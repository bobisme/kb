//! `kb forget <target>` — remove an ingested source from the knowledge base.
//!
//! A personal KB is append-only by default, but users sometimes want to
//! retire a source (renamed file, experimental note, outright mistake).
//! `kb forget` provides the one blessed path for doing so.
//!
//! Design decisions:
//!
//! - **Move, don't rm.** Affected directories are moved into
//!   `trash/<src_id>-<timestamp>/` so a mistaken forget is recoverable. No
//!   TTL is enforced today — the user is expected to clean `trash/` manually
//!   when they're satisfied nothing valuable went in. A future pass can add
//!   a GC hook (e.g. `kb doctor --clean-trash-older-than 30d`).
//!
//! - **Best-effort backlinks.** After removing the wiki source page we run
//!   `run_backlinks_pass` so concept pages stop linking to a dead URL.
//!
//! - **Cascade (bn-did).** Concept pages whose `source_document_ids` becomes
//!   empty after forgetting `<src>` are orphans by definition; the forget
//!   command enumerates them in the pre-flight plan and moves them into the
//!   same trash bundle on approval. Promoted question pages citing the
//!   forgotten src are flagged via an `orphaned_sources:` frontmatter field
//!   but are left in place (full rewriting is a future pass). Stale build
//!   records whose `metadata.output_paths` names files under the forgotten
//!   source are also trashed. `--no-cascade` opts out and restores the pre-
//!   bn-did behavior (source-only).
//!
//! - **Hash-state cleanup.** `state/hashes.json` is rewritten without the
//!   forgotten `normalized/<id>` key so `kb status` doesn't flag its now-
//!   missing normalized dir as a "never compiled" changed input.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use kb_compile::{HashState, backlinks};
use kb_core::{BuildRecord, build_records_dir, frontmatter::read_frontmatter, frontmatter::write_frontmatter};
use serde::Serialize;
use serde_yaml::{Mapping, Value};

/// Cascade effects discovered by the pre-flight analysis for a given src.
///
/// Empty lists are valid and mean "no cascade effects of that kind" — the
/// caller still emits the header but skips the sub-section.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct CascadePlan {
    /// Concept pages (absolute paths) whose `source_document_ids`
    /// becomes empty after removing this src; they'll be moved to trash.
    pub orphaned_concept_pages: Vec<PathBuf>,
    /// Promoted question pages (absolute paths) whose `source_document_ids`
    /// lists this src; we add an `orphaned_sources:` marker to the
    /// frontmatter instead of removing/rewriting the page.
    pub flagged_question_pages: Vec<PathBuf>,
    /// Build records (absolute paths to the JSON files) whose
    /// `metadata.output_paths` names files under the forgotten source
    /// (normalized/<src>/... or wiki/sources/<src>.md); they'll be moved to
    /// trash.
    pub stale_build_records: Vec<PathBuf>,
}

impl CascadePlan {
    /// `true` when this cascade does not move or flag any file.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.orphaned_concept_pages.is_empty()
            && self.flagged_question_pages.is_empty()
            && self.stale_build_records.is_empty()
    }
}

/// What a single `kb forget` call would move (or moved) to trash.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgetPlan {
    /// Source-document id (`src-<hex>`) being removed.
    pub src_id: String,
    /// Human-readable origin (filesystem path or URL) recovered from
    /// `raw/inbox/<src>/source_document.json::stable_location`, when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
    /// Destination trash directory (absolute) that receives every moved path.
    pub trash_dir: PathBuf,
    /// Absolute paths that will be (or were) moved into the trash dir.
    /// Entries missing on disk at plan time are omitted so the plan reflects
    /// only real work.
    pub moves: Vec<PathBuf>,
    /// Cascade effects discovered by the pre-flight analysis. Empty when
    /// `--no-cascade` was passed; non-empty otherwise (but individual fields
    /// may still be empty).
    #[serde(default, skip_serializing_if = "CascadePlan::is_empty")]
    pub cascade: CascadePlan,
}

/// Result of a `kb forget` invocation, suitable for `--json` output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgetOutcome {
    pub plan: ForgetPlan,
    pub dry_run: bool,
    /// Whether we walked `wiki/concepts/*.md` to refresh backlinks after
    /// removing the source page. Dry-run skips this pass.
    pub backlinks_refreshed: bool,
}

/// Resolve `<target>` to a concrete `src-<hex>` id.
///
/// Accepts either:
///   - a bare src-id (`src-0639ebb0`), which is verified to exist under
///     `raw/inbox/<id>/` or `normalized/<id>/`, or
///   - a filesystem path, which is canonicalized, normalized via
///     `kb_core::normalize_file_stable_location`, and matched against
///     `stable_location` in each `raw/inbox/*/source_document.json`.
///
/// Returns the src-id, plus (when available) the `stable_location` string so
/// the caller can echo it in the confirmation prompt and the job manifest.
///
/// # Errors
///
/// Returns an error when:
///   - the target looks like a src-id but no matching `raw/inbox/<id>/` nor
///     `normalized/<id>/` directory exists,
///   - the target is a path but no ingested source records it as their
///     `stable_location`,
///   - reading `raw/inbox/` fails.
pub fn resolve_target(root: &Path, target: &str) -> Result<(String, Option<String>)> {
    if is_src_id(target) {
        let raw_dir = root.join("raw/inbox").join(target);
        let normalized_dir = root.join("normalized").join(target);
        if !raw_dir.exists() && !normalized_dir.exists() {
            bail!(
                "no source with id '{target}' found under {} or {}",
                raw_dir.display(),
                normalized_dir.display()
            );
        }
        let origin = read_stable_location(root, target);
        return Ok((target.to_string(), origin));
    }

    // Path target: canonicalize + normalize like ingest does, then scan
    // `raw/inbox/*/source_document.json` for a matching stable_location.
    let candidate = PathBuf::from(target);
    let canonical = fs::canonicalize(&candidate)
        .with_context(|| format!("failed to canonicalize '{target}' (does the file exist?)"))?;
    let stable_location = kb_core::normalize_file_stable_location(&canonical)
        .with_context(|| format!("failed to normalize '{target}'"))?;

    let inbox = root.join("raw/inbox");
    if !inbox.exists() {
        bail!(
            "no ingested sources found: '{}' does not exist",
            inbox.display()
        );
    }
    for entry in fs::read_dir(&inbox)
        .with_context(|| format!("read {}", inbox.display()))?
    {
        let entry = entry.with_context(|| format!("read entry in {}", inbox.display()))?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let id = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("non-utf8 directory name under raw/inbox"))?;
        if !is_src_id(&id) {
            continue;
        }
        if let Some(found) = read_stable_location(root, &id)
            && found == stable_location
        {
            return Ok((id, Some(stable_location)));
        }
    }

    bail!(
        "no ingested source matches path '{target}' (normalized: '{stable_location}')"
    )
}

/// Build a `ForgetPlan` for `src_id` without touching disk.
///
/// When `cascade` is `true` (default; `--no-cascade` passes `false`), walks
/// `wiki/concepts/*.md`, `wiki/questions/*.md`, and `state/build_records/*.json`
/// and populates `plan.cascade` with the derived effects. A concept page is
/// marked orphaned iff its `source_document_ids` list contains `src_id` and
/// has length 1 (so removing `src_id` empties it); a concept with two or more
/// sources survives because removing this one leaves at least one grounding
/// source. Question pages are flagged (not rewritten) whenever the list
/// merely *contains* `src_id`, regardless of the other sources present.
///
/// # Errors
///
/// Returns an error when the system clock is before `UNIX_EPOCH`
/// (unreachable on sane hosts; used only to derive the trash timestamp), or
/// when the cascade walk encounters an unreadable directory.
pub fn plan(
    root: &Path,
    src_id: &str,
    origin: Option<String>,
    cascade: bool,
) -> Result<ForgetPlan> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock before UNIX_EPOCH")?
        .as_secs();
    let trash_dir = root
        .join("trash")
        .join(format!("{src_id}-{timestamp}"));

    let candidates = [
        root.join("normalized").join(src_id),
        root.join("raw/inbox").join(src_id),
        root.join(kb_compile::source_page::source_page_path_for_id(src_id)),
    ];
    let moves: Vec<PathBuf> = candidates
        .into_iter()
        .filter(|p| p.exists())
        .collect();

    let cascade_plan = if cascade {
        analyze_cascade(root, src_id)?
    } else {
        CascadePlan::default()
    };

    Ok(ForgetPlan {
        src_id: src_id.to_string(),
        origin,
        trash_dir,
        moves,
        cascade: cascade_plan,
    })
}

/// Walk the KB and enumerate concept pages, promoted question pages, and
/// build records affected by forgetting `src_id`.
///
/// See `plan` for the precise definition of "orphaned" vs "flagged".
///
/// # Errors
///
/// Returns an error only when a directory listing itself fails. Unreadable
/// or malformed individual files (bad YAML frontmatter, unreadable JSON)
/// are silently skipped — they would be stale regardless of this cascade
/// and `kb lint` will surface them on the next run.
fn analyze_cascade(root: &Path, src_id: &str) -> Result<CascadePlan> {
    let orphaned_concept_pages =
        scan_frontmatter_matches(&root.join("wiki/concepts"), src_id, /*orphan_only=*/ true)?;
    let flagged_question_pages =
        scan_frontmatter_matches(&root.join("wiki/questions"), src_id, /*orphan_only=*/ false)?;
    let stale_build_records = scan_stale_build_records(root, src_id)?;
    Ok(CascadePlan {
        orphaned_concept_pages,
        flagged_question_pages,
        stale_build_records,
    })
}

/// Walk `dir` (a `wiki/*` subdirectory) and return `.md` files whose
/// frontmatter `source_document_ids` references `src_id`.
///
/// `orphan_only=true`: only return files where the list *equals* `[src_id]`
/// (the 1-element case). After forgetting `src_id`, the list becomes empty
/// and the page is orphaned.
///
/// `orphan_only=false`: return files where the list merely *contains*
/// `src_id`. Used for promoted questions, which we flag but don't remove
/// even when they'd otherwise be orphaned.
fn scan_frontmatter_matches(
    dir: &Path,
    src_id: &str,
    orphan_only: bool,
) -> Result<Vec<PathBuf>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut matches = Vec::new();
    for entry in fs::read_dir(dir)
        .with_context(|| format!("read dir {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("read entry in {}", dir.display()))?;
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        // Malformed frontmatter (lint will flag it) — skip silently.
        let Ok((frontmatter, _body)) = read_frontmatter(&path) else {
            continue;
        };
        let ids = frontmatter_source_ids(&frontmatter);
        let contains = ids.iter().any(|id| id == src_id);
        if !contains {
            continue;
        }
        if orphan_only {
            // Orphan: list is exactly [src_id] after the forget.
            if ids.len() == 1 {
                matches.push(path);
            }
        } else {
            matches.push(path);
        }
    }
    matches.sort();
    Ok(matches)
}

/// Extract the `source_document_ids` list from a frontmatter mapping.
/// Returns an empty vec when the field is missing, non-sequence, or contains
/// non-string entries (mixed entries are taken at their string parts).
fn frontmatter_source_ids(frontmatter: &Mapping) -> Vec<String> {
    let Some(value) = frontmatter.get(Value::String("source_document_ids".into())) else {
        return Vec::new();
    };
    match value {
        Value::Sequence(seq) => seq
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    }
}

/// Walk `state/build_records/*.json` and return those whose
/// `metadata.output_paths` includes a file under `normalized/<src>/...` or
/// `wiki/sources/<src>.md` (the two directly-forgotten bucket prefixes).
fn scan_stale_build_records(root: &Path, src_id: &str) -> Result<Vec<PathBuf>> {
    let dir = build_records_dir(root);
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let normalized_prefix = PathBuf::from("normalized").join(src_id);
    let wiki_source_path = kb_compile::source_page::source_page_path_for_id(src_id);
    let mut matches = Vec::new();
    for entry in fs::read_dir(&dir)
        .with_context(|| format!("read build_records dir {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("read entry in {}", dir.display()))?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let Ok(bytes) = fs::read(&path) else {
            continue;
        };
        let Ok(record) = serde_json::from_slice::<BuildRecord>(&bytes) else {
            continue;
        };
        let references_src = record.metadata.output_paths.iter().any(|out| {
            out.starts_with(&normalized_prefix) || out == &wiki_source_path
        });
        if references_src {
            matches.push(path);
        }
    }
    matches.sort();
    Ok(matches)
}

/// Execute a previously-built `ForgetPlan`, moving real bytes on disk.
///
/// Each entry in `plan.moves` is moved into `plan.trash_dir` under its
/// original relative path (e.g. `normalized/src-XXXX` → `trash/src-XXXX-T/normalized/`,
/// `wiki/sources/src-XXXX.md` → `trash/src-XXXX-T/wiki/sources/src-XXXX.md`).
/// Layout preserves the original parent directory name so the trash dir is
/// self-describing — a user poking at `trash/src-XXXX-1700000000/` can tell
/// which bucket each file came from.
///
/// Cascade effects execute after the core moves:
///   - `plan.cascade.orphaned_concept_pages` — each page is moved under
///     `trash/<bundle>/wiki/concepts/`.
///   - `plan.cascade.flagged_question_pages` — each page gets (or appends)
///     `orphaned_sources: [<src_id>]` to its frontmatter. The body is not
///     rewritten; a future pass can handle full rewrites.
///   - `plan.cascade.stale_build_records` — each JSON file is moved under
///     `trash/<bundle>/state/build_records/`.
///
/// After moves succeed, the `normalized/<id>` key is dropped from
/// `state/hashes.json` (if present) and `run_backlinks_pass` is invoked so
/// concept pages no longer link into a deleted source. Backlink refresh is
/// best-effort: a failure emits a warning and returns success, because the
/// user has already agreed to the destructive operation and rerunning `kb
/// compile` will correct any lingering staleness.
///
/// # Errors
///
/// Returns an error when a move fails or the hash-state file cannot be
/// updated. Partial moves are NOT rolled back — callers should treat a
/// failure as "inspect manually and finish by hand".
pub fn execute(root: &Path, plan: &ForgetPlan) -> Result<bool> {
    fs::create_dir_all(&plan.trash_dir)
        .with_context(|| format!("create trash dir {}", plan.trash_dir.display()))?;

    // Core moves first (normalized/, raw/inbox/, wiki/sources/).
    for src in &plan.moves {
        move_into_trash(root, src, &plan.trash_dir)?;
    }

    // Cascade: orphaned concept pages → trash.
    for concept in &plan.cascade.orphaned_concept_pages {
        if concept.exists() {
            move_into_trash(root, concept, &plan.trash_dir)?;
        }
    }

    // Cascade: flag (not rewrite) promoted question pages.
    for question in &plan.cascade.flagged_question_pages {
        if question.exists() {
            flag_orphaned_source(question, &plan.src_id)?;
        }
    }

    // Cascade: stale build records → trash.
    for record in &plan.cascade.stale_build_records {
        if record.exists() {
            move_into_trash(root, record, &plan.trash_dir)?;
        }
    }

    // Drop the forgotten source from the hash state so `kb status` doesn't
    // immediately re-flag it as "never compiled". Missing file is fine.
    let hash_state_path = root.join("state/hashes.json");
    if hash_state_path.exists() {
        let mut state = HashState::load_from_root(root)?;
        let key = format!("normalized/{}", plan.src_id);
        if state.hashes.remove(&key).is_some() {
            state.save_to_root(root)?;
        }
    }

    // Refresh backlinks so concept pages stop pointing at the (now-gone)
    // source page. Best-effort: a failure here doesn't undo the forget.
    let backlinks_refreshed = match backlinks::run_backlinks_pass(root) {
        Ok(artifacts) => match backlinks::persist_backlinks_artifacts(&artifacts) {
            Ok(()) => true,
            Err(err) => {
                eprintln!(
                    "warning: backlinks refresh failed after forget: {err:#}"
                );
                false
            }
        },
        Err(err) => {
            eprintln!(
                "warning: backlinks refresh failed after forget: {err:#}"
            );
            false
        }
    };

    Ok(backlinks_refreshed)
}

/// Move `src` (absolute) into `trash_dir`, preserving its path relative to
/// `root`. Parent directories under `trash_dir` are created as needed.
fn move_into_trash(root: &Path, src: &Path, trash_dir: &Path) -> Result<()> {
    let relative = src.strip_prefix(root).unwrap_or(src).to_path_buf();
    let dest = trash_dir.join(&relative);
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create trash subdir {}", parent.display()))?;
    }
    // `fs::rename` works across the same filesystem and is atomic; the
    // KB root and its trash/ subdir are always on the same mount so
    // we don't need fallback-copy semantics here.
    fs::rename(src, &dest)
        .with_context(|| format!("move {} → {}", src.display(), dest.display()))?;
    Ok(())
}

/// Append `src_id` to the `orphaned_sources:` frontmatter field on `page`,
/// creating the field as a single-element list when it is absent. Idempotent
/// — a second forget of the same src does not duplicate the entry. The body
/// of the page is left untouched; v1 of the cascade treats this as a marker
/// for later sweeps (full rewriting is tracked as future work).
fn flag_orphaned_source(page: &Path, src_id: &str) -> Result<()> {
    let (mut frontmatter, body) = read_frontmatter(page)
        .with_context(|| format!("read frontmatter for {}", page.display()))?;
    let key = Value::String("orphaned_sources".into());
    let entry = Value::String(src_id.to_string());
    match frontmatter.get_mut(&key) {
        Some(Value::Sequence(seq)) => {
            if !seq.iter().any(|v| v.as_str() == Some(src_id)) {
                seq.push(entry);
            }
        }
        _ => {
            frontmatter.insert(key, Value::Sequence(vec![entry]));
        }
    }
    write_frontmatter(page, &frontmatter, body)
        .with_context(|| format!("write frontmatter for {}", page.display()))?;
    Ok(())
}

/// Render the prompt shown in the default (interactive) flow.
pub fn prompt_text(plan: &ForgetPlan) -> String {
    let label = plan
        .origin
        .as_deref()
        .map_or_else(|| plan.src_id.clone(), origin_label);
    format!(
        "remove source '{label}' ({})? This will delete wiki/sources/{}.md. [y/N] ",
        plan.src_id, plan.src_id
    )
}

/// Short human label for a `stable_location` — basename for local paths,
/// the full string for URLs.
fn origin_label(origin: &str) -> String {
    if origin.starts_with("http://") || origin.starts_with("https://") {
        origin.to_string()
    } else {
        Path::new(origin)
            .file_name()
            .and_then(|s| s.to_str())
            .map_or_else(|| origin.to_string(), ToOwned::to_owned)
    }
}

/// Ask the user `y/N` on stderr; default No. Any read error (closed stdin,
/// non-tty) aborts as a safety rail — we never forget silently.
pub fn confirm_on_stderr(plan: &ForgetPlan) -> Result<bool> {
    eprint!("{}", prompt_text(plan));
    io::stderr().flush().ok();
    let mut answer = String::new();
    io::stdin()
        .read_line(&mut answer)
        .context("failed to read confirmation from stdin")?;
    let trimmed = answer.trim().to_ascii_lowercase();
    Ok(trimmed == "y" || trimmed == "yes")
}

/// Readable string for `kb forget` text output.
///
/// The cascade section mirrors the bone spec's sample:
///
///   forget src-aa540111 would also affect:
///     concept pages orphaned by this forget (will be moved to trash):
///       - wiki/concepts/zab.md
///     promoted questions that cite this source (will be flagged but not rewritten):
///       - wiki/questions/how-does-zookeeper-x.md
///     stale build records (will be removed):
///       - state/build_records/build:source-summary:src-aa540111.json
pub fn render_plan(plan: &ForgetPlan, dry_run: bool) -> String {
    use std::fmt::Write as _;
    let verb = if dry_run { "would move" } else { "moved" };
    let mut out = String::new();
    if let Some(origin) = &plan.origin {
        // Writes to a String are infallible; unwrap is safe.
        let _ = writeln!(out, "forget {} (origin: {origin})", plan.src_id);
    } else {
        let _ = writeln!(out, "forget {}", plan.src_id);
    }
    if plan.moves.is_empty() && plan.cascade.is_empty() {
        out.push_str(
            "  nothing to move — no normalized/, raw/inbox/, or wiki/sources/ entry found\n",
        );
        return out;
    }
    for src in &plan.moves {
        let _ = writeln!(out, "  {verb}: {}", src.display());
    }
    if !plan.cascade.is_empty() {
        out.push_str("  cascade:\n");
        if !plan.cascade.orphaned_concept_pages.is_empty() {
            out.push_str(
                "    concept pages orphaned by this forget (will be moved to trash):\n",
            );
            for page in &plan.cascade.orphaned_concept_pages {
                let _ = writeln!(out, "      - {}", page.display());
            }
        }
        if !plan.cascade.flagged_question_pages.is_empty() {
            out.push_str(
                "    promoted questions that cite this source (will be flagged but not rewritten):\n",
            );
            for page in &plan.cascade.flagged_question_pages {
                let _ = writeln!(out, "      - {}", page.display());
            }
        }
        if !plan.cascade.stale_build_records.is_empty() {
            out.push_str("    stale build records (will be removed):\n");
            for rec in &plan.cascade.stale_build_records {
                let _ = writeln!(out, "      - {}", rec.display());
            }
        }
    }
    if !plan.moves.is_empty() || !plan.cascade.orphaned_concept_pages.is_empty()
        || !plan.cascade.stale_build_records.is_empty()
    {
        let _ = writeln!(out, "  trash:    {}", plan.trash_dir.display());
    }
    out
}

fn is_src_id(token: &str) -> bool {
    token.starts_with("src-") && token.len() > 4 && token[4..].chars().all(|c| c.is_ascii_hexdigit())
}

/// Best-effort read of `raw/inbox/<src>/source_document.json::stable_location`.
fn read_stable_location(root: &Path, src_id: &str) -> Option<String> {
    let path = root
        .join("raw/inbox")
        .join(src_id)
        .join("source_document.json");
    let bytes = fs::read(&path).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    value
        .get("stable_location")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
}

/// `true` when `origin` looks like a filesystem path we can stat (i.e. not
/// a URL scheme). Used by `gather_status` to skip URL sources when flagging
/// sources with missing origins.
#[must_use]
pub fn origin_is_local_path(origin: &str) -> bool {
    !(origin.starts_with("http://")
        || origin.starts_with("https://")
        || origin.starts_with("ftp://")
        || origin.starts_with("ftps://"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn is_src_id_accepts_hex_ids() {
        assert!(is_src_id("src-0639ebb0"));
        assert!(is_src_id("src-abcdef"));
        assert!(!is_src_id("src-"));
        assert!(!is_src_id("foo"));
        assert!(!is_src_id("src-xyz"));
    }

    #[test]
    fn origin_is_local_path_rejects_urls() {
        assert!(origin_is_local_path("/tmp/foo.md"));
        assert!(origin_is_local_path("relative/path.md"));
        assert!(!origin_is_local_path("https://example.com/a"));
        assert!(!origin_is_local_path("http://example.com/a"));
    }

    fn write_source_document(root: &Path, src_id: &str, stable_location: &str) {
        let dir = root.join("raw/inbox").join(src_id);
        fs::create_dir_all(&dir).expect("mkdir raw/inbox");
        let body = serde_json::json!({
            "metadata": {
                "id": src_id,
                "entity_type": "source-document",
                "display_name": stable_location,
                "canonical_path": "inbox/x.md",
                "content_hashes": [],
                "output_paths": [],
                "status": "Fresh",
            },
            "source_kind": "file",
            "stable_location": stable_location,
            "discovered_at_millis": 0u64,
        });
        fs::write(dir.join("source_document.json"), body.to_string())
            .expect("write source_document.json");
    }

    /// Write a minimal concept page with `source_document_ids:` frontmatter.
    fn write_concept_page(root: &Path, slug: &str, source_ids: &[&str]) -> PathBuf {
        let dir = root.join("wiki/concepts");
        fs::create_dir_all(&dir).expect("mkdir wiki/concepts");
        let ids_yaml = source_ids
            .iter()
            .map(|id| format!("  - {id}"))
            .collect::<Vec<_>>()
            .join("\n");
        let body = format!(
            "---\nid: concept-{slug}\nname: {slug}\nsource_document_ids:\n{ids_yaml}\n---\n\n# {slug}\n",
        );
        let path = dir.join(format!("{slug}.md"));
        fs::write(&path, body).expect("write concept page");
        path
    }

    #[test]
    fn resolve_target_by_src_id_returns_origin() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        write_source_document(root, "src-deadbeef", "/tmp/fake/foo.md");
        let (id, origin) =
            resolve_target(root, "src-deadbeef").expect("resolve by id");
        assert_eq!(id, "src-deadbeef");
        assert_eq!(origin.as_deref(), Some("/tmp/fake/foo.md"));
    }

    #[test]
    fn resolve_target_unknown_src_id_errors() {
        // Valid-hex id that doesn't exist in raw/inbox nor normalized/.
        let dir = tempdir().expect("tempdir");
        let err = resolve_target(dir.path(), "src-00ff00ff")
            .expect_err("unknown src-id should error");
        assert!(
            err.to_string().contains("no source with id"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn plan_lists_only_existing_paths() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let src = "src-cafef00d";
        fs::create_dir_all(root.join("normalized").join(src)).expect("create normalized");
        // No wiki page, no raw/inbox entry yet.
        let plan = plan(root, src, None, true).expect("plan");
        assert_eq!(plan.src_id, src);
        assert_eq!(plan.moves.len(), 1);
        assert!(plan.moves[0].ends_with(format!("normalized/{src}")));
    }

    #[test]
    fn cascade_detects_orphaned_concept_and_skips_multi_sourced_one() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let src = "src-abc00001";
        let other = "src-abc00002";
        fs::create_dir_all(root.join("normalized").join(src)).expect("create normalized");
        let orphan = write_concept_page(root, "zab", &[src]);
        let multi = write_concept_page(root, "shared", &[src, other]);

        let plan = plan(root, src, None, true).expect("plan with cascade");
        assert_eq!(
            plan.cascade.orphaned_concept_pages,
            vec![orphan],
            "only solely-sourced concepts should orphan; got {:?}",
            plan.cascade.orphaned_concept_pages
        );
        assert!(
            !plan.cascade.orphaned_concept_pages.contains(&multi),
            "multi-sourced concept must NOT be flagged as orphaned"
        );
    }

    #[test]
    fn cascade_disabled_yields_empty_cascade_plan() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let src = "src-abc00003";
        fs::create_dir_all(root.join("normalized").join(src)).expect("create normalized");
        write_concept_page(root, "zab", &[src]);

        let plan = plan(root, src, None, false).expect("plan without cascade");
        assert!(
            plan.cascade.is_empty(),
            "no-cascade plan must not enumerate cascade effects: {:?}",
            plan.cascade
        );
    }

    #[test]
    fn flag_orphaned_source_adds_new_field() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let src = "src-q00001";
        let question_dir = root.join("wiki/questions");
        fs::create_dir_all(&question_dir).expect("mkdir");
        let page = question_dir.join("q.md");
        fs::write(
            &page,
            format!(
                "---\nid: q-1\ntitle: How\nsource_document_ids:\n  - {src}\n---\n\n# q\n"
            ),
        )
        .expect("write q");

        flag_orphaned_source(&page, src).expect("flag");
        let (fm, _body) = read_frontmatter(&page).expect("re-read");
        let value = fm
            .get(Value::String("orphaned_sources".into()))
            .expect("orphaned_sources added");
        let seq = value.as_sequence().expect("sequence");
        assert_eq!(seq.len(), 1);
        assert_eq!(seq[0].as_str(), Some(src));
    }

    #[test]
    fn flag_orphaned_source_is_idempotent() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let src = "src-q00002";
        let page = root.join("wiki/questions/q.md");
        fs::create_dir_all(page.parent().expect("parent")).expect("mkdir");
        fs::write(
            &page,
            format!(
                "---\nid: q-2\ntitle: How\nsource_document_ids:\n  - {src}\norphaned_sources:\n  - {src}\n---\n\n# q\n"
            ),
        )
        .expect("write q");

        flag_orphaned_source(&page, src).expect("flag again");
        let (fm, _body) = read_frontmatter(&page).expect("re-read");
        let seq = fm
            .get(Value::String("orphaned_sources".into()))
            .expect("present")
            .as_sequence()
            .expect("seq");
        assert_eq!(seq.len(), 1, "duplicate flags must collapse");
    }

    #[test]
    fn execute_moves_existing_dirs_and_updates_hash_state() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let src = "src-abcdef01";

        let normalized = root.join("normalized").join(src);
        fs::create_dir_all(&normalized).expect("mkdir normalized");
        fs::write(normalized.join("source.md"), "body").expect("write source.md");

        let raw = root.join("raw/inbox").join(src);
        fs::create_dir_all(&raw).expect("mkdir raw");
        fs::write(raw.join("source_document.json"), "{}").expect("write sd.json");

        let wiki_page = root.join(kb_compile::source_page::source_page_path_for_id(src));
        fs::create_dir_all(wiki_page.parent().expect("wiki page has parent"))
            .expect("mkdir wiki/sources");
        fs::write(&wiki_page, "---\nid: x\n---\n").expect("write wiki page");

        let mut state = HashState::default();
        state
            .hashes
            .insert(format!("normalized/{src}"), "fp".to_string());
        state.save_to_root(root).expect("save hash state");

        let plan = plan(root, src, None, true).expect("plan");
        assert_eq!(plan.moves.len(), 3);
        execute(root, &plan).expect("execute forget");

        assert!(!normalized.exists(), "normalized dir should be gone");
        assert!(!raw.exists(), "raw/inbox dir should be gone");
        assert!(!wiki_page.exists(), "wiki page should be gone");
        assert!(plan.trash_dir.exists(), "trash dir should exist");
        assert!(
            plan.trash_dir.join(format!("normalized/{src}")).exists(),
            "normalized dir preserved under trash"
        );

        let reloaded = HashState::load_from_root(root).expect("reload hash state");
        assert!(
            !reloaded.hashes.contains_key(&format!("normalized/{src}")),
            "hash state should no longer list the forgotten src"
        );
    }
}
