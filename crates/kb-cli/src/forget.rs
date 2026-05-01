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
use kb_compile::{Graph, HashState, backlinks, index_page};
use kb_core::{
    BuildRecord, build_records_dir, frontmatter::read_frontmatter,
    frontmatter::write_frontmatter, normalized_dir, normalized_rel, trash_dir,
};
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
    /// Whether each of the three post-trash layout refreshes ran
    /// successfully (bn-i5r: index pages, frontmatter scrub, lexical index).
    /// Dry-run always reports `false` since nothing is executed.
    #[serde(default, skip_serializing_if = "CascadeRefresh::is_noop")]
    pub cascade_refresh: CascadeRefresh,
}

/// Accounting for the three structural refreshes that run after the trash
/// moves complete — separate from the cascade *plan* because these operate on
/// the whole wiki, not a fixed per-src list.
///
/// Each field is best-effort: a failure is logged as a warning and recorded
/// here as `false`/`0` so callers (and humans reading `--json`) can tell what
/// actually happened without interpreting the exit code.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct CascadeRefresh {
    /// `true` when `wiki/index.md`, `wiki/sources/index.md`, and
    /// `wiki/concepts/index.md` were regenerated from the current on-disk
    /// layout.
    pub index_pages_refreshed: bool,
    /// Number of surviving concept pages whose frontmatter
    /// `source_document_ids` was rewritten to drop the forgotten src-id.
    pub frontmatter_scrubbed: usize,
    /// `true` when `state/indexes/lexical.json` was rebuilt from the current
    /// on-disk wiki pages.
    pub lexical_index_refreshed: bool,
    /// Number of dependency-graph nodes surgically removed from
    /// `state/graph.json` (bn-3f6). Zero when `state/graph.json` was absent
    /// or carried no nodes referencing the forgotten src.
    #[serde(default)]
    pub graph_nodes_pruned: usize,
}

impl CascadeRefresh {
    /// `true` when no refresh ran (useful for skipping the JSON field on
    /// `--dry-run` / "nothing to remove" paths).
    #[must_use]
    pub const fn is_noop(&self) -> bool {
        !self.index_pages_refreshed
            && self.frontmatter_scrubbed == 0
            && !self.lexical_index_refreshed
            && self.graph_nodes_pruned == 0
    }
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
        let normalized_source_dir = normalized_dir(root).join(target);
        if raw_dir.exists() || normalized_source_dir.exists() {
            let origin = read_stable_location(root, target);
            return Ok((target.to_string(), origin));
        }
        // bn-1525: the target looks like a src-id but isn't on disk as-is.
        // Try prefix resolution against known src ids before bailing — users
        // often type `src-a7` expecting to hit `src-a7x3q9`.
        match crate::id_resolve::resolve(root, target) {
            Ok(resolved) if resolved.kind == crate::id_resolve::IdKind::Source => {
                let origin = read_stable_location(root, &resolved.id);
                return Ok((resolved.id, origin));
            }
            Err(err) if err.to_string().contains("ambiguous") => {
                // Surface the candidate list verbatim; the user needs to type
                // a longer prefix.
                return Err(err);
            }
            // Resolver found a concept/question instead (forget only
            // operates on sources), or came up empty — fall through to the
            // regular "no source with id" error.
            Ok(_) | Err(_) => {}
        }
        bail!(
            "no source with id '{target}' found under {} or {}",
            raw_dir.display(),
            normalized_source_dir.display()
        );
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
    let trash_bucket = trash_dir(root).join(format!("{src_id}-{timestamp}"));

    // bn-nlw9: the wiki source page lives at either `wiki/sources/<src>.md`
    // (legacy) or `wiki/sources/<src>-<slug>.md` (current). Resolve the actual
    // on-disk path so forget trashes whichever exists.
    let mut candidates: Vec<PathBuf> = vec![
        normalized_dir(root).join(src_id),
        root.join("raw/inbox").join(src_id),
    ];
    if let Some(page) = kb_compile::source_page::resolve_source_page_path(root, src_id) {
        candidates.push(page);
    }
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
        trash_dir: trash_bucket,
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

/// Walk `wiki/concepts/*.md` and return pages whose frontmatter
/// `source_document_ids` contains `src_id` with *other* ids alongside it.
/// These are the pages that survive cascade (not trashed) but still need
/// their frontmatter scrubbed so `kb lint orphans` stops flagging a dangling
/// reference.
///
/// # Errors
///
/// Returns an error only when the directory listing itself fails.
pub fn concept_pages_needing_scrub(root: &Path, src_id: &str) -> Result<Vec<PathBuf>> {
    let dir = root.join("wiki/concepts");
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut matches = Vec::new();
    for entry in fs::read_dir(&dir)
        .with_context(|| format!("read dir {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("read entry in {}", dir.display()))?;
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        let Ok((frontmatter, _body)) = read_frontmatter(&path) else {
            continue;
        };
        let ids = frontmatter_source_ids(&frontmatter);
        // Survivor: contains the forgotten src and at least one other.
        if ids.iter().any(|id| id == src_id) && ids.len() >= 2 {
            matches.push(path);
        }
    }
    matches.sort();
    Ok(matches)
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

/// Walk `state/build_records/*.json` and return those attributable to
/// `src_id` by any of the following criteria (bn-3f6 expansion):
///
/// 1. `metadata.output_paths` includes a path under `normalized/<src>/...`
///    (covers passes that emit into the normalized bucket, e.g. hypothetical
///    `source_summary` variants).
/// 2. `metadata.output_paths` equals `wiki/sources/<src>.md` (the canonical
///    source-summary destination).
/// 3. `metadata.output_paths` includes a path under `state/concept_candidates/<src>.*`
///    (covers `extract_concepts`, which pass-11 found was being missed).
/// 4. `metadata.id` contains `:<src>` as a suffix component (covers every
///    pass whose record id follows the `build:<pass>:<src_id>` convention,
///    including future passes that haven't been written yet).
///
/// Together these guarantee that every build record scoped to a single
/// forgotten source is trashed — the pass-11 audit found that before this
/// change only `build:source-summary:<src>` was removed, leaving
/// `build:extract-concepts:<src>` on disk.
fn scan_stale_build_records(root: &Path, src_id: &str) -> Result<Vec<PathBuf>> {
    let dir = build_records_dir(root);
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let normalized_prefix = normalized_rel(src_id);
    // Legacy (id-only) wiki source page path plus the id-slug prefix. bn-nlw9
    // introduced `wiki/sources/<src-id>-<slug>.md`, so we accept any
    // `wiki/sources/<src-id>*.md`-shaped output here.
    let wiki_source_id_only = kb_compile::source_page::source_page_path_for_id(src_id);
    let wiki_source_prefix = format!("wiki/sources/{src_id}-");
    let concept_candidates_prefix = PathBuf::from(kb_core::KB_DIR)
        .join(kb_core::STATE_SUBDIR)
        .join("concept_candidates")
        .join(src_id);
    // `build:<pass>:<src-id>` → the id ends with `:<src-id>`. A suffix check
    // is narrow enough to avoid collisions with records for unrelated srcs
    // whose id happens to embed this src-id as a substring.
    let id_suffix = format!(":{src_id}");
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
        let references_src_via_paths = record.metadata.output_paths.iter().any(|out| {
            let out_str = out.to_string_lossy();
            out.starts_with(&normalized_prefix)
                || out == &wiki_source_id_only
                || out_str.starts_with(&wiki_source_prefix)
                // `starts_with` on `state/concept_candidates/<src>` catches both
                // `<src>.json` (exact file) and any future subpath.
                || out.starts_with(&concept_candidates_prefix)
                || out_str.starts_with(
                    &format!(
                        "{}/{}/concept_candidates/{src_id}.",
                        kb_core::KB_DIR,
                        kb_core::STATE_SUBDIR,
                    ),
                )
        });
        let references_src_via_id = record.metadata.id.ends_with(&id_suffix);
        if references_src_via_paths || references_src_via_id {
            matches.push(path);
        }
    }
    matches.sort();
    Ok(matches)
}

/// Aggregate result of running [`execute`] — tracks which post-trash passes
/// ran successfully so the CLI (and `--json` consumers) can report them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExecuteOutcome {
    /// Whether backlinks were regenerated (same semantics as pre-bn-i5r).
    pub backlinks_refreshed: bool,
    /// Post-trash structural refreshes added by bn-i5r.
    pub cascade_refresh: CascadeRefresh,
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
/// After moves succeed we:
///   1. drop the `normalized/<id>` key from `state/hashes.json` (if present),
///   2. refresh backlinks (`run_backlinks_pass`) so concept pages no longer
///      point at the deleted source,
///   3. **scrub** surviving concept frontmatter — any multi-sourced concept
///      still listing `src_id` in `source_document_ids` has that entry
///      removed via an atomic frontmatter rewrite (bn-i5r step 2),
///   4. **regenerate** `wiki/index.md`, `wiki/sources/index.md`, and
///      `wiki/concepts/index.md` from the current on-disk layout so their
///      link lists no longer point into `trash/` (bn-i5r step 1),
///   5. **rebuild** the lexical search index at `state/indexes/lexical.json`
///      from surviving wiki pages so `kb ask` doesn't crash trying to read a
///      trashed candidate (bn-i5r step 3),
///   6. **prune** `state/graph.json` — surgically remove every node that
///      references the forgotten src (`source-document-<src>`,
///      `wiki-page-<src>`, and path-style nodes containing `/<src>/`) along
///      with their edges so `kb inspect` / `kb status` don't surface dangling
///      references until the next full compile (bn-3f6).
///
/// Steps 2–6 are all best-effort: a failure emits a warning and marks the
/// corresponding field in [`ExecuteOutcome`] as `false`/`0` but the forget as
/// a whole still succeeds — rerunning `kb compile` will correct any lingering
/// staleness.
///
/// # Errors
///
/// Returns an error when a move fails or the hash-state file cannot be
/// updated. Partial moves are NOT rolled back — callers should treat a
/// failure as "inspect manually and finish by hand".
#[allow(clippy::too_many_lines)]
pub fn execute(root: &Path, plan: &ForgetPlan) -> Result<ExecuteOutcome> {
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
    let hash_state_path = kb_core::state_dir(root).join("hashes.json");
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

    // bn-i5r step 2: scrub the forgotten src from surviving concept pages.
    // Must run BEFORE the lexical index rebuild so the index sees the
    // post-scrub state.
    let frontmatter_scrubbed = match scrub_concept_frontmatter(root, &plan.src_id) {
        Ok(n) => n,
        Err(err) => {
            eprintln!(
                "warning: frontmatter scrub failed after forget: {err:#}"
            );
            0
        }
    };

    // bn-i5r step 1: regenerate the three index pages from current on-disk
    // state so `wiki/*/index.md` stops listing trashed entries.
    let index_pages_refreshed = match index_page::generate_indexes(root) {
        Ok(artifacts) => match index_page::persist_index_artifacts(&artifacts) {
            Ok(()) => true,
            Err(err) => {
                eprintln!(
                    "warning: index-page refresh failed after forget: {err:#}"
                );
                false
            }
        },
        Err(err) => {
            eprintln!(
                "warning: index-page refresh failed after forget: {err:#}"
            );
            false
        }
    };

    // bn-i5r step 3: rebuild the lexical search index from surviving wiki
    // pages so `kb ask` doesn't try to read a candidate that lives in
    // `trash/`.
    let lexical_index_refreshed = match kb_query::build_lexical_index(root) {
        Ok(index) => match index.save(root) {
            Ok(()) => true,
            Err(err) => {
                eprintln!(
                    "warning: lexical index rebuild failed after forget: {err:#}"
                );
                false
            }
        },
        Err(err) => {
            eprintln!(
                "warning: lexical index rebuild failed after forget: {err:#}"
            );
            false
        }
    };

    // bn-3f6 step 4: surgically prune `state/graph.json` so stale
    // `source-document-<src>` / `wiki-page-<src>` nodes don't linger until
    // the next full compile. Missing graph.json is fine (`load_from` returns
    // an empty Graph).
    let graph_nodes_pruned = match prune_graph_for_src(root, &plan.src_id) {
        Ok(n) => n,
        Err(err) => {
            eprintln!(
                "warning: graph.json prune failed after forget: {err:#}"
            );
            0
        }
    };

    // bn-3qsj step 5: eagerly drop the forgotten source's embedding row
    // (per design-doc open question 1, "eager forget"). The next compile
    // would prune it lazily anyway, but eager is cheaper to reason about
    // and avoids a forget→search window where stale semantic hits would
    // still surface. Missing embedding DB is fine — fresh kbs that never
    // ran compile have nothing to drop.
    if let Err(err) = drop_embedding_for_src(root, &plan.src_id) {
        eprintln!(
            "warning: embedding row drop failed after forget: {err:#}"
        );
    }

    Ok(ExecuteOutcome {
        backlinks_refreshed,
        cascade_refresh: CascadeRefresh {
            index_pages_refreshed,
            frontmatter_scrubbed,
            lexical_index_refreshed,
            graph_nodes_pruned,
        },
    })
}

/// Open the embedding store and delete the wiki source's row.
///
/// The embedding row is keyed by `wiki/sources/<filename>.md`, where
/// `<filename>` is the on-disk filename for the source (which may include
/// a slug suffix). We discover it by listing surviving wiki source pages
/// that begin with `<src_id>` — there should be at most one. After the
/// trash move ran above, the source's wiki page is already gone, so we
/// look for it via the trash bucket too as a fallback.
fn drop_embedding_for_src(root: &Path, src_id: &str) -> Result<()> {
    use rusqlite::params;
    let db_path = kb_query::embedding_db_path(root);
    if !db_path.exists() {
        return Ok(());
    }
    let conn = rusqlite::Connection::open(&db_path)
        .with_context(|| format!("open embedding db {}", db_path.display()))?;
    // Match either the bare `wiki/sources/<src_id>.md` form or the
    // slugged `wiki/sources/<src_id>-<slug>.md` form. SQLite GLOB makes
    // the prefix match readable.
    let pattern_bare = format!("wiki/sources/{src_id}.md");
    let pattern_slug = format!("wiki/sources/{src_id}-*.md");
    // bn-3rzz: per-chunk schema. One source can have multiple rows
    // (one per chunk) — the GLOB / `=` predicates on item_id strip them all.
    conn.execute(
        "DELETE FROM chunk_embeddings
         WHERE item_id = ?1
            OR item_id GLOB ?2",
        params![pattern_bare, pattern_slug],
    )
    .with_context(|| format!("delete embedding for {src_id}"))?;
    Ok(())
}

/// Load `state/graph.json`, remove every node referencing `src_id`, and
/// persist back atomically. Returns the number of nodes pruned (0 when the
/// file is absent or carries no matching nodes).
///
/// # Errors
///
/// Returns an error when the graph fails to load or persist (including the
/// post-prune `validate()` round-trip).
fn prune_graph_for_src(root: &Path, src_id: &str) -> Result<usize> {
    let graph_path = Graph::graph_path(root);
    if !graph_path.exists() {
        return Ok(0);
    }
    let mut graph = Graph::load_from(root)
        .with_context(|| format!("load {}", graph_path.display()))?;
    let removed = graph.prune_for_src(src_id);
    if removed == 0 {
        return Ok(0);
    }
    graph
        .persist_to(root)
        .with_context(|| format!("persist {} after prune", graph_path.display()))?;
    Ok(removed)
}

/// Walk `wiki/concepts/*.md` and remove `src_id` from each page's
/// `source_document_ids` frontmatter list when it still appears alongside
/// other ids. Returns the number of pages rewritten.
///
/// Pages where the list becomes empty (or is empty) are left alone — they
/// should have been moved to trash already by the cascade's
/// `orphaned_concept_pages` step. If we find one, we conservatively skip it
/// and let `kb lint` surface the discrepancy on the next run.
///
/// The write uses `write_frontmatter`, which goes through `atomic_write`, so
/// a crash mid-rewrite leaves the original file intact.
///
/// # Errors
///
/// Returns an error when the `wiki/concepts/` directory listing itself
/// fails. Per-file read / parse / write failures are logged as warnings and
/// the page is skipped.
fn scrub_concept_frontmatter(root: &Path, src_id: &str) -> Result<usize> {
    let pages = concept_pages_needing_scrub(root, src_id)?;
    let mut scrubbed = 0;
    for page in pages {
        match scrub_single_concept(&page, src_id) {
            Ok(true) => scrubbed += 1,
            Ok(false) => {}
            Err(err) => {
                eprintln!(
                    "warning: could not scrub {} : {err:#}",
                    page.display()
                );
            }
        }
    }
    Ok(scrubbed)
}

/// Rewrite a single concept page's `source_document_ids` to drop `src_id`.
///
/// Returns `Ok(true)` when the file was rewritten, `Ok(false)` when nothing
/// needed to change (the id was not present, or the rewrite would produce an
/// empty list — which means cascade should have trashed it already).
fn scrub_single_concept(page: &Path, src_id: &str) -> Result<bool> {
    let (mut frontmatter, body) = read_frontmatter(page)
        .with_context(|| format!("read frontmatter for {}", page.display()))?;
    let key = Value::String("source_document_ids".into());
    let Some(value) = frontmatter.get_mut(&key) else {
        return Ok(false);
    };
    let Value::Sequence(seq) = value else {
        return Ok(false);
    };
    let before = seq.len();
    seq.retain(|v| v.as_str() != Some(src_id));
    let after = seq.len();
    if before == after {
        return Ok(false);
    }
    // Empty list means cascade should have trashed this page; leave it alone
    // so the next `kb lint` run flags the discrepancy rather than us silently
    // producing a page with no grounding.
    if after == 0 {
        return Ok(false);
    }
    write_frontmatter(page, &frontmatter, body)
        .with_context(|| format!("write frontmatter for {}", page.display()))?;
    Ok(true)
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
    // Use the actual on-disk wiki source page filename when we can find one
    // — bn-nlw9 may have slug-suffixed it. Fall back to the id-only form so
    // the prompt still reads sensibly before the page has been compiled.
    let page_label = plan
        .moves
        .iter()
        .find_map(|p| {
            let rel = p.file_name()?.to_str()?;
            if rel.starts_with(&format!("{}.", plan.src_id))
                || rel.starts_with(&format!("{}-", plan.src_id))
            {
                Some(format!("wiki/sources/{rel}"))
            } else {
                None
            }
        })
        .unwrap_or_else(|| format!("wiki/sources/{}.md", plan.src_id));
    format!(
        "remove source '{label}' ({})? This will delete {page_label}. [y/N] ",
        plan.src_id
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

/// Render the "post-trash refresh" footer used by `--dry-run` and live runs.
///
/// The dry-run variant previews each step so users know what live execution
/// would do beyond the trash moves. The live variant reports what actually
/// happened (and distinguishes "ran" vs. "failed, warned, kept going" via the
/// fields in [`CascadeRefresh`]).
///
/// Returns an empty string when `refresh` is a no-op (e.g. `--dry-run` path
/// that computed 0 scrub candidates, or `execute` before any refresh ran).
#[must_use]
pub fn render_refresh_footer(plan: &ForgetPlan, refresh: &CascadeRefresh, dry_run: bool) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    if dry_run {
        let _ = writeln!(
            out,
            "  will also refresh: wiki/index.md, wiki/sources/index.md, wiki/concepts/index.md, wiki/questions/index.md, state/indexes/lexical.json, frontmatter scrub on {} page(s), graph.json prune of {} node(s)",
            refresh.frontmatter_scrubbed, refresh.graph_nodes_pruned,
        );
        // Moves / cascade bookkeeping already printed above; nothing else.
        let _ = &plan;
        return out;
    }
    if refresh.is_noop() {
        return out;
    }
    if refresh.index_pages_refreshed {
        out.push_str("  index pages refreshed\n");
    }
    if refresh.lexical_index_refreshed {
        out.push_str("  lexical index rebuilt\n");
    }
    if refresh.frontmatter_scrubbed > 0 {
        let _ = writeln!(
            out,
            "  scrubbed {} concept frontmatter page(s)",
            refresh.frontmatter_scrubbed
        );
    }
    if refresh.graph_nodes_pruned > 0 {
        let _ = writeln!(
            out,
            "  pruned {} graph.json node(s)",
            refresh.graph_nodes_pruned
        );
    }
    out
}

/// Build the `CascadeRefresh` preview shown by `--dry-run` — reports what
/// live execution *would* do without touching disk. Used by the CLI to drive
/// `render_refresh_footer(_, _, true)` and the `--json` payload.
///
/// `index_pages_refreshed` and `lexical_index_refreshed` are reported as
/// `true` in the preview because both passes run unconditionally on live
/// execute. `frontmatter_scrubbed` is the number of concept pages that would
/// be rewritten (0 when the cascade removes all cites).
///
/// # Errors
///
/// Returns an error when the scrub-candidate walk fails (missing
/// directories are tolerated; see [`concept_pages_needing_scrub`]).
pub fn preview_refresh(root: &Path, src_id: &str) -> Result<CascadeRefresh> {
    let frontmatter_scrubbed = concept_pages_needing_scrub(root, src_id)?.len();
    let graph_nodes_pruned = preview_graph_prune(root, src_id).unwrap_or(0);
    Ok(CascadeRefresh {
        index_pages_refreshed: true,
        frontmatter_scrubbed,
        lexical_index_refreshed: true,
        graph_nodes_pruned,
    })
}

/// Count how many `state/graph.json` nodes a live execute would prune without
/// modifying the file. Returns `Ok(0)` when the graph file is missing. Shared
/// between `preview_refresh` and any future diagnostic command.
fn preview_graph_prune(root: &Path, src_id: &str) -> Result<usize> {
    let graph_path = Graph::graph_path(root);
    if !graph_path.exists() {
        return Ok(0);
    }
    // Clone-and-prune so the on-disk file is untouched.
    let mut graph = Graph::load_from(root)?;
    Ok(graph.prune_for_src(src_id))
}

fn is_src_id(token: &str) -> bool {
    // terseid hashes are base36 (lowercase alphanumeric). Older KBs minted
    // 8-char hex ids; base36 is a strict superset of hex so this check
    // accepts both — legacy KBs continue to resolve until they regenerate.
    token.starts_with("src-")
        && token.len() > 4
        && token[4..]
            .chars()
            .all(|c| c.is_ascii_digit() || c.is_ascii_lowercase())
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
    fn is_src_id_accepts_base36_ids() {
        // New terseid-minted ids are base36 (lowercase alphanumeric).
        assert!(is_src_id("src-a7x"));
        assert!(is_src_id("src-a7x3q9"));
        // Legacy 8-hex ids still round-trip via the base36 superset.
        assert!(is_src_id("src-0639ebb0"));
        assert!(is_src_id("src-abcdef"));
        assert!(!is_src_id("src-"));
        assert!(!is_src_id("foo"));
        // Mixed case / punctuation is rejected: our minter always outputs
        // lowercase, and the upstream parser keeps the namespace tight.
        assert!(!is_src_id("src-ABC"));
        assert!(!is_src_id("src-a.b"));
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
        fs::create_dir_all(normalized_dir(root).join(src)).expect("create normalized");
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
        fs::create_dir_all(normalized_dir(root).join(src)).expect("create normalized");
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
        fs::create_dir_all(normalized_dir(root).join(src)).expect("create normalized");
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

        let normalized = normalized_dir(root).join(src);
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
        let outcome = execute(root, &plan).expect("execute forget");
        // bn-i5r: execute now reports the three post-trash refreshes. This
        // test only writes a wiki source page (no concept pages), so the
        // scrub counter stays at 0; the index/lexical rebuilds still run.
        assert!(outcome.cascade_refresh.index_pages_refreshed);
        assert!(outcome.cascade_refresh.lexical_index_refreshed);
        assert_eq!(outcome.cascade_refresh.frontmatter_scrubbed, 0);

        assert!(!normalized.exists(), "normalized dir should be gone");
        assert!(!raw.exists(), "raw/inbox dir should be gone");
        assert!(!wiki_page.exists(), "wiki page should be gone");
        assert!(plan.trash_dir.exists(), "trash dir should exist");
        assert!(
            plan.trash_dir
                .join(kb_core::KB_DIR)
                .join(format!("normalized/{src}"))
                .exists(),
            "normalized dir preserved under trash"
        );

        let reloaded = HashState::load_from_root(root).expect("reload hash state");
        assert!(
            !reloaded.hashes.contains_key(&format!("normalized/{src}")),
            "hash state should no longer list the forgotten src"
        );
    }

    // -- bn-i5r: post-trash structural refreshes ---------------------------

    #[test]
    fn concept_pages_needing_scrub_returns_only_multi_sourced_survivors() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let src = "src-i5r00001";
        let other = "src-i5r00002";
        // Orphan (single src) — should NOT appear; cascade will trash it.
        write_concept_page(root, "alone", &[src]);
        // Survivor (two srcs) — should appear.
        let survivor = write_concept_page(root, "shared", &[src, other]);
        // Unrelated (only other src) — should NOT appear.
        write_concept_page(root, "other", &[other]);

        let pages = concept_pages_needing_scrub(root, src).expect("scan");
        assert_eq!(pages, vec![survivor], "only multi-sourced survivors should need scrub; got {pages:?}");
    }

    #[test]
    fn scrub_single_concept_drops_forgotten_id_and_keeps_others() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let src = "src-i5r00010";
        let other = "src-i5r00011";
        let page = write_concept_page(root, "shared", &[src, other]);

        let rewrote = scrub_single_concept(&page, src).expect("scrub");
        assert!(rewrote, "scrub must report that it rewrote the file");

        let (fm, _body) = read_frontmatter(&page).expect("reread");
        let ids = frontmatter_source_ids(&fm);
        assert_eq!(ids, vec![other.to_string()]);
    }

    #[test]
    fn scrub_single_concept_leaves_empty_list_alone() {
        // Guard: cascade should have trashed the page; don't silently leave
        // a concept page with zero grounding behind.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let src = "src-i5r00020";
        let page = write_concept_page(root, "alone", &[src]);

        let rewrote = scrub_single_concept(&page, src).expect("scrub");
        assert!(!rewrote, "scrub must skip pages where the list becomes empty");
        // File content unchanged.
        let (fm, _body) = read_frontmatter(&page).expect("reread");
        let ids = frontmatter_source_ids(&fm);
        assert_eq!(ids, vec![src.to_string()]);
    }

    #[test]
    fn scrub_single_concept_is_noop_when_id_absent() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let kept = "src-i5r00030";
        let page = write_concept_page(root, "nope", &[kept]);

        let rewrote = scrub_single_concept(&page, "src-absent01").expect("scrub");
        assert!(!rewrote);
    }

    #[test]
    fn execute_refreshes_index_pages_and_lexical_index() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let src = "src-i5ra0001";
        let other = "src-i5ra0002";

        // Ingest-like layout: normalized + wiki source page.
        fs::create_dir_all(normalized_dir(root).join(src))
            .expect("mkdir normalized");
        let wiki_page = root.join(kb_compile::source_page::source_page_path_for_id(src));
        fs::create_dir_all(wiki_page.parent().expect("parent"))
            .expect("mkdir wiki/sources");
        fs::write(
            &wiki_page,
            format!("---\nid: x\ntitle: Forgotten\nsource_document_id: {src}\n---\n\n# Forgotten\n"),
        )
        .expect("write wiki source");

        // One orphan concept (single src) and one survivor (multi-sourced).
        let orphan = write_concept_page(root, "orphan", &[src]);
        let survivor = write_concept_page(root, "survivor", &[src, other]);

        let plan = plan(root, src, None, true).expect("plan");
        let outcome = execute(root, &plan).expect("execute");

        // The three bn-i5r refreshes ran.
        assert!(outcome.cascade_refresh.index_pages_refreshed);
        assert!(outcome.cascade_refresh.lexical_index_refreshed);
        assert_eq!(outcome.cascade_refresh.frontmatter_scrubbed, 1);

        // Orphan was trashed; survivor remains but no longer references src.
        assert!(!orphan.exists());
        assert!(survivor.exists());
        let (fm, _body) = read_frontmatter(&survivor).expect("reread survivor");
        let ids = frontmatter_source_ids(&fm);
        assert_eq!(ids, vec![other.to_string()], "survivor frontmatter must not list forgotten src");

        // Index pages exist and don't reference the trashed concept / source.
        let global = fs::read_to_string(root.join("wiki/index.md")).expect("read global index");
        assert!(!global.contains(&format!("sources/{src}.md")),
            "global index still lists trashed source:\n{global}");
        assert!(!global.contains("orphan.md"),
            "global index still lists trashed orphan concept:\n{global}");

        // Lexical index doesn't mention the trashed pages either.
        let lexical_json = fs::read_to_string(kb_core::state_dir(root).join("indexes/lexical.json"))
            .expect("read lexical index");
        assert!(!lexical_json.contains(&format!("wiki/sources/{src}.md")),
            "lexical index still points at trashed source page");
        assert!(!lexical_json.contains("orphan.md"),
            "lexical index still points at trashed orphan concept");
        // Survivor remains indexed.
        assert!(lexical_json.contains("wiki/concepts/survivor.md"),
            "lexical index must retain survivor page");
    }

    #[test]
    fn preview_refresh_counts_scrub_candidates() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let src = "src-i5rp0001";
        let other = "src-i5rp0002";
        write_concept_page(root, "alone", &[src]);
        write_concept_page(root, "shared", &[src, other]);

        let preview = preview_refresh(root, src).expect("preview");
        assert!(preview.index_pages_refreshed);
        assert!(preview.lexical_index_refreshed);
        assert_eq!(preview.frontmatter_scrubbed, 1);
    }

    // -- bn-3f6: expanded build-record matching ----------------------------

    /// Write a build record under `state/build_records/<id>.json` with the
    /// given id and output paths. Leaves every other metadata field at
    /// defaults, which is enough for [`scan_stale_build_records`] since it
    /// only inspects `id` and `output_paths`.
    fn write_build_record(root: &Path, id: &str, output_paths: &[&str]) -> PathBuf {
        let dir = build_records_dir(root);
        fs::create_dir_all(&dir).expect("mkdir build_records");
        let record = serde_json::json!({
            "metadata": {
                "id": id,
                "created_at_millis": 0u64,
                "updated_at_millis": 0u64,
                "source_hashes": [],
                "dependencies": [],
                "output_paths": output_paths,
                "status": "fresh",
            },
            "pass_name": "test",
            "input_ids": [],
            "output_ids": [],
            "manifest_hash": "m",
        });
        let path = dir.join(format!("{id}.json"));
        fs::write(&path, record.to_string()).expect("write build record");
        path
    }

    #[test]
    fn scan_stale_build_records_matches_source_summary_and_extract_concepts() {
        // bn-3f6: pass-11 found that `build:extract-concepts:<src>` was NOT
        // scooped up by the cascade because its `output_paths` points at
        // `state/concept_candidates/<src>.json`, not `normalized/<src>/` nor
        // `wiki/sources/<src>.md`. Both must be trashed after the fix.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let src = "src-3f600001";
        let other = "src-3f600002";

        // Source-summary record — caught today via `output_paths`.
        let summary = write_build_record(
            root,
            &format!("build:source-summary:{src}"),
            &[&format!("wiki/sources/{src}.md")],
        );
        // Extract-concepts record — caught ONLY after bn-3f6 (id suffix
        // match OR state/concept_candidates path match).
        let extract = write_build_record(
            root,
            &format!("build:extract-concepts:{src}"),
            &[&format!("state/concept_candidates/{src}.json")],
        );
        // Unrelated src — must NOT match.
        let unrelated = write_build_record(
            root,
            &format!("build:source-summary:{other}"),
            &[&format!("wiki/sources/{other}.md")],
        );

        let found = scan_stale_build_records(root, src).expect("scan");
        assert!(
            found.contains(&summary),
            "source-summary record must match; got {found:?}"
        );
        assert!(
            found.contains(&extract),
            "extract-concepts record must match; got {found:?}"
        );
        assert!(
            !found.contains(&unrelated),
            "unrelated src must NOT match: {found:?}"
        );
        assert_eq!(found.len(), 2, "only the two src-owned records match");
    }

    #[test]
    fn scan_stale_build_records_matches_future_pass_by_id_suffix() {
        // Future passes that emit `build:<pass>:<src>` with output paths
        // outside normalized/, wiki/sources/, and state/concept_candidates/
        // are still caught by the id-suffix rule — the whole point of the
        // bn-3f6 expansion is to be forward-compatible.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let src = "src-3f600010";
        let hypothetical = write_build_record(
            root,
            &format!("build:future-pass:{src}"),
            &["state/somewhere-else/blob.json"],
        );

        let found = scan_stale_build_records(root, src).expect("scan");
        assert_eq!(found, vec![hypothetical]);
    }

    // -- bn-3f6: graph.json surgical prune ---------------------------------

    #[test]
    fn execute_prunes_graph_json_nodes_referencing_src() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let src = "src-3f60a001";

        // Minimal on-disk layout so the forget plan has something to move.
        fs::create_dir_all(normalized_dir(root).join(src))
            .expect("mkdir normalized");
        let wiki_page = root.join(kb_compile::source_page::source_page_path_for_id(src));
        fs::create_dir_all(wiki_page.parent().expect("parent"))
            .expect("mkdir wiki/sources");
        fs::write(&wiki_page, "---\nid: x\n---\n").expect("write wiki page");

        // Persist a graph that references this src plus an unrelated doc.
        let mut graph = kb_compile::Graph::default();
        graph.record(
            [format!("source-document-{src}")],
            [format!("wiki-page-{src}")],
        );
        graph.record(
            ["source-document-src-keepalive"],
            ["wiki-page-src-keepalive"],
        );
        graph.persist_to(root).expect("persist graph");

        let plan = plan(root, src, None, true).expect("plan");
        let outcome = execute(root, &plan).expect("execute");

        // Both src-owned nodes went away in a single prune.
        assert_eq!(
            outcome.cascade_refresh.graph_nodes_pruned, 2,
            "source-document + wiki-page nodes must be pruned"
        );

        let reloaded = kb_compile::Graph::load_from(root).expect("reload graph");
        assert!(
            !reloaded.nodes.contains_key(&format!("source-document-{src}")),
            "source-document-{src} must be gone"
        );
        assert!(
            !reloaded.nodes.contains_key(&format!("wiki-page-{src}")),
            "wiki-page-{src} must be gone"
        );
        // Unrelated src survives — surgical, not a rebuild.
        assert!(reloaded.nodes.contains_key("source-document-src-keepalive"));
        assert!(reloaded.nodes.contains_key("wiki-page-src-keepalive"));
    }

    #[test]
    fn execute_tolerates_missing_graph_json() {
        // No graph.json on disk → prune reports 0 and forget still succeeds.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let src = "src-3f60a010";
        fs::create_dir_all(normalized_dir(root).join(src))
            .expect("mkdir normalized");

        let plan = plan(root, src, None, true).expect("plan");
        let outcome = execute(root, &plan).expect("execute");

        assert_eq!(outcome.cascade_refresh.graph_nodes_pruned, 0);
        assert!(!kb_compile::Graph::graph_path(root).exists());
    }

    #[test]
    fn preview_refresh_counts_graph_nodes() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let src = "src-3f60b001";

        let mut graph = kb_compile::Graph::default();
        graph.record(
            [format!("source-document-{src}")],
            [format!("wiki-page-{src}")],
        );
        graph.persist_to(root).expect("persist graph");

        let preview = preview_refresh(root, src).expect("preview");
        assert_eq!(preview.graph_nodes_pruned, 2);
        // Preview must NOT mutate the on-disk graph.
        let reloaded = kb_compile::Graph::load_from(root).expect("reload graph");
        assert!(reloaded.nodes.contains_key(&format!("source-document-{src}")));
        assert!(reloaded.nodes.contains_key(&format!("wiki-page-{src}")));
    }
}
