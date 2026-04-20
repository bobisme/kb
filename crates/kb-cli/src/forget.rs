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
//!   `run_backlinks_pass` so concept pages stop linking to a dead URL. We
//!   don't cascade-delete orphaned concept pages (bn-1fq F3 scope); `kb
//!   lint` already surfaces unreferenced concepts.
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
use serde::Serialize;

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
/// # Errors
///
/// Returns an error when the system clock is before `UNIX_EPOCH`
/// (unreachable on sane hosts; used only to derive the trash timestamp).
pub fn plan(root: &Path, src_id: &str, origin: Option<String>) -> Result<ForgetPlan> {
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

    Ok(ForgetPlan {
        src_id: src_id.to_string(),
        origin,
        trash_dir,
        moves,
    })
}

/// Execute a previously-built `ForgetPlan`, moving real bytes on disk.
///
/// Each entry in `plan.moves` is moved into `plan.trash_dir` under its
/// original basename (e.g. `normalized/src-XXXX` → `trash/src-XXXX-T/normalized/`,
/// `wiki/sources/src-XXXX.md` → `trash/src-XXXX-T/wiki/sources/src-XXXX.md`).
/// Layout preserves the original parent directory name so the trash dir is
/// self-describing — a user poking at `trash/src-XXXX-1700000000/` can tell
/// which bucket each file came from.
///
/// After moves succeed, the `normalized/<id>` key is dropped from
/// `state/hashes.json` (if present) and `run_backlinks_pass` is invoked so
/// concept pages no longer link into a deleted source. Backlink refresh is
/// best-effort: a failure emits a warning and returns success, because the
/// user has already agreed to the destructive operation and rerunning `kb
/// compile` will correct any lingering staleness.
///
/// F3 note: cascade-deleting concept pages whose only source was the
/// forgotten one is explicitly out of scope. `kb lint` already surfaces
/// unreferenced concepts; future work can promote that into an automatic
/// cascade when user feedback warrants it.
///
/// # Errors
///
/// Returns an error when a move fails or the hash-state file cannot be
/// updated. Partial moves are NOT rolled back — callers should treat a
/// failure as "inspect manually and finish by hand".
pub fn execute(root: &Path, plan: &ForgetPlan) -> Result<bool> {
    fs::create_dir_all(&plan.trash_dir)
        .with_context(|| format!("create trash dir {}", plan.trash_dir.display()))?;

    for src in &plan.moves {
        let relative = src
            .strip_prefix(root)
            .unwrap_or(src)
            .to_path_buf();
        let dest = plan.trash_dir.join(&relative);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create trash subdir {}", parent.display()))?;
        }
        // `fs::rename` works across the same filesystem and is atomic; the
        // KB root and its trash/ subdir are always on the same mount so
        // we don't need fallback-copy semantics here.
        fs::rename(src, &dest).with_context(|| {
            format!("move {} → {}", src.display(), dest.display())
        })?;
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
/// # Errors
///
/// Returns an error only if writing to the in-memory `String` buffer fails,
/// which is not expected in practice. Surfaced as `Result` so callers can
/// decide whether to `?` or panic.
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
    if plan.moves.is_empty() {
        out.push_str(
            "  nothing to move — no normalized/, raw/inbox/, or wiki/sources/ entry found\n",
        );
        return out;
    }
    for src in &plan.moves {
        let _ = writeln!(out, "  {verb}: {}", src.display());
    }
    let _ = writeln!(out, "  trash:    {}", plan.trash_dir.display());
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
        let plan = plan(root, src, None).expect("plan");
        assert_eq!(plan.src_id, src);
        assert_eq!(plan.moves.len(), 1);
        assert!(plan.moves[0].ends_with(format!("normalized/{src}")));
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

        let plan = plan(root, src, None).expect("plan");
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
