//! `kb migrate` — one-shot, explicit upgrade from the pre-`.kb/` layout
//! AND (bn-nlw9) from id-only browseable filenames to slug-suffixed names.
//!
//! Older kb vaults put `cache/`, `logs/`, `state/`, `trash/`, `normalized/`,
//! and `prompts/` at the root alongside the browseable tree (`raw/`, `wiki/`,
//! `outputs/`, `reviews/`). bn-2xbq moves every one of those into a hidden
//! `.kb/` directory so the vault root only shows user-facing content.
//!
//! bn-nlw9 adds a second, independent pass: rename
//! `wiki/sources/src-<id>.md` → `wiki/sources/src-<id>-<title-slug>.md`
//! and
//! `outputs/questions/q-<id>/` → `outputs/questions/q-<id>-<question-slug>/`
//! so browseable artifacts read like their contents. Titles come from
//! frontmatter (`title:`) and `question.json` respectively.
//!
//! This command is the bridge. It is **opt-in**: no other command will touch
//! a legacy layout. Every mutating command (`compile`, `ask`, `ingest`, …)
//! instead refuses to run and points the user here — see
//! [`bail_if_legacy_layout`]. The move is an atomic `std::fs::rename` per
//! directory, which costs O(1) on a same-filesystem vault.
//!
//! Idempotent: if the layout is already current, we print a note and exit 0.
//! Re-running after a successful bn-nlw9 rename is a no-op because the
//! scanner skips entries that already carry a slug suffix.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::emit_json;

/// Per-directory migration outcome. Written into the JSON envelope under
/// `data.moves` and also printed as human-readable lines.
#[derive(Debug, Clone, Serialize)]
pub struct MoveRecord {
    pub subdir: String,
    pub from: String,
    pub to: String,
}

/// bn-nlw9 rename outcome: either a source wiki page file or a question
/// output directory.
#[derive(Debug, Clone, Serialize)]
pub struct RenameRecord {
    pub kind: String,
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct MigrateReport {
    pub already_migrated: bool,
    pub moved: Vec<MoveRecord>,
    /// bn-nlw9 source/question renames. Empty when the browseable layout
    /// is already slug-augmented or when no sources/questions exist.
    #[serde(default)]
    pub renamed: Vec<RenameRecord>,
    /// `true` when the backlinks refresh after renames succeeded (or when
    /// no renames happened so nothing needed refreshing).
    #[serde(default)]
    pub backlinks_refreshed: bool,
}

/// Entry point for `kb migrate`.
///
/// # Errors
///
/// Returns an error when:
/// - `<root>/.kb/<subdir>` already exists for a legacy dir we want to move
///   (partial prior migration — the user must resolve manually), or
/// - `std::fs::rename` fails (typically cross-device moves or EBUSY).
pub fn run_migrate(root: &Path, json: bool) -> Result<()> {
    let legacy_present = detect_legacy(root);

    // Phase A: .kb/ layout migration (bn-2xbq). Atomic rename per dir.
    let mut moved = Vec::new();
    if !legacy_present.is_empty() {
        let kb_root = kb_core::kb_dir(root);
        fs::create_dir_all(&kb_root)
            .with_context(|| format!("create {} directory", kb_root.display()))?;

        for subdir in &legacy_present {
            let from = root.join(subdir);
            let to = kb_root.join(subdir);

            if to.exists() {
                bail!(
                    "both {} and {} exist — refusing to clobber. \
                     Merge or delete one manually and rerun `kb migrate`.",
                    from.display(),
                    to.display()
                );
            }

            fs::rename(&from, &to).with_context(|| {
                format!(
                    "rename {} -> {} (if the .kb tree is on a different filesystem, \
                     move the directory manually)",
                    from.display(),
                    to.display()
                )
            })?;

            moved.push(MoveRecord {
                subdir: (*subdir).to_string(),
                from: from.display().to_string(),
                to: to.display().to_string(),
            });
        }
    }

    // Phase B: bn-nlw9 browseable-filename slugging. Runs every time so a
    // fresh-from-legacy vault picks up both rewrites in one `kb migrate`.
    let renamed = rename_sources_and_questions(root)?;
    let backlinks_refreshed = if renamed.is_empty() {
        true
    } else {
        refresh_backlinks(root)
    };

    let already_migrated = moved.is_empty() && renamed.is_empty();

    if json {
        emit_json("migrate", MigrateReport {
            already_migrated,
            moved,
            renamed,
            backlinks_refreshed,
        })?;
    } else if already_migrated {
        println!(
            "Already migrated: no legacy directories and no unslugged \
             sources/questions at {}",
            root.display()
        );
    } else {
        for mv in &moved {
            let subdir = &mv.subdir;
            let kb = kb_core::KB_DIR;
            println!("Moved {subdir}/ → {kb}/{subdir}/");
        }
        if !moved.is_empty() {
            println!(
                "Migrated {} directory/directories into {}/",
                moved.len(),
                kb_core::kb_dir(root).display()
            );
        }
        for rn in &renamed {
            println!("Renamed {} ({} → {})", rn.kind, rn.from, rn.to);
        }
        if !renamed.is_empty() {
            println!("Renamed {} browseable artifact(s).", renamed.len());
            if backlinks_refreshed {
                println!("Refreshed backlinks across concept/source pages.");
            } else {
                println!(
                    "Warning: backlinks refresh failed — rerun `kb compile` \
                     to reconcile concept pages."
                );
            }
        }
    }

    Ok(())
}

/// Walk `wiki/sources/` and `outputs/questions/` and rename legacy id-only
/// entries to the bn-nlw9 `<id>-<slug>` form. Idempotent: entries already
/// carrying a slug suffix are left alone.
///
/// Title/question lookup strategy:
///
/// - Sources: parse the wiki page's YAML frontmatter and read `title:`.
/// - Questions: read `question.json` → `raw_query`.
///
/// When the title slugs to empty, the entry is skipped (id-only is the
/// correct rendering). When the title can't be read (no frontmatter, no
/// question.json, malformed YAML) we also skip — the entry is left as-is
/// rather than guessed, and the next `kb compile`/`kb ask` will rewrite it
/// with a slug the proper way.
fn rename_sources_and_questions(root: &Path) -> Result<Vec<RenameRecord>> {
    let mut renamed = Vec::new();

    // Sources.
    let sources_dir = root.join("wiki/sources");
    if sources_dir.exists() {
        for entry in fs::read_dir(&sources_dir)
            .with_context(|| format!("read {}", sources_dir.display()))?
        {
            let entry =
                entry.with_context(|| format!("read entry in {}", sources_dir.display()))?;
            let path = entry.path();
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            if path.extension().and_then(|s| s.to_str()) != Some("md") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if stem == "index" {
                continue;
            }
            // Only rename id-only stems (`src-<hash>` with no trailing `-<slug>`).
            // After bn-nlw9, anything already carrying a dash after the id
            // hash is the new slugged form.
            if !looks_like_id_only_src_stem(stem) {
                continue;
            }
            let Some(title) = read_source_title(&path) else {
                continue;
            };
            let slug = kb_core::slug_for_filename(
                &title,
                kb_core::DEFAULT_FILENAME_SLUG_MAX_CHARS,
            );
            if slug.is_empty() {
                continue;
            }
            let new_name = format!("{stem}-{slug}.md");
            let new_path = sources_dir.join(&new_name);
            if new_path.exists() {
                // Shouldn't happen on a sane vault — the old id-only file
                // AND a slugged file for the same id? Skip rather than clobber.
                continue;
            }
            fs::rename(&path, &new_path).with_context(|| {
                format!("rename {} -> {}", path.display(), new_path.display())
            })?;
            renamed.push(RenameRecord {
                kind: "source".to_string(),
                from: relative_for_display(root, &path),
                to: relative_for_display(root, &new_path),
            });
        }
    }

    // Questions.
    let questions_dir = root.join("outputs/questions");
    if questions_dir.exists() {
        for entry in fs::read_dir(&questions_dir)
            .with_context(|| format!("read {}", questions_dir.display()))?
        {
            let entry =
                entry.with_context(|| format!("read entry in {}", questions_dir.display()))?;
            let path = entry.path();
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            if !looks_like_id_only_q_name(name) {
                continue;
            }
            let Some(text) = read_question_text(&path) else {
                continue;
            };
            let slug = kb_core::slug_for_filename(
                &text,
                kb_core::DEFAULT_FILENAME_SLUG_MAX_CHARS,
            );
            if slug.is_empty() {
                continue;
            }
            let new_name = format!("{name}-{slug}");
            let new_path = questions_dir.join(&new_name);
            if new_path.exists() {
                continue;
            }
            fs::rename(&path, &new_path).with_context(|| {
                format!("rename {} -> {}", path.display(), new_path.display())
            })?;
            renamed.push(RenameRecord {
                kind: "question".to_string(),
                from: relative_for_display(root, &path),
                to: relative_for_display(root, &new_path),
            });
        }
    }

    Ok(renamed)
}

/// Regenerate backlinks after bn-nlw9 renames so concept pages' managed
/// `backlinks` regions point at the new filenames.
///
/// Best-effort: returns `true` on success, `false` on failure. A compile
/// run will reconcile anyway; migration shouldn't hard-fail just because
/// one wiki page couldn't be re-read.
fn refresh_backlinks(root: &Path) -> bool {
    match kb_compile::backlinks::run_backlinks_pass(root) {
        Ok(artifacts) => {
            let to_persist: Vec<_> = artifacts
                .into_iter()
                .filter(kb_compile::backlinks::BacklinksArtifact::needs_update)
                .collect();
            match kb_compile::backlinks::persist_backlinks_artifacts(&to_persist) {
                Ok(()) => true,
                Err(err) => {
                    tracing::warn!("migrate: persist backlinks failed: {err}");
                    false
                }
            }
        }
        Err(err) => {
            tracing::warn!("migrate: backlinks pass failed: {err}");
            false
        }
    }
}

/// `src-<hash>` with no trailing `-<slug>` (i.e. dashes only inside the
/// `src-` prefix). `src-1wz` matches, `src-1wz-hello` does not.
fn looks_like_id_only_src_stem(stem: &str) -> bool {
    let Some(rest) = stem.strip_prefix("src-") else {
        return false;
    };
    if rest.is_empty() {
        return false;
    }
    !rest.contains('-')
}

/// `q-<hash>` with no trailing `-<slug>`. Same rule as the src checker.
fn looks_like_id_only_q_name(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("q-") else {
        return false;
    };
    if rest.is_empty() {
        return false;
    }
    !rest.contains('-')
}

/// Parse YAML frontmatter out of a wiki source markdown file and return the
/// `title:` field, trimmed. Returns `None` when the file can't be read,
/// lacks a frontmatter block, or the `title` is empty.
fn read_source_title(path: &Path) -> Option<String> {
    let raw = fs::read_to_string(path).ok()?;
    let rest = raw
        .strip_prefix("---\n")
        .or_else(|| raw.strip_prefix("---\r\n"))?;
    let mut yaml = String::new();
    for line in rest.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed == "---" {
            break;
        }
        yaml.push_str(line);
    }
    let parsed: serde_yaml::Value = serde_yaml::from_str(&yaml).ok()?;
    let title = parsed.get("title")?.as_str()?.trim().to_string();
    if title.is_empty() { None } else { Some(title) }
}

/// Pull the question text out of `question.json` — the `raw_query` field —
/// and return it. Returns `None` when the file is missing or malformed.
fn read_question_text(q_dir: &Path) -> Option<String> {
    let q_file = q_dir.join("question.json");
    let raw = fs::read_to_string(&q_file).ok()?;
    let parsed: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let text = parsed.get("raw_query")?.as_str()?.trim().to_string();
    if text.is_empty() { None } else { Some(text) }
}

/// Format a path for display in the migration log, preferring a relative
/// path under `root` when possible.
fn relative_for_display(root: &Path, path: &Path) -> String {
    path.strip_prefix(root).map_or_else(
        |_| path.display().to_string(),
        |p| p.to_string_lossy().replace('\\', "/"),
    )
}


/// Return the list of legacy subdir names that still exist at the root.
/// Does **not** filter on whether `.kb/<subdir>` also exists — callers
/// that see the same name on both sides should bail, because a partial
/// migration shouldn't silently look "done". Ordering matches
/// `kb_core::LEGACY_MIGRATABLE_SUBDIRS`.
#[allow(clippy::redundant_pub_crate)]
pub(crate) fn detect_legacy(root: &Path) -> Vec<&'static str> {
    kb_core::LEGACY_MIGRATABLE_SUBDIRS
        .iter()
        .copied()
        .filter(|subdir| root.join(subdir).is_dir())
        .collect()
}

/// Bail with an actionable error when the root still looks like the
/// pre-`.kb/` layout. Called at the entry of every mutating command —
/// stale state won't silently break or corrupt a compile.
///
/// # Errors
///
/// Returns an `Err(anyhow)` when any legacy dir is present without its
/// `.kb/` counterpart.
pub fn bail_if_legacy_layout(root: &Path) -> Result<()> {
    let legacy = detect_legacy(root);
    if legacy.is_empty() {
        return Ok(());
    }
    bail!(
        "detected pre-.kb/ layout ({} still at the vault root). \
         Run `kb migrate` to upgrade this kb in place.",
        legacy.join("/, ") + "/"
    );
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn write_legacy(root: &Path) {
        for sub in kb_core::LEGACY_MIGRATABLE_SUBDIRS {
            let dir = root.join(sub);
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("marker.txt"), sub.as_bytes()).unwrap();
        }
    }

    #[test]
    fn migrate_moves_every_legacy_dir_into_kb() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        write_legacy(root);

        run_migrate(root, false).unwrap();

        for sub in kb_core::LEGACY_MIGRATABLE_SUBDIRS {
            assert!(!root.join(sub).exists(), "{sub} should have moved");
            let moved = kb_core::kb_dir(root).join(sub);
            assert!(moved.is_dir(), "{sub} should land under .kb/");
            let marker = moved.join("marker.txt");
            let contents = fs::read_to_string(&marker).unwrap();
            assert_eq!(contents, sub);
        }
    }

    #[test]
    fn migrate_is_idempotent_on_current_layout() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        // Already-migrated layout: .kb/ exists, nothing at root level.
        fs::create_dir_all(kb_core::kb_dir(root).join("state")).unwrap();

        run_migrate(root, false).unwrap();
        // Second run must also be a no-op.
        run_migrate(root, false).unwrap();
    }

    #[test]
    fn migrate_refuses_to_clobber_existing_target() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        fs::create_dir_all(root.join("state")).unwrap();
        fs::create_dir_all(kb_core::kb_dir(root).join("state")).unwrap();

        let err = run_migrate(root, false).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("refusing to clobber"), "got: {msg}");
    }

    #[test]
    fn bail_if_legacy_layout_detects_partial_legacy() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        fs::create_dir_all(root.join("state")).unwrap();
        let err = bail_if_legacy_layout(root).unwrap_err();
        assert!(err.to_string().contains("pre-.kb/ layout"));
    }

    #[test]
    fn bail_if_legacy_layout_passes_current_layout() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        fs::create_dir_all(kb_core::kb_dir(root).join("state")).unwrap();
        bail_if_legacy_layout(root).unwrap();
    }

    #[test]
    fn bail_if_legacy_layout_passes_empty_root() {
        let dir = tempdir().unwrap();
        bail_if_legacy_layout(dir.path()).unwrap();
    }

    // bn-nlw9: phase B tests — rename id-only wiki sources and question
    // directories into id-slug form, and stay idempotent on a second run.

    fn write_source_page(root: &Path, src_id: &str, title: &str) {
        let dir = root.join("wiki/sources");
        fs::create_dir_all(&dir).unwrap();
        let md = format!(
            "---\nid: wiki-source-{src_id}\ntype: source\ntitle: {title}\n\
source_document_id: {src_id}\nsource_revision_id: rev-1\ngenerated_at: 0\n\
build_record_id: build-1\n---\n\n# Source\n",
        );
        fs::write(dir.join(format!("{src_id}.md")), md).unwrap();
    }

    fn write_question(root: &Path, q_id: &str, text: &str) {
        let dir = root.join("outputs/questions").join(q_id);
        fs::create_dir_all(&dir).unwrap();
        let payload = serde_json::json!({
            "metadata": { "id": q_id },
            "raw_query": text,
        });
        fs::write(dir.join("question.json"), payload.to_string()).unwrap();
    }

    #[test]
    fn migrate_renames_id_only_sources_to_id_slug_form() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        write_source_page(root, "src-abc", "Hello World Of Rust");
        write_source_page(root, "src-xyz", "Another Doc");

        run_migrate(root, false).unwrap();

        assert!(!root.join("wiki/sources/src-abc.md").exists());
        assert!(
            root.join("wiki/sources/src-abc-hello-world-of-rust.md").exists(),
            "expected slugged source page"
        );
        assert!(root.join("wiki/sources/src-xyz-another-doc.md").exists());
    }

    #[test]
    fn migrate_renames_id_only_question_dirs() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        write_question(root, "q-abc", "Produce a mermaid graph of USB team");
        write_question(root, "q-xyz", "List rust lints");

        run_migrate(root, false).unwrap();

        assert!(!root.join("outputs/questions/q-abc").exists());
        assert!(
            root.join("outputs/questions/q-abc-produce-a-mermaid-graph-of-usb-team")
                .is_dir(),
            "expected slugged question dir"
        );
        assert!(
            root.join("outputs/questions/q-xyz-list-rust-lints").is_dir()
        );
    }

    #[test]
    fn migrate_phase_b_is_idempotent() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        write_source_page(root, "src-abc", "Hello");
        write_question(root, "q-abc", "world");

        run_migrate(root, false).unwrap();
        // Second run: no-op.
        run_migrate(root, false).unwrap();

        assert!(root.join("wiki/sources/src-abc-hello.md").exists());
        assert!(root.join("outputs/questions/q-abc-world").is_dir());
    }

    #[test]
    fn migrate_phase_b_skips_empty_title_sources() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        // Title that slugs to empty — the file must be left alone (id-only
        // is the correct rendering for an empty slug).
        write_source_page(root, "src-empty", "!!!");

        run_migrate(root, false).unwrap();

        assert!(
            root.join("wiki/sources/src-empty.md").exists(),
            "empty-slug source must not be renamed"
        );
    }

    #[test]
    fn migrate_phase_b_skips_when_no_frontmatter() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let src_dir = root.join("wiki/sources");
        fs::create_dir_all(&src_dir).unwrap();
        // Plain markdown, no frontmatter block.
        fs::write(src_dir.join("src-plain.md"), "# Just a page\n").unwrap();

        run_migrate(root, false).unwrap();

        assert!(
            src_dir.join("src-plain.md").exists(),
            "source without readable frontmatter must not be renamed"
        );
    }

    #[test]
    fn migrate_phase_b_skips_already_slugged() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let src_dir = root.join("wiki/sources");
        fs::create_dir_all(&src_dir).unwrap();
        fs::write(
            src_dir.join("src-abc-existing-slug.md"),
            "---\ntitle: Something Else\n---\n",
        )
        .unwrap();

        run_migrate(root, false).unwrap();

        assert!(src_dir.join("src-abc-existing-slug.md").exists());
        assert!(!src_dir.join("src-abc-something-else.md").exists());
    }

    #[test]
    fn migrate_phase_b_after_legacy_layout_migration() {
        // Full end-to-end: legacy root + legacy names → fully migrated.
        let dir = tempdir().unwrap();
        let root = dir.path();
        write_legacy(root);
        write_source_page(root, "src-abc", "Combined Migration");
        write_question(root, "q-abc", "combined migration");

        run_migrate(root, false).unwrap();

        for sub in kb_core::LEGACY_MIGRATABLE_SUBDIRS {
            assert!(!root.join(sub).exists(), "{sub} should have moved");
            assert!(kb_core::kb_dir(root).join(sub).is_dir());
        }
        assert!(
            root.join("wiki/sources/src-abc-combined-migration.md").exists()
        );
        assert!(
            root.join("outputs/questions/q-abc-combined-migration").is_dir()
        );
    }
}
