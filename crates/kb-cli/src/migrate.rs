//! `kb migrate` — one-shot, explicit upgrade from the pre-`.kb/` layout.
//!
//! Older kb vaults put `cache/`, `logs/`, `state/`, `trash/`, `normalized/`,
//! and `prompts/` at the root alongside the browseable tree (`raw/`, `wiki/`,
//! `outputs/`, `reviews/`). bn-2xbq moves every one of those into a hidden
//! `.kb/` directory so the vault root only shows user-facing content.
//!
//! This command is the bridge. It is **opt-in**: no other command will touch
//! a legacy layout. Every mutating command (`compile`, `ask`, `ingest`, …)
//! instead refuses to run and points the user here — see
//! [`bail_if_legacy_layout`]. The move is an atomic `std::fs::rename` per
//! directory, which costs O(1) on a same-filesystem vault.
//!
//! Idempotent: if the layout is already current, we print a note and exit 0.

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

#[derive(Debug, Clone, Serialize)]
pub struct MigrateReport {
    pub already_migrated: bool,
    pub moved: Vec<MoveRecord>,
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

    if legacy_present.is_empty() {
        if json {
            emit_json("migrate", MigrateReport {
                already_migrated: true,
                moved: Vec::new(),
            })?;
        } else {
            println!("Already migrated: no legacy directories found at {}", root.display());
        }
        return Ok(());
    }

    let kb_root = kb_core::kb_dir(root);
    fs::create_dir_all(&kb_root)
        .with_context(|| format!("create {} directory", kb_root.display()))?;

    let mut moved = Vec::new();
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

    if json {
        emit_json("migrate", MigrateReport {
            already_migrated: false,
            moved,
        })?;
    } else {
        for mv in &moved {
            let subdir = &mv.subdir;
            let kb = kb_core::KB_DIR;
            println!("Moved {subdir}/ → {kb}/{subdir}/");
        }
        println!(
            "Migrated {} directory/directories into {}/",
            moved.len(),
            kb_root.display()
        );
    }

    Ok(())
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
}
