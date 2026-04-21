//! Canonical path helpers for the kb layout.
//!
//! Every caller that needs one of the "internal" directories (cache, logs,
//! state, trash, normalized, prompts) must route through these helpers so
//! the layout stays centralized. Browseable dirs (`raw/`, `wiki/`,
//! `outputs/`, `reviews/`) are intentionally *not* exposed here — they
//! remain at the vault root and are the user's to rearrange.
//!
//! The on-disk layout after this module landed is:
//!
//! ```text
//! <root>/
//! ├── .kb/               ← all internal plumbing
//! │   ├── cache/
//! │   ├── logs/
//! │   ├── state/
//! │   ├── trash/
//! │   ├── normalized/
//! │   └── prompts/       ← user prompt overrides, if any
//! ├── kb.toml
//! ├── raw/               ← original ingested inputs
//! ├── wiki/              ← compiled wiki pages
//! ├── outputs/           ← ask answers, slides, charts
//! └── reviews/           ← review queue items
//! ```
//!
//! Old layouts (pre-`.kb/`) are not read — `kb migrate` is the one-shot
//! bridge for upgrading existing vaults.

use std::path::{Path, PathBuf};

/// Name of the hidden directory that houses all internal plumbing.
pub const KB_DIR: &str = ".kb";

/// Trailing dir-name constants.
///
/// Exposed for the rare caller that only needs the leaf component (e.g. when
/// checking if a path *ends with* a legacy segment). Prefer the `*_dir(root)`
/// helpers for everything else.
pub const CACHE_SUBDIR: &str = "cache";
pub const LOGS_SUBDIR: &str = "logs";
pub const STATE_SUBDIR: &str = "state";
pub const TRASH_SUBDIR: &str = "trash";
pub const NORMALIZED_SUBDIR: &str = "normalized";
pub const PROMPTS_SUBDIR: &str = "prompts";

/// The six legacy segment names that `kb migrate` relocates into `.kb/`.
pub const LEGACY_MIGRATABLE_SUBDIRS: [&str; 6] = [
    CACHE_SUBDIR,
    LOGS_SUBDIR,
    STATE_SUBDIR,
    TRASH_SUBDIR,
    NORMALIZED_SUBDIR,
    PROMPTS_SUBDIR,
];

/// `<root>/.kb`
#[must_use]
pub fn kb_dir(root: &Path) -> PathBuf {
    root.join(KB_DIR)
}

/// `<root>/.kb/cache`
#[must_use]
pub fn cache_dir(root: &Path) -> PathBuf {
    kb_dir(root).join(CACHE_SUBDIR)
}

/// `<root>/.kb/logs`
#[must_use]
pub fn logs_dir(root: &Path) -> PathBuf {
    kb_dir(root).join(LOGS_SUBDIR)
}

/// `<root>/.kb/state`
#[must_use]
pub fn state_dir(root: &Path) -> PathBuf {
    kb_dir(root).join(STATE_SUBDIR)
}

/// `<root>/.kb/trash`
#[must_use]
pub fn trash_dir(root: &Path) -> PathBuf {
    kb_dir(root).join(TRASH_SUBDIR)
}

/// `<root>/.kb/normalized`
#[must_use]
pub fn normalized_dir(root: &Path) -> PathBuf {
    kb_dir(root).join(NORMALIZED_SUBDIR)
}

/// `<root>/.kb/prompts` — user template overrides for the LLM prompts.
#[must_use]
pub fn prompts_dir(root: &Path) -> PathBuf {
    kb_dir(root).join(PROMPTS_SUBDIR)
}

/// Relative path (rooted at the KB root) for the normalized dir of
/// `source_id`. Used when we need to store a portable reference in a
/// metadata record rather than an absolute path.
#[must_use]
pub fn normalized_rel(source_id: &str) -> PathBuf {
    PathBuf::from(KB_DIR)
        .join(NORMALIZED_SUBDIR)
        .join(source_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirs_are_all_under_dot_kb() {
        let root = Path::new("/tmp/example");
        assert_eq!(kb_dir(root), root.join(".kb"));
        assert_eq!(cache_dir(root), root.join(".kb/cache"));
        assert_eq!(logs_dir(root), root.join(".kb/logs"));
        assert_eq!(state_dir(root), root.join(".kb/state"));
        assert_eq!(trash_dir(root), root.join(".kb/trash"));
        assert_eq!(normalized_dir(root), root.join(".kb/normalized"));
        assert_eq!(prompts_dir(root), root.join(".kb/prompts"));
    }

    #[test]
    fn normalized_rel_joins_segments() {
        assert_eq!(
            normalized_rel("src-abc"),
            PathBuf::from(".kb").join("normalized").join("src-abc")
        );
    }

    #[test]
    fn legacy_subdirs_covers_all_internal_names() {
        assert!(LEGACY_MIGRATABLE_SUBDIRS.contains(&CACHE_SUBDIR));
        assert!(LEGACY_MIGRATABLE_SUBDIRS.contains(&LOGS_SUBDIR));
        assert!(LEGACY_MIGRATABLE_SUBDIRS.contains(&STATE_SUBDIR));
        assert!(LEGACY_MIGRATABLE_SUBDIRS.contains(&TRASH_SUBDIR));
        assert!(LEGACY_MIGRATABLE_SUBDIRS.contains(&NORMALIZED_SUBDIR));
        assert!(LEGACY_MIGRATABLE_SUBDIRS.contains(&PROMPTS_SUBDIR));
        assert_eq!(LEGACY_MIGRATABLE_SUBDIRS.len(), 6);
    }
}
