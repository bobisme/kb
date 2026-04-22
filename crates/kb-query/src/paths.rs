//! Path resolution helpers for question artifacts.
//!
//! Question output directories may live at either:
//!
//! - `outputs/questions/q-<id>/` — legacy, id-only layout, or
//! - `outputs/questions/q-<id>-<slug>/` — bn-nlw9 layout that appends a
//!   slugified snippet of the question text for browseability.
//!
//! All code that looks up a question directory by its stable `q-<id>` id
//! must route through [`resolve_question_dir`] rather than constructing
//! the path directly.

use std::path::{Path, PathBuf};

/// Resolve the on-disk question directory for `q_id`, returning `None` when
/// no matching directory exists.
///
/// Prefers the slug-suffixed form (`q-<id>-<slug>/`) when both forms
/// happen to coexist — normally they won't because the writer cleans up
/// during recompile, but during partial migration both may appear
/// temporarily. `q_id` is the stable id prefix (e.g. `q-abc`).
#[must_use]
pub fn resolve_question_dir(root: &Path, q_id: &str) -> Option<PathBuf> {
    let dir = root.join("outputs/questions");
    if !dir.exists() {
        return None;
    }
    let id_prefix = format!("{q_id}-");
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let Ok(name) = entry.file_name().into_string() else {
                continue;
            };
            let Ok(ft) = entry.file_type() else { continue };
            if ft.is_dir() && name.starts_with(&id_prefix) {
                return Some(entry.path());
            }
        }
    }
    let bare = dir.join(q_id);
    if bare.is_dir() {
        return Some(bare);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn prefers_slug_form() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        fs::create_dir_all(root.join("outputs/questions/q-abc-hello-world")).expect("mkdir");
        let resolved = resolve_question_dir(root, "q-abc").expect("resolved");
        assert_eq!(
            resolved,
            root.join("outputs/questions/q-abc-hello-world")
        );
    }

    #[test]
    fn falls_back_to_id_only() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        fs::create_dir_all(root.join("outputs/questions/q-abc")).expect("mkdir");
        let resolved = resolve_question_dir(root, "q-abc").expect("resolved");
        assert_eq!(resolved, root.join("outputs/questions/q-abc"));
    }

    #[test]
    fn returns_none_when_missing() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        assert!(resolve_question_dir(root, "q-missing").is_none());
    }

    #[test]
    fn returns_none_when_only_prefix_collision() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        fs::create_dir_all(root.join("outputs/questions/q-abcdef-hello")).expect("mkdir");
        // `q-abc` is a prefix of `q-abcdef` but should NOT resolve because
        // the `-` separator after `q-abc` is what we look for.
        let resolved = resolve_question_dir(root, "q-abc");
        assert!(
            resolved.is_none(),
            "resolver must not match unrelated q ids by string prefix"
        );
    }
}
