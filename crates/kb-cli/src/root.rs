use std::{
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};

const KB_DIR: &str = "kb";
const KB_CONFIG: &str = "kb.toml";

/// Handle representing a discovered KB root directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KbRoot {
    pub path: PathBuf,
}

/// Resolve the KB root for command execution.
///
/// Resolution order:
/// 1. `--root` override (if set) — accepts either a dir with `kb.toml` directly,
///    or a dir containing a `kb/` subdir with `kb.toml`.
/// 2. Walk up from the current directory. At each ancestor, try
///    `<ancestor>/kb.toml` first (the ancestor *is* the KB root), then
///    `<ancestor>/kb/kb.toml` (project layout with a `kb/` subdir).
/// 3. Fall back to `~/kb` when present.
/// 4. Fail with an actionable error message.
pub fn discover_root(explicit_root: Option<&Path>) -> Result<KbRoot> {
    let cwd = env::current_dir().context("failed to resolve current working directory")?;
    let home = env::var("HOME").context("missing HOME environment variable")?;
    discover_root_in(explicit_root, &cwd, Path::new(&home))
}

fn discover_root_in(explicit_root: Option<&Path>, cwd: &Path, home: &Path) -> Result<KbRoot> {
    if let Some(root_override) = explicit_root {
        return validate_candidate(&resolve_root_override(root_override, cwd)?);
    }

    for ancestor in cwd.ancestors() {
        if let Some(path) = find_kb_root_at(ancestor) {
            return Ok(KbRoot { path });
        }
    }

    if is_kb_root(&home.join(KB_DIR)) {
        return Ok(KbRoot {
            path: home.join(KB_DIR),
        });
    }
    if is_kb_root(home) {
        return Ok(KbRoot {
            path: home.to_path_buf(),
        });
    }

    Err(anyhow!(
        "no KB root found.\n  \
         Looked for '{KB_CONFIG}' (global-style root) and '{KB_DIR}/{KB_CONFIG}' (project-style root) from '{}' up to filesystem root.\n  \
         Run 'kb init' here, or set '--root <path>'.",
        cwd.display(),
    ))
}

/// Look for a KB root at `dir`. Prefers `<dir>/kb.toml` (dir is the root),
/// falls back to `<dir>/kb/kb.toml` (project layout).
fn find_kb_root_at(dir: &Path) -> Option<PathBuf> {
    if is_kb_root(dir) {
        return Some(dir.to_path_buf());
    }
    let subdir = dir.join(KB_DIR);
    if is_kb_root(&subdir) {
        return Some(subdir);
    }
    None
}

fn resolve_root_override(explicit_root: &Path, cwd: &Path) -> Result<PathBuf> {
    let resolved = if explicit_root.is_absolute() {
        explicit_root.to_path_buf()
    } else {
        cwd.join(explicit_root)
    };

    if !resolved.is_dir() {
        return Err(anyhow!(
            "'--root {}' is not an existing directory.\n  \
             Run 'kb init {}' to initialize a KB root there.",
            resolved.display(),
            resolved.display(),
        ));
    }

    fs::canonicalize(&resolved)
        .with_context(|| format!("failed to resolve --root path '{}'", resolved.display()))
}

fn validate_candidate(path: &Path) -> Result<KbRoot> {
    find_kb_root_at(path).map_or_else(
        || {
            Err(anyhow!(
                "'{}' is not a KB root.\n  \
                 Looked for '{}/{KB_CONFIG}' (global-style root) or '{}/{KB_DIR}/{KB_CONFIG}' (project-style root).\n  \
                 Run 'kb init {}' to initialize one.",
                path.display(),
                path.display(),
                path.display(),
                path.display(),
            ))
        },
        |found| Ok(KbRoot { path: found }),
    )
}

fn is_kb_root(candidate: &Path) -> bool {
    candidate.join(KB_CONFIG).is_file()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn touch_kb_toml(path: &Path) {
        fs::create_dir_all(path).expect("create kb root");
        fs::write(path.join(KB_CONFIG), b"\n").expect("write kb config");
    }

    #[test]
    fn root_override_is_used() {
        let tmp = tempdir().expect("temp root");
        let repo = tmp.path().join("repo");
        let explicit = repo.join("forced");
        touch_kb_toml(&explicit);

        let cwd = repo.join("work");
        fs::create_dir_all(&cwd).expect("create cwd");

        let result = discover_root_in(Some(explicit.as_path()), &cwd, Path::new("/home/missing"));

        assert_eq!(
            result.expect("discover root").path,
            explicit.canonicalize().expect("canonicalize")
        );
    }

    #[test]
    fn discovery_walks_up_for_project_kb() {
        let tmp = tempdir().expect("temp root");
        let project = tmp.path().join("project");
        let kb_root = project.join("kb");
        touch_kb_toml(&kb_root);

        let cwd = project.join("src");
        fs::create_dir_all(&cwd).expect("create cwd");

        let result = discover_root_in(None, &cwd, Path::new("/home/missing"));

        assert_eq!(
            result.expect("discover root").path,
            kb_root.canonicalize().expect("canonicalize")
        );
    }

    #[test]
    fn discovery_falls_back_to_home_kb() {
        let cwd_tmp = tempdir().expect("temp cwd");
        let home_tmp = tempdir().expect("temp home");
        let fallback = home_tmp.path().join("kb");
        touch_kb_toml(&fallback);

        let result = discover_root_in(None, cwd_tmp.path(), home_tmp.path());

        assert_eq!(
            result.expect("discover root").path,
            fallback.canonicalize().expect("canonicalize")
        );
    }

    #[test]
    fn discovery_accepts_pwd_as_root_when_kb_toml_present() {
        // Simulates `kb init` in an arbitrary dir which writes kb.toml directly.
        let tmp = tempdir().expect("temp root");
        let root = tmp.path().join("my-kb");
        touch_kb_toml(&root);

        let result = discover_root_in(None, &root, Path::new("/home/missing"));

        assert_eq!(
            result.expect("discover root").path,
            root.canonicalize().expect("canonicalize")
        );
    }

    #[test]
    fn discovery_accepts_ancestor_as_root_when_kb_toml_present() {
        // pwd sits inside a KB root (kb.toml directly in an ancestor).
        let tmp = tempdir().expect("temp root");
        let root = tmp.path().join("my-kb");
        touch_kb_toml(&root);
        let cwd = root.join("notes/sub");
        fs::create_dir_all(&cwd).expect("create cwd");

        let result = discover_root_in(None, &cwd, Path::new("/home/missing"));

        assert_eq!(
            result.expect("discover root").path,
            root.canonicalize().expect("canonicalize")
        );
    }

    #[test]
    fn root_override_accepts_dir_with_kb_toml_directly() {
        // `kb --root <dir-with-kb.toml>` accepts the dir itself as root.
        let tmp = tempdir().expect("temp root");
        let explicit = tmp.path().join("direct-root");
        touch_kb_toml(&explicit);

        let cwd = tmp.path().join("elsewhere");
        fs::create_dir_all(&cwd).expect("create cwd");

        let result = discover_root_in(Some(explicit.as_path()), &cwd, Path::new("/home/missing"));

        assert_eq!(
            result.expect("discover root").path,
            explicit.canonicalize().expect("canonicalize")
        );
    }

    #[test]
    fn root_override_accepts_project_with_kb_subdir() {
        // `kb --root <project>` where project has a `kb/` subdir still resolves.
        let tmp = tempdir().expect("temp root");
        let project = tmp.path().join("project");
        let kb_root = project.join("kb");
        touch_kb_toml(&kb_root);

        let cwd = tmp.path().join("elsewhere");
        fs::create_dir_all(&cwd).expect("create cwd");

        let result = discover_root_in(Some(project.as_path()), &cwd, Path::new("/home/missing"));

        assert_eq!(
            result.expect("discover root").path,
            kb_root.canonicalize().expect("canonicalize")
        );
    }

    #[test]
    fn missing_root_reports_init_guidance() {
        let cwd_tmp = tempdir().expect("temp cwd");
        let home_tmp = tempdir().expect("temp home");

        let result = discover_root_in(None, cwd_tmp.path(), home_tmp.path());

        let err = result.expect_err("root should be missing");
        assert!(
            err.to_string().contains("kb init"),
            "missing-root error should mention kb init"
        );
    }

    #[test]
    fn discovery_error_message_names_both_layouts() {
        // Both walk-up-from-cwd and --root failures should name both valid
        // layouts so the user isn't misled into thinking only one works.
        let cwd_tmp = tempdir().expect("temp cwd");
        let home_tmp = tempdir().expect("temp home");

        // Case 1: walk-up failure.
        let walkup_err = discover_root_in(None, cwd_tmp.path(), home_tmp.path())
            .expect_err("walk-up should fail");
        let walkup_msg = walkup_err.to_string();
        assert!(
            walkup_msg.contains("kb.toml"),
            "walk-up error should mention 'kb.toml': {walkup_msg}"
        );
        assert!(
            walkup_msg.contains("kb/kb.toml"),
            "walk-up error should mention 'kb/kb.toml': {walkup_msg}"
        );
        assert!(
            walkup_msg.contains(&cwd_tmp.path().display().to_string()),
            "walk-up error should cite cwd '{}': {walkup_msg}",
            cwd_tmp.path().display()
        );

        // Case 2: --root points at an existing dir with no kb config.
        let empty = tempdir().expect("temp empty root");
        let override_err =
            discover_root_in(Some(empty.path()), cwd_tmp.path(), home_tmp.path())
                .expect_err("override should fail");
        let override_msg = override_err.to_string();
        assert!(
            override_msg.contains("kb.toml"),
            "--root error should mention 'kb.toml': {override_msg}"
        );
        assert!(
            override_msg.contains("kb/kb.toml"),
            "--root error should mention 'kb/kb.toml': {override_msg}"
        );
        let empty_canon = empty
            .path()
            .canonicalize()
            .expect("canonicalize empty root");
        assert!(
            override_msg.contains(&empty_canon.display().to_string()),
            "--root error should cite provided path '{}': {override_msg}",
            empty_canon.display()
        );
    }
}
