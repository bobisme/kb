use std::{env, fs, path::{Path, PathBuf}};

use anyhow::{anyhow, Context, Result};

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
/// 1. `--root` override (if set)
/// 2. Walk up from the current directory for `*/kb/kb.toml`
/// 3. Fall back to `~/kb` when present
/// 4. Fail with an actionable error message.
pub fn discover_root(explicit_root: Option<&Path>) -> Result<KbRoot> {
    let cwd = env::current_dir().context("failed to resolve current working directory")?;
    let home = env::var("HOME").context("missing HOME environment variable")?;
    discover_root_in(explicit_root, &cwd, Path::new(&home))
}

fn discover_root_in(explicit_root: Option<&Path>, cwd: &Path, home: &Path) -> Result<KbRoot> {
    if let Some(root_override) = explicit_root {
        return validate_candidate(resolve_root_override(root_override, cwd)?);
    }

    for ancestor in cwd.ancestors() {
        let candidate = ancestor.join(KB_DIR);
        if is_kb_root(&candidate) {
            return Ok(KbRoot {
                path: candidate,
            });
        }
    }

    let global = home.join(KB_DIR);
    if is_kb_root(&global) {
        return Ok(KbRoot { path: global });
    }

    Err(anyhow!(
        "No KB root found. Initialize one with `kb init` or provide --root."
    ))
}

fn resolve_root_override(explicit_root: &Path, cwd: &Path) -> Result<PathBuf> {
    let resolved = if explicit_root.is_absolute() {
        explicit_root.to_path_buf()
    } else {
        cwd.join(explicit_root)
    };

    if !resolved.is_dir() {
        return Err(anyhow!(
            "Provided --root '{}' is not an existing directory. Run `kb init` to create it.",
            resolved.display()
        ));
    }

    fs::canonicalize(&resolved)
        .with_context(|| format!("failed to resolve --root path '{}'", resolved.display()))
}

fn validate_candidate(path: PathBuf) -> Result<KbRoot> {
    if is_kb_root(&path) {
        Ok(KbRoot { path })
    } else {
        Err(anyhow!(
            "Provided --root '{}' does not contain '{}'. Run `kb init` to initialize a KB root.",
            path.display(),
            format!("{KB_DIR}/{KB_CONFIG}")
        ))
    }
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

        let result =
            discover_root_in(Some(explicit.as_path()), &cwd, Path::new("/home/missing"));

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
}
