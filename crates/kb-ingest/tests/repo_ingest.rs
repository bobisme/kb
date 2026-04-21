//! End-to-end tests for `kb ingest <git-url>`.
//!
//! We seed a throwaway upstream repo with `git init` under `TempDir`, then
//! point `ingest_repo` at its file:// URL. Using a local bare/regular repo
//! means the test has no network dependency and finishes in well under a
//! second.

#![allow(clippy::unwrap_used)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use kb_core::normalized_dir;
use kb_ingest::{IngestOutcome, RepoIngestOptions, ingest_repo, is_git_url};
use tempfile::TempDir;

/// Runs `git` with the given args in `dir`, failing the test if the command
/// exits non-zero or can't be spawned. Stderr is attached on failure.
fn git(dir: &Path, args: &[&str]) {
    let output = Command::new("git")
        .current_dir(dir)
        // Disable global/system config so the test is hermetic on dev boxes
        // with exotic git settings (hooks, signing keys, init.defaultBranch).
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .env("GIT_AUTHOR_NAME", "kb test")
        .env("GIT_AUTHOR_EMAIL", "kb@test.invalid")
        .env("GIT_COMMITTER_NAME", "kb test")
        .env("GIT_COMMITTER_EMAIL", "kb@test.invalid")
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn git {args:?}: {e}"));
    assert!(
        output.status.success(),
        "git {args:?} failed: {}\nstdout: {}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout),
    );
}

/// Creates a git repo at `dir` with the given files (`rel_path` -> contents)
/// and a single commit. Returns the `file://` URL pointing at the repo.
fn init_seeded_repo(dir: &Path, files: &[(&str, &str)]) -> String {
    git(dir, &["init", "-q", "-b", "main"]);
    for (rel, contents) in files {
        let abs = dir.join(rel);
        if let Some(parent) = abs.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&abs, contents).unwrap();
    }
    git(dir, &["add", "."]);
    git(dir, &["commit", "-q", "-m", "initial"]);
    format!("file://{}", dir.display())
}

fn init_kb_root() -> TempDir {
    let root = TempDir::new().unwrap();
    fs::create_dir_all(root.path().join("raw/repos")).unwrap();
    fs::create_dir_all(root.path().join("raw/inbox")).unwrap();
    fs::create_dir_all(root.path().join("raw/web")).unwrap();
    root
}

/// Collect ingested `repo_paths` into a sorted `Vec<String>` so assertions
/// don't depend on `WalkBuilder`'s ordering.
fn sorted_paths(files: &[kb_ingest::RepoFileReport]) -> Vec<String> {
    let mut v: Vec<String> = files.iter().map(|f| f.repo_path.clone()).collect();
    v.sort();
    v
}

#[test]
fn is_git_url_routes_known_and_suffixed_urls() {
    assert!(is_git_url("https://github.com/microsoft/markitdown"));
    assert!(is_git_url("https://github.com/microsoft/markitdown.git"));
    assert!(is_git_url("git@github.com:microsoft/markitdown.git"));
    assert!(is_git_url("git+https://github.com/microsoft/markitdown"));
    // Plain web URLs should not route through repo ingest.
    assert!(!is_git_url("https://example.com/article"));
}

#[test]
fn ingest_repo_walks_default_docs_filter() {
    let upstream = TempDir::new().unwrap();
    let url = init_seeded_repo(
        upstream.path(),
        &[
            ("README.md", "# Hello\n\nBody\n"),
            ("CHANGELOG.md", "# Changes\n\n- v0.1\n"),
            ("docs/guide.md", "# Guide\n\nStuff\n"),
            ("src/main.rs", "fn main() {}\n"), // should be filtered out
            ("CONTRIBUTING", "Contributions welcome\n"),
        ],
    );

    let kb = init_kb_root();
    let report = ingest_repo(kb.path(), &url, &RepoIngestOptions::default()).unwrap();

    let paths = sorted_paths(&report.files);
    assert_eq!(
        paths,
        vec![
            "CHANGELOG.md".to_string(),
            "CONTRIBUTING".to_string(),
            "README.md".to_string(),
            "docs/guide.md".to_string(),
        ],
        "default walk should pick up docs but skip src/"
    );
    assert!(!report.commit_sha.is_empty(), "commit SHA must be recorded");
    assert_eq!(report.files[0].outcome, IngestOutcome::NewSource);

    // Each ingested file should have its own normalized document.
    for file in &report.files {
        let normalized = normalized_dir(kb.path()).join(&file.source_document_id)
            .join("source.md");
        assert!(
            normalized.is_file(),
            "normalized source.md missing for {}: {}",
            file.repo_path,
            normalized.display()
        );
        // Origin sidecar must record the remote URL + commit SHA.
        let origin_path = normalized_dir(kb.path()).join(&file.source_document_id)
            .join("origin.json");
        let origin: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&origin_path).unwrap()).unwrap();
        assert_eq!(
            origin.get("commit_sha").and_then(|v| v.as_str()),
            Some(report.commit_sha.as_str()),
            "origin.json must record the exact cloned commit SHA"
        );
        assert!(
            origin
                .get("normalized_url")
                .and_then(|v| v.as_str())
                .is_some_and(|s| s.starts_with("file://")),
            "origin.json must record the normalized remote URL"
        );
    }

    // Repo-level metadata sidecar.
    let meta_path = kb
        .path()
        .join("raw/repos")
        .join(&report.repo_source_id)
        .join("metadata.json");
    let meta: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&meta_path).unwrap()).unwrap();
    assert_eq!(
        meta.get("commit_sha").and_then(|v| v.as_str()),
        Some(report.commit_sha.as_str())
    );
    assert_eq!(
        meta.get("original_url").and_then(|v| v.as_str()),
        Some(url.as_str())
    );
}

#[test]
fn reingest_same_repo_skips_unchanged_files() {
    let upstream = TempDir::new().unwrap();
    let url = init_seeded_repo(
        upstream.path(),
        &[
            ("README.md", "# Hello\n\nBody\n"),
            ("docs/guide.md", "# Guide\n\nStuff\n"),
        ],
    );

    let kb = init_kb_root();
    let first = ingest_repo(kb.path(), &url, &RepoIngestOptions::default()).unwrap();
    assert!(first
        .files
        .iter()
        .all(|f| f.outcome == IngestOutcome::NewSource));

    // Second pass with no upstream changes must be a no-op per file.
    let second = ingest_repo(kb.path(), &url, &RepoIngestOptions::default()).unwrap();
    assert_eq!(second.files.len(), first.files.len());
    for file in &second.files {
        assert_eq!(
            file.outcome,
            IngestOutcome::Skipped,
            "{} should be skipped on no-op re-ingest",
            file.repo_path
        );
    }
    assert_eq!(
        first.commit_sha, second.commit_sha,
        "pinning the same commit should round-trip"
    );
}

#[test]
fn reingest_after_upstream_change_creates_new_revision() {
    let upstream = TempDir::new().unwrap();
    let url = init_seeded_repo(
        upstream.path(),
        &[
            ("README.md", "# Hello\n\nVersion one\n"),
            ("docs/guide.md", "# Guide\n\nUnchanged\n"),
        ],
    );

    let kb = init_kb_root();
    let first = ingest_repo(kb.path(), &url, &RepoIngestOptions::default()).unwrap();

    // Modify README upstream and commit.
    fs::write(
        upstream.path().join("README.md"),
        "# Hello\n\nVersion TWO\n",
    )
    .unwrap();
    git(upstream.path(), &["add", "README.md"]);
    git(upstream.path(), &["commit", "-q", "-m", "update readme"]);

    // Need to remove the existing cached clone so the second ingest sees the
    // new commit; our v1 doesn't do `git fetch` on re-ingest yet.
    let repo_dir: PathBuf = kb
        .path()
        .join("raw/repos")
        .join(&first.repo_source_id);
    fs::remove_dir_all(&repo_dir).unwrap();

    let second = ingest_repo(kb.path(), &url, &RepoIngestOptions::default()).unwrap();

    let readme = second
        .files
        .iter()
        .find(|f| f.repo_path == "README.md")
        .expect("README.md should appear in second ingest");
    assert_eq!(
        readme.outcome,
        IngestOutcome::NewRevision,
        "README should get a new revision after upstream change"
    );

    let guide = second
        .files
        .iter()
        .find(|f| f.repo_path == "docs/guide.md")
        .expect("docs/guide.md should appear in second ingest");
    assert_eq!(
        guide.outcome,
        IngestOutcome::Skipped,
        "unchanged docs/guide.md should still be skipped"
    );

    assert_ne!(
        first.commit_sha, second.commit_sha,
        "new upstream commit must advance recorded SHA"
    );
}

#[test]
fn include_glob_pulls_rust_sources() {
    let upstream = TempDir::new().unwrap();
    let url = init_seeded_repo(
        upstream.path(),
        &[
            ("README.md", "# Hello\n\nBody\n"),
            ("src/main.rs", "fn main() {\n    println!(\"hi\");\n}\n"),
            ("src/lib.rs", "pub fn add(a: i32, b: i32) -> i32 { a + b }\n"),
        ],
    );

    let kb = init_kb_root();
    let report = ingest_repo(
        kb.path(),
        &url,
        &RepoIngestOptions {
            includes: vec!["**/*.rs".to_string()],
            ..RepoIngestOptions::default()
        },
    )
    .unwrap();

    let paths = sorted_paths(&report.files);
    assert_eq!(
        paths,
        vec!["src/lib.rs".to_string(), "src/main.rs".to_string()],
        "explicit include should bypass the default doc filter"
    );
}

#[test]
fn exclude_glob_drops_matching_files() {
    let upstream = TempDir::new().unwrap();
    let url = init_seeded_repo(
        upstream.path(),
        &[
            ("README.md", "# Hello\n"),
            ("docs/guide.md", "# Guide\n"),
            ("docs/internal.md", "# Internal\n"),
        ],
    );

    let kb = init_kb_root();
    let report = ingest_repo(
        kb.path(),
        &url,
        &RepoIngestOptions {
            excludes: vec!["docs/internal.md".to_string()],
            ..RepoIngestOptions::default()
        },
    )
    .unwrap();

    let paths = sorted_paths(&report.files);
    assert_eq!(
        paths,
        vec!["README.md".to_string(), "docs/guide.md".to_string()],
        "exclude glob should drop internal.md"
    );
}

#[test]
fn gitignored_files_are_not_ingested() {
    let upstream = TempDir::new().unwrap();
    // `.gitignore` is committed; `secret.md` is also committed but typically
    // a user would only ship one or the other. The point is that WalkBuilder
    // inside the checkout sees the .gitignore and honors it. Since we want
    // the file to actually exist in the clone, leave `secret.md` committed
    // and verify the walker still respects the ignore rule.
    let url = init_seeded_repo(
        upstream.path(),
        &[
            (".gitignore", "secret.md\n"),
            ("README.md", "# ok\n"),
            // secret.md gets written but must be excluded by ignore walker.
            ("secret.md", "shhh\n"),
        ],
    );

    let kb = init_kb_root();
    let report = ingest_repo(kb.path(), &url, &RepoIngestOptions::default()).unwrap();

    let paths = sorted_paths(&report.files);
    assert!(
        paths.iter().any(|p| p == "README.md"),
        "README.md should ingest, got {paths:?}"
    );
    assert!(
        !paths.iter().any(|p| p == "secret.md"),
        ".gitignored file should not ingest, got {paths:?}"
    );
}

#[test]
fn pinned_commit_checks_out_that_sha() {
    let upstream = TempDir::new().unwrap();
    let url = init_seeded_repo(
        upstream.path(),
        &[("README.md", "# first\n")],
    );

    // Capture the first commit's SHA.
    let first_sha = String::from_utf8(
        Command::new("git")
            .current_dir(upstream.path())
            .args(["rev-parse", "HEAD"])
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap()
    .trim()
    .to_string();

    // Push a second commit upstream.
    fs::write(upstream.path().join("README.md"), "# second\n").unwrap();
    git(upstream.path(), &["add", "README.md"]);
    git(upstream.path(), &["commit", "-q", "-m", "second"]);

    let kb = init_kb_root();
    let report = ingest_repo(
        kb.path(),
        &url,
        &RepoIngestOptions {
            commit: Some(first_sha.clone()),
            ..RepoIngestOptions::default()
        },
    )
    .unwrap();

    assert_eq!(
        report.commit_sha, first_sha,
        "commit pin should check out the requested SHA"
    );

    // README content at that revision should be "first", not "second".
    let readme = report
        .files
        .iter()
        .find(|f| f.repo_path == "README.md")
        .unwrap();
    let normalized_src = normalized_dir(kb.path()).join(&readme.source_document_id)
        .join("source.md");
    let body = fs::read_to_string(&normalized_src).unwrap();
    assert!(
        body.contains("first"),
        "pinned commit should yield the 'first' README, got: {body}"
    );
    assert!(!body.contains("second"));
}

#[test]
fn binary_files_in_default_walk_are_skipped() {
    let upstream = TempDir::new().unwrap();
    // Default filter won't pick up *.bin, but a user-provided include can.
    // We hand an include that matches binaries to prove that the per-file
    // text-check still guards against them.
    let url = init_seeded_repo(upstream.path(), &[("README.md", "# text\n")]);

    // Write an ELF-shaped binary and commit it under an allowed extension so
    // the walker includes it but the per-file text check rejects it.
    fs::write(
        upstream.path().join("binary.md"),
        [0x7fu8, b'E', b'L', b'F', 0, 0, 0, 1, 2, 3],
    )
    .unwrap();
    git(upstream.path(), &["add", "binary.md"]);
    git(upstream.path(), &["commit", "-q", "-m", "add binary"]);

    let kb = init_kb_root();
    let report = ingest_repo(kb.path(), &url, &RepoIngestOptions::default()).unwrap();

    let paths = sorted_paths(&report.files);
    assert_eq!(
        paths,
        vec!["README.md".to_string()],
        "binary .md should be skipped by the text probe"
    );
}
