//! Repository ingestion: shallow-clone a git remote and ingest its docs.
//!
//! The high-level flow is:
//! 1. Detect that the ingest target is a git URL (see [`is_git_url`]).
//! 2. Shallow-clone (`git clone --depth 1`) into `raw/repos/<src-id>/`, pinning
//!    `--branch` / `--commit` when requested.
//! 3. Walk the checkout with a default filter that keeps only docs-ish text
//!    files (README, CHANGELOG, CONTRIBUTING, docs/**/*.md, top-level *.md).
//!    Callers can override with `--include` / `--exclude` glob flags.
//! 4. Each matched file is ingested via the same code path as local files,
//!    but tagged with `SourceKind::Repo` and stable locations of the form
//!    `git+<normalized-url>#<repo-relative-path>`. The remote URL + commit
//!    SHA land in `raw/repos/<src>/metadata.json` for provenance.
//! 5. Re-ingest dedupes by content hash per file (the existing file-ingest
//!    pipeline handles the `Skipped` outcome).

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use ignore::WalkBuilder;
use ignore::overrides::OverrideBuilder;
use kb_core::fs::atomic_write;
use kb_core::{
    EntityMetadata, NormalizedDocument, SourceDocument, SourceKind, SourceRevision, Status,
    mint_source_document_id, mint_source_revision_id, normalized_dir, normalized_rel,
    source_revision_content_hash, write_normalized_document,
};
use serde::{Deserialize, Serialize};

use crate::IngestOutcome;
use crate::headings::extract_heading_ids;

const SOURCE_DOCUMENT_RECORD: &str = "source_document.json";
const SOURCE_REVISION_RECORD: &str = "source_revision.json";
const REPO_METADATA_RECORD: &str = "metadata.json";

/// Known git hosts that don't always carry a `.git` suffix in user-typed URLs.
/// Keep this narrow: matching a host here auto-routes the URL to repo ingest,
/// so false positives would pull web pages through a clone attempt.
const KNOWN_GIT_HOSTS: &[&str] = &[
    "github.com",
    "www.github.com",
    "gitlab.com",
    "bitbucket.org",
    "codeberg.org",
    "git.sr.ht",
];

/// Files worth ingesting out of the box. If the caller passes `--include`,
/// this default list is ignored entirely and only their globs apply.
///
/// Patterns live in the `globset` flavor that `ignore::overrides` uses
/// (extended gitignore-style globs). They're whitelist patterns — anything
/// not matched is skipped.
const DEFAULT_INCLUDES: &[&str] = &[
    // Top-level docs
    "README*",
    "CHANGELOG*",
    "CONTRIBUTING*",
    "ROADMAP*",
    "HACKING*",
    "AUTHORS*",
    "NOTICE*",
    "*.md",
    "*.mdx",
    "*.markdown",
    "*.rst",
    "*.txt",
    // Subdirectory READMEs at any depth
    "**/README*",
    "**/CHANGELOG*",
    "**/CONTRIBUTING*",
    // docs/ and doc/ trees
    "docs/**/*.md",
    "docs/**/*.mdx",
    "docs/**/*.rst",
    "doc/**/*.md",
    "doc/**/*.mdx",
    "doc/**/*.rst",
];

/// Options controlling a single `kb ingest <git-url>` invocation.
#[derive(Debug, Clone, Default)]
pub struct RepoIngestOptions {
    /// Extra include globs. When non-empty, overrides [`DEFAULT_INCLUDES`].
    pub includes: Vec<String>,
    /// Exclude globs (applied after includes).
    pub excludes: Vec<String>,
    /// Checkout this branch rather than the remote's default.
    pub branch: Option<String>,
    /// Pin to this commit SHA after clone.
    pub commit: Option<String>,
    /// If true, compute outcomes and report paths without writing files.
    pub dry_run: bool,
    /// Passthrough to file ingest — allow frontmatter-only / empty files.
    pub allow_empty: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepoFileReport {
    /// Path relative to the repo root (forward-slash).
    pub repo_path: String,
    pub outcome: IngestOutcome,
    pub source_document_id: String,
    pub source_revision_id: String,
    pub content_path: PathBuf,
    pub metadata_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepoIngestReport {
    pub source_url: String,
    pub normalized_url: String,
    pub repo_source_id: String,
    pub commit_sha: String,
    pub branch: Option<String>,
    pub files: Vec<RepoFileReport>,
}

/// On-disk sidecar that records a repo's remote + pinned revision. Lives at
/// `raw/repos/<src>/metadata.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RepoMetadata {
    original_url: String,
    normalized_url: String,
    commit_sha: String,
    branch: Option<String>,
    cloned_at_millis: u64,
    tool_version: String,
    /// Per-file commit SHA from the last successful ingest. Keyed by repo-
    /// relative path. Used to short-circuit re-ingest when nothing changed.
    #[serde(default)]
    last_ingest_commits: std::collections::BTreeMap<String, String>,
}

/// Per-file sidecar written next to the archived copy under
/// `raw/repos/<src>/.ingest/<src-id>/<rev-id>/<file>.metadata.json`.
#[derive(Serialize)]
struct RepoFileSidecar<'a> {
    original_path: &'a str,
    repo_url: &'a str,
    commit_sha: &'a str,
    size_bytes: u64,
    content_hash: &'a str,
    imported_at_millis: u64,
}

/// Companion to the normalized-document metadata. Kept separate so the
/// canonical `metadata.json` schema stays untouched across source kinds.
#[derive(Serialize)]
struct RepoFileOrigin<'a> {
    original_url: &'a str,
    normalized_url: &'a str,
    repo_source_id: &'a str,
    repo_path: &'a str,
    commit_sha: &'a str,
    fetched_at_millis: u64,
}

/// Returns true if `source` looks like a git remote rather than an HTTP(S) URL
/// or local path.
///
/// Recognized shapes:
/// - explicit `git+...` (e.g. `git+https://...`) — pip / npm convention
/// - `git@<host>:<owner>/<repo>[.git]` (SSH)
/// - `git://<host>/...` / `ssh://git@...`
/// - any URL whose path ends in `.git` or `.git/`
/// - known hosts: `github.com`, `gitlab.com`, `bitbucket.org`, `codeberg.org`,
///   `sr.ht` — even without a `.git` suffix
///
/// The returned bool is **disjoint** from [`crate::is_url`] in the caller's
/// dispatch: the CLI must test `is_git_url` first so that
/// `https://github.com/foo/bar` goes through the repo path, not the web path.
#[must_use]
pub fn is_git_url(source: &str) -> bool {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return false;
    }

    // `git+https://...`, `git+ssh://...`, `git://...`.
    let lower_prefix_len = trimmed.len().min(10);
    let lower_prefix = trimmed
        .get(..lower_prefix_len)
        .unwrap_or("")
        .to_ascii_lowercase();
    if lower_prefix.starts_with("git+")
        || lower_prefix.starts_with("git://")
        || lower_prefix.starts_with("git@")
        || lower_prefix.starts_with("ssh://")
    {
        return true;
    }

    // `.git` suffix on the final path segment of any URL-like / path-like
    // input. Case-insensitive so `FOO.GIT` (rare but valid) still routes
    // through. `Path::extension()` would strip the leading dot for us but
    // breaks on `git@host:foo.git` (no path separator before `foo.git`) and
    // `//repo.git/` (trailing slash), so we split on `/` manually.
    let stripped = trimmed.trim_end_matches('/');
    if let Some(last_seg) = stripped.rsplit('/').next()
        && let Some(ext) = Path::new(last_seg).extension()
        && ext.eq_ignore_ascii_case("git")
    {
        return true;
    }

    // Known git hosting URL — require http(s) scheme to avoid false positives
    // on non-URL text that happens to mention the host.
    if lower_prefix.starts_with("https://") || lower_prefix.starts_with("http://") {
        let body = trimmed.split_once("//").map_or(trimmed, |(_, rest)| rest);
        let host = body.split(['/', '?', '#']).next().unwrap_or("");
        let host_lower = host.to_ascii_lowercase();
        // Strip trailing `:port` if present.
        let host_only = host_lower.split(':').next().unwrap_or(&host_lower);
        if KNOWN_GIT_HOSTS.contains(&host_only) {
            // For known hosts, require at least `/owner/repo` in the path so
            // a bare `https://github.com/` doesn't get misrouted as a repo.
            let path = body.get(host.len()..).unwrap_or("");
            let path = path.split(['?', '#']).next().unwrap_or("");
            let segment_count = path.split('/').filter(|s| !s.is_empty()).count();
            return segment_count >= 2;
        }
    }

    false
}

/// Produces a stable-location string for a git remote, stripping the optional
/// `git+` scheme prefix used in Python/npm-style requirements. The result
/// becomes the seed for the repo's `src-` id.
fn normalize_git_url(raw: &str) -> String {
    let raw = raw.trim();
    let stripped = raw.strip_prefix("git+").unwrap_or(raw);
    // Drop a trailing `.git` for the id seed so `/foo/bar` and `/foo/bar.git`
    // collide onto the same source document. This matches how users and most
    // UIs treat the two as interchangeable.
    let no_git = stripped
        .strip_suffix(".git")
        .or_else(|| stripped.strip_suffix(".git/"))
        .unwrap_or(stripped);
    // Collapse a trailing slash on non-root paths; mirrors the URL
    // normalization used for web sources.
    no_git.trim_end_matches('/').to_string()
}

/// Top-level entry point for `kb ingest <git-url>`.
///
/// Clones `raw_url` into `raw/repos/<src-id>/`, walks the checkout with the
/// caller's filters, and ingests each matched file as its own
/// `SourceDocument` under a shared `src-<repo-hash>-<subpath>` scheme.
///
/// # Errors
/// Returns an error if the clone fails, the commit pin is invalid, or any
/// per-file ingest fails catastrophically. Per-file binary/empty rejections
/// are warned and skipped, as they are for local directory ingest.
pub fn ingest_repo(
    root: &Path,
    raw_url: &str,
    options: &RepoIngestOptions,
) -> Result<RepoIngestReport> {
    let normalized_url = normalize_git_url(raw_url);
    let repo_source_id = mint_repo_source_id(root, &normalized_url);
    let repo_dir = root.join("raw").join("repos").join(&repo_source_id);

    // Fresh clone if the directory doesn't exist. On re-ingest we leave the
    // existing working tree in place — the metadata sidecar tells us the
    // pinned commit, and the per-file dedupe handled by ingest_file still
    // kicks in. A future enhancement can `git fetch + reset` here, but the
    // bone scopes that out for v1.
    let need_clone = !repo_dir.exists();
    if need_clone {
        if options.dry_run {
            // Dry-run shouldn't touch disk. Abort before we clone gigabytes.
            bail!(
                "dry-run ingest for {raw_url} requires an existing checkout at {}",
                repo_dir.display()
            );
        }
        let parent = repo_dir
            .parent()
            .context("raw/repos/<src> must have a parent")?;
        fs::create_dir_all(parent).with_context(|| {
            format!("failed to create raw/repos under {}", root.display())
        })?;
        clone_repo(&strip_git_plus(raw_url), &repo_dir, options.branch.as_deref())?;
        if let Some(commit) = &options.commit {
            checkout_commit(&repo_dir, commit)?;
        }
    } else if let Some(commit) = &options.commit {
        // Best-effort: if the caller pinned a commit but we already have a
        // checkout, make sure it's at the right SHA.
        checkout_commit(&repo_dir, commit)?;
    }

    let commit_sha = rev_parse_head(&repo_dir)?;

    // Walk the working tree with our include/exclude globs. We pass ownership
    // to a vec so we can sort for deterministic output order.
    let mut files = collect_repo_files(&repo_dir, &options.includes, &options.excludes)?;
    files.sort();

    // Load (or create) the repo sidecar so we can short-circuit files whose
    // per-path commit hasn't advanced since the last ingest.
    let metadata_path = repo_dir.join(REPO_METADATA_RECORD);
    let mut repo_metadata = load_repo_metadata(&metadata_path).unwrap_or_else(|| RepoMetadata {
        original_url: raw_url.to_string(),
        normalized_url: normalized_url.clone(),
        commit_sha: commit_sha.clone(),
        branch: options.branch.clone(),
        cloned_at_millis: now_millis().unwrap_or(0),
        tool_version: env!("CARGO_PKG_VERSION").to_string(),
        last_ingest_commits: std::collections::BTreeMap::new(),
    });

    let mut file_reports = Vec::with_capacity(files.len());
    for rel in files {
        let abs = repo_dir.join(&rel);
        let rel_str = rel.to_string_lossy().replace('\\', "/");

        // Short-circuit: if the per-file commit recorded on the sidecar
        // matches the file's current blob commit AND the on-disk ingest
        // record is still present, we can emit a Skipped report without
        // re-hashing the content. This is the v1 "skip unchanged files on
        // re-ingest" optimization — full git-native sync is out of scope.
        let blob_commit = last_commit_for_path(&repo_dir, &rel_str).ok();
        if let Some(prev) = repo_metadata.last_ingest_commits.get(&rel_str)
            && let Some(current) = &blob_commit
            && prev == current
        {
            // Fabricate a minimal report pointing at the existing on-disk
            // location so the CLI summary still shows the file.
            let src_id = repo_file_src_id(&repo_source_id, &rel_str);
            let rev_id = existing_revision_id(root, &src_id).unwrap_or_default();
            file_reports.push(RepoFileReport {
                repo_path: rel_str.clone(),
                outcome: IngestOutcome::Skipped,
                source_document_id: src_id.clone(),
                source_revision_id: rev_id,
                content_path: PathBuf::from("raw/repos")
                    .join(&repo_source_id)
                    .join(&rel_str),
                metadata_path: normalized_rel(&src_id).join("metadata.json"),
            });
            continue;
        }

        if let Some(report) = ingest_repo_file(
            root,
            &repo_source_id,
            &rel_str,
            &abs,
            &normalized_url,
            &commit_sha,
            options,
        )? {
            if let Some(commit) = &blob_commit {
                repo_metadata
                    .last_ingest_commits
                    .insert(rel_str.clone(), commit.clone());
            }
            file_reports.push(report);
        }
        // Files rejected by the binary/empty/unreadable filters are silently
        // dropped — and not recorded on the sidecar — so a future invocation
        // re-checks them if the ingest gates are relaxed.
    }

    // Persist updated sidecar. We always write it (even on dry-run=false with
    // zero changes) so a stale file from a partial previous run gets
    // refreshed.
    if !options.dry_run {
        commit_sha.clone_into(&mut repo_metadata.commit_sha);
        repo_metadata.branch = options.branch.clone().or(repo_metadata.branch);
        write_repo_metadata(&metadata_path, &repo_metadata)?;
    }

    Ok(RepoIngestReport {
        source_url: raw_url.to_string(),
        normalized_url,
        repo_source_id,
        commit_sha,
        branch: options.branch.clone(),
        files: file_reports,
    })
}

/// Collects the relative paths and in-memory records for a single repo file.
/// Split out from the side-effectful `ingest_repo_file` so the latter stays
/// under clippy's 100-line-per-function budget.
struct RepoFilePlan {
    rel_document_record: PathBuf,
    rel_revision_record: PathBuf,
    rel_copied: PathBuf,
    rel_metadata_sidecar: PathBuf,
    document: SourceDocument,
    revision: SourceRevision,
    content_hash: String,
    outcome: IngestOutcome,
}

impl RepoFilePlan {
    #[allow(clippy::too_many_arguments)]
    fn build(
        root: &Path,
        rel_path: &str,
        src_id: &str,
        rev_id: &str,
        stable_location: &str,
        content: &[u8],
        now: u64,
    ) -> Result<Self> {
        // Per-file ingest records share the `raw/inbox/<src-id>/` layout with
        // local-file ingest. Storing them there — rather than nested under
        // `raw/repos/<repo-src>/.ingest/` — means wiping a cached clone
        // doesn't wipe the file's ingest history, so re-ingesting after an
        // upstream change correctly reports NewRevision instead of NewSource.
        let content_hash = source_revision_content_hash(content);
        let rel_revision_dir = PathBuf::from("raw")
            .join("inbox")
            .join(src_id)
            .join(rev_id);
        let file_name =
            Path::new(rel_path)
                .file_name()
                .map_or_else(|| "source".to_string(), |n| n.to_string_lossy().into_owned());
        let rel_document_record = PathBuf::from("raw")
            .join("inbox")
            .join(src_id)
            .join(SOURCE_DOCUMENT_RECORD);
        let rel_revision_record = rel_revision_dir.join(SOURCE_REVISION_RECORD);
        let rel_copied = rel_revision_dir.join(&file_name);
        let rel_metadata_sidecar =
            rel_revision_dir.join(format!("{file_name}.metadata.json"));

        let document_record_path = root.join(&rel_document_record);
        let revision_record_path = root.join(&rel_revision_record);
        let document_exists = document_record_path.exists();
        let revision_exists = revision_record_path.exists();

        let document = if document_exists {
            read_json::<SourceDocument>(&document_record_path)?
        } else {
            SourceDocument {
                metadata: EntityMetadata {
                    id: src_id.to_string(),
                    created_at_millis: now,
                    updated_at_millis: now,
                    source_hashes: vec![content_hash.clone()],
                    model_version: None,
                    tool_version: Some(env!("CARGO_PKG_VERSION").to_string()),
                    prompt_template_hash: None,
                    dependencies: Vec::new(),
                    output_paths: vec![rel_document_record.clone()],
                    status: Status::Fresh,
                },
                source_kind: SourceKind::Repo,
                stable_location: stable_location.to_string(),
                discovered_at_millis: now,
            }
        };

        let revision = if revision_exists {
            read_json::<SourceRevision>(&revision_record_path)?
        } else {
            SourceRevision {
                metadata: EntityMetadata {
                    id: rev_id.to_string(),
                    created_at_millis: now,
                    updated_at_millis: now,
                    source_hashes: vec![content_hash.clone()],
                    model_version: None,
                    tool_version: Some(env!("CARGO_PKG_VERSION").to_string()),
                    prompt_template_hash: None,
                    dependencies: vec![document.metadata.id.clone()],
                    output_paths: vec![
                        rel_copied.clone(),
                        rel_metadata_sidecar.clone(),
                        rel_revision_record.clone(),
                    ],
                    status: Status::Fresh,
                },
                source_document_id: document.metadata.id.clone(),
                fetched_revision_hash: content_hash.clone(),
                fetched_path: rel_copied.clone(),
                fetched_size_bytes: content.len() as u64,
                fetched_at_millis: now,
            }
        };

        let outcome = if !document_exists {
            IngestOutcome::NewSource
        } else if !revision_exists {
            IngestOutcome::NewRevision
        } else {
            IngestOutcome::Skipped
        };

        Ok(Self {
            rel_document_record,
            rel_revision_record,
            rel_copied,
            rel_metadata_sidecar,
            document,
            revision,
            content_hash,
            outcome,
        })
    }
}

/// Ingest a single file inside the clone. Returns `Ok(None)` if the file was
/// silently skipped (binary / empty / unreadable), `Ok(Some(report))`
/// otherwise.
#[allow(clippy::too_many_arguments)]
fn ingest_repo_file(
    root: &Path,
    repo_source_id: &str,
    rel_path: &str,
    abs_path: &Path,
    normalized_url: &str,
    commit_sha: &str,
    options: &RepoIngestOptions,
) -> Result<Option<RepoFileReport>> {
    let content = match fs::read(abs_path) {
        Ok(bytes) => bytes,
        Err(err) => {
            eprintln!("warning: skipping {}: {err}", abs_path.display());
            return Ok(None);
        }
    };

    if !crate::looks_like_text(&content) {
        eprintln!(
            "warning: skipping {} (binary content detected)",
            abs_path.display()
        );
        return Ok(None);
    }

    if !options.allow_empty && crate::is_semantically_empty(&content) {
        eprintln!(
            "warning: skipping empty source {} (use --allow-empty to override)",
            abs_path.display()
        );
        return Ok(None);
    }

    let stable_location = format!("git+{normalized_url}#{rel_path}");
    let src_id = repo_file_src_id(repo_source_id, rel_path);
    let rev_id = mint_source_revision_id(&content);
    let now = now_millis()?;

    let plan = RepoFilePlan::build(
        root,
        rel_path,
        &src_id,
        &rev_id,
        &stable_location,
        &content,
        now,
    )?;

    if !options.dry_run {
        persist_repo_file(
            root,
            repo_source_id,
            rel_path,
            normalized_url,
            commit_sha,
            &content,
            now,
            &plan,
        )?;
    }

    Ok(Some(RepoFileReport {
        repo_path: rel_path.to_string(),
        outcome: plan.outcome,
        source_document_id: src_id,
        source_revision_id: rev_id,
        content_path: plan.rel_copied,
        metadata_path: normalized_rel(&plan.document.metadata.id).join("metadata.json"),
    }))
}

/// Writes every on-disk artifact implied by `plan`: the source-document and
/// source-revision records, the archived file copy, the per-file sidecar,
/// the normalized-document view, and the origin sidecar for inspect tooling.
#[allow(clippy::too_many_arguments)]
fn persist_repo_file(
    root: &Path,
    repo_source_id: &str,
    rel_path: &str,
    normalized_url: &str,
    commit_sha: &str,
    content: &[u8],
    now: u64,
    plan: &RepoFilePlan,
) -> Result<()> {
    let document_record_path = root.join(&plan.rel_document_record);
    let revision_record_path = root.join(&plan.rel_revision_record);

    if !document_record_path.exists() {
        write_json(&document_record_path, &plan.document)?;
    }

    if !revision_record_path.exists() {
        let copied = root.join(&plan.rel_copied);
        fs::create_dir_all(
            copied
                .parent()
                .context("ingest revision dir should have a parent")?,
        )
        .with_context(|| format!("failed to create dir for {}", copied.display()))?;
        fs::write(&copied, content)
            .with_context(|| format!("failed to copy to {}", copied.display()))?;

        let sidecar = RepoFileSidecar {
            original_path: rel_path,
            repo_url: normalized_url,
            commit_sha,
            size_bytes: content.len() as u64,
            content_hash: &plan.content_hash,
            imported_at_millis: now,
        };
        write_json(&root.join(&plan.rel_metadata_sidecar), &sidecar)?;
        write_json(&revision_record_path, &plan.revision)?;
    }

    // Produce the canonical normalized view so compile/query can consume the
    // repo doc the same way they consume local files.
    let raw_text = std::str::from_utf8(content).map_or_else(
        |_| String::from_utf8_lossy(content).into_owned(),
        ToOwned::to_owned,
    );
    let heading_ids = extract_heading_ids(&raw_text);
    let normalized_metadata = EntityMetadata {
        id: plan.document.metadata.id.clone(),
        created_at_millis: plan.document.metadata.created_at_millis,
        updated_at_millis: now,
        source_hashes: vec![plan.content_hash.clone()],
        model_version: None,
        tool_version: Some(env!("CARGO_PKG_VERSION").to_string()),
        prompt_template_hash: None,
        dependencies: vec![plan.revision.metadata.id.clone()],
        output_paths: vec![
            normalized_rel(&plan.document.metadata.id).join("source.md"),
            normalized_rel(&plan.document.metadata.id).join("metadata.json"),
        ],
        status: Status::Fresh,
    };
    let normalized = NormalizedDocument {
        metadata: normalized_metadata,
        source_revision_id: plan.revision.metadata.id.clone(),
        canonical_text: raw_text,
        normalized_assets: Vec::new(),
        heading_ids,
    };
    write_normalized_document(root, &normalized).with_context(|| {
        format!(
            "failed to write normalized document {}",
            plan.document.metadata.id
        )
    })?;

    let origin = RepoFileOrigin {
        original_url: normalized_url,
        normalized_url,
        repo_source_id,
        repo_path: rel_path,
        commit_sha,
        fetched_at_millis: now,
    };
    atomic_write(
        normalized_dir(root)
            .join(&plan.document.metadata.id)
            .join("origin.json"),
        serde_json::to_vec_pretty(&origin)?.as_slice(),
    )
    .context("failed to write repo-file origin sidecar")?;

    Ok(())
}

/// Seeds the per-file `src-` id from the repo's source id plus the file's
/// in-repo path, so the collision probe only has to inspect the file's own
/// document record.
fn repo_file_src_id(repo_source_id: &str, rel_path: &str) -> String {
    // Use kb_core's id minter with a file-kind tag so ids live in the same
    // `src-` namespace as local files. The stable_location we seed is the
    // composite `<repo>/<path>` so two repos with the same path mint distinct
    // ids.
    let seed = format!("{repo_source_id}/{rel_path}");
    mint_source_document_id(SourceKind::Repo, &seed, 0, |_| false)
}

/// Mints the `src-` id for the repo as a whole. Re-ingesting the same remote
/// must round-trip to the same id; we key off the normalized URL.
fn mint_repo_source_id(root: &Path, normalized_url: &str) -> String {
    let existing = existing_source_count(root);
    mint_source_document_id(
        SourceKind::Repo,
        normalized_url,
        existing,
        |candidate| {
            root.join("raw")
                .join("repos")
                .join(candidate)
                .join(REPO_METADATA_RECORD)
                .exists()
                && !repo_metadata_matches(root, candidate, normalized_url)
        },
    )
}

/// Returns true iff the `<candidate>/metadata.json` on disk records the same
/// normalized URL — i.e. re-ingesting this remote should reuse the candidate
/// id rather than mint a fresh one.
fn repo_metadata_matches(root: &Path, candidate: &str, normalized_url: &str) -> bool {
    let path = root
        .join("raw")
        .join("repos")
        .join(candidate)
        .join(REPO_METADATA_RECORD);
    load_repo_metadata(&path).is_some_and(|md| md.normalized_url == normalized_url)
}

fn existing_source_count(root: &Path) -> usize {
    let mut total = 0;
    for bucket in ["raw/inbox", "raw/web", "raw/repos"] {
        if let Ok(entries) = fs::read_dir(root.join(bucket)) {
            total += entries
                .filter_map(Result::ok)
                .filter(|entry| entry.file_type().is_ok_and(|ft| ft.is_dir()))
                .count();
        }
    }
    total
}

/// Reads `normalized/<src>/metadata.json` (the canonical normalized-document
/// record written by `write_normalized_document`) and returns its
/// `source_revision_id`, if present.
fn existing_revision_id(root: &Path, src_id: &str) -> Option<String> {
    #[derive(Deserialize)]
    struct Probe {
        source_revision_id: String,
    }
    let path = normalized_dir(root).join(src_id).join("metadata.json");
    let contents = fs::read_to_string(&path).ok()?;
    serde_json::from_str::<Probe>(&contents)
        .ok()
        .map(|probe| probe.source_revision_id)
}

fn load_repo_metadata(path: &Path) -> Option<RepoMetadata> {
    let contents = fs::read_to_string(path).ok()?;
    serde_json::from_str(&contents).ok()
}

fn write_repo_metadata(path: &Path, md: &RepoMetadata) -> Result<()> {
    atomic_write(path, serde_json::to_vec_pretty(md)?.as_slice())
        .context("failed to write repo metadata sidecar")
}

/// `git clone --depth 1 [--branch <b>] <url> <dest>`. We shell out rather
/// than adding a `git2` dep — the clone cost dwarfs the subprocess overhead,
/// and `git` is already a hard dependency of any kb user.
fn clone_repo(url: &str, dest: &Path, branch: Option<&str>) -> Result<()> {
    let mut cmd = Command::new("git");
    cmd.arg("clone").arg("--depth").arg("1");
    // Don't follow submodules — v1 policy per the bone.
    cmd.arg("--no-tags");
    if let Some(branch) = branch {
        cmd.arg("--branch").arg(branch).arg("--single-branch");
    }
    cmd.arg(url).arg(dest);

    let output = cmd
        .output()
        .with_context(|| format!("failed to spawn `git clone {url}`"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("git clone {url} failed: {stderr}");
    }
    Ok(())
}

fn checkout_commit(repo: &Path, commit: &str) -> Result<()> {
    // A shallow clone doesn't fetch arbitrary history, so `git checkout <sha>`
    // will fail unless the SHA is the tip of `HEAD`. Unshallow first.
    let unshallow = Command::new("git")
        .current_dir(repo)
        .args(["fetch", "--unshallow", "--no-tags"])
        .output();
    if let Ok(out) = unshallow
        && !out.status.success()
    {
        // `--unshallow` errors on an already-complete clone with
        // "fatal: --unshallow on a complete repository does not make sense",
        // which is fine — keep going.
        let stderr = String::from_utf8_lossy(&out.stderr);
        if !stderr.contains("--unshallow on a complete repository") {
            // Only propagate unexpected fetch errors.
            if !out.status.success() {
                eprintln!(
                    "warning: `git fetch --unshallow` in {} failed: {stderr}",
                    repo.display()
                );
            }
        }
    }

    let output = Command::new("git")
        .current_dir(repo)
        .args(["checkout", "--detach", commit])
        .output()
        .with_context(|| format!("failed to spawn `git checkout {commit}`"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("git checkout {commit} failed: {stderr}");
    }
    Ok(())
}

fn rev_parse_head(repo: &Path) -> Result<String> {
    let output = Command::new("git")
        .current_dir(repo)
        .args(["rev-parse", "HEAD"])
        .output()
        .context("failed to spawn `git rev-parse HEAD`")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("git rev-parse HEAD failed: {stderr}");
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Returns the last-commit SHA that touched `rel_path`. Used only as a fast
/// skip signal on re-ingest; failures (e.g. a shallow clone that doesn't
/// reach the file's prior commits) just force a re-check.
fn last_commit_for_path(repo: &Path, rel_path: &str) -> Result<String> {
    let output = Command::new("git")
        .current_dir(repo)
        .args(["log", "-1", "--format=%H", "--", rel_path])
        .output()
        .context("failed to spawn `git log`")?;
    if !output.status.success() {
        bail!("git log for {rel_path} failed");
    }
    let sha = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if sha.is_empty() {
        bail!("no commit history for {rel_path}");
    }
    Ok(sha)
}

/// Builds the final file list using `ignore::WalkBuilder` so the repo's own
/// `.gitignore` is respected (per acceptance criteria). The include/exclude
/// globs are layered on top via `Override`.
fn collect_repo_files(
    repo_dir: &Path,
    includes: &[String],
    excludes: &[String],
) -> Result<Vec<PathBuf>> {
    let includes = if includes.is_empty() {
        DEFAULT_INCLUDES
            .iter()
            .map(|s| (*s).to_string())
            .collect::<Vec<_>>()
    } else {
        includes.to_vec()
    };

    let mut builder = OverrideBuilder::new(repo_dir);
    for pat in &includes {
        builder
            .add(pat)
            .with_context(|| format!("invalid include glob: {pat}"))?;
    }
    for pat in excludes {
        // `ignore` uses the gitignore convention: a `!` prefix means "include"
        // and a bare glob means "exclude". OverrideBuilder flips this (bare
        // means include) so we prefix user excludes with `!`.
        let excl = if pat.starts_with('!') {
            pat.trim_start_matches('!').to_string()
        } else {
            format!("!{pat}")
        };
        builder
            .add(&excl)
            .with_context(|| format!("invalid exclude glob: {pat}"))?;
    }
    let overrides = builder.build().context("failed to build override glob set")?;

    // Walk with stdout's `.gitignore` support enabled and drop hidden/.git
    // entries. `require_git(false)` lets us walk clones that lost their
    // `.git` dir (e.g. shallow exports).
    let mut seen: HashSet<PathBuf> = HashSet::new();
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in WalkBuilder::new(repo_dir)
        .standard_filters(true)
        .overrides(overrides)
        .hidden(true)
        .require_git(false)
        .build()
    {
        let entry = entry.with_context(|| format!("failed to walk {}", repo_dir.display()))?;
        if !entry.file_type().is_some_and(|kind| kind.is_file()) {
            continue;
        }
        let path = entry.path();
        // `.git` subtree isn't hidden-filtered on bare-clone workdirs in some
        // git versions; belt-and-suspenders exclude it explicitly.
        if path
            .components()
            .any(|c| c.as_os_str() == ".git" || c.as_os_str() == ".ingest")
        {
            continue;
        }
        let rel = path
            .strip_prefix(repo_dir)
            .context("walked entry outside repo root")?;
        if seen.insert(rel.to_path_buf()) {
            files.push(rel.to_path_buf());
        }
    }
    Ok(files)
}

/// Strips a leading `git+` so the URL is something `git clone` actually
/// accepts.
fn strip_git_plus(url: &str) -> String {
    url.trim().strip_prefix("git+").unwrap_or(url).to_string()
}

fn read_json<T>(path: &Path) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let contents =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse JSON {}", path.display()))
}

fn write_json<T>(path: &Path, value: &T) -> Result<()>
where
    T: Serialize,
{
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }
    let body = serde_json::to_string_pretty(value).context("failed to serialize JSON")?;
    fs::write(path, format!("{body}\n"))
        .with_context(|| format!("failed to write {}", path.display()))
}

fn now_millis() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock before Unix epoch")?
        .as_millis()
        .try_into()
        .context("timestamp overflow converting to u64")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn git_url_detects_common_shapes() {
        assert!(is_git_url("https://github.com/foo/bar"));
        assert!(is_git_url("https://github.com/foo/bar.git"));
        assert!(is_git_url("https://github.com/foo/bar.git/"));
        assert!(is_git_url("git@github.com:foo/bar.git"));
        assert!(is_git_url("git@github.com:foo/bar"));
        assert!(is_git_url("git+https://example.com/foo/bar.git"));
        assert!(is_git_url("git+ssh://git@example.com/foo/bar"));
        assert!(is_git_url("git://example.com/foo/bar"));
        assert!(is_git_url("ssh://git@example.com/foo/bar.git"));
        assert!(is_git_url("https://gitlab.com/group/project"));
        assert!(is_git_url("https://codeberg.org/user/repo"));
        assert!(is_git_url("https://example.com/foo.git"));
    }

    #[test]
    fn git_url_rejects_non_git() {
        assert!(!is_git_url("https://example.com/"));
        assert!(!is_git_url("https://github.com/"));
        assert!(!is_git_url("https://github.com/foo"));
        assert!(!is_git_url("/tmp/notes.md"));
        assert!(!is_git_url("./README.md"));
        assert!(!is_git_url(""));
        assert!(!is_git_url("ftp://foo/bar"));
    }

    #[test]
    fn normalize_git_url_strips_prefix_and_suffix() {
        assert_eq!(
            normalize_git_url("git+https://github.com/foo/bar.git"),
            "https://github.com/foo/bar"
        );
        assert_eq!(
            normalize_git_url("https://github.com/foo/bar.git/"),
            "https://github.com/foo/bar"
        );
        assert_eq!(
            normalize_git_url("git@github.com:foo/bar.git"),
            "git@github.com:foo/bar"
        );
    }
}
