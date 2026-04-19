#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use ignore::WalkBuilder;
use kb_core::{
    EntityMetadata, NormalizedDocument, SourceDocument, SourceKind, SourceRevision, Status,
    mint_source_document_id, mint_source_revision_id, normalize_file_stable_location,
    source_revision_content_hash, write_normalized_document,
};
use serde::{Deserialize, Serialize};

const SOURCE_DOCUMENT_RECORD: &str = "source_document.json";
const SOURCE_REVISION_RECORD: &str = "source_revision.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IngestOutcome {
    NewSource,
    NewRevision,
    Skipped,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalFileMetadata {
    pub original_path: String,
    pub size_bytes: u64,
    pub modified_at_millis: Option<u64>,
    pub content_hash: String,
    pub imported_at_millis: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestedSource {
    pub document: SourceDocument,
    pub revision: SourceRevision,
    pub copied_path: PathBuf,
    pub metadata_sidecar_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalIngestReport {
    pub source_path: PathBuf,
    pub outcome: IngestOutcome,
    pub ingested: IngestedSource,
}

/// Origin of a collected file — whether it was named explicitly on the command
/// line, or discovered via a directory walk. Binary rejection behavior differs
/// between the two: explicit files fail hard, directory-walk files are skipped
/// with a warning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileOrigin {
    Explicit,
    DirectoryWalk,
}

/// Maximum number of bytes inspected when deciding whether a file looks like
/// text. Larger files are only partially sampled.
const TEXT_PROBE_BYTES: usize = 8 * 1024;

/// Fraction of non-printable, non-whitespace bytes (in the probed prefix) at
/// which we decide a file is binary. Kept as a ratio out of 1000 so all math
/// stays in integers.
const NON_PRINTABLE_THRESHOLD_PER_THOUSAND: usize = 300;

/// Heuristic: does the first slice of a file look like UTF-8 text?
///
/// Rejects on any of:
/// - a NUL byte in the probed prefix
/// - the probed prefix is not valid UTF-8
/// - more than ~30% of the probed bytes are non-printable and not common
///   whitespace (space, tab, newline, carriage return, form feed)
#[must_use]
pub fn looks_like_text(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return true;
    }
    let probe_len = bytes.len().min(TEXT_PROBE_BYTES);
    let probe = &bytes[..probe_len];

    if probe.contains(&0) {
        return false;
    }

    // Must be valid UTF-8 up to the probe boundary. A multibyte char may be
    // sliced at the boundary, so accept a trailing-invalid error whose valid
    // prefix covers everything except a short tail.
    match std::str::from_utf8(probe) {
        Ok(_) => {}
        Err(err) => {
            let valid_up_to = err.valid_up_to();
            let tail = probe_len - valid_up_to;
            // Allow up to 3 trailing bytes (max UTF-8 continuation) to be cut
            // off only when we actually truncated (bytes longer than probe).
            if tail > 3 || bytes.len() <= probe_len {
                return false;
            }
        }
    }

    let non_printable = probe
        .iter()
        .filter(|&&b| !is_text_byte(b))
        .count();
    // non_printable / probe_len > threshold/1000
    non_printable * 1000 <= NON_PRINTABLE_THRESHOLD_PER_THOUSAND * probe_len
}

const fn is_text_byte(b: u8) -> bool {
    // Printable ASCII, common whitespace, or the high bit (possible UTF-8
    // continuation — UTF-8 validity was already checked above).
    matches!(b, 0x09 | 0x0A | 0x0C | 0x0D | 0x20..=0x7E) || b >= 0x80
}

/// # Errors
/// Returns an error if any source path cannot be read, walked, or ingested.
pub fn ingest_paths(root: &Path, sources: &[PathBuf]) -> Result<Vec<IngestedSource>> {
    Ok(ingest_paths_with_options(root, sources, false)?
        .into_iter()
        .map(|report| report.ingested)
        .collect())
}

/// # Errors
/// Returns an error if any source path cannot be read, walked, or ingested.
pub fn ingest_paths_with_options(
    root: &Path,
    sources: &[PathBuf],
    dry_run: bool,
) -> Result<Vec<LocalIngestReport>> {
    let mut files: Vec<(PathBuf, FileOrigin)> = Vec::new();
    for source in sources {
        collect_files(source, &mut files)?;
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));

    let mut ingested = Vec::with_capacity(files.len());
    for (file, origin) in files {
        match ingest_file(root, &file, dry_run, origin) {
            Ok(Some(report)) => ingested.push(report),
            Ok(None) => {}
            Err(err) => return Err(err),
        }
    }

    Ok(ingested)
}

fn collect_files(path: &Path, files: &mut Vec<(PathBuf, FileOrigin)>) -> Result<()> {
    if path.is_file() {
        files.push((path.to_path_buf(), FileOrigin::Explicit));
        return Ok(());
    }

    if path.is_dir() {
        for entry in WalkBuilder::new(path)
            .standard_filters(true)
            .require_git(false)
            .build()
        {
            let entry = entry.with_context(|| format!("failed to walk {}", path.display()))?;
            let is_hidden = entry
                .path()
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with('.'));
            if entry.file_type().is_some_and(|kind| kind.is_file()) && !is_hidden {
                files.push((entry.into_path(), FileOrigin::DirectoryWalk));
            }
        }
        return Ok(());
    }

    bail!(
        "source path does not exist or is not a regular file/directory: {}",
        path.display()
    )
}

/// Returns `Ok(true)` if the file should be ingested, `Ok(false)` if it was a
/// binary file from a directory walk that should be silently skipped, and
/// `Err` if it was an explicit binary file that should abort the run.
fn check_text_or_skip(
    canonical_source: &Path,
    content: &[u8],
    origin: FileOrigin,
) -> Result<bool> {
    if looks_like_text(content) {
        return Ok(true);
    }
    match origin {
        FileOrigin::Explicit => {
            bail!(
                "{} does not appear to be text (binary content detected)",
                canonical_source.display()
            );
        }
        FileOrigin::DirectoryWalk => {
            eprintln!(
                "warning: skipping {} (binary content detected)",
                canonical_source.display()
            );
            Ok(false)
        }
    }
}

fn ingest_file(
    root: &Path,
    source_path: &Path,
    dry_run: bool,
    origin: FileOrigin,
) -> Result<Option<LocalIngestReport>> {
    let canonical_source = fs::canonicalize(source_path)
        .with_context(|| format!("failed to canonicalize {}", source_path.display()))?;
    let stable_location = normalize_file_stable_location(&canonical_source)
        .with_context(|| format!("failed to normalize {}", source_path.display()))?;
    let content = fs::read(&canonical_source)
        .with_context(|| format!("failed to read {}", canonical_source.display()))?;
    let fs_metadata = fs::metadata(&canonical_source)
        .with_context(|| format!("failed to read metadata for {}", canonical_source.display()))?;

    if !check_text_or_skip(&canonical_source, &content, origin)? {
        return Ok(None);
    }

    let imported_at_millis = now_millis()?;
    let modified_at_millis = fs_metadata.modified().ok().and_then(system_time_to_millis);
    let source_document_id = mint_source_document_id(SourceKind::File, &stable_location);
    let source_revision_id = mint_source_revision_id(&content);
    let content_hash = source_revision_content_hash(&content);

    let original_name = canonical_source
        .file_name()
        .with_context(|| format!("missing file name for {}", canonical_source.display()))?;

    let relative_document_dir = PathBuf::from("raw").join("inbox").join(&source_document_id);
    let relative_revision_dir = relative_document_dir.join(&source_revision_id);
    let relative_copied_path = relative_revision_dir.join(original_name);
    let relative_metadata_sidecar_path =
        relative_revision_dir.join(format!("{}.metadata.json", original_name.to_string_lossy()));
    let relative_document_record_path = relative_document_dir.join(SOURCE_DOCUMENT_RECORD);
    let relative_revision_record_path = relative_revision_dir.join(SOURCE_REVISION_RECORD);

    let document_record_path = root.join(&relative_document_record_path);
    let revision_record_path = root.join(&relative_revision_record_path);
    let copied_path = root.join(&relative_copied_path);
    let metadata_sidecar_path = root.join(&relative_metadata_sidecar_path);

    let document_exists = document_record_path.exists();
    let revision_exists = revision_record_path.exists();

    let document = if document_exists {
        read_json::<SourceDocument>(&document_record_path)?
    } else {
        build_document(
            source_document_id,
            stable_location,
            imported_at_millis,
            content_hash.clone(),
            relative_document_record_path,
        )
    };

    let revision = if revision_exists {
        read_json::<SourceRevision>(&revision_record_path)?
    } else {
        build_revision(
            &document,
            source_revision_id,
            content_hash,
            imported_at_millis,
            fs_metadata.len(),
            &relative_copied_path,
            &relative_metadata_sidecar_path,
            &relative_revision_record_path,
        )
    };

    if !dry_run {
        if !document_exists {
            write_json(&document_record_path, &document)?;
        }
        if !revision_exists {
            write_new_revision(
                &canonical_source,
                &copied_path,
                &content,
                &fs_metadata,
                &metadata_sidecar_path,
                &revision_record_path,
                modified_at_millis,
                &revision,
            )?;
        }

        // Also produce the canonical NormalizedDocument so downstream passes
        // (compile, query) can consume the ingested source. The raw/ tree
        // remains the immutable source archive; normalized/ is the derived
        // view keyed by source-document id.
        write_normalized_for_file(root, &document, &revision, &content)?;
    }

    let outcome = if !document_exists {
        IngestOutcome::NewSource
    } else if !revision_exists {
        IngestOutcome::NewRevision
    } else {
        IngestOutcome::Skipped
    };

    Ok(Some(LocalIngestReport {
        source_path: canonical_source,
        outcome,
        ingested: IngestedSource {
            document,
            revision,
            copied_path: relative_copied_path,
            metadata_sidecar_path: relative_metadata_sidecar_path,
        },
    }))
}

fn build_document(
    source_document_id: String,
    stable_location: String,
    imported_at_millis: u64,
    content_hash: String,
    relative_document_record_path: PathBuf,
) -> SourceDocument {
    let metadata = EntityMetadata {
        id: source_document_id,
        created_at_millis: imported_at_millis,
        updated_at_millis: imported_at_millis,
        source_hashes: vec![content_hash],
        model_version: None,
        tool_version: Some(env!("CARGO_PKG_VERSION").to_string()),
        prompt_template_hash: None,
        dependencies: Vec::new(),
        output_paths: vec![relative_document_record_path],
        status: Status::Fresh,
    };

    SourceDocument {
        metadata,
        source_kind: SourceKind::File,
        stable_location,
        discovered_at_millis: imported_at_millis,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_revision(
    document: &SourceDocument,
    source_revision_id: String,
    content_hash: String,
    imported_at_millis: u64,
    size_bytes: u64,
    relative_copied_path: &Path,
    relative_metadata_sidecar_path: &Path,
    relative_revision_record_path: &Path,
) -> SourceRevision {
    let metadata = EntityMetadata {
        id: source_revision_id,
        created_at_millis: imported_at_millis,
        updated_at_millis: imported_at_millis,
        source_hashes: vec![content_hash.clone()],
        model_version: None,
        tool_version: Some(env!("CARGO_PKG_VERSION").to_string()),
        prompt_template_hash: None,
        dependencies: vec![document.metadata.id.clone()],
        output_paths: vec![
            relative_copied_path.to_path_buf(),
            relative_metadata_sidecar_path.to_path_buf(),
            relative_revision_record_path.to_path_buf(),
        ],
        status: Status::Fresh,
    };

    SourceRevision {
        metadata,
        source_document_id: document.metadata.id.clone(),
        fetched_revision_hash: content_hash,
        fetched_path: relative_copied_path.to_path_buf(),
        fetched_size_bytes: size_bytes,
        fetched_at_millis: imported_at_millis,
    }
}

#[allow(clippy::too_many_arguments)]
fn write_new_revision(
    canonical_source: &Path,
    copied_path: &Path,
    content: &[u8],
    fs_metadata: &fs::Metadata,
    metadata_sidecar_path: &Path,
    revision_record_path: &Path,
    modified_at_millis: Option<u64>,
    revision: &SourceRevision,
) -> Result<()> {
    fs::create_dir_all(
        copied_path
            .parent()
            .context("copied path should have a parent directory")?,
    )
    .with_context(|| format!("failed to create directory for {}", copied_path.display()))?;
    fs::write(copied_path, content).with_context(|| {
        format!(
            "failed to copy {} to {}",
            canonical_source.display(),
            copied_path.display()
        )
    })?;

    let sidecar = LocalFileMetadata {
        original_path: canonical_source.to_string_lossy().into_owned(),
        size_bytes: fs_metadata.len(),
        modified_at_millis,
        content_hash: revision.fetched_revision_hash.clone(),
        imported_at_millis: revision.fetched_at_millis,
    };
    write_json(metadata_sidecar_path, &sidecar)?;
    write_json(revision_record_path, revision)?;
    Ok(())
}

/// Writes a canonical `NormalizedDocument` for an ingested local file so that
/// `kb compile` and other downstream passes can consume it from `normalized/`.
///
/// The raw file already lives under `raw/inbox/<src>/<rev>/`; here we produce
/// the derived view keyed by the source-document id. For non-UTF8 files we
/// fall back to lossy decoding rather than refusing ingest — the upstream
/// archive copy still preserves exact bytes.
fn write_normalized_for_file(
    root: &Path,
    document: &SourceDocument,
    revision: &SourceRevision,
    content: &[u8],
) -> Result<()> {
    let canonical_text = std::str::from_utf8(content).map_or_else(
        |_| String::from_utf8_lossy(content).into_owned(),
        ToOwned::to_owned,
    );
    let heading_ids = extract_heading_ids(&canonical_text);

    let now = now_millis()?;
    let normalized_metadata = EntityMetadata {
        id: document.metadata.id.clone(),
        created_at_millis: document.metadata.created_at_millis,
        updated_at_millis: now,
        source_hashes: vec![revision.fetched_revision_hash.clone()],
        model_version: None,
        tool_version: Some(env!("CARGO_PKG_VERSION").to_string()),
        prompt_template_hash: None,
        dependencies: vec![revision.metadata.id.clone()],
        output_paths: vec![
            PathBuf::from("normalized")
                .join(&document.metadata.id)
                .join("source.md"),
            PathBuf::from("normalized")
                .join(&document.metadata.id)
                .join("metadata.json"),
        ],
        status: Status::Fresh,
    };

    let normalized = NormalizedDocument {
        metadata: normalized_metadata,
        source_revision_id: revision.metadata.id.clone(),
        canonical_text,
        normalized_assets: Vec::new(),
        heading_ids,
    };

    write_normalized_document(root, &normalized)
        .with_context(|| format!("failed to write normalized document {}", document.metadata.id))?;
    Ok(())
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

fn system_time_to_millis(value: SystemTime) -> Option<u64> {
    value
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| duration.as_millis().try_into().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn init_kb_root() -> TempDir {
        let temp = TempDir::new().expect("create tempdir");
        fs::create_dir_all(temp.path().join("raw/inbox")).expect("create raw inbox");
        temp
    }

    #[test]
    fn ingest_file_copies_content_and_writes_records() {
        let kb_root = init_kb_root();
        let source_root = TempDir::new().expect("create source tempdir");
        let file = source_root.path().join("notes.md");
        fs::write(&file, "hello world\n").expect("write source file");

        let ingested = ingest_paths(kb_root.path(), &[file]).expect("ingest succeeds");
        assert_eq!(ingested.len(), 1);

        let item = &ingested[0];
        assert!(kb_root.path().join(&item.copied_path).is_file());
        assert!(kb_root.path().join(&item.metadata_sidecar_path).is_file());
        assert!(
            kb_root
                .path()
                .join("raw/inbox")
                .join(&item.document.metadata.id)
                .join(SOURCE_DOCUMENT_RECORD)
                .is_file()
        );
        assert_eq!(item.revision.source_document_id, item.document.metadata.id);
    }

    #[test]
    fn reingesting_same_file_is_idempotent() {
        let kb_root = init_kb_root();
        let source_root = TempDir::new().expect("create source tempdir");
        let file = source_root.path().join("notes.md");
        fs::write(&file, "same contents\n").expect("write source file");

        let first =
            ingest_paths(kb_root.path(), std::slice::from_ref(&file)).expect("first ingest");
        let second = ingest_paths(kb_root.path(), &[file]).expect("second ingest");

        assert_eq!(
            first[0].document.metadata.id,
            second[0].document.metadata.id
        );
        assert_eq!(
            first[0].revision.metadata.id,
            second[0].revision.metadata.id
        );
    }

    #[test]
    fn changed_content_produces_new_revision_for_same_document() {
        let kb_root = init_kb_root();
        let source_root = TempDir::new().expect("create source tempdir");
        let file = source_root.path().join("notes.md");
        fs::write(&file, "version one\n").expect("write source file");
        let first =
            ingest_paths(kb_root.path(), std::slice::from_ref(&file)).expect("first ingest");

        fs::write(&file, "version two\n").expect("rewrite source file");
        let second = ingest_paths(kb_root.path(), &[file]).expect("second ingest");

        assert_eq!(
            first[0].document.metadata.id,
            second[0].document.metadata.id
        );
        assert_ne!(
            first[0].revision.metadata.id,
            second[0].revision.metadata.id
        );
    }

    #[test]
    fn directory_ingest_honors_gitignore() {
        let kb_root = init_kb_root();
        let source_root = TempDir::new().expect("create source tempdir");
        fs::write(source_root.path().join(".gitignore"), "ignored.md\n").expect("write gitignore");
        fs::write(source_root.path().join("kept.md"), "keep me\n").expect("write kept file");
        fs::write(source_root.path().join("ignored.md"), "ignore me\n")
            .expect("write ignored file");

        let ingested = ingest_paths(kb_root.path(), &[source_root.path().to_path_buf()])
            .expect("directory ingest succeeds");

        assert_eq!(ingested.len(), 1);
        assert!(ingested[0].copied_path.ends_with("kept.md"));
    }

    #[test]
    fn file_ingest_writes_normalized_document_roundtrip() {
        let kb_root = init_kb_root();
        let source_root = TempDir::new().expect("create source tempdir");
        let file = source_root.path().join("notes.md");
        fs::write(
            &file,
            "# Alpha\n\nBody text.\n\n## Beta section\n\nmore\n\n### Gamma\n",
        )
        .expect("write source file");

        let ingested = ingest_paths(kb_root.path(), &[file]).expect("ingest succeeds");
        let src_id = ingested[0].document.metadata.id.clone();
        let rev_id = ingested[0].revision.metadata.id.clone();

        // The on-disk metadata.json must deserialize via the canonical reader.
        let doc = kb_core::read_normalized_document(kb_root.path(), &src_id)
            .expect("read_normalized_document round-trip");

        assert_eq!(doc.metadata.id, src_id);
        assert_eq!(doc.source_revision_id, rev_id);
        assert!(doc.canonical_text.starts_with("# Alpha"));
        assert_eq!(
            doc.heading_ids,
            vec![
                "alpha".to_string(),
                "beta-section".to_string(),
                "gamma".to_string(),
            ]
        );
        assert!(doc.normalized_assets.is_empty());
    }

    #[test]
    fn looks_like_text_handles_common_cases() {
        assert!(looks_like_text(b""));
        assert!(looks_like_text(b"hello world\n"));
        assert!(looks_like_text(b"# Heading\n\nPara\twith\ttabs\r\n"));
        assert!(looks_like_text("utf-8 text: café — résumé".as_bytes()));

        // NUL byte -> binary
        assert!(!looks_like_text(b"oops\0still text"));
        // Invalid UTF-8 (lone continuation)
        assert!(!looks_like_text(&[0x80, 0x80, 0x80, 0x80]));
        // Mostly control bytes -> binary
        let binary: Vec<u8> = (0..100u8).map(|i| i % 32).collect();
        assert!(!looks_like_text(&binary));
    }

    #[test]
    fn ingest_rejects_binary_file() {
        let kb_root = init_kb_root();
        let source_root = TempDir::new().expect("create source tempdir");
        let file = source_root.path().join("program.bin");
        // ELF-ish: contains NUL bytes
        let bytes: Vec<u8> = vec![0x7f, b'E', b'L', b'F', 0, 0, 0, 1, 2, 3, 0, 0, 0, 0xff];
        fs::write(&file, &bytes).expect("write binary file");

        let err = ingest_paths(kb_root.path(), &[file]).expect_err("ingest should fail");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("does not appear to be text"),
            "error should mention binary detection, got: {msg}"
        );
    }

    #[test]
    fn ingest_directory_skips_binary_files_and_keeps_text() {
        let kb_root = init_kb_root();
        let source_root = TempDir::new().expect("create source tempdir");
        fs::write(source_root.path().join("notes.md"), "hello text\n")
            .expect("write text file");
        let binary: Vec<u8> = vec![0x7f, b'E', b'L', b'F', 0, 0, 0, 1, 2, 3];
        fs::write(source_root.path().join("program.bin"), &binary)
            .expect("write binary file");

        let ingested =
            ingest_paths(kb_root.path(), &[source_root.path().to_path_buf()])
                .expect("directory ingest succeeds despite binary file");

        assert_eq!(ingested.len(), 1, "only the text file should be ingested");
        assert!(ingested[0].copied_path.ends_with("notes.md"));
    }

    #[test]
    fn dry_run_reports_changes_without_writing_files() {
        let kb_root = init_kb_root();
        let source_root = TempDir::new().expect("create source tempdir");
        let file = source_root.path().join("notes.md");
        fs::write(&file, "hello dry run\n").expect("write source file");

        let reports =
            ingest_paths_with_options(kb_root.path(), &[file], true).expect("dry run succeeds");

        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].outcome, IngestOutcome::NewSource);
        assert!(
            !kb_root
                .path()
                .join(&reports[0].ingested.copied_path)
                .exists()
        );
        assert!(
            fs::read_dir(kb_root.path().join("raw/inbox"))
                .expect("read raw inbox")
                .next()
                .is_none()
        );
    }
}

mod headings;
mod url;

pub use headings::extract_heading_ids;
pub use url::{UrlIngestReport, ingest_url, ingest_url_with_options};
