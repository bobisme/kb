use std::collections::HashSet;
use std::io;
use std::path::{Path, PathBuf};

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::{EntityMetadata, NormalizedDocument};
use crate::fs::atomic_write;

const METADATA_FILE_NAME: &str = "metadata.json";
const SOURCE_MARKDOWN_FILE_NAME: &str = "source.md";
const ASSETS_DIR_NAME: &str = "assets";

#[derive(Debug, Serialize, Deserialize)]
struct NormalizedDocumentMetadata {
    metadata: EntityMetadata,
    source_revision_id: String,
    normalized_assets: Vec<PathBuf>,
    heading_ids: Vec<String>,
}

impl NormalizedDocumentMetadata {
    fn from_document(doc: &NormalizedDocument) -> Self {
        Self {
            metadata: doc.metadata.clone(),
            source_revision_id: doc.source_revision_id.clone(),
            normalized_assets: doc.normalized_assets.clone(),
            heading_ids: doc.heading_ids.clone(),
        }
    }

    fn into_document(self, canonical_text: String) -> NormalizedDocument {
        NormalizedDocument {
            metadata: self.metadata,
            source_revision_id: self.source_revision_id,
            canonical_text,
            normalized_assets: self.normalized_assets,
            heading_ids: self.heading_ids,
        }
    }
}

/// Writes a NormalizedDocument to the standard storage layout.
///
/// Layout under `root`:
/// - `normalized/<doc-id>/source.md`
/// - `normalized/<doc-id>/metadata.json`
/// - `normalized/<doc-id>/assets/` (directory created if needed)
pub fn write_normalized_document(
    root: impl AsRef<Path>,
    doc: &NormalizedDocument,
) -> io::Result<()> {
    let base_dir = root.as_ref().join("normalized").join(&doc.metadata.id);
    std::fs::create_dir_all(&base_dir)?;
    let assets_dir = base_dir.join(ASSETS_DIR_NAME);
    std::fs::create_dir_all(&assets_dir)?;

    let referenced_assets = extract_referenced_assets(&doc.canonical_text);

    copy_and_validate_assets(doc, &assets_dir)?;
    validate_asset_references(&referenced_assets, doc)?;

    let source_path = base_dir.join(SOURCE_MARKDOWN_FILE_NAME);
    atomic_write(&source_path, doc.canonical_text.as_bytes())?;

    let metadata = NormalizedDocumentMetadata::from_document(doc);
    let metadata_path = base_dir.join(METADATA_FILE_NAME);
    let metadata_bytes = serde_json::to_vec_pretty(&metadata).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to serialize normalized metadata: {err}"),
        )
    })?;
    atomic_write(&metadata_path, &metadata_bytes)?;

    Ok(())
}

/// Reads a NormalizedDocument from the standard storage layout.
pub fn read_normalized_document(root: impl AsRef<Path>, id: &str) -> io::Result<NormalizedDocument> {
    let base_dir = root.as_ref().join("normalized").join(id);
    let metadata_path = base_dir.join(METADATA_FILE_NAME);
    let source_path = base_dir.join(SOURCE_MARKDOWN_FILE_NAME);

    let metadata_bytes = std::fs::read(metadata_path)?;
    let metadata: NormalizedDocumentMetadata = serde_json::from_slice(&metadata_bytes).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to deserialize normalized metadata: {err}"),
        )
    })?;

    let canonical_text = std::fs::read_to_string(source_path)?;
    Ok(metadata.into_document(canonical_text))
}

fn copy_and_validate_assets(doc: &NormalizedDocument, assets_dir: &Path) -> io::Result<()> {
    let mut written: HashSet<&std::ffi::OsStr> = HashSet::new();

    for asset_path in &doc.normalized_assets {
        let file_name = match asset_path.file_name() {
            Some(name) => name,
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "normalized asset path must include a filename",
                ));
            }
        };

        if !written.insert(file_name) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("duplicate asset filename: {}", file_name.display()),
            ));
        }

        if !asset_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("normalized asset not found: {}", asset_path.display()),
            ));
        }

        let destination = assets_dir.join(file_name);
        let asset_bytes = std::fs::read(asset_path)?;
        atomic_write(&destination, &asset_bytes)?;
    }

    Ok(())
}

fn validate_asset_references(
    referenced_assets: &[String],
    doc: &NormalizedDocument,
) -> io::Result<()> {
    if referenced_assets.is_empty() {
        return Ok(());
    }

    let available = doc
        .normalized_assets
        .iter()
        .filter_map(|path| path.file_name())
        .collect::<HashSet<_>>();

    for asset_ref in referenced_assets {
        let asset_file = Path::new(asset_ref)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(asset_ref.as_str());

        if !available.contains(Path::new(asset_file).as_os_str()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unresolved asset reference in source.md: {asset_ref}"),
            ));
        }
    }

    Ok(())
}

fn extract_referenced_assets(source_text: &str) -> Vec<String> {
    let mut references = Vec::new();
    let re = Regex::new(r#"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)|\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?"#)
        .expect("asset regex compile");

    for capture in re.captures_iter(source_text) {
        let raw_path = match (capture.get(1), capture.get(2)) {
            (Some(path), _) => path.as_str(),
            (_, Some(path)) => path.as_str(),
            _ => continue,
        };

        if let Some(asset_ref) = normalize_asset_reference(raw_path) {
            references.push(asset_ref);
        }
    }

    references
}

fn normalize_asset_reference(raw_path: &str) -> Option<String> {
    let normalized = raw_path.trim().trim_start_matches("./");

    if normalized.is_empty()
        || normalized.starts_with("http://")
        || normalized.starts_with("https://")
        || normalized.starts_with("mailto:")
        || normalized.starts_with("#")
    {
        return None;
    }

    let normalized = normalized.trim_start_matches('/');
    if normalized.starts_with("assets/") {
        return Some(normalized.trim_start_matches("assets/").to_string());
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn round_trip_normalized_document() -> io::Result<()> {
        let temp = TempDir::new()?;
        let root = temp.path();

        let asset_path = root.join("img.png");
        fs::write(&asset_path, b"image bytes")?;

        let document = NormalizedDocument {
            metadata: crate::EntityMetadata {
                id: "doc-001".to_string(),
                created_at_millis: 1,
                updated_at_millis: 2,
                source_hashes: vec!["abc".to_string()],
                model_version: None,
                tool_version: None,
                prompt_template_hash: None,
                dependencies: vec![],
                output_paths: vec![],
                status: crate::Status::Fresh,
            },
            source_revision_id: "rev-001".to_string(),
            canonical_text: "# Title\n![logo](assets/img.png)\n".to_string(),
            normalized_assets: vec![asset_path.clone()],
            heading_ids: vec!["title".to_string()],
        };

        write_normalized_document(root, &document)?;

        let written_doc = read_normalized_document(root, &document.metadata.id)?;
        assert_eq!(written_doc.source_revision_id, document.source_revision_id);
        assert_eq!(written_doc.canonical_text, document.canonical_text);
        assert_eq!(written_doc.normalized_assets, document.normalized_assets);
        assert!(root
            .join("normalized")
            .join(document.metadata.id.clone())
            .join("source.md")
            .is_file());
        assert!(root
            .join("normalized")
            .join(document.metadata.id.clone())
            .join("assets")
            .join("img.png")
            .is_file());
        let metadata_path = root
            .join("normalized")
            .join(&document.metadata.id)
            .join("metadata.json");
        let metadata_text = fs::read_to_string(metadata_path)?;
        assert!(metadata_text.contains("\"heading_ids\""));
        assert!(!metadata_text.contains("\"canonical_text\""));

        Ok(())
    }

    #[test]
    fn metadata_file_excludes_canonical_text() -> io::Result<()> {
        let temp = TempDir::new()?;
        let root = temp.path();

        let document = NormalizedDocument {
            metadata: crate::EntityMetadata {
                id: "doc-no-body".to_string(),
                created_at_millis: 1,
                updated_at_millis: 2,
                source_hashes: vec!["abc".to_string()],
                model_version: None,
                tool_version: None,
                prompt_template_hash: None,
                dependencies: vec![],
                output_paths: vec![],
                status: crate::Status::Fresh,
            },
            source_revision_id: "rev-no-body".to_string(),
            canonical_text: "plain body".to_string(),
            normalized_assets: vec![],
            heading_ids: vec![],
        };

        write_normalized_document(root, &document)?;

        let metadata = fs::read_to_string(root.join("normalized").join("doc-no-body").join("metadata.json"))?;
        assert!(metadata.contains("\"heading_ids\""));
        assert!(!metadata.contains("\"canonical_text\""));
        Ok(())
    }

    #[test]
    fn unresolved_asset_reference_is_an_error() -> io::Result<()> {
        let temp = TempDir::new()?;
        let root = temp.path();

        let document = NormalizedDocument {
            metadata: crate::EntityMetadata {
                id: "doc-bad".to_string(),
                created_at_millis: 1,
                updated_at_millis: 2,
                source_hashes: vec!["abc".to_string()],
                model_version: None,
                tool_version: None,
                prompt_template_hash: None,
                dependencies: vec![],
                output_paths: vec![],
                status: crate::Status::Fresh,
            },
            source_revision_id: "rev-bad".to_string(),
            canonical_text: "![missing](assets/does-not-exist.png)".to_string(),
            normalized_assets: vec![],
            heading_ids: vec![],
        };

        let result = write_normalized_document(root, &document);
        assert!(result.is_err());
        Ok(())
    }
}
