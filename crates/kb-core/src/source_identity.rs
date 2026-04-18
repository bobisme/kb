use std::fs;
use std::io;
use std::path::Path;

use anyhow::{Context, Result};
use url::Url;

use crate::{SourceKind, hash_bytes};

pub const SOURCE_DOCUMENT_ID_PREFIX: &str = "src";
pub const SOURCE_REVISION_ID_PREFIX: &str = "rev";
const SHORT_HASH_LEN: usize = 8;

/// Returns a canonical stable location string for a file source.
///
/// File identities are based on the canonicalized absolute path so that repeated
/// ingests of the same logical file mint the same `SourceDocument` ID.
///
/// # Errors
/// Returns an error if the path cannot be canonicalized.
pub fn normalize_file_stable_location(path: impl AsRef<Path>) -> io::Result<String> {
    let canonical = fs::canonicalize(path)?;
    Ok(canonical.to_string_lossy().into_owned())
}

/// Returns a normalized stable location string for a URL source.
///
/// Normalization rules:
/// - lowercase scheme and host
/// - preserve the path produced by URL parsing
/// - sort query parameters by key then value
/// - drop fragments because they identify sublocations within a document, not the
///   logical source itself
///
/// # Errors
/// Returns an error if the URL cannot be parsed.
pub fn normalize_url_stable_location(raw_url: &str) -> Result<String> {
    let mut url = Url::parse(raw_url).with_context(|| format!("invalid URL: {raw_url}"))?;
    url.set_fragment(None);

    let mut query_pairs = url
        .query_pairs()
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect::<Vec<_>>();
    query_pairs.sort();

    {
        let mut pairs = url.query_pairs_mut();
        pairs.clear();
        for (key, value) in query_pairs {
            pairs.append_pair(&key, &value);
        }
    }

    Ok(url.into())
}

/// Mints a short source document ID from a source kind and normalized location.
#[must_use]
pub fn mint_source_document_id(source_kind: SourceKind, stable_location: &str) -> String {
    let kind = match source_kind {
        SourceKind::File => "file",
        SourceKind::Url => "url",
        SourceKind::Repo => "repo",
        SourceKind::Image => "image",
        SourceKind::Dataset => "dataset",
        SourceKind::Other => "other",
    };

    mint_short_id(
        SOURCE_DOCUMENT_ID_PREFIX,
        &format!("{kind}\n{stable_location}"),
    )
}

/// Returns the full BLAKE3 content hash for a fetched source revision.
#[must_use]
pub fn source_revision_content_hash(content: &[u8]) -> String {
    hash_bytes(content).to_hex()
}

/// Mints a short source revision ID from fetched content bytes.
#[must_use]
pub fn mint_source_revision_id(content: &[u8]) -> String {
    mint_short_id(
        SOURCE_REVISION_ID_PREFIX,
        &source_revision_content_hash(content),
    )
}

/// Canonicalizes a file path and mints the corresponding source document ID.
///
/// # Errors
/// Returns an error if the path cannot be canonicalized.
pub fn source_document_id_for_file(path: impl AsRef<Path>) -> io::Result<String> {
    let stable_location = normalize_file_stable_location(path)?;
    Ok(mint_source_document_id(SourceKind::File, &stable_location))
}

/// Normalizes a URL and mints the corresponding source document ID.
///
/// # Errors
/// Returns an error if the URL cannot be parsed.
pub fn source_document_id_for_url(raw_url: &str) -> Result<String> {
    let stable_location = normalize_url_stable_location(raw_url)?;
    Ok(mint_source_document_id(SourceKind::Url, &stable_location))
}

fn mint_short_id(prefix: &str, material: &str) -> String {
    let hash = hash_bytes(material.as_bytes()).to_hex();
    format!("{prefix}-{}", &hash[..SHORT_HASH_LEN])
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn file_reingest_produces_same_source_document_id() {
        let temp = TempDir::new().unwrap();
        let nested = temp.path().join("nested");
        fs::create_dir_all(&nested).unwrap();

        let file = nested.join("source.txt");
        fs::write(&file, b"hello world").unwrap();

        let via_direct = source_document_id_for_file(&file).unwrap();
        let via_relative_segments =
            source_document_id_for_file(nested.join("../nested/source.txt")).unwrap();

        assert_eq!(via_direct, via_relative_segments);
        assert_eq!(
            normalize_file_stable_location(&file).unwrap(),
            normalize_file_stable_location(nested.join("../nested/source.txt")).unwrap()
        );
    }

    #[test]
    fn url_normalization_lowercases_host_and_sorts_query_params() {
        let normalized =
            normalize_url_stable_location("HTTPS://Example.COM/a/path?z=last&b=two&a=one#section")
                .unwrap();

        assert_eq!(normalized, "https://example.com/a/path?a=one&b=two&z=last");
    }

    #[test]
    fn url_refetch_with_unchanged_content_reuses_revision() {
        let first = mint_source_revision_id(b"same bytes");
        let second = mint_source_revision_id(b"same bytes");

        assert_eq!(first, second);
        assert_eq!(
            source_revision_content_hash(b"same bytes"),
            source_revision_content_hash(b"same bytes")
        );
    }

    #[test]
    fn url_refetch_with_changed_content_mints_new_revision() {
        let first = mint_source_revision_id(b"version one");
        let second = mint_source_revision_id(b"version two");

        assert_ne!(first, second);
        assert_ne!(
            source_revision_content_hash(b"version one"),
            source_revision_content_hash(b"version two")
        );
    }

    #[test]
    fn url_document_id_is_stable_for_equivalent_urls() {
        let first = source_document_id_for_url("https://Example.com/a?b=2&a=1").unwrap();
        let second = source_document_id_for_url("https://example.com/a?a=1&b=2").unwrap();

        assert_eq!(first, second);
    }
}
