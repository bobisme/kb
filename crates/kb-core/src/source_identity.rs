use std::fs;
use std::io;
use std::path::Path;

use anyhow::{Context, Result};
use terseid::{IdConfig, IdGenerator};
use url::Url;

use crate::{SourceKind, hash_bytes};

pub const SOURCE_DOCUMENT_ID_PREFIX: &str = "src";
pub const SOURCE_REVISION_ID_PREFIX: &str = "rev";

/// Fixed length (in base36 chars) of the content-addressed revision hash. The
/// rev-id is derived purely from the content hash, so we skip `IdGenerator`'s
/// collision-retry machinery — two different revisions with different content
/// should always produce different rev-ids at this length, and two identical
/// contents must round-trip to the same rev-id on ingest/re-ingest.
const REVISION_HASH_LEN: usize = 6;

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
/// - lowercase scheme and host (the `url` crate lowercases the scheme
///   automatically during parsing, which also covers mixed-case inputs like
///   `HTTPS://` / `Http://`)
/// - collapse a single trailing slash on non-root paths so that
///   `https://foo.com/bar/` and `https://foo.com/bar` are treated as the same
///   source. The root path `/` is preserved because `https://foo.com/` is the
///   canonical form produced by URL parsers (empty-path URLs round-trip to a
///   trailing slash anyway).
/// - sort query parameters by key then value
/// - drop fragments because they identify sublocations within a document, not the
///   logical source itself
///
/// # Errors
/// Returns an error if the URL cannot be parsed.
pub fn normalize_url_stable_location(raw_url: &str) -> Result<String> {
    let mut url = Url::parse(raw_url).with_context(|| format!("invalid URL: {raw_url}"))?;
    url.set_fragment(None);

    // Collapse a trailing slash on non-root paths. Keep root `/` intact: an
    // input with no path (e.g. `https://foo.com`) is normalized by the `url`
    // crate to `https://foo.com/`, which is the canonical root form.
    let path = url.path().to_owned();
    if path.len() > 1 && path.ends_with('/') {
        let trimmed = path.trim_end_matches('/');
        // `trim_end_matches` would leave an empty string for `////`; fall back
        // to `/` so we never produce a path-less URL.
        let new_path = if trimmed.is_empty() { "/" } else { trimmed };
        url.set_path(new_path);
    }

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
    // `query_pairs_mut` always leaves a `?` even when no pairs were appended,
    // so clear it manually when there were none to avoid a dangling `?`.
    if url.query() == Some("") {
        url.set_query(None);
    }

    Ok(url.into())
}

/// Mints a short source document ID from a source kind and normalized
/// location, using `terseid` for adaptive-length hashes with collision retry.
///
/// `existing_count` is the number of source documents already in the KB; it
/// drives `terseid`'s adaptive-length heuristic so small KBs get 3-char ids
/// and larger KBs automatically grow. `exists` is the caller-supplied
/// collision probe — it should return `true` only when the candidate id is
/// already taken by a *different* source (otherwise re-ingesting the same
/// file would mint a fresh id on every pass).
#[must_use]
pub fn mint_source_document_id(
    source_kind: SourceKind,
    stable_location: &str,
    existing_count: usize,
    exists: impl Fn(&str) -> bool,
) -> String {
    let kind = source_kind_tag(source_kind);
    let seed_material = format!("{kind}\n{stable_location}");

    let generator = IdGenerator::new(IdConfig::new(SOURCE_DOCUMENT_ID_PREFIX));
    generator.generate(
        |nonce| {
            if nonce == 0 {
                seed_material.as_bytes().to_vec()
            } else {
                // Nonce escalation: terseid retries with a bumped seed when
                // the primary candidate collides. Appending the nonce keeps
                // nonce=0 identical to the content-free seed so the common
                // "no collision" path is deterministic across re-ingests.
                format!("{seed_material}|{nonce}").into_bytes()
            }
        },
        existing_count,
        |candidate| exists(candidate),
    )
}

const fn source_kind_tag(source_kind: SourceKind) -> &'static str {
    match source_kind {
        SourceKind::File => "file",
        SourceKind::Url => "url",
        SourceKind::Repo => "repo",
        SourceKind::Image => "image",
        SourceKind::Dataset => "dataset",
        SourceKind::Other => "other",
    }
}

/// Returns the full BLAKE3 content hash for a fetched source revision.
#[must_use]
pub fn source_revision_content_hash(content: &[u8]) -> String {
    hash_bytes(content).to_hex()
}

/// Mints a short, content-addressable source revision ID from fetched bytes.
///
/// We deliberately bypass `IdGenerator`'s collision-retry machinery here: the
/// rev-id must be a pure function of the content so that re-fetching the
/// same bytes (even under a different URL) yields the same rev-id. We use
/// `terseid::hash` with a fixed 6-char length, prefixed with `rev-`.
#[must_use]
pub fn mint_source_revision_id(content: &[u8]) -> String {
    let content_hash = source_revision_content_hash(content);
    let short = terseid::hash(&content_hash, REVISION_HASH_LEN);
    format!("{SOURCE_REVISION_ID_PREFIX}-{short}")
}

/// Canonicalizes a file path and mints the corresponding source document ID.
///
/// # Errors
/// Returns an error if the path cannot be canonicalized.
pub fn source_document_id_for_file(
    path: impl AsRef<Path>,
    existing_count: usize,
    exists: impl Fn(&str) -> bool,
) -> io::Result<String> {
    let stable_location = normalize_file_stable_location(path)?;
    Ok(mint_source_document_id(
        SourceKind::File,
        &stable_location,
        existing_count,
        exists,
    ))
}

/// Normalizes a URL and mints the corresponding source document ID.
///
/// # Errors
/// Returns an error if the URL cannot be parsed.
pub fn source_document_id_for_url(
    raw_url: &str,
    existing_count: usize,
    exists: impl Fn(&str) -> bool,
) -> Result<String> {
    let stable_location = normalize_url_stable_location(raw_url)?;
    Ok(mint_source_document_id(
        SourceKind::Url,
        &stable_location,
        existing_count,
        exists,
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Convenience: no-collision existence predicate for tests that aren't
    /// exercising the retry path.
    const fn never_exists(_: &str) -> bool {
        false
    }

    #[test]
    fn source_document_id_has_src_prefix_and_short_hash() {
        let id = mint_source_document_id(SourceKind::File, "/tmp/x.md", 0, never_exists);
        assert!(id.starts_with("src-"), "expected src- prefix, got {id}");
        // 3 chars at count=0 is terseid's adaptive minimum for tiny corpora.
        let hash = id.trim_start_matches("src-");
        assert_eq!(
            hash.len(),
            3,
            "expected 3-char adaptive hash for empty KB, got {id}"
        );
    }

    #[test]
    fn source_revision_id_has_rev_prefix_and_six_char_hash() {
        let id = mint_source_revision_id(b"hello");
        assert!(id.starts_with("rev-"), "expected rev- prefix, got {id}");
        let hash = id.trim_start_matches("rev-");
        assert_eq!(hash.len(), 6, "rev-id hash must be fixed-length 6, got {id}");
    }

    #[test]
    fn file_reingest_produces_same_source_document_id() {
        let temp = TempDir::new().unwrap();
        let nested = temp.path().join("nested");
        fs::create_dir_all(&nested).unwrap();

        let file = nested.join("source.txt");
        fs::write(&file, b"hello world").unwrap();

        let via_direct = source_document_id_for_file(&file, 0, never_exists).unwrap();
        let via_relative_segments =
            source_document_id_for_file(nested.join("../nested/source.txt"), 0, never_exists)
                .unwrap();

        assert_eq!(via_direct, via_relative_segments);
        assert_eq!(
            normalize_file_stable_location(&file).unwrap(),
            normalize_file_stable_location(nested.join("../nested/source.txt")).unwrap()
        );
    }

    #[test]
    fn same_content_different_paths_share_rev_id_but_not_src_id() {
        // Content-addressable rev-ids: the same bytes at two different
        // stable_locations must mint distinct src-ids but an identical
        // rev-id. This is the contract downstream code relies on when it
        // dedupes revisions across URLs / filesystem paths.
        let src_a =
            mint_source_document_id(SourceKind::File, "/tmp/a.md", 0, never_exists);
        let src_b =
            mint_source_document_id(SourceKind::File, "/tmp/b.md", 0, never_exists);
        assert_ne!(src_a, src_b);

        let rev_a = mint_source_revision_id(b"identical bytes");
        let rev_b = mint_source_revision_id(b"identical bytes");
        assert_eq!(rev_a, rev_b);
    }

    #[test]
    fn collision_forces_length_extension() {
        // Simulate every 3-char id for this prefix being taken. terseid's
        // nonce-escalation exhausts at length 3, then bumps to length 4.
        let id = mint_source_document_id(
            SourceKind::File,
            "/tmp/needs-longer-id.md",
            0,
            |candidate| candidate.trim_start_matches("src-").len() == 3,
        );
        let hash = id.trim_start_matches("src-");
        assert!(
            hash.len() >= 4,
            "exists-check should have pushed past 3 chars, got {id}"
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
        let first =
            source_document_id_for_url("https://Example.com/a?b=2&a=1", 0, never_exists).unwrap();
        let second =
            source_document_id_for_url("https://example.com/a?a=1&b=2", 0, never_exists).unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn url_document_id_ignores_scheme_case() {
        // Per RFC 3986, URL schemes are case-insensitive. `HTTPS://` and
        // `https://` must collide, otherwise `kb ingest HTTPS://example.com/foo`
        // and `kb ingest https://example.com/foo` would mint different
        // src-ids for the same resource.
        let upper =
            source_document_id_for_url("HTTPS://example.com/foo", 0, never_exists).unwrap();
        let lower =
            source_document_id_for_url("https://example.com/foo", 0, never_exists).unwrap();

        assert_eq!(upper, lower);
    }

    #[test]
    fn url_document_id_collapses_trailing_slash_on_nonroot_path() {
        let with_slash =
            source_document_id_for_url("https://foo.com/bar/", 0, never_exists).unwrap();
        let without_slash =
            source_document_id_for_url("https://foo.com/bar", 0, never_exists).unwrap();

        assert_eq!(with_slash, without_slash);
    }

    #[test]
    fn url_document_id_preserves_root_slash() {
        // `https://foo.com/` is the canonical root form that URL parsers
        // produce from `https://foo.com`. We leave it intact — there's no
        // shorter non-empty path to collapse it to.
        let with_slash =
            source_document_id_for_url("https://foo.com/", 0, never_exists).unwrap();
        let implicit = source_document_id_for_url("https://foo.com", 0, never_exists).unwrap();

        // Both inputs normalize to the same root form, so the IDs still match.
        assert_eq!(with_slash, implicit);

        // And the stable location keeps the trailing slash rather than
        // degenerating to `https://foo.com` with no path at all.
        let loc = normalize_url_stable_location("https://foo.com/").unwrap();
        assert_eq!(loc, "https://foo.com/");
    }

    #[test]
    fn url_document_id_handles_mixed_case_and_trailing_slash_together() {
        // Exercises both normalization paths in one go, as called out in
        // bn-nnd. Host lowercasing is already covered by the `url` crate and
        // is asserted here for completeness.
        let messy =
            source_document_id_for_url("HTTPS://FOO.COM/bar/", 0, never_exists).unwrap();
        let clean = source_document_id_for_url("https://foo.com/bar", 0, never_exists).unwrap();

        assert_eq!(messy, clean);
    }

    #[test]
    fn url_document_id_collapses_repeated_trailing_slashes() {
        // Defensive: if a user hand-types `//` at the end, we still collapse
        // to a single canonical form rather than minting a distinct src-id.
        let triple = normalize_url_stable_location("https://foo.com/bar///").unwrap();
        assert_eq!(triple, "https://foo.com/bar");
    }
}
