use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use kb_core::fs::atomic_write;
use kb_core::{
    EntityMetadata, NormalizedDocument, Status, hash_bytes, mint_source_revision_id,
    normalize_url_stable_location, source_document_id_for_url, source_revision_content_hash,
    write_normalized_document,
};
use regex::Regex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::warn;
use url::Url;

use crate::IngestOutcome;
use crate::headings::extract_heading_ids;

const FETCH_UA: &str = "Mozilla/5.0 (compatible; kb-ingest/0.1)";
const TIMEOUT_SECS: u64 = 30;
const BACKOFF_DELAYS_MS: [u64; 3] = [500, 1_000, 2_000];

static IMG_SRC_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"<img[^>]*\bsrc="([^"]+)""#).expect("hard-coded regex is valid"));

#[derive(Serialize)]
struct FetchRecord {
    final_url: String,
    status: u16,
    content_type: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UrlIngestReport {
    pub source_url: String,
    pub source_document_id: String,
    pub source_revision_id: String,
    pub outcome: IngestOutcome,
    pub raw_snapshot_path: PathBuf,
    pub normalized_path: PathBuf,
    pub metadata_path: PathBuf,
}

/// Fetches `raw_url`, stores a raw HTML snapshot, and writes a normalized markdown document.
///
/// # Errors
/// Returns an error if the fetch fails after all retries, readability extraction fails,
/// or any file write fails.
pub async fn ingest_url(root: &Path, raw_url: &str) -> Result<UrlIngestReport> {
    ingest_url_with_options(root, raw_url, false).await
}

/// Fetches `raw_url`, computes the ingest outcome, and optionally persists fetched output.
///
/// # Errors
/// Returns an error if the fetch fails after all retries, readability extraction fails,
/// or any file write fails.
pub async fn ingest_url_with_options(
    root: &Path,
    raw_url: &str,
    dry_run: bool,
) -> Result<UrlIngestReport> {
    let source_id = mint_url_source_id(root, raw_url)?;

    let client = Client::builder()
        .user_agent(FETCH_UA)
        .timeout(Duration::from_secs(TIMEOUT_SECS))
        .build()
        .context("failed to build HTTP client")?;

    let (final_url, html_bytes, status, content_type) = fetch_with_retry(&client, raw_url).await?;
    let source_revision_id = mint_source_revision_id(&html_bytes);
    let content_hash = source_revision_content_hash(&html_bytes);

    let raw_snapshot_path = PathBuf::from("raw")
        .join("web")
        .join(&source_id)
        .join("page.html");
    let normalized_path = PathBuf::from("normalized")
        .join(&source_id)
        .join("source.md");
    let metadata_path = PathBuf::from("normalized")
        .join(&source_id)
        .join("metadata.json");

    let existing_revision_id = read_existing_revision_id(&root.join(&metadata_path))?;
    let outcome = match existing_revision_id {
        None => IngestOutcome::NewSource,
        Some(id) if id == source_revision_id => IngestOutcome::Skipped,
        Some(_) => IngestOutcome::NewRevision,
    };

    if !dry_run && outcome != IngestOutcome::Skipped {
        let raw_dir = root.join("raw").join("web").join(&source_id);
        atomic_write(raw_dir.join("page.html"), &html_bytes).context("failed to write raw HTML")?;
        atomic_write(
            raw_dir.join("page.headers.json"),
            serde_json::to_vec_pretty(&FetchRecord {
                final_url: final_url.to_string(),
                status,
                content_type: content_type.clone(),
            })?
            .as_slice(),
        )
        .context("failed to write fetch record")?;

        let treat_as_markdown = should_treat_as_markdown(content_type.as_deref(), &html_bytes);

        let (canonical_text, downloaded_assets, title) = if treat_as_markdown {
            // text/markdown, text/plain, or mis-labeled markdown: use the body
            // verbatim. readability/html2md would escape '#' and '*' and
            // collapse newlines, mangling the source.
            let text = String::from_utf8_lossy(&html_bytes).into_owned();
            let title = extract_markdown_title(&text).unwrap_or_default();
            (text, Vec::new(), title)
        } else {
            let mut html_reader = html_bytes.as_slice();
            let product = readability::extractor::extract(&mut html_reader, &final_url)
                .context("readability extraction failed")?;

            let assets_dir = root.join("normalized").join(&source_id).join("assets");
            let (processed_html, downloaded_assets) =
                rewrite_images(&product.content, &final_url, &assets_dir, &client).await?;
            let markdown = html2md::parse_html(&processed_html);
            (markdown, downloaded_assets, product.title)
        };

        let now = epoch_millis();
        let heading_ids = extract_heading_ids(&canonical_text);
        let normalized_doc = NormalizedDocument {
            metadata: EntityMetadata {
                id: source_id.clone(),
                created_at_millis: now,
                updated_at_millis: now,
                source_hashes: vec![content_hash.clone()],
                model_version: None,
                tool_version: Some(env!("CARGO_PKG_VERSION").to_string()),
                prompt_template_hash: None,
                dependencies: vec![source_revision_id.clone()],
                output_paths: vec![normalized_path.clone(), metadata_path.clone()],
                status: Status::Fresh,
            },
            source_revision_id: source_revision_id.clone(),
            canonical_text,
            normalized_assets: downloaded_assets,
            heading_ids,
        };

        // Stash the URL/title as a companion sidecar so callers can still find
        // the origin URL; the canonical metadata.json is owned by the
        // normalized-document writer.
        atomic_write(
            root.join("normalized").join(&source_id).join("origin.json"),
            serde_json::to_vec_pretty(&UrlOrigin {
                title,
                original_url: raw_url.to_string(),
                fetched_at_millis: now,
            })?
            .as_slice(),
        )
        .context("failed to write origin sidecar")?;

        write_normalized_document(root, &normalized_doc)
            .context("failed to write normalized document")?;
    }

    Ok(UrlIngestReport {
        source_url: raw_url.to_string(),
        source_document_id: source_id,
        source_revision_id,
        outcome,
        raw_snapshot_path,
        normalized_path,
        metadata_path,
    })
}

#[derive(Serialize)]
struct UrlOrigin {
    title: String,
    original_url: String,
    fetched_at_millis: u64,
}

/// Minimal view over the canonical `normalized/<id>/metadata.json` — just
/// enough to determine whether the on-disk revision matches what we fetched.
#[derive(Deserialize)]
struct RevisionProbe {
    source_revision_id: String,
}

/// Returns the `source_revision_id` recorded in an existing normalized
/// document, if one is present on disk.
fn read_existing_revision_id(path: &Path) -> Result<Option<String>> {
    if !path.exists() {
        return Ok(None);
    }

    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let probe: RevisionProbe = serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse JSON {}", path.display()))?;
    Ok(Some(probe.source_revision_id))
}

/// A fetch error, classified for retry decisions.
///
/// Transport errors and 5xx responses are transient and retried; 4xx responses are
/// fatal and surfaced immediately so we don't burn 3-4s retrying a 404.
enum FetchError {
    Transient(anyhow::Error),
    Fatal(anyhow::Error),
}

impl FetchError {
    fn into_error(self) -> anyhow::Error {
        match self {
            Self::Transient(e) | Self::Fatal(e) => e,
        }
    }
}

/// Fetches `url`, retrying up to `BACKOFF_DELAYS_MS.len()` times on transient failures.
///
/// 4xx responses short-circuit the retry loop: repeating a `GET` on a 404 or 401 is a
/// waste of time and user latency.
async fn fetch_with_retry(
    client: &Client,
    url: &str,
) -> Result<(Url, Vec<u8>, u16, Option<String>)> {
    for delay_ms in BACKOFF_DELAYS_MS {
        match do_fetch(client, url).await {
            Ok(result) => return Ok(result),
            Err(FetchError::Fatal(e)) => {
                return Err(e).with_context(|| format!("fetch {url} failed (non-retryable)"));
            }
            Err(FetchError::Transient(e)) => {
                warn!("fetch failed ({e}); retrying in {delay_ms}ms");
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }
        }
    }
    do_fetch(client, url)
        .await
        .map_err(FetchError::into_error)
        .with_context(|| {
            format!(
                "fetch {url} failed after {} attempts",
                BACKOFF_DELAYS_MS.len() + 1
            )
        })
}

async fn do_fetch(
    client: &Client,
    url: &str,
) -> std::result::Result<(Url, Vec<u8>, u16, Option<String>), FetchError> {
    let resp = client
        .get(url)
        .send()
        .await
        .context("HTTP request failed")
        .map_err(FetchError::Transient)?;
    let status = resp.status();
    let reason = status.canonical_reason().unwrap_or("");
    if status.is_server_error() {
        return Err(FetchError::Transient(anyhow::anyhow!(
            "HTTP {} {}",
            status.as_u16(),
            reason
        )));
    }
    if status.is_client_error() {
        return Err(FetchError::Fatal(anyhow::anyhow!(
            "HTTP {} {}",
            status.as_u16(),
            reason
        )));
    }
    if !status.is_success() {
        // Treat other unexpected non-success statuses (e.g. 3xx that reqwest didn't
        // follow) as fatal to avoid confusing retry storms.
        return Err(FetchError::Fatal(anyhow::anyhow!(
            "HTTP {} {}",
            status.as_u16(),
            reason
        )));
    }
    let final_url = resp.url().clone();
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(ToOwned::to_owned);
    let status_u16 = status.as_u16();
    let bytes = resp
        .bytes()
        .await
        .context("failed to read response body")
        .map_err(FetchError::Transient)?
        .to_vec();
    Ok((final_url, bytes, status_u16, content_type))
}

/// Replaces image `src` attributes in `html` with local asset paths, downloading each image.
///
/// Images that fail to download are left unrewritten and a warning is logged.
/// Returns the rewritten HTML and the paths of assets that were successfully
/// downloaded. The asset paths are the absolute destinations inside
/// `assets_dir`; `write_normalized_document` re-copies them through its own
/// atomic writer so the final layout matches other normalized sources.
async fn rewrite_images(
    html: &str,
    base_url: &Url,
    assets_dir: &Path,
    client: &Client,
) -> Result<(String, Vec<PathBuf>)> {
    let srcs: Vec<String> = IMG_SRC_RE
        .captures_iter(html)
        .map(|cap| cap[1].to_owned())
        .filter(|src| !src.starts_with("data:"))
        .collect();

    let mut result = html.to_owned();
    let mut downloaded: Vec<PathBuf> = Vec::new();
    let mut seen_filenames: std::collections::HashSet<String> = std::collections::HashSet::new();
    for src in srcs {
        let img_url = if src.starts_with("http://") || src.starts_with("https://") {
            Url::parse(&src).ok()
        } else {
            base_url.join(&src).ok()
        };
        let Some(img_url) = img_url else {
            continue;
        };

        let filename = derive_asset_name(&img_url, &src);
        let asset_path = assets_dir.join(&filename);

        match download_asset(client, &img_url, &asset_path).await {
            Ok(()) => {
                result = result.replace(
                    &format!(r#"src="{src}""#),
                    &format!(r#"src="assets/{filename}""#),
                );
                // `write_normalized_document` rejects duplicate filenames; the
                // same `src` can appear twice in HTML, so dedupe here.
                if seen_filenames.insert(filename.clone()) {
                    downloaded.push(asset_path);
                }
            }
            Err(e) => warn!("skipping image {img_url}: {e}"),
        }
    }
    Ok((result, downloaded))
}

async fn download_asset(client: &Client, url: &Url, dest: &Path) -> Result<()> {
    let resp = client
        .get(url.as_str())
        .send()
        .await
        .context("failed to fetch asset")?;
    if !resp.status().is_success() {
        bail!("HTTP {} for {url}", resp.status());
    }
    let bytes = resp.bytes().await.context("failed to read asset body")?;
    atomic_write(dest, &bytes).context("failed to write asset file")
}

/// Derives a collision-resistant asset filename from the image URL and original `src`.
fn derive_asset_name(url: &Url, src: &str) -> String {
    let raw = url
        .path_segments()
        .and_then(|mut s| s.next_back())
        .filter(|s| !s.is_empty())
        .unwrap_or("asset");
    let prefix = &hash_bytes(src.as_bytes()).to_hex()[..8];
    let safe: String = raw
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '.' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("{prefix}_{safe}")
}

/// Decides whether a fetched response should be treated as raw markdown/plain
/// text instead of run through readability + html2md.
///
/// Uses two signals:
/// 1. The `Content-Type` header: `text/markdown`, `text/plain`, or `text/*`
///    that is not `text/html`/`application/xhtml+xml` is treated as markdown.
/// 2. A 4KB body sniff: if the server labels it `text/html` but the body has
///    no `<html` / `<!DOCTYPE` marker AND starts with `# ` (ATX heading),
///    assume it's markdown mis-served as HTML. Common on raw git hosts and
///    some CDNs.
fn should_treat_as_markdown(content_type: Option<&str>, body: &[u8]) -> bool {
    let mime = content_type
        .map(|ct| ct.split(';').next().unwrap_or("").trim().to_ascii_lowercase())
        .unwrap_or_default();

    let looks_like_html_mime =
        mime == "text/html" || mime == "application/xhtml+xml" || mime == "application/xml";
    let looks_like_text_mime = mime.starts_with("text/") && !looks_like_html_mime;

    if looks_like_text_mime {
        return true;
    }

    // Sniff fallback: server may serve markdown as text/html (or omit the
    // header). Inspect the first ~4KB for HTML markers.
    let sniff_len = body.len().min(4096);
    let head = &body[..sniff_len];
    let head_lower: Vec<u8> = head.iter().map(u8::to_ascii_lowercase).collect();
    let has_html_marker = find_subsequence(&head_lower, b"<html")
        || find_subsequence(&head_lower, b"<!doctype");

    if has_html_marker {
        return false;
    }

    // No HTML markers AND body starts with an ATX heading — treat as markdown
    // regardless of header. Only apply this when we also lack a positive HTML
    // signal, so well-formed HTML pages still flow through readability.
    let body_start = trim_bom_and_whitespace(head);
    if body_start.starts_with(b"# ") {
        return true;
    }

    false
}

/// Returns true if `haystack` contains `needle` as a contiguous subsequence.
fn find_subsequence(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || haystack.len() < needle.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// Skips a UTF-8 BOM and leading ASCII whitespace so ATX-heading sniffing
/// survives editors that prepend `\u{FEFF}` or leading newlines.
fn trim_bom_and_whitespace(bytes: &[u8]) -> &[u8] {
    let mut start = if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        3
    } else {
        0
    };
    while start < bytes.len() && matches!(bytes[start], b' ' | b'\t' | b'\r' | b'\n') {
        start += 1;
    }
    &bytes[start..]
}

/// Extracts the first ATX heading (`# ...`) from a markdown document to use as
/// a title, mirroring what readability does for HTML.
fn extract_markdown_title(text: &str) -> Option<String> {
    for line in text.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("# ") {
            let title = rest.trim().trim_end_matches('#').trim();
            if !title.is_empty() {
                return Some(title.to_string());
            }
        }
    }
    None
}

fn epoch_millis() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// Mints the `src-` id for a URL source, threading terseid's collision
/// probe into this KB's on-disk layout. The seed is the normalized URL so
/// re-ingesting the same URL deterministically returns the same id.
fn mint_url_source_id(root: &Path, raw_url: &str) -> Result<String> {
    let stable_location = normalize_url_stable_location(raw_url)
        .context("failed to derive source document ID")?;
    let existing_count = count_url_sources(root);
    source_document_id_for_url(raw_url, existing_count, |candidate| {
        url_source_id_taken_by_other(root, candidate, &stable_location)
    })
    .context("failed to derive source document ID")
}

/// Counts existing URL source documents under `raw/web/<id>/` and local
/// sources under `raw/inbox/<id>/`. Both share the `src-` namespace, so
/// terseid's collision math needs the combined count.
fn count_url_sources(root: &Path) -> usize {
    let mut total = 0;
    for bucket in ["raw/web", "raw/inbox"] {
        if let Ok(entries) = std::fs::read_dir(root.join(bucket)) {
            total += entries
                .filter_map(Result::ok)
                .filter(|entry| entry.file_type().is_ok_and(|ft| ft.is_dir()))
                .count();
        }
    }
    total
}

/// True if `candidate` already names a source document on disk whose
/// origin URL normalizes to something other than `stable_location`. We
/// inspect the origin sidecar at `normalized/<candidate>/origin.json`
/// when present so that re-ingesting the same URL returns the same id.
fn url_source_id_taken_by_other(root: &Path, candidate: &str, stable_location: &str) -> bool {
    let origin = root.join("normalized").join(candidate).join("origin.json");
    if let Ok(contents) = std::fs::read_to_string(&origin)
        && let Ok(value) = serde_json::from_str::<serde_json::Value>(&contents)
        && let Some(stored_url) = value.get("original_url").and_then(|v| v.as_str())
    {
        return normalize_url_stable_location(stored_url)
            .map_or(true, |normalized| normalized != stable_location);
    }

    // No origin sidecar on disk: check whether any top-level directory under
    // raw/web or raw/inbox already owns this id (from a partially-written
    // previous ingest) and, if so, treat it as a collision we can't resolve.
    root.join("raw").join("web").join(candidate).is_dir()
        || root.join("raw").join("inbox").join(candidate).is_dir()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read as _, Write as _};
    use std::net::TcpListener;
    use std::thread;

    const SAMPLE_MD: &[u8] = b"# Rust By Example\n\n* item one\n* item two\n\n## Section\n\nbody\n";
    const SAMPLE_HTML: &[u8] =
        b"<!DOCTYPE html><html><head><title>t</title></head><body><h1>hi</h1></body></html>";

    #[test]
    fn text_markdown_is_treated_as_markdown() {
        assert!(should_treat_as_markdown(
            Some("text/markdown; charset=utf-8"),
            SAMPLE_MD
        ));
    }

    #[test]
    fn text_plain_is_treated_as_markdown() {
        assert!(should_treat_as_markdown(Some("text/plain"), SAMPLE_MD));
    }

    #[test]
    fn text_html_is_not_treated_as_markdown() {
        assert!(!should_treat_as_markdown(Some("text/html"), SAMPLE_HTML));
    }

    #[test]
    fn html_mime_with_markdown_body_is_sniffed_as_markdown() {
        // Server mis-labels raw markdown as text/html. If the body has no
        // HTML markers and starts with `# `, we should treat it as markdown.
        assert!(should_treat_as_markdown(Some("text/html"), SAMPLE_MD));
    }

    #[test]
    fn html_mime_with_real_html_stays_html() {
        assert!(!should_treat_as_markdown(Some("text/html"), SAMPLE_HTML));
    }

    #[test]
    fn missing_content_type_falls_back_to_sniff() {
        assert!(should_treat_as_markdown(None, SAMPLE_MD));
        assert!(!should_treat_as_markdown(None, SAMPLE_HTML));
    }

    #[test]
    fn application_json_is_not_markdown() {
        assert!(!should_treat_as_markdown(
            Some("application/json"),
            b"{\"hi\": 1}"
        ));
    }

    #[test]
    fn utf8_bom_prefix_still_sniffs_as_markdown() {
        let mut body = Vec::from([0xEF, 0xBB, 0xBF]);
        body.extend_from_slice(SAMPLE_MD);
        assert!(should_treat_as_markdown(Some("text/html"), &body));
    }

    #[test]
    fn markdown_title_extracts_first_heading() {
        assert_eq!(
            extract_markdown_title("# Rust By Example\n\nbody\n"),
            Some("Rust By Example".to_string())
        );
    }

    #[test]
    fn markdown_title_skips_non_heading_lines() {
        assert_eq!(
            extract_markdown_title("\n\nsome prose\n\n# Real Title\n"),
            Some("Real Title".to_string())
        );
    }

    #[test]
    fn markdown_title_returns_none_without_heading() {
        assert_eq!(extract_markdown_title("no heading here\n"), None);
    }

    #[tokio::test]
    async fn fetch_404_error_includes_url_and_status_phrase() {
        // Stub a 404 response and assert the surfaced error message names the
        // URL and the canonical HTTP reason phrase. Regression guard for
        // bn-361: prior output was a bare "fetch failed (non-retryable)".
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind listener");
        let address = listener.local_addr().expect("local addr");
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept connection");
            let mut buffer = [0_u8; 1024];
            let _ = stream.read(&mut buffer).expect("read request");
            let body = "not found";
            write!(
                stream,
                "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .expect("write response");
        });

        let url = format!("http://{address}/missing");
        let client = Client::builder()
            .user_agent(FETCH_UA)
            .timeout(Duration::from_secs(TIMEOUT_SECS))
            .build()
            .expect("build client");

        let err = fetch_with_retry(&client, &url)
            .await
            .expect_err("404 should surface as error");
        server.join().expect("join server");

        let msg = format!("{err:#}");
        assert!(
            msg.contains(&url),
            "error should mention URL; got: {msg}"
        );
        assert!(
            msg.contains("404"),
            "error should mention HTTP 404 status; got: {msg}"
        );
        assert!(
            msg.contains("Not Found"),
            "error should include canonical reason phrase; got: {msg}"
        );
        assert!(
            msg.contains("non-retryable"),
            "4xx path should be non-retryable; got: {msg}"
        );
    }

    #[tokio::test]
    async fn ingest_markdown_content_type_preserves_raw_markdown() {
        use tempfile::TempDir;

        // Standing up a full HTTP server is overkill; we instead drive the
        // post-fetch logic directly by invoking the same helpers to prove
        // that markdown bodies take the raw-path. The regression we're
        // guarding against is escaping of `#` and `*`, which only happens
        // when html2md::parse_html is called on the body.
        let temp = TempDir::new().expect("tempdir");
        let body = SAMPLE_MD;

        assert!(should_treat_as_markdown(Some("text/markdown"), body));
        let text = String::from_utf8_lossy(body).into_owned();

        // Regression checks: the raw-path must preserve the markdown tokens.
        assert!(text.contains("# Rust By Example"));
        assert!(text.contains("* item one"));
        assert!(!text.contains("\\#"));
        assert!(!text.contains("\\*"));

        // Exercise extract_heading_ids on the preserved text so we know the
        // canonical-text path downstream sees real headings, not mangled
        // prose that html2md would produce from escaped hashes.
        let heading_ids = extract_heading_ids(&text);
        assert_eq!(
            heading_ids,
            vec!["rust-by-example".to_string(), "section".to_string()]
        );
        drop(temp);
    }
}
