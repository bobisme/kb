use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use kb_core::fs::atomic_write;
use kb_core::{
    EntityMetadata, NormalizedDocument, Status, hash_bytes, mint_source_revision_id,
    source_document_id_for_url, source_revision_content_hash, write_normalized_document,
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
    let source_id =
        source_document_id_for_url(raw_url).context("failed to derive source document ID")?;

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
                content_type,
            })?
            .as_slice(),
        )
        .context("failed to write fetch record")?;

        let mut html_reader = html_bytes.as_slice();
        let product = readability::extractor::extract(&mut html_reader, &final_url)
            .context("readability extraction failed")?;

        let assets_dir = root.join("normalized").join(&source_id).join("assets");
        let (processed_html, downloaded_assets) =
            rewrite_images(&product.content, &final_url, &assets_dir, &client).await?;
        let markdown = html2md::parse_html(&processed_html);

        let now = epoch_millis();
        let heading_ids = extract_heading_ids(&markdown);
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
            canonical_text: markdown,
            normalized_assets: downloaded_assets,
            heading_ids,
        };

        // Stash the URL/title as a companion sidecar so callers can still find
        // the origin URL; the canonical metadata.json is owned by the
        // normalized-document writer.
        atomic_write(
            root.join("normalized").join(&source_id).join("origin.json"),
            serde_json::to_vec_pretty(&UrlOrigin {
                title: product.title,
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
                return Err(e).context("fetch failed (non-retryable)");
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
                "fetch failed after {} attempts",
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
    if status.is_server_error() {
        return Err(FetchError::Transient(anyhow::anyhow!(
            "server error {status}"
        )));
    }
    if status.is_client_error() {
        return Err(FetchError::Fatal(anyhow::anyhow!("HTTP {status}")));
    }
    if !status.is_success() {
        // Treat other unexpected non-success statuses (e.g. 3xx that reqwest didn't
        // follow) as fatal to avoid confusing retry storms.
        return Err(FetchError::Fatal(anyhow::anyhow!("HTTP {status}")));
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

fn epoch_millis() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}
