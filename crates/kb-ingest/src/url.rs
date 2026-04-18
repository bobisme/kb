use std::path::Path;
use std::sync::LazyLock;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use kb_core::fs::atomic_write;
use kb_core::{
    hash_bytes, mint_source_revision_id, source_document_id_for_url, source_revision_content_hash,
};
use regex::Regex;
use reqwest::Client;
use serde::Serialize;
use tracing::warn;
use url::Url;

const FETCH_UA: &str = "Mozilla/5.0 (compatible; kb-ingest/0.1)";
const TIMEOUT_SECS: u64 = 30;
const BACKOFF_DELAYS_MS: [u64; 3] = [500, 1_000, 2_000];

static IMG_SRC_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"<img[^>]*\bsrc="([^"]+)""#).expect("hard-coded regex is valid")
});

#[derive(Serialize)]
struct FetchRecord {
    final_url: String,
    status: u16,
    content_type: Option<String>,
}

#[derive(Serialize)]
struct NormalizedMetadata {
    title: String,
    original_url: String,
    fetched_at_millis: u64,
    content_hash: String,
    source_revision_id: String,
}

/// Fetches `raw_url`, stores a raw HTML snapshot, and writes a normalized markdown document.
///
/// Produces under `root`:
/// - `raw/web/<source-id>/page.html` — raw HTML bytes
/// - `raw/web/<source-id>/page.headers.json` — fetch metadata
/// - `normalized/<source-id>/source.md` — readability-extracted markdown
/// - `normalized/<source-id>/metadata.json` — structured provenance metadata
/// - `normalized/<source-id>/assets/<hash>_<name>` — downloaded images
///
/// # Errors
/// Returns an error if the fetch fails after all retries, readability extraction fails,
/// or any file write fails.
pub async fn ingest_url(root: &Path, raw_url: &str) -> Result<()> {
    let source_id = source_document_id_for_url(raw_url)
        .context("failed to derive source document ID")?;

    let client = Client::builder()
        .user_agent(FETCH_UA)
        .timeout(Duration::from_secs(TIMEOUT_SECS))
        .build()
        .context("failed to build HTTP client")?;

    let (final_url, html_bytes, status, content_type) =
        fetch_with_retry(&client, raw_url).await?;

    // Persist raw snapshot
    let raw_dir = root.join("raw").join("web").join(&source_id);
    atomic_write(raw_dir.join("page.html"), &html_bytes)
        .context("failed to write raw HTML")?;
    atomic_write(
        raw_dir.join("page.headers.json"),
        serde_json::to_vec_pretty(&FetchRecord { final_url: final_url.to_string(), status, content_type })?.as_slice(),
    )
    .context("failed to write fetch record")?;

    // Extract readable content (readability takes a sync reader over the already-fetched bytes)
    let mut html_reader = html_bytes.as_slice();
    let product = readability::extractor::extract(&mut html_reader, &final_url)
        .context("readability extraction failed")?;

    // Download images, rewrite src references, convert to markdown
    let assets_dir = root.join("normalized").join(&source_id).join("assets");
    let processed_html = rewrite_images(&product.content, &final_url, &assets_dir, &client).await?;
    let markdown = html2md::parse_html(&processed_html);

    // Persist normalized document
    let normalized_dir = root.join("normalized").join(&source_id);
    atomic_write(normalized_dir.join("source.md"), markdown.as_bytes())
        .context("failed to write normalized markdown")?;
    atomic_write(
        normalized_dir.join("metadata.json"),
        serde_json::to_vec_pretty(&NormalizedMetadata {
            title: product.title,
            original_url: raw_url.to_string(),
            fetched_at_millis: epoch_millis(),
            content_hash: source_revision_content_hash(&html_bytes),
            source_revision_id: mint_source_revision_id(&html_bytes),
        })?
        .as_slice(),
    )
    .context("failed to write metadata")?;

    Ok(())
}

/// Fetches `url`, retrying up to `BACKOFF_DELAYS_MS.len()` times on transient failures.
async fn fetch_with_retry(
    client: &Client,
    url: &str,
) -> Result<(Url, Vec<u8>, u16, Option<String>)> {
    for delay_ms in BACKOFF_DELAYS_MS {
        match do_fetch(client, url).await {
            Ok(result) => return Ok(result),
            Err(e) => {
                warn!("fetch failed ({e}); retrying in {delay_ms}ms");
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }
        }
    }
    do_fetch(client, url)
        .await
        .with_context(|| format!("fetch failed after {} attempts", BACKOFF_DELAYS_MS.len() + 1))
}

async fn do_fetch(client: &Client, url: &str) -> Result<(Url, Vec<u8>, u16, Option<String>)> {
    let resp = client.get(url).send().await.context("HTTP request failed")?;
    let status = resp.status();
    if status.is_server_error() {
        bail!("server error {status}");
    }
    if !status.is_success() {
        bail!("HTTP {status}");
    }
    let final_url = resp.url().clone();
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(ToOwned::to_owned);
    let status_u16 = status.as_u16();
    let bytes = resp.bytes().await.context("failed to read response body")?.to_vec();
    Ok((final_url, bytes, status_u16, content_type))
}

/// Replaces image `src` attributes in `html` with local asset paths, downloading each image.
///
/// Images that fail to download are left unrewritten and a warning is logged.
async fn rewrite_images(
    html: &str,
    base_url: &Url,
    assets_dir: &Path,
    client: &Client,
) -> Result<String> {
    let srcs: Vec<String> = IMG_SRC_RE
        .captures_iter(html)
        .map(|cap| cap[1].to_owned())
        .filter(|src| !src.starts_with("data:"))
        .collect();

    let mut result = html.to_owned();
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
            }
            Err(e) => warn!("skipping image {img_url}: {e}"),
        }
    }
    Ok(result)
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
        .map(|c| if c.is_alphanumeric() || c == '.' || c == '-' { c } else { '_' })
        .collect();
    format!("{prefix}_{safe}")
}

fn epoch_millis() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}
