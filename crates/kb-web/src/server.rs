//! Axum router + handlers for the kb web UI.

use std::net::SocketAddr;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::Router;
use axum::extract::{Path as AxumPath, Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use kb_query::{
    HybridOptions, HybridResult, LexicalIndex, SemanticBackend, SemanticBackendConfig,
    SemanticBackendKind,
};
use serde::Deserialize;

use crate::markdown;

/// Shared state handed to every handler.
///
/// Cheap to clone (`Arc`-backed) so axum can hand each handler its own copy.
/// The lexical index is loaded once at server start; the router does not
/// hot-reload it — a `kb compile` run that happens while the server is up
/// will not be reflected until the server is restarted. That is fine for
/// v1 (punt-scope, local tool) and avoids having to synchronize writers.
#[derive(Clone)]
pub struct WebState {
    inner: Arc<WebStateInner>,
}

struct WebStateInner {
    root: PathBuf,
    index: LexicalIndex,
    /// Embedding backend used to embed inbound search queries. Loaded once
    /// at server start so the ONNX session (when `MiniLM` is configured) is
    /// reused across requests rather than rebuilt per query. bn-1rww.
    backend: SemanticBackend,
}

impl WebState {
    /// Build a state from a kb root path using the always-available hash
    /// embedding backend.
    ///
    /// Pins to [`SemanticBackendKind::Hash`] regardless of platform default
    /// so callers (and tests) that haven't selected a specific backend can
    /// construct a `WebState` even when the binary wasn't compiled with the
    /// `semantic-ort` feature. Production paths should use
    /// [`Self::with_backend_config`] with a config derived from `kb.toml`.
    ///
    /// # Errors
    ///
    /// Returns an error if the lexical index at `<root>/state/indexes/lexical.json`
    /// exists but cannot be parsed. A missing index is not an error — search
    /// simply returns no results until `kb compile` is run.
    pub fn new(root: PathBuf) -> Result<Self> {
        Self::with_backend_config(
            root,
            &SemanticBackendConfig {
                kind: SemanticBackendKind::Hash,
                ..SemanticBackendConfig::default()
            },
        )
    }

    /// Build a state with an explicit embedding backend config. The kb-cli
    /// `Command::Serve` path uses this so `kb.toml [semantic]` selections
    /// are honored — otherwise queries embedded with hash-embed against
    /// MiniLM-stored vectors would produce nonsense rankings.
    ///
    /// # Errors
    ///
    /// Returns an error if the lexical index cannot be parsed or the
    /// configured backend fails to load.
    pub fn with_backend_config(root: PathBuf, backend: &SemanticBackendConfig) -> Result<Self> {
        let index = LexicalIndex::load(&root)
            .with_context(|| format!("load lexical index at {}", root.display()))?;
        let backend = SemanticBackend::from_config(backend)
            .context("load semantic backend for kb-web")?;
        Ok(Self {
            inner: Arc::new(WebStateInner {
                root,
                index,
                backend,
            }),
        })
    }

    fn root(&self) -> &Path {
        &self.inner.root
    }

    fn index(&self) -> &LexicalIndex {
        &self.inner.index
    }

    fn backend(&self) -> &SemanticBackend {
        &self.inner.backend
    }
}

/// Build the axum router with all kb-web routes wired up.
pub fn router(state: WebState) -> Router {
    Router::new()
        .route("/", get(index_handler))
        .route("/wiki/{*path}", get(wiki_handler))
        .route("/search", get(search_handler))
        .route("/ask", post(ask_handler).get(ask_get_handler))
        .route("/static/style.css", get(style_handler))
        .with_state(state)
}

/// Bind an HTTP listener on `host:port` and serve the router.
///
/// # Errors
///
/// Returns an error if the listener cannot bind, or if the server exits
/// abnormally.
pub async fn serve(host: &str, port: u16, state: WebState) -> Result<()> {
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .with_context(|| format!("parse bind address {host}:{port}"))?;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    tracing::info!("kb-web listening on http://{addr}");
    eprintln!("kb serve: listening on http://{addr}/");
    eprintln!("  (Ctrl-C to stop)");
    axum::serve(listener, router(state))
        .await
        .context("axum server crashed")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn index_handler(State(state): State<WebState>) -> Response {
    // `wiki/index.md` is produced by `kb compile`; if absent, fall back to a
    // small greeting page so the server still serves something useful.
    let index_path = state.root().join("wiki/index.md");
    let body = if index_path.exists() {
        match std::fs::read_to_string(&index_path) {
            Ok(md) => markdown::render(&md),
            Err(e) => format!(
                "<p><em>failed to read wiki/index.md:</em> {}</p>",
                escape_html(&e.to_string())
            ),
        }
    } else {
        "<p><em>No <code>wiki/index.md</code> yet.</em> \
         Run <code>kb compile</code> to generate one.</p>"
            .to_string()
    };
    Html(shell("kb", &body, "/")).into_response()
}

async fn wiki_handler(
    State(state): State<WebState>,
    AxumPath(rel): AxumPath<String>,
) -> Response {
    // Only files under `<root>/wiki/` are served. We sanitize by refusing
    // any path component that is `..` or absolute, then verify the final
    // resolved path still starts with `<root>/wiki/`.
    let rel_path = PathBuf::from(&rel);
    if !is_safe_relative(&rel_path) {
        return (StatusCode::BAD_REQUEST, "invalid path").into_response();
    }

    // Accept both `/wiki/foo` and `/wiki/foo.md`. If the request omits the
    // `.md` suffix, try appending it so links like `[foo](foo)` resolve.
    let wiki_root = state.root().join("wiki");
    let direct = wiki_root.join(&rel_path);
    let with_md = if rel_path.extension().is_none() {
        Some(wiki_root.join(format!("{rel}.md")))
    } else {
        None
    };

    let target = if direct.is_file() {
        direct
    } else if let Some(p) = with_md.filter(|p| p.is_file()) {
        p
    } else {
        return (
            StatusCode::NOT_FOUND,
            Html(shell(
                "Not found",
                &format!("<p>No wiki page at <code>{}</code>.</p>", escape_html(&rel)),
                "/",
            )),
        )
            .into_response();
    };

    // Defense-in-depth: even if `is_safe_relative` slipped, the canonical
    // parent must still be under wiki_root.
    if let Ok(canon) = target.canonicalize()
        && let Ok(wiki_canon) = wiki_root.canonicalize()
        && !canon.starts_with(&wiki_canon)
    {
        return (StatusCode::BAD_REQUEST, "path escapes wiki root").into_response();
    }

    match std::fs::read_to_string(&target) {
        Ok(md) => {
            let title = rel_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("wiki");
            let body = markdown::render(&md);
            Html(shell(title, &body, "/")).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("read {}: {}", target.display(), e),
        )
            .into_response(),
    }
}

#[derive(Debug, Deserialize)]
struct SearchQuery {
    q: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
}

/// `GET /search?q=...&limit=10`
///
/// Returns JSON. A missing or whitespace-only `q` returns `{ "results": [] }`
/// with status 200 — the UI uses this to clear stale results.
async fn search_handler(
    State(state): State<WebState>,
    Query(q): Query<SearchQuery>,
) -> Response {
    let query = q.q.unwrap_or_default();
    if query.trim().is_empty() || kb_query::query_reduced_to_stopwords(&query) {
        return axum::Json(SearchResponse { results: Vec::new() }).into_response();
    }
    let limit = q.limit.unwrap_or(10).clamp(1, 100);
    let results = match kb_query::hybrid_search_with_index_and_backend(
        state.root(),
        state.index(),
        &query,
        limit,
        HybridOptions::for_backend(state.backend().kind()),
        state.backend(),
    ) {
        Ok(hits) => hits,
        Err(err) => {
            tracing::warn!("hybrid search failed, falling back to empty: {err:#}");
            Vec::new()
        }
    };
    axum::Json(SearchResponse { results }).into_response()
}

#[derive(serde::Serialize)]
struct SearchResponse {
    results: Vec<HybridResult>,
}

#[derive(Debug, Deserialize)]
struct AskForm {
    q: Option<String>,
}

/// `POST /ask` — accepts a form or JSON body with `q=<question>`. Shells
/// out to `kb ask --json --no-render` against the current root and returns
/// the answer body.
///
/// Shelling out keeps this crate decoupled from the full kb-cli config
/// machinery; the CLI already handles root discovery, LLM adapter setup,
/// artifact writing, and citation validation. The web handler just surfaces
/// what the CLI produced.
async fn ask_handler(State(state): State<WebState>, body: String) -> Response {
    // Accept either `q=...` form bodies or a raw question string (for `curl
    // -d "what is rust"`). JSON is also fine — clients that post
    // `{"q": "..."}` end up here via `application/json`, and we parse
    // defensively below.
    let question = parse_ask_body(&body);
    run_ask_and_respond(state.root(), &question).await
}

async fn ask_get_handler(
    State(state): State<WebState>,
    Query(q): Query<AskForm>,
) -> Response {
    let question = q.q.unwrap_or_default();
    run_ask_and_respond(state.root(), &question).await
}

fn parse_ask_body(body: &str) -> String {
    // Try JSON first.
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(body)
        && let Some(q) = v.get("q").and_then(|s| s.as_str())
    {
        return q.to_string();
    }
    // Form-encoded: `q=hello+world&other=ignored`.
    for pair in body.split('&') {
        if let Some(val) = pair.strip_prefix("q=") {
            return urldecode(val);
        }
    }
    // Plain text body: treat the whole thing as the question.
    body.trim().to_string()
}

fn urldecode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hi = hex_digit(bytes[i + 1]);
                let lo = hex_digit(bytes[i + 2]);
                if let (Some(hi), Some(lo)) = (hi, lo) {
                    out.push((hi << 4) | lo);
                    i += 3;
                } else {
                    out.push(bytes[i]);
                    i += 1;
                }
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8(out).unwrap_or_default()
}

const fn hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[derive(serde::Serialize)]
struct AskResponse {
    question: String,
    answer: String,
    artifact_path: Option<String>,
    error: Option<String>,
}

async fn run_ask_and_respond(root: &Path, question: &str) -> Response {
    let question = question.trim();
    if question.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            axum::Json(AskResponse {
                question: String::new(),
                answer: String::new(),
                artifact_path: None,
                error: Some("question cannot be empty".to_string()),
            }),
        )
            .into_response();
    }

    let root = root.to_path_buf();
    let question_owned = question.to_string();
    // tokio::process would be nicer but this keeps the blocking I/O off the
    // reactor thread via `spawn_blocking`.
    let question_for_closure = question_owned.clone();
    let join = tokio::task::spawn_blocking(move || {
        std::process::Command::new(kb_binary())
            .arg("--root")
            .arg(&root)
            .arg("--json")
            .arg("ask")
            .arg("--no-render")
            .arg(&question_for_closure)
            .output()
    })
    .await;

    let output = match join {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => {
            return axum::Json(AskResponse {
                question: question_owned,
                answer: String::new(),
                artifact_path: None,
                error: Some(format!("failed to spawn kb ask: {e}")),
            })
            .into_response();
        }
        Err(e) => {
            return axum::Json(AskResponse {
                question: question_owned,
                answer: String::new(),
                artifact_path: None,
                error: Some(format!("kb ask task panicked: {e}")),
            })
            .into_response();
        }
    };

    if !output.status.success() {
        return axum::Json(AskResponse {
            question: question_owned,
            answer: String::new(),
            artifact_path: None,
            error: Some(format!(
                "kb ask exited {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim(),
            )),
        })
        .into_response();
    }

    // The CLI wraps responses in `JsonEnvelope` (schema_version=1). We pull
    // out `data.artifact_path` and read the file body.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let envelope: serde_json::Value = match serde_json::from_str(&stdout) {
        Ok(v) => v,
        Err(e) => {
            return axum::Json(AskResponse {
                question: question_owned,
                answer: String::new(),
                artifact_path: None,
                error: Some(format!("kb ask JSON parse failed: {e}")),
            })
            .into_response();
        }
    };
    let artifact_path = envelope
        .pointer("/data/artifact_path")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let answer_body = artifact_path
        .as_deref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .unwrap_or_default();
    // Artifact files written by `kb ask` open with a YAML frontmatter block
    // (`---\nid: art-…\ngenerated_at: …\n---\n\n<body>`). The web UI surfaces
    // the body only — the metadata belongs in `artifact_path`, not the
    // rendered answer. Reuse the same stripper that the /wiki/* path uses
    // so the two surfaces stay in sync.
    let answer_body = markdown::strip_frontmatter(&answer_body).to_string();

    axum::Json(AskResponse {
        question: question_owned,
        answer: answer_body,
        artifact_path,
        error: None,
    })
    .into_response()
}

/// Resolve the `kb` binary path. Honors `KB_BIN` for tests, falls back to
/// `kb` in `$PATH` for production.
fn kb_binary() -> PathBuf {
    if let Ok(p) = std::env::var("KB_BIN") {
        return PathBuf::from(p);
    }
    PathBuf::from("kb")
}

async fn style_handler() -> Response {
    ([("content-type", "text/css; charset=utf-8")], STYLE_CSS).into_response()
}

// ---------------------------------------------------------------------------
// Template + helpers
// ---------------------------------------------------------------------------

fn shell(title: &str, body_html: &str, home: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — kb</title>
<link rel="stylesheet" href="/static/style.css">
</head>
<body>
<header>
  <a href="{home}" class="brand">kb</a>
  <form action="/search" method="get" class="searchbar">
    <input type="search" name="q" placeholder="search wiki…" autocomplete="off">
  </form>
</header>
<main>
<section class="ask">
  <form id="ask-form">
    <input type="text" id="ask-q" placeholder="ask a question…">
    <button type="submit">ask</button>
  </form>
  <div id="ask-result"></div>
</section>
<article>
{body}
</article>
</main>
<script>
document.getElementById('ask-form').addEventListener('submit', async (e) => {{
  e.preventDefault();
  const q = document.getElementById('ask-q').value.trim();
  if (!q) return;
  const out = document.getElementById('ask-result');
  out.textContent = 'thinking…';
  try {{
    const res = await fetch('/ask', {{
      method: 'POST',
      headers: {{ 'content-type': 'application/x-www-form-urlencoded' }},
      body: 'q=' + encodeURIComponent(q),
    }});
    const j = await res.json();
    if (j.error) {{
      out.textContent = 'error: ' + j.error;
    }} else {{
      out.innerHTML = '<pre>' + escapeHtml(j.answer || '(no answer)') + '</pre>';
    }}
  }} catch (err) {{
    out.textContent = 'network error: ' + err;
  }}
}});
function escapeHtml(s) {{
  return s.replace(/[&<>"']/g, c => ({{
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }})[c]);
}}
</script>
</body>
</html>
"#,
        title = escape_html(title),
        home = escape_html(home),
        body = body_html,
    )
}

fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// Reject paths that contain `..`, absolute roots, or host/prefix components.
/// We only want simple descendants of `<root>/wiki/`.
fn is_safe_relative(p: &Path) -> bool {
    for c in p.components() {
        match c {
            Component::Normal(_) => {}
            _ => return false,
        }
    }
    true
}

const STYLE_CSS: &str = include_str!("../assets/style.css");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ask_body_form() {
        assert_eq!(parse_ask_body("q=what+is+rust"), "what is rust");
        assert_eq!(parse_ask_body("q=hello%20world&other=x"), "hello world");
    }

    #[test]
    fn parse_ask_body_json() {
        assert_eq!(parse_ask_body(r#"{"q":"hi there"}"#), "hi there");
    }

    #[test]
    fn parse_ask_body_plain() {
        assert_eq!(parse_ask_body("what is rust"), "what is rust");
    }

    #[test]
    fn is_safe_relative_rejects_parent() {
        assert!(!is_safe_relative(Path::new("../etc/passwd")));
        assert!(!is_safe_relative(Path::new("/etc/passwd")));
        assert!(!is_safe_relative(Path::new("foo/../bar")));
    }

    #[test]
    fn is_safe_relative_accepts_normal() {
        assert!(is_safe_relative(Path::new("foo/bar.md")));
        assert!(is_safe_relative(Path::new("concepts/rust.md")));
    }

    #[test]
    fn escape_html_basic() {
        assert_eq!(escape_html("<a>"), "&lt;a&gt;");
        assert_eq!(escape_html("\"'&"), "&quot;&#39;&amp;");
    }
}
