//! Integration tests for kb-web that drive the axum router via
//! `tower::ServiceExt::oneshot` — no real network, no real LLM.

use std::fs;
use std::path::Path;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use kb_web::{WebState, router};
use tempfile::TempDir;
use tower::ServiceExt;

fn write(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("mkdir");
    }
    fs::write(path, contents).expect("write");
}

/// Build a minimal kb root on disk that is just enough for the web handlers
/// to exercise `/`, `/wiki/*`, and `/search`.
fn fixture() -> TempDir {
    let tmp = TempDir::new().expect("tempdir");
    let root = tmp.path();

    write(&root.join("kb.toml"), "\n");
    write(
        &root.join("wiki/index.md"),
        "# Knowledge Base\n\nWelcome to kb.\n\n- [Rust](concepts/rust.md)\n",
    );
    write(
        &root.join("wiki/concepts/rust.md"),
        "# Rust\n\nRust is a systems programming language.\n",
    );
    // A page with YAML frontmatter — used by the regression test for the
    // bn-125l bug where `kb serve` was leaking frontmatter keys into the
    // rendered body.
    write(
        &root.join("wiki/concepts/aries.md"),
        "---\n\
         id: concept:aries\n\
         name: ARIES\n\
         aliases:\n  - recovery\n  - durability\n\
         ---\n\
         # ARIES\n\n\
         ARIES is a recovery algorithm.\n",
    );

    // Hand-build a lexical index so /search returns something deterministic
    // without needing to run `kb compile`.
    let idx = serde_json::json!({
        "entries": [
            {
                "id": "wiki/concepts/rust.md",
                "title": "Rust",
                "aliases": [],
                "headings": ["Rust"],
                "summary": "Rust is a systems programming language."
            }
        ]
    });
    write(
        &root.join("state/indexes/lexical.json"),
        &serde_json::to_string(&idx).expect("json"),
    );

    tmp
}

fn make_app(tmp: &TempDir) -> axum::Router {
    let state = WebState::new(tmp.path().to_path_buf()).expect("state");
    router(state)
}

async fn body_string(resp: axum::response::Response) -> String {
    let bytes = resp
        .into_body()
        .collect()
        .await
        .expect("collect body")
        .to_bytes();
    String::from_utf8(bytes.to_vec()).expect("utf8")
}

#[tokio::test]
async fn root_renders_wiki_index() {
    let tmp = fixture();
    let app = make_app(&tmp);

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/")
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("serve");

    assert_eq!(resp.status(), StatusCode::OK);
    let html = body_string(resp).await;
    assert!(html.contains("<title>kb — kb</title>"));
    assert!(html.contains("Knowledge Base"));
    assert!(html.contains("Welcome to kb"));
}

#[tokio::test]
async fn wiki_page_renders_markdown() {
    let tmp = fixture();
    let app = make_app(&tmp);

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/wiki/concepts/rust.md")
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("serve");

    assert_eq!(resp.status(), StatusCode::OK);
    let html = body_string(resp).await;
    assert!(html.contains("<h1>Rust</h1>"));
    assert!(html.contains("systems programming language"));
}

#[tokio::test]
async fn wiki_page_strips_yaml_frontmatter() {
    // Regression test for bn-125l: YAML frontmatter must not leak into the
    // rendered HTML body.
    let tmp = fixture();
    let app = make_app(&tmp);

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/wiki/concepts/aries.md")
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("serve");

    assert_eq!(resp.status(), StatusCode::OK);
    let html = body_string(resp).await;
    // The real body must render.
    assert!(html.contains("<h1>ARIES</h1>"));
    assert!(html.contains("recovery algorithm"));
    // But the frontmatter keys must not appear anywhere in the response.
    assert!(
        !html.contains("id: concept:aries"),
        "frontmatter id leaked into body: {html}"
    );
    assert!(
        !html.contains("name: ARIES"),
        "frontmatter name leaked into body: {html}"
    );
    assert!(
        !html.contains("aliases:"),
        "frontmatter aliases leaked into body: {html}"
    );
}

#[tokio::test]
async fn wiki_page_allows_omitting_md_suffix() {
    let tmp = fixture();
    let app = make_app(&tmp);

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/wiki/concepts/rust")
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("serve");

    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn wiki_rejects_path_traversal() {
    let tmp = fixture();
    let app = make_app(&tmp);

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/wiki/..%2F..%2Fkb.toml")
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("serve");

    // Axum decodes the path before routing, so `..` components show up and
    // our sanitizer rejects them with 400.
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn wiki_missing_page_is_404() {
    let tmp = fixture();
    let app = make_app(&tmp);

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/wiki/nope.md")
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("serve");

    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn search_returns_json_results() {
    let tmp = fixture();
    let app = make_app(&tmp);

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/search?q=rust")
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("serve");

    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_string(resp).await;
    let json: serde_json::Value = serde_json::from_str(&body).expect("json");
    let results = json
        .get("results")
        .and_then(|v| v.as_array())
        .expect("results array");
    assert!(!results.is_empty(), "expected at least one hit");
    assert_eq!(results[0]["id"], "wiki/concepts/rust.md");
}

#[tokio::test]
async fn search_empty_query_returns_empty_results() {
    let tmp = fixture();
    let app = make_app(&tmp);

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/search?q=")
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("serve");

    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_string(resp).await;
    assert!(body.contains("\"results\":[]"));
}

#[tokio::test]
async fn ask_rejects_empty_question() {
    let tmp = fixture();
    let app = make_app(&tmp);

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/ask")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("q="))
                .expect("req"),
        )
        .await
        .expect("serve");

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_string(resp).await;
    let json: serde_json::Value = serde_json::from_str(&body).expect("json");
    assert!(
        json["error"]
            .as_str()
            .unwrap_or("")
            .contains("cannot be empty")
    );
}

#[tokio::test]
async fn style_endpoint_serves_css() {
    let tmp = fixture();
    let app = make_app(&tmp);

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/static/style.css")
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("serve");

    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(ct.contains("text/css"));
}
