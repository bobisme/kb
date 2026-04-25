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
        &kb_core::state_dir(root).join("indexes/lexical.json"),
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
    // bn-3qsj: `/search` now returns HybridResult, which uses `item_id`
    // (matching the lexical+semantic store) instead of the legacy `id`.
    assert_eq!(results[0]["item_id"], "wiki/concepts/rust.md");
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
async fn ask_strips_frontmatter_from_answer_field() {
    // Regression test for bn-1hyu: the /ask handler reads the artifact file
    // written by `kb ask` (which opens with a YAML frontmatter block) and
    // returns its contents as the `answer` JSON field. Frontmatter metadata
    // must be stripped before surfacing the body to the UI.
    let tmp = fixture();
    let root = tmp.path();

    // Seed an artifact that looks exactly like what `kb ask` would produce.
    let artifact_abs = root.join("outputs/questions/q-test/answer.md");
    let body = "# Memo: What is ARIES?\n\n\
         ARIES is a recovery algorithm for databases.\n";
    let with_fm = format!(
        "---\n\
         id: art-1yc\n\
         generated_at: 2026-04-20T00:00:00Z\n\
         source_document_ids:\n  - doc:rust\n\
         ---\n\n\
         {body}"
    );
    write(&artifact_abs, &with_fm);

    // Build a fake `kb` binary that ignores its args and prints the JSON
    // envelope the real CLI would produce. We write it as a shell script
    // and point `KB_BIN` at it for the duration of the test. The handler
    // reads `artifact_path` directly, so we hand it the absolute path.
    let fake_kb = tmp.path().join("fake-kb.sh");
    let artifact_str = artifact_abs.to_str().expect("utf-8 path");
    let envelope = serde_json::json!({
        "schema_version": 1,
        "data": { "artifact_path": artifact_str },
    });
    let script = format!(
        "#!/bin/sh\nprintf '%s' '{}'\n",
        serde_json::to_string(&envelope).expect("serialize envelope"),
    );
    fs::write(&fake_kb, script).expect("write fake kb");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perm = fs::metadata(&fake_kb).expect("meta").permissions();
        perm.set_mode(0o755);
        fs::set_permissions(&fake_kb, perm).expect("chmod");
    }

    // SAFETY: Rust 2024 requires `unsafe` for env mutation. No other test
    // in this crate reads or writes `KB_BIN`, so we accept the race risk
    // here. If more /ask tests are added, serialize them behind a Mutex.
    unsafe {
        std::env::set_var("KB_BIN", &fake_kb);
    }

    let app = make_app(&tmp);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/ask")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("q=what+is+aries"))
                .expect("req"),
        )
        .await
        .expect("serve");

    unsafe {
        std::env::remove_var("KB_BIN");
    }

    assert_eq!(resp.status(), StatusCode::OK);
    let body_str = body_string(resp).await;
    let json: serde_json::Value = serde_json::from_str(&body_str).expect("json");
    let answer = json["answer"].as_str().expect("answer is a string");

    // Frontmatter delimiters and keys must not leak through.
    assert!(
        !answer.starts_with("---"),
        "answer starts with frontmatter delimiter: {answer:?}"
    );
    assert!(
        !answer.contains("id: art-"),
        "answer contains frontmatter id: {answer:?}"
    );
    assert!(
        !answer.contains("generated_at:"),
        "answer contains generated_at: {answer:?}"
    );
    assert!(
        !answer.contains("source_document_ids:"),
        "answer contains source_document_ids: {answer:?}"
    );
    // But the real body is still there, unchanged.
    assert!(
        answer.contains("# Memo: What is ARIES?"),
        "answer missing heading: {answer:?}"
    );
    assert!(
        answer.contains("ARIES is a recovery algorithm"),
        "answer missing body text: {answer:?}"
    );

    // And the artifact_path is surfaced so the UI can link to the file.
    assert_eq!(json["artifact_path"].as_str(), Some(artifact_str));
    assert!(json["error"].is_null());
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
