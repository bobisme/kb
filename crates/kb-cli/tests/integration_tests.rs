mod common;

use common::{kb_cmd, make_temp_kb};
use kb_compile::Graph;
use regex::Regex;
use serde_json::Value;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::thread;

fn init_kb(root: &std::path::Path) {
    let mut cmd = kb_cmd(root);
    cmd.arg("init");
    let output = cmd.output().expect("failed to run kb init");
    assert!(
        output.status.success(),
        "kb init failed with output: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn normalize_temp_paths(output: &str) -> String {
    // Replace all temp paths with a placeholder to make snapshots deterministic
    Regex::new(r"/tmp/[.a-zA-Z0-9]+")
        .expect("regex is valid")
        .replace_all(output, "/tmp/tmpXXXXXX")
        .to_string()
}

#[test]
fn smoke_test_kb_init_creates_directory() {
    let (_temp_dir, kb_root) = make_temp_kb();

    // The kb root directory should be empty initially
    assert!(
        fs::read_dir(&kb_root)
            .expect("failed to read kb root directory")
            .next()
            .is_none()
    );
}

#[test]
fn smoke_test_kb_init_command() {
    let (_temp_dir, kb_root) = make_temp_kb();

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("init");

    // Run the init command - should succeed even though init is not fully implemented
    let output = cmd.output().expect("failed to run kb init");

    // Command should exit successfully
    assert!(
        output.status.success(),
        "kb init failed with output: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Output should indicate the command is not fully implemented yet
    let stdout = String::from_utf8_lossy(&output.stdout);
    let normalized = normalize_temp_paths(&stdout);
    insta::assert_snapshot!(normalized);
}

#[test]
fn smoke_test_kb_init_with_explicit_path() {
    let (_temp_dir, kb_root) = make_temp_kb();
    let init_path = kb_root.join("my_kb");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("init").arg(&init_path);

    let output = cmd.output().expect("failed to run kb init with path");

    // Command should exit successfully
    assert!(
        output.status.success(),
        "kb init with path failed with output: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let normalized = normalize_temp_paths(&stdout);
    insta::assert_snapshot!(normalized);
}

#[test]
fn init_creates_empty_state_files() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let manifest = fs::read_to_string(kb_root.join("state/manifest.json"))
        .expect("read manifest state file");
    let hashes =
        fs::read_to_string(kb_root.join("state/hashes.json")).expect("read hashes state file");

    let manifest_json: Value = serde_json::from_str(&manifest).expect("parse manifest json");
    let hashes_json: Value = serde_json::from_str(&hashes).expect("parse hashes json");

    assert_eq!(manifest_json, serde_json::json!({ "artifacts": {} }));
    assert_eq!(hashes_json, serde_json::json!({ "inputs": {} }));
}

#[test]
fn ingest_file_registers_source_and_sidecar() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let source = kb_root.join("example.md");
    fs::write(&source, "# hello\n").expect("write source");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json").arg("ingest").arg(&source);
    let output = cmd.output().expect("run kb ingest");

    assert!(
        output.status.success(),
        "kb ingest failed with stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let payload: Value = serde_json::from_slice(&output.stdout).expect("parse ingest json");
    let items = payload["results"]
        .as_array()
        .expect("ingest results should be an array");
    assert_eq!(items.len(), 1);
    assert_eq!(payload["summary"]["new_sources"], 1);

    let copied_path = items[0]["content_path"]
        .as_str()
        .expect("content_path should be a string");
    let sidecar_path = items[0]["metadata_path"]
        .as_str()
        .expect("metadata_path should be a string");

    assert!(kb_root.join(copied_path).is_file());
    assert!(kb_root.join(sidecar_path).is_file());
}

#[test]
fn ingest_directory_respects_gitignore() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let corpus = kb_root.join("corpus");
    fs::create_dir_all(&corpus).expect("create corpus dir");
    fs::write(corpus.join(".gitignore"), "ignored.md\n").expect("write gitignore");
    fs::write(corpus.join("kept.md"), "keep\n").expect("write kept");
    fs::write(corpus.join("ignored.md"), "ignore\n").expect("write ignored");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json").arg("ingest").arg(&corpus);
    let output = cmd.output().expect("run kb ingest directory");

    assert!(
        output.status.success(),
        "kb ingest directory failed with stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let payload: Value = serde_json::from_slice(&output.stdout).expect("parse ingest json");
    let items = payload["results"]
        .as_array()
        .expect("ingest results should be an array");
    assert_eq!(items.len(), 1);
    assert_eq!(
        items[0]["content_path"]
            .as_str()
            .expect("content_path should be a string")
            .rsplit('/')
            .next()
            .expect("content_path should have a filename"),
        "kept.md"
    );
}

#[test]
fn ingest_dry_run_makes_no_filesystem_changes() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let source = kb_root.join("example.md");
    fs::write(&source, "# hello\n").expect("write source");

    let before: Vec<_> = fs::read_dir(&kb_root)
        .expect("read kb root")
        .map(|entry| entry.expect("dir entry").file_name())
        .collect();

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--dry-run")
        .arg("--json")
        .arg("ingest")
        .arg(&source);
    let output = cmd.output().expect("run kb ingest dry run");

    assert!(
        output.status.success(),
        "dry run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let payload: Value = serde_json::from_slice(&output.stdout).expect("parse ingest json");
    assert_eq!(payload["dry_run"], true);
    assert_eq!(payload["summary"]["new_sources"], 1);

    let after: Vec<_> = fs::read_dir(&kb_root)
        .expect("read kb root")
        .map(|entry| entry.expect("dir entry").file_name())
        .collect();
    assert_eq!(before, after);
}

#[test]
fn inspect_reads_dependency_graph() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut graph = Graph::default();
    graph.record(["raw/inbox/example.md"], ["normalized/example.json"]);
    graph.record(["normalized/example.json"], ["wiki/sources/example.md"]);
    graph.record(["wiki/sources/example.md"], ["wiki/index.md"]);
    graph.persist_to(&kb_root).expect("persist graph");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("inspect").arg("wiki/index.md");
    let output = cmd.output().expect("run kb inspect");

    assert!(
        output.status.success(),
        "kb inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("node: wiki/index.md"));
    assert!(stdout.contains("- wiki/sources/example.md"));
    assert!(stdout.contains("- raw/inbox/example.md"));
}

#[test]
fn ingest_mixed_file_directory_and_url_reports_summary() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let file = kb_root.join("single.md");
    fs::write(&file, "# file\n").expect("write file");

    let corpus = kb_root.join("corpus");
    fs::create_dir_all(&corpus).expect("create corpus dir");
    fs::write(corpus.join("nested.md"), "# nested\n").expect("write nested");

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind listener");
    let address = listener.local_addr().expect("local addr");
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept connection");
        let mut buffer = [0_u8; 1024];
        let _ = stream.read(&mut buffer).expect("read request");
        let body = "<html><head><title>Example</title></head><body><article><h1>hello</h1><p>world</p></article></body></html>";
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        )
        .expect("write response");
    });

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json")
        .arg("ingest")
        .arg(&file)
        .arg(format!("http://{address}/article"))
        .arg(&corpus);
    let output = cmd.output().expect("run mixed ingest");
    server.join().expect("join server");

    assert!(
        output.status.success(),
        "mixed ingest failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let payload: Value = serde_json::from_slice(&output.stdout).expect("parse ingest json");
    assert_eq!(payload["summary"]["total"], 3);
    assert_eq!(payload["summary"]["new_sources"], 3);
    assert_eq!(
        payload["results"]
            .as_array()
            .expect("results array")
            .iter()
            .filter(|item| item["source_kind"] == "url")
            .count(),
        1
    );
}

fn write_source_page(root: &Path, slug: &str, title: &str, summary: &str) {
    let dir = root.join("wiki/sources");
    fs::create_dir_all(&dir).expect("create wiki/sources");
    let content = format!(
        "---\nid: wiki-source-{slug}\ntype: source\ntitle: {title}\n---\n\
         \n# Source\n<!-- kb:begin id=title -->\n{title}\n<!-- kb:end id=title -->\n\
         \n## Summary\n<!-- kb:begin id=summary -->\n{summary}\n<!-- kb:end id=summary -->\n"
    );
    fs::write(dir.join(format!("{slug}.md")), content).expect("write source page");
}

fn write_concept_page(root: &Path, slug: &str, name: &str, aliases: &[&str]) {
    use std::fmt::Write as _;
    let dir = root.join("wiki/concepts");
    fs::create_dir_all(&dir).expect("create wiki/concepts");
    let mut content = format!("---\nid: concept:{slug}\nname: {name}\n");
    if !aliases.is_empty() {
        content.push_str("aliases:\n");
        for alias in aliases {
            writeln!(content, "  - {alias}").expect("write alias");
        }
    }
    writeln!(content, "---\n\n# {name}\n\nThis concept page covers {name}.").expect("write body");
    fs::write(dir.join(format!("{slug}.md")), content).expect("write concept page");
}

#[test]
fn compile_builds_lexical_index() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    write_source_page(&kb_root, "rust-book", "The Rust Programming Language", "Memory safety.");
    write_concept_page(&kb_root, "borrow-checker", "Borrow checker", &["borrowck"]);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("compile");
    let output = cmd.output().expect("run kb compile");
    assert!(
        output.status.success(),
        "kb compile failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let index_path = kb_root.join("state/indexes/lexical.json");
    assert!(index_path.exists(), "lexical index should exist after compile");

    let raw = fs::read_to_string(&index_path).expect("read index");
    let json: Value = serde_json::from_str(&raw).expect("parse index");
    let entries = json["entries"].as_array().expect("entries array");
    assert_eq!(entries.len(), 2, "should index 2 pages");
}

#[test]
fn search_returns_ranked_results_after_compile() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    write_source_page(
        &kb_root,
        "rust-overview",
        "Rust Overview",
        "An introduction to the Rust programming language.",
    );
    write_source_page(
        &kb_root,
        "python-intro",
        "Python Introduction",
        "Getting started with Python.",
    );
    write_concept_page(&kb_root, "ownership", "Rust Ownership", &["borrow"]);

    let mut compile_cmd = kb_cmd(&kb_root);
    compile_cmd.arg("compile");
    compile_cmd.output().expect("run kb compile");

    let mut search_cmd = kb_cmd(&kb_root);
    search_cmd.arg("search").arg("rust");
    let output = search_cmd.output().expect("run kb search");
    assert!(
        output.status.success(),
        "kb search failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Rust"),
        "search results should contain 'Rust' pages: {stdout}"
    );
    assert!(
        !stdout.contains("Python"),
        "Python page should not appear in 'rust' search: {stdout}"
    );
}

#[test]
fn search_with_no_index_reports_helpful_tip() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("search").arg("anything");
    let output = cmd.output().expect("run kb search with no index");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("compile"),
        "should suggest running compile: {stdout}"
    );
}
