mod common;

use common::{kb_cmd, make_temp_kb};
use kb_compile::Graph;
use kb_core::{
    BuildRecord, EntityMetadata, ReviewItem, ReviewKind, ReviewStatus, Status, save_build_record,
    save_review_item,
};
use regex::Regex;
use serde_json::Value;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
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

fn write_executable(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write executable");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(path).expect("metadata").permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions).expect("chmod");
    }
}

fn install_fake_harnesses(root: &Path) -> PathBuf {
    let bin_dir = root.join("fake-bin");
    fs::create_dir_all(&bin_dir).expect("create fake bin dir");
    write_executable(bin_dir.join("opencode").as_path(), "#!/bin/sh\nprintf 'OK'");
    write_executable(
        bin_dir.join("claude").as_path(),
        "#!/bin/sh\nprintf '{\"result\":\"OK\"}'",
    );
    bin_dir
}

fn prepend_path(dir: &Path) -> String {
    let existing = std::env::var_os("PATH").unwrap_or_default();
    let mut paths = vec![dir.to_path_buf()];
    paths.extend(std::env::split_paths(&existing));
    std::env::join_paths(paths)
        .expect("join PATH")
        .to_string_lossy()
        .into_owned()
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

    let manifest =
        fs::read_to_string(kb_root.join("state/manifest.json")).expect("read manifest state file");
    let manifest_json: Value = serde_json::from_str(&manifest).expect("parse manifest json");
    assert_eq!(manifest_json, serde_json::json!({ "artifacts": {} }));

    // hashes.json is not created at init — it's written by the first
    // successful `kb compile` in the canonical HashState schema
    // (see bn-1pw: removed the stale Hashes default-write from init).
    assert!(
        !kb_root.join("state/hashes.json").exists(),
        "hashes.json should not exist until first compile"
    );
    assert!(
        kb_root.join("state/build_records").exists(),
        "state/build_records must exist after init"
    );
}

#[test]
fn ask_creates_question_record_and_placeholder_artifact() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json")
        .arg("ask")
        .arg("How does the pipeline work?");
    let output = cmd.output().expect("run kb ask");

    assert!(
        output.status.success(),
        "kb ask failed with stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse ask json");
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "ask");
    let payload = &envelope["data"];
    let question_path = kb_root.join(
        payload["question_path"]
            .as_str()
            .expect("question_path should be a string"),
    );
    let artifact_path = kb_root.join(
        payload["artifact_path"]
            .as_str()
            .expect("artifact_path should be a string"),
    );

    assert!(question_path.is_file());
    assert!(artifact_path.is_file());

    let question_record: Value =
        serde_json::from_str(&fs::read_to_string(&question_path).expect("read question record"))
            .expect("parse question record");
    assert_eq!(question_record["raw_query"], "How does the pipeline work?");
    assert_eq!(question_record["requested_format"], "md");
    assert_eq!(question_record["requesting_context"], "project_kb");

    let retrieval_plan_path = kb_root.join(
        question_record["retrieval_plan"]
            .as_str()
            .expect("retrieval_plan should be a string"),
    );
    assert!(retrieval_plan_path.is_file());

    let retrieval_plan: Value = serde_json::from_str(
        &fs::read_to_string(&retrieval_plan_path).expect("read retrieval plan"),
    )
    .expect("parse retrieval plan");
    assert_eq!(retrieval_plan["query"], "How does the pipeline work?");
    assert_eq!(retrieval_plan["token_budget"], 20_000);

    let artifact = fs::read_to_string(&artifact_path).expect("read artifact placeholder");
    assert!(artifact.contains("question_id:"));
    assert!(artifact.contains("requested_format: md"));
    assert!(
        artifact.contains("LLM unavailable") || artifact.contains("type: question_answer"),
        "artifact should contain LLM unavailable message or valid artifact header"
    );

    let metadata_path = artifact_path.with_file_name("metadata.json");
    assert!(metadata_path.is_file(), "metadata.json sidecar should exist");
}

#[test]
fn ask_persists_ranked_retrieval_plan_after_compile() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    write_source_page(
        &kb_root,
        "rust-overview",
        "Rust Overview",
        "Rust ownership and borrowing basics.",
    );
    write_concept_page(&kb_root, "borrow-checker", "Borrow checker", &["borrowck"]);

    let mut compile_cmd = kb_cmd(&kb_root);
    compile_cmd.arg("compile");
    let compile_output = compile_cmd.output().expect("run kb compile");
    assert!(
        compile_output.status.success(),
        "kb compile failed: {}",
        String::from_utf8_lossy(&compile_output.stderr)
    );

    let mut ask_cmd = kb_cmd(&kb_root);
    ask_cmd.arg("--json").arg("ask").arg("borrowck rust checker");
    let output = ask_cmd.output().expect("run kb ask");
    assert!(
        output.status.success(),
        "kb ask failed with stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse ask json");
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "ask");
    let payload = &envelope["data"];
    let question_path = kb_root.join(
        payload["question_path"]
            .as_str()
            .expect("question_path should be a string"),
    );
    let question_record: Value = serde_json::from_str(
        &fs::read_to_string(&question_path).expect("read question record"),
    )
    .expect("parse question record");
    let retrieval_plan_path = kb_root.join(
        question_record["retrieval_plan"]
            .as_str()
            .expect("retrieval_plan should be a string"),
    );
    let retrieval_plan: Value = serde_json::from_str(
        &fs::read_to_string(&retrieval_plan_path).expect("read retrieval plan"),
    )
    .expect("parse retrieval plan");

    let candidates = retrieval_plan["candidates"]
        .as_array()
        .expect("candidates array");
    assert_eq!(candidates.len(), 2);
    assert_eq!(candidates[0]["id"], "wiki/concepts/borrow-checker.md");
    let first_score = candidates[0]["score"].as_u64().expect("first score");
    let second_score = candidates[1]["score"].as_u64().expect("second score");
    assert!(first_score >= second_score);
    let reasons = candidates[0]["reasons"].as_array().expect("reasons array");
    assert!(reasons.iter().any(|reason| {
        reason
            .as_str()
            .is_some_and(|text| text.contains("alias matched 'borrowck'"))
    }));
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

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse ingest json");
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "ingest");
    let payload = &envelope["data"];
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

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse ingest json");
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "ingest");
    let payload = &envelope["data"];
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

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse ingest json");
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "ingest");
    let payload = &envelope["data"];
    assert_eq!(payload["dry_run"], true);
    assert_eq!(payload["summary"]["new_sources"], 1);

    let after: Vec<_> = fs::read_dir(&kb_root)
        .expect("read kb root")
        .map(|entry| entry.expect("dir entry").file_name())
        .collect();
    assert_eq!(before, after);
}

fn test_metadata(id: &str) -> EntityMetadata {
    EntityMetadata {
        id: id.to_string(),
        created_at_millis: 1_700_000_000_000,
        updated_at_millis: 1_700_000_000_500,
        source_hashes: vec!["hash-1".to_string()],
        model_version: Some("test-model".to_string()),
        tool_version: Some("kb-test".to_string()),
        prompt_template_hash: Some("tmpl-1".to_string()),
        dependencies: Vec::new(),
        output_paths: Vec::new(),
        status: Status::Fresh,
    }
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
    assert!(stdout.contains("resolved_id: wiki/index.md"));
    assert!(stdout.contains("direct inputs:"));
    assert!(stdout.contains("- wiki/sources/example.md"));
    assert!(stdout.contains("- raw/inbox/example.md"));
}

#[test]
fn inspect_json_trace_and_build_records_are_reported() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    fs::write(
        kb_root.join("wiki/index.md"),
        "# Index\n\n## Citations\n- [[wiki/sources/example.md]]\n",
    )
    .expect("write wiki index");

    let mut graph = Graph::default();
    graph.record(["raw/inbox/example.md"], ["normalized/example.json"]);
    graph.record(["normalized/example.json"], ["wiki/sources/example.md"]);
    graph.record(["wiki/sources/example.md"], ["wiki/index.md"]);
    graph.persist_to(&kb_root).expect("persist graph");

    save_build_record(
        &kb_root,
        &BuildRecord {
            metadata: test_metadata("build-index"),
            pass_name: "index".to_string(),
            input_ids: vec!["wiki/sources/example.md".to_string()],
            output_ids: vec!["wiki/index.md".to_string()],
            manifest_hash: "manifest-1".to_string(),
        },
    )
    .expect("save build record");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json")
        .arg("inspect")
        .arg("--trace")
        .arg("wiki/index.md");
    let output = cmd.output().expect("run kb inspect --json --trace");

    assert!(
        output.status.success(),
        "kb inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse inspect payload");
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "inspect");
    let payload = &envelope["data"];
    assert_eq!(payload["resolved_id"], "wiki/index.md");
    assert_eq!(payload["kind"], "wiki_page");
    assert_eq!(payload["freshness"], "fresh");
    assert_eq!(payload["graph"]["direct_inputs"][0], "wiki/sources/example.md");
    assert_eq!(payload["citations"][0], "- [[wiki/sources/example.md]]");
    assert_eq!(payload["build_records"][0]["id"], "build-index");
    assert_eq!(payload["trace"][0]["id"], "wiki/sources/example.md");
}

#[test]
fn inspect_missing_target_has_actionable_error() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("inspect").arg("missing-target");
    let output = cmd.output().expect("run kb inspect missing target");

    assert!(!output.status.success(), "kb inspect unexpectedly succeeded");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Run 'kb compile' first"));
    assert!(stderr.contains("was not found"));
}

#[test]
fn inspect_resolves_by_frontmatter_id() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let sources_dir = kb_root.join("wiki/sources");
    fs::create_dir_all(&sources_dir).expect("create wiki/sources");
    let wiki_file = sources_dir.join("foo.md");
    fs::write(
        &wiki_file,
        "---\nid: my-test-id\n---\n\n# Foo\n\nSome body.\n",
    )
    .expect("write wiki file");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("inspect").arg("my-test-id");
    let output = cmd.output().expect("run kb inspect my-test-id");

    assert!(
        output.status.success(),
        "kb inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("resolved_id: wiki/sources/foo.md"),
        "expected resolved_id to point at foo.md, got:\n{stdout}"
    );
}

#[test]
fn inspect_ambiguous_frontmatter_id_reports_matches() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let sources_dir = kb_root.join("wiki/sources");
    fs::create_dir_all(&sources_dir).expect("create wiki/sources");
    fs::write(
        sources_dir.join("one.md"),
        "---\nid: dup-id\n---\n\n# One\n",
    )
    .expect("write one.md");
    fs::write(
        sources_dir.join("two.md"),
        "---\nid: dup-id\n---\n\n# Two\n",
    )
    .expect("write two.md");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("inspect").arg("dup-id");
    let output = cmd.output().expect("run kb inspect dup-id");

    assert!(!output.status.success(), "kb inspect unexpectedly succeeded");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("ambiguous"), "stderr was: {stderr}");
    assert!(stderr.contains("wiki/sources/one.md"), "stderr was: {stderr}");
    assert!(stderr.contains("wiki/sources/two.md"), "stderr was: {stderr}");
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

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse ingest json");
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "ingest");
    let payload = &envelope["data"];
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

#[test]
fn error_output_includes_inner_cause_chain() {
    // Regression guard for bn-1g5: `eprintln!("error: {err}")` only printed
    // anyhow's outermost context, so the inner cause (HTTP status, IO error,
    // ...) was hidden. Using `{err:#}` joins the full chain with `: ` so
    // every context level reaches the user.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

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
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("ingest").arg(&url);
    let output = cmd.output().expect("run kb ingest against 404 stub");
    server.join().expect("join server");

    assert!(
        !output.status.success(),
        "kb ingest against 404 URL should fail"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("fetch"),
        "stderr should include outer 'fetch' context; got: {stderr}"
    );
    assert!(
        stderr.contains("404"),
        "stderr should include inner HTTP 404 cause; got: {stderr}"
    );
    assert!(
        stderr.contains(&url),
        "stderr should include the URL; got: {stderr}"
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
    writeln!(
        content,
        "---\n\n# {name}\n\nThis concept page covers {name}."
    )
    .expect("write body");
    fs::write(dir.join(format!("{slug}.md")), content).expect("write concept page");
}

#[test]
fn compile_builds_lexical_index() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    write_source_page(
        &kb_root,
        "rust-book",
        "The Rust Programming Language",
        "Memory safety.",
    );
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
    assert!(
        index_path.exists(),
        "lexical index should exist after compile"
    );

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

#[test]
fn search_includes_ranking_reasons() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    write_concept_page(&kb_root, "borrowck", "Borrow Checker", &["borrowck"]);
    write_source_page(
        &kb_root,
        "memory",
        "Memory Safety",
        "The borrow checker prevents memory errors.",
    );

    let mut compile_cmd = kb_cmd(&kb_root);
    compile_cmd.arg("compile");
    compile_cmd.output().expect("run kb compile");

    let mut search_cmd = kb_cmd(&kb_root);
    search_cmd.arg("search").arg("borrowck");
    let output = search_cmd.output().expect("run kb search");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("reason:"),
        "search results should include reasons: {stdout}"
    );
    assert!(
        stdout.contains("alias matched") || stdout.contains("title matched"),
        "reasons should explain which field matched: {stdout}"
    );
}

#[test]
fn search_with_limit_flag() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    for i in 0..5 {
        write_source_page(
            &kb_root,
            &format!("rust-{i}"),
            &format!("Rust Guide {i}"),
            "Rust programming language.",
        );
    }

    let mut compile_cmd = kb_cmd(&kb_root);
    compile_cmd.arg("compile");
    compile_cmd.output().expect("run kb compile");

    let mut search_cmd = kb_cmd(&kb_root);
    search_cmd.arg("search").arg("rust").arg("--limit").arg("2");
    let output = search_cmd.output().expect("run kb search with limit");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line_count = stdout.lines().count();
    assert!(
        line_count <= 10,
        "with --limit 2, output should have <=10 lines (title + reason lines): {stdout}"
    );
}

#[test]
fn search_with_json_output() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    write_concept_page(&kb_root, "ownership", "Rust Ownership", &[]);
    write_source_page(
        &kb_root,
        "rust-intro",
        "Rust Introduction",
        "Introduction to Rust.",
    );

    let mut compile_cmd = kb_cmd(&kb_root);
    compile_cmd.arg("compile");
    compile_cmd.output().expect("run kb compile");

    let mut search_cmd = kb_cmd(&kb_root);
    search_cmd
        .arg("search")
        .arg("rust")
        .arg("--json");
    let output = search_cmd.output().expect("run kb search with --json");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    let envelope: Value = serde_json::from_str(&stdout)
        .expect("search with --json should return valid JSON envelope");
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "search");

    let results = envelope["data"].as_array().expect("data should be an array");
    assert!(!results.is_empty(), "should have search results");

    let first = &results[0];
    assert!(
        first.get("id").is_some() && first.get("title").is_some() && first.get("score").is_some(),
        "JSON results should have id, title, and score fields: {first}"
    );
    assert!(
        first.get("reasons").is_some(),
        "JSON results should include reasons field: {first}"
    );
}

#[test]
fn lint_reports_broken_links_with_line_numbers_and_suggestions() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    write_concept_page(&kb_root, "rust", "Rust", &[]);
    fs::write(
        kb_root.join("wiki/sources/page.md"),
        "# Page\nSee [[wiki/concepts/rsut]].\nSee [[wiki/concepts/rust#summry]].\n",
    )
    .expect("write page with broken links");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("lint");
    let output = cmd.output().expect("run kb lint");

    assert!(
        !output.status.success(),
        "kb lint should fail when broken links exist"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("wiki/sources/page.md:2"),
        "stdout: {stdout}"
    );
    assert!(
        stdout.contains("wiki/sources/page.md:3"),
        "stdout: {stdout}"
    );
    assert!(
        stdout.contains("suggested fix: wiki/concepts/rust"),
        "stdout: {stdout}"
    );
    assert!(
        stdout.contains("suggested fix: wiki/concepts/rust#rust")
            || stdout.contains("suggested fix: wiki/concepts/rust#summary"),
        "stdout: {stdout}"
    );
}

#[test]
fn lint_json_reports_clean_relative_links() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    write_concept_page(&kb_root, "borrow-checker", "Borrow checker", &[]);
    fs::write(
        kb_root.join("wiki/sources/page.md"),
        "# Page\nSee [borrow checker](../concepts/borrow-checker.md).\n",
    )
    .expect("write page with relative link");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json").arg("lint").arg("--check").arg("broken-links");
    let output = cmd.output().expect("run kb lint json");

    assert!(
        output.status.success(),
        "kb lint should pass for valid relative links: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse lint json");
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "lint");
    let payload = &envelope["data"];
    let checks = payload["checks"].as_array().expect("checks array");
    assert_eq!(checks.len(), 1);
    let broken_links = &checks[0];
    assert_eq!(broken_links["check"], "broken-links");
    assert_eq!(broken_links["issue_count"], 0);
    assert_eq!(
        broken_links["issues"].as_array().expect("issues array").len(),
        0
    );
}

#[test]
fn lint_reports_orphan_pages_as_json() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    fs::create_dir_all(kb_root.join("wiki/sources")).expect("create wiki sources");
    fs::write(
        kb_root.join("wiki/sources/orphan.md"),
        "---\nsource_document_id: doc-1\nsource_revision_id: rev-1\n---\n# Orphan\n",
    )
    .expect("write orphan page");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json").arg("lint").arg("--check").arg("orphans");
    let output = cmd.output().expect("run kb lint");

    assert!(
        !output.status.success(),
        "lint should fail when issues are found"
    );

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse lint json");
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "lint");
    let payload = &envelope["data"];
    let checks = payload["checks"].as_array().expect("checks array");
    assert_eq!(checks.len(), 1);
    let orphans = &checks[0];
    assert_eq!(orphans["issue_count"], 2);
    let issues = orphans["issues"].as_array().expect("issues array");
    assert!(issues.iter().any(|issue| issue["kind"] == "source_document_missing"));
    assert!(issues.iter().any(|issue| issue["kind"] == "source_revision_missing"));
}

#[test]
fn lint_exits_zero_when_only_warnings() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    fs::create_dir_all(kb_root.join("wiki/sources")).expect("create wiki sources");
    fs::write(
        kb_root.join("wiki/sources/page.md"),
        "---\nsource_document_id: doc-1\nsource_revision_id: rev-1\n---\n# Page\n\n## Summary\n<!-- kb:begin id=summary -->\nSynthetic summary.\n<!-- kb:end id=summary -->\n",
    )
    .expect("write page");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("lint").arg("--check").arg("missing-citations");
    let output = cmd.output().expect("run kb lint");

    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stdout.contains("[warn] missing-citations"),
        "stdout: {stdout}"
    );
    // No "error:" prefix on warning-only runs
    assert!(
        !stderr.contains("error:"),
        "warning-only lint must not print 'error:' prefix, stderr: {stderr}"
    );
    assert!(
        !stdout.contains("error:"),
        "warning-only lint must not print 'error:' prefix, stdout: {stdout}"
    );
    assert!(
        stdout.contains("use --strict to fail on warnings"),
        "expected informational hint about --strict, stdout: {stdout}"
    );
}

#[test]
fn lint_exits_one_when_any_error() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    write_concept_page(&kb_root, "rust", "Rust", &[]);
    fs::write(
        kb_root.join("wiki/sources/page.md"),
        "# Page\nSee [[wiki/concepts/rsut]].\n",
    )
    .expect("write page with broken link");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("lint").arg("--check").arg("broken-links");
    let output = cmd.output().expect("run kb lint");

    assert_eq!(
        output.status.code(),
        Some(1),
        "stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn lint_strict_exits_one_on_warnings() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    fs::create_dir_all(kb_root.join("wiki/sources")).expect("create wiki sources");
    fs::write(
        kb_root.join("wiki/sources/page.md"),
        "---\nsource_document_id: doc-1\nsource_revision_id: rev-1\n---\n# Page\n\n## Summary\n<!-- kb:begin id=summary -->\nSynthetic summary.\n<!-- kb:end id=summary -->\n",
    )
    .expect("write page");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("lint")
        .arg("--check")
        .arg("missing-citations")
        .arg("--strict");
    let output = cmd.output().expect("run kb lint --strict");

    assert_eq!(
        output.status.code(),
        Some(1),
        "stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn lint_missing_citations_can_fail_when_configured_as_error() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    fs::write(
        kb_root.join("kb.toml"),
        "[llm]\ndefault_model = \"test-model\"\n\n[compile]\ntoken_budget = 12000\n\n[lint]\nmissing_citations_level = \"error\"\n",
    )
    .expect("rewrite kb.toml");

    fs::create_dir_all(kb_root.join("wiki/sources")).expect("create wiki sources");
    fs::write(
        kb_root.join("wiki/sources/page.md"),
        "---\nsource_document_id: doc-1\nsource_revision_id: rev-1\n---\n# Page\n\n## Summary\n<!-- kb:begin id=summary -->\nSynthetic summary.\n<!-- kb:end id=summary -->\n",
    )
    .expect("write page");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json").arg("lint").arg("--check").arg("missing-citations");
    let output = cmd.output().expect("run kb lint");

    assert_eq!(
        output.status.code(),
        Some(1),
        "stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse lint json");
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "lint");
    let payload = &envelope["data"];
    let checks = payload["checks"].as_array().expect("checks array");
    assert_eq!(checks.len(), 1);
    let missing_citations = &checks[0];
    assert_eq!(missing_citations["issue_count"], 1);
    assert_eq!(missing_citations["issues"][0]["severity"], "error");
    assert_eq!(missing_citations["issues"][0]["kind"], "missing_citations");
}

#[test]
fn doctor_returns_zero_when_all_checks_pass() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    let fake_bin = install_fake_harnesses(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.env("PATH", prepend_path(&fake_bin));
    cmd.arg("--json").arg("doctor");
    let output = cmd.output().expect("run kb doctor");

    assert!(
        output.status.success(),
        "stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.status.code(), Some(0));

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse doctor json");
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "doctor");
    let payload = &envelope["data"];
    assert_eq!(payload["status"], "ok");
    assert_eq!(payload["exit_code"], 0);
    assert_eq!(payload["error_count"], 0);
    assert_eq!(payload["warning_count"], 0);
}

#[test]
fn doctor_returns_error_when_harness_is_missing() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    fs::create_dir_all(kb_root.join("empty-bin")).expect("create empty bin dir");

    let mut cmd = kb_cmd(&kb_root);
    cmd.env("PATH", kb_root.join("empty-bin"));
    cmd.arg("doctor");
    let output = cmd.output().expect("run kb doctor");

    assert!(!output.status.success());
    assert_eq!(output.status.code(), Some(2));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("harness:opencode") || stdout.contains("harness:claude"));
    assert!(stdout.contains("missing"));
}

#[test]
fn doctor_returns_warning_when_interrupted_jobs_exist() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    let fake_bin = install_fake_harnesses(&kb_root);

    fs::write(
        kb_root.join("state/jobs/interrupted-test.json"),
        serde_json::json!({
            "metadata": {
                "id": "interrupted-test",
                "created_at_millis": 1,
                "updated_at_millis": 1,
                "source_hashes": [],
                "model_version": null,
                "tool_version": "kb-cli/0.1.0",
                "prompt_template_hash": null,
                "dependencies": [],
                "output_paths": [],
                "status": "fresh"
            },
            "command": "compile",
            "root_path": kb_root,
            "started_at_millis": 1,
            "ended_at_millis": null,
            "status": "running",
            "log_path": null,
            "affected_outputs": [],
            "pid": 999_999,
            "exit_code": null
        })
        .to_string(),
    )
    .expect("write interrupted job manifest");

    let mut cmd = kb_cmd(&kb_root);
    cmd.env("PATH", prepend_path(&fake_bin));
    cmd.arg("--json").arg("doctor");
    let output = cmd.output().expect("run kb doctor");

    assert!(!output.status.success());
    assert_eq!(output.status.code(), Some(1));
    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse doctor json");
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "doctor");
    let payload = &envelope["data"];
    assert_eq!(payload["status"], "warn");
    assert_eq!(payload["exit_code"], 1);
    assert!(
        payload["checks"]
            .as_array()
            .expect("checks array")
            .iter()
            .any(|check| check["name"] == "interrupted_jobs" && check["status"] == "warn")
    );
}

#[test]
fn ask_dry_run_prints_retrieval_plan_without_calling_llm() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    write_source_page(&kb_root, "rust-guide", "Rust Guide", "A guide to Rust.");

    let mut compile_cmd = kb_cmd(&kb_root);
    compile_cmd.arg("compile");
    compile_cmd.output().expect("compile");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json")
        .arg("--dry-run")
        .arg("ask")
        .arg("How does Rust work?");
    let output = cmd.output().expect("run kb ask --dry-run");

    assert!(
        output.status.success(),
        "kb ask --dry-run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse dry-run json");
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "ask");
    let plan = &envelope["data"];
    assert!(plan["query"].is_string());
    assert!(plan["token_budget"].is_number());
    assert!(plan["candidates"].is_array());

    let outputs_dir = kb_root.join("outputs/questions");
    if outputs_dir.exists() {
        let entries: Vec<_> = fs::read_dir(&outputs_dir)
            .expect("read outputs/questions")
            .filter_map(Result::ok)
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with("question-")
            })
            .collect();
        assert!(
            entries.is_empty(),
            "dry-run should not create question output directories, found {}",
            entries.len()
        );
    }
}

#[test]
fn ask_promote_creates_review_item() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json")
        .arg("ask")
        .arg("--promote")
        .arg("What is testing?");
    let output = cmd.output().expect("run kb ask --promote");

    assert!(
        output.status.success(),
        "kb ask --promote failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse ask json");
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "ask");
    let payload = &envelope["data"];
    let question_id = payload["question_id"]
        .as_str()
        .expect("question_id should be a string");

    let review_dir = kb_root.join("reviews/promotions");
    assert!(review_dir.is_dir(), "reviews/promotions directory should exist");

    let review_path = review_dir.join(format!("review-{question_id}.json"));
    assert!(
        review_path.is_file(),
        "ReviewItem file should exist at {}",
        review_path.display()
    );

    let review: Value =
        serde_json::from_str(&fs::read_to_string(&review_path).expect("read review item"))
            .expect("parse review item");
    assert_eq!(review["kind"], "promotion");
    assert_eq!(review["status"], "pending");
    assert_eq!(review["proposed_destination"], "wiki/questions/what-is-testing.md");
    assert!(review["affected_pages"].as_array().is_some_and(|pages| !pages.is_empty()));
    assert!(review["comment"]
        .as_str()
        .expect("comment")
        .contains("What is testing?"));
}

// ── Snapshot schema tests ────────────────────────────────────────────────────
// Each test verifies the stable envelope wrapper and the set of keys in `data`.
// Dynamic values (timestamps, IDs, paths) are not included in the snapshot.

fn sorted_object_keys(value: &Value) -> String {
    let mut keys: Vec<&str> = value
        .as_object()
        .expect("expected JSON object")
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    keys.join(", ")
}

#[test]
fn json_schema_envelope_ingest() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    let source = kb_root.join("example.md");
    fs::write(&source, "# hello\n").expect("write source");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json").arg("ingest").arg(&source);
    let output = cmd.output().expect("run kb ingest");
    assert!(output.status.success());

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse json");
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "ingest");
    insta::assert_snapshot!("ingest_envelope_keys", sorted_object_keys(&envelope));
    insta::assert_snapshot!("ingest_data_keys", sorted_object_keys(&envelope["data"]));
}

#[test]
fn json_schema_envelope_compile() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    write_source_page(&kb_root, "a", "Title A", "Body A.");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json").arg("compile");
    let output = cmd.output().expect("run kb compile");
    assert!(output.status.success());

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse json");
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "compile");
    insta::assert_snapshot!("compile_envelope_keys", sorted_object_keys(&envelope));
    insta::assert_snapshot!("compile_data_keys", sorted_object_keys(&envelope["data"]));
}

#[test]
fn json_schema_envelope_search_empty() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json").arg("search").arg("anything");
    let output = cmd.output().expect("run kb search");
    assert!(output.status.success());

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse json");
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "search");
    assert!(envelope["data"].is_array(), "search data should be an array");
    assert_eq!(
        envelope["data"].as_array().expect("data array").len(),
        0,
        "empty search should have zero results in data"
    );
    insta::assert_snapshot!("search_empty_envelope_keys", sorted_object_keys(&envelope));
}

#[test]
fn json_schema_envelope_inspect() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut graph = kb_compile::Graph::default();
    graph.record(["raw/inbox/doc.md"], ["wiki/sources/doc.md"]);
    graph.persist_to(&kb_root).expect("persist graph");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json").arg("inspect").arg("wiki/sources/doc.md");
    let output = cmd.output().expect("run kb inspect");
    assert!(output.status.success());

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse json");
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "inspect");
    insta::assert_snapshot!("inspect_envelope_keys", sorted_object_keys(&envelope));
    insta::assert_snapshot!("inspect_data_keys", sorted_object_keys(&envelope["data"]));
}

#[test]
fn json_schema_envelope_lint() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json").arg("lint").arg("--check").arg("broken-links");
    let output = cmd.output().expect("run kb lint");
    assert!(output.status.success());

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse json");
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "lint");
    insta::assert_snapshot!("lint_envelope_keys", sorted_object_keys(&envelope));
    insta::assert_snapshot!("lint_data_keys", sorted_object_keys(&envelope["data"]));
}

// ── Review command tests ────────────────────────────────────────────────────

fn make_review_item(id: &str, status: ReviewStatus) -> ReviewItem {
    ReviewItem {
        metadata: EntityMetadata {
            id: id.to_string(),
            created_at_millis: 1_000,
            updated_at_millis: 1_000,
            source_hashes: vec!["hash-1".to_string()],
            model_version: None,
            tool_version: Some("kb-test".to_string()),
            prompt_template_hash: None,
            dependencies: vec!["artifact-q1".to_string()],
            output_paths: vec![PathBuf::from("outputs/questions/q1/answer.md")],
            status: Status::NeedsReview,
        },
        kind: ReviewKind::Promotion,
        target_entity_id: "artifact-q1".to_string(),
        proposed_destination: Some(PathBuf::from("wiki/questions/test.md")),
        citations: vec!["src-1#intro".to_string()],
        affected_pages: vec![PathBuf::from("wiki/questions/test.md")],
        created_at_millis: 1_000,
        status,
        comment: "Promote test answer".to_string(),
    }
}

#[test]
fn review_list_shows_pending_items() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let item = make_review_item("review-list-1", ReviewStatus::Pending);
    save_review_item(&kb_root, &item).expect("save review item");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("review").arg("list");
    let output = cmd.output().expect("run kb review list");
    assert!(
        output.status.success(),
        "kb review list failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("review-list-1"));
    assert!(stdout.contains("promotion"));
}

#[test]
fn review_list_json_output() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let item = make_review_item("review-json-1", ReviewStatus::Pending);
    save_review_item(&kb_root, &item).expect("save review item");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json").arg("review").arg("list");
    let output = cmd.output().expect("run kb review list --json");
    assert!(output.status.success());

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse json");
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "review.list");
    let data = &envelope["data"];
    assert_eq!(data["counts"]["pending"], 1);
    assert_eq!(data["items"][0]["id"], "review-json-1");
}

#[test]
fn review_show_displays_item_details() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let item = make_review_item("review-show-1", ReviewStatus::Pending);
    save_review_item(&kb_root, &item).expect("save review item");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("review").arg("show").arg("review-show-1");
    let output = cmd.output().expect("run kb review show");
    assert!(
        output.status.success(),
        "kb review show failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("review-show-1"));
    assert!(stdout.contains("pending"));
    assert!(stdout.contains("Promote test answer"));
}

#[test]
fn review_reject_marks_item_rejected() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let item = make_review_item("review-reject-1", ReviewStatus::Pending);
    save_review_item(&kb_root, &item).expect("save review item");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("review")
        .arg("reject")
        .arg("review-reject-1")
        .arg("--reason")
        .arg("not relevant");
    let output = cmd.output().expect("run kb review reject");
    assert!(
        output.status.success(),
        "kb review reject failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let path = kb_root.join("reviews/promotions/review-reject-1.json");
    let saved: Value =
        serde_json::from_str(&fs::read_to_string(&path).expect("read review")).expect("parse");
    assert_eq!(saved["status"], "rejected");
    assert!(saved["comment"].as_str().expect("comment string").contains("not relevant"));
}

#[test]
fn review_reject_json_output() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let item = make_review_item("review-reject-j1", ReviewStatus::Pending);
    save_review_item(&kb_root, &item).expect("save review item");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json")
        .arg("review")
        .arg("reject")
        .arg("review-reject-j1");
    let output = cmd.output().expect("run kb review reject --json");
    assert!(output.status.success());

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse json");
    assert_eq!(envelope["command"], "review.reject");
    assert_eq!(envelope["data"]["action"], "rejected");
}

#[test]
fn review_list_empty_queue() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("review").arg("list");
    let output = cmd.output().expect("run kb review list");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("No pending review items"));
}

#[test]
fn approved_promotion_passes_orphan_lint_with_real_source_ids() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // Seed a normalized source that the retrieval plan pointed at. The lint
    // orphans check looks for `normalized/<doc_id>/metadata.json`.
    let doc_id = "wiki/sources/rust-overview.md";
    let normalized_dir = kb_root.join("normalized").join(doc_id);
    fs::create_dir_all(&normalized_dir).expect("create normalized dir");
    fs::write(
        normalized_dir.join("metadata.json"),
        "{\"source_revision_id\":\"rev-1\"}",
    )
    .expect("write normalized metadata");

    // Also create the wiki page itself so the kb tree is well-formed.
    fs::create_dir_all(kb_root.join("wiki/sources")).expect("create wiki sources");
    fs::write(
        kb_root.join(doc_id),
        "---\nsource_document_id: wiki/sources/rust-overview.md\nsource_revision_id: rev-1\n---\n# Rust Overview\n",
    )
    .expect("write source wiki page");

    // Seed the answer artifact that `execute_promotion` will read. Its frontmatter
    // carries the real `source_document_ids` (as `kb_query::write_artifact` would).
    let question_id = "question-promote-orphan";
    let artifact_id = "artifact-promote-orphan";
    let q_dir = kb_root.join("outputs/questions").join(question_id);
    fs::create_dir_all(&q_dir).expect("create question dir");
    fs::write(
        q_dir.join("answer.md"),
        format!(
            "---\nid: {artifact_id}\ntype: question_answer\nquestion_id: {question_id}\nsource_document_ids:\n- {doc_id}\n---\n\nRust is a systems programming language.\n"
        ),
    )
    .expect("write answer.md");

    // Seed the ReviewItem with dependencies = [question_id, artifact_id] so we
    // exercise the same dependency-split logic the real CLI uses.
    let review_item = ReviewItem {
        metadata: EntityMetadata {
            id: "review-promote-orphan".to_string(),
            created_at_millis: 1_000,
            updated_at_millis: 1_000,
            source_hashes: vec!["hash-1".to_string()],
            model_version: None,
            tool_version: Some("kb-test".to_string()),
            prompt_template_hash: None,
            dependencies: vec![question_id.to_string(), artifact_id.to_string()],
            output_paths: vec![
                PathBuf::from(format!("outputs/questions/{question_id}/answer.md")),
                PathBuf::from("wiki/questions/rust-overview-answer.md"),
            ],
            status: Status::NeedsReview,
        },
        kind: ReviewKind::Promotion,
        target_entity_id: artifact_id.to_string(),
        proposed_destination: Some(PathBuf::from("wiki/questions/rust-overview-answer.md")),
        citations: vec![format!("{doc_id}#intro")],
        affected_pages: vec![PathBuf::from("wiki/questions/rust-overview-answer.md")],
        created_at_millis: 1_000,
        status: ReviewStatus::Pending,
        comment: "Promote answer for: What is Rust?".to_string(),
    };
    save_review_item(&kb_root, &review_item).expect("save review item");

    // Approve -> triggers execute_promotion.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("review").arg("approve").arg("review-promote-orphan");
    let output = cmd.output().expect("run kb review approve");
    assert!(
        output.status.success(),
        "kb review approve failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let promoted_path = kb_root.join("wiki/questions/rust-overview-answer.md");
    let promoted = fs::read_to_string(&promoted_path).expect("read promoted page");
    assert!(
        promoted.contains(&format!("- {doc_id}")),
        "promoted page missing real source id: {promoted}"
    );
    assert!(
        promoted.contains("derived_from:"),
        "promoted page missing derived_from block: {promoted}"
    );
    // source_document_ids block in the frontmatter must not list the question/artifact IDs.
    let fm_block = promoted
        .strip_prefix("---\n")
        .and_then(|rest| rest.split_once("\n---\n").map(|(fm, _)| fm))
        .expect("frontmatter block");
    let parsed: Value = serde_yaml::from_str(fm_block).expect("parse frontmatter");
    let sources = parsed["source_document_ids"]
        .as_array()
        .expect("source_document_ids array");
    let source_strs: Vec<&str> = sources.iter().filter_map(Value::as_str).collect();
    assert!(!source_strs.iter().any(|s| s.starts_with("question-")));
    assert!(!source_strs.iter().any(|s| s.starts_with("artifact-")));
    assert!(source_strs.contains(&doc_id));

    // Now run kb lint --check orphans. It must pass.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json").arg("lint").arg("--check").arg("orphans");
    let output = cmd.output().expect("run kb lint");
    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse lint json");
    let checks = envelope["data"]["checks"].as_array().expect("checks");
    let orphans = checks
        .iter()
        .find(|c| c["check"] == "orphans")
        .expect("orphans check entry");
    assert_eq!(
        orphans["issue_count"], 0,
        "promoted page should produce zero orphan findings: {}",
        serde_json::to_string_pretty(orphans).unwrap_or_default()
    );
}
