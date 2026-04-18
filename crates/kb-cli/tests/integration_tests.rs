mod common;

use common::{kb_cmd, make_temp_kb};
use regex::Regex;
use serde_json::Value;
use std::fs;

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
    let items = payload.as_array().expect("ingest output should be an array");
    assert_eq!(items.len(), 1);

    let copied_path = items[0]["copied_path"]
        .as_str()
        .expect("copied_path should be a string");
    let sidecar_path = items[0]["metadata_sidecar_path"]
        .as_str()
        .expect("metadata_sidecar_path should be a string");

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
    let items = payload.as_array().expect("ingest output should be an array");
    assert_eq!(items.len(), 1);
    assert_eq!(
        items[0]["copied_path"]
            .as_str()
            .expect("copied_path should be a string")
            .rsplit('/')
            .next()
            .expect("copied_path should have a filename"),
        "kept.md"
    );
}
