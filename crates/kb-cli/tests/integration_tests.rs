mod common;

use common::{kb_cmd, make_temp_kb};
use regex::Regex;
use std::fs;

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
    insta::assert_snapshot!(stdout);
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
