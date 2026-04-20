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
fn init_force_preserves_existing_kb_toml() {
    // bn-2so: `kb init --force` must preserve a user-customized kb.toml so
    // re-running init (e.g. after being told to by a manifest-corruption
    // error) does not wipe publish targets or runner overrides.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let config_path = kb_root.join("kb.toml");
    let original = fs::read_to_string(&config_path).expect("read kb.toml");
    let customized =
        format!("{original}\n[publish.targets.test]\npath = \"/x\"\n");
    fs::write(&config_path, &customized).expect("write customized kb.toml");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("init").arg("--force");
    let output = cmd.output().expect("run kb init --force");
    assert!(
        output.status.success(),
        "kb init --force failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let after = fs::read_to_string(&config_path).expect("read kb.toml after --force");
    assert_eq!(
        after, customized,
        "kb init --force must preserve the existing kb.toml byte-for-byte"
    );
    assert!(
        after.contains("[publish.targets.test]"),
        "publish target section must still be present after --force"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("kb.toml preserved"),
        "stdout should acknowledge preservation; got: {stdout}"
    );

    // State scaffolding must still be regenerated under --force.
    assert!(
        kb_root.join("state/manifest.json").exists(),
        "manifest.json must be (re)created under --force"
    );
}

#[test]
fn init_force_reset_config_overwrites_kb_toml() {
    // bn-2so: explicit escape hatch when a user does want a fresh config.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let config_path = kb_root.join("kb.toml");
    let customized = "# custom marker\n[publish.targets.test]\npath = \"/x\"\n";
    fs::write(&config_path, customized).expect("write customized kb.toml");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("init").arg("--force").arg("--reset-config");
    let output = cmd
        .output()
        .expect("run kb init --force --reset-config");
    assert!(
        output.status.success(),
        "kb init --force --reset-config failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let after = fs::read_to_string(&config_path).expect("read kb.toml after reset");
    assert!(
        !after.contains("# custom marker"),
        "--reset-config must overwrite the customized kb.toml; got: {after}"
    );
    assert!(
        !after.contains("[publish.targets.test]"),
        "--reset-config must drop custom publish targets; got: {after}"
    );
}

#[test]
fn init_force_replaces_unparseable_kb_toml() {
    // bn-2so: if the existing kb.toml does not parse, --force still writes a
    // fresh one so the KB becomes usable again.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let config_path = kb_root.join("kb.toml");
    fs::write(&config_path, "this is not = valid = toml [[[").expect("write broken kb.toml");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("init").arg("--force");
    let output = cmd.output().expect("run kb init --force on broken config");
    assert!(
        output.status.success(),
        "kb init --force failed on broken config: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let after = fs::read_to_string(&config_path).expect("read kb.toml after repair");
    assert!(
        !after.contains("this is not = valid = toml"),
        "broken kb.toml must be replaced by --force; got: {after}"
    );
    // Sanity: the new file parses as TOML.
    toml::from_str::<toml::Value>(&after).expect("regenerated kb.toml parses");
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

#[test]
fn ingest_skips_empty_file_with_warning_and_exit_zero() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // Zero-byte file.
    let empty = kb_root.join("empty.md");
    fs::write(&empty, b"").expect("write empty");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("ingest").arg(&empty);
    let output = cmd.output().expect("run kb ingest on empty");

    assert!(
        output.status.success(),
        "kb ingest on empty file should exit 0, got stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("warning: skipping empty source"),
        "expected emptiness warning on stderr, got: {stderr}"
    );
    assert!(
        !kb_root.join("normalized").exists()
            || fs::read_dir(kb_root.join("normalized"))
                .expect("read normalized dir")
                .next()
                .is_none(),
        "no normalized/ entries should be created for an empty source"
    );
}

/// After a clean empty-file skip, neither `kb status` nor `kb doctor` should
/// flag any interrupted job runs. Regression guard for bn-36x: the bn-40r
/// empty-skip path used to leave the ingest job manifest in its initial
/// `Running` state so the stale-job reaper later relabeled it "interrupted",
/// polluting both `kb status` and `kb doctor` forever.
#[test]
fn ingest_empty_file_does_not_leave_interrupted_job() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    let fake_bin = install_fake_harnesses(&kb_root);

    let empty = kb_root.join("empty.md");
    fs::write(&empty, b"").expect("write empty");

    let mut ingest_cmd = kb_cmd(&kb_root);
    ingest_cmd.arg("ingest").arg(&empty);
    let ingest_output = ingest_cmd.output().expect("run kb ingest on empty");
    assert!(
        ingest_output.status.success(),
        "kb ingest on empty file should exit 0, got stderr: {}",
        String::from_utf8_lossy(&ingest_output.stderr)
    );

    // `kb status --json` must report zero interrupted job runs after the
    // cleanly-skipped empty ingest.
    let mut status_cmd = kb_cmd(&kb_root);
    status_cmd.arg("--json").arg("status");
    let status_output = status_cmd.output().expect("run kb status");
    assert!(
        status_output.status.success(),
        "kb status should exit 0, got stderr: {}",
        String::from_utf8_lossy(&status_output.stderr)
    );
    let status_envelope: Value =
        serde_json::from_slice(&status_output.stdout).expect("parse status json");
    let interrupted = status_envelope["data"]["interrupted_jobs"]
        .as_array()
        .expect("interrupted_jobs array");
    assert!(
        interrupted.is_empty(),
        "empty-file ingest must not leave an interrupted job run, got: {interrupted:?}"
    );

    // The ingest job itself should be recorded as succeeded.
    let recent = status_envelope["data"]["recent_jobs"]
        .as_array()
        .expect("recent_jobs array");
    let ingest_jobs: Vec<&Value> = recent
        .iter()
        .filter(|job| job["command"] == "ingest")
        .collect();
    assert_eq!(
        ingest_jobs.len(),
        1,
        "exactly one ingest job expected, got: {ingest_jobs:?}"
    );
    assert_eq!(
        ingest_jobs[0]["status"], "succeeded",
        "empty-file ingest job should be marked succeeded, got: {:?}",
        ingest_jobs[0]
    );
    assert_eq!(ingest_jobs[0]["exit_code"], 0);

    // `kb doctor` must exit 0 with no interrupted-job warning.
    let mut doctor_cmd = kb_cmd(&kb_root);
    doctor_cmd.env("PATH", prepend_path(&fake_bin));
    doctor_cmd.arg("--json").arg("doctor");
    let doctor_output = doctor_cmd.output().expect("run kb doctor");
    assert!(
        doctor_output.status.success(),
        "kb doctor should exit 0 after empty-file ingest, got stderr: {}",
        String::from_utf8_lossy(&doctor_output.stderr)
    );
    let doctor_envelope: Value =
        serde_json::from_slice(&doctor_output.stdout).expect("parse doctor json");
    let checks = doctor_envelope["data"]["checks"]
        .as_array()
        .expect("checks array");
    let interrupted_check = checks
        .iter()
        .find(|check| check["name"] == "interrupted_jobs")
        .expect("interrupted_jobs check present");
    assert_eq!(
        interrupted_check["status"], "ok",
        "interrupted_jobs check should be OK, got: {interrupted_check:?}"
    );
}

#[test]
fn ingest_skips_frontmatter_only_file() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let fm = kb_root.join("only-fm.md");
    fs::write(&fm, "---\ntitle: x\n---\n").expect("write fm-only");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("ingest").arg(&fm);
    let output = cmd.output().expect("run kb ingest on fm-only");

    assert!(output.status.success(), "should exit 0");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("warning: skipping empty source"),
        "expected emptiness warning, got: {stderr}"
    );
}

#[test]
fn ingest_allow_empty_accepts_zero_byte_file() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let empty = kb_root.join("placeholder.md");
    fs::write(&empty, b"").expect("write placeholder");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json")
        .arg("ingest")
        .arg("--allow-empty")
        .arg(&empty);
    let output = cmd.output().expect("run kb ingest --allow-empty");

    assert!(
        output.status.success(),
        "kb ingest --allow-empty should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse ingest json");
    let items = envelope["data"]["results"]
        .as_array()
        .expect("results array");
    assert_eq!(items.len(), 1, "empty file should be ingested with --allow-empty");
}

#[test]
fn ingest_mixed_directory_keeps_non_empty_and_warns_on_empty() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let corpus = kb_root.join("corpus");
    fs::create_dir_all(&corpus).expect("mkdir");
    fs::write(corpus.join("good.md"), "# real\n\ncontent\n").expect("write good");
    fs::write(corpus.join("empty.md"), b"").expect("write empty");
    fs::write(corpus.join("fm-only.md"), "---\ntitle: x\n---\n").expect("write fm-only");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json").arg("ingest").arg(&corpus);
    let output = cmd.output().expect("run mixed ingest");

    assert!(
        output.status.success(),
        "mixed ingest should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.matches("warning: skipping empty source").count() >= 2,
        "expected two emptiness warnings (empty.md + fm-only.md), got: {stderr}"
    );
    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse json");
    let items = envelope["data"]["results"]
        .as_array()
        .expect("results array");
    assert_eq!(items.len(), 1, "only good.md should be ingested");
    let content_path = items[0]["content_path"]
        .as_str()
        .expect("content_path str");
    assert!(content_path.ends_with("good.md"), "got: {content_path}");
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
    // I5: citations are parsed clean of their list-marker prefix so the
    // inspect renderer does not emit "- - [[...]]".
    assert_eq!(payload["citations"][0], "[[wiki/sources/example.md]]");
    assert_eq!(payload["build_records"][0]["id"], "build-index");
    assert_eq!(payload["trace"][0]["id"], "wiki/sources/example.md");
}

#[test]
fn ask_rejects_empty_argument() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("ask").arg("");
    let output = cmd.output().expect("run kb ask with empty argument");

    assert!(
        !output.status.success(),
        "kb ask with empty argument unexpectedly succeeded"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("empty"),
        "expected stderr to mention 'empty', got: {stderr}"
    );
}

#[test]
fn ask_rejects_whitespace_only_argument() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("ask").arg("   ");
    let output = cmd
        .output()
        .expect("run kb ask with whitespace-only argument");

    assert!(
        !output.status.success(),
        "kb ask with whitespace-only argument unexpectedly succeeded"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("empty"),
        "expected stderr to mention 'empty', got: {stderr}"
    );
}

#[test]
fn ask_stdin_empty_still_errors() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("ask").arg("-").write_stdin("");
    let output = cmd.output().expect("run kb ask - with empty stdin");

    assert!(
        !output.status.success(),
        "kb ask - with empty stdin unexpectedly succeeded"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no question provided"),
        "expected stderr to mention 'no question provided', got: {stderr}"
    );
}

#[test]
fn ask_format_png_refuses_cleanly() {
    // Regression (bn-iiq): `--format png` used to silently produce `answer.md`.
    // It now errors with a clear "not yet supported" message and writes no files.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("ask")
        .arg("--format")
        .arg("png")
        .arg("What is testing?");
    let output = cmd.output().expect("run kb ask --format=png");

    assert!(
        !output.status.success(),
        "kb ask --format=png unexpectedly succeeded"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--format png is not yet supported"),
        "expected stderr to mention png-not-supported, got: {stderr}"
    );
    assert!(
        stderr.contains("md, marp, json"),
        "expected stderr to list supported formats, got: {stderr}"
    );

    // No question directory should have been created.
    let outputs_dir = kb_root.join("outputs/questions");
    if outputs_dir.exists() {
        let count = fs::read_dir(&outputs_dir)
            .expect("read outputs/questions")
            .filter_map(Result::ok)
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with("question-")
            })
            .count();
        assert_eq!(
            count, 0,
            "failed --format=png run should not create question artifacts"
        );
    }
}

#[test]
fn ask_format_md_writes_markdown_answer() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json")
        .arg("ask")
        .arg("--format")
        .arg("md")
        .arg("What is md?");
    let output = cmd.output().expect("run kb ask --format=md");
    assert!(
        output.status.success(),
        "kb ask --format=md failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse ask json");
    let artifact_path = kb_root.join(
        envelope["data"]["artifact_path"]
            .as_str()
            .expect("artifact_path"),
    );
    assert!(
        artifact_path.extension().and_then(|e| e.to_str()) == Some("md"),
        "md format should produce .md file, got: {}",
        artifact_path.display()
    );
    let content = fs::read_to_string(&artifact_path).expect("read answer");
    assert!(content.starts_with("---\n"), "md answer should have YAML frontmatter");
    assert!(content.contains("requested_format: md"));
}

#[test]
fn ask_format_marp_writes_markdown_with_marp_flag() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json")
        .arg("ask")
        .arg("--format")
        .arg("marp")
        .arg("What is marp?");
    let output = cmd.output().expect("run kb ask --format=marp");
    assert!(
        output.status.success(),
        "kb ask --format=marp failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse ask json");
    let artifact_path = kb_root.join(
        envelope["data"]["artifact_path"]
            .as_str()
            .expect("artifact_path"),
    );
    // Marp IS markdown; keeps the .md extension.
    assert_eq!(
        artifact_path.extension().and_then(|e| e.to_str()),
        Some("md"),
        "marp format should produce .md file, got: {}",
        artifact_path.display()
    );
    let content = fs::read_to_string(&artifact_path).expect("read answer");
    assert!(content.contains("marp: true"), "marp answer should set marp: true");
    assert!(content.contains("requested_format: marp"));
}

#[test]
fn ask_format_json_writes_structured_artifact() {
    // Regression (bn-iiq): `--format json` used to silently produce `answer.md`
    // with a lying `requested_format: json` frontmatter line. It now emits a
    // structured `answer.json` artifact.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json")
        .arg("ask")
        .arg("--format")
        .arg("json")
        .arg("What is json?");
    let output = cmd.output().expect("run kb ask --format=json");
    assert!(
        output.status.success(),
        "kb ask --format=json failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse ask json");
    let artifact_path = kb_root.join(
        envelope["data"]["artifact_path"]
            .as_str()
            .expect("artifact_path"),
    );
    assert_eq!(
        artifact_path.extension().and_then(|e| e.to_str()),
        Some("json"),
        "json format should produce .json file, got: {}",
        artifact_path.display()
    );

    // The file must parse as JSON and carry the documented fields.
    let raw = fs::read_to_string(&artifact_path).expect("read answer.json");
    let answer: Value = serde_json::from_str(&raw).expect("answer.json is valid JSON");
    assert_eq!(answer["type"], "question_answer");
    assert_eq!(answer["requested_format"], "json");
    assert!(answer["id"].is_string(), "id should be a string");
    assert!(answer["question_id"].is_string(), "question_id should be a string");
    assert!(answer["generated_at"].is_number(), "generated_at should be a number");
    assert!(answer["body"].is_string(), "body should be a string");
    assert!(answer["source_document_ids"].is_array());
    assert!(answer["retrieval_candidates"].is_array());
    assert!(answer["citations"].is_array());

    // No stray answer.md sitting next to answer.json.
    let stray_md = artifact_path.with_file_name("answer.md");
    assert!(
        !stray_md.exists(),
        "json format must not also write answer.md at {}",
        stray_md.display()
    );
}

#[test]
fn inspect_rejects_empty_target() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("inspect").arg("");
    let output = cmd.output().expect("run kb inspect with empty target");

    assert!(
        !output.status.success(),
        "kb inspect with empty target unexpectedly succeeded"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("empty"),
        "expected stderr to mention 'empty', got: {stderr}"
    );
}

#[test]
fn inspect_rejects_whitespace_only_target() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("inspect").arg("   ");
    let output = cmd
        .output()
        .expect("run kb inspect with whitespace-only target");

    assert!(
        !output.status.success(),
        "kb inspect with whitespace-only target unexpectedly succeeded"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("empty"),
        "expected stderr to mention 'empty', got: {stderr}"
    );
}

#[test]
fn search_rejects_empty_query() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("search").arg("");
    let output = cmd.output().expect("run kb search with empty query");

    assert!(
        !output.status.success(),
        "kb search with empty query unexpectedly succeeded"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("empty"),
        "expected stderr to mention 'empty', got: {stderr}"
    );
}

#[test]
fn search_rejects_whitespace_only_query() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("search").arg("   ");
    let output = cmd
        .output()
        .expect("run kb search with whitespace-only query");

    assert!(
        !output.status.success(),
        "kb search with whitespace-only query unexpectedly succeeded"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("empty"),
        "expected stderr to mention 'empty', got: {stderr}"
    );
}

#[test]
fn search_without_matches_after_compile_omits_compile_tip() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    write_source_page(
        &kb_root,
        "rust-overview",
        "Rust Overview",
        "An introduction to the Rust programming language.",
    );

    let mut compile_cmd = kb_cmd(&kb_root);
    compile_cmd.arg("compile");
    let compile_output = compile_cmd.output().expect("run kb compile");
    assert!(
        compile_output.status.success(),
        "kb compile failed: {}",
        String::from_utf8_lossy(&compile_output.stderr)
    );

    // Confirm the lexical index was written.
    assert!(
        kb_root.join("state/indexes/lexical.json").exists(),
        "compile should have produced a lexical index"
    );

    let mut search_cmd = kb_cmd(&kb_root);
    search_cmd.arg("search").arg("nonexistentqueryvalue");
    let output = search_cmd
        .output()
        .expect("run kb search for a term with no matches");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("No results"),
        "expected zero-result message, got: {stdout}"
    );
    assert!(
        !stdout.contains("run 'kb compile'"),
        "should not suggest running compile after a successful compile: {stdout}"
    );
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

// I1: bare `src-<hex>` identifier resolves to the wiki/sources page.
#[test]
fn inspect_resolves_bare_src_id_to_wiki_sources_page() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let sources_dir = kb_root.join("wiki/sources");
    fs::create_dir_all(&sources_dir).expect("create wiki/sources");
    fs::write(
        sources_dir.join("src-302b46ff.md"),
        "---\nid: wiki-source-src-302b46ff\nsource_document_id: src-302b46ff\nsource_revision_id: rev-1\n---\n\n# Src\n",
    )
    .expect("write source page");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("inspect").arg("src-302b46ff");
    let output = cmd.output().expect("run kb inspect src-302b46ff");

    assert!(
        output.status.success(),
        "kb inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("resolved_id: wiki/sources/src-302b46ff.md"),
        "expected resolved_id to point at wiki/sources page, got:\n{stdout}"
    );
}

// I1 fallback: bare `src-<hex>` identifier with no wiki page falls back to
// the normalized/<id>/source.md file.
#[test]
fn inspect_resolves_bare_src_id_to_normalized_dir() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let normalized_dir = kb_root.join("normalized/src-deadbeef");
    fs::create_dir_all(&normalized_dir).expect("create normalized dir");
    fs::write(normalized_dir.join("source.md"), "# Only normalized\n")
        .expect("write source.md");
    fs::write(
        normalized_dir.join("metadata.json"),
        r#"{"metadata":{"id":"src-deadbeef","created_at_millis":0,"updated_at_millis":0,"source_hashes":[],"dependencies":[],"output_paths":[],"status":"fresh"},"source_revision_id":"rev-1","normalized_assets":[],"heading_ids":[]}"#,
    )
    .expect("write metadata.json");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("inspect").arg("src-deadbeef");
    let output = cmd.output().expect("run kb inspect src-deadbeef");

    assert!(
        output.status.success(),
        "kb inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("resolved_id: normalized/src-deadbeef/source.md"),
        "expected resolved_id to point at normalized source, got:\n{stdout}"
    );
}

// I2: absolute paths outside the KB root must error.
#[test]
fn inspect_rejects_paths_outside_kb_root() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("inspect").arg("/etc/hosts");
    let output = cmd
        .output()
        .expect("run kb inspect on absolute path outside root");

    assert!(
        !output.status.success(),
        "kb inspect on /etc/hosts unexpectedly succeeded: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("outside the KB root"),
        "expected 'outside the KB root' in stderr, got: {stderr}"
    );
}

// I3: source-wiki page freshness reflects the normalized document's current
// source_revision_id.
#[test]
fn inspect_reports_fresh_when_source_revision_matches_normalized() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let sources_dir = kb_root.join("wiki/sources");
    fs::create_dir_all(&sources_dir).expect("create wiki/sources");
    fs::write(
        sources_dir.join("src-abcd1234.md"),
        "---\nid: wiki-source-src-abcd1234\nsource_document_id: src-abcd1234\nsource_revision_id: rev-match\n---\n\n# Src\n",
    )
    .expect("write source page");

    let normalized_dir = kb_root.join("normalized/src-abcd1234");
    fs::create_dir_all(&normalized_dir).expect("create normalized dir");
    fs::write(normalized_dir.join("source.md"), "# body\n").expect("write source");
    fs::write(
        normalized_dir.join("metadata.json"),
        r#"{"metadata":{"id":"src-abcd1234","created_at_millis":0,"updated_at_millis":0,"source_hashes":[],"dependencies":[],"output_paths":[],"status":"fresh"},"source_revision_id":"rev-match","normalized_assets":[],"heading_ids":[]}"#,
    )
    .expect("write metadata");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("inspect").arg("src-abcd1234");
    let output = cmd.output().expect("run kb inspect src-abcd1234");
    assert!(
        output.status.success(),
        "kb inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("freshness: fresh"),
        "expected fresh, got:\n{stdout}"
    );
}

#[test]
fn inspect_reports_stale_when_source_revision_diverges() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let sources_dir = kb_root.join("wiki/sources");
    fs::create_dir_all(&sources_dir).expect("create wiki/sources");
    fs::write(
        sources_dir.join("src-aaaa1111.md"),
        "---\nid: wiki-source-src-aaaa1111\nsource_document_id: src-aaaa1111\nsource_revision_id: rev-old\n---\n\n# Src\n",
    )
    .expect("write source page");

    let normalized_dir = kb_root.join("normalized/src-aaaa1111");
    fs::create_dir_all(&normalized_dir).expect("create normalized dir");
    fs::write(normalized_dir.join("source.md"), "# body\n").expect("write source");
    fs::write(
        normalized_dir.join("metadata.json"),
        r#"{"metadata":{"id":"src-aaaa1111","created_at_millis":0,"updated_at_millis":0,"source_hashes":[],"dependencies":[],"output_paths":[],"status":"fresh"},"source_revision_id":"rev-new","normalized_assets":[],"heading_ids":[]}"#,
    )
    .expect("write metadata");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("inspect").arg("src-aaaa1111");
    let output = cmd.output().expect("run kb inspect src-aaaa1111");
    assert!(
        output.status.success(),
        "kb inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("freshness: stale"),
        "expected stale, got:\n{stdout}"
    );
}

// I4: build records whose metadata.output_paths name the inspected file are
// surfaced, even when output_ids is empty.
#[test]
fn inspect_surfaces_build_records_matched_by_output_paths() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let concepts_dir = kb_root.join("wiki/concepts");
    fs::create_dir_all(&concepts_dir).expect("create wiki/concepts");
    fs::write(
        concepts_dir.join("tokio.md"),
        "---\nid: concept:tokio\n---\n\n# Tokio\n",
    )
    .expect("write concept page");

    let mut record_metadata = test_metadata("build-concept-tokio");
    record_metadata.output_paths = vec![PathBuf::from("wiki/concepts/tokio.md")];
    save_build_record(
        &kb_root,
        &BuildRecord {
            metadata: record_metadata,
            pass_name: "concept_extraction".to_string(),
            input_ids: vec!["wiki/sources/example.md".to_string()],
            output_ids: Vec::new(),
            manifest_hash: "manifest-concept".to_string(),
        },
    )
    .expect("save build record");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("inspect").arg("wiki/concepts/tokio.md");
    let output = cmd.output().expect("run kb inspect concept");
    assert!(
        output.status.success(),
        "kb inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("id: build-concept-tokio"),
        "expected build-concept-tokio in build records, got:\n{stdout}"
    );
}

// I5: the citations section must not emit double-bullet artifacts when the
// source markdown already includes a `- ` list prefix.
#[test]
fn inspect_citations_do_not_double_bullet() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let sources_dir = kb_root.join("wiki/sources");
    fs::create_dir_all(&sources_dir).expect("create wiki/sources");
    fs::write(
        sources_dir.join("src-cafe9999.md"),
        "---\nid: wiki-source-src-cafe9999\nsource_document_id: src-cafe9999\nsource_revision_id: rev-1\n---\n\n## Citations\n\n- [[wiki/sources/src-cafe9999]]\n",
    )
    .expect("write source page");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("inspect").arg("src-cafe9999");
    let output = cmd.output().expect("run kb inspect src-cafe9999");
    assert!(
        output.status.success(),
        "kb inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.contains("- - [["),
        "citations should not double-bullet, got:\n{stdout}"
    );
    assert!(
        stdout.contains("- [[wiki/sources/src-cafe9999]]"),
        "expected single-bullet citation, got:\n{stdout}"
    );
}

// L1: `kb inspect wiki/sources/<id>` (no `.md`) should resolve to the
// on-disk `.md` file — users follow the Obsidian/wiki convention of
// omitting the extension.
#[test]
fn inspect_accepts_wiki_sources_path_without_md_extension() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let sources_dir = kb_root.join("wiki/sources");
    fs::create_dir_all(&sources_dir).expect("create wiki/sources");
    fs::write(
        sources_dir.join("src-0e2e3f8b.md"),
        "---\nid: wiki-source-src-0e2e3f8b\nsource_document_id: src-0e2e3f8b\nsource_revision_id: rev-1\n---\n\n# Src\n",
    )
    .expect("write source page");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("inspect").arg("wiki/sources/src-0e2e3f8b");
    let output = cmd
        .output()
        .expect("run kb inspect wiki/sources/src-0e2e3f8b");

    assert!(
        output.status.success(),
        "kb inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("resolved_id: wiki/sources/src-0e2e3f8b.md"),
        "expected resolved_id to include .md suffix, got:\n{stdout}"
    );
    assert!(
        stdout.contains("kind: wiki_page"),
        "expected kind: wiki_page, got:\n{stdout}"
    );
}

// L1: concept pages should resolve the same way.
#[test]
fn inspect_accepts_wiki_concepts_path_without_md_extension() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let concepts_dir = kb_root.join("wiki/concepts");
    fs::create_dir_all(&concepts_dir).expect("create wiki/concepts");
    fs::write(concepts_dir.join("tokio.md"), "# Tokio\n").expect("write concept");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("inspect").arg("wiki/concepts/tokio");
    let output = cmd.output().expect("run kb inspect wiki/concepts/tokio");

    assert!(
        output.status.success(),
        "kb inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("resolved_id: wiki/concepts/tokio.md"),
        "expected resolved_id to include .md suffix, got:\n{stdout}"
    );
}

// L2: inspecting a directory (e.g. `raw/inbox`) should classify the kind as
// `directory` rather than inheriting the `source_revision` prefix heuristic.
#[test]
fn inspect_directory_reports_directory_kind() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // `kb init` creates `raw/inbox/` as a directory.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("inspect").arg("raw/inbox");
    let output = cmd.output().expect("run kb inspect raw/inbox");

    assert!(
        output.status.success(),
        "kb inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("kind: directory"),
        "expected kind: directory for raw/inbox, got:\n{stdout}"
    );
    assert!(
        !stdout.contains("kind: source_revision"),
        "raw/inbox should not be classified source_revision, got:\n{stdout}"
    );
}

// L2 (regression): an actual revision file under raw/inbox should still be
// classified `source_revision`.
#[test]
fn inspect_source_revision_file_still_reports_source_revision_kind() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let rev_dir = kb_root.join("raw/inbox/src-abc12345/rev-1");
    fs::create_dir_all(&rev_dir).expect("create revision dir");
    fs::write(rev_dir.join("document.md"), "# doc\n").expect("write revision file");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("inspect")
        .arg("raw/inbox/src-abc12345/rev-1/document.md");
    let output = cmd.output().expect("run kb inspect revision file");

    assert!(
        output.status.success(),
        "kb inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("kind: source_revision"),
        "expected kind: source_revision for revision file, got:\n{stdout}"
    );
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

// bn-k79: promoted question pages must be surfaced in both the top-level
// wiki/index.md (as a dedicated Questions section) and a per-category
// wiki/questions/index.md, mirroring the existing sources/concepts layout.
#[test]
fn compile_lists_promoted_questions_in_indexes() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // A source so the regular index sections aren't empty — isolates the
    // assertion to the new Questions behavior.
    write_source_page(
        &kb_root,
        "rust-book",
        "The Rust Programming Language",
        "Memory safety.",
    );

    // Two promoted question pages. Test both `title:` and `question:`
    // frontmatter keys since real promotion pipelines use either.
    let questions_dir = kb_root.join("wiki/questions");
    fs::create_dir_all(&questions_dir).expect("mkdir wiki/questions");
    fs::write(
        questions_dir.join("what-is-rust.md"),
        "---\nid: q-what-is-rust\ntype: question_answer\ntitle: What is Rust?\n---\n\n## Answer\nRust is a systems language.\n",
    )
    .expect("write promoted question a");
    fs::write(
        questions_dir.join("why-borrow-checker.md"),
        "---\nid: q-why-borrow-checker\ntype: question_answer\nquestion: Why a borrow checker?\n---\n\n## Answer\nMemory safety without GC.\n",
    )
    .expect("write promoted question b");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("compile");
    let output = cmd.output().expect("run kb compile");
    assert!(
        output.status.success(),
        "kb compile failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // wiki/questions/index.md exists and lists both questions with
    // file-relative links (bare filenames from the category dir).
    let questions_index =
        fs::read_to_string(kb_root.join("wiki/questions/index.md")).expect("read questions index");
    assert!(
        questions_index.contains("# Questions"),
        "questions index missing header: {questions_index}"
    );
    assert!(
        questions_index.contains("[What is Rust?]")
            && questions_index.contains("(what-is-rust.md)"),
        "questions index missing first entry: {questions_index}"
    );
    assert!(
        questions_index.contains("[Why a borrow checker?]")
            && questions_index.contains("(why-borrow-checker.md)"),
        "questions index missing second entry (question: fallback): {questions_index}"
    );

    // Global wiki/index.md gains a ## Questions section with links
    // relative to wiki/ (i.e., questions/<slug>.md).
    let global = fs::read_to_string(kb_root.join("wiki/index.md")).expect("read global index");
    assert!(
        global.contains("## Questions"),
        "global index missing Questions section: {global}"
    );
    assert!(
        global.contains("2 question(s)"),
        "global index missing question count: {global}"
    );
    assert!(
        global.contains("(questions/what-is-rust.md)"),
        "global index missing file-relative link to first question: {global}"
    );
    assert!(
        global.contains("(questions/why-borrow-checker.md)"),
        "global index missing file-relative link to second question: {global}"
    );
}

// bn-k79: with zero promoted questions, keep the output tidy — no
// Questions section in wiki/index.md, no wiki/questions/index.md file.
#[test]
fn compile_without_questions_omits_questions_index() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    write_source_page(
        &kb_root,
        "rust-book",
        "The Rust Programming Language",
        "Memory safety.",
    );

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("compile");
    let output = cmd.output().expect("run kb compile");
    assert!(
        output.status.success(),
        "kb compile failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(
        !kb_root.join("wiki/questions/index.md").exists(),
        "wiki/questions/index.md must not be emitted when there are no promoted questions"
    );
    let global = fs::read_to_string(kb_root.join("wiki/index.md")).expect("read global index");
    assert!(
        !global.contains("## Questions"),
        "global index must not contain Questions section when there are no questions: {global}"
    );
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

/// Regression test for bn-2sv: a syntactically broken kb.toml must cause every
/// command to fail upfront (not silently run with defaults on read-only commands).
/// `kb init --force` is the documented escape hatch and must still succeed.
#[test]
fn broken_kb_toml_fails_read_only_commands() {
    let (_temp, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // Corrupt kb.toml so the parser will reject it.
    let config_path = kb_root.join("kb.toml");
    fs::write(&config_path, "invalid toml ===").expect("write broken kb.toml");

    // `kb status` is a read-only command that previously ignored config errors.
    // It must now exit non-zero with an error naming the file and the parse error.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("status");
    let output = cmd.output().expect("run kb status");
    assert!(
        !output.status.success(),
        "kb status must fail with broken kb.toml; stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("kb.toml"),
        "stderr should name the offending file path, got: {stderr}"
    );
    // The TOML parser reports a line/column range like "1:15" for the bad token;
    // accept either the explicit locator or the underlying "invalid" description.
    assert!(
        stderr.contains("line")
            || stderr.contains("column")
            || stderr.contains("invalid")
            || stderr.contains("expected"),
        "stderr should include parse error details, got: {stderr}"
    );

    // `kb init --force` is the documented way to recover — it must succeed
    // even when the existing kb.toml is unparseable, because it overwrites it.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--force").arg("init");
    let output = cmd.output().expect("run kb init --force");
    assert!(
        output.status.success(),
        "kb init --force must succeed even with broken kb.toml; stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );

    // After `init --force`, the config is valid again and status works.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("status");
    let output = cmd.output().expect("run kb status after recovery");
    assert!(
        output.status.success(),
        "kb status should succeed after init --force; stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Fabricate `normalized/<src_id>/metadata.json` at the given revision,
/// plus a minimal `source.md`. Used by the bn-2gn re-ingest status test.
fn write_fake_normalized_source(kb_root: &Path, src_id: &str, revision: &str) {
    let normalized_dir = kb_root.join("normalized").join(src_id);
    fs::create_dir_all(&normalized_dir).expect("create normalized dir");
    let metadata = serde_json::json!({
        "metadata": {
            "id": src_id,
            "entity_type": "source-document",
            "display_name": src_id,
            "canonical_path": "inbox/fake.md",
            "content_hashes": [],
            "output_paths": [],
            "status": "Fresh",
        },
        "source_revision_id": revision,
        "normalized_assets": [],
        "heading_ids": [],
    });
    fs::write(normalized_dir.join("metadata.json"), metadata.to_string())
        .expect("write metadata.json");
    fs::write(normalized_dir.join("source.md"), "# body\n").expect("write source.md");
}

/// Fabricate a minimal `wiki/sources/<src_id>.md` with frontmatter pinned
/// to the given revision. Used by the bn-2gn re-ingest status test.
fn write_fake_wiki_source_page(kb_root: &Path, src_id: &str, revision: &str) {
    let wiki_dir = kb_root.join("wiki").join("sources");
    fs::create_dir_all(&wiki_dir).expect("create wiki/sources dir");
    let markdown = format!(
        "---\nid: wiki-source-{src_id}\ntype: source\ntitle: {src_id}\n\
source_document_id: {src_id}\nsource_revision_id: {revision}\n\
generated_at: 0\nbuild_record_id: build-1\n---\n\n# Source\n",
    );
    fs::write(wiki_dir.join(format!("{src_id}.md")), markdown).expect("write wiki page");
}

/// Fabricate a `state/hashes.json` that claims the given normalized ids were
/// all compiled at some fingerprint. Values don't matter — `kb status` only
/// checks for key presence to decide whether a source was ever compiled.
fn write_fake_hash_state(kb_root: &Path, src_ids: &[&str]) {
    let state_dir = kb_root.join("state");
    fs::create_dir_all(&state_dir).expect("create state dir");
    let mut hashes = serde_json::Map::new();
    for id in src_ids {
        hashes.insert(
            format!("normalized/{id}"),
            serde_json::Value::String(format!("fingerprint-{id}")),
        );
    }
    let body = serde_json::json!({ "hashes": serde_json::Value::Object(hashes) });
    fs::write(state_dir.join("hashes.json"), body.to_string())
        .expect("write state/hashes.json");
}

/// Run `kb --json status` and return the parsed `changed_inputs_not_compiled` array.
fn status_changed_inputs(kb_root: &Path) -> Vec<String> {
    let mut cmd = kb_cmd(kb_root);
    cmd.arg("--json").arg("status");
    let output = cmd.output().expect("run kb status");
    assert!(
        output.status.success(),
        "kb status must succeed; stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse status json");
    envelope["data"]["changed_inputs_not_compiled"]
        .as_array()
        .expect("changed_inputs_not_compiled array")
        .iter()
        .map(|v| {
            v.get("normalized_path")
                .and_then(Value::as_str)
                .expect("changed entry has normalized_path string")
                .to_string()
        })
        .collect()
}

/// Regression test for bn-2gn. When a user re-ingests a source whose content
/// changed, `normalized/<src>/metadata.json` gains a new `source_revision_id`
/// while `wiki/sources/<slug>.md` frontmatter is still pinned to the old
/// revision until `kb compile` runs. `kb status` must surface this under
/// "changed inputs not yet compiled" so the user knows the wiki is stale.
///
/// We fabricate the on-disk state directly (no `kb ingest`, no LLM) so the
/// test is deterministic and does not depend on the pipeline.
#[test]
fn kb_status_flags_reingested_source_with_stale_wiki_page() {
    let (_temp, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let src_id = "src-2gn";

    // Starting state: fully compiled at rev-A.
    write_fake_normalized_source(&kb_root, src_id, "rev-A");
    write_fake_wiki_source_page(&kb_root, src_id, "rev-A");
    write_fake_hash_state(&kb_root, &[src_id]);

    // Clean compile state → no changed inputs.
    let changed = status_changed_inputs(&kb_root);
    assert!(
        changed.is_empty(),
        "clean rev-A state should not flag any changed inputs, got: {changed:?}",
    );

    // Simulate a re-ingest: metadata bumps to rev-B, wiki page still at rev-A.
    write_fake_normalized_source(&kb_root, src_id, "rev-B");

    let changed = status_changed_inputs(&kb_root);
    assert_eq!(
        changed.len(),
        1,
        "re-ingested source must be flagged, got: {changed:?}",
    );
    assert!(
        changed[0].ends_with(&format!("normalized/{src_id}")),
        "reported path should point at normalized/{src_id}, got: {}",
        changed[0],
    );

    // After the wiki page is regenerated at rev-B (what `kb compile` does),
    // the list clears.
    write_fake_wiki_source_page(&kb_root, src_id, "rev-B");

    let changed = status_changed_inputs(&kb_root);
    assert!(
        changed.is_empty(),
        "after recompile, changed_inputs_not_compiled should clear, got: {changed:?}",
    );
}

// --- bn-2cr: kb status readability regressions -----------------------------

/// Seed a failed `JobRun` manifest directly under `state/jobs/` so we don't
/// have to actually provoke N different failure modes from the CLI. The
/// shape mirrors `jobs::load_job_run`'s serde expectations — if this drifts,
/// update alongside the `JobRun`/`EntityMetadata` structs.
fn seed_failed_job(root: &Path, id: &str, started_at_millis: u64) {
    let jobs_dir = root.join("state/jobs");
    fs::create_dir_all(&jobs_dir).expect("create state/jobs");
    let manifest = serde_json::json!({
        "metadata": {
            "id": id,
            "created_at_millis": started_at_millis,
            "updated_at_millis": started_at_millis,
            "source_hashes": [],
            "model_version": null,
            "tool_version": "kb-cli/0.1.0",
            "prompt_template_hash": null,
            "dependencies": [],
            "output_paths": [],
            "status": "fresh"
        },
        "command": "ingest",
        "root_path": root,
        "started_at_millis": started_at_millis,
        "ended_at_millis": started_at_millis + 1,
        "status": "failed",
        "log_path": null,
        "affected_outputs": [],
        "pid": 12_345,
        "exit_code": 1
    });
    fs::write(
        jobs_dir.join(format!("{id}.json")),
        manifest.to_string(),
    )
    .expect("write failed job manifest");
}

/// ST1: `kb status` must show the original filename alongside the src-id
/// for changed-but-not-yet-compiled inputs. A bare `normalized/src-<id>`
/// path is meaningless to users; the filename lets them recognize what's
/// pending at a glance and grep by it.
#[test]
fn status_changed_inputs_show_filename_and_src_id() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // Ingest a file with a recognizable name but *don't* compile, so it
    // stays in the "changed inputs not yet compiled" list.
    let source = kb_root.join("concurrent-a.md");
    fs::write(&source, "# Concurrent A\nBody.\n").expect("write source");
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("ingest").arg(&source);
    let ingest_output = cmd.output().expect("run kb ingest");
    assert!(
        ingest_output.status.success(),
        "kb ingest failed: {}",
        String::from_utf8_lossy(&ingest_output.stderr)
    );

    // Text output: filename appears, and the src-id is present as a suffix
    // so operators can still grep by hash.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("status");
    let output = cmd.output().expect("run kb status");
    assert!(
        output.status.success(),
        "kb status failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("changed inputs not yet compiled:"),
        "expected changed-inputs section; got: {stdout}"
    );
    assert!(
        stdout.contains("concurrent-a.md"),
        "expected filename in changed-inputs list; got: {stdout}"
    );
    // The src-id is a deterministic sha-prefixed string; we don't know the
    // exact value here, but we know it appears in parentheses after the
    // filename and starts with `src-`.
    let has_src_id_annotation = stdout
        .lines()
        .filter(|line| line.contains("concurrent-a.md"))
        .any(|line| line.contains("(src-"));
    assert!(
        has_src_id_annotation,
        "expected '(src-...)' annotation next to filename; got: {stdout}"
    );

    // JSON output: each changed input carries the original_path field,
    // plus normalized_path and src_id.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json").arg("status");
    let output = cmd.output().expect("run kb --json status");
    assert!(output.status.success());
    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse status json");
    let entries = envelope["data"]["changed_inputs_not_compiled"]
        .as_array()
        .expect("changed_inputs_not_compiled should be an array");
    assert_eq!(entries.len(), 1, "one pending input; got {entries:?}");
    let entry = &entries[0];
    assert!(
        entry["src_id"].as_str().unwrap_or("").starts_with("src-"),
        "src_id should be a src-<hash> string: {entry}"
    );
    assert!(
        entry["normalized_path"]
            .as_str()
            .unwrap_or("")
            .contains("normalized"),
        "normalized_path should reference the normalized/ dir: {entry}"
    );
    let original = entry["original_path"]
        .as_str()
        .expect("original_path should be populated for a freshly ingested file");
    assert!(
        original.ends_with("concurrent-a.md"),
        "original_path should recover the ingest filename; got {original}"
    );
}

/// ST2: when every source falls in the same kind bucket, the per-kind
/// breakdown carries no information and must be suppressed. The header
/// line (`wiki source pages: N`) already conveys the total.
#[test]
fn status_hides_single_kind_source_breakdown() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // One file-type source page is enough to populate a single "other"
    // bucket via `extract_source_kind`.
    write_source_page(&kb_root, "only", "Only", "body.");
    let mut compile = kb_cmd(&kb_root);
    compile.arg("compile");
    let compile_out = compile.output().expect("run kb compile");
    assert!(
        compile_out.status.success(),
        "compile failed: {}",
        String::from_utf8_lossy(&compile_out.stderr)
    );

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("status");
    let output = cmd.output().expect("run kb status");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("wiki source pages:"),
        "expected wiki source pages header; got: {stdout}"
    );
    // The key regression: no `other: N` (or any per-kind) line when the
    // breakdown is a single bucket.
    for line in stdout.lines() {
        let trimmed = line.trim_start();
        assert!(
            !trimmed.starts_with("other:"),
            "single-kind breakdown must be hidden; got line: {line}\nfull: {stdout}"
        );
    }

    // JSON schema stays unchanged — `by_kind` is still emitted so callers
    // that consume it don't break.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json").arg("status");
    let output = cmd.output().expect("run kb --json status");
    assert!(output.status.success());
    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse status json");
    assert!(
        envelope["data"]["sources"]["by_kind"].is_object(),
        "JSON payload must still expose sources.by_kind"
    );
}

/// ST3: failed jobs accumulate indefinitely (every typo, every bad path
/// adds one). The text list must cap at 10 and emit a "... and N more"
/// hint when exceeded; the header shows the true total so users see scope.
#[test]
fn status_caps_failed_jobs_at_ten_with_more_hint() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // Seed 15 failed jobs with strictly-increasing timestamps so sort
    // order is deterministic.
    for i in 0u32..15 {
        seed_failed_job(&kb_root, &format!("fail-{i:02}"), 1_000 + u64::from(i));
    }

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("status");
    let output = cmd.output().expect("run kb status");
    assert!(
        output.status.success(),
        "kb status failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Header shows the real total.
    assert!(
        stdout.contains("failed job runs (15):"),
        "header should report true total of 15; got: {stdout}"
    );

    // Body is capped at 10 detail lines. Count lines that look like a
    // job entry (`  <id> | Failed | ...`).
    let detail_lines = stdout
        .lines()
        .filter(|line| line.starts_with("  fail-"))
        .count();
    assert_eq!(
        detail_lines, 10,
        "exactly 10 failed-job detail lines should render; got {detail_lines}\nfull: {stdout}"
    );

    // "... and 5 more (run 'kb jobs --failed' to inspect)" hint.
    assert!(
        stdout.contains("... and 5 more"),
        "expected '... and 5 more' hint; got: {stdout}"
    );
    assert!(
        stdout.contains("kb jobs --failed"),
        "hint should point at future `kb jobs --failed` entry point; got: {stdout}"
    );

    // JSON payload: `failed_jobs_total` carries the true total, and
    // `failed_jobs` is capped at the display limit so JSON consumers get a
    // bounded list too.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json").arg("status");
    let output = cmd.output().expect("run kb --json status");
    assert!(output.status.success());
    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse status json");
    assert_eq!(envelope["data"]["failed_jobs_total"], 15);
    assert_eq!(
        envelope["data"]["failed_jobs"]
            .as_array()
            .expect("failed_jobs array")
            .len(),
        10
    );
}

/// Seed an interrupted `JobRun` manifest for cap + prune tests. Mirrors
/// `seed_failed_job` but with `"interrupted"` status and a default
/// command of `"compile"` (the most realistic source — user hits ^C
/// during a long compile, leaves a manifest behind).
fn seed_interrupted_job(root: &Path, id: &str, started_at_millis: u64) {
    let jobs_dir = root.join("state/jobs");
    fs::create_dir_all(&jobs_dir).expect("create state/jobs");
    let manifest = serde_json::json!({
        "metadata": {
            "id": id,
            "created_at_millis": started_at_millis,
            "updated_at_millis": started_at_millis,
            "source_hashes": [],
            "model_version": null,
            "tool_version": "kb-cli/0.1.0",
            "prompt_template_hash": null,
            "dependencies": [],
            "output_paths": [],
            "status": "fresh"
        },
        "command": "compile",
        "root_path": root,
        "started_at_millis": started_at_millis,
        "ended_at_millis": started_at_millis + 42,
        "status": "interrupted",
        "log_path": null,
        "affected_outputs": [],
        "pid": 99_999,
        "exit_code": 130
    });
    fs::write(
        jobs_dir.join(format!("{id}.json")),
        manifest.to_string(),
    )
    .expect("write interrupted job manifest");
}

/// bn-3qn part 1: `kb status` must cap interrupted-job detail lines at
/// 5 (mirrors bn-2cr's 10-cap for failed jobs, but tighter — each
/// interrupted entry also gets a per-section "Inspect the logs" footer
/// so 20+ entries turn status into a wall of remediation text).
#[test]
fn status_caps_interrupted_jobs_at_five_with_more_hint() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // Seed 20 interrupted jobs with strictly-increasing timestamps so
    // sort order is deterministic.
    for i in 0u32..20 {
        seed_interrupted_job(&kb_root, &format!("intr-{i:02}"), 1_000 + u64::from(i));
    }

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("status");
    let output = cmd.output().expect("run kb status");
    assert!(
        output.status.success(),
        "kb status failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Header reflects the true total.
    assert!(
        stdout.contains("interrupted job runs (20):"),
        "header should report true total of 20; got: {stdout}"
    );

    // Body is capped at 5 detail lines (ids like `intr-NN`).
    let detail_lines = stdout
        .lines()
        .filter(|line| line.starts_with("  intr-"))
        .count();
    assert_eq!(
        detail_lines, 5,
        "exactly 5 interrupted detail lines should render; got {detail_lines}\nfull: {stdout}"
    );

    // "... and 15 more (run 'kb jobs prune --interrupted' ...)" hint.
    assert!(
        stdout.contains("... and 15 more"),
        "expected '... and 15 more' hint; got: {stdout}"
    );
    assert!(
        stdout.contains("kb jobs prune --interrupted"),
        "hint should point at `kb jobs prune --interrupted`; got: {stdout}"
    );

    // JSON payload: `interrupted_jobs_total` = 20 (honest), while
    // `interrupted_jobs` array is capped at 5 for consumers.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json").arg("status");
    let output = cmd.output().expect("run kb --json status");
    assert!(output.status.success());
    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse status json");
    assert_eq!(envelope["data"]["interrupted_jobs_total"], 20);
    assert_eq!(
        envelope["data"]["interrupted_jobs"]
            .as_array()
            .expect("interrupted_jobs array")
            .len(),
        5
    );
}

/// Negative: with fewer than 5 interrupted jobs the "more" hint must
/// not appear — no dangling "... and 0 more" line.
#[test]
fn status_below_interrupted_cap_has_no_more_hint() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    for i in 0u32..3 {
        seed_interrupted_job(&kb_root, &format!("intr-{i}"), 1_000 + u64::from(i));
    }

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("status");
    let output = cmd.output().expect("run kb status");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(stdout.contains("interrupted job runs (3):"));
    assert!(
        !stdout.contains("... and"),
        "no more-hint expected below the cap; got: {stdout}"
    );
}

/// bn-3qn part 2: `kb jobs prune --interrupted --older-than 0` moves
/// all interrupted manifests into `trash/jobs-<ts>/` and a subsequent
/// `kb status` reports zero interrupted runs. `--older-than 0` is the
/// test-friendly "prune everything" escape hatch.
#[test]
fn jobs_prune_interrupted_moves_to_trash_and_clears_status() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // Seed 20 interrupted manifests + a `.log` sidecar for one to prove
    // the log is also moved (forensic preservation).
    for i in 0u32..20 {
        seed_interrupted_job(&kb_root, &format!("intr-{i:02}"), 1_000 + u64::from(i));
    }
    fs::write(
        kb_root.join("state/jobs/intr-00.log"),
        "some log content\n",
    )
    .expect("write log sidecar");

    // Prune.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("jobs")
        .arg("prune")
        .arg("--interrupted")
        .arg("--older-than")
        .arg("0");
    let output = cmd.output().expect("run kb jobs prune --interrupted");
    assert!(
        output.status.success(),
        "kb jobs prune failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Pruned 20 job run(s)"),
        "expected prune confirmation, got: {stdout}"
    );

    // Trash directory exists and contains all 20 manifests + the log.
    let trash_root = kb_root.join("trash");
    let trash_dirs: Vec<PathBuf> = fs::read_dir(&trash_root)
        .expect("read trash")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("jobs-"))
        })
        .collect();
    assert_eq!(trash_dirs.len(), 1, "exactly one jobs-<ts>/ bucket expected");
    let trash = &trash_dirs[0];
    let trashed: Vec<String> = fs::read_dir(trash)
        .expect("read trash bucket")
        .filter_map(Result::ok)
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    assert_eq!(
        trashed
            .iter()
            .filter(|n| Path::new(n)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("json")))
            .count(),
        20,
        "all 20 manifests should be in trash: {trashed:?}"
    );
    assert!(
        trashed.iter().any(|n| n == "intr-00.log"),
        "log sidecar for intr-00 must be preserved in trash, got: {trashed:?}"
    );

    // `state/jobs/` is now clean — no stray manifests.
    let remaining: Vec<String> = fs::read_dir(kb_root.join("state/jobs"))
        .expect("read jobs dir")
        .filter_map(Result::ok)
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| {
            Path::new(n)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
        })
        .filter(|n| n.starts_with("intr-"))
        .collect();
    assert!(
        remaining.is_empty(),
        "no interrupted manifests should remain; got: {remaining:?}"
    );

    // Subsequent `kb status --json`: zero interrupted jobs.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json").arg("status");
    let output = cmd.output().expect("run kb status");
    assert!(output.status.success());
    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse status json");
    assert_eq!(
        envelope["data"]["interrupted_jobs_total"], 0,
        "status should show zero interrupted jobs after prune"
    );
}

/// `kb jobs prune --interrupted` with the default `--older-than 7`
/// must KEEP recently-created manifests — we don't want a stray prune
/// to wipe out a still-investigatable failure. The stamp needs to be
/// within the 7-day window for this test; seeding with `now - 1s`
/// ensures that.
#[test]
fn jobs_prune_respects_older_than_window() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let now_ms = u64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_millis(),
    )
    .expect("now-millis fits in u64");
    // Brand-new interrupted job (< 7d old) — must NOT be pruned.
    seed_interrupted_job(&kb_root, "fresh-intr", now_ms - 1_000);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("jobs").arg("prune").arg("--interrupted"); // default --older-than 7
    let output = cmd.output().expect("run kb jobs prune --interrupted");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("No job runs older than 7 day(s)"),
        "fresh manifests should be kept by default; got: {stdout}"
    );

    // Manifest still on disk.
    assert!(
        kb_root.join("state/jobs/fresh-intr.json").exists(),
        "fresh manifest should still be on disk under default window"
    );
}

/// Missing status selectors is a user error — "prune nothing" deserves
/// a non-zero exit, not silent success.
#[test]
fn jobs_prune_without_selector_errors() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("jobs").arg("prune");
    let output = cmd.output().expect("run kb jobs prune");
    assert!(
        !output.status.success(),
        "kb jobs prune without a status selector should exit non-zero"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--interrupted") && stderr.contains("--failed"),
        "error message should mention the missing selectors; got stderr: {stderr}"
    );
}

/// `kb jobs list --interrupted` surfaces manifests with their log
/// paths. JSON round-trips through the `jobs.list` envelope so
/// downstream tooling can page through.
#[test]
fn jobs_list_interrupted_reports_ids_and_log_paths() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    for i in 0u32..3 {
        seed_interrupted_job(&kb_root, &format!("intr-{i}"), 1_000 + u64::from(i));
    }
    // Log sidecar for one; list must surface it for forensic jump-to.
    fs::write(kb_root.join("state/jobs/intr-1.log"), "trace\n")
        .expect("write log sidecar");

    // Text listing.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("jobs").arg("list").arg("--interrupted");
    let output = cmd.output().expect("run kb jobs list --interrupted");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("intr-0"));
    assert!(stdout.contains("intr-1"));
    assert!(stdout.contains("intr-2"));
    assert!(
        stdout.contains("intr-1.log"),
        "log path should appear for the manifest with a sidecar; got: {stdout}"
    );

    // JSON listing: 3 entries, total 3, interrupted status.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json")
        .arg("jobs")
        .arg("list")
        .arg("--interrupted");
    let output = cmd.output().expect("run kb --json jobs list");
    assert!(output.status.success());
    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse json");
    assert_eq!(envelope["data"]["total"], 3);
    let arr = envelope["data"]["jobs"]
        .as_array()
        .expect("jobs array");
    assert_eq!(arr.len(), 3);
    for entry in arr {
        assert_eq!(entry["status"], "interrupted");
    }
}

/// ST3 negative: with fewer than 10 failures the "more" hint must not
/// appear — we don't want a dangling "... and 0 more" line.
#[test]
fn status_below_failed_cap_has_no_more_hint() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    for i in 0u32..3 {
        seed_failed_job(&kb_root, &format!("fail-{i}"), 1_000 + u64::from(i));
    }

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("status");
    let output = cmd.output().expect("run kb status");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(stdout.contains("failed job runs (3):"));
    assert!(
        !stdout.contains("... and"),
        "no more-hint expected below the cap; got: {stdout}"
    );
}

// ── concept_merge auto-apply tests ───────────────────────────────────────────

fn seed_concept_page(root: &Path, rel: &str, frontmatter: &str, body: &str) -> PathBuf {
    let full = root.join(rel);
    fs::create_dir_all(full.parent().expect("parent")).expect("create dirs");
    let content = format!("---\n{frontmatter}---\n{body}");
    fs::write(&full, content).expect("write concept page");
    full
}

fn seed_concept_merge_review(
    root: &Path,
    id: &str,
    canonical_rel: &str,
    members: &[&str],
) -> PathBuf {
    // `dependencies` carries the raw candidate names — the canonical page's
    // candidate plus the ones being subsumed. `proposed_destination` is the
    // canonical concept page path.
    let item = ReviewItem {
        metadata: EntityMetadata {
            id: id.to_string(),
            created_at_millis: 1_000,
            updated_at_millis: 1_000,
            source_hashes: vec!["hash-merge-1".to_string()],
            model_version: None,
            tool_version: Some("kb-test".to_string()),
            prompt_template_hash: None,
            dependencies: members.iter().map(|s| (*s).to_string()).collect(),
            output_paths: vec![PathBuf::from(canonical_rel)],
            status: Status::NeedsReview,
        },
        kind: ReviewKind::ConceptMerge,
        target_entity_id: id.to_string(),
        proposed_destination: Some(PathBuf::from(canonical_rel)),
        citations: vec![],
        affected_pages: vec![PathBuf::from(canonical_rel)],
        created_at_millis: 1_000,
        status: ReviewStatus::Pending,
        comment: format!(
            "Proposed canonical: Borrow checker\nMembers: {}\n",
            members.join(", ")
        ),
    };
    save_review_item(root, &item).expect("save review item");
    root.join(format!("reviews/merges/{id}.json"))
}

#[test]
fn review_approve_applies_concept_merge() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // Canonical concept page with a single alias and one source id.
    let canonical = seed_concept_page(
        &kb_root,
        "wiki/concepts/borrow-checker.md",
        "id: concept:borrow-checker\nname: Borrow checker\naliases:\n  - borrowck\nsource_document_ids:\n  - src-aaa\n",
        "\n# Borrow checker\n\nValidates references at compile time.\n",
    );

    // Subsumed member with its own alias + source_document_ids + sources.
    let member_path = seed_concept_page(
        &kb_root,
        "wiki/concepts/borrow-checker-pass.md",
        "id: concept:borrow-checker-pass\nname: Borrow checker pass\naliases:\n  - bc pass\nsource_document_ids:\n  - src-bbb\nsources:\n  - heading_anchor: ownership\n    quote: The borrow checker validates references.\n",
        "\n# Borrow checker pass\n\nAnother view of the checker.\n",
    );

    seed_concept_merge_review(
        &kb_root,
        "merge:borrow-checker",
        "wiki/concepts/borrow-checker.md",
        &["borrow checker", "borrow checker pass"],
    );

    // First approve: applies the merge.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("review").arg("approve").arg("merge:borrow-checker");
    let output = cmd.output().expect("run kb review approve");
    assert!(
        output.status.success(),
        "kb review approve failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Approved: merge:borrow-checker"));
    assert!(stdout.contains("concept_merge"));
    assert!(stdout.contains("wiki/concepts/borrow-checker.md"));
    assert!(stdout.contains("wiki/concepts/borrow-checker-pass.md"));
    assert!(stdout.contains("kb compile"));

    // Subsumed member file is gone.
    assert!(
        !member_path.exists(),
        "subsumed concept page should be deleted"
    );

    // Canonical page has been updated: frontmatter carries merged aliases +
    // source_document_ids + the member name itself as an alias.
    let canonical_text = fs::read_to_string(&canonical).expect("read canonical");
    assert!(
        canonical_text.contains("- borrowck"),
        "pre-existing alias preserved; content: {canonical_text}"
    );
    assert!(
        canonical_text.contains("- bc pass"),
        "member alias merged; content: {canonical_text}"
    );
    assert!(
        canonical_text.contains("- borrow checker pass"),
        "member name added as alias; content: {canonical_text}"
    );
    assert!(
        canonical_text.contains("src-aaa"),
        "pre-existing source id preserved"
    );
    assert!(
        canonical_text.contains("src-bbb"),
        "member source id merged"
    );
    assert!(
        canonical_text.contains("sources:"),
        "member sources list merged; content: {canonical_text}"
    );
    assert!(
        canonical_text.contains("Validates references at compile time."),
        "body preserved"
    );

    // Review item status flipped to approved.
    let review_path = kb_root.join("reviews/merges/merge:borrow-checker.json");
    let saved: Value =
        serde_json::from_str(&fs::read_to_string(&review_path).expect("read review")).expect("parse");
    assert_eq!(saved["status"], "approved");

    // Idempotency: re-approve is a no-op that re-prints the summary.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("review").arg("approve").arg("merge:borrow-checker");
    let output = cmd.output().expect("run kb review approve (reapply)");
    assert!(
        output.status.success(),
        "re-approve must succeed without error: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Already approved")
            || stdout.contains("Approved: merge:borrow-checker"),
        "re-apply output should acknowledge prior approval: {stdout}"
    );
    // Canonical page is unchanged on re-apply.
    let canonical_text_after = fs::read_to_string(&canonical).expect("read canonical");
    assert_eq!(
        canonical_text, canonical_text_after,
        "re-apply must not mutate canonical page"
    );
}

#[test]
fn review_approve_concept_merge_missing_canonical_errors() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // Member exists but canonical does not — apply must bail cleanly, leaving
    // the review item pending.
    seed_concept_page(
        &kb_root,
        "wiki/concepts/borrowck.md",
        "id: concept:borrowck\nname: borrowck\n",
        "\n# borrowck\n",
    );

    seed_concept_merge_review(
        &kb_root,
        "merge:missing-canonical",
        "wiki/concepts/borrow-checker.md",
        &["borrow checker", "borrowck"],
    );

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("review").arg("approve").arg("merge:missing-canonical");
    let output = cmd.output().expect("run kb review approve");
    assert!(
        !output.status.success(),
        "kb review approve should fail when canonical page missing"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("canonical") || stderr.contains("borrow-checker"),
        "stderr should mention missing canonical: {stderr}"
    );

    // Review item still pending.
    let review_path = kb_root.join("reviews/merges/merge:missing-canonical.json");
    let saved: Value =
        serde_json::from_str(&fs::read_to_string(&review_path).expect("read review")).expect("parse");
    assert_eq!(saved["status"], "pending");

    // Member file untouched.
    assert!(kb_root.join("wiki/concepts/borrowck.md").exists());
}

#[test]
fn review_show_mentions_auto_apply_for_concept_merge() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    seed_concept_merge_review(
        &kb_root,
        "merge:show-me",
        "wiki/concepts/x.md",
        &["x", "y"],
    );

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("review").arg("show").arg("merge:show-me");
    let output = cmd.output().expect("run kb review show");
    assert!(
        output.status.success(),
        "kb review show failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("On approve"),
        "show output must describe approve behavior: {stdout}"
    );
    assert!(
        stdout.to_lowercase().contains("delete")
            || stdout.to_lowercase().contains("subsumed")
            || stdout.to_lowercase().contains("absorb"),
        "show output must mention that files are deleted/absorbed: {stdout}"
    );
}

/// Seed an in-flight compile lock so `kb lint` can observe it via the
/// sidecar-metadata peek. We write the JSON file directly (mirroring the
/// shape `KbLock::acquire` writes) and use the current test pid, which is
/// guaranteed to be alive, so the peek treats the holder as live.
///
/// The `root.lock` file itself is also touched so the on-disk layout matches
/// a real holder; we do not flock it because the peek does not contest the
/// advisory lock — it only reads the sidecar JSON.
fn seed_fake_compile_lock(kb_root: &Path, command: &str) -> u32 {
    let locks_dir = kb_root.join("state").join("locks");
    fs::create_dir_all(&locks_dir).expect("create locks dir");
    fs::write(locks_dir.join("root.lock"), b"").expect("touch root.lock");
    let pid = std::process::id();
    let metadata = serde_json::json!({
        "command": command,
        "pid": pid,
        "started_at_millis": 1,
    });
    fs::write(
        locks_dir.join("root.lock.json"),
        serde_json::to_vec_pretty(&metadata).expect("serialize metadata"),
    )
    .expect("write root.lock.json");
    pid
}

/// When a compile is in flight, `kb lint` (default mode) must warn on stderr
/// that the tree is mid-rewrite but continue to run normally and return its
/// usual exit code. The warning names the holder pid + command so an operator
/// can correlate.
#[test]
fn lint_warns_on_stderr_when_compile_in_flight() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    let holder_pid = seed_fake_compile_lock(&kb_root, "compile");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("lint").arg("--check").arg("broken-links");
    let output = cmd.output().expect("run kb lint");

    // A clean tree with no broken links must still pass (exit 0). The in-flight
    // compile must not fail default lint; it only adds a stderr preamble.
    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("kb compile is in flight"),
        "stderr must warn about in-flight compile: {stderr}"
    );
    assert!(
        stderr.contains(&format!("pid {holder_pid}")),
        "stderr warning must name the holder pid: {stderr}"
    );
}

/// `--strict` must refuse to run while a compile holds the lock: stale
/// warnings in --strict mode would fail CI for reasons that disappear on
/// retry, and that's exactly what --strict is meant to prevent.
#[test]
fn lint_strict_refuses_while_compile_in_flight() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    let holder_pid = seed_fake_compile_lock(&kb_root, "compile");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("lint")
        .arg("--check")
        .arg("broken-links")
        .arg("--strict");
    let output = cmd.output().expect("run kb lint --strict");

    assert_eq!(
        output.status.code(),
        Some(1),
        "stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("refusing to run --strict"),
        "stderr must explain the refusal: {stderr}"
    );
    assert!(
        stderr.contains(&format!("pid {holder_pid}")),
        "stderr must name holder pid: {stderr}"
    );
    assert!(
        stderr.contains("compile"),
        "stderr must name the holder command: {stderr}"
    );
}

/// Stale sidecar metadata (a sidecar that points at a dead pid) must not
/// trigger the warning — the compile that wrote it is long gone and the
/// tree is no longer mid-rewrite. Without the pid-alive check we would
/// false-positive on every sidecar left behind by a SIGKILL'd compile.
#[test]
fn lint_ignores_stale_compile_metadata_from_dead_pid() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let locks_dir = kb_root.join("state").join("locks");
    fs::create_dir_all(&locks_dir).expect("create locks dir");
    fs::write(locks_dir.join("root.lock"), b"").expect("touch root.lock");
    // u32::MAX / 2 is virtually guaranteed to not correspond to a running pid.
    let dead_pid: u32 = u32::MAX / 2;
    let metadata = serde_json::json!({
        "command": "compile",
        "pid": dead_pid,
        "started_at_millis": 1,
    });
    fs::write(
        locks_dir.join("root.lock.json"),
        serde_json::to_vec_pretty(&metadata).expect("serialize metadata"),
    )
    .expect("write root.lock.json");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("lint").arg("--check").arg("broken-links");
    let output = cmd.output().expect("run kb lint");

    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("kb compile is in flight"),
        "stale metadata from a dead pid must not trigger the warning: {stderr}"
    );
}

// ── lint:duplicate-concepts auto-apply tests ─────────────────────────────────

/// Seed a `lint:duplicate-concepts:...` review item that mimics what
/// `kb-lint`'s duplicate-concepts check produces: `proposed_destination` points
/// at the proposal JSON sidecar (NOT a concept page), and `dependencies` holds
/// the two concept ids. The auto-apply path has to recover the canonical
/// concept page from the review id.
fn seed_lint_duplicate_concepts_review(
    root: &Path,
    merged_from_concept_id: &str,
    canonical_concept_id: &str,
) -> (String, PathBuf) {
    let id = format!("lint:duplicate-concepts:{merged_from_concept_id}:{canonical_concept_id}");
    // Mirror the lint emitter: proposed_destination is a slugged JSON path
    // under reviews/merges/ (NOT a wiki/concepts/ page path). This is what
    // bn-1e9's auto-apply mistakenly tried to read as a concept page.
    let slug_a = merged_from_concept_id.replace(':', "-");
    let slug_b = canonical_concept_id.replace(':', "-");
    let proposal_rel = PathBuf::from(format!(
        "reviews/merges/lint-duplicate-concepts-{slug_a}-{slug_b}.json"
    ));
    let item = ReviewItem {
        metadata: EntityMetadata {
            id: id.clone(),
            created_at_millis: 1_000,
            updated_at_millis: 1_000,
            source_hashes: vec![],
            model_version: None,
            tool_version: Some("kb-test".to_string()),
            prompt_template_hash: None,
            dependencies: vec![
                merged_from_concept_id.to_string(),
                canonical_concept_id.to_string(),
            ],
            output_paths: vec![proposal_rel.clone()],
            status: Status::NeedsReview,
        },
        kind: ReviewKind::ConceptMerge,
        target_entity_id: merged_from_concept_id.to_string(),
        proposed_destination: Some(proposal_rel),
        citations: vec![],
        affected_pages: vec![],
        created_at_millis: 1_000,
        status: ReviewStatus::Pending,
        comment: format!(
            "Near-duplicate of '{canonical_concept_id}' (similarity: 1.00; matched: 'x' \u{2248} 'x')"
        ),
    };
    save_review_item(root, &item).expect("save lint-duplicate review item");
    let saved_path = root.join(format!("reviews/merges/{id}.json"));
    (id, saved_path)
}

#[test]
fn review_approve_applies_lint_duplicate_concepts() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // By convention, the *second* concept id in the review id is the canonical
    // (the page that survives). Here: `raft-consensus` is canonical,
    // `byzantine-consensus` is folded in and deleted.
    let canonical = seed_concept_page(
        &kb_root,
        "wiki/concepts/raft-consensus.md",
        "id: concept:raft-consensus\nname: Raft consensus\naliases:\n  - raft\n",
        "\n# Raft consensus\n\nLeader-based consensus.\n",
    );
    let merged_from = seed_concept_page(
        &kb_root,
        "wiki/concepts/byzantine-consensus.md",
        "id: concept:byzantine-consensus\nname: Byzantine consensus\naliases:\n  - bft\nsource_document_ids:\n  - src-bft\n",
        "\n# Byzantine consensus\n\nBFT.\n",
    );

    let (id, review_path) = seed_lint_duplicate_concepts_review(
        &kb_root,
        "concept:byzantine-consensus",
        "concept:raft-consensus",
    );

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("review").arg("approve").arg(&id);
    let output = cmd.output().expect("run kb review approve");
    assert!(
        output.status.success(),
        "kb review approve failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("concept_merge"), "stdout: {stdout}");
    assert!(
        stdout.contains("wiki/concepts/raft-consensus.md"),
        "stdout should show canonical path: {stdout}"
    );
    assert!(
        stdout.contains("byzantine-consensus"),
        "stdout should mention subsumed member: {stdout}"
    );

    // Merged-from file is deleted; canonical remains.
    assert!(
        !merged_from.exists(),
        "subsumed concept page should be deleted"
    );
    assert!(canonical.exists(), "canonical concept page must survive");

    // Canonical absorbed the member's alias + source id.
    let canonical_text = fs::read_to_string(&canonical).expect("read canonical");
    assert!(
        canonical_text.contains("- bft"),
        "member alias merged: {canonical_text}"
    );
    assert!(
        canonical_text.contains("src-bft"),
        "member source id merged: {canonical_text}"
    );
    assert!(
        canonical_text.contains("- raft"),
        "pre-existing alias preserved: {canonical_text}"
    );

    // Review item flipped to approved; proposed_destination is preserved
    // unchanged on disk (we only rewrite it in-memory for the apply call).
    let saved: Value =
        serde_json::from_str(&fs::read_to_string(&review_path).expect("read review"))
            .expect("parse");
    assert_eq!(saved["status"], "approved");

    // Re-approve is an idempotent no-op.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("review").arg("approve").arg(&id);
    let output = cmd.output().expect("run kb review approve (reapply)");
    assert!(
        output.status.success(),
        "re-approve must succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let canonical_text_after = fs::read_to_string(&canonical).expect("read canonical again");
    assert_eq!(
        canonical_text, canonical_text_after,
        "re-apply must not mutate canonical page"
    );
}

#[test]
fn review_approve_lint_duplicate_concepts_missing_canonical_bails() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // Only the merged-from page exists; the canonical is missing. Apply must
    // error cleanly and leave the review item pending.
    let merged_from = seed_concept_page(
        &kb_root,
        "wiki/concepts/byzantine-consensus.md",
        "id: concept:byzantine-consensus\nname: Byzantine consensus\n",
        "\n# Byzantine consensus\n",
    );

    let (id, review_path) = seed_lint_duplicate_concepts_review(
        &kb_root,
        "concept:byzantine-consensus",
        "concept:raft-consensus",
    );

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("review").arg("approve").arg(&id);
    let output = cmd.output().expect("run kb review approve");
    assert!(
        !output.status.success(),
        "approve should fail when canonical page is missing"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("canonical") || stderr.contains("raft-consensus"),
        "stderr should mention missing canonical: {stderr}"
    );

    // Review still pending; merged-from file untouched.
    let saved: Value =
        serde_json::from_str(&fs::read_to_string(&review_path).expect("read review"))
            .expect("parse");
    assert_eq!(saved["status"], "pending");
    assert!(merged_from.exists(), "merged-from page must not be deleted");
}

// ---------------------------------------------------------------------------
// `kb forget` — bn-1fq F1 + F2
// ---------------------------------------------------------------------------

/// Return the single `src-<hex>` id produced by an ingest of `source` via
/// `kb --json ingest`. Panics on any missing structure, because every test
/// that calls this expects a successful ingest before proceeding.
fn ingest_single_and_get_src_id(kb_root: &Path, source: &Path) -> String {
    let mut cmd = kb_cmd(kb_root);
    cmd.arg("--json").arg("ingest").arg(source);
    let output = cmd.output().expect("run kb ingest");
    assert!(
        output.status.success(),
        "ingest failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let envelope: Value =
        serde_json::from_slice(&output.stdout).expect("parse ingest envelope");
    envelope["data"]["results"][0]["source_document_id"]
        .as_str()
        .expect("source_document_id in ingest result")
        .to_string()
}

/// Drop a minimal `wiki/sources/<src>.md` file so `kb forget` has a wiki
/// page to remove. Avoids needing a full `kb compile` in the test path.
fn stub_wiki_source_page(kb_root: &Path, src_id: &str) {
    let dir = kb_root.join("wiki/sources");
    fs::create_dir_all(&dir).expect("mkdir wiki/sources");
    let markdown = format!(
        "---\nid: wiki-source-{src_id}\ntype: source\ntitle: {src_id}\n\
source_document_id: {src_id}\nsource_revision_id: rev-stub\n\
generated_at: 0\nbuild_record_id: build-stub\n---\n\n# Source\n"
    );
    fs::write(dir.join(format!("{src_id}.md")), markdown).expect("write wiki source page");
}

/// F1: `kb forget src-xxxx` moves normalized/ + raw/inbox/ + wiki/sources/
/// entries into `trash/<src>-<ts>/` and records a succeeded job run.
#[test]
fn forget_by_src_id_moves_entries_to_trash_and_succeeds() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let source = kb_root.join("note.md");
    fs::write(&source, "# hi\n\ncontent\n").expect("write source");
    let src_id = ingest_single_and_get_src_id(&kb_root, &source);
    stub_wiki_source_page(&kb_root, &src_id);

    let normalized_dir = kb_root.join("normalized").join(&src_id);
    let raw_dir = kb_root.join("raw/inbox").join(&src_id);
    let wiki_page = kb_root.join("wiki/sources").join(format!("{src_id}.md"));
    assert!(normalized_dir.exists());
    assert!(raw_dir.exists());
    assert!(wiki_page.exists());

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--force").arg("forget").arg(&src_id);
    let output = cmd.output().expect("run kb forget");
    assert!(
        output.status.success(),
        "kb forget failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(!normalized_dir.exists(), "normalized dir should be moved");
    assert!(!raw_dir.exists(), "raw/inbox dir should be moved");
    assert!(!wiki_page.exists(), "wiki source page should be moved");

    // A `trash/<src_id>-*` dir should exist and contain the three entries
    // preserved under their original parent names.
    let trash_root = kb_root.join("trash");
    let trash_entries: Vec<_> = fs::read_dir(&trash_root)
        .expect("read trash dir")
        .filter_map(Result::ok)
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with(&format!("{src_id}-"))
        })
        .collect();
    assert_eq!(trash_entries.len(), 1, "exactly one trash bundle expected");
    let bundle = trash_entries
        .into_iter()
        .next()
        .expect("first trash bundle")
        .path();
    assert!(bundle.join("normalized").join(&src_id).exists());
    assert!(bundle.join("raw/inbox").join(&src_id).exists());
    assert!(
        bundle
            .join("wiki/sources")
            .join(format!("{src_id}.md"))
            .exists()
    );

    // The forget job must be recorded as succeeded in `kb --json status`.
    let mut status_cmd = kb_cmd(&kb_root);
    status_cmd.arg("--json").arg("status");
    let status_output = status_cmd.output().expect("run kb status");
    assert!(
        status_output.status.success(),
        "kb status failed: {}",
        String::from_utf8_lossy(&status_output.stderr)
    );
    let envelope: Value =
        serde_json::from_slice(&status_output.stdout).expect("parse status json");
    let recent = envelope["data"]["recent_jobs"]
        .as_array()
        .expect("recent_jobs");
    let forget_jobs: Vec<&Value> = recent
        .iter()
        .filter(|j| j["command"] == "forget")
        .collect();
    assert_eq!(forget_jobs.len(), 1, "exactly one forget job expected");
    assert_eq!(forget_jobs[0]["status"], "succeeded");

    // After forget, neither normalized nor wiki source listings should
    // include the removed src_id.
    assert_eq!(envelope["data"]["normalized_source_count"], 0);
}

/// F1: `kb forget <path>` resolves the path to the right src-id and
/// removes exactly the same entries as the bare-id form.
#[test]
fn forget_by_path_resolves_and_removes_source() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let source = kb_root.join("byhand.md");
    fs::write(&source, "# hi\n\nbody\n").expect("write source");
    let src_id = ingest_single_and_get_src_id(&kb_root, &source);
    stub_wiki_source_page(&kb_root, &src_id);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--force").arg("forget").arg(&source);
    let output = cmd.output().expect("run kb forget by path");
    assert!(
        output.status.success(),
        "kb forget by path failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(!kb_root.join("normalized").join(&src_id).exists());
    assert!(!kb_root.join("raw/inbox").join(&src_id).exists());
}

/// F1: `kb forget --dry-run <src-id>` prints the plan and touches no files.
#[test]
fn forget_dry_run_prints_plan_and_keeps_disk_intact() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let source = kb_root.join("kept.md");
    fs::write(&source, "# hi\n\nbody\n").expect("write source");
    let src_id = ingest_single_and_get_src_id(&kb_root, &source);
    stub_wiki_source_page(&kb_root, &src_id);

    let normalized_dir = kb_root.join("normalized").join(&src_id);
    let raw_dir = kb_root.join("raw/inbox").join(&src_id);
    let wiki_page = kb_root.join("wiki/sources").join(format!("{src_id}.md"));

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--dry-run")
        .arg("--json")
        .arg("forget")
        .arg(&src_id);
    let output = cmd.output().expect("run kb forget --dry-run");
    assert!(
        output.status.success(),
        "dry-run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value =
        serde_json::from_slice(&output.stdout).expect("parse forget json");
    assert_eq!(envelope["command"], "forget");
    assert_eq!(envelope["data"]["dry_run"], true);
    assert_eq!(envelope["data"]["plan"]["src_id"], src_id);
    let moves = envelope["data"]["plan"]["moves"]
        .as_array()
        .expect("moves array");
    assert_eq!(moves.len(), 3, "expected 3 moves (normalized/raw/wiki)");

    // Nothing should have been moved.
    assert!(normalized_dir.exists(), "normalized must still exist");
    assert!(raw_dir.exists(), "raw/inbox must still exist");
    assert!(wiki_page.exists(), "wiki page must still exist");
    // `kb init` seeds an empty trash/ dir, so we can't assert it's absent;
    // assert instead that no src-id-prefixed bundle got created under it.
    let trash_dir = kb_root.join("trash");
    if trash_dir.exists() {
        let stray_bundles: Vec<_> = fs::read_dir(&trash_dir)
            .expect("read trash")
            .filter_map(Result::ok)
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with(&format!("{src_id}-"))
            })
            .collect();
        assert!(
            stray_bundles.is_empty(),
            "dry-run must not create trash bundles, got: {:?}",
            stray_bundles
                .iter()
                .map(std::fs::DirEntry::path)
                .collect::<Vec<_>>()
        );
    }
}

/// F1: `kb forget` with no target is a clap usage error (exits non-zero).
#[test]
fn forget_without_target_is_usage_error() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("forget");
    let output = cmd.output().expect("run kb forget with no arg");

    assert!(
        !output.status.success(),
        "kb forget without target unexpectedly succeeded"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    // clap emits a "required ... <TARGET>" style error. Accept either
    // "required" or "usage" so we don't over-fit to clap's exact phrasing.
    assert!(
        stderr.to_lowercase().contains("required")
            || stderr.to_lowercase().contains("usage"),
        "expected clap required/usage error, got: {stderr}"
    );
}

/// F1: running `kb compile` after `kb forget` must NOT recreate the
/// removed wiki/sources page — normalized/ is gone, so compile has no
/// input to regenerate from.
#[test]
fn compile_after_forget_does_not_recreate_source_pages() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let source = kb_root.join("once.md");
    fs::write(&source, "# hi\n\nbody\n").expect("write source");
    let src_id = ingest_single_and_get_src_id(&kb_root, &source);
    stub_wiki_source_page(&kb_root, &src_id);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--force").arg("forget").arg(&src_id);
    let output = cmd.output().expect("run kb forget");
    assert!(output.status.success());

    // Run compile; it must exit 0 and leave the wiki source page absent.
    let mut compile_cmd = kb_cmd(&kb_root);
    compile_cmd.arg("--dry-run").arg("compile");
    let compile_output = compile_cmd.output().expect("run kb compile --dry-run");
    assert!(
        compile_output.status.success(),
        "kb compile after forget failed: {}",
        String::from_utf8_lossy(&compile_output.stderr)
    );

    let wiki_page = kb_root.join("wiki/sources").join(format!("{src_id}.md"));
    assert!(
        !wiki_page.exists(),
        "wiki source page must not be recreated after forget"
    );
}

/// F2: after the origin file is deleted, `kb --json status` exposes the
/// src-id under `sources_with_missing_origin`.
#[test]
fn status_surfaces_sources_with_missing_origin() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let source = kb_root.join("vanishing.md");
    fs::write(&source, "# hi\n\nbody\n").expect("write source");
    let src_id = ingest_single_and_get_src_id(&kb_root, &source);

    // Sanity: no missing origins yet.
    let mut status_cmd = kb_cmd(&kb_root);
    status_cmd.arg("--json").arg("status");
    let output = status_cmd.output().expect("run kb status");
    assert!(output.status.success());
    let envelope: Value =
        serde_json::from_slice(&output.stdout).expect("parse status");
    assert!(
        envelope["data"]["sources_with_missing_origin"]
            .as_array()
            .expect("array")
            .is_empty(),
        "origin still present: {:?}",
        envelope["data"]["sources_with_missing_origin"]
    );

    // Remove the original source file on disk and re-check.
    fs::remove_file(&source).expect("rm source");

    let mut status_cmd = kb_cmd(&kb_root);
    status_cmd.arg("--json").arg("status");
    let output = status_cmd.output().expect("run kb status after rm");
    assert!(output.status.success());
    let envelope: Value =
        serde_json::from_slice(&output.stdout).expect("parse status");
    let missing = envelope["data"]["sources_with_missing_origin"]
        .as_array()
        .expect("missing array");
    assert_eq!(missing.len(), 1, "expected exactly one missing origin");
    assert_eq!(missing[0]["src_id"], src_id);

    // Text output must include the hint line pointing at kb forget.
    let mut text_cmd = kb_cmd(&kb_root);
    text_cmd.arg("status");
    let text_output = text_cmd.output().expect("run kb status text");
    assert!(text_output.status.success());
    let stdout = String::from_utf8_lossy(&text_output.stdout);
    assert!(
        stdout.contains("sources with missing origin"),
        "status text must mention missing origins, got:\n{stdout}"
    );
    assert!(
        stdout.contains("kb forget"),
        "status text must point at kb forget, got:\n{stdout}"
    );
}

/// bn-32t: declining the `[y/N]` confirmation prompt is the happy path
/// ("user changed their mind"), not an error. `kb forget src-X` with `n`
/// must exit 0, leave the source on disk, and NOT surface as a failed job
/// in `kb status`.
#[test]
fn forget_declined_prompt_exits_zero_and_is_noop() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let source = kb_root.join("keep.md");
    fs::write(&source, "# hi\n\ncontent\n").expect("write source");
    let src_id = ingest_single_and_get_src_id(&kb_root, &source);
    stub_wiki_source_page(&kb_root, &src_id);

    let normalized_dir = kb_root.join("normalized").join(&src_id);
    let raw_dir = kb_root.join("raw/inbox").join(&src_id);
    let wiki_page = kb_root.join("wiki/sources").join(format!("{src_id}.md"));

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("forget").arg(&src_id).write_stdin("n\n");
    let output = cmd.output().expect("run kb forget with declined prompt");

    assert!(
        output.status.success(),
        "kb forget should exit 0 when user declines the prompt; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("forget cancelled"),
        "expected 'forget cancelled' in stdout, got: {stdout}"
    );

    // Source must still be on disk — declining is a no-op.
    assert!(normalized_dir.exists(), "normalized dir must remain");
    assert!(raw_dir.exists(), "raw/inbox dir must remain");
    assert!(wiki_page.exists(), "wiki source page must remain");

    // Declined forget must not surface as a failed job in `kb status`.
    let mut status_cmd = kb_cmd(&kb_root);
    status_cmd.arg("--json").arg("status");
    let status_output = status_cmd.output().expect("run kb status");
    assert!(status_output.status.success());
    let envelope: Value =
        serde_json::from_slice(&status_output.stdout).expect("parse status json");
    let recent = envelope["data"]["recent_jobs"]
        .as_array()
        .expect("recent_jobs");
    let failed_forgets: Vec<&Value> = recent
        .iter()
        .filter(|j| j["command"] == "forget" && j["status"] == "failed")
        .collect();
    assert!(
        failed_forgets.is_empty(),
        "declined forget should not appear as a failed job: {failed_forgets:?}"
    );
}

/// bn-32t: pressing Enter at the `[y/N]` prompt (empty input) also declines
/// and exits 0. Mirrors the `n`/`N` behavior.
#[test]
fn forget_empty_prompt_response_declines_and_exits_zero() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let source = kb_root.join("keep2.md");
    fs::write(&source, "# hi\n\ncontent\n").expect("write source");
    let src_id = ingest_single_and_get_src_id(&kb_root, &source);
    stub_wiki_source_page(&kb_root, &src_id);

    let normalized_dir = kb_root.join("normalized").join(&src_id);

    let mut cmd = kb_cmd(&kb_root);
    // Empty line: just press Enter.
    cmd.arg("forget").arg(&src_id).write_stdin("\n");
    let output = cmd.output().expect("run kb forget with empty prompt");

    assert!(
        output.status.success(),
        "kb forget should exit 0 on empty (default-No) input; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(normalized_dir.exists(), "normalized dir must remain");
}

/// F1/F2: after `kb forget`, `kb status` must no longer list the removed
/// source — both `normalized_source_count` and `sources_with_missing_origin`
/// should be empty with respect to it.
#[test]
fn status_after_forget_does_not_list_source() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let source = kb_root.join("gone.md");
    fs::write(&source, "# hi\n\nbody\n").expect("write source");
    let src_id = ingest_single_and_get_src_id(&kb_root, &source);
    stub_wiki_source_page(&kb_root, &src_id);
    // Simulate a deleted origin so missing_origin would have flagged it.
    fs::remove_file(&source).expect("rm source");

    let mut forget_cmd = kb_cmd(&kb_root);
    forget_cmd.arg("--force").arg("forget").arg(&src_id);
    let output = forget_cmd.output().expect("run kb forget");
    assert!(
        output.status.success(),
        "forget failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let mut status_cmd = kb_cmd(&kb_root);
    status_cmd.arg("--json").arg("status");
    let status_output = status_cmd.output().expect("run kb status");
    assert!(status_output.status.success());
    let envelope: Value =
        serde_json::from_slice(&status_output.stdout).expect("parse status");
    assert_eq!(envelope["data"]["normalized_source_count"], 0);
    assert_eq!(envelope["data"]["wiki_pages"], 0);
    let missing = envelope["data"]["sources_with_missing_origin"]
        .as_array()
        .expect("missing array");
    assert!(
        missing.is_empty(),
        "forgotten source should not linger in sources_with_missing_origin: {missing:?}"
    );
}

// ---------------------------------------------------------------------------
// `kb forget` cascade — bn-did F3a/F3b/F3c
// ---------------------------------------------------------------------------

/// Drop a minimal `wiki/concepts/<slug>.md` with a `source_document_ids` list.
fn stub_concept_page(kb_root: &Path, slug: &str, source_ids: &[&str]) -> PathBuf {
    let dir = kb_root.join("wiki/concepts");
    fs::create_dir_all(&dir).expect("mkdir wiki/concepts");
    let ids = source_ids
        .iter()
        .map(|id| format!("  - {id}"))
        .collect::<Vec<_>>()
        .join("\n");
    let body = format!(
        "---\nid: concept-{slug}\nname: {slug}\nsource_document_ids:\n{ids}\n---\n\n# {slug}\n",
    );
    let path = dir.join(format!("{slug}.md"));
    fs::write(&path, body).expect("write concept page");
    path
}

/// Drop a minimal `wiki/questions/<slug>.md` with a `source_document_ids` list.
fn stub_question_page(kb_root: &Path, slug: &str, source_ids: &[&str]) -> PathBuf {
    let dir = kb_root.join("wiki/questions");
    fs::create_dir_all(&dir).expect("mkdir wiki/questions");
    let ids = source_ids
        .iter()
        .map(|id| format!("  - {id}"))
        .collect::<Vec<_>>()
        .join("\n");
    let body = format!(
        "---\nid: q-{slug}\ntitle: {slug}\nsource_document_ids:\n{ids}\n---\n\n# {slug}\n",
    );
    let path = dir.join(format!("{slug}.md"));
    fs::write(&path, body).expect("write question page");
    path
}

/// Write a `state/build_records/<id>.json` whose `metadata.output_paths`
/// points at `normalized/<src>/...`, simulating the `build:source-summary:<src>`
/// record observed in pass-9.
fn stub_source_build_record(kb_root: &Path, src_id: &str) -> PathBuf {
    let record_id = format!("build:source-summary:{src_id}");
    let mut metadata = test_metadata(&record_id);
    metadata
        .output_paths
        .push(PathBuf::from(format!("normalized/{src_id}/summary.md")));
    save_build_record(
        kb_root,
        &BuildRecord {
            metadata,
            pass_name: "source_summary".to_string(),
            input_ids: vec![src_id.to_string()],
            output_ids: vec![record_id.clone()],
            manifest_hash: "m1".to_string(),
        },
    )
    .expect("save build record");
    kb_root
        .join("state/build_records")
        .join(format!("{record_id}.json"))
}

/// bn-did F3b: concepts solely sourced from the forgotten src get trashed,
/// concepts grounded by additional sources survive, and the trash bundle
/// receives the orphans alongside the source.
#[test]
fn forget_cascade_trashes_orphaned_concept_pages() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let source = kb_root.join("note.md");
    fs::write(&source, "# hi\n\ncontent\n").expect("write source");
    let src_id = ingest_single_and_get_src_id(&kb_root, &source);
    stub_wiki_source_page(&kb_root, &src_id);

    let orphan_a = stub_concept_page(&kb_root, "zab", &[&src_id]);
    let orphan_b = stub_concept_page(&kb_root, "zookeeper", &[&src_id]);
    let survivor =
        stub_concept_page(&kb_root, "shared", &[&src_id, "src-deadbeef"]);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--force").arg("forget").arg(&src_id);
    let output = cmd.output().expect("run kb forget");
    assert!(
        output.status.success(),
        "kb forget failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(!orphan_a.exists(), "orphaned concept zab should be trashed");
    assert!(
        !orphan_b.exists(),
        "orphaned concept zookeeper should be trashed"
    );
    assert!(
        survivor.exists(),
        "multi-sourced concept must NOT be trashed: {survivor:?}"
    );

    // Orphans land under `trash/<src>-*/wiki/concepts/`.
    let trash_root = kb_root.join("trash");
    let bundle = fs::read_dir(&trash_root)
        .expect("read trash")
        .filter_map(Result::ok)
        .find(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with(&format!("{src_id}-"))
        })
        .expect("trash bundle for src")
        .path();
    assert!(bundle.join("wiki/concepts/zab.md").exists());
    assert!(bundle.join("wiki/concepts/zookeeper.md").exists());
}

/// bn-did F3b: promoted question pages citing the forgotten source are
/// flagged with `orphaned_sources:` in frontmatter (not rewritten, not
/// removed).
#[test]
fn forget_cascade_flags_promoted_question_pages() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let source = kb_root.join("seed.md");
    fs::write(&source, "# hi\n\nbody\n").expect("write source");
    let src_id = ingest_single_and_get_src_id(&kb_root, &source);
    stub_wiki_source_page(&kb_root, &src_id);

    let question = stub_question_page(&kb_root, "how-does-x", &[&src_id]);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--force").arg("forget").arg(&src_id);
    let output = cmd.output().expect("run kb forget");
    assert!(
        output.status.success(),
        "kb forget failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Page survives.
    assert!(question.exists(), "question page must not be trashed");
    let body = fs::read_to_string(&question).expect("read question");
    assert!(
        body.contains("orphaned_sources:"),
        "question frontmatter must carry orphaned_sources marker:\n{body}"
    );
    assert!(
        body.contains(&src_id),
        "marker must list the forgotten src_id"
    );
}

/// bn-did F3b: build records whose `metadata.output_paths` reference
/// `normalized/<src>/...` or `wiki/sources/<src>.md` are moved to trash.
#[test]
fn forget_cascade_trashes_stale_build_records() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let source = kb_root.join("seed.md");
    fs::write(&source, "# hi\n\nbody\n").expect("write source");
    let src_id = ingest_single_and_get_src_id(&kb_root, &source);
    stub_wiki_source_page(&kb_root, &src_id);
    let record_path = stub_source_build_record(&kb_root, &src_id);
    assert!(record_path.exists(), "sanity: build record written");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--force").arg("forget").arg(&src_id);
    let output = cmd.output().expect("run kb forget");
    assert!(
        output.status.success(),
        "kb forget failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(!record_path.exists(), "stale build record must be trashed");
    let bundle = fs::read_dir(kb_root.join("trash"))
        .expect("read trash")
        .filter_map(Result::ok)
        .find(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with(&format!("{src_id}-"))
        })
        .expect("trash bundle")
        .path();
    let trashed_record = bundle
        .join("state/build_records")
        .join(format!("build:source-summary:{src_id}.json"));
    assert!(
        trashed_record.exists(),
        "build record preserved under trash: expected {trashed_record:?}"
    );
}

/// bn-3f6: both `build:source-summary:<src>` AND
/// `build:extract-concepts:<src>` records are scooped up by the cascade.
/// Before bn-3f6, only source-summary matched (via `output_paths` under
/// `wiki/sources/<src>.md`); extract-concepts was missed because its
/// `output_paths` points at `state/concept_candidates/<src>.json`.
#[test]
fn forget_cascade_trashes_both_source_summary_and_extract_concepts_records() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let source = kb_root.join("seed.md");
    fs::write(&source, "# hi\n\nbody\n").expect("write source");
    let src_id = ingest_single_and_get_src_id(&kb_root, &source);
    stub_wiki_source_page(&kb_root, &src_id);

    // Stub both record types. source-summary → wiki/sources/<src>.md.
    let summary_record = stub_source_build_record(&kb_root, &src_id);
    // extract-concepts → state/concept_candidates/<src>.json.
    let extract_record_id = format!("build:extract-concepts:{src_id}");
    let mut extract_meta = test_metadata(&extract_record_id);
    extract_meta
        .output_paths
        .push(PathBuf::from(format!(
            "state/concept_candidates/{src_id}.json"
        )));
    save_build_record(
        &kb_root,
        &BuildRecord {
            metadata: extract_meta,
            pass_name: "extract_concepts".to_string(),
            input_ids: vec![src_id.clone()],
            output_ids: vec![extract_record_id.clone()],
            manifest_hash: "m2".to_string(),
        },
    )
    .expect("save extract-concepts build record");
    let extract_record = kb_root
        .join("state/build_records")
        .join(format!("{extract_record_id}.json"));
    assert!(summary_record.exists(), "sanity: summary record written");
    assert!(extract_record.exists(), "sanity: extract record written");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--force").arg("forget").arg(&src_id);
    let output = cmd.output().expect("run kb forget");
    assert!(
        output.status.success(),
        "kb forget failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(
        !summary_record.exists(),
        "source-summary build record must be trashed"
    );
    assert!(
        !extract_record.exists(),
        "extract-concepts build record must be trashed (bn-3f6 regression guard)"
    );

    // Acceptance: grep -rl <src> under state/ returns nothing.
    for entry in walkdir_files(&kb_root.join("state")) {
        let contents = fs::read_to_string(&entry).unwrap_or_default();
        assert!(
            !contents.contains(&src_id),
            "state/{} still references forgotten src {src_id}",
            entry
                .strip_prefix(&kb_root)
                .unwrap_or(&entry)
                .display()
        );
    }

    // Cascade plan (stdout) lists BOTH records.
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains(&format!("build:source-summary:{src_id}")),
        "cascade plan should name source-summary record:\n{stdout}"
    );
    assert!(
        stdout.contains(&format!("build:extract-concepts:{src_id}")),
        "cascade plan should name extract-concepts record:\n{stdout}"
    );
}

/// bn-3f6: `state/graph.json` is surgically pruned of nodes referencing the
/// forgotten src, without a full compile rebuild.
#[test]
fn forget_prunes_graph_json_nodes_for_src() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let source = kb_root.join("seed.md");
    fs::write(&source, "# hi\n\nbody\n").expect("write source");
    let src_id = ingest_single_and_get_src_id(&kb_root, &source);
    stub_wiki_source_page(&kb_root, &src_id);

    // Pre-seed graph.json with nodes referencing this src plus an unrelated
    // keepalive lane.
    let mut graph = Graph::default();
    graph.record(
        [format!("source-document-{src_id}")],
        [format!("wiki-page-{src_id}")],
    );
    graph.record(
        ["source-document-src-keepalive"],
        ["wiki-page-src-keepalive"],
    );
    graph.persist_to(&kb_root).expect("persist pre-forget graph");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--force").arg("forget").arg(&src_id);
    let output = cmd.output().expect("run kb forget");
    assert!(
        output.status.success(),
        "kb forget failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Acceptance: graph.json exists and references no forgotten src ids.
    let graph_json = fs::read_to_string(Graph::graph_path(&kb_root))
        .expect("read graph.json after forget");
    assert!(
        !graph_json.contains(&src_id),
        "graph.json still references forgotten src {src_id}:\n{graph_json}"
    );

    // Surgical: unrelated lane survives.
    let reloaded = Graph::load_from(&kb_root).expect("reload graph");
    assert!(
        reloaded.nodes.contains_key("source-document-src-keepalive"),
        "surgical prune must not touch unrelated src: {:?}",
        reloaded.nodes.keys().collect::<Vec<_>>()
    );
    assert!(
        reloaded.nodes.contains_key("wiki-page-src-keepalive"),
        "surgical prune must keep unrelated wiki-page node"
    );
}

/// Walk `dir` recursively and collect every file path into `out`. Tiny
/// helper to avoid pulling in `walkdir` just for the one bn-3f6 acceptance
/// test. Symlinks are followed implicitly by `file_type`.
fn walkdir_visit(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        match entry.file_type() {
            Ok(ft) if ft.is_dir() => walkdir_visit(&path, out),
            Ok(ft) if ft.is_file() => out.push(path),
            _ => {}
        }
    }
}

fn walkdir_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walkdir_visit(dir, &mut out);
    out
}

/// bn-did F3c: `--no-cascade` preserves bn-1fq behavior: orphaned concepts,
/// cited questions, and stale build records stay put.
#[test]
fn forget_no_cascade_preserves_legacy_behavior() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let source = kb_root.join("seed.md");
    fs::write(&source, "# hi\n\nbody\n").expect("write source");
    let src_id = ingest_single_and_get_src_id(&kb_root, &source);
    stub_wiki_source_page(&kb_root, &src_id);

    let orphan = stub_concept_page(&kb_root, "zab", &[&src_id]);
    let question = stub_question_page(&kb_root, "how", &[&src_id]);
    let record_path = stub_source_build_record(&kb_root, &src_id);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--force")
        .arg("forget")
        .arg("--no-cascade")
        .arg(&src_id);
    let output = cmd.output().expect("run kb forget --no-cascade");
    assert!(
        output.status.success(),
        "kb forget --no-cascade failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Source itself was forgotten.
    assert!(!kb_root.join("normalized").join(&src_id).exists());
    // Cascade targets were NOT touched.
    assert!(orphan.exists(), "concept should survive --no-cascade");
    let q_body = fs::read_to_string(&question).expect("read question");
    assert!(
        !q_body.contains("orphaned_sources:"),
        "question must NOT be flagged under --no-cascade:\n{q_body}"
    );
    assert!(record_path.exists(), "build record must survive --no-cascade");
}

/// bn-did F3a: `--dry-run` includes the cascade plan in JSON output and
/// moves no files.
#[test]
fn forget_dry_run_reports_cascade_items() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let source = kb_root.join("seed.md");
    fs::write(&source, "# hi\n\nbody\n").expect("write source");
    let src_id = ingest_single_and_get_src_id(&kb_root, &source);
    stub_wiki_source_page(&kb_root, &src_id);

    let orphan = stub_concept_page(&kb_root, "zab", &[&src_id]);
    let question = stub_question_page(&kb_root, "how", &[&src_id]);
    let record_path = stub_source_build_record(&kb_root, &src_id);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--dry-run")
        .arg("--json")
        .arg("forget")
        .arg(&src_id);
    let output = cmd.output().expect("run kb forget --dry-run --json");
    assert!(
        output.status.success(),
        "dry-run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value =
        serde_json::from_slice(&output.stdout).expect("parse json");
    assert_eq!(envelope["data"]["dry_run"], true);
    let cascade = &envelope["data"]["plan"]["cascade"];
    let concepts = cascade["orphaned_concept_pages"]
        .as_array()
        .expect("orphaned_concept_pages array");
    assert_eq!(
        concepts.len(),
        1,
        "one orphaned concept expected, got {concepts:?}"
    );
    let questions = cascade["flagged_question_pages"]
        .as_array()
        .expect("flagged_question_pages array");
    assert_eq!(questions.len(), 1);
    let records = cascade["stale_build_records"]
        .as_array()
        .expect("stale_build_records array");
    assert_eq!(records.len(), 1);

    // Nothing touched on disk.
    assert!(orphan.exists(), "dry-run must not move concepts");
    let q_body = fs::read_to_string(&question).expect("read question");
    assert!(
        !q_body.contains("orphaned_sources:"),
        "dry-run must not flag questions"
    );
    assert!(record_path.exists(), "dry-run must not move build records");
}

/// bn-i5r F1/F2/F3: after `kb forget --force` on a cascade-heavy setup
/// (3 orphan concepts + 1 survivor that loses one of its two srcs), the KB
/// must be immediately query-able:
///   - `wiki/*/index.md` no longer lists the trashed concepts or source,
///   - the surviving concept's frontmatter no longer cites the forgotten src,
///   - `state/indexes/lexical.json` only references surviving pages,
///   - `kb lint` reports no orphan / broken-links errors for the forgotten src,
///   - `kb ask` runs cleanly (the old "No such file or directory" crash).
#[test]
#[allow(clippy::too_many_lines)]
fn forget_cascade_fully_refreshes_index_pages_frontmatter_and_lexical() {
    use kb_query::LexicalIndex;

    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let source = kb_root.join("seed.md");
    fs::write(&source, "# hi\n\nbody\n").expect("write source");
    let src_id = ingest_single_and_get_src_id(&kb_root, &source);
    // Give forget a wiki source page to remove (avoids full kb compile).
    stub_wiki_source_page(&kb_root, &src_id);

    // Populate wiki/concepts/ with real post-compile layout: 3 orphans, 1
    // survivor with a second source, and one unrelated concept.
    let orphan_a = stub_concept_page(&kb_root, "orphan-a", &[&src_id]);
    let orphan_b = stub_concept_page(&kb_root, "orphan-b", &[&src_id]);
    let orphan_c = stub_concept_page(&kb_root, "orphan-c", &[&src_id]);
    let survivor = stub_concept_page(&kb_root, "survivor", &[&src_id, "src-deadbeef"]);
    let unrelated = stub_concept_page(&kb_root, "unrelated", &["src-deadbeef"]);

    // Seed a stale lexical index and stale index pages referencing everything
    // — the state the KB is left in after `kb compile` but before `kb forget`.
    let pre_index = kb_query::build_lexical_index(&kb_root).expect("pre-index");
    pre_index.save(&kb_root).expect("save pre-index");
    let pre_artifacts =
        kb_compile::index_page::generate_indexes(&kb_root).expect("pre-index-pages");
    kb_compile::index_page::persist_index_artifacts(&pre_artifacts)
        .expect("persist pre-index-pages");
    // Sanity: pre-state references everything we're about to forget.
    let pre_lex = fs::read_to_string(kb_root.join("state/indexes/lexical.json"))
        .expect("read pre-lex");
    assert!(
        pre_lex.contains(&format!("wiki/sources/{src_id}.md")),
        "pre-state lexical index must reference the source page; got:\n{pre_lex}"
    );
    assert!(
        pre_lex.contains("wiki/concepts/orphan-a.md"),
        "pre-state lexical index must reference orphan concepts"
    );

    // Now forget the source — cascade + all three bn-i5r refreshes.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--force").arg("forget").arg(&src_id);
    let output = cmd.output().expect("run kb forget");
    assert!(
        output.status.success(),
        "kb forget failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // 1. Source + orphans moved to trash, survivor + unrelated preserved.
    assert!(!orphan_a.exists());
    assert!(!orphan_b.exists());
    assert!(!orphan_c.exists());
    assert!(survivor.exists());
    assert!(unrelated.exists());

    // 2. Surviving concept's frontmatter no longer cites the forgotten src.
    let survivor_body = fs::read_to_string(&survivor).expect("read survivor");
    assert!(
        !survivor_body.contains(&src_id),
        "surviving concept frontmatter still lists forgotten src:\n{survivor_body}"
    );
    assert!(
        survivor_body.contains("src-deadbeef"),
        "surviving concept must still list its other src:\n{survivor_body}"
    );

    // 3. Index pages no longer list trashed pages.
    let global_index = fs::read_to_string(kb_root.join("wiki/index.md"))
        .expect("read wiki/index.md");
    assert!(
        !global_index.contains("orphan-a.md"),
        "global index still lists trashed concept:\n{global_index}"
    );
    assert!(
        !global_index.contains(&format!("sources/{src_id}.md")),
        "global index still lists trashed source page:\n{global_index}"
    );
    let concepts_index = fs::read_to_string(kb_root.join("wiki/concepts/index.md"))
        .expect("read wiki/concepts/index.md");
    assert!(!concepts_index.contains("orphan-a.md"));
    assert!(!concepts_index.contains("orphan-b.md"));
    assert!(!concepts_index.contains("orphan-c.md"));
    assert!(concepts_index.contains("survivor.md"));
    assert!(concepts_index.contains("unrelated.md"));

    // 4. Lexical index was rebuilt from on-disk state.
    let post_index =
        LexicalIndex::load(&kb_root).expect("reload lexical index after forget");
    let ids: Vec<&str> = post_index.entries.iter().map(|e| e.id.as_str()).collect();
    assert!(
        !ids.iter().any(|id| id.contains(&format!("sources/{src_id}.md"))),
        "lexical index still points at trashed source page: {ids:?}"
    );
    assert!(
        !ids.iter().any(|id| id.contains("orphan-")),
        "lexical index still points at trashed concepts: {ids:?}"
    );
    assert!(
        ids.contains(&"wiki/concepts/survivor.md"),
        "lexical index must retain surviving concept: {ids:?}"
    );

    // 5. `kb lint --check orphans` no longer flags the forgotten src.
    let mut lint_cmd = kb_cmd(&kb_root);
    lint_cmd.arg("--json").arg("lint").arg("--check").arg("orphans");
    let lint_output = lint_cmd.output().expect("run kb lint --check orphans");
    let lint_stdout = String::from_utf8_lossy(&lint_output.stdout);
    assert!(
        !lint_stdout.contains(&format!("refers to source_document_id {src_id}")),
        "orphan-lint still mentions forgotten src:\n{lint_stdout}"
    );

    // 6. `kb lint --check broken-links` reports no errors pointing into trash.
    let mut broken_cmd = kb_cmd(&kb_root);
    broken_cmd
        .arg("--json")
        .arg("lint")
        .arg("--check")
        .arg("broken-links");
    let broken_output = broken_cmd
        .output()
        .expect("run kb lint --check broken-links");
    let broken_stdout = String::from_utf8_lossy(&broken_output.stdout);
    assert!(
        !broken_stdout.contains("trash/"),
        "broken-links lint still points into trash/:\n{broken_stdout}"
    );

    // 7. `kb ask` runs cleanly (no "No such file or directory" crash).
    let mut ask_cmd = kb_cmd(&kb_root);
    ask_cmd.arg("--json").arg("ask").arg("survivor concept");
    let ask_output = ask_cmd.output().expect("run kb ask");
    assert!(
        ask_output.status.success(),
        "kb ask crashed after forget — stale lexical index not refreshed:\nstderr: {}\nstdout: {}",
        String::from_utf8_lossy(&ask_output.stderr),
        String::from_utf8_lossy(&ask_output.stdout)
    );
    let ask_stderr = String::from_utf8_lossy(&ask_output.stderr);
    assert!(
        !ask_stderr.contains("No such file or directory"),
        "kb ask surfaced stale lexical candidates:\n{ask_stderr}"
    );
}

/// bn-i5r F3/F4: `--dry-run` previews the three post-trash refresh steps
/// (index pages, lexical index, frontmatter scrub count) without touching
/// disk.
#[test]
fn forget_dry_run_mentions_post_trash_refresh_steps() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let source = kb_root.join("seed.md");
    fs::write(&source, "# hi\n\nbody\n").expect("write source");
    let src_id = ingest_single_and_get_src_id(&kb_root, &source);
    stub_wiki_source_page(&kb_root, &src_id);

    // One orphan + one survivor → 1 scrub candidate in the preview.
    stub_concept_page(&kb_root, "orphan", &[&src_id]);
    stub_concept_page(&kb_root, "survivor", &[&src_id, "src-deadbeef"]);

    let pre_index = kb_query::build_lexical_index(&kb_root).expect("pre-index");
    pre_index.save(&kb_root).expect("save pre-index");

    // Text dry-run: footer names each refresh target.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--dry-run").arg("forget").arg(&src_id);
    let output = cmd.output().expect("run kb forget --dry-run");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("will also refresh"),
        "dry-run must mention the refresh plan:\n{stdout}"
    );
    assert!(stdout.contains("wiki/index.md"), "names global index");
    assert!(
        stdout.contains("wiki/concepts/index.md"),
        "names concepts index"
    );
    assert!(
        stdout.contains("wiki/sources/index.md"),
        "names sources index"
    );
    assert!(
        stdout.contains("lexical.json"),
        "names lexical index target"
    );
    assert!(
        stdout.contains("scrub on 1 page"),
        "names scrub count; got:\n{stdout}"
    );

    // JSON dry-run: `cascade_refresh` previews the three steps.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--dry-run")
        .arg("--json")
        .arg("forget")
        .arg(&src_id);
    let output = cmd
        .output()
        .expect("run kb forget --dry-run --json");
    assert!(output.status.success());
    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse json");
    let refresh = &envelope["data"]["cascade_refresh"];
    assert_eq!(refresh["index_pages_refreshed"], true);
    assert_eq!(refresh["lexical_index_refreshed"], true);
    assert_eq!(refresh["frontmatter_scrubbed"], 1);

    // Nothing was actually moved or rewritten.
    assert!(
        kb_root.join("wiki/concepts/orphan.md").exists(),
        "dry-run must not trash the orphan"
    );
    let survivor_body = fs::read_to_string(kb_root.join("wiki/concepts/survivor.md"))
        .expect("read survivor");
    assert!(
        survivor_body.contains(&src_id),
        "dry-run must not scrub the survivor frontmatter:\n{survivor_body}"
    );
}

/// bn-did acceptance: after a cascade forget, `kb lint` reports 0 orphan
/// errors — the orphan-lint rule that triggered the bone is silenced.
#[test]
fn lint_after_cascade_forget_has_no_orphan_errors() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let source = kb_root.join("seed.md");
    fs::write(&source, "# hi\n\nbody\n").expect("write source");
    let src_id = ingest_single_and_get_src_id(&kb_root, &source);
    stub_wiki_source_page(&kb_root, &src_id);
    stub_concept_page(&kb_root, "zab", &[&src_id]);
    stub_concept_page(&kb_root, "zookeeper", &[&src_id]);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--force").arg("forget").arg(&src_id);
    let output = cmd.output().expect("run kb forget");
    assert!(output.status.success());

    let mut lint_cmd = kb_cmd(&kb_root);
    lint_cmd.arg("--json").arg("lint");
    let lint_output = lint_cmd.output().expect("run kb lint");
    // Lint may still exit non-zero for other issues, but the `orphans`
    // check for our forgotten src should report zero issues.
    let stdout = String::from_utf8_lossy(&lint_output.stdout);
    // Cheap guard: the orphan messages would name src_id; after cascade,
    // no concept page still points at it.
    assert!(
        !stdout.contains(&format!("refers to source_document_id {src_id}")),
        "lint still flags orphaned concepts after cascade forget:\n{stdout}"
    );
}

/// Count job manifests (`state/jobs/*.json`) under a KB root. Used by the
/// bn-1jx acceptance tests to assert that validation rejections leave no
/// trace in the jobs directory.
fn count_job_manifests(root: &Path) -> usize {
    let jobs_dir = root.join("state/jobs");
    if !jobs_dir.exists() {
        return 0;
    }
    fs::read_dir(&jobs_dir)
        .expect("read state/jobs")
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .path()
                .extension()
                .and_then(|ext| ext.to_str())
                == Some("json")
        })
        .count()
}

/// bn-1jx acceptance: `kb ask --format=png` is a pure input-validation
/// rejection (we reject before any LLM call or file write). It must exit
/// 1 with a clear message AND leave no failed-job manifest behind, so
/// `kb status` / `kb doctor` stay clean.
#[test]
fn ask_format_png_leaves_no_failed_job_manifest() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("ask").arg("--format").arg("png").arg("x");
    let output = cmd.output().expect("run kb ask --format=png");
    assert!(
        !output.status.success(),
        "kb ask --format=png must still exit non-zero"
    );
    assert_eq!(
        count_job_manifests(&kb_root),
        0,
        "validation rejection must not write a job manifest; manifests present: {:?}",
        fs::read_dir(kb_root.join("state/jobs"))
            .ok()
            .map(|iter| iter.filter_map(Result::ok).map(|e| e.file_name()).collect::<Vec<_>>())
    );

    // `kb status --json` reports zero failed jobs.
    let mut status_cmd = kb_cmd(&kb_root);
    status_cmd.arg("--json").arg("status");
    let status_output = status_cmd.output().expect("run kb --json status");
    assert!(status_output.status.success(), "kb status failed");
    let envelope: Value =
        serde_json::from_slice(&status_output.stdout).expect("parse status json");
    assert_eq!(
        envelope["data"]["failed_jobs_total"], 0,
        "validation rejection must not appear as a failed job"
    );
    assert!(
        envelope["data"]["failed_jobs"]
            .as_array()
            .expect("failed_jobs array")
            .is_empty(),
        "failed_jobs list must be empty after a pure validation rejection"
    );
}

/// bn-1jx acceptance: `kb ingest /nonexistent` — the path-does-not-exist
/// check fires before any normalize work. Must not count as a failed job.
#[test]
fn ingest_nonexistent_path_leaves_no_failed_job_manifest() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let missing = kb_root.join("does-not-exist.md");
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("ingest").arg(&missing);
    let output = cmd.output().expect("run kb ingest");
    assert!(
        !output.status.success(),
        "kb ingest <missing> must exit non-zero"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("does not exist"),
        "error message must mention the missing path; got: {stderr}"
    );
    assert_eq!(
        count_job_manifests(&kb_root),
        0,
        "missing-path rejection must not write a job manifest"
    );
}

/// bn-1jx acceptance: `kb publish <unknown-target>` rejects in the CLI
/// dispatch, before the publish job would otherwise acquire the root
/// lock. No manifest should be left behind.
#[test]
fn publish_unknown_target_leaves_no_failed_job_manifest() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("publish").arg("no-such-target");
    let output = cmd.output().expect("run kb publish");
    assert!(
        !output.status.success(),
        "kb publish <unknown> must exit non-zero"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no-such-target") && stderr.contains("not found"),
        "error message must name the target and say 'not found'; got: {stderr}"
    );
    assert_eq!(
        count_job_manifests(&kb_root),
        0,
        "unknown publish target must not write a job manifest"
    );
}

/// bn-1jx acceptance: `kb review approve <unknown-id>` rejects up front.
/// No manifest should be left behind.
#[test]
fn review_approve_unknown_id_leaves_no_failed_job_manifest() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("review").arg("approve").arg("review-no-such-id");
    let output = cmd.output().expect("run kb review approve");
    assert!(
        !output.status.success(),
        "kb review approve <unknown> must exit non-zero"
    );
    assert_eq!(
        count_job_manifests(&kb_root),
        0,
        "unknown review id must not write a job manifest"
    );
}

/// bn-1jx acceptance: real system failures (not validation) MUST still
/// be recorded as failed jobs. This is the contrast case for the tests
/// above — we must not over-classify and hide actual bugs.
///
/// We simulate a system failure by seeding a `JobRun` manifest directly
/// on disk (same shape the CLI writes when the action's closure bubbles
/// a non-`ValidationError`). `kb status --json` must count it.
#[test]
fn real_failure_is_still_recorded_as_failed_job() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // Seed a system-failure manifest directly — the same shape
    // `execute_mutating_command_with_handle` writes when the inner
    // action returns a non-validation error (e.g. an LLM timeout).
    seed_failed_job(&kb_root, "fail-real-llm-timeout", 1_000);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json").arg("status");
    let output = cmd.output().expect("run kb --json status");
    assert!(output.status.success());
    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse status json");
    assert_eq!(
        envelope["data"]["failed_jobs_total"], 1,
        "a real system failure must be counted in failed_jobs_total"
    );
    let ids: Vec<String> = envelope["data"]["failed_jobs"]
        .as_array()
        .expect("failed_jobs array")
        .iter()
        .map(|job| {
            job["metadata"]["id"]
                .as_str()
                .expect("metadata.id")
                .to_string()
        })
        .collect();
    assert!(
        ids.iter().any(|id| id == "fail-real-llm-timeout"),
        "expected the seeded real failure to show up in failed_jobs; got: {ids:?}"
    );
}
