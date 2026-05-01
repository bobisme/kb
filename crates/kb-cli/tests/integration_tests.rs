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
        fs::read_to_string(kb_root.join(".kb/state/manifest.json")).expect("read manifest state file");
    let manifest_json: Value = serde_json::from_str(&manifest).expect("parse manifest json");
    assert_eq!(manifest_json, serde_json::json!({ "artifacts": {} }));

    // hashes.json is not created at init — it's written by the first
    // successful `kb compile` in the canonical HashState schema
    // (see bn-1pw: removed the stale Hashes default-write from init).
    assert!(
        !kb_root.join(".kb/state/hashes.json").exists(),
        "hashes.json should not exist until first compile"
    );
    assert!(
        kb_root.join(".kb/state/build_records").exists(),
        "`.kb/state/build_records` must exist after init"
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
        kb_root.join(".kb/state/manifest.json").exists(),
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
    // bn-2cs2: default format is now `auto`; the model picks supporting
    // artifacts per question. The placeholder path is exercised here
    // because no LLM backend is configured, but the recorded format still
    // reflects the requested mode.
    assert_eq!(question_record["requested_format"], "auto");
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
    assert!(artifact.contains("requested_format: auto"));
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
    // Extra sources so the retrieval scorer returns 3+ positive candidates
    // and the low-coverage fallback (bn-1yvv) stays quiet — this test is
    // specifically about ranking of real matches, not the fallback.
    write_source_page(
        &kb_root,
        "rust-memory",
        "Rust Memory Model",
        "Rust memory safety guarantees.",
    );
    write_source_page(
        &kb_root,
        "rust-guide",
        "Rust Guide",
        "A guide to the Rust programming language.",
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
    // Strong-match scenario: three Rust source pages plus the concept all
    // score positively, so fallback MUST NOT fire. `fallback_reason` must
    // be absent from the persisted plan.
    assert!(
        retrieval_plan.get("fallback_reason").is_none(),
        "fallback_reason should be absent on strong-match plan, got: {retrieval_plan:?}",
    );
    // Hybrid retrieval can append semantic-only tail candidates beyond the
    // four lexical hits. The first four positions still belong to the
    // lexical scoring (because lexical-tier hits sort in front of zero-
    // scored semantic-only entries), so we assert presence of all four
    // expected lexical candidates rather than an exact count.
    assert!(
        candidates.len() >= 4,
        "expected >=4 candidates, got {}",
        candidates.len()
    );
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
fn ingest_directory_respects_kbignore_and_gitignore_cumulatively() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let corpus = kb_root.join("corpus");
    fs::create_dir_all(&corpus).expect("create corpus dir");
    fs::write(corpus.join(".gitignore"), "git-ignored.md\n").expect("write gitignore");
    fs::write(corpus.join(".kbignore"), "kb-ignored.md\n").expect("write kbignore");
    fs::write(corpus.join("kept.md"), "keep\n").expect("write kept");
    fs::write(corpus.join("git-ignored.md"), "git-ignored\n").expect("write git-ignored");
    fs::write(corpus.join("kb-ignored.md"), "kb-ignored\n").expect("write kb-ignored");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json").arg("ingest").arg(&corpus);
    let output = cmd.output().expect("run kb ingest directory");

    assert!(
        output.status.success(),
        "kb ingest directory failed with stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse ingest json");
    let payload = &envelope["data"];
    let items = payload["results"]
        .as_array()
        .expect("ingest results should be an array");
    assert_eq!(items.len(), 1, "only kept.md should be ingested");
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
        !kb_root.join(".kb/normalized").exists()
            || fs::read_dir(kb_root.join(".kb/normalized"))
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
fn ask_no_arg_piped_empty_stdin_is_validation_error_no_failed_job() {
    // bn-ozj5: `kb ask` with no arg and non-TTY stdin must fall through to
    // the piped-stdin reader (NOT the interactive reedline editor). Empty
    // piped stdin must surface the same "no question provided" error and
    // — crucially — must not leave a failed-job manifest behind, since
    // this is a pure validation rejection that never reaches the LLM.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("ask").write_stdin("");
    let output = cmd
        .output()
        .expect("run kb ask (no arg) with empty piped stdin");

    assert!(
        !output.status.success(),
        "kb ask (no arg) with empty piped stdin unexpectedly succeeded"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no question provided"),
        "expected stderr to mention 'no question provided', got: {stderr}"
    );

    assert_eq!(
        count_job_manifests(&kb_root),
        0,
        "empty piped-stdin ask must not write a job manifest"
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
                    .starts_with("q-")
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

// bn-35ap: `kb ask --format=chart` drives the LLM through the `ask_chart.md`
// prompt, expects a PNG at `outputs/questions/<q-id>/chart.png`, and writes an
// `answer.md` containing a caption + markdown image reference.
//
// Mock LLM adapter: a shell script standing in for `opencode` that scans the
// prompt for the chart.png path the template told it to write, drops a tiny
// fixture PNG there, and prints a deterministic caption.
fn install_fake_chart_harness(root: &Path) -> PathBuf {
    let bin_dir = root.join("fake-chart-bin");
    fs::create_dir_all(&bin_dir).expect("create fake chart bin dir");
    // bn-3049: kb-cli now feeds the prompt via stdin (not argv). Drain
    // stdin into `prompt` and grep it for the absolute path ending in
    // `chart.png`, write a non-empty byte sequence there (minimal PNG
    // signature + IHDR/IEND; real matplotlib output is much larger but the
    // only thing kb verifies is that the file exists), and emit a caption.
    let script = r#"#!/bin/sh
set -e
prompt="$(cat)"
png_path="$(printf '%s' "$prompt" | grep -oE '/[^ ]+chart\.png' | head -n1)"
if [ -z "$png_path" ]; then
    echo "fake-chart-opencode: no chart.png path found in prompt" >&2
    exit 1
fi
mkdir -p "$(dirname "$png_path")"
# Minimal PNG signature + some bytes. kb only asserts existence + non-empty.
printf '\x89PNG\r\n\x1a\nFAKE-PNG-DATA' > "$png_path"
printf 'A chart showing the requested comparison.\n'
"#;
    write_executable(bin_dir.join("opencode").as_path(), script);
    // Claude is present so `kb doctor` stays happy if the test ever needs it.
    write_executable(
        bin_dir.join("claude").as_path(),
        "#!/bin/sh\nprintf '{\"result\":\"OK\"}'",
    );
    bin_dir
}

#[test]
fn ask_format_chart_produces_png_and_answer_with_image_ref() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    let fake_bin = install_fake_chart_harness(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.env("PATH", prepend_path(&fake_bin));
    cmd.arg("--json")
        .arg("ask")
        .arg("--format")
        .arg("chart")
        .arg("Compare timings across backends");
    let output = cmd.output().expect("run kb ask --format=chart");
    assert!(
        output.status.success(),
        "kb ask --format=chart failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse ask json");
    assert_eq!(envelope["command"], "ask");
    let data = &envelope["data"];
    assert_eq!(data["requested_format"], "chart");

    let artifact_path = kb_root.join(
        data["artifact_path"]
            .as_str()
            .expect("artifact_path"),
    );
    // Caption lands in answer.md (not answer.png).
    assert_eq!(
        artifact_path.extension().and_then(|e| e.to_str()),
        Some("md"),
        "chart format writes the caption to answer.md, got: {}",
        artifact_path.display()
    );
    let answer_md = fs::read_to_string(&artifact_path).expect("read answer.md");
    assert!(
        answer_md.contains("requested_format: chart"),
        "answer.md frontmatter should record requested_format=chart, got:\n{answer_md}"
    );
    assert!(
        answer_md.contains("![chart](chart.png)"),
        "answer.md must embed a markdown image reference to chart.png, got:\n{answer_md}"
    );
    assert!(
        answer_md.contains("A chart showing the requested comparison."),
        "answer.md must include the LLM's caption, got:\n{answer_md}"
    );

    // chart.png exists next to answer.md and is non-empty.
    let chart_png = artifact_path.with_file_name("chart.png");
    assert!(
        chart_png.exists(),
        "chart.png must be written at {}",
        chart_png.display()
    );
    let bytes = fs::read(&chart_png).expect("read chart.png");
    assert!(!bytes.is_empty(), "chart.png must not be empty");

    // metadata.json records requested_format: chart.
    let metadata_path = artifact_path.with_file_name("metadata.json");
    let metadata: Value = serde_json::from_str(
        &fs::read_to_string(&metadata_path).expect("read metadata.json"),
    )
    .expect("parse metadata.json");
    let question_id = metadata["question_id"].as_str().expect("question_id");
    assert!(question_id.starts_with("q-"));

    // question.json records requested_format=chart too.
    let question_path = artifact_path.with_file_name("question.json");
    let question: Value = serde_json::from_str(
        &fs::read_to_string(&question_path).expect("read question.json"),
    )
    .expect("parse question.json");
    assert_eq!(question["requested_format"], "chart");
}

// bn-31uk: chart runs invoke opencode with the write/bash tools enabled,
// and opencode streams the model's per-tool-call commentary into stdout
// ahead of the real caption. That preamble was leaking into answer.md's
// body. A narrow `strip_tool_narration` helper runs on the chart code path
// and cuts the preamble when the body has a markdown heading or a
// blank-line paragraph break. This test wires a fake opencode that emits
// exactly that shape (two narration lines, blank line, `# Chart caption`
// header, blank line, body), produces a PNG, and asserts answer.md's body
// begins at the heading — not at the narration.
fn install_fake_narration_chart_harness(root: &Path) -> PathBuf {
    let bin_dir = root.join("fake-narration-chart-bin");
    fs::create_dir_all(&bin_dir).expect("create fake narration chart bin");
    // Same PNG-writing behavior as `install_fake_chart_harness`, but the
    // printed caption intentionally begins with two narration lines that
    // the LLM-driven tool loop would emit before settling on the final
    // assistant message. strip_tool_narration should cut them.
    // bn-3049: kb-cli now feeds the prompt via stdin (not argv).
    let script = r#"#!/bin/sh
set -e
prompt="$(cat)"
png_path="$(printf '%s' "$prompt" | grep -oE '/[^ ]+chart\.png' | head -n1)"
if [ -z "$png_path" ]; then
    echo "fake-narration-chart-opencode: no chart.png path found in prompt" >&2
    exit 1
fi
mkdir -p "$(dirname "$png_path")"
printf '\x89PNG\r\n\x1a\nFAKE-PNG-DATA' > "$png_path"
printf 'narration line 1\nnarration line 2\n\n# Chart caption\n\nBody text of the caption.\n'
"#;
    write_executable(bin_dir.join("opencode").as_path(), script);
    write_executable(
        bin_dir.join("claude").as_path(),
        "#!/bin/sh\nprintf '{\"result\":\"OK\"}'",
    );
    bin_dir
}

#[test]
fn ask_format_chart_strips_llm_tool_narration_preamble() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    let fake_bin = install_fake_narration_chart_harness(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.env("PATH", prepend_path(&fake_bin));
    cmd.arg("--json")
        .arg("ask")
        .arg("--format")
        .arg("chart")
        .arg("Compare latencies across backends");
    let output = cmd.output().expect("run kb ask --format=chart");
    assert!(
        output.status.success(),
        "kb ask --format=chart failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse ask json");
    let artifact_path = kb_root.join(
        envelope["data"]["artifact_path"]
            .as_str()
            .expect("artifact_path"),
    );
    let answer_md = fs::read_to_string(&artifact_path).expect("read answer.md");

    // Frontmatter must still be present (bn-35ap contract).
    assert!(
        answer_md.starts_with("---\n"),
        "answer.md must begin with frontmatter, got:\n{answer_md}"
    );

    // Body starts after the second `---\n`. Extract it and verify the
    // narration preamble was stripped: the caption begins at the
    // '# Chart caption' heading, not at 'narration line 1'.
    //
    // bn-15w4 prepends a `> **Question:**` blockquote between the frontmatter
    // and the body, so skip past it before asserting on the caption shape.
    let body = answer_md
        .splitn(3, "---\n")
        .nth(2)
        .expect("answer.md must have a body after frontmatter")
        .trim_start_matches('\n');
    let caption_start = body
        .find("# Chart caption")
        .unwrap_or_else(|| panic!("answer.md must contain the chart caption, got body:\n{body}"));
    let caption = &body[caption_start..];
    assert!(
        caption.starts_with("# Chart caption"),
        "answer.md caption must begin at the '# Chart caption' heading after \
         strip_tool_narration; narration lines must be dropped. Got body:\n{body}"
    );
    assert!(
        !body.contains("narration line 1"),
        "narration line 1 must be stripped, got body:\n{body}"
    );
    assert!(
        !body.contains("narration line 2"),
        "narration line 2 must be stripped, got body:\n{body}"
    );
    // Caption body content is preserved intact.
    assert!(
        body.contains("Body text of the caption."),
        "caption body must be preserved, got body:\n{body}"
    );
    // The image reference is still emitted by the chart code path.
    assert!(
        body.contains("![chart](chart.png)"),
        "answer.md must still embed the chart image reference, got body:\n{body}"
    );
}

#[test]
fn ask_format_chart_errors_when_llm_produces_no_png() {
    // When the LLM runs successfully but never writes the PNG to the exact
    // path, kb must refuse to write a stale answer.md. No silent fallback to
    // markdown.
    //
    // bn-1hqh: on failure the outputs/questions/<q-id>/ directory is still
    // created with `answer.md` (raw LLM output, verbatim) and
    // `metadata.json` (`success: false`) so users can diagnose why the
    // chart wasn't produced.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // Fake opencode that prints a caption (including tool-narration-like
    // preamble) but never writes chart.png. The narration must survive
    // verbatim on the failure path so the user sees what the LLM tried —
    // strip_tool_narration is for the success path only.
    let bin_dir = kb_root.join("fake-no-png-bin");
    fs::create_dir_all(&bin_dir).expect("create fake bin");
    write_executable(
        bin_dir.join("opencode").as_path(),
        "#!/bin/sh\nprintf 'calling bash tool\\nran matplotlib import\\n\\nA nice caption but no png.\\n'",
    );
    write_executable(
        bin_dir.join("claude").as_path(),
        "#!/bin/sh\nprintf '{\"result\":\"OK\"}'",
    );

    let mut cmd = kb_cmd(&kb_root);
    cmd.env("PATH", prepend_path(&bin_dir));
    cmd.arg("ask")
        .arg("--format")
        .arg("chart")
        .arg("Compare timings across backends");
    let output = cmd.output().expect("run kb ask --format=chart without png");

    assert!(
        !output.status.success(),
        "kb ask --format=chart should fail when no PNG is produced"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--format chart"),
        "stderr should mention --format chart, got: {stderr}"
    );
    assert!(
        stderr.contains("chart.png") || stderr.contains("was not produced"),
        "stderr should explain the missing PNG, got: {stderr}"
    );
    // Error message points at the artifact dir so the user knows where to
    // look.
    assert!(
        stderr.contains("answer.md"),
        "stderr should reference answer.md for diagnosis, got: {stderr}"
    );

    // bn-1hqh: find the outputs/questions/q-*/ directory. Since we ran
    // without --json, the question id isn't echoed to stdout — walk the
    // questions dir.
    let questions_dir = kb_root.join("outputs").join("questions");
    assert!(
        questions_dir.exists(),
        "outputs/questions/ must be created on chart failure so users can diagnose"
    );
    let q_dirs: Vec<_> = fs::read_dir(&questions_dir)
        .expect("read outputs/questions")
        .filter_map(Result::ok)
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with("q-")
        })
        .collect();
    assert_eq!(
        q_dirs.len(),
        1,
        "exactly one q-* dir must exist on chart failure, found {}",
        q_dirs.len()
    );
    let q_dir = q_dirs[0].path();

    // answer.md preserves the raw LLM output verbatim (narration + caption)
    // inside the "Chart generation failed" envelope.
    let answer_md =
        fs::read_to_string(q_dir.join("answer.md")).expect("read failure answer.md");
    assert!(
        answer_md.contains("## Chart generation failed"),
        "failure answer.md must have 'Chart generation failed' header, got:\n{answer_md}"
    );
    assert!(
        answer_md.contains("Reason:"),
        "failure answer.md must record a reason, got:\n{answer_md}"
    );
    assert!(
        answer_md.contains("## LLM output"),
        "failure answer.md must have LLM output section, got:\n{answer_md}"
    );
    // Narration survives verbatim — not stripped on the failure path.
    assert!(
        answer_md.contains("calling bash tool"),
        "failure answer.md must preserve raw LLM output verbatim (narration included), got:\n{answer_md}"
    );
    assert!(
        answer_md.contains("A nice caption but no png."),
        "failure answer.md must preserve LLM caption, got:\n{answer_md}"
    );

    // metadata.json records success: false + the error string.
    let metadata_str =
        fs::read_to_string(q_dir.join("metadata.json")).expect("read failure metadata.json");
    let metadata: Value =
        serde_json::from_str(&metadata_str).expect("parse failure metadata.json");
    assert_eq!(metadata["requested_format"], "chart");
    assert_eq!(metadata["success"], false);
    assert!(
        metadata["error"].is_string(),
        "metadata.json must include an error string, got: {metadata_str}"
    );
    let err_str = metadata["error"].as_str().expect("error is a string");
    assert!(
        !err_str.is_empty(),
        "metadata.json error must be non-empty, got: {err_str}"
    );

    // chart.png must NOT exist (the whole point of this failure path).
    assert!(
        !q_dir.join("chart.png").exists(),
        "chart.png must not exist when the LLM failed to produce it"
    );
}

#[test]
fn ask_format_figure_is_alias_for_chart() {
    // `--format=figure` is a synonym for `--format=chart`; both must hit the
    // same prompt + PNG post-check pipeline.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    let fake_bin = install_fake_chart_harness(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.env("PATH", prepend_path(&fake_bin));
    cmd.arg("--json")
        .arg("ask")
        .arg("--format")
        .arg("figure")
        .arg("Same thing please");
    let output = cmd.output().expect("run kb ask --format=figure");
    assert!(
        output.status.success(),
        "kb ask --format=figure failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse ask json");
    // `figure` normalizes to `chart` at the artifact layer.
    assert_eq!(envelope["data"]["requested_format"], "chart");
    let artifact_path = kb_root.join(
        envelope["data"]["artifact_path"]
            .as_str()
            .expect("artifact_path"),
    );
    assert!(artifact_path.with_file_name("chart.png").exists());
}

// bn-1m02: `kb ask` renders the answer body to stdout after generation. These
// tests exercise the opt-out paths; rendered-body coverage requires a real LLM
// and is captured by gold_harness.
#[test]
fn ask_json_output_has_no_markdown_rendering() {
    // --json must stay a pure JSON stream — no markdown body leaks to stdout.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json").arg("ask").arg("Does --json stay clean?");
    let output = cmd.output().expect("run kb ask --json");
    assert!(
        output.status.success(),
        "kb ask --json failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // The entire stdout must parse as a single JSON envelope.
    let envelope: Value = serde_json::from_slice(&output.stdout)
        .expect("stdout must be a single JSON envelope with no extra text");
    assert_eq!(envelope["command"], "ask");
    assert!(envelope["data"]["artifact_path"].is_string());
}

#[test]
fn ask_no_render_flag_is_accepted_and_prints_footer() {
    // --no-render must be accepted by clap and must not break the ask flow.
    // Without a real LLM the placeholder branch runs; we verify the artifact
    // path footer is present and no rendered body (which only fires when an
    // LLM actually answered) sneaks in.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("ask").arg("--no-render").arg("anything");
    let output = cmd.output().expect("run kb ask --no-render");
    assert!(
        output.status.success(),
        "kb ask --no-render failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Artifact written:"),
        "expected path footer, got: {stdout}"
    );
}

#[test]
fn ask_piped_non_tty_stdout_contains_footer() {
    // Piped stdout (non-TTY — which is always the case in an assert_cmd
    // child process) must still produce the path footer so scripts that
    // `kb ask "..." | head -N` can locate the artifact.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("ask").arg("piped stdout path");
    let output = cmd.output().expect("run kb ask (piped)");
    assert!(
        output.status.success(),
        "kb ask (piped) failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Artifact written:"),
        "piped ask must print 'Artifact written:' footer, got: {stdout}"
    );
    // No ANSI escape bytes should appear on piped stdout.
    assert!(
        !stdout.contains('\u{1b}'),
        "piped ask must not emit ANSI escapes, got: {stdout:?}"
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
        kb_root.join(".kb/state/indexes/lexical.json").exists(),
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

    let normalized_dir = kb_root.join(".kb/normalized/src-deadbeef");
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
        stdout.contains("resolved_id: .kb/normalized/src-deadbeef/source.md"),
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

    let normalized_dir = kb_root.join(".kb/normalized/src-abcd1234");
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

    let normalized_dir = kb_root.join(".kb/normalized/src-aaaa1111");
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

    let index_path = kb_root.join(".kb/state/indexes/lexical.json");
    assert!(
        index_path.exists(),
        "lexical index should exist after compile"
    );

    let raw = fs::read_to_string(&index_path).expect("read index");
    let json: Value = serde_json::from_str(&raw).expect("parse index");
    let entries = json["entries"].as_array().expect("entries array");
    assert_eq!(entries.len(), 2, "should index 2 pages");
}

// bn-327j: when stdout is piped (not a TTY) the compile output must fall back
// to plain line-by-line text so downstream `grep`/`sed` keep working, even
// though a TTY invocation would render indicatif progress bars. `assert_cmd`
// always captures stdout via a pipe, so the IsTerminal check naturally
// returns false — we assert the stderr is pure ASCII text and contains the
// legacy `compile: N source(s)` banner with no terminal escape sequences.
#[test]
fn compile_piped_stdout_emits_plain_text_without_escape_codes() {
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

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    // indicatif renders bars with ANSI escape sequences (ESC + `[`). Piped
    // stdout must drop into the line-by-line fallback where neither stream
    // ever contains those escapes.
    assert!(
        !stderr.contains('\x1b'),
        "piped stderr must not contain ANSI escape codes; got {stderr:?}"
    );
    assert!(
        !stdout.contains('\x1b'),
        "piped stdout must not contain ANSI escape codes; got {stdout:?}"
    );

    // The final report is printed to stdout in plain text format ("compile:
    // N source(s), K stale"). This also runs as the stderr banner but the
    // stdout copy is what scripts parse.
    assert!(
        stdout.contains("compile:"),
        "expected 'compile:' banner on stdout for plain-text fallback; got stdout={stdout:?}"
    );
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
    // Counting result titles (one per `[score:` occurrence) is more robust
    // than line-counting against per-hit reason lists, whose width varies
    // with the active embedding backend (MiniLM emits an extra `semantic
    // match` reason when scores clear the floor; hash often doesn't).
    let result_count = stdout.matches("[score:").count();
    assert_eq!(
        result_count, 2,
        "with --limit 2 we expect exactly 2 result titles in the output: {stdout}"
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
    // bn-3qsj: search now returns HybridResult, which carries `item_id`
    // (matching the lexical+semantic store) instead of the legacy `id`
    // SearchResult field. `score` is now an f32 RRF score, not a bare
    // lexical count, but the field is still present.
    assert!(
        first.get("item_id").is_some()
            && first.get("title").is_some()
            && first.get("score").is_some(),
        "JSON results should have item_id, title, and score fields: {first}"
    );
    assert!(
        first.get("reasons").is_some(),
        "JSON results should include reasons field: {first}"
    );
    // Hybrid result also exposes per-tier ranks so the user can see
    // which tier matched.
    assert!(
        first.get("lexical_rank").is_some() || first.get("semantic_rank").is_some(),
        "JSON results should expose at least one of lexical_rank/semantic_rank: {first}"
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
fn lint_missing_concepts_flags_term_appearing_in_three_sources() {
    // bn-31lt: the missing-concepts lint walks normalized source bodies and
    // emits a review item for terms mentioned in >= min_sources distinct
    // documents that don't correspond to an existing concept page.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // Seed three normalized source documents that all mention the term
    // "FooBar System". The term appears 5+ times across the corpus and is
    // not a concept — it should produce a concept_candidate review item.
    let normalized_root = kb_root.join(".kb/normalized");
    for (doc_id, body) in [
        (
            "doc-a",
            "# A\n\nThe FooBar System is interesting. FooBar System has many parts.\n",
        ),
        (
            "doc-b",
            "# B\n\nFooBar System unique aspects. The FooBar System model.\n",
        ),
        (
            "doc-c",
            "# C\n\nAnother look at FooBar System for comparison.\n",
        ),
    ] {
        let dir = normalized_root.join(doc_id);
        fs::create_dir_all(&dir).expect("create normalized doc dir");
        fs::write(
            dir.join("metadata.json"),
            format!("{{\"source_revision_id\":\"rev-{doc_id}\"}}"),
        )
        .expect("write metadata");
        fs::write(dir.join("source.md"), body).expect("write source body");
    }

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json")
        .arg("lint")
        .arg("--check")
        .arg("missing-concepts");
    let output = cmd.output().expect("run kb lint");

    // Missing-concepts findings are warnings, so exit code should be 0 in
    // default (non-strict) mode.
    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse lint json");
    assert_eq!(envelope["command"], "lint");
    let payload = &envelope["data"];
    let checks = payload["checks"].as_array().expect("checks array");
    assert_eq!(checks.len(), 1);
    let check = &checks[0];
    assert_eq!(check["check"], "missing-concepts");
    let issues = check["issues"].as_array().expect("issues array");
    assert!(
        issues.iter().any(|i| {
            i["kind"] == "concept_candidate"
                && i["severity"] == "warning"
                && i["target"].as_str().is_some_and(|t| t.contains("FooBar System"))
        }),
        "expected concept_candidate warning for 'FooBar System', got: {issues:?}"
    );

    // Running the lint must also persist a review item with the
    // concept_candidate kind so users can `kb review approve` it later.
    let review_path = kb_root
        .join("reviews")
        .join("concept_candidates")
        .join("lint:concept-candidate:foobar-system.json");
    assert!(
        review_path.is_file(),
        "expected review item file at {}",
        review_path.display()
    );
    let raw = fs::read_to_string(&review_path).expect("read review item");
    let item: Value = serde_json::from_str(&raw).expect("parse review item json");
    assert_eq!(item["kind"], "concept_candidate");
    assert_eq!(item["status"], "pending");
    assert!(
        item["comment"]
            .as_str()
            .is_some_and(|c| c.contains("FooBar System")),
        "review item comment must name the term: {raw}"
    );
    let deps = item["metadata"]["dependencies"].as_array().expect("deps");
    assert_eq!(
        deps.len(),
        3,
        "expected three source-document ids in deps: {deps:?}"
    );
}

#[test]
fn lint_contradictions_emits_review_item_when_llm_detects_conflict() {
    // bn-3axp: cross-source contradictions are LLM-powered. The pass is
    // gated behind `--check contradictions` (never part of the default
    // sweep) and relies on the configured runner to decide whether the
    // concept's cited quotes disagree.
    //
    // Harness-fake strategy: install a fake `opencode` binary on PATH that
    // prints the exact JSON body the adapter expects to see from the real
    // runner (a `DetectContradictionsResponse` with `contradiction: true`).
    // The kb binary calls it, parses the JSON, and must persist a
    // `reviews/contradictions/*.json` review item.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // Seed two normalized source documents — the heading-anchor on each
    // quote is how the lint attributes a quote to its source.
    let normalized_root = kb_root.join(".kb/normalized");
    for (doc_id, heading) in [("src-alpha", "claim-alpha"), ("src-beta", "claim-beta")] {
        let dir = normalized_root.join(doc_id);
        fs::create_dir_all(&dir).expect("create normalized doc dir");
        let metadata = serde_json::json!({
            "source_revision_id": format!("rev-{doc_id}"),
            "heading_ids": [heading],
        });
        fs::write(dir.join("metadata.json"), metadata.to_string())
            .expect("write metadata");
        fs::write(
            dir.join("source.md"),
            format!("# {doc_id}\n\n## {heading}\n\nsome body\n"),
        )
        .expect("write body");
    }

    // Seed a concept page that cites one quote from each source.
    let concepts = kb_root.join("wiki").join("concepts");
    fs::create_dir_all(&concepts).expect("create concepts dir");
    let concept_body = r"---
id: concept:widget
name: Widget
source_document_ids:
- src-alpha
- src-beta
sources:
- heading_anchor: claim-alpha
  quote: Widgets are always blue.
- heading_anchor: claim-beta
  quote: Widgets are always red.
---

# Widget
";
    fs::write(concepts.join("widget.md"), concept_body).expect("write concept page");

    // Install a fake `opencode` that prints the contradiction JSON on
    // stdout. Any arguments are ignored. The adapter parses the stdout
    // directly (no JSON envelope — opencode output is the raw response).
    let bin_dir = kb_root.join("fake-bin");
    fs::create_dir_all(&bin_dir).expect("create fake bin dir");
    let fake_script = r#"#!/bin/sh
# Ignore every argument; print the exact JSON the adapter expects. The
# `run` subcommand's output is returned verbatim by the opencode adapter.
cat <<'JSON'
{"contradiction": true, "explanation": "One quote says widgets are blue, the other says they are red — these cannot both be true.", "conflicting_quotes": [0, 1]}
JSON
"#;
    write_executable(bin_dir.join("opencode").as_path(), fake_script);

    let mut cmd = kb_cmd(&kb_root);
    cmd.env("PATH", prepend_path(&bin_dir));
    cmd.arg("--json")
        .arg("lint")
        .arg("--check")
        .arg("contradictions");
    let output = cmd.output().expect("run kb lint --check contradictions");

    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse lint json");
    assert_eq!(envelope["command"], "lint");
    let checks = envelope["data"]["checks"].as_array().expect("checks array");
    assert_eq!(checks.len(), 1);
    assert_eq!(checks[0]["check"], "contradictions");
    let issues = checks[0]["issues"].as_array().expect("issues array");
    assert_eq!(issues.len(), 1, "expected exactly one contradiction issue: {issues:?}");
    assert_eq!(issues[0]["kind"], "contradiction");
    assert_eq!(issues[0]["severity"], "warning");
    assert_eq!(issues[0]["target"], "concept:widget");

    // The lint must also have persisted a ReviewKind::Contradiction item
    // under reviews/contradictions/ so `kb review` can surface it.
    let review_path = kb_root
        .join("reviews")
        .join("contradictions")
        .join("lint:contradiction:widget.json");
    assert!(
        review_path.is_file(),
        "expected review item at {}",
        review_path.display()
    );
    let raw = fs::read_to_string(&review_path).expect("read review item");
    let item: Value = serde_json::from_str(&raw).expect("parse review item json");
    assert_eq!(item["kind"], "contradiction");
    assert_eq!(item["status"], "pending");
    assert_eq!(item["target_entity_id"], "concept:widget");
    let comment = item["comment"].as_str().expect("comment string");
    assert!(
        comment.contains("Widget") && comment.contains("blue"),
        "review comment should reference the concept + LLM explanation: {comment}"
    );
    // Citations should list each conflicting quote with its source label.
    let citations = item["citations"].as_array().expect("citations array");
    assert_eq!(citations.len(), 2, "expected two citations: {citations:?}");
    assert!(
        citations.iter().any(|c| c.as_str().is_some_and(|s| s.contains("src-alpha"))),
        "missing src-alpha citation: {citations:?}"
    );
    assert!(
        citations.iter().any(|c| c.as_str().is_some_and(|s| s.contains("src-beta"))),
        "missing src-beta citation: {citations:?}"
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn lint_impute_queues_imputed_fix_review_items_from_thin_concept_body() {
    // bn-xt4o: `kb lint --impute` walks concept pages with thin bodies,
    // calls a web-search-capable LLM, and queues the draft as a
    // `ReviewKind::ImputedFix` review item. No direct edits — the user
    // approves via `kb review approve lint:imputed-fix:<id>`.
    //
    // Harness-fake strategy mirrors the contradictions test: install a
    // fake opencode on PATH that prints the exact JSON an
    // `ImputeGapResponse` serializes to. The adapter parses the stdout
    // and the impute pass wraps it into a review item.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // Seed a concept page with a body short enough to trip the thin-body
    // detector (the default threshold is 12 words; "thin." has one).
    let concepts = kb_root.join("wiki").join("concepts");
    fs::create_dir_all(&concepts).expect("create concepts dir");
    let concept_body = "---\nid: concept:widget\nname: Widget\n---\n\n# Widget\n\nthin.\n";
    fs::write(concepts.join("widget.md"), concept_body).expect("write concept");

    // Fake opencode prints the JSON the impute adapter expects.
    let bin_dir = kb_root.join("fake-bin");
    fs::create_dir_all(&bin_dir).expect("create fake bin dir");
    let fake_script = r#"#!/bin/sh
cat <<'JSON'
{"definition": "Widget is a general-purpose example concept used in kb integration tests to exercise the impute pass.", "sources": [{"url": "https://example.com/widget", "title": "Widget overview", "note": "Canonical test reference."}], "confidence": "medium", "rationale": "Picked the general test variant because the local KB body is thin."}
JSON
"#;
    write_executable(bin_dir.join("opencode").as_path(), fake_script);
    // `just check`-style `claude` stub needed in case the adapter path
    // ever falls back to it (it won't in this config, but kb-cli doctors
    // expect it on PATH when resolving runners).
    write_executable(
        bin_dir.join("claude").as_path(),
        "#!/bin/sh\nprintf '{\"result\":\"OK\"}'",
    );

    let mut cmd = kb_cmd(&kb_root);
    cmd.env("PATH", prepend_path(&bin_dir));
    cmd.arg("--json").arg("lint").arg("--impute");
    let output = cmd.output().expect("run kb lint --impute");

    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout: {} stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse lint json");
    assert_eq!(envelope["command"], "lint");
    let checks = envelope["data"]["checks"].as_array().expect("checks array");
    let impute_check = checks
        .iter()
        .find(|c| c["check"] == "impute")
        .expect("expected an 'impute' check entry");
    let issues = impute_check["issues"].as_array().expect("issues array");
    assert!(
        !issues.is_empty(),
        "impute pass should have emitted at least one issue: {issues:?}"
    );

    // The impute pass must have persisted an imputed-fix review item and
    // its payload sidecar so `kb review approve` can find it.
    let review_path = kb_root
        .join("reviews")
        .join("imputed_fixes")
        .join("lint:imputed-fix:thin_concept_body:widget.json");
    assert!(
        review_path.is_file(),
        "expected review item at {}",
        review_path.display()
    );
    let raw = fs::read_to_string(&review_path).expect("read review item");
    let item: Value = serde_json::from_str(&raw).expect("parse review item json");
    assert_eq!(item["kind"], "imputed_fix");
    assert_eq!(item["status"], "pending");
    assert_eq!(item["target_entity_id"], "concept:widget");
    let citations = item["citations"].as_array().expect("citations array");
    assert!(
        citations
            .iter()
            .any(|c| c.as_str().is_some_and(|s| s.contains("https://example.com/widget"))),
        "expected web-source citation: {citations:?}"
    );

    let sidecar_path = kb_root
        .join("reviews")
        .join("imputed_fixes")
        .join("lint:imputed-fix:thin_concept_body:widget.payload.json");
    assert!(
        sidecar_path.is_file(),
        "expected payload sidecar at {}",
        sidecar_path.display()
    );
    let payload_raw = fs::read_to_string(&sidecar_path).expect("read payload sidecar");
    let payload: Value = serde_json::from_str(&payload_raw).expect("parse payload sidecar");
    assert_eq!(payload["gap_kind"], "thin_concept_body");
    assert_eq!(payload["concept_id"], "concept:widget");
    assert!(
        payload["definition"]
            .as_str()
            .is_some_and(|s| s.contains("Widget is a")),
        "definition should start with concept name: {payload:?}"
    );
    assert!(
        payload["sources"]
            .as_array()
            .is_some_and(|arr| !arr.is_empty()),
        "payload must carry at least one web source"
    );

    // Approve should rewrite the body while preserving the frontmatter.
    let mut approve_cmd = kb_cmd(&kb_root);
    approve_cmd.env("PATH", prepend_path(&bin_dir));
    approve_cmd
        .arg("--json")
        .arg("review")
        .arg("approve")
        .arg("lint:imputed-fix:thin_concept_body:widget");
    let approve_output = approve_cmd
        .output()
        .expect("run kb review approve lint:imputed-fix:...");
    assert_eq!(
        approve_output.status.code(),
        Some(0),
        "approve stdout: {} stderr: {}",
        String::from_utf8_lossy(&approve_output.stdout),
        String::from_utf8_lossy(&approve_output.stderr)
    );

    let written = fs::read_to_string(concepts.join("widget.md")).expect("read concept post-approve");
    assert!(
        written.contains("id: concept:widget"),
        "frontmatter id preserved: {written}"
    );
    assert!(
        written.contains("Widget is a general-purpose example concept"),
        "body replaced with imputed definition: {written}"
    );
    assert!(
        written.contains("## Sources (imputed)"),
        "sources section rendered: {written}"
    );
    assert!(
        written.contains("https://example.com/widget"),
        "cited URL rendered in sources section: {written}"
    );
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
    insta::assert_snapshot!("doctor_envelope_keys", sorted_object_keys(&envelope));
    let payload = &envelope["data"];
    insta::assert_snapshot!("doctor_data_keys", sorted_object_keys(payload));
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
        kb_root.join(".kb/state/jobs/interrupted-test.json"),
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
    insta::assert_snapshot!("ask_dry_run_data_keys", sorted_object_keys(plan));

    let outputs_dir = kb_root.join("outputs/questions");
    if outputs_dir.exists() {
        let entries: Vec<_> = fs::read_dir(&outputs_dir)
            .expect("read outputs/questions")
            .filter_map(Result::ok)
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with("q-")
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
fn status_json_exposes_chief_freshness_contract_fields() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    let source = kb_root.join("chief-contract.md");
    fs::write(&source, "# Chief Contract\n\nProvider fields.\n").expect("write source");

    let mut ingest_cmd = kb_cmd(&kb_root);
    ingest_cmd.arg("ingest").arg(&source);
    let ingest_output = ingest_cmd.output().expect("run kb ingest");
    assert!(
        ingest_output.status.success(),
        "ingest failed: {}",
        String::from_utf8_lossy(&ingest_output.stderr)
    );

    let mut compile_cmd = kb_cmd(&kb_root);
    compile_cmd.arg("compile");
    let compile_output = compile_cmd.output().expect("run kb compile");
    assert!(
        compile_output.status.success(),
        "compile failed: {}",
        String::from_utf8_lossy(&compile_output.stderr)
    );

    let mut status_cmd = kb_cmd(&kb_root);
    status_cmd.arg("--json").arg("status");
    let output = status_cmd.output().expect("run kb status");
    assert!(output.status.success());
    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse status json");
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "status");
    let data = &envelope["data"];
    assert_eq!(data["total_sources"], data["normalized_source_count"]);
    assert_eq!(data["stale_sources"], data["stale_count"]);
    assert!(
        data["wiki_page_count"].as_u64().expect("wiki_page_count") >= data["wiki_pages"]
            .as_u64()
            .expect("wiki_pages")
    );
    assert!(
        data["last_compile_at_millis"].is_number(),
        "last compile timestamp should be present after compile: {data}"
    );
}

#[test]
fn resolve_json_resolves_kb_uri_with_metadata() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    write_concept_page(&kb_root, "borrow-checker", "Borrow checker", &["borrowck"]);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json")
        .arg("resolve")
        .arg("kb://wiki/concepts/borrow-checker.md");
    let output = cmd.output().expect("run kb resolve");
    assert!(
        output.status.success(),
        "resolve failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse resolve json");
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "resolve");
    insta::assert_snapshot!("resolve_envelope_keys", sorted_object_keys(&envelope));
    insta::assert_snapshot!("resolve_data_keys", sorted_object_keys(&envelope["data"]));

    let data = &envelope["data"];
    assert_eq!(data["target"], "wiki/concepts/borrow-checker.md");
    assert_eq!(data["stable_id"], "concept:borrow-checker");
    assert_eq!(data["current_path"], "wiki/concepts/borrow-checker.md");
    assert_eq!(data["title"], "Borrow checker");
    assert!(data["content_hash"].is_string());
    assert_eq!(data["broken"], false);
    assert_eq!(data["kind"], "wiki_page");
}

#[test]
fn resolve_json_reports_broken_kb_uri_without_failing() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json")
        .arg("resolve")
        .arg("kb://wiki/concepts/missing.md");
    let output = cmd.output().expect("run kb resolve missing");
    assert!(
        output.status.success(),
        "broken references should be represented in JSON rather than command failure: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse resolve json");
    assert_eq!(envelope["command"], "resolve");
    let data = &envelope["data"];
    assert_eq!(data["target"], "wiki/concepts/missing.md");
    assert_eq!(data["broken"], true);
    assert_eq!(data["freshness"], "missing");
    assert!(data["broken_reason"].is_string());
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
            dependencies: vec!["art-q1".to_string()],
            output_paths: vec![PathBuf::from("outputs/questions/q-q1/answer.md")],
            status: Status::NeedsReview,
        },
        kind: ReviewKind::Promotion,
        target_entity_id: "art-q1".to_string(),
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
    let normalized_dir = kb_root.join(".kb/normalized").join(doc_id);
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
    let question_id = "q-promote-orphan";
    let artifact_id = "art-promote-orphan";
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
    assert!(!source_strs.iter().any(|s| s.starts_with("q-")));
    assert!(!source_strs.iter().any(|s| s.starts_with("art-")));
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
    let normalized_dir = kb_root.join(".kb/normalized").join(src_id);
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
    let state_dir = kb_root.join(".kb/state");
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
    let jobs_dir = root.join(".kb/state/jobs");
    fs::create_dir_all(&jobs_dir).expect("create .kb/state/jobs");
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
    let jobs_dir = root.join(".kb/state/jobs");
    fs::create_dir_all(&jobs_dir).expect("create .kb/state/jobs");
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
        kb_root.join(".kb/state/jobs/intr-00.log"),
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
    let trash_root = kb_root.join(".kb/trash");
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
    let remaining: Vec<String> = fs::read_dir(kb_root.join(".kb/state/jobs"))
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
        kb_root.join(".kb/state/jobs/fresh-intr.json").exists(),
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
    fs::write(kb_root.join(".kb/state/jobs/intr-1.log"), "trace\n")
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
    let locks_dir = kb_root.join(".kb/state").join("locks");
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

    let locks_dir = kb_root.join(".kb/state").join("locks");
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

    let normalized_dir = kb_root.join(".kb/normalized").join(&src_id);
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
    let trash_root = kb_root.join(".kb/trash");
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
    assert!(bundle.join(".kb/normalized").join(&src_id).exists());
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

    assert!(!kb_root.join(".kb/normalized").join(&src_id).exists());
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

    let normalized_dir = kb_root.join(".kb/normalized").join(&src_id);
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
    let trash_dir = kb_root.join(".kb/trash");
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

    let normalized_dir = kb_root.join(".kb/normalized").join(&src_id);
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

    let normalized_dir = kb_root.join(".kb/normalized").join(&src_id);

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
        .push(PathBuf::from(format!(".kb/normalized/{src_id}/summary.md")));
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
        .join(".kb/state/build_records")
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
    let trash_root = kb_root.join(".kb/trash");
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
    let bundle = fs::read_dir(kb_root.join(".kb/trash"))
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
        .join(".kb/state/build_records")
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
            ".kb/state/concept_candidates/{src_id}.json"
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
        .join(".kb/state/build_records")
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
    for entry in walkdir_files(&kb_root.join(".kb/state")) {
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
    assert!(!kb_root.join(".kb/normalized").join(&src_id).exists());
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
    let pre_lex = fs::read_to_string(kb_root.join(".kb/state/indexes/lexical.json"))
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
    let jobs_dir = root.join(".kb/state/jobs");
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
        fs::read_dir(kb_root.join(".kb/state/jobs"))
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

/// bn-2zy acceptance (Part A + Part B):
///
/// Approving a promotion must refresh `wiki/index.md` and
/// `wiki/questions/index.md` inline — no follow-up `kb compile` required —
/// and both must render the properly-cased raw question (from the promoted
/// page's `question:` frontmatter), not the URL-safe slug.
#[test]
fn approve_refreshes_indexes_and_renders_question_title() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // Seed the answer artifact that `execute_promotion` reads.
    let question_id = "q-arc-vs-lru";
    let artifact_id = "art-arc-vs-lru";
    let q_dir = kb_root.join("outputs/questions").join(question_id);
    fs::create_dir_all(&q_dir).expect("create question dir");
    fs::write(
        q_dir.join("answer.md"),
        format!(
            "---\nid: {artifact_id}\ntype: question_answer\nquestion_id: {question_id}\n---\n\nARC keeps a ghost list of evicted entries; LRU does not.\n"
        ),
    )
    .expect("write answer.md");

    // Seed the pending ReviewItem. The `comment` prefix is the contract the
    // writer uses to recover the raw question for the `question:` frontmatter.
    let raw_question = "How does ARC cache compare to LRU?";
    let destination = PathBuf::from("wiki/questions/how-does-arc-cache-compare-to-lru.md");
    let review_id = "review-arc-vs-lru";
    let review_item = ReviewItem {
        metadata: EntityMetadata {
            id: review_id.to_string(),
            created_at_millis: 1_000,
            updated_at_millis: 1_000,
            source_hashes: vec!["hash-arc".to_string()],
            model_version: None,
            tool_version: Some("kb-test".to_string()),
            prompt_template_hash: None,
            dependencies: vec![question_id.to_string(), artifact_id.to_string()],
            output_paths: vec![
                PathBuf::from(format!("outputs/questions/{question_id}/answer.md")),
                destination.clone(),
            ],
            status: Status::NeedsReview,
        },
        kind: ReviewKind::Promotion,
        target_entity_id: artifact_id.to_string(),
        proposed_destination: Some(destination.clone()),
        citations: vec![],
        affected_pages: vec![destination.clone()],
        created_at_millis: 1_000,
        status: ReviewStatus::Pending,
        comment: format!("Promote answer for: {raw_question}"),
    };
    save_review_item(&kb_root, &review_item).expect("save review item");

    // Approve.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("review").arg("approve").arg(review_id);
    let output = cmd.output().expect("run kb review approve");
    assert!(
        output.status.success(),
        "kb review approve failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // The promoted page must carry the raw question in frontmatter.
    let promoted = fs::read_to_string(kb_root.join(&destination)).expect("read promoted page");
    assert!(
        promoted.contains(&format!("question: {raw_question}")),
        "promoted page missing raw-question frontmatter: {promoted}"
    );

    // Part A: indexes must be refreshed inline.
    let global_index = fs::read_to_string(kb_root.join("wiki/index.md"))
        .expect("wiki/index.md should exist after approve");
    assert!(
        global_index.contains("## Questions"),
        "global index missing Questions section after approve:\n{global_index}"
    );
    let questions_index = fs::read_to_string(kb_root.join("wiki/questions/index.md"))
        .expect("wiki/questions/index.md should exist after approve");

    // Part B: both indexes must render the properly-cased question title,
    // not the slug.
    assert!(
        global_index.contains(&format!("[{raw_question}]")),
        "global index should show proper-cased title [{raw_question}]; got:\n{global_index}"
    );
    assert!(
        questions_index.contains(&format!("[{raw_question}]")),
        "questions index should show proper-cased title [{raw_question}]; got:\n{questions_index}"
    );
    // Slug-as-title should NOT appear as link text in either index.
    assert!(
        !global_index.contains("[how-does-arc-cache-compare-to-lru]")
            && !global_index.contains("[how does arc cache compare to lru]"),
        "global index leaked slug as title:\n{global_index}"
    );
    assert!(
        !questions_index.contains("[how-does-arc-cache-compare-to-lru]")
            && !questions_index.contains("[how does arc cache compare to lru]"),
        "questions index leaked slug as title:\n{questions_index}"
    );
}

// -- bn-1525: short-prefix id resolution (terseid::IdResolver) ---------------

/// Helper: lay down a minimal normalized+wiki pair for a src id so the inspect
/// file-report code has something to read. Kept local to the bn-1525 tests to
/// avoid cross-test coupling.
fn seed_src(kb_root: &Path, src_id: &str) {
    let normalized = kb_root.join(".kb/normalized").join(src_id);
    fs::create_dir_all(&normalized).expect("create normalized dir");
    fs::write(normalized.join("source.md"), "# seed\n").expect("write source.md");
    fs::write(
        normalized.join("metadata.json"),
        format!(
            r#"{{"metadata":{{"id":"{src_id}","created_at_millis":0,"updated_at_millis":0,"source_hashes":[],"dependencies":[],"output_paths":[],"status":"fresh"}},"source_revision_id":"rev-1","normalized_assets":[],"heading_ids":[]}}"#
        ),
    )
    .expect("write metadata.json");
}

#[test]
fn inspect_resolves_unique_src_prefix() {
    // `kb inspect src-a7` should route to the unique `src-a7x3q9` when that's
    // the only src whose hash starts with `a7`.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    seed_src(&kb_root, "src-a7x3q9");
    seed_src(&kb_root, "src-b2k1m5");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("inspect").arg("src-a7");
    let output = cmd.output().expect("run kb inspect src-a7");
    assert!(
        output.status.success(),
        "kb inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("src-a7x3q9"),
        "expected prefix to resolve to src-a7x3q9; got:\n{stdout}"
    );
}

#[test]
fn inspect_reports_ambiguous_src_prefix_with_candidates() {
    // Two srcs share the `a7` prefix → ambiguity error lists both and hints
    // at typing a longer prefix.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    seed_src(&kb_root, "src-a7x3q9");
    seed_src(&kb_root, "src-a7b2k1");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("inspect").arg("src-a7");
    let output = cmd.output().expect("run kb inspect src-a7");
    assert!(
        !output.status.success(),
        "ambiguous prefix must fail; stdout=\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("ambiguous"), "expected ambiguity: {stderr}");
    assert!(stderr.contains("src-a7x3q9"), "candidates: {stderr}");
    assert!(stderr.contains("src-a7b2k1"), "candidates: {stderr}");
    assert!(stderr.contains("longer prefix"), "hint: {stderr}");
}

#[test]
fn inspect_no_match_reports_clearly() {
    // Nothing on disk matches — the error should point at the actual problem,
    // not silently succeed.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    seed_src(&kb_root, "src-a7x3q9");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("inspect").arg("src-zzzz");
    let output = cmd.output().expect("run kb inspect src-zzzz");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("was not found"),
        "expected not-found error: {stderr}"
    );
}

#[test]
fn inspect_exact_full_src_id_still_matches_after_bn_1525() {
    // Regression guard: the existing exact-id path must still resolve to the
    // wiki/sources page without invoking the new prefix resolver.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let sources_dir = kb_root.join("wiki/sources");
    fs::create_dir_all(&sources_dir).expect("create wiki/sources");
    fs::write(
        sources_dir.join("src-bb550011.md"),
        "---\nid: wiki-source-src-bb550011\nsource_document_id: src-bb550011\nsource_revision_id: rev-1\n---\n\n# Src\n",
    )
    .expect("write source page");
    seed_src(&kb_root, "src-bb550011");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("inspect").arg("src-bb550011");
    let output = cmd.output().expect("run kb inspect");
    assert!(
        output.status.success(),
        "kb inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("resolved_id: wiki/sources/src-bb550011.md"),
        "full id should hit wiki page first; got:\n{stdout}"
    );
}

#[test]
fn forget_resolves_unique_src_prefix_and_confirms_src_id() {
    // `kb forget src-a7 --dry-run` should route to the full id in its plan
    // even though the user typed a prefix.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    seed_src(&kb_root, "src-a7x3q9");
    seed_src(&kb_root, "src-b2k1m5");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--dry-run").arg("forget").arg("src-a7");
    let output = cmd.output().expect("run kb forget --dry-run");
    assert!(
        output.status.success(),
        "kb forget dry-run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("forget src-a7x3q9"),
        "plan must mention full id; got:\n{stdout}"
    );
    // The plan should show the normalized dir about to move.
    assert!(
        stdout.contains("normalized/src-a7x3q9"),
        "plan should list normalized dir; got:\n{stdout}"
    );
}

#[test]
fn forget_reports_ambiguous_src_prefix_with_candidates() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    seed_src(&kb_root, "src-a7x3q9");
    seed_src(&kb_root, "src-a7b2k1");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--dry-run").arg("forget").arg("src-a7");
    let output = cmd.output().expect("run kb forget --dry-run");
    assert!(
        !output.status.success(),
        "ambiguous prefix must fail; stdout=\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("ambiguous"), "expected ambiguity: {stderr}");
    assert!(stderr.contains("src-a7x3q9"), "candidates: {stderr}");
    assert!(stderr.contains("src-a7b2k1"), "candidates: {stderr}");
}

#[test]
fn forget_exact_full_src_id_still_works_after_bn_1525() {
    // Regression guard: `kb forget src-<full>` must continue to work without
    // touching the prefix resolver.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    seed_src(&kb_root, "src-cc770033");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--dry-run").arg("forget").arg("src-cc770033");
    let output = cmd.output().expect("run kb forget --dry-run");
    assert!(
        output.status.success(),
        "kb forget dry-run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("forget src-cc770033"),
        "plan must mention full id; got:\n{stdout}"
    );
}

// ---------------------------------------------------------------------------
// bn-1dar: `kb ask --editor` / `-e`
//
// The flag opens $VISUAL/$EDITOR (falling back to `vi`) on a tempfile and
// uses the resulting content as the question. Lines starting with `#` are
// stripped like `git commit`. Empty content after stripping is a
// ValidationError — it must NOT leave a failed-job manifest behind.
// ---------------------------------------------------------------------------

/// Write a fake editor script that overwrites the tempfile passed as its
/// first argument with `body`. Returns the script path.
fn write_fake_editor(dir: &Path, name: &str, body: &str) -> PathBuf {
    let script = dir.join(name);
    // Quote `body` using a shell-friendly heredoc so newlines round-trip.
    let contents = format!(
        "#!/bin/sh\ncat > \"$1\" <<'FAKE_EDITOR_EOF'\n{body}\nFAKE_EDITOR_EOF\n"
    );
    write_executable(&script, &contents);
    script
}

#[test]
fn ask_editor_flag_reads_question_from_editor() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let script = write_fake_editor(&kb_root, "fake-editor.sh", "my test question");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json")
        .arg("ask")
        .arg("--editor")
        .env("EDITOR", &script)
        .env_remove("VISUAL");
    let output = cmd.output().expect("run kb ask --editor");
    assert!(
        output.status.success(),
        "kb ask --editor failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse ask json");
    assert_eq!(envelope["command"], "ask");
    let question_path = kb_root.join(
        envelope["data"]["question_path"]
            .as_str()
            .expect("question_path string"),
    );
    let record: Value = serde_json::from_str(
        &fs::read_to_string(&question_path).expect("read question record"),
    )
    .expect("parse question record");
    assert_eq!(
        record["raw_query"], "my test question",
        "editor content must become the raw_query; record: {record}"
    );
}

#[test]
fn ask_editor_short_flag_equivalent() {
    // `-e` is the short form of `--editor` and behaves identically.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let script = write_fake_editor(&kb_root, "fake-editor.sh", "short-flag question");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("ask")
        .arg("-e")
        .env("EDITOR", &script)
        .env_remove("VISUAL");
    let output = cmd.output().expect("run kb ask -e");
    assert!(
        output.status.success(),
        "kb ask -e failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn ask_editor_visual_takes_precedence_over_editor() {
    // $VISUAL wins when both are set — standard Unix convention.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let visual_script =
        write_fake_editor(&kb_root, "visual-editor.sh", "from VISUAL wins");
    // The EDITOR script writes a DIFFERENT body AND would fail the test
    // (nonzero exit). If $VISUAL isn't preferred, we'd run this one and
    // either get the wrong body or a launch error.
    let editor_script = kb_root.join("editor-fail.sh");
    write_executable(
        &editor_script,
        "#!/bin/sh\necho 'EDITOR was used, but VISUAL should have won' >&2\nexit 17\n",
    );

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json")
        .arg("ask")
        .arg("--editor")
        .env("VISUAL", &visual_script)
        .env("EDITOR", &editor_script);
    let output = cmd.output().expect("run kb ask --editor with VISUAL set");
    assert!(
        output.status.success(),
        "kb ask --editor with $VISUAL set failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse ask json");
    let question_path = kb_root.join(
        envelope["data"]["question_path"]
            .as_str()
            .expect("question_path string"),
    );
    let record: Value = serde_json::from_str(
        &fs::read_to_string(&question_path).expect("read question record"),
    )
    .expect("parse question record");
    assert_eq!(
        record["raw_query"], "from VISUAL wins",
        "VISUAL content must win; record: {record}"
    );
}

#[test]
fn ask_editor_strips_comment_lines() {
    // git-style: lines starting with `#` are dropped before validation.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let script = write_fake_editor(
        &kb_root,
        "fake-editor.sh",
        "# This is a comment\n   # indented comment\nreal question line\n# trailing comment",
    );

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json")
        .arg("ask")
        .arg("--editor")
        .env("EDITOR", &script)
        .env_remove("VISUAL");
    let output = cmd.output().expect("run kb ask --editor with comments");
    assert!(
        output.status.success(),
        "kb ask --editor with comments failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse ask json");
    let question_path = kb_root.join(
        envelope["data"]["question_path"]
            .as_str()
            .expect("question_path string"),
    );
    let record: Value = serde_json::from_str(
        &fs::read_to_string(&question_path).expect("read question record"),
    )
    .expect("parse question record");
    let raw_query = record["raw_query"]
        .as_str()
        .expect("raw_query should be string");
    assert_eq!(
        raw_query, "real question line",
        "comment lines must be stripped; got: {raw_query}"
    );
}

#[test]
fn ask_editor_empty_content_is_validation_error_no_failed_job() {
    // bn-1jx acceptance: empty editor content is a pure validation
    // rejection — exit 1, but NO failed-job manifest.
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // Editor writes only comments → after stripping, body is empty.
    let script = write_fake_editor(
        &kb_root,
        "empty-editor.sh",
        "# only comments here\n# nothing of substance",
    );

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("ask")
        .arg("--editor")
        .env("EDITOR", &script)
        .env_remove("VISUAL");
    let output = cmd.output().expect("run kb ask --editor with only-comments");
    assert!(
        !output.status.success(),
        "empty editor content must cause kb ask to fail"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("empty"),
        "expected stderr to mention 'empty', got: {stderr}"
    );

    assert_eq!(
        count_job_manifests(&kb_root),
        0,
        "validation rejection must not write a job manifest"
    );

    let mut status_cmd = kb_cmd(&kb_root);
    status_cmd.arg("--json").arg("status");
    let status_output = status_cmd.output().expect("run kb --json status");
    assert!(status_output.status.success(), "kb status failed");
    let envelope: Value =
        serde_json::from_slice(&status_output.stdout).expect("parse status json");
    assert_eq!(
        envelope["data"]["failed_jobs_total"], 0,
        "empty --editor content must not count as a failed job"
    );
    assert!(
        envelope["data"]["failed_jobs"]
            .as_array()
            .expect("failed_jobs array")
            .is_empty(),
        "failed_jobs list must be empty after validation rejection"
    );
}

// ── concept_candidate auto-apply tests (bn-lw06) ─────────────────────────────

/// Install a fake `opencode` script that returns a fixed
/// `GenerateConceptFromCandidateResponse` JSON payload. Matches the
/// bundled `generate_concept_from_candidate.md` prompt's output schema so
/// the CLI's parser accepts it. We also stub `generate_concept_body` by
/// returning a plain sentence whenever the prompt mentions "general
/// definition" — the primary draft already looks healthy, so the two-step
/// refinement should not trigger; but being ready means we don't fail
/// spuriously if the heuristic flips.
fn install_fake_concept_candidate_harness(root: &Path, canonical: &str, category: &str) -> PathBuf {
    let bin_dir = root.join("fake-concept-candidate-bin");
    fs::create_dir_all(&bin_dir).expect("create fake bin dir");
    // bn-3049: kb-cli now feeds the prompt via stdin (not argv). If it's
    // the generate_concept_from_candidate prompt (which names the candidate
    // term), return the canonical JSON. Otherwise fall through to a
    // plain-text reply used by the concept_body two-step prompt.
    let script = format!(
        r#"#!/bin/sh
set -e
prompt="$(cat)"
case "$prompt" in
  *"drafting a canonical concept entry"*)
    printf '{{"canonical_name":"{canonical}","aliases":["FooBar"],"category":"{category}","definition":"{canonical} is a coordinated system described across multiple sources that covers its design, failure modes, and operational characteristics."}}'
    ;;
  *"general definition for a knowledge base concept"*)
    printf '{canonical} is a coordinated system described across multiple sources that covers its design, failure modes, and operational characteristics.'
    ;;
  *)
    printf '{{"canonical_name":"{canonical}","aliases":[],"category":null,"definition":"{canonical} is a coordinated system."}}'
    ;;
esac
"#
    );
    write_executable(bin_dir.join("opencode").as_path(), &script);
    write_executable(
        bin_dir.join("claude").as_path(),
        "#!/bin/sh\nprintf '{\"result\":\"OK\"}'",
    );
    bin_dir
}

/// bn-lw06: `kb review approve concept-candidate:<slug>` drafts a concept
/// page via the LLM, writes `wiki/concepts/<slug>.md` with proper
/// frontmatter, refreshes backlinks + indexes inline, and flips the review
/// item to approved.
#[test]
#[allow(clippy::too_many_lines)]
fn review_approve_applies_concept_candidate_with_stub_llm() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // Seed three normalized source bodies that all mention "FooBar System"
    // (matches the existing lint test pattern so the snippet extractor has
    // real prose to work with, not just the candidate term in isolation).
    let normalized_root = kb_root.join(".kb/normalized");
    for (doc_id, body) in [
        (
            "doc-a",
            "# A\n\nThe FooBar System is a distributed store. FooBar System partitions data.\n",
        ),
        (
            "doc-b",
            "# B\n\nFooBar System has unique consistency semantics. The FooBar System model.\n",
        ),
        (
            "doc-c",
            "# C\n\nAnother look at FooBar System for comparison purposes.\n",
        ),
    ] {
        let dir = normalized_root.join(doc_id);
        fs::create_dir_all(&dir).expect("create normalized doc dir");
        fs::write(
            dir.join("metadata.json"),
            format!("{{\"source_revision_id\":\"rev-{doc_id}\"}}"),
        )
        .expect("write metadata");
        fs::write(dir.join("source.md"), body).expect("write source body");
    }

    // Seed the concept_candidate review item directly (bypass the lint; the
    // review-item format is stable contract from bn-31lt).
    let slug = "foobar-system";
    let id = format!("lint:concept-candidate:{slug}");
    let dest = PathBuf::from(format!("wiki/concepts/{slug}.md"));
    let item = ReviewItem {
        metadata: EntityMetadata {
            id: id.clone(),
            created_at_millis: 1_000,
            updated_at_millis: 1_000,
            source_hashes: vec!["hash-foobar".to_string()],
            model_version: None,
            tool_version: Some("kb-test".to_string()),
            prompt_template_hash: None,
            dependencies: vec![
                "doc-a".to_string(),
                "doc-b".to_string(),
                "doc-c".to_string(),
            ],
            output_paths: vec![dest.clone()],
            status: Status::NeedsReview,
        },
        kind: ReviewKind::ConceptCandidate,
        target_entity_id: id.clone(),
        proposed_destination: Some(dest.clone()),
        citations: vec![
            "doc-a".to_string(),
            "doc-b".to_string(),
            "doc-c".to_string(),
        ],
        affected_pages: vec![dest],
        created_at_millis: 1_000,
        status: ReviewStatus::Pending,
        comment: "Term 'FooBar System' is mentioned in 3 source(s) (6 total mention(s)) but has no concept page. Sources: doc-a, doc-b, doc-c. Approve to generate wiki/concepts/foobar-system.md from the mentions.".to_string(),
    };
    save_review_item(&kb_root, &item).expect("save review item");

    let fake_bin = install_fake_concept_candidate_harness(&kb_root, "FooBar System", "storage");

    let mut cmd = kb_cmd(&kb_root);
    cmd.env("PATH", prepend_path(&fake_bin));
    cmd.arg("--json").arg("review").arg("approve").arg(&id);
    let output = cmd.output().expect("run kb review approve");
    assert!(
        output.status.success(),
        "kb review approve failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse approve json");
    assert_eq!(envelope["command"], "review.approve");
    let data = &envelope["data"];
    assert_eq!(data["kind"], "concept_candidate");
    assert_eq!(data["action"], "approved");
    assert_eq!(data["canonical_name"], "FooBar System");
    assert_eq!(data["category"], "storage");
    assert_eq!(
        data["concept_path"].as_str(),
        Some("wiki/concepts/foobar-system.md")
    );
    assert_eq!(
        data["source_document_ids"]
            .as_array()
            .expect("source_document_ids array")
            .len(),
        3
    );

    // Concept page exists with the expected frontmatter + body.
    let page_path = kb_root.join("wiki/concepts/foobar-system.md");
    assert!(
        page_path.is_file(),
        "wiki/concepts/foobar-system.md must be written"
    );
    let page = fs::read_to_string(&page_path).expect("read concept page");
    assert!(page.contains("id: concept:foobar-system"), "page: {page}");
    assert!(page.contains("name: FooBar System"), "page: {page}");
    assert!(page.contains("category: storage"), "page: {page}");
    assert!(page.contains("- doc-a"), "page: {page}");
    assert!(page.contains("- doc-b"), "page: {page}");
    assert!(page.contains("- doc-c"), "page: {page}");
    assert!(
        page.contains("FooBar System is a coordinated system"),
        "page body from LLM stub: {page}"
    );

    // Review item flipped to approved on disk.
    let review_path = kb_root
        .join("reviews")
        .join("concept_candidates")
        .join(format!("{id}.json"));
    let saved: Value =
        serde_json::from_str(&fs::read_to_string(&review_path).expect("read review"))
            .expect("parse review");
    assert_eq!(saved["status"], "approved");

    // Wiki index updated to include the new concept (bn-2zy inline refresh).
    let concepts_index = kb_root.join("wiki/concepts/index.md");
    assert!(
        concepts_index.is_file(),
        "wiki/concepts/index.md must be generated after approve"
    );
    let concepts_index_text = fs::read_to_string(&concepts_index).expect("read concepts index");
    assert!(
        concepts_index_text.contains("FooBar System")
            || concepts_index_text.contains("foobar-system"),
        "concepts index must list the new concept: {concepts_index_text}"
    );

    // Wiki global index exists and has been rebuilt.
    assert!(
        kb_root.join("wiki/index.md").is_file(),
        "wiki/index.md must be generated"
    );
}

/// Re-approving a `concept_candidate` that's already been approved must
/// fail with a clear error (not silently overwrite the concept page).
#[test]
fn review_approve_concept_candidate_rejects_already_approved() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let slug = "already-done";
    let id = format!("lint:concept-candidate:{slug}");
    let dest = PathBuf::from(format!("wiki/concepts/{slug}.md"));
    let item = ReviewItem {
        metadata: EntityMetadata {
            id: id.clone(),
            created_at_millis: 1_000,
            updated_at_millis: 1_000,
            source_hashes: vec!["hash-done".to_string()],
            model_version: None,
            tool_version: Some("kb-test".to_string()),
            prompt_template_hash: None,
            dependencies: vec!["doc-a".to_string()],
            output_paths: vec![dest.clone()],
            status: Status::NeedsReview,
        },
        kind: ReviewKind::ConceptCandidate,
        target_entity_id: id.clone(),
        proposed_destination: Some(dest.clone()),
        citations: vec!["doc-a".to_string()],
        affected_pages: vec![dest],
        created_at_millis: 1_000,
        status: ReviewStatus::Approved, // already approved
        comment: "Term 'Already Done' is mentioned".to_string(),
    };
    save_review_item(&kb_root, &item).expect("save review item");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("review").arg("approve").arg(&id);
    let output = cmd.output().expect("run kb review approve");
    assert!(
        !output.status.success(),
        "re-approving a concept_candidate must fail: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("approved") || stderr.contains("only pending"),
        "error must mention status: {stderr}"
    );
}

// ── bn-2xbq: kb migrate ────────────────────────────────────────────────────

/// Build a legacy-shaped vault by hand: `kb.toml` at the root plus every
/// subdir that bn-2xbq relocated (`cache/`, `logs/`, `state/`, `trash/`,
/// `normalized/`, `prompts/`). A sentinel file in each directory lets us
/// assert that the move preserved contents.
fn write_legacy_layout(root: &Path) {
    fs::write(root.join("kb.toml"), "\n").expect("write kb.toml");
    for sub in ["cache", "logs", "state", "trash", "normalized", "prompts"] {
        let dir = root.join(sub);
        fs::create_dir_all(&dir).expect("mkdir legacy subdir");
        fs::write(dir.join("marker.txt"), sub).expect("write marker");
    }
    // Browseable tree stays put — migrate must not touch these.
    for sub in ["raw", "wiki", "outputs", "reviews"] {
        fs::create_dir_all(root.join(sub)).expect("mkdir browseable");
    }
}

/// bn-2xbq: running `kb migrate` on a pre-.kb/ layout relocates every
/// internal plumbing directory into `.kb/`, preserves contents byte-for-byte,
/// and leaves the browseable tree alone.
#[test]
fn migrate_relocates_every_legacy_dir_into_dot_kb() {
    let (_temp_dir, kb_root) = make_temp_kb();
    write_legacy_layout(&kb_root);

    let output = kb_cmd(&kb_root)
        .arg("migrate")
        .output()
        .expect("run kb migrate");
    assert!(
        output.status.success(),
        "kb migrate failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Each legacy directory is now gone from the root and present at .kb/<sub>
    // with its marker file intact.
    for sub in ["cache", "logs", "state", "trash", "normalized", "prompts"] {
        assert!(
            !kb_root.join(sub).exists(),
            "{sub}/ should no longer exist at root"
        );
        let moved = kb_root.join(".kb").join(sub);
        assert!(
            moved.is_dir(),
            "{sub}/ should live under .kb/ — got {}",
            moved.display()
        );
        let marker = moved.join("marker.txt");
        let contents = fs::read_to_string(&marker).expect("read marker after migrate");
        assert_eq!(contents, sub, "marker bytes preserved");
    }

    // Browseable dirs untouched.
    for sub in ["raw", "wiki", "outputs", "reviews"] {
        assert!(
            kb_root.join(sub).is_dir(),
            "{sub}/ must stay at root after migrate"
        );
    }
}

/// bn-2xbq: a second `kb migrate` on an already-current layout is a no-op
/// (exit 0, friendly message).
#[test]
fn migrate_is_idempotent_on_already_current_layout() {
    let (_temp_dir, kb_root) = make_temp_kb();
    write_legacy_layout(&kb_root);
    // First migrate moves everything.
    kb_cmd(&kb_root)
        .arg("migrate")
        .output()
        .expect("first migrate");

    // Second migrate must still succeed and print the "already migrated"
    // message without touching the filesystem.
    let output = kb_cmd(&kb_root)
        .arg("migrate")
        .output()
        .expect("second migrate");
    assert!(
        output.status.success(),
        "second kb migrate failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Already migrated"),
        "second migrate should print Already migrated; got: {stdout}"
    );
}

/// bn-2xbq: mutating commands (`kb compile`, `kb ask`, …) refuse to run
/// against a legacy layout and point the user at `kb migrate`.
#[test]
fn legacy_layout_sentinel_blocks_compile_with_helpful_error() {
    let (_temp_dir, kb_root) = make_temp_kb();
    write_legacy_layout(&kb_root);

    let output = kb_cmd(&kb_root)
        .arg("compile")
        .output()
        .expect("run kb compile");
    assert!(!output.status.success(), "kb compile must fail on legacy layout");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("kb migrate"),
        "error must point users at `kb migrate`: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// bn-nlw9: slugified titles in wiki/sources/ filenames and outputs/questions/
// directory names.
//
// End-to-end checks that exercise the compile pipeline, ask flow, and the
// migrate rename pass. The compile pipeline skips its LLM-driven per-doc
// passes when no backend is configured, so these tests either drive the
// pipeline through the CLI (which exercises the id-lookup resolvers) or
// stub wiki pages directly on disk.
// ---------------------------------------------------------------------------

/// Seed the hash-state on-disk so kb's freshness tracker treats a
/// hand-written `wiki/sources/<slug>.md` as a fully compiled fresh output.
/// Test helper only — mirrors what the compile pipeline would have stored.
fn stub_nlw9_source_page(kb_root: &Path, filename: &str, src_id: &str, title: &str) -> PathBuf {
    let sources_dir = kb_root.join("wiki/sources");
    fs::create_dir_all(&sources_dir).expect("create wiki/sources");
    let path = sources_dir.join(filename);
    let md = format!(
        "---\nid: wiki-source-{src_id}\ntype: source\ntitle: {title}\n\
source_document_id: {src_id}\nsource_revision_id: rev-1\ngenerated_at: 0\n\
build_record_id: build-1\n---\n\n# Source\n\n<!-- kb:begin id=title -->\n\
{title}\n<!-- kb:end id=title -->\n",
    );
    fs::write(&path, md).expect("write wiki source page");
    path
}

/// bn-nlw9 P7.1: `kb inspect <src-id>` resolves both the legacy id-only
/// and new id-slug source wiki-page filenames. The resolver is the piece
/// that makes prefix-match lookups keep working after the rename.
#[test]
fn inspect_resolves_both_id_only_and_id_slug_source_pages() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    stub_nlw9_source_page(
        &kb_root,
        "src-slugged-hello-world-of-rust.md",
        "src-slugged",
        "Hello World of Rust",
    );
    stub_nlw9_source_page(&kb_root, "src-plain.md", "src-plain", "Plain");

    for src_id in ["src-slugged", "src-plain"] {
        let mut cmd = kb_cmd(&kb_root);
        cmd.arg("inspect").arg(src_id);
        let output = cmd.output().expect("run kb inspect");
        assert!(
            output.status.success(),
            "kb inspect {src_id} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("wiki/sources/"),
            "kb inspect {src_id} must resolve to a wiki source page; got: {stdout}"
        );
    }
}

/// bn-nlw9 P7.2: `kb forget <src-id>` moves the slug-augmented page to
/// trash when no id-only form exists on disk.
#[test]
fn forget_trashes_id_slug_source_page() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // Fabricate a normalized doc for this src so forget has something to
    // act on besides the wiki page (forget cascades through all three
    // buckets: normalized/, raw/inbox/, wiki/sources/).
    let normalized_dir = kb_core::normalized_dir(&kb_root).join("src-abc");
    fs::create_dir_all(&normalized_dir).expect("mkdir normalized");
    fs::write(
        normalized_dir.join("metadata.json"),
        r#"{"metadata": {"id": "src-abc", "status": "fresh"}, "source_revision_id": "rev-1"}"#,
    )
    .expect("write normalized metadata");
    fs::write(normalized_dir.join("source.md"), "# body\n").expect("write source.md");

    let slugged = stub_nlw9_source_page(
        &kb_root,
        "src-abc-a-meaningful-title.md",
        "src-abc",
        "A meaningful title",
    );
    assert!(slugged.exists(), "precondition: slugged page exists");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--force").arg("forget").arg("src-abc");
    let output = cmd.output().expect("run kb forget");
    assert!(
        output.status.success(),
        "kb forget failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !slugged.exists(),
        "slugged source page should have been trashed"
    );
}

/// bn-nlw9 P7.3: `kb migrate` on a vault with legacy id-only sources +
/// question dirs renames them to id-slug form and is idempotent on a
/// second run. This is the migration test backing the migrate tests at
/// the unit level with an end-to-end CLI invocation.
#[test]
fn migrate_renames_id_only_sources_and_questions_end_to_end() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // Seed an id-only source page and an id-only question dir.
    let sources_dir = kb_root.join("wiki/sources");
    fs::create_dir_all(&sources_dir).expect("mkdir sources");
    fs::write(
        sources_dir.join("src-abc.md"),
        "---\ntitle: Rust Concurrency Guide\n\
source_document_id: src-abc\n\
source_revision_id: rev-1\n---\n\n# Source\n",
    )
    .expect("write id-only source");

    let q_dir = kb_root.join("outputs/questions/q-abc");
    fs::create_dir_all(&q_dir).expect("mkdir q dir");
    fs::write(
        q_dir.join("question.json"),
        r#"{"metadata":{"id":"q-abc"},"raw_query":"What is ownership in Rust?"}"#,
    )
    .expect("write question.json");

    // First migrate: renames happen.
    let output = kb_cmd(&kb_root)
        .arg("migrate")
        .output()
        .expect("run kb migrate");
    assert!(
        output.status.success(),
        "kb migrate failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!sources_dir.join("src-abc.md").exists());
    assert!(
        sources_dir.join("src-abc-rust-concurrency-guide.md").exists(),
        "source page should be slug-augmented"
    );
    let renamed_q = kb_root
        .join("outputs/questions/q-abc-what-is-ownership-in-rust");
    assert!(
        renamed_q.is_dir(),
        "question dir should be slug-augmented: {}",
        renamed_q.display()
    );

    // Second migrate: idempotent.
    let output2 = kb_cmd(&kb_root)
        .arg("migrate")
        .output()
        .expect("run kb migrate again");
    assert!(output2.status.success());
    let stdout2 = String::from_utf8_lossy(&output2.stdout);
    assert!(
        stdout2.contains("Already migrated"),
        "second migrate should print Already migrated; got: {stdout2}"
    );
    // Both renamed artifacts still there, in slug form.
    assert!(
        sources_dir.join("src-abc-rust-concurrency-guide.md").exists()
    );
    assert!(renamed_q.is_dir());
}

/// bn-nlw9 P7.5: `kb ask` produces a question dir whose name includes a
/// slug derived from the question text, and the artifact path in the JSON
/// envelope reflects the slugged dir. Runs without an LLM — the ask flow
/// still creates the dir and writes a placeholder answer when no backend
/// is available.
#[test]
fn ask_creates_slugged_question_output_dir() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json")
        .arg("ask")
        .arg("--format")
        .arg("md")
        .arg("Produce a mermaid graph of USB team");
    let output = cmd.output().expect("run kb ask");
    assert!(
        output.status.success(),
        "kb ask failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse json");
    let artifact_rel = envelope["data"]["artifact_path"]
        .as_str()
        .expect("artifact_path");
    assert!(
        artifact_rel.starts_with("outputs/questions/q-"),
        "artifact_path must live under outputs/questions/q-*: {artifact_rel}"
    );
    assert!(
        artifact_rel.contains("-produce-a-mermaid-graph-of-usb-team"),
        "artifact_path must include the question slug: {artifact_rel}"
    );
    let artifact_abs = kb_root.join(artifact_rel);
    assert!(
        artifact_abs.exists(),
        "artifact file must exist on disk: {}",
        artifact_abs.display()
    );
    let q_dir = artifact_abs
        .parent()
        .expect("artifact has parent dir");
    let dir_name = q_dir
        .file_name()
        .expect("dir has name")
        .to_string_lossy()
        .into_owned();
    assert!(
        dir_name.starts_with("q-"),
        "question dir must start with q-: {dir_name}"
    );
    assert!(
        dir_name.contains("-produce-a-mermaid-graph-of-usb-team"),
        "question dir must include question slug: {dir_name}"
    );
}

/// bn-nlw9 P7.6: `kb ask` with a query that collapses to an empty slug
/// (e.g. only punctuation/symbols) falls back to the id-only dir name.
#[test]
fn ask_falls_back_to_id_only_dir_when_query_slugs_to_empty() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--json")
        .arg("ask")
        .arg("--format")
        .arg("md")
        .arg("!!!@#$%");
    let output = cmd.output().expect("run kb ask");
    assert!(
        output.status.success(),
        "kb ask failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse json");
    let artifact_rel = envelope["data"]["artifact_path"]
        .as_str()
        .expect("artifact_path");
    // Empty slug → no trailing `-<slug>`, just `q-<id>/answer.md`.
    let q_dir_name = Path::new(artifact_rel)
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .expect("parent dir name");
    assert!(
        q_dir_name.starts_with("q-") && !q_dir_name[2..].contains('-'),
        "empty-slug query should produce id-only dir, got: {q_dir_name}"
    );
}

/// bn-nlw9 P7.4: `kb migrate --json` surfaces the rename records under
/// `data.renamed[]` so programmatic callers can tell what moved.
#[test]
fn migrate_json_emits_rename_records() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let sources_dir = kb_root.join("wiki/sources");
    fs::create_dir_all(&sources_dir).expect("mkdir sources");
    fs::write(
        sources_dir.join("src-abc.md"),
        "---\ntitle: Short Title\nsource_document_id: src-abc\nsource_revision_id: rev-1\n---\n",
    )
    .expect("write");

    let output = kb_cmd(&kb_root)
        .arg("--json")
        .arg("migrate")
        .output()
        .expect("run kb --json migrate");
    assert!(output.status.success());
    let envelope: Value =
        serde_json::from_slice(&output.stdout).expect("parse migrate json envelope");
    let renamed = envelope["data"]["renamed"]
        .as_array()
        .expect("data.renamed should be an array");
    assert_eq!(renamed.len(), 1, "one rename; got: {renamed:?}");
    assert_eq!(renamed[0]["kind"], "source");
    assert!(
        renamed[0]["from"]
            .as_str()
            .unwrap_or("")
            .ends_with("src-abc.md")
    );
    assert!(
        renamed[0]["to"]
            .as_str()
            .unwrap_or("")
            .ends_with("src-abc-short-title.md")
    );
}

// ---------------------------------------------------------------------------
// bn-3qsj: hybrid retrieval coverage
// ---------------------------------------------------------------------------

/// Build a small fixture corpus where the relevant source uses **different
/// vocabulary** from the query. Lexical search alone would miss the right
/// page because the query word ("login") never appears in the body; the
/// page is about "credential-exchange protocol stuff" instead. A working
/// hybrid retrieval surfaces it via the semantic tier.
fn write_paraphrase_fixture(root: &Path) {
    // src-cred is the relevant page. Its body uses the morphological
    // sibling "authn" (and its variants "authenticate", "authenticated")
    // throughout — the query word "authentication" tokenizes apart from
    // "authn", so the lexical tier alone misses it. The hash-embed
    // backend is morphology-aware (its 3..=5-char n-grams overlap on
    // "auth"), so the semantic tier is what surfaces this page.
    write_source_page(
        root,
        "src-cred",
        "Authn token flow",
        "Service issues a signed authn token on a successful authenticate \
         call. The caller presents the authn token on each request; the \
         server runs an authenticate verify before accepting. Authn token \
         rotation happens once per hour. Authn pack contents include \
         issuer, subject, expiry. Authenticated callers stay authenticated \
         for the token lifetime.",
    );
    // Decoys: completely unrelated subject matter. Avoid the literal
    // "auth" prefix entirely so the n-gram overlap with the query is
    // negligible.
    write_source_page(
        root,
        "src-frontend",
        "Frontend rendering pipeline",
        "Pixel buffer composition, vertex transforms, and rasterization \
         routines. Shaders compile to a vendor-specific bytecode then run \
         on the GPU.",
    );
    write_source_page(
        root,
        "src-billing",
        "Billing pipeline overview",
        "Invoice generation, ledger reconciliation, and tax engine \
         integration. Currency conversion happens at the edge.",
    );
}

#[test]
fn compile_populates_embedding_db_and_second_compile_is_no_op() {
    let (_temp, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    write_paraphrase_fixture(&kb_root);

    // First compile should populate .kb/state/embeddings.db.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("compile");
    let output = cmd.output().expect("compile 1");
    assert!(
        output.status.success(),
        "first compile failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let db_path = kb_root.join(".kb/state/embeddings.db");
    assert!(
        db_path.exists(),
        "kb compile must populate {}",
        db_path.display()
    );

    let initial_size = fs::metadata(&db_path).expect("stat db").len();
    assert!(initial_size > 0, "embedding db should not be empty");

    // Second compile with no changes is essentially free for embeddings —
    // every page hash matches and no row is rewritten.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("compile");
    let output = cmd.output().expect("compile 2");
    assert!(
        output.status.success(),
        "second compile failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let after_size = fs::metadata(&db_path).expect("stat db 2").len();
    assert_eq!(
        initial_size, after_size,
        "second compile changed embedding db size: {initial_size} -> {after_size}"
    );
}

#[test]
fn search_paraphrased_query_beats_lexical_only_via_semantic_tier() {
    let (_temp, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    write_paraphrase_fixture(&kb_root);

    let mut compile_cmd = kb_cmd(&kb_root);
    compile_cmd.arg("compile");
    let output = compile_cmd.output().expect("compile");
    assert!(output.status.success(), "compile failed");

    // The fixture's relevant page uses "authn" / "authenticate" — the
    // query word "authentication" never appears verbatim. Pure-lexical
    // retrieval misses it (different tokens); hybrid catches it via the
    // hash-embed backend's character-n-gram overlap on "auth".
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("search").arg("authentication").arg("--json");
    let output = cmd.output().expect("hybrid search");
    assert!(
        output.status.success(),
        "search failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let envelope: Value =
        serde_json::from_slice(&output.stdout).expect("hybrid search returns json");
    let results = envelope["data"]
        .as_array()
        .expect("data array")
        .clone();
    assert!(
        !results.is_empty(),
        "hybrid search should find the credential-exchange page despite vocab mismatch"
    );
    let top = &results[0];
    assert!(
        top["item_id"]
            .as_str()
            .is_some_and(|id| id.contains("src-cred")),
        "expected the credential-exchange source on top, got: {top}"
    );
    // Reasons list should explicitly cite the semantic tier.
    let reasons = top["reasons"].as_array().expect("reasons array");
    assert!(
        reasons
            .iter()
            .any(|r| r.as_str().is_some_and(|t| t.contains("semantic match"))),
        "top hit should have a semantic-tier reason: {top}"
    );
}

#[test]
fn search_with_kb_semantic_disabled_falls_back_to_lexical_only() {
    let (_temp, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    write_paraphrase_fixture(&kb_root);
    write_source_page(
        &kb_root,
        "src-creds-keyword",
        "Authentication overview",
        "This page contains the literal word authentication so the \
         lexical tier finds it on its own without help from the \
         semantic tier.",
    );

    let mut compile_cmd = kb_cmd(&kb_root);
    compile_cmd.arg("compile");
    assert!(compile_cmd.output().expect("compile").status.success());

    // With KB_SEMANTIC=0, hybrid reduces to lexical-only — no semantic
    // reasons should appear on any result.
    let mut cmd = kb_cmd(&kb_root);
    cmd.env("KB_SEMANTIC", "0");
    cmd.arg("search").arg("authentication").arg("--json");
    let output = cmd.output().expect("disabled search");
    assert!(output.status.success());
    let envelope: Value = serde_json::from_slice(&output.stdout).expect("json envelope");
    let results = envelope["data"]
        .as_array()
        .expect("data array")
        .clone();
    assert!(
        !results.is_empty(),
        "lexical-only should still surface the keyword page"
    );
    for r in &results {
        let reasons = r["reasons"].as_array().expect("reasons");
        assert!(
            reasons
                .iter()
                .all(|reason| !reason.as_str().is_some_and(|t| t.contains("semantic match"))),
            "semantic match should not appear when KB_SEMANTIC=0; got: {r}"
        );
    }
}

#[test]
fn search_with_sqlite_vec_disabled_uses_rust_cosine_fallback_with_same_top_k() {
    let (_temp, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    write_paraphrase_fixture(&kb_root);

    let mut compile_cmd = kb_cmd(&kb_root);
    compile_cmd.arg("compile");
    assert!(compile_cmd.output().expect("compile").status.success());

    let run_search = |env: Option<(&str, &str)>| -> Vec<String> {
        let mut cmd = kb_cmd(&kb_root);
        if let Some((k, v)) = env {
            cmd.env(k, v);
        }
        cmd.arg("search").arg("authentication").arg("--json");
        let output = cmd.output().expect("search");
        assert!(output.status.success());
        let envelope: Value = serde_json::from_slice(&output.stdout).expect("json");
        envelope["data"]
            .as_array()
            .expect("data array")
            .iter()
            .filter_map(|r| r["item_id"].as_str().map(str::to_owned))
            .collect()
    };

    let with_extension = run_search(None);
    let without_extension = run_search(Some(("KB_SQLITE_VEC_AUTO", "0")));
    assert_eq!(
        with_extension, without_extension,
        "KB_SQLITE_VEC_AUTO=0 should produce identical top-K via Rust cosine fallback"
    );
}

/// bn-3rzz: a multi-section source must produce multiple rows in
/// `chunk_embeddings` (one per H2 chunk after merging) and the chunk
/// metadata (`item_id`, `chunk_idx`, heading) must be queryable via the new
/// schema.
#[test]
fn multi_section_source_produces_multiple_chunk_embeddings() {
    let (_temp, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // Two H2 sections, each comfortably above the 100-token chunk floor
    // so the merger doesn't collapse them. The chunker's body extraction
    // runs on the page body (managed-region fences stripped), so the
    // sections live alongside the standard summary region.
    let dir = kb_root.join("wiki/sources");
    fs::create_dir_all(&dir).expect("create wiki/sources");
    let body = format!(
        "---\nid: wiki-source-multi\ntype: source\ntitle: Multi-section source\n---\n\
         \n# Source\n\
         \n## Authentication\n\n{}\
         \n## Authorization\n\n{}\n",
        "auth ".repeat(200),
        "perms ".repeat(200),
    );
    fs::write(dir.join("multi.md"), body).expect("write multi-section source");

    // Drive the embedding sync directly (avoids a full `kb compile`,
    // which needs an LLM for per-doc passes).
    let backend = kb_query::HashEmbedBackend::new();
    kb_query::sync_embeddings(&kb_root, &backend).expect("sync embeddings");

    let db_path = kb_root.join(".kb/state/embeddings.db");
    assert!(db_path.exists(), "embedding db should exist after sync");
    let conn = rusqlite::Connection::open(&db_path).expect("open db");

    // The new schema MUST be in place. The legacy `item_embeddings`
    // table is dropped on first sync.
    let chunk_table_exists: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'chunk_embeddings'",
            [],
            |row| row.get(0),
        )
        .expect("check chunk_embeddings table");
    assert_eq!(chunk_table_exists, 1, "chunk_embeddings table must exist");

    // Two H2 sections → at least 2 chunk rows for this item.
    let item_id = "wiki/sources/multi.md".to_string();
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM chunk_embeddings WHERE item_id = ?1",
            rusqlite::params![item_id],
            |row| row.get(0),
        )
        .expect("count rows");
    assert!(
        count >= 2,
        "expected >=2 chunks for multi-section source, got {count}"
    );

    // Each chunk row carries the H2 heading text.
    let headings: Vec<String> = conn
        .prepare("SELECT heading FROM chunk_embeddings WHERE item_id = ?1 ORDER BY chunk_idx")
        .expect("prep")
        .query_map(rusqlite::params![item_id], |row| {
            row.get::<_, Option<String>>(0)
        })
        .expect("query")
        .filter_map(Result::ok)
        .map(Option::unwrap_or_default)
        .collect();
    assert!(
        headings.iter().any(|h| h == "Authentication"),
        "expected an Authentication heading among chunks: {headings:?}"
    );
    assert!(
        headings.iter().any(|h| h == "Authorization"),
        "expected an Authorization heading among chunks: {headings:?}"
    );
}

/// bn-3rzz: a query that matches a non-first section's content must surface
/// the source via the chunk-aggregation aggregator. We bypass the full
/// `kb search` CLI and exercise the pipeline directly so the test is
/// hermetic (no LLM, no fancy backend).
#[test]
fn search_finds_source_via_non_first_section_chunk() {
    let (_temp, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // First section is generic prose; second section contains the
    // distinctive token "ratchet" only at the bottom of the source. With
    // whole-document embedding, that token would average out under the
    // 256-token MiniLM window. Per-chunk embedding lets the second
    // chunk's vector light up against a "ratchet" query.
    let dir = kb_root.join("wiki/sources");
    fs::create_dir_all(&dir).expect("create wiki/sources");
    let body = format!(
        "---\nid: wiki-source-chunked\ntype: source\ntitle: Chunked retrieval\n---\n\
         \n## Background\n\n{}\
         \n## Ratchet specifics\n\n{}\n",
        "background unrelated material ".repeat(80),
        "ratchet ratchet ratchet protocol step by step ".repeat(80),
    );
    fs::write(dir.join("chunked.md"), body).expect("write source");

    let backend = kb_query::HashEmbedBackend::new();
    kb_query::sync_embeddings(&kb_root, &backend).expect("sync");

    // Embed the query and KNN against chunk_embeddings, then aggregate.
    let db_path = kb_root.join(".kb/state/embeddings.db");
    let conn = rusqlite::Connection::open(&db_path).expect("open db");
    let qvec = kb_query::EmbeddingBackend::embed(&backend, "ratchet protocol")
        .expect("embed query");
    let chunk_hits = kb_query::knn_search(&conn, &qvec, 16).expect("knn");
    assert!(!chunk_hits.is_empty(), "knn should find chunks");

    // The ratchet section chunk must outrank the background chunk for
    // this query — chunk-level embedding is the whole point of bn-3rzz.
    let item_id = "wiki/sources/chunked.md".to_string();
    let our_chunks: Vec<&kb_query::SemanticChunkHit> = chunk_hits
        .iter()
        .filter(|c| c.item_id == item_id)
        .collect();
    assert!(
        our_chunks.len() >= 2,
        "expected >=2 chunks indexed for the source, got {}",
        our_chunks.len()
    );
    let top_for_item = our_chunks[0];
    assert_eq!(
        top_for_item.heading.as_deref(),
        Some("Ratchet specifics"),
        "ratchet query should rank the ratchet chunk above background; \
         got top heading {:?}",
        top_for_item.heading
    );

    // Aggregator: the source surfaces with the ratchet chunk attached.
    let item_hits = kb_query::aggregate_chunks_to_items(chunk_hits, 5);
    let our_item = item_hits
        .iter()
        .find(|h| h.item_id == item_id)
        .expect("source aggregated");
    let best = our_item.best_chunk.as_ref().expect("best chunk attached");
    assert_eq!(best.heading.as_deref(), Some("Ratchet specifics"));
}

#[test]
fn forget_eagerly_drops_embedding_row_for_source() {
    let (_temp, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // Ingest + stub a wiki source page so we have something forget can act
    // on, then sync the embedding pipeline directly. This avoids a full
    // `kb compile` (which would need an LLM to do per-doc passes); the
    // embedding sync only needs the wiki page on disk.
    let source = kb_root.join("seed.md");
    fs::write(&source, "# hi\n\nbody\n").expect("write source");
    let src_id = ingest_single_and_get_src_id(&kb_root, &source);
    stub_wiki_source_page(&kb_root, &src_id);

    let backend = kb_query::HashEmbedBackend::new();
    kb_query::sync_embeddings(&kb_root, &backend).expect("sync embeddings");

    let db_path = kb_root.join(".kb/state/embeddings.db");
    assert!(db_path.exists(), "embedding store should exist after sync");
    let conn = rusqlite::Connection::open(&db_path).expect("open db");
    let pre_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM chunk_embeddings WHERE item_id = ?1",
            rusqlite::params![format!("wiki/sources/{src_id}.md")],
            |row| row.get(0),
        )
        .expect("pre count");
    assert!(pre_count >= 1, "expected >=1 chunk row for {src_id}, got {pre_count}");
    drop(conn);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("--force").arg("forget").arg(&src_id);
    let output = cmd.output().expect("run kb forget");
    assert!(
        output.status.success(),
        "kb forget failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Eager: row gone before any subsequent compile runs.
    let conn = rusqlite::Connection::open(&db_path).expect("reopen db");
    let post_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM chunk_embeddings WHERE item_id = ?1",
            rusqlite::params![format!("wiki/sources/{src_id}.md")],
            |row| row.get(0),
        )
        .expect("post count");
    assert_eq!(post_count, 0, "kb forget must eagerly drop the embedding rows");
}

// bn-166d: a fake opencode that emits a deterministic answer body
// containing one verifiable and one fabricated quote, both citing
// `src-rust-quote`. Used to exercise the `verified: N/M quotes...`
// footer the post-process appends to ask answers.
fn install_fake_quote_harness(root: &Path) -> PathBuf {
    let bin_dir = root.join("fake-quote-bin");
    fs::create_dir_all(&bin_dir).expect("create fake quote bin dir");
    let script = r#"#!/bin/sh
# Drain stdin (prompt) and ignore — the answer is fully canned. The
# answer must be markdown that lands as `artifact_body` so the post-
# process verifier sees the quoted spans.
cat >/dev/null
cat <<'ANSWER'
Per the docs: "memory safety without garbage collection" [src-rust-quote].

Some agents claim "rust runs natively on quantum hardware" [src-rust-quote],
which would be neat.
ANSWER
"#;
    write_executable(bin_dir.join("opencode").as_path(), script);
    write_executable(
        bin_dir.join("claude").as_path(),
        "#!/bin/sh\nprintf '{\"result\":\"OK\"}'",
    );
    bin_dir
}

#[test]
fn ask_answer_footer_counts_verified_and_unverified_quotes() {
    let (_temp_dir, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // Source page must contain the *real* quote so it verifies and the
    // *fake* quote stays missing. The slug must include the `src-`
    // prefix so the verifier resolves `[src-rust-quote]` to this file
    // (production src-ids always start with `src-`; the
    // `write_source_page` helper takes the bare slug).
    write_source_page(
        &kb_root,
        "src-rust-quote",
        "Rust language",
        "Rust offers memory safety without garbage collection — and zero-cost abstractions.",
    );

    let fake_bin = install_fake_quote_harness(&kb_root);
    let mut cmd = kb_cmd(&kb_root);
    cmd.env("PATH", prepend_path(&fake_bin));
    cmd.arg("--json")
        .arg("ask")
        .arg("--format")
        .arg("md")
        .arg("Tell me about Rust safety.");
    let output = cmd.output().expect("run kb ask --format=md");
    assert!(
        output.status.success(),
        "kb ask failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse ask json");
    let artifact_path = kb_root.join(
        envelope["data"]["artifact_path"]
            .as_str()
            .expect("artifact_path"),
    );
    let content = fs::read_to_string(&artifact_path).expect("read answer");

    // Footer must report 1/2 verified — the real quote hits the source
    // page, the quantum-hardware quote does not.
    assert!(
        content.contains("**verified: 1/2 quotes found in sources**"),
        "expected verified footer, got:\n{content}"
    );
    assert!(
        content.contains("Unverified quotes:"),
        "expected unverified-quote section in footer:\n{content}"
    );
    assert!(
        content.contains("rust runs natively on quantum hardware"),
        "expected the fabricated quote to be listed in the footer:\n{content}"
    );
    // The verified quote must NOT appear under the unverified section,
    // even though it shares the src-id. We assert the order: footer is
    // appended after the body, so check the substring positions.
    let footer_start = content
        .find("**verified:")
        .expect("footer marker present");
    let unverified_section = &content[footer_start..];
    assert!(
        !unverified_section.contains("memory safety without garbage collection"),
        "verified quote should not be re-listed under Unverified quotes:\n{unverified_section}"
    );
}

// ---------------------------------------------------------------------------
// bn-3sco: golden Q/A retrieval-eval harness
// ---------------------------------------------------------------------------

#[test]
#[allow(clippy::too_many_lines, reason = "end-to-end test exercises three CLI invocations and validates output shape")]
fn eval_run_writes_json_and_markdown_with_finite_metrics() {
    let (_temp, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    // Tiny corpus: paraphrase fixture is enough to give the hybrid
    // retriever something to rank.
    write_paraphrase_fixture(&kb_root);

    // Compile builds the lexical (+ semantic) index. No LLM call is
    // needed because the eval harness skips LLMs entirely.
    let mut compile_cmd = kb_cmd(&kb_root);
    compile_cmd.arg("compile");
    let compile_output = compile_cmd.output().expect("compile");
    assert!(
        compile_output.status.success(),
        "compile failed: {}",
        String::from_utf8_lossy(&compile_output.stderr)
    );

    // Drop a small golden.toml at <kb-root>/evals/golden.toml.
    let evals_dir = kb_root.join("evals");
    fs::create_dir_all(&evals_dir).expect("mkdir evals");
    let golden_toml = r#"
[[query]]
id = "auth-paraphrase"
query = "how does authentication work?"
expected_sources = ["src-cred"]

[[query]]
id = "billing-overview"
query = "explain the billing pipeline"
expected_sources = ["src-billing"]
"#;
    fs::write(evals_dir.join("golden.toml"), golden_toml).expect("write golden.toml");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("eval").arg("run");
    let output = cmd.output().expect("kb eval run");
    assert!(
        output.status.success(),
        "kb eval run failed: stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    // The latest.md summary should always be present after a successful run.
    let latest_md = kb_root.join("evals/results/latest.md");
    assert!(
        latest_md.exists(),
        "expected {} to be written",
        latest_md.display()
    );
    let md = fs::read_to_string(&latest_md).expect("read latest.md");
    assert!(md.contains("auth-paraphrase"), "got latest.md: {md}");
    assert!(md.contains("Aggregate"), "got latest.md: {md}");

    // Locate the JSON result file (single file in evals/results/ apart
    // from latest.md).
    let results_dir = kb_root.join("evals/results");
    let json_path = fs::read_dir(&results_dir)
        .expect("read results dir")
        .filter_map(Result::ok)
        .find(|e| e.path().extension().and_then(|s| s.to_str()) == Some("json"))
        .map(|e| e.path())
        .expect("json result file present");

    let raw = fs::read_to_string(&json_path).expect("read json result");
    let payload: Value = serde_json::from_str(&raw).expect("parse eval json");

    // Sanity: top-level shape.
    assert!(payload["run_id"].is_string(), "run_id missing: {payload}");
    assert!(payload["backend_id"].is_string(), "backend_id missing");
    let queries = payload["queries"].as_array().expect("queries array");
    assert_eq!(queries.len(), 2, "expected two query results");

    // Aggregate metrics must exist and be finite numbers (not NaN, not
    // null) so callers can serialize them onward.
    let agg = &payload["aggregate"];
    for metric in ["p_at_5", "p_at_10", "mrr", "ndcg_10"] {
        let value = agg[metric].as_f64();
        assert!(
            value.is_some_and(f64::is_finite),
            "aggregate.{metric} should be a finite number, got {:?}",
            agg[metric]
        );
    }

    // Per-query metrics must also be finite.
    for q in queries {
        for metric in ["p_at_5", "p_at_10", "mrr", "ndcg_10"] {
            let value = q[metric].as_f64();
            assert!(
                value.is_some_and(f64::is_finite),
                "query.{metric} should be a finite number for {}, got {:?}",
                q["id"], q[metric]
            );
        }
        // ranking is a (possibly empty) array of strings.
        assert!(q["ranking"].is_array(), "ranking missing on {q}");
    }

    // Now exercise --baseline + --save-as in a single follow-up run.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("eval")
        .arg("run")
        .arg("--save-as")
        .arg("snap1");
    let output = cmd.output().expect("save-as run");
    assert!(
        output.status.success(),
        "save-as run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let snap_path = results_dir.join("snap1.json");
    assert!(snap_path.exists(), "snap1.json should exist after --save-as");

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("eval")
        .arg("run")
        .arg("--baseline")
        .arg("snap1");
    let output = cmd.output().expect("baseline run");
    assert!(
        output.status.success(),
        "baseline run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Eval diff"),
        "baseline output should include the diff header, got:\n{stdout}"
    );

    // `kb eval list` should now show at least the snap1 + two timestamped
    // runs in the listing.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("eval").arg("list");
    let output = cmd.output().expect("list");
    assert!(output.status.success(), "list failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("snap1"),
        "list should include snap1: {stdout}"
    );
    assert!(
        stdout.contains("(latest)"),
        "list should mark a latest entry: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// bn-o6wv: conversational `kb ask --session` and `kb session ...`
// ---------------------------------------------------------------------------

/// Fake opencode that records the *last* prompt it was given to a sentinel
/// file under `<root>/.kb/state/last-prompt.txt` and emits a deterministic
/// answer that depends on whether the prompt looks like a rewrite call or
/// an answer call. We need both branches to behave differently for the
/// turn-2-uses-prior-context assertion to be meaningful.
fn install_fake_session_harness(root: &Path) -> PathBuf {
    let bin_dir = root.join("fake-session-bin");
    fs::create_dir_all(&bin_dir).expect("create fake session bin dir");
    // Pre-create the state dir so the sentinel write doesn't race the
    // first `kb` invocation.
    fs::create_dir_all(root.join(".kb").join("state"))
        .expect("create state dir for sentinel");

    let prompt_path = root.join(".kb").join("state").join("last-prompt.txt");
    let prompt_path_str = prompt_path.to_string_lossy().into_owned();

    let script_template = "#!/bin/sh\n\
# Drain stdin into the sentinel so the test can assert what the prompt\n\
# actually looked like.\n\
prompt_path=__PROMPT_PATH__\n\
tmp=$(mktemp)\n\
cat >\"$tmp\"\n\
cp \"$tmp\" \"$prompt_path\"\n\
\n\
# Rewrite calls end with the literal 'Rewritten retrieval query' header.\n\
# Answer calls go through ask.md or ask_session.md and start with the\n\
# question banner.\n\
if grep -q 'Rewritten retrieval query' \"$tmp\" >/dev/null 2>&1; then\n\
    if grep -q 'consensus' \"$tmp\" >/dev/null 2>&1; then\n\
        printf 'consensus algorithm raft differences from paxos\\n'\n\
    else\n\
        printf 'rewritten retrieval query placeholder\\n'\n\
    fi\n\
else\n\
    # Answer call. When the prompt has any role-tagged turn ('[user]' or\n\
    # '[assistant]') in the conversation block, the session run is on\n\
    # turn 2+; emit an answer that proves we considered the prior turn.\n\
    if grep -qE '\\[(user|assistant)\\]' \"$tmp\" >/dev/null 2>&1; then\n\
        printf 'Building on the earlier turn about Paxos, Raft achieves consensus through leader election [1].\\n'\n\
    else\n\
        printf 'Consensus is the problem of agreeing on a single value across distributed nodes [1].\\n'\n\
    fi\n\
fi\n";
    let script = script_template.replace(
        "__PROMPT_PATH__",
        &shell_quote_single(&prompt_path_str),
    );
    write_executable(bin_dir.join("opencode").as_path(), &script);
    write_executable(
        bin_dir.join("claude").as_path(),
        "#!/bin/sh\nprintf '{\"result\":\"OK\"}'",
    );
    bin_dir
}

/// Shell-quote `s` with single quotes for safe inclusion in `/bin/sh` heredocs.
fn shell_quote_single(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for ch in s.chars() {
        if ch == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

#[test]
fn session_first_turn_creates_session_file_with_two_turns() {
    let (_temp, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    write_source_page(
        &kb_root,
        "paxos",
        "Paxos",
        "Paxos is a family of protocols for solving consensus.",
    );

    let fake_bin = install_fake_session_harness(&kb_root);
    let mut cmd = kb_cmd(&kb_root);
    cmd.env("PATH", prepend_path(&fake_bin));
    cmd.arg("--json")
        .arg("ask")
        .arg("--session")
        .arg("s-test")
        .arg("what is consensus?");
    let output = cmd.output().expect("kb ask --session");
    assert!(
        output.status.success(),
        "kb ask --session failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Session file must exist with two turns (user + assistant).
    let session_path = kb_root.join(".kb/sessions/s-test.json");
    assert!(
        session_path.is_file(),
        "session file should exist at {}",
        session_path.display()
    );
    let raw = fs::read_to_string(&session_path).expect("read session file");
    let parsed: Value = serde_json::from_str(&raw).expect("parse session json");
    assert_eq!(parsed["id"], "s-test");
    let turns = parsed["turns"].as_array().expect("turns array");
    assert_eq!(turns.len(), 2, "expected 2 turns after first ask, got {turns:?}");
    assert_eq!(turns[0]["role"], "user");
    assert_eq!(turns[0]["text"], "what is consensus?");
    assert_eq!(turns[1]["role"], "assistant");
    assert!(
        turns[1]["text"]
            .as_str()
            .is_some_and(|t| t.contains("Consensus is")),
        "first-turn answer should be the canned consensus reply, got {}",
        turns[1]["text"]
    );

    // The JSON envelope echoes the session metadata.
    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse json envelope");
    assert_eq!(envelope["data"]["session_id"], "s-test");
    assert_eq!(envelope["data"]["turn_count"], 2);
}

#[test]
fn session_second_turn_uses_prior_context_via_rewrite() {
    let (_temp, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    // Sources for both consensus and Paxos so retrieval has something to
    // pull on each turn — the rewrite injects "consensus" into turn 2's
    // retrieval query, which finds the Paxos page.
    write_source_page(
        &kb_root,
        "paxos",
        "Paxos",
        "Paxos is the canonical consensus algorithm.",
    );
    write_source_page(
        &kb_root,
        "raft",
        "Raft",
        "Raft is an alternative consensus algorithm to Paxos.",
    );

    let fake_bin = install_fake_session_harness(&kb_root);

    // Turn 1: seed the session.
    let mut cmd = kb_cmd(&kb_root);
    cmd.env("PATH", prepend_path(&fake_bin));
    cmd.arg("ask")
        .arg("--session")
        .arg("s-multi")
        .arg("what is consensus?");
    let output = cmd.output().expect("kb ask --session turn 1");
    assert!(
        output.status.success(),
        "turn 1 failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Turn 2: a follow-up question. The rewrite step should rewrite
    // "what about raft?" against the prior turn's "consensus" context.
    let mut cmd = kb_cmd(&kb_root);
    cmd.env("PATH", prepend_path(&fake_bin));
    cmd.arg("--json")
        .arg("ask")
        .arg("--session")
        .arg("s-multi")
        .arg("what about raft?");
    let output = cmd.output().expect("kb ask --session turn 2");
    assert!(
        output.status.success(),
        "turn 2 failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // After turn 2 the file must hold 4 turns.
    let session_path = kb_root.join(".kb/sessions/s-multi.json");
    let raw = fs::read_to_string(&session_path).expect("read session file");
    let parsed: Value = serde_json::from_str(&raw).expect("parse session json");
    let turns = parsed["turns"].as_array().expect("turns array");
    assert_eq!(turns.len(), 4, "expected 4 turns after two asks, got {}", turns.len());
    assert_eq!(turns[2]["role"], "user");
    assert_eq!(turns[2]["text"], "what about raft?");
    let retrieved_ids = turns[2]["retrieved_ids"]
        .as_array()
        .expect("retrieved_ids array");
    assert!(
        !retrieved_ids.is_empty(),
        "turn 2 user retrieved_ids must reflect the rewrite-driven retrieval"
    );

    // The JSON envelope from turn 2 also echoes the rewritten query.
    let envelope: Value = serde_json::from_slice(&output.stdout).expect("parse envelope");
    let rewritten = envelope["data"]["rewritten_query"]
        .as_str()
        .expect("rewritten_query string");
    assert!(
        rewritten.to_lowercase().contains("consensus")
            || rewritten.to_lowercase().contains("paxos"),
        "rewritten query should pull in prior-turn context, got {rewritten:?}"
    );

    // Turn 2's answer should reference the prior-turn topic ("Building
    // on the earlier turn about Paxos") because the answer prompt
    // includes the conversation history.
    let answer = turns[3]["text"].as_str().expect("turn 3 text");
    assert!(
        answer.contains("Building on the earlier turn"),
        "turn 2 answer should include prior-context reference, got {answer:?}"
    );

    // The answer prompt that was fed to the LLM must include the
    // conversation transcript section.
    let prompt = fs::read_to_string(kb_root.join(".kb/state/last-prompt.txt"))
        .expect("read last-prompt sentinel");
    assert!(
        prompt.contains("Conversation so far"),
        "answer prompt must include the conversation block, got:\n{prompt}"
    );
    assert!(
        prompt.contains("[user]") && prompt.contains("[assistant]"),
        "conversation block must list role-tagged turns, got:\n{prompt}"
    );
}

#[test]
fn session_list_and_show_inspect_a_session() {
    let (_temp, kb_root) = make_temp_kb();
    init_kb(&kb_root);
    write_source_page(&kb_root, "paxos", "Paxos", "Paxos is consensus.");
    let fake_bin = install_fake_session_harness(&kb_root);

    // Add one turn so a session file lands on disk.
    let mut cmd = kb_cmd(&kb_root);
    cmd.env("PATH", prepend_path(&fake_bin));
    cmd.arg("ask").arg("--session").arg("s-inspect").arg("what is consensus?");
    let output = cmd.output().expect("kb ask --session");
    assert!(
        output.status.success(),
        "kb ask failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // `kb session list` should mention the session.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("session").arg("list");
    let output = cmd.output().expect("kb session list");
    assert!(
        output.status.success(),
        "kb session list failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("s-inspect"), "list should mention session: {stdout}");

    // `kb session show <id>` should print the transcript with both turns.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("session").arg("show").arg("s-inspect");
    let output = cmd.output().expect("kb session show");
    assert!(
        output.status.success(),
        "kb session show failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("what is consensus?"),
        "show should include the user turn: {stdout}"
    );
    assert!(
        stdout.contains("[user]") && stdout.contains("[assistant]"),
        "show should label both roles: {stdout}"
    );
}

#[test]
fn session_new_creates_empty_session() {
    let (_temp, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("session").arg("new").arg("custom-id");
    let output = cmd.output().expect("kb session new");
    assert!(
        output.status.success(),
        "kb session new failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let session_path = kb_root.join(".kb/sessions/custom-id.json");
    assert!(session_path.is_file(), "session file should exist");

    let raw = fs::read_to_string(&session_path).expect("read session file");
    let parsed: Value = serde_json::from_str(&raw).expect("parse json");
    assert_eq!(parsed["id"], "custom-id");
    assert_eq!(
        parsed["turns"].as_array().map_or(99, Vec::len),
        0,
        "fresh session must start with no turns"
    );

    // Re-running with the same id must error rather than clobber.
    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("session").arg("new").arg("custom-id");
    let output = cmd.output().expect("kb session new dup");
    assert!(
        !output.status.success(),
        "second `session new` with same id should fail"
    );
}

#[test]
fn session_rejects_path_escape_in_id() {
    let (_temp, kb_root) = make_temp_kb();
    init_kb(&kb_root);

    let mut cmd = kb_cmd(&kb_root);
    cmd.arg("ask")
        .arg("--session")
        .arg("../escape")
        .arg("what is consensus?");
    let output = cmd.output().expect("kb ask --session ../escape");
    assert!(
        !output.status.success(),
        "session id with path-escape characters must be rejected"
    );
}
