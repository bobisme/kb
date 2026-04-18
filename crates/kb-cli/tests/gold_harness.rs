mod common;

use common::{kb_cmd, make_temp_kb};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

const QUESTIONS_FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/gold/questions.toml");
const CORPUS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/gold/corpus");
const BASELINE_FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/gold/baseline.json");

// ── Data types ───────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct GoldQuestion {
    id: String,
    query: String,
    expected_sources: Vec<String>,
    #[allow(dead_code)]
    notes: Option<String>,
}

#[derive(Debug, Deserialize)]
struct QuestionsFile {
    questions: Vec<GoldQuestion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuestionResult {
    id: String,
    query: String,
    /// Fraction of expected sources that appeared in the retrieval plan (0.0–1.0).
    retrieval_coverage: f64,
    /// Number of expected sources found in the retrieval plan.
    retrieved_expected: usize,
    /// Total expected sources for this question.
    total_expected: usize,
    /// Candidate titles that appeared in the retrieval plan.
    retrieved_titles: Vec<String>,
    /// Number of valid LLM citation references (`None` when LLM unavailable).
    valid_citations: Option<usize>,
    /// Number of invalid (hallucinated) LLM citation references.
    invalid_citations: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
struct BaselineRun {
    /// Date of the run (ISO 8601).
    date: String,
    /// Git commit or "unknown".
    git_ref: String,
    /// Average retrieval coverage across all questions.
    mean_retrieval_coverage: f64,
    /// Average LLM citation coverage (`None` when all questions lack LLM data).
    mean_llm_citation_coverage: Option<f64>,
    /// Average hallucination rate (`None` when all questions lack LLM data).
    mean_hallucination_rate: Option<f64>,
    /// Per-question results.
    questions: Vec<QuestionResult>,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn write_fake_harnesses(bin_dir: &Path) {
    use std::os::unix::fs::PermissionsExt;
    fs::create_dir_all(bin_dir).expect("create fake bin dir");

    for name in &["opencode", "claude"] {
        let path = bin_dir.join(name);
        let content = if *name == "claude" {
            "#!/bin/sh\nprintf '{\"result\":\"OK\"}'"
        } else {
            "#!/bin/sh\nprintf 'OK'"
        };
        fs::write(&path, content).expect("write fake harness");
        let mut perms = fs::metadata(&path).expect("metadata").permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&path, perms).expect("chmod");
    }
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

fn init_kb(root: &Path) {
    let out = kb_cmd(root).arg("init").output().expect("kb init");
    assert!(
        out.status.success(),
        "kb init failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn stage_corpus_as_wiki_sources(root: &Path, corpus: &Path) {
    let wiki_dir = root.join("wiki/sources");
    fs::create_dir_all(&wiki_dir).expect("create wiki/sources");

    let mut entries: Vec<_> = fs::read_dir(corpus)
        .expect("read gold corpus dir")
        .collect::<Result<Vec<_>, _>>()
        .expect("collect gold corpus entries");
    entries.sort_by_key(std::fs::DirEntry::path);

    for entry in entries {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("md") {
            continue;
        }

        let slug = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .expect("gold corpus file stem");
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

        let mut lines = source.lines();
        let title = lines
            .next()
            .and_then(|line| line.strip_prefix("# "))
            .unwrap_or(slug)
            .trim();
        let body = lines.collect::<Vec<_>>().join("\n").trim().to_string();

        let page = format!(
            "---\nid: wiki-source-{slug}\ntype: source\ntitle: {title}\n---\n\n{}\n",
            if body.is_empty() {
                format!("# {title}")
            } else {
                body
            }
        );
        fs::write(wiki_dir.join(format!("{slug}.md")), page)
            .unwrap_or_else(|e| panic!("write staged corpus page {slug}: {e}"));
    }
}

fn compile_kb(root: &Path, path_env: &str) {
    let out = kb_cmd(root)
        .env("PATH", path_env)
        .arg("compile")
        .output()
        .expect("kb compile");
    assert!(
        out.status.success(),
        "kb compile failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Run `kb ask --json <query>` and return the parsed JSON envelope.
fn run_ask(root: &Path, query: &str, path_env: &str) -> Value {
    let out = kb_cmd(root)
        .env("PATH", path_env)
        .arg("--json")
        .arg("ask")
        .arg(query)
        .output()
        .expect("kb ask");
    assert!(
        out.status.success(),
        "kb ask failed for query {query:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("parse ask JSON")
}

/// Read a JSON file from the KB root relative to the ask output paths.
fn read_json_file(root: &Path, rel_path: &str) -> Value {
    let path = root.join(rel_path);
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&content).expect("parse JSON")
}

/// Return the parent directory of an artifact path as a relative string.
fn artifact_parent(artifact_path: &str) -> String {
    PathBuf::from(artifact_path)
        .parent()
        .expect("artifact parent dir")
        .to_string_lossy()
        .into_owned()
}

/// Check what fraction of `expected` titles appear (case-insensitive substring)
/// among the `candidates` titles from the retrieval plan.
#[allow(clippy::cast_precision_loss)]
fn retrieval_coverage(candidates: &[String], expected: &[String]) -> (f64, usize) {
    if expected.is_empty() {
        return (1.0, 0);
    }
    let found = expected
        .iter()
        .filter(|exp| {
            let exp_lower = exp.to_lowercase();
            candidates
                .iter()
                .any(|c| c.to_lowercase().contains(&exp_lower))
        })
        .count();
    let coverage = found as f64 / expected.len() as f64;
    (coverage, found)
}

fn load_questions() -> Vec<GoldQuestion> {
    let content = fs::read_to_string(QUESTIONS_FILE).expect("read questions.toml");
    let parsed: QuestionsFile = toml::from_str(&content).expect("parse questions.toml");
    parsed.questions
}

fn git_ref() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok().map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn score_questions(
    kb_root: &Path,
    questions: &[GoldQuestion],
    path_env: &str,
) -> Vec<QuestionResult> {
    questions
        .iter()
        .map(|q| {
            let envelope = run_ask(kb_root, &q.query, path_env);
            let artifact_path = envelope["data"]["artifact_path"]
                .as_str()
                .expect("artifact_path in ask output");

            let parent = artifact_parent(artifact_path);
            let plan = read_json_file(kb_root, &format!("{parent}/retrieval_plan.json"));
            let sidecar = read_json_file(kb_root, &format!("{parent}/metadata.json"));

            let candidate_titles: Vec<String> = plan["candidates"]
                .as_array()
                .map_or_else(Vec::new, |arr| {
                    arr.iter()
                        .filter_map(|c| c["title"].as_str().map(String::from))
                        .collect()
                });

            let (coverage, found) = retrieval_coverage(&candidate_titles, &q.expected_sources);
            let valid_cits = sidecar["valid_citations"].as_array().map(Vec::len);
            let invalid_cits = sidecar["invalid_citations"].as_array().map(Vec::len);

            QuestionResult {
                id: q.id.clone(),
                query: q.query.clone(),
                retrieval_coverage: coverage,
                retrieved_expected: found,
                total_expected: q.expected_sources.len(),
                retrieved_titles: candidate_titles,
                valid_citations: valid_cits,
                invalid_citations: invalid_cits,
            }
        })
        .collect()
}

// ── Tests ────────────────────────────────────────────────────────────────────

/// Measures retrieval coverage: for each gold question, checks that the expected
/// source documents appear in the retrieval plan produced by `kb ask`.
///
/// This test runs in CI using a fake LLM backend. It verifies the lexical index
/// and retrieval plan construction, not LLM citation quality.
#[allow(clippy::cast_precision_loss)]
#[test]
fn gold_retrieval_coverage() {
    let corpus_path = PathBuf::from(CORPUS_DIR);
    if !corpus_path.exists() {
        eprintln!(
            "gold corpus not found at {CORPUS_DIR} — skipping gold_retrieval_coverage"
        );
        return;
    }

    let questions = load_questions();
    assert!(
        !questions.is_empty(),
        "questions.toml must contain at least one question"
    );

    let (_temp_dir, kb_root) = make_temp_kb();
    let bin_dir = kb_root.join("fake-bin");
    write_fake_harnesses(&bin_dir);
    let path_env = prepend_path(&bin_dir);

    init_kb(&kb_root);
    stage_corpus_as_wiki_sources(&kb_root, &corpus_path);
    compile_kb(&kb_root, &path_env);

    let results = score_questions(&kb_root, &questions, &path_env);

    let total_coverage: f64 = results.iter().map(|r| r.retrieval_coverage).sum();
    let mean_coverage = total_coverage / questions.len() as f64;
    print_coverage_report(&results, mean_coverage);

    if std::env::var("GOLD_RECORD_BASELINE").is_ok() {
        write_baseline(&results, mean_coverage, None, None);
    }

    // Retrieval coverage must be at least 70% on average to pass.
    assert!(
        mean_coverage >= 0.70,
        "mean retrieval coverage {mean_coverage:.2} is below threshold 0.70 — \
         check that the gold corpus was ingested and compiled correctly"
    );
}

/// Measures LLM citation coverage and hallucination rate against a real backend.
///
/// Run with:
/// ```text
/// cargo test --test gold_harness -- --ignored
/// ```
///
/// To record the results as the new baseline:
/// ```text
/// GOLD_RECORD_BASELINE=1 cargo test --test gold_harness -- --ignored
/// ```
#[allow(clippy::cast_precision_loss)]
#[test]
#[ignore = "requires real LLM backend; run manually and set GOLD_RECORD_BASELINE=1 to update"]
fn gold_llm_citation_quality() {
    let corpus_path = PathBuf::from(CORPUS_DIR);
    assert!(
        corpus_path.exists(),
        "gold corpus not found at {CORPUS_DIR}"
    );

    let questions = load_questions();
    let (_temp_dir, kb_root) = make_temp_kb();
    let path_env = std::env::var("PATH").unwrap_or_default();

    init_kb(&kb_root);
    stage_corpus_as_wiki_sources(&kb_root, &corpus_path);
    compile_kb(&kb_root, &path_env);

    let results = score_questions(&kb_root, &questions, &path_env);

    let total_retrieval: f64 = results.iter().map(|r| r.retrieval_coverage).sum();
    let mean_retrieval = total_retrieval / questions.len() as f64;

    let mut total_llm_citation = 0.0_f64;
    let mut total_hallucination = 0.0_f64;
    let mut llm_question_count = 0_usize;

    for (r, q) in results.iter().zip(questions.iter()) {
        let valid = r.valid_citations.unwrap_or(0);
        let invalid = r.invalid_citations.unwrap_or(0);
        let total_cits = valid + invalid;

        // Only count questions where the LLM produced citations.
        if total_cits > 0 || r.valid_citations.is_some() {
            let hallucination_rate = if total_cits > 0 {
                invalid as f64 / total_cits as f64
            } else {
                0.0
            };
            let llm_cit_cov = if q.expected_sources.is_empty() {
                1.0
            } else {
                valid as f64 / q.expected_sources.len() as f64
            };
            total_llm_citation += llm_cit_cov;
            total_hallucination += hallucination_rate;
            llm_question_count += 1;
        }
    }

    let mean_llm_cov = if llm_question_count > 0 {
        Some(total_llm_citation / llm_question_count as f64)
    } else {
        None
    };
    let mean_hallucination = if llm_question_count > 0 {
        Some(total_hallucination / llm_question_count as f64)
    } else {
        None
    };

    print_coverage_report(&results, mean_retrieval);
    if let Some(h) = mean_hallucination {
        println!("mean hallucination rate: {h:.2}");
    }

    if std::env::var("GOLD_RECORD_BASELINE").is_ok() {
        write_baseline(&results, mean_retrieval, mean_llm_cov, mean_hallucination);
    }
}

// ── Reporting ────────────────────────────────────────────────────────────────

fn print_coverage_report(results: &[QuestionResult], mean_coverage: f64) {
    println!("\n── Gold Question Retrieval Coverage ────────────────────────────────");
    for r in results {
        let mark = if r.retrieval_coverage >= 1.0 { "✓" } else { "✗" };
        println!(
            "  {mark} {} [{}/{} expected] {:.0}%  \"{}\"",
            r.id,
            r.retrieved_expected,
            r.total_expected,
            r.retrieval_coverage * 100.0,
            r.query
        );
    }
    println!("────────────────────────────────────────────────────────────────────");
    println!(
        "  mean retrieval coverage: {mean_coverage:.2} ({:.0}%)",
        mean_coverage * 100.0
    );
}

fn write_baseline(
    results: &[QuestionResult],
    mean_retrieval: f64,
    mean_llm_cov: Option<f64>,
    mean_hallucination: Option<f64>,
) {
    let baseline = BaselineRun {
        date: chrono_now(),
        git_ref: git_ref(),
        mean_retrieval_coverage: mean_retrieval,
        mean_llm_citation_coverage: mean_llm_cov,
        mean_hallucination_rate: mean_hallucination,
        questions: results.to_vec(),
    };

    let json = serde_json::to_string_pretty(&baseline).expect("serialize baseline");
    let path = PathBuf::from(BASELINE_FILE);
    fs::write(&path, json)
        .unwrap_or_else(|e| eprintln!("warning: could not write baseline: {e}"));
    println!("baseline recorded to {}", path.display());
}

fn chrono_now() -> String {
    std::process::Command::new("date")
        .arg("+%Y-%m-%dT%H:%M:%SZ")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok().map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}
