use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use kb_core::{
    hash_bytes, hash_many, normalized_dir, read_normalized_document, save_build_record,
    slug_from_title,
};
#[cfg(test)]
use kb_core::{prompts_dir, state_dir};
use kb_llm::{ConceptCandidate, LlmAdapter};

use crate::progress::{LineLogReporter, ProgressReporter};
use crate::{Graph, HashState, StaleNode, detect_stale};

/// No-op reporter used when `options.progress = false` and no reporter is
/// configured on `CompileOptions`. Keeps the pipeline's event emission
/// unconditional so log-sink forwarding doesn't need a second code path.
struct NullReporter;

impl ProgressReporter for NullReporter {
    fn pass_start(&self, _pass: &str, _total: usize) {}
    fn pass_item_start(&self, _pass: &str, _item: &str) {}
    fn pass_item_done(&self, _pass: &str, _item: &str, _elapsed: Duration) {}
    fn pass_done(&self, _pass: &str, _affected: usize, _elapsed: Duration) {}
    fn info(&self, _message: &str) {}
    fn error(&self, _message: &str) {}
}

/// Abstract sink that receives per-pass progress events so the compile
/// pipeline does not need to depend on the CLI job-run machinery directly.
///
/// Implementations typically append each message as a line in the `JobRun`
/// log file, but tests use an in-memory buffer and the dry-run code path
/// uses `None` to skip log emission entirely. Messages mirror the
/// progress-to-stderr strings emitted by the `progress` flag, so the
/// two destinations stay in sync.
pub trait LogSink: Send + Sync {
    /// Append a single message as a log line. Implementations should not
    /// panic on IO errors — a failed log write must never abort a compile.
    fn append_log(&self, message: &str);
}

/// Max bytes of subprocess stderr to embed in a single log line when an
/// LLM call fails. Keeps a single failed pass from flooding the log with
/// megabytes of noisy backend output while still giving enough context to
/// diagnose the failure.
const ERR_TAIL_BYTES: usize = 2 * 1024;

/// Trim `text` to its last `ERR_TAIL_BYTES` on a char boundary, prefixing
/// with `...` when bytes were dropped. Safe for non-ASCII input because it
/// scans backwards for a valid UTF-8 boundary.
fn stderr_tail(text: &str) -> String {
    if text.len() <= ERR_TAIL_BYTES {
        return text.to_string();
    }
    let mut start = text.len() - ERR_TAIL_BYTES;
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    format!("...{}", &text[start..])
}

#[derive(Clone, Default)]
pub struct CompileOptions {
    pub force: bool,
    pub dry_run: bool,
    /// When true and no explicit [`Self::reporter`] is supplied, a verbose
    /// [`LineLogReporter`] is constructed automatically so the compile
    /// continues to emit `[run]`/`[ok]` lines to stderr (legacy behavior).
    /// When `reporter` is set, this flag is ignored — the reporter owns
    /// rendering. Should still be disabled under `--json` so structured
    /// consumers see clean stderr.
    pub progress: bool,
    /// Optional sink that receives the same progress messages as plain text,
    /// plus `[err]` lines on LLM-call failures (with a tail of subprocess
    /// stderr embedded in the error). `None` disables log forwarding.
    ///
    /// The CLI typically supplies a sink backed by the active `JobRun` log
    /// file so `state/jobs/<id>.log` captures per-pass events instead of
    /// containing only the initial "job started" line. Messages are always
    /// plain text — indicatif escape codes must never reach the log file.
    pub log_sink: Option<Arc<dyn LogSink>>,
    /// Custom progress renderer. When `Some`, the pipeline routes per-pass
    /// lifecycle events through this reporter (indicatif bars, plain lines,
    /// etc.) *instead of* the default built from `progress`. Leave `None` to
    /// preserve the legacy `progress: bool` branch.
    pub reporter: Option<Arc<dyn ProgressReporter>>,
}

impl std::fmt::Debug for CompileOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompileOptions")
            .field("force", &self.force)
            .field("dry_run", &self.dry_run)
            .field("progress", &self.progress)
            .field("log_sink", &self.log_sink.as_ref().map(|_| "<LogSink>"))
            .field(
                "reporter",
                &self.reporter.as_ref().map(|_| "<ProgressReporter>"),
            )
            .finish()
    }
}

/// Resolve the effective progress reporter for a compile invocation.
///
/// Priority:
/// 1. `options.reporter` when explicitly set (CLI dispatch path).
/// 2. `LineLogReporter::new()` when the legacy `progress: true` flag is set.
/// 3. A [`NullReporter`] otherwise — so the pipeline can unconditionally fire
///    lifecycle events without branching on `Option` at every callsite.
fn resolve_reporter(options: &CompileOptions) -> Arc<dyn ProgressReporter> {
    if let Some(reporter) = options.reporter.as_ref() {
        return Arc::clone(reporter);
    }
    if options.progress {
        return Arc::new(LineLogReporter::new());
    }
    Arc::new(NullReporter)
}

/// Append `message` to the configured log sink, if any. The sink must only
/// ever see plain text — indicatif escape sequences are kept out of the log
/// file so `kb jobs logs` stays readable.
fn log_message(options: &CompileOptions, message: &str) {
    if let Some(sink) = options.log_sink.as_ref() {
        sink.append_log(message);
    }
}

#[derive(Debug, Clone)]
pub struct CompileReport {
    pub total_sources: usize,
    pub stale_sources: usize,
    pub build_records_emitted: usize,
    pub passes: Vec<(String, PassStatus)>,
}

#[derive(Debug, Clone)]
pub enum PassStatus {
    Executed { affected: usize },
    Skipped { reason: String },
    DryRun { would_process: Vec<String> },
}

impl CompileReport {
    #[must_use]
    pub fn render(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "compile: {} source(s), {} stale",
            self.total_sources, self.stale_sources,
        ));

        for (name, status) in &self.passes {
            match status {
                PassStatus::Executed { affected } => {
                    lines.push(format!("  [ok] {name} ({affected} affected)"));
                }
                PassStatus::Skipped { reason } => {
                    lines.push(format!("  [skip] {name} — {reason}"));
                }
                PassStatus::DryRun { would_process } => {
                    lines.push(format!(
                        "  [dry-run] {name} — would process {} node(s)",
                        would_process.len()
                    ));
                    for node in would_process {
                        lines.push(format!("    - {node}"));
                    }
                }
            }
        }

        if self.build_records_emitted > 0 {
            lines.push(format!(
                "compile: {} build record(s) emitted",
                self.build_records_emitted
            ));
        }

        lines.join("\n")
    }
}

fn discover_normalized_ids(root: &Path) -> Result<Vec<String>> {
    let normalized_root = normalized_dir(root);
    if !normalized_root.exists() {
        return Ok(Vec::new());
    }

    let mut ids = Vec::new();
    for entry in std::fs::read_dir(&normalized_root)
        .with_context(|| format!("scan normalized dir {}", normalized_root.display()))?
    {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            if let Some(name) = entry.file_name().to_str() {
                ids.push(name.to_string());
            }
        }
    }
    ids.sort();
    Ok(ids)
}

/// Compute a combined hash of all compile-time prompt templates.
/// A change to any template invalidates all cached LLM outputs.
fn compile_template_hash(root: &Path) -> String {
    let templates = [
        "summarize_document.md",
        "extract_concepts.md",
        "merge_concept_candidates.md",
    ];
    let mut parts: Vec<Vec<u8>> = Vec::new();
    for name in &templates {
        let tmpl = kb_llm::Template::load(name, Some(root)).ok();
        let hash = tmpl
            .map(|t| t.template_hash.to_hex())
            .unwrap_or_default();
        parts.push(hash.into_bytes());
        parts.push(b"\0".to_vec());
    }
    let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    hash_many(&refs).to_hex()
}

/// Key used in `HashState.hashes` for the global `concept_merge` fingerprint.
/// Not a valid node id (contains ':'), so it cannot collide with per-doc keys.
const CONCEPT_MERGE_FINGERPRINT_KEY: &str = "concept_merge:global";

/// Compute a fingerprint over the concept-merge pass inputs: the sorted
/// blake3 hashes of every candidate JSON in `state/concept_candidates/` and
/// the hash of the merge prompt template. When this is unchanged across
/// compiles, re-running the merge LLM call is wasteful, so the pipeline
/// short-circuits.
///
/// Returns an empty string when there are no candidates (caller already
/// short-circuits on empty candidates separately).
fn compute_concept_merge_fingerprint(root: &Path) -> Result<String> {
    let candidates_dir = crate::concept_extraction::concept_candidates_dir(root);
    let mut candidate_hashes: Vec<String> = Vec::new();
    if candidates_dir.exists() {
        for entry in std::fs::read_dir(&candidates_dir)
            .with_context(|| format!("read {}", candidates_dir.display()))?
        {
            let entry = entry?;
            if entry.file_type()?.is_file()
                && entry.path().extension().and_then(|s| s.to_str()) == Some("json")
            {
                let bytes = std::fs::read(entry.path())
                    .with_context(|| format!("read {}", entry.path().display()))?;
                candidate_hashes.push(hash_bytes(&bytes).to_hex());
            }
        }
    }
    candidate_hashes.sort();

    let template_hash = kb_llm::Template::load("merge_concept_candidates.md", Some(root))
        .map(|t| t.template_hash.to_hex())
        .unwrap_or_default();

    let mut parts: Vec<Vec<u8>> = Vec::new();
    for h in &candidate_hashes {
        parts.push(h.clone().into_bytes());
        parts.push(b"\0".to_vec());
    }
    parts.push(b"template=".to_vec());
    parts.push(template_hash.into_bytes());
    let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    Ok(hash_many(&refs).to_hex())
}

fn build_input_nodes(
    root: &Path,
    doc_ids: &[String],
    template_hash: &str,
) -> Result<Vec<StaleNode>> {
    let mut nodes = Vec::with_capacity(doc_ids.len());
    for id in doc_ids {
        let doc = read_normalized_document(root, id)
            .with_context(|| format!("read normalized document {id}"))?;
        nodes.push(StaleNode {
            id: format!("normalized/{id}"),
            dependencies: vec![],
            content_hash: hash_bytes(doc.canonical_text.as_bytes()).to_hex(),
            prompt_template_hash: Some(template_hash.to_string()),
            model_version: None,
        });
    }
    Ok(nodes)
}

/// Run the compile pipeline without an LLM adapter.
///
/// Stale detection still runs, and the batch passes (backlinks, lexical
/// index, index pages) still execute — but per-document LLM passes (source
/// summary, concept extraction, concept merge) are skipped. Useful for unit
/// tests and for `kb compile --dry-run`-style inspection.
///
/// # Errors
///
/// Returns an error when normalized documents cannot be read or state files
/// cannot be persisted.
pub fn run_compile(root: &Path, options: &CompileOptions) -> Result<CompileReport> {
    run_compile_with_llm(root, options, None::<&dyn LlmAdapter>)
}

/// Run the full compile pipeline with incremental stale detection and an
/// optional LLM adapter for per-document passes.
///
/// When `adapter` is `Some`, stale normalized documents are passed through
/// `source_summary` → `source_page` rendering → `concept_extraction`, and a
/// global `concept_merge` pass runs afterwards. Batch passes (backlinks,
/// lexical index, index pages) always run because they are cheap and their
/// inputs may have changed outside the normalized-doc pipeline.
///
/// When `adapter` is `None`, the per-document LLM passes are skipped and the
/// function behaves like [`run_compile`].
///
/// # Errors
///
/// Returns an error when normalized documents cannot be read, passes fail,
/// or state files cannot be persisted.
#[allow(clippy::too_many_lines)]
pub fn run_compile_with_llm(
    root: &Path,
    options: &CompileOptions,
    adapter: Option<&(dyn LlmAdapter + '_)>,
) -> Result<CompileReport> {
    let doc_ids = discover_normalized_ids(root)?;
    let total_sources = doc_ids.len();

    // Stale detection on normalized documents. We keep the full StaleReport so the
    // per-document LLM passes below can iterate the stale doc IDs. We also
    // carry the previous concept_merge fingerprint forward into the new
    // state map so it survives a persist even when no per-doc pass ran.
    let previous_merge_fingerprint: Option<String>;
    let (stale_sources, stale_doc_ids, mut hash_state) = if total_sources == 0 {
        previous_merge_fingerprint = HashState::load_from_root(root)
            .ok()
            .and_then(|s| s.hashes.get(CONCEPT_MERGE_FINGERPRINT_KEY).cloned());
        (0, Vec::new(), None)
    } else {
        let raw_previous = HashState::load_from_root(root)?;
        previous_merge_fingerprint = raw_previous
            .hashes
            .get(CONCEPT_MERGE_FINGERPRINT_KEY)
            .cloned();
        // Filter out non-node keys (e.g. "concept_merge:global") before running
        // stale detection — they would otherwise be reported as "changed" on
        // every compile because detect_stale rebuilds current_state from doc
        // nodes only.
        let previous = HashState {
            hashes: raw_previous
                .hashes
                .iter()
                .filter(|(k, _)| !k.contains(':'))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        };
        let tmpl_hash = compile_template_hash(root);
        let nodes = build_input_nodes(root, &doc_ids, &tmpl_hash)?;

        let report = if options.force {
            let all: BTreeSet<String> = nodes.iter().map(|n| n.id.clone()).collect();
            let current_state = HashState {
                hashes: nodes
                    .iter()
                    .map(|n| (n.id.clone(), n.fingerprint().to_hex()))
                    .collect(),
            };
            crate::StaleReport {
                changed_nodes: all.clone(),
                stale_nodes: all,
                current_state,
            }
        } else {
            detect_stale(&previous, &nodes)
        };

        let count = report.stale_nodes.len();
        let ids: Vec<String> = report
            .stale_nodes
            .iter()
            .filter_map(|node_id| node_id.strip_prefix("normalized/").map(str::to_string))
            .collect();
        let mut current_state = report.current_state;
        // Preserve prior merge fingerprint so a save after skipping merge
        // (or after a no-op compile) doesn't drop it.
        if let Some(prev) = &previous_merge_fingerprint {
            current_state
                .hashes
                .insert(CONCEPT_MERGE_FINGERPRINT_KEY.to_string(), prev.clone());
        }
        (count, ids, Some(current_state))
    };

    if options.dry_run {
        let mut passes = Vec::new();

        // Per-document LLM passes come first in the live run, so mirror that order.
        // Dry-run is a *preview*: we always report what would happen once the
        // pipeline runs for real, regardless of whether the caller wired up an
        // adapter. The CLI intentionally skips adapter construction under
        // `--dry-run` (no credentials needed to inspect what *would* run), so
        // adapter=None here is the normal case — not a misconfiguration. Emit
        // DryRun so users don't read "no LLM adapter configured" and assume
        // their kb.toml is broken.
        if !stale_doc_ids.is_empty() {
            let stale_node_ids: Vec<String> = stale_doc_ids
                .iter()
                .map(|id| format!("normalized/{id}"))
                .collect();

            passes.push((
                "source_summary".to_string(),
                PassStatus::DryRun {
                    would_process: stale_node_ids.clone(),
                },
            ));
            passes.push((
                "concept_extraction".to_string(),
                PassStatus::DryRun {
                    would_process: stale_node_ids,
                },
            ));
            passes.push((
                "concept_merge".to_string(),
                PassStatus::DryRun {
                    would_process: vec!["wiki/concepts/* (global)".to_string()],
                },
            ));
        }

        passes.push((
            "backlinks".to_string(),
            PassStatus::DryRun {
                would_process: vec!["wiki/concepts/*".to_string()],
            },
        ));
        passes.push((
            "lexical_index".to_string(),
            PassStatus::DryRun {
                would_process: vec!["wiki/**/*.md".to_string()],
            },
        ));
        passes.push((
            "index_pages".to_string(),
            PassStatus::DryRun {
                would_process: vec![
                    "wiki/index.md".to_string(),
                    "wiki/sources/index.md".to_string(),
                    "wiki/concepts/index.md".to_string(),
                    "wiki/questions/index.md".to_string(),
                ],
            },
        ));

        return Ok(CompileReport {
            total_sources,
            stale_sources,
            build_records_emitted: 0,
            passes,
        });
    }

    let mut passes = Vec::new();
    let mut build_records_emitted: usize = 0;

    let reporter = resolve_reporter(options);
    let banner = format!("compile: {total_sources} source(s), {stale_sources} stale");
    reporter.info(&banner);
    log_message(options, &banner);

    // Per-document LLM passes (only when an adapter is configured and we have stale docs).
    if let Some(adapter) = adapter {
        if !stale_doc_ids.is_empty() {
            match run_per_document_passes(
                root,
                &stale_doc_ids,
                adapter,
                options,
                reporter.as_ref(),
            ) {
                Ok(report) => {
                    build_records_emitted += report.build_records_emitted;
                    passes.push((
                        "source_summary".to_string(),
                        PassStatus::Executed {
                            affected: report.source_pages_written,
                        },
                    ));
                    passes.push((
                        "concept_extraction".to_string(),
                        PassStatus::Executed {
                            affected: report.candidate_files_written,
                        },
                    ));
                }
                Err(err) => {
                    tracing::warn!("per-document passes failed: {err}");
                    passes.push((
                        "source_summary".to_string(),
                        PassStatus::Skipped {
                            reason: format!("error: {err}"),
                        },
                    ));
                    passes.push((
                        "concept_extraction".to_string(),
                        PassStatus::Skipped {
                            reason: "upstream pass failed".to_string(),
                        },
                    ));
                }
            }
        }

        // Global concept merge (reads all candidate JSONs so prior-run candidates still count).
        // Short-circuit when candidates + template haven't changed since the
        // last successful merge — the LLM call is 80s+ for real corpora and
        // would produce identical output. bn-1op.
        let new_merge_fingerprint = compute_concept_merge_fingerprint(root).ok();
        let candidates_dir = crate::concept_extraction::concept_candidates_dir(root);
        let has_candidates = candidates_dir.exists()
            && std::fs::read_dir(&candidates_dir)
                .map(|it| {
                    it.flatten().any(|e| {
                        e.path().extension().and_then(|s| s.to_str()) == Some("json")
                    })
                })
                .unwrap_or(false);
        let fingerprint_matches = !options.force
            && has_candidates
            && match (&new_merge_fingerprint, &previous_merge_fingerprint) {
                (Some(new), Some(prev)) => new == prev,
                _ => false,
            };

        if fingerprint_matches {
            let skip_line = "  [skip] concept_merge — no candidate changes";
            reporter.info(skip_line);
            log_message(options, skip_line);
            passes.push((
                "concept_merge".to_string(),
                PassStatus::Skipped {
                    reason: "no candidate changes".to_string(),
                },
            ));
        } else {
            match run_concept_merge_from_state(root, adapter, options, reporter.as_ref()) {
                Ok(report) => {
                    build_records_emitted += report.build_records_emitted;
                    // Persist the fingerprint so the next compile can skip
                    // the merge when nothing changed. Only update when the
                    // merge actually produced something (pages or reviews),
                    // otherwise fall back to the prior value so a degenerate
                    // zero-candidate run doesn't wipe the record.
                    if report.pages_written + report.reviews_written > 0
                        && let (Some(fp), Some(state)) =
                            (&new_merge_fingerprint, hash_state.as_mut())
                    {
                        state
                            .hashes
                            .insert(CONCEPT_MERGE_FINGERPRINT_KEY.to_string(), fp.clone());
                    }
                    passes.push((
                        "concept_merge".to_string(),
                        PassStatus::Executed {
                            affected: report.pages_written + report.reviews_written,
                        },
                    ));
                }
                Err(err) => {
                    tracing::warn!("concept merge pass failed: {err}");
                    passes.push((
                        "concept_merge".to_string(),
                        PassStatus::Skipped {
                            reason: format!("error: {err}"),
                        },
                    ));
                }
            }
        }
    } else if !stale_doc_ids.is_empty() {
        // No adapter: record what we skipped so users see why summaries didn't appear.
        // Keep parity with the dry-run block above so `dry_run_matches_live_run_stale_count`
        // sees the same set of pass names regardless of mode.
        passes.push((
            "source_summary".to_string(),
            PassStatus::Skipped {
                reason: "no LLM adapter configured".to_string(),
            },
        ));
        passes.push((
            "concept_extraction".to_string(),
            PassStatus::Skipped {
                reason: "no LLM adapter configured".to_string(),
            },
        ));
        passes.push((
            "concept_merge".to_string(),
            PassStatus::Skipped {
                reason: "no LLM adapter configured".to_string(),
            },
        ));
    }

    // Pass: backlinks (no LLM — scans wiki links and updates concept pages)
    match crate::backlinks::run_backlinks_pass(root) {
        Ok(artifacts) => {
            let mut updated = 0;
            for artifact in &artifacts {
                if artifact.needs_update() {
                    kb_core::fs::atomic_write(&artifact.path, artifact.updated_markdown.as_bytes())
                        .with_context(|| {
                            format!("write backlinks for {}", artifact.path.display())
                        })?;
                    updated += 1;
                }
            }
            passes.push((
                "backlinks".to_string(),
                PassStatus::Executed { affected: updated },
            ));
        }
        Err(err) => {
            tracing::warn!("backlinks pass failed: {err}");
            passes.push((
                "backlinks".to_string(),
                PassStatus::Skipped {
                    reason: format!("error: {err}"),
                },
            ));
        }
    }

    // Pass: lexical index
    let index = kb_query::build_lexical_index(root)?;
    let entry_count = index.entries.len();
    index.save(root)?;
    passes.push((
        "lexical_index".to_string(),
        PassStatus::Executed {
            affected: entry_count,
        },
    ));

    // Pass: index pages
    let index_artifacts = crate::index_page::generate_indexes(root)?;
    let index_count = index_artifacts.len();
    crate::index_page::persist_index_artifacts(&index_artifacts)?;
    passes.push((
        "index_pages".to_string(),
        PassStatus::Executed {
            affected: index_count,
        },
    ));

    // Persist updated hash state (only when we computed one)
    if let Some(state) = &hash_state {
        state.save_to_root(root)?;
    }

    // Persist the dependency graph so `kb status` and `kb inspect` can see
    // the compiled corpus. bn-3w0. Individual passes emit `BuildRecord`s but
    // nothing previously assembled them into the persisted graph used by
    // status/inspect, so their counters always read zero after a compile.
    match build_graph_from_state(root) {
        Ok(graph) => {
            if let Err(err) = graph.persist_to(root) {
                tracing::warn!("failed to persist compile graph: {err}");
            }
        }
        Err(err) => {
            tracing::warn!("failed to assemble compile graph: {err}");
        }
    }

    Ok(CompileReport {
        total_sources,
        stale_sources,
        build_records_emitted,
        passes,
    })
}

/// Assemble a dependency graph describing the current compiled state of the KB.
///
/// Nodes use the `source-document-*`, `wiki-page-*`, and `concept-*` prefixes
/// that `gather_status` and `run_inspect` rely on. Edges connect each source
/// document to its rendered `wiki/sources/*.md` page, and each source page to
/// every concept that shares a build record with it.
///
/// The graph is rebuilt from the filesystem + `state/build_records/` on every
/// compile so it stays consistent even when passes are skipped or fail; a
/// missing or malformed build record simply omits its edges rather than
/// aborting the compile.
#[allow(clippy::too_many_lines)]
fn build_graph_from_state(root: &Path) -> Result<Graph> {
    let mut graph = Graph::default();

    // Source-document nodes: one per `.kb/normalized/<src-id>/` directory.
    let normalized_root = normalized_dir(root);
    let mut doc_ids: Vec<String> = Vec::new();
    if normalized_root.exists() {
        for entry in std::fs::read_dir(&normalized_root)
            .with_context(|| format!("scan {}", normalized_root.display()))?
        {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    doc_ids.push(name.to_string());
                    graph.add_node(format!("source-document-{name}"));
                }
            }
        }
    }

    // Wiki-page nodes: one per `wiki/sources/*.md` (skip `index.md`). The slug
    // is the filename stem; `source_page_path_for_id` derives the same slug
    // for a given doc id, so we match them up to draw edges.
    let sources_dir = root.join("wiki/sources");
    let mut wiki_page_slugs: BTreeSet<String> = BTreeSet::new();
    if sources_dir.exists() {
        for entry in std::fs::read_dir(&sources_dir)
            .with_context(|| format!("scan {}", sources_dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if !entry.file_type()?.is_file() {
                continue;
            }
            if path.extension().and_then(|s| s.to_str()) != Some("md") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if stem == "index" {
                continue;
            }
            wiki_page_slugs.insert(stem.to_string());
            graph.add_node(format!("wiki-page-{stem}"));
        }
    }

    // Concept nodes: one per `wiki/concepts/*.md` (skip `index.md`).
    let concepts_dir = root.join("wiki/concepts");
    let mut concept_slugs: BTreeSet<String> = BTreeSet::new();
    if concepts_dir.exists() {
        for entry in std::fs::read_dir(&concepts_dir)
            .with_context(|| format!("scan {}", concepts_dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if !entry.file_type()?.is_file() {
                continue;
            }
            if path.extension().and_then(|s| s.to_str()) != Some("md") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if stem == "index" {
                continue;
            }
            concept_slugs.insert(stem.to_string());
            graph.add_node(format!("concept-{stem}"));
        }
    }

    // Edges: source-document -> wiki-page. Source pages are now written at
    // `wiki/sources/<src-id>-<title-slug>.md` (bn-nlw9), so match by id
    // prefix: any on-disk wiki-page slug that is either the src id itself
    // or starts with `<src-id>-` is owned by that doc.
    for doc_id in &doc_ids {
        let page_stem_from_id = {
            let p = crate::source_page::source_page_path_for_id(doc_id);
            p.file_stem().and_then(|s| s.to_str()).map(str::to_string)
        };
        let Some(id_stem) = page_stem_from_id else {
            continue;
        };
        let id_prefix = format!("{id_stem}-");
        for page_slug in &wiki_page_slugs {
            if page_slug == &id_stem || page_slug.starts_with(&id_prefix) {
                graph.add_edge(
                    format!("source-document-{doc_id}"),
                    format!("wiki-page-{page_slug}"),
                );
            }
        }
    }

    // Edges: wiki-page -> concept. Walk every build record and, for each
    // record whose output paths fall under `wiki/concepts/`, connect each
    // input path under `wiki/sources/` to the concept. Records for other
    // passes (backlinks, lexical index, etc.) are ignored because they do
    // not reshape the source -> concept topology.
    let doc_id_set: BTreeSet<String> = doc_ids.iter().cloned().collect();
    let build_records_dir = kb_core::build_records_dir(root);
    if build_records_dir.exists() {
        for entry in std::fs::read_dir(&build_records_dir)
            .with_context(|| format!("scan {}", build_records_dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let raw = match std::fs::read_to_string(&path) {
                Ok(text) => text,
                Err(err) => {
                    tracing::warn!("skip build record {}: {err}", path.display());
                    continue;
                }
            };
            let record: kb_core::BuildRecord = match serde_json::from_str(&raw) {
                Ok(record) => record,
                Err(err) => {
                    tracing::warn!("skip build record {}: {err}", path.display());
                    continue;
                }
            };

            // Inputs/outputs can be a wiki source page slug, a source
            // document short id (source_summary uses the latter), a concept
            // slug, or a filesystem path; normalize all known shapes to the
            // prefixed graph node format used above.
            let mut raw_outputs: Vec<String> = record.output_ids.clone();
            for path in &record.metadata.output_paths {
                if let Some(s) = path.to_str() {
                    raw_outputs.push(s.to_string());
                }
            }
            let input_nodes: Vec<String> = record
                .input_ids
                .iter()
                .filter_map(|id| {
                    graph_node_for_id(id, &doc_id_set, &wiki_page_slugs, &concept_slugs)
                })
                .collect();
            let output_nodes: Vec<String> = raw_outputs
                .iter()
                .filter_map(|id| {
                    graph_node_for_id(id, &doc_id_set, &wiki_page_slugs, &concept_slugs)
                })
                .collect();

            for input in &input_nodes {
                for output in &output_nodes {
                    if input != output {
                        graph.add_edge(input.clone(), output.clone());
                    }
                }
            }
        }
    }

    // Normalize explicitly: `add_edge` appends without deduping, so repeated
    // build records between the same pair of nodes would otherwise leave
    // duplicates in the persisted JSON. `validate()` returns a normalized
    // clone, which is what callers expect to round-trip via `persist_to`.
    graph.validate()
}

/// Map a build-record identifier (node id, wiki path, or source id) onto the
/// prefixed graph node format used by status/inspect. Returns `None` when the
/// identifier does not correspond to one of the prefixed node kinds.
fn graph_node_for_id(
    id: &str,
    doc_ids: &BTreeSet<String>,
    wiki_page_slugs: &BTreeSet<String>,
    concept_slugs: &BTreeSet<String>,
) -> Option<String> {
    if id.starts_with("source-document-")
        || id.starts_with("wiki-page-")
        || id.starts_with("concept-")
    {
        return Some(id.to_string());
    }

    // Wiki source/concept markdown paths emitted by build records.
    if let Some(rest) = id.strip_prefix("wiki/sources/") {
        let stem = std::path::Path::new(rest)
            .file_stem()
            .and_then(|s| s.to_str())?;
        if stem == "index" {
            return None;
        }
        return Some(format!("wiki-page-{stem}"));
    }
    if let Some(rest) = id.strip_prefix("wiki/concepts/") {
        let stem = std::path::Path::new(rest)
            .file_stem()
            .and_then(|s| s.to_str())?;
        if stem == "index" {
            return None;
        }
        return Some(format!("concept-{stem}"));
    }

    // Slug-only match against known wiki pages (for passes that reference
    // the page by frontmatter id like `wiki-source-<slug>`).
    if let Some(rest) = id.strip_prefix("wiki-source-") {
        if wiki_page_slugs.contains(rest) {
            return Some(format!("wiki-page-{rest}"));
        }
    }

    // Source document short ids (e.g. `src-abc123`) appear in source_summary
    // build records' input_ids directly.
    if doc_ids.contains(id) {
        return Some(format!("source-document-{id}"));
    }

    // Fallback: exact slug match against known wiki pages or concepts.
    if wiki_page_slugs.contains(id) {
        return Some(format!("wiki-page-{id}"));
    }
    if concept_slugs.contains(id) {
        return Some(format!("concept-{id}"));
    }

    None
}

struct PerDocumentReport {
    source_pages_written: usize,
    candidate_files_written: usize,
    build_records_emitted: usize,
}

/// For each stale normalized document, run source summary + source page render +
/// concept extraction. Writes are atomic; a failure for one document still lets the
/// others finish (so a single bad source does not poison the whole compile).
#[allow(clippy::too_many_lines)]
fn run_per_document_passes(
    root: &Path,
    stale_doc_ids: &[String],
    adapter: &(dyn LlmAdapter + '_),
    options: &CompileOptions,
    reporter: &dyn ProgressReporter,
) -> Result<PerDocumentReport> {
    let mut source_pages_written = 0;
    let mut candidate_files_written = 0;
    let mut build_records_emitted = 0;

    let total = stale_doc_ids.len();
    let summary_overall_start = Instant::now();
    reporter.pass_start("source_summary", total);
    // concept_extraction has the same total and runs interleaved with
    // source_summary below; start its bar up-front so both render together.
    reporter.pass_start("concept_extraction", total);
    let concept_overall_start = Instant::now();
    let mut source_summary_affected = 0usize;
    let mut concept_extraction_affected = 0usize;

    for doc_id in stale_doc_ids {
        let doc = read_normalized_document(root, doc_id)
            .with_context(|| format!("read normalized document {doc_id}"))?;
        // bn-6puc: before falling through to the src-id, prefer the ingested
        // file's filename stem as a title when the body has no H1 and no
        // frontmatter title. URL-ingested sources have no filesystem stem, so
        // the stem helper returns `None` and we fall through to `doc_id` as
        // before.
        let stem_fallback = stable_location_file_stem(root, doc_id);
        let title = derive_title(&doc.canonical_text, stem_fallback.as_deref(), doc_id);
        // New page path includes the title slug (bn-nlw9). Read any existing
        // page (id-only OR prior-slug form) so managed regions / frontmatter
        // from the previous compile still propagate into the new file.
        let page_path = crate::source_page::source_page_path_for(doc_id, &title);
        let page_abs_path = root.join(&page_path);
        let existing_page_abs = crate::source_page::resolve_source_page_path(root, doc_id)
            .unwrap_or_else(|| page_abs_path.clone());
        let existing_markdown = std::fs::read_to_string(&existing_page_abs).ok();
        let existing_body = existing_markdown
            .as_deref()
            .and_then(split_body_from_frontmatter);
        let page_id = format!("wiki-source-{}", slug_from_title(doc_id));

        reporter.pass_item_start("source_summary", doc_id);
        log_message(options, &format!("  [run] source_summary: {doc_id}..."));
        let summary_started = Instant::now();
        let summary_artifact = match crate::source_summary::run_source_summary_pass(
            adapter,
            &doc,
            &title,
            120,
            &page_id,
            &page_path,
            existing_body.as_deref(),
        ) {
            Ok(artifact) => artifact,
            Err(err) => {
                tracing::warn!("source_summary failed for {doc_id}: {err}");
                // Adapter errors already embed subprocess stderr (opencode/claude
                // both format `stderr` into their LlmAdapterError message) — trim
                // to ERR_TAIL_BYTES so a single failed pass can't flood the log.
                let err_str = stderr_tail(&err.to_string());
                let err_line = format!(
                    "  [err] source_summary: {doc_id} ({:.1}s) — {err_str}",
                    summary_started.elapsed().as_secs_f64(),
                );
                reporter.error(&err_line);
                log_message(options, &err_line);
                // Advance the progress bar past this item even on error so the
                // counter still reaches total once every doc has been tried.
                reporter.pass_item_done(
                    "source_summary",
                    doc_id,
                    summary_started.elapsed(),
                );
                continue;
            }
        };
        let summary_elapsed = summary_started.elapsed();
        reporter.pass_item_done("source_summary", doc_id, summary_elapsed);
        log_message(
            options,
            &format!(
                "  [ok]  source_summary: {doc_id} ({:.1}s)",
                summary_elapsed.as_secs_f64(),
            ),
        );
        source_summary_affected += 1;

        save_build_record(root, &summary_artifact.build_record)
            .with_context(|| format!("save build record for source_summary {doc_id}"))?;
        build_records_emitted += 1;

        let citations_display: Vec<String> = summary_artifact
            .citations
            .iter()
            .map(|c| {
                let anchor = c
                    .heading_anchor
                    .as_deref()
                    .map(|h| format!("#{h}"))
                    .unwrap_or_default();
                format!("{}{}", c.source_revision_id, anchor)
            })
            .collect();

        let generated_at = summary_artifact.build_record.metadata.created_at_millis;
        // Re-anchor any `![alt](assets/foo.png)` refs the LLM preserved in
        // the summary. The normalized source stores them relative to
        // `normalized/<src>/`; the wiki source page is at
        // `wiki/sources/<src>.md` (two levels deep from root) so it needs
        // `../../normalized/<src>/assets/...` to resolve the same file.
        let rewritten_summary = crate::source_page::rewrite_summary_image_refs(
            &summary_artifact.summary,
            doc_id,
        );
        let page_input = crate::source_page::SourcePageInput {
            page_id: &page_id,
            title: &title,
            source_document_id: doc_id,
            source_revision_id: &doc.source_revision_id,
            generated_at,
            build_record_id: &summary_artifact.build_record.metadata.id,
            summary: &rewritten_summary,
            key_topics: &summary_artifact.key_headings,
            citations: &citations_display,
        };

        let page_artifact = crate::source_page::render_source_page(
            &page_input,
            existing_markdown.as_deref(),
        )?;
        kb_core::fs::atomic_write(&page_abs_path, page_artifact.markdown.as_bytes())
            .with_context(|| format!("write source page {}", page_abs_path.display()))?;
        // Clean up any stale id-only or id-slug source pages left over from a
        // previous compile (title changed → slug changed). bn-nlw9 keeps the
        // new file on disk; every other match goes.
        match crate::source_page::clean_stale_source_pages(root, doc_id, &page_path) {
            Ok(removed) => {
                for p in &removed {
                    tracing::debug!("cleaned stale source page: {}", p.display());
                }
            }
            Err(err) => {
                tracing::warn!("could not clean stale source pages for {doc_id}: {err}");
            }
        }
        source_pages_written += 1;

        let candidates_path = crate::concept_extraction::concept_candidates_path(root, doc_id);
        reporter.pass_item_start("concept_extraction", doc_id);
        log_message(options, &format!("  [run] concept_extraction: {doc_id}..."));
        let extract_started = Instant::now();
        match crate::concept_extraction::run_concept_extraction_pass(
            adapter,
            &doc,
            &title,
            Some(&summary_artifact.summary),
            Some(20),
            &candidates_path,
        ) {
            Ok(artifact) => {
                crate::concept_extraction::persist_concept_candidates(
                    &artifact.output_path,
                    &artifact.output_json,
                )?;
                save_build_record(root, &artifact.build_record).with_context(|| {
                    format!("save build record for concept_extraction {doc_id}")
                })?;
                candidate_files_written += 1;
                build_records_emitted += 1;
                concept_extraction_affected += 1;
                let elapsed = extract_started.elapsed();
                reporter.pass_item_done("concept_extraction", doc_id, elapsed);
                log_message(
                    options,
                    &format!(
                        "  [ok]  concept_extraction: {doc_id} ({:.1}s)",
                        elapsed.as_secs_f64(),
                    ),
                );
            }
            Err(err) => {
                tracing::warn!("concept_extraction failed for {doc_id}: {err}");
                let err_str = stderr_tail(&err.to_string());
                let err_line = format!(
                    "  [err] concept_extraction: {doc_id} ({:.1}s) — {err_str}",
                    extract_started.elapsed().as_secs_f64(),
                );
                reporter.error(&err_line);
                log_message(options, &err_line);
                reporter.pass_item_done(
                    "concept_extraction",
                    doc_id,
                    extract_started.elapsed(),
                );
            }
        }
    }

    reporter.pass_done(
        "source_summary",
        source_summary_affected,
        summary_overall_start.elapsed(),
    );
    reporter.pass_done(
        "concept_extraction",
        concept_extraction_affected,
        concept_overall_start.elapsed(),
    );

    Ok(PerDocumentReport {
        source_pages_written,
        candidate_files_written,
        build_records_emitted,
    })
}

struct MergeRunReport {
    pages_written: usize,
    reviews_written: usize,
    build_records_emitted: usize,
}

/// Load every candidate JSON from `state/concept_candidates/` and run the global
/// merge pass. Candidates from non-stale sources still participate so renames and
/// canonicalization can propagate across the whole KB on every run.
#[allow(clippy::too_many_lines)]
fn run_concept_merge_from_state(
    root: &Path,
    adapter: &(dyn LlmAdapter + '_),
    options: &CompileOptions,
    reporter: &dyn ProgressReporter,
) -> Result<MergeRunReport> {
    let candidates_dir = crate::concept_extraction::concept_candidates_dir(root);
    let mut all_candidates: Vec<ConceptCandidate> = Vec::new();
    // Side-map of candidate.name -> sorted list of originating source document IDs
    // (the filename stem of each `state/concept_candidates/<src-id>.json`). Same
    // candidate name can show up under multiple sources, so we accumulate a set
    // rather than overwriting. Passed into the merge pass so concept pages can
    // emit `source_document_ids:` in frontmatter for the missing-citations lint.
    let mut candidate_origins: std::collections::HashMap<String, std::collections::BTreeSet<String>> =
        std::collections::HashMap::new();
    if candidates_dir.exists() {
        for entry in std::fs::read_dir(&candidates_dir)
            .with_context(|| format!("read {}", candidates_dir.display()))?
        {
            let entry = entry?;
            if entry.file_type()?.is_file()
                && entry.path().extension().and_then(|s| s.to_str()) == Some("json")
            {
                let path = entry.path();
                let src_id = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(std::string::ToString::to_string);
                let text = std::fs::read_to_string(&path)
                    .with_context(|| format!("read {}", path.display()))?;
                let mut batch: Vec<ConceptCandidate> = serde_json::from_str(&text)
                    .with_context(|| format!("parse {}", path.display()))?;
                if let Some(src_id) = src_id {
                    for c in &batch {
                        candidate_origins
                            .entry(c.name.clone())
                            .or_default()
                            .insert(src_id.clone());
                    }
                }
                all_candidates.append(&mut batch);
            }
        }
    }
    let candidate_origins: std::collections::HashMap<String, Vec<String>> = candidate_origins
        .into_iter()
        .map(|(k, v)| (k, v.into_iter().collect()))
        .collect();

    if all_candidates.is_empty() {
        return Ok(MergeRunReport {
            pages_written: 0,
            reviews_written: 0,
            build_records_emitted: 0,
        });
    }

    // concept_merge is a single opaque LLM call — render as a spinner by
    // declaring total=1. Reporter implementations (indicatif) detect this and
    // draw a spinner instead of a bar.
    reporter.pass_start("concept_merge", 1);
    let candidate_label = format!("merging {} candidate(s)", all_candidates.len());
    reporter.pass_item_start("concept_merge", &candidate_label);
    log_message(
        options,
        &format!(
            "  [run] concept_merge: {} candidate(s)...",
            all_candidates.len(),
        ),
    );
    let merge_started = Instant::now();
    let artifact = match crate::concept_merge::run_concept_merge_pass_with_origins(
        adapter,
        all_candidates,
        root,
        &candidate_origins,
    ) {
        Ok(a) => a,
        Err(err) => {
            let err_str = stderr_tail(&err.to_string());
            let err_line = format!(
                "  [err] concept_merge ({:.1}s) — {err_str}",
                merge_started.elapsed().as_secs_f64(),
            );
            reporter.error(&err_line);
            log_message(options, &err_line);
            reporter.pass_item_done(
                "concept_merge",
                &candidate_label,
                merge_started.elapsed(),
            );
            reporter.pass_done("concept_merge", 0, merge_started.elapsed());
            return Err(anyhow::anyhow!("{err}"));
        }
    };
    let merge_elapsed = merge_started.elapsed();
    reporter.pass_item_done("concept_merge", &candidate_label, merge_elapsed);
    reporter.pass_done("concept_merge", 1, merge_elapsed);
    log_message(
        options,
        &format!(
            "  [ok]  concept_merge ({:.1}s)",
            merge_elapsed.as_secs_f64(),
        ),
    );
    for page in &artifact.concept_pages {
        crate::concept_merge::persist_concept_page(page)?;
    }
    for review in &artifact.review_records {
        crate::concept_merge::persist_merge_review(root, review)?;
    }
    save_build_record(root, &artifact.build_record)
        .context("save build record for concept_merge")?;

    Ok(MergeRunReport {
        pages_written: artifact.concept_pages.len(),
        reviews_written: artifact.review_records.len(),
        build_records_emitted: 1,
    })
}

/// Extract a human-readable title with the following fallback order:
///
/// 1. The first `title:` line in a leading YAML frontmatter block (if any).
/// 2. The first `# ` heading in the canonical text.
/// 3. `stem_fallback` — the ingested file's filename stem (bn-6puc). URL
///    ingests pass `None` here so step 4 applies instead.
/// 4. `fallback_id` — the src id (last resort).
///
/// Note: the ingest pipeline strips YAML frontmatter from `canonical_text`
/// before persisting, so step 1 is rarely hit in practice — it's there as
/// defense against future paths that might preserve frontmatter and to make
/// the fallback order explicit.
fn derive_title(canonical_text: &str, stem_fallback: Option<&str>, fallback_id: &str) -> String {
    if let Some(title) = frontmatter_title(canonical_text) {
        return title;
    }
    for line in canonical_text.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("# ") {
            let candidate = rest.trim();
            if !candidate.is_empty() {
                return candidate.to_string();
            }
        }
    }
    if let Some(stem) = stem_fallback {
        let trimmed = stem.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    fallback_id.to_string()
}

/// Parse a leading YAML frontmatter block out of `text` and return the
/// `title:` field, trimmed. Returns `None` when the text has no frontmatter,
/// no `title:` key, or the title is empty after trimming.
fn frontmatter_title(text: &str) -> Option<String> {
    let rest = text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"))?;
    let mut yaml = String::new();
    for line in rest.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed == "---" {
            break;
        }
        yaml.push_str(line);
    }
    let parsed: serde_yaml::Value = serde_yaml::from_str(&yaml).ok()?;
    let title = parsed.get("title")?.as_str()?.trim().to_string();
    if title.is_empty() { None } else { Some(title) }
}

/// Read the filename stem from `raw/inbox/<src_id>/source_document.json`'s
/// `stable_location` field. Returns `None` when:
///
/// - the `source_document.json` is missing or malformed,
/// - the `stable_location` is URL-shaped (`scheme://...`) — URL sources don't
///   carry a meaningful filesystem stem, and
/// - the stem is empty.
///
/// The stem is returned verbatim (preserving hyphens, case, and extension-
/// less basename). Slug normalization happens later in the write path.
fn stable_location_file_stem(root: &Path, src_id: &str) -> Option<String> {
    let path = root
        .join("raw/inbox")
        .join(src_id)
        .join("source_document.json");
    let bytes = std::fs::read(&path).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    let stable_location = value.get("stable_location")?.as_str()?;
    kb_core::file_stem_from_stable_location(stable_location)
}

/// Split body text out of a frontmatter-prefixed markdown document, returning None if
/// there is no frontmatter block. Used to hand the body (without YAML) to passes that
/// only operate on the managed-region body.
fn split_body_from_frontmatter(markdown: &str) -> Option<String> {
    let rest = markdown.strip_prefix("---\n").or_else(|| markdown.strip_prefix("---\r\n"))?;
    // Find the next `---` line.
    let mut offset = 0;
    for line in rest.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed == "---" {
            return Some(rest[offset + line.len()..].to_string());
        }
        offset += line.len();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use kb_core::{EntityMetadata, NormalizedDocument, Status, write_normalized_document};
    use tempfile::tempdir;

    fn test_doc(id: &str, content: &str) -> NormalizedDocument {
        NormalizedDocument {
            metadata: EntityMetadata {
                id: id.to_string(),
                created_at_millis: 1,
                updated_at_millis: 1,
                source_hashes: vec![],
                model_version: None,
                tool_version: Some("kb/test".to_string()),
                prompt_template_hash: None,
                dependencies: vec![],
                output_paths: vec![],
                status: Status::Fresh,
            },
            source_revision_id: format!("rev-{id}"),
            canonical_text: content.to_string(),
            normalized_assets: vec![],
            heading_ids: vec![],
        }
    }

    fn setup_kb(root: &Path) {
        std::fs::create_dir_all(state_dir(root)).expect("create state dir");
        std::fs::create_dir_all(root.join("wiki/sources")).expect("create wiki sources");
        std::fs::create_dir_all(root.join("wiki/concepts")).expect("create wiki concepts");
    }

    #[test]
    fn compile_options_default_suppresses_progress() {
        let opts = CompileOptions::default();
        assert!(!opts.progress, "progress must default to false (quiet)");
        assert!(!opts.force);
        assert!(!opts.dry_run);
    }

    #[test]
    fn empty_kb_with_progress_disabled_succeeds_silently() {
        // With progress=false, running compile on an empty KB must not panic and
        // must not emit progress lines to stderr. We cannot intercept stderr from
        // within a library test, but we exercise the code path to guarantee that
        // every eprintln call is gated on `options.progress` — any ungated call
        // would still fire and a reviewer can observe it in test output.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        let report = run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
                progress: false,
                log_sink: None,
                reporter: None,
            },
        )
        .expect("compile");

        assert_eq!(report.total_sources, 0);
        assert_eq!(report.stale_sources, 0);
    }

    #[test]
    fn compile_with_progress_enabled_does_not_panic() {
        // Smoke test: progress=true must run cleanly end-to-end without an adapter.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        write_normalized_document(root, &test_doc("doc-1", "# Hello\nWorld")).expect("write");

        let report = run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
                progress: true,
                log_sink: None,
                reporter: None,
            },
        )
        .expect("compile with progress");

        assert_eq!(report.total_sources, 1);
        assert_eq!(report.stale_sources, 1);
    }

    #[test]
    fn compile_on_empty_kb_runs_batch_passes() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        let report = run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
                progress: false,
                log_sink: None,
                reporter: None,
            },
        )
        .expect("compile");

        assert_eq!(report.total_sources, 0);
        assert_eq!(report.stale_sources, 0);
        assert_eq!(report.build_records_emitted, 0);
        assert!(
            report.passes.len() >= 2,
            "batch passes (lexical_index, index_pages) should always run"
        );
    }

    #[test]
    fn first_compile_marks_all_sources_stale() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        write_normalized_document(root, &test_doc("doc-1", "# Hello\nWorld")).expect("write doc");
        write_normalized_document(root, &test_doc("doc-2", "# Second\nDocument"))
            .expect("write doc");

        let report = run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
                progress: false,
                log_sink: None,
                reporter: None,
            },
        )
        .expect("compile");

        assert_eq!(report.total_sources, 2);
        assert_eq!(report.stale_sources, 2);
        assert!(!report.passes.is_empty());
    }

    #[test]
    fn second_compile_with_no_changes_has_zero_stale() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        write_normalized_document(root, &test_doc("doc-1", "# Hello\nWorld")).expect("write doc");

        run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
                progress: false,
                log_sink: None,
                reporter: None,
            },
        )
        .expect("first compile");

        let report = run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
                progress: false,
                log_sink: None,
                reporter: None,
            },
        )
        .expect("second compile");

        assert_eq!(report.total_sources, 1);
        assert_eq!(report.stale_sources, 0);
        assert_eq!(report.build_records_emitted, 0);
    }

    #[test]
    fn changed_doc_triggers_rebuild() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        write_normalized_document(root, &test_doc("doc-1", "# Original")).expect("write doc");

        run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
                progress: false,
                log_sink: None,
                reporter: None,
            },
        )
        .expect("first compile");

        write_normalized_document(root, &test_doc("doc-1", "# Updated content"))
            .expect("update doc");

        let report = run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
                progress: false,
                log_sink: None,
                reporter: None,
            },
        )
        .expect("second compile");

        assert_eq!(report.total_sources, 1);
        assert_eq!(report.stale_sources, 1);
    }

    #[test]
    fn force_flag_rebuilds_everything() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        write_normalized_document(root, &test_doc("doc-1", "# Hello")).expect("write doc");

        run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
                progress: false,
                log_sink: None,
                reporter: None,
            },
        )
        .expect("first compile");

        let report = run_compile(
            root,
            &CompileOptions {
                force: true,
                dry_run: false,
                progress: false,
                log_sink: None,
                reporter: None,
            },
        )
        .expect("force compile");

        assert_eq!(report.stale_sources, 1);
        assert!(!report.passes.is_empty());
    }

    #[test]
    fn dry_run_reports_without_executing() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        write_normalized_document(root, &test_doc("doc-1", "# Hello")).expect("write doc");

        let report = run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: true,
                progress: false,
                log_sink: None,
                reporter: None,
            },
        )
        .expect("dry run");

        assert_eq!(report.total_sources, 1);
        assert_eq!(report.stale_sources, 1);
        assert_eq!(report.build_records_emitted, 0);

        // Dry-run is a preview, so every pass — per-doc and batch — must be
        // reported as DryRun. The CLI deliberately skips adapter construction
        // under --dry-run, so the absence of an adapter is normal here and
        // must NOT be surfaced as "no LLM adapter configured" (bn-1xf).
        let names: Vec<&str> = report
            .passes
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        for expected in [
            "source_summary",
            "concept_extraction",
            "concept_merge",
            "backlinks",
            "lexical_index",
            "index_pages",
        ] {
            assert!(
                names.contains(&expected),
                "dry-run should list `{expected}` pass; got {names:?}"
            );
        }

        for (name, status) in &report.passes {
            assert!(
                matches!(status, PassStatus::DryRun { .. }),
                "dry-run pass `{name}` should be DryRun; got {status:?}"
            );
        }

        assert!(
            HashState::load_from_root(root)
                .expect("load")
                .hashes
                .is_empty(),
            "hash state should not be persisted during dry run"
        );
    }

    #[test]
    fn dry_run_with_adapter_previews_per_doc_passes() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        write_normalized_document(root, &test_doc("doc-a", "# A")).expect("write doc");
        write_normalized_document(root, &test_doc("doc-b", "# B")).expect("write doc");

        let adapter = RecordingAdapter::default();
        let report = run_compile_with_llm(
            root,
            &CompileOptions {
                force: false,
                dry_run: true,
                progress: false,
                log_sink: None,
                reporter: None,
            },
            Some(&adapter),
        )
        .expect("dry run with adapter");

        // No LLM calls should have been made during a dry run.
        assert_eq!(
            adapter
                .summarize_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0,
            "dry run must not invoke summarize_document"
        );
        assert_eq!(
            adapter
                .extract_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0,
            "dry run must not invoke extract_concepts"
        );
        assert_eq!(
            adapter
                .merge_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0,
            "dry run must not invoke merge_concept_candidates"
        );

        // Per-doc passes should list the stale doc IDs.
        let summary_pass = report
            .passes
            .iter()
            .find(|(n, _)| n == "source_summary")
            .expect("source_summary entry");
        match &summary_pass.1 {
            PassStatus::DryRun { would_process } => {
                assert_eq!(would_process.len(), 2, "one entry per stale doc");
                assert!(would_process.iter().any(|s| s == "normalized/doc-a"));
                assert!(would_process.iter().any(|s| s == "normalized/doc-b"));
            }
            other => panic!("expected DryRun, got {other:?}"),
        }

        let extract_pass = report
            .passes
            .iter()
            .find(|(n, _)| n == "concept_extraction")
            .expect("concept_extraction entry");
        assert!(matches!(&extract_pass.1, PassStatus::DryRun { would_process } if would_process.len() == 2));

        let merge_pass = report
            .passes
            .iter()
            .find(|(n, _)| n == "concept_merge")
            .expect("concept_merge entry");
        assert!(matches!(&merge_pass.1, PassStatus::DryRun { would_process } if would_process.len() == 1));
    }

    #[test]
    fn dry_run_reports_per_doc_passes_regardless_of_adapter() {
        // bn-1xf regression: `kb compile --dry-run` deliberately skips adapter
        // construction, so run_compile_with_llm is called with adapter=None
        // under dry_run=true. Previously that path emitted
        // `Skipped { reason: "no LLM adapter configured" }`, which misled
        // users into thinking their kb.toml was broken. Dry-run is a preview;
        // per-doc passes must be reported as DryRun with the stale doc IDs.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        write_normalized_document(root, &test_doc("doc-a", "# A")).expect("write doc");

        let report = run_compile_with_llm(
            root,
            &CompileOptions {
                force: false,
                dry_run: true,
                progress: false,
                log_sink: None,
                reporter: None,
            },
            None::<&dyn LlmAdapter>,
        )
        .expect("dry run without adapter");

        for name in ["source_summary", "concept_extraction", "concept_merge"] {
            let entry = report
                .passes
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("missing `{name}` pass"));
            assert!(
                matches!(&entry.1, PassStatus::DryRun { .. }),
                "`{name}` must be DryRun under --dry-run (adapter absence is expected); got {:?}",
                entry.1
            );
        }

        // source_summary/concept_extraction previews list the stale doc IDs;
        // concept_merge previews the global merge scope.
        let summary_pass = report
            .passes
            .iter()
            .find(|(n, _)| n == "source_summary")
            .expect("source_summary entry");
        match &summary_pass.1 {
            PassStatus::DryRun { would_process } => {
                assert_eq!(would_process, &vec!["normalized/doc-a".to_string()]);
            }
            other => panic!("expected DryRun, got {other:?}"),
        }
    }

    #[test]
    fn dry_run_with_no_stale_docs_omits_per_doc_passes() {
        // When nothing is stale, the per-doc LLM passes wouldn't run in a live
        // compile either, so the dry-run must not list them.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        write_normalized_document(root, &test_doc("doc-1", "# Hello")).expect("write doc");

        // First compile writes hash state so the second run sees zero stale.
        run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
                progress: false,
                log_sink: None,
                reporter: None,
            },
        )
        .expect("first compile");

        let report = run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: true,
                progress: false,
                log_sink: None,
                reporter: None,
            },
        )
        .expect("dry run with clean state");

        assert_eq!(report.stale_sources, 0);
        for name in ["source_summary", "concept_extraction", "concept_merge"] {
            assert!(
                !report.passes.iter().any(|(n, _)| n == name),
                "`{name}` must not appear when no docs are stale"
            );
        }
    }

    #[test]
    fn dry_run_matches_live_run_stale_count() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        write_normalized_document(root, &test_doc("doc-1", "# A")).expect("write doc");
        write_normalized_document(root, &test_doc("doc-2", "# B")).expect("write doc");

        let dry = run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: true,
                progress: false,
                log_sink: None,
                reporter: None,
            },
        )
        .expect("dry run");

        let live = run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
                progress: false,
                log_sink: None,
                reporter: None,
            },
        )
        .expect("live run");

        assert_eq!(dry.total_sources, live.total_sources);
        assert_eq!(dry.stale_sources, live.stale_sources);

        // Dry-run must mention every pass that the live run actually executed
        // or skipped, so users see the same set of pass names in both modes.
        let live_names: std::collections::BTreeSet<&str> =
            live.passes.iter().map(|(n, _)| n.as_str()).collect();
        let dry_names: std::collections::BTreeSet<&str> =
            dry.passes.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            dry_names, live_names,
            "dry-run pass names must match live-run pass names"
        );
    }

    #[test]
    fn render_shows_pass_details() {
        let report = CompileReport {
            total_sources: 3,
            stale_sources: 1,
            build_records_emitted: 0,
            passes: vec![
                (
                    "backlinks".to_string(),
                    PassStatus::Executed { affected: 2 },
                ),
                (
                    "lexical_index".to_string(),
                    PassStatus::Skipped {
                        reason: "no stale nodes".to_string(),
                    },
                ),
            ],
        };

        let rendered = report.render();
        assert!(rendered.contains("3 source(s), 1 stale"));
        assert!(rendered.contains("[ok] backlinks (2 affected)"));
        assert!(rendered.contains("[skip] lexical_index"));
    }

    #[test]
    fn template_change_triggers_rebuild() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        write_normalized_document(root, &test_doc("doc-1", "# Hello")).expect("write doc");

        run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
                progress: false,
                log_sink: None,
                reporter: None,
            },
        )
        .expect("first compile");

        let report = run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
                progress: false,
                log_sink: None,
                reporter: None,
            },
        )
        .expect("second compile — no changes");
        assert_eq!(report.stale_sources, 0);

        // Write a custom template override to simulate a template change
        let overrides = prompts_dir(root);
        std::fs::create_dir_all(&overrides).expect("create prompts dir");
        std::fs::write(
            overrides.join("summarize_document.md"),
            "Updated template: {{title}}\n{{body}}",
        )
        .expect("write template");

        let report = run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
                progress: false,
                log_sink: None,
                reporter: None,
            },
        )
        .expect("third compile — template changed");
        assert_eq!(
            report.stale_sources, 1,
            "template change should mark all sources stale"
        );
    }

    // A minimal fake adapter that records its calls. Used only by pipeline tests to
    // verify the LLM passes are actually invoked when an adapter is supplied.
    #[derive(Default)]
    #[allow(clippy::struct_field_names)]
    struct RecordingAdapter {
        summarize_calls: std::sync::atomic::AtomicUsize,
        extract_calls: std::sync::atomic::AtomicUsize,
        merge_calls: std::sync::atomic::AtomicUsize,
    }

    impl RecordingAdapter {
        fn new_provenance() -> kb_llm::ProvenanceRecord {
            kb_llm::ProvenanceRecord {
                harness: "test".into(),
                harness_version: None,
                model: "test-model".into(),
                prompt_template_name: "t.md".into(),
                prompt_template_hash: kb_core::Hash::from([1u8; 32]),
                prompt_render_hash: kb_core::Hash::from([2u8; 32]),
                started_at: 0,
                ended_at: 1,
                latency_ms: 1,
                retries: 0,
                tokens: None,
                cost_estimate: None,
            }
        }
    }

    impl kb_llm::LlmAdapter for RecordingAdapter {
        fn summarize_document(
            &self,
            _req: kb_llm::SummarizeDocumentRequest,
        ) -> Result<(kb_llm::SummarizeDocumentResponse, kb_llm::ProvenanceRecord), kb_llm::LlmAdapterError>
        {
            self.summarize_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok((
                kb_llm::SummarizeDocumentResponse {
                    summary: "Fake summary.".into(),
                },
                Self::new_provenance(),
            ))
        }

        fn extract_concepts(
            &self,
            _req: kb_llm::ExtractConceptsRequest,
        ) -> Result<(kb_llm::ExtractConceptsResponse, kb_llm::ProvenanceRecord), kb_llm::LlmAdapterError>
        {
            self.extract_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok((
                kb_llm::ExtractConceptsResponse {
                    concepts: vec![kb_llm::ConceptCandidate {
                        name: "Rust".into(),
                        aliases: vec!["rustlang".into()],
                        definition_hint: Some("A systems language.".into()),
                        source_anchors: vec![],
                    }],
                },
                Self::new_provenance(),
            ))
        }

        fn merge_concept_candidates(
            &self,
            req: kb_llm::MergeConceptCandidatesRequest,
        ) -> Result<
            (kb_llm::MergeConceptCandidatesResponse, kb_llm::ProvenanceRecord),
            kb_llm::LlmAdapterError,
        > {
            self.merge_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let groups = req
                .candidates
                .into_iter()
                .map(|c| kb_llm::MergeGroup {
                    canonical_name: c.name.clone(),
                    aliases: c.aliases.clone(),
                    category: None,
                    confident: true,
                    rationale: None,
                    members: vec![c],
                })
                .collect();
            Ok((
                kb_llm::MergeConceptCandidatesResponse { groups },
                Self::new_provenance(),
            ))
        }

        fn answer_question(
            &self,
            _req: kb_llm::AnswerQuestionRequest,
        ) -> Result<(kb_llm::AnswerQuestionResponse, kb_llm::ProvenanceRecord), kb_llm::LlmAdapterError>
        {
            unreachable!("answer_question is not called during compile")
        }

        fn generate_slides(
            &self,
            _req: kb_llm::GenerateSlidesRequest,
        ) -> Result<(kb_llm::GenerateSlidesResponse, kb_llm::ProvenanceRecord), kb_llm::LlmAdapterError>
        {
            unreachable!()
        }

        fn run_health_check(
            &self,
            _req: kb_llm::RunHealthCheckRequest,
        ) -> Result<(kb_llm::RunHealthCheckResponse, kb_llm::ProvenanceRecord), kb_llm::LlmAdapterError>
        {
            unreachable!()
        }
    }

    #[test]
    fn compile_with_llm_runs_per_document_and_merge_passes() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        write_normalized_document(root, &test_doc("doc-a", "# Alpha\nBody A")).expect("write");
        write_normalized_document(root, &test_doc("doc-b", "# Beta\nBody B")).expect("write");

        let adapter = RecordingAdapter::default();
        let report = run_compile_with_llm(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
                progress: false,
                log_sink: None,
                reporter: None,
            },
            Some(&adapter),
        )
        .expect("compile with LLM");

        // Both documents were stale — each gets a summary + extraction call.
        assert_eq!(
            adapter
                .summarize_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            2,
            "summarize_document must run once per stale source"
        );
        assert_eq!(
            adapter
                .extract_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            2,
            "extract_concepts must run once per stale source"
        );
        assert_eq!(
            adapter
                .merge_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1,
            "concept merge runs once globally"
        );

        // Source pages should exist on disk. bn-nlw9: the filename now
        // includes a slug derived from the source title ("Alpha" → `alpha`).
        let page_a = root.join("wiki/sources/doc-a-alpha.md");
        assert!(
            page_a.exists(),
            "source page for doc-a missing at {}",
            page_a.display()
        );
        let body_a = std::fs::read_to_string(&page_a).expect("read page");
        assert!(body_a.contains("Fake summary."));
        assert!(body_a.contains("build_record_id:"));

        // Concept page should exist.
        assert!(
            root.join("wiki/concepts/rust.md").exists(),
            "concept page missing"
        );

        // Build records recorded (source_summary + extraction × 2 + merge × 1 = 5).
        assert_eq!(report.build_records_emitted, 5);

        // Second compile is a no-op: zero additional LLM calls.
        let report2 = run_compile_with_llm(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
                progress: false,
                log_sink: None,
                reporter: None,
            },
            Some(&adapter),
        )
        .expect("second compile");
        assert_eq!(report2.stale_sources, 0);
        assert_eq!(
            adapter
                .summarize_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            2,
            "second compile must not call summarize again"
        );
    }

    /// bn-nlw9: when a source's title changes between compiles, the wiki
    /// source page is renamed to reflect the new slug and the old file is
    /// cleaned up. Managed regions + frontmatter carry over because the
    /// pipeline reads the prior file via the id-or-slug resolver.
    #[test]
    fn compile_renames_source_page_on_title_change() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        // First compile: title "Alpha" → doc-a-alpha.md.
        write_normalized_document(root, &test_doc("doc-a", "# Alpha\nBody"))
            .expect("write doc-a");
        let adapter = RecordingAdapter::default();
        run_compile_with_llm(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
                progress: false,
                log_sink: None,
                reporter: None,
            },
            Some(&adapter),
        )
        .expect("first compile");
        assert!(
            root.join("wiki/sources/doc-a-alpha.md").exists(),
            "first compile must write slugged page"
        );

        // Edit the normalized text so the derived title changes. We also
        // have to bump source_revision_id so hash-state sees the doc as
        // stale and the per-doc passes re-run.
        let mut doc = kb_core::read_normalized_document(root, "doc-a").expect("read");
        doc.canonical_text = "# Beta\nBody".to_string();
        doc.source_revision_id = "rev-beta".to_string();
        kb_core::write_normalized_document(root, &doc).expect("rewrite");
        // Force rebuild — we're using a recording adapter with a fixed
        // provenance; the `content_hash` path alone wouldn't fire without
        // the rebuild pass below touching it.
        run_compile_with_llm(
            root,
            &CompileOptions {
                force: true,
                dry_run: false,
                progress: false,
                log_sink: None,
                reporter: None,
            },
            Some(&adapter),
        )
        .expect("second compile");

        assert!(
            root.join("wiki/sources/doc-a-beta.md").exists(),
            "second compile must write the new slug"
        );
        assert!(
            !root.join("wiki/sources/doc-a-alpha.md").exists(),
            "second compile must clean up the stale slug"
        );
        // Id-only form should not have appeared either.
        assert!(!root.join("wiki/sources/doc-a.md").exists());
    }

    /// bn-3w0: after a successful compile, `state/graph.json` must exist and
    /// contain nodes under the prefixes that `gather_status` and `run_inspect`
    /// count. Without this the CLI always reported zero wiki pages/concepts.
    #[test]
    fn compile_persists_graph_json_with_prefixed_nodes() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        write_normalized_document(root, &test_doc("doc-a", "# Alpha\nBody A"))
            .expect("write doc-a");
        write_normalized_document(root, &test_doc("doc-b", "# Beta\nBody B"))
            .expect("write doc-b");

        let adapter = RecordingAdapter::default();
        run_compile_with_llm(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
                progress: false,
                log_sink: None,
                reporter: None,
            },
            Some(&adapter),
        )
        .expect("compile with LLM");

        let graph_path = Graph::graph_path(root);
        assert!(
            graph_path.exists(),
            "compile must persist state/graph.json (found nothing at {})",
            graph_path.display()
        );

        let graph = Graph::load_from(root).expect("load persisted graph");

        let source_nodes: Vec<&String> = graph
            .nodes
            .keys()
            .filter(|k| k.starts_with("source-document-"))
            .collect();
        assert_eq!(
            source_nodes.len(),
            2,
            "expected 2 source-document nodes, got {source_nodes:?}"
        );

        let wiki_nodes: Vec<&String> = graph
            .nodes
            .keys()
            .filter(|k| k.starts_with("wiki-page-"))
            .collect();
        assert_eq!(
            wiki_nodes.len(),
            2,
            "expected 2 wiki-page nodes, got {wiki_nodes:?}"
        );

        let has_concept_node = graph
            .nodes
            .keys()
            .any(|k| k.starts_with("concept-"));
        assert!(
            has_concept_node,
            "expected at least one concept node after merge; graph: {:?}",
            graph.nodes.keys().collect::<Vec<_>>()
        );

        // Each source document should link to its rendered wiki page so
        // `kb inspect <wiki-page-slug>` can trace back to the source.
        // bn-nlw9: the wiki-page node carries the id-slug stem
        // (`doc-a-alpha` for the "Alpha" title) — edge matching is by
        // `<src-id>-*` prefix.
        let src_a = graph
            .nodes
            .get("source-document-doc-a")
            .expect("source-document-doc-a node present");
        assert!(
            src_a.outputs.iter().any(|o| o.starts_with("wiki-page-doc-a")),
            "source-document-doc-a must connect to a wiki-page-doc-a* node; outputs: {:?}",
            src_a.outputs
        );
    }

    #[test]
    fn compile_without_llm_skips_per_document_passes() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        write_normalized_document(root, &test_doc("doc-1", "# Hello\nWorld")).expect("write");

        let report = run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
                progress: false,
                log_sink: None,
                reporter: None,
            },
        )
        .expect("compile");

        assert_eq!(report.stale_sources, 1);
        assert_eq!(
            report.build_records_emitted, 0,
            "no adapter means no per-doc BuildRecords"
        );
        assert!(
            report.passes.iter().any(|(name, status)| name == "source_summary"
                && matches!(status, PassStatus::Skipped { .. })),
            "source_summary should be skipped when no adapter is configured"
        );
        assert!(
            !root.join("wiki/sources/doc-1.md").exists(),
            "without adapter, source pages must not be written"
        );
    }

    /// In-memory `LogSink` used by the log-streaming tests. Collects every
    /// message so assertions can look for `[run]`/`[ok]`/`[err]` markers.
    #[derive(Default, Clone)]
    struct MemorySink {
        lines: Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl MemorySink {
        fn snapshot(&self) -> Vec<String> {
            self.lines.lock().expect("lock").clone()
        }
    }

    impl LogSink for MemorySink {
        fn append_log(&self, message: &str) {
            self.lines.lock().expect("lock").push(message.to_string());
        }
    }

    /// An adapter whose `summarize_document` always fails. Mirrors the shape
    /// of a real opencode/claude error, including a large stderr tail that
    /// we expect the pipeline to truncate before embedding in the log.
    struct FailingAdapter {
        stderr_blob: String,
    }

    impl kb_llm::LlmAdapter for FailingAdapter {
        fn summarize_document(
            &self,
            _req: kb_llm::SummarizeDocumentRequest,
        ) -> Result<(kb_llm::SummarizeDocumentResponse, kb_llm::ProvenanceRecord), kb_llm::LlmAdapterError>
        {
            Err(kb_llm::LlmAdapterError::Transport(format!(
                "opencode exited with error: stderr: {}",
                self.stderr_blob
            )))
        }

        fn extract_concepts(
            &self,
            _req: kb_llm::ExtractConceptsRequest,
        ) -> Result<(kb_llm::ExtractConceptsResponse, kb_llm::ProvenanceRecord), kb_llm::LlmAdapterError>
        {
            unreachable!("extract should not run when summary fails")
        }

        fn merge_concept_candidates(
            &self,
            _req: kb_llm::MergeConceptCandidatesRequest,
        ) -> Result<(kb_llm::MergeConceptCandidatesResponse, kb_llm::ProvenanceRecord), kb_llm::LlmAdapterError>
        {
            // When every per-doc pass fails there are no candidates, so merge
            // short-circuits before calling this. Panic-loud guard anyway.
            unreachable!("merge should not run when every per-doc pass fails")
        }

        fn answer_question(
            &self,
            _req: kb_llm::AnswerQuestionRequest,
        ) -> Result<(kb_llm::AnswerQuestionResponse, kb_llm::ProvenanceRecord), kb_llm::LlmAdapterError>
        {
            unreachable!()
        }

        fn generate_slides(
            &self,
            _req: kb_llm::GenerateSlidesRequest,
        ) -> Result<(kb_llm::GenerateSlidesResponse, kb_llm::ProvenanceRecord), kb_llm::LlmAdapterError>
        {
            unreachable!()
        }

        fn run_health_check(
            &self,
            _req: kb_llm::RunHealthCheckRequest,
        ) -> Result<(kb_llm::RunHealthCheckResponse, kb_llm::ProvenanceRecord), kb_llm::LlmAdapterError>
        {
            unreachable!()
        }
    }

    /// A successful compile must stream a `[run]` and matching `[ok]` line
    /// per pass into the log sink. This is the core bn-3ny regression: before
    /// the fix, the `JobRun` log only ever contained "job started".
    #[test]
    fn compile_streams_run_and_ok_events_to_log_sink() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        write_normalized_document(root, &test_doc("doc-ok", "# Title\nBody.")).expect("write");

        let sink = MemorySink::default();
        let adapter = RecordingAdapter::default();
        let options = CompileOptions {
            force: false,
            dry_run: false,
            progress: false,
            log_sink: Some(Arc::new(sink.clone())),
            reporter: None,
        };
        run_compile_with_llm(root, &options, Some(&adapter)).expect("compile");

        let lines = sink.snapshot();
        assert!(
            lines.iter().any(|l| l.contains("[run] source_summary: doc-ok")),
            "missing [run] source_summary line; got {lines:?}"
        );
        assert!(
            lines.iter().any(|l| l.contains("[ok]  source_summary: doc-ok")),
            "missing [ok] source_summary line; got {lines:?}"
        );
        assert!(
            lines.iter().any(|l| l.contains("[run] concept_extraction: doc-ok")),
            "missing [run] concept_extraction line; got {lines:?}"
        );
        assert!(
            lines.iter().any(|l| l.contains("[ok]  concept_extraction: doc-ok")),
            "missing [ok] concept_extraction line; got {lines:?}"
        );
        assert!(
            lines.iter().any(|l| l.contains("[run] concept_merge")),
            "missing [run] concept_merge line; got {lines:?}"
        );
        assert!(
            lines.iter().any(|l| l.contains("[ok]  concept_merge")),
            "missing [ok] concept_merge line; got {lines:?}"
        );
        assert!(
            lines.iter().any(|l| l.starts_with("compile:")),
            "missing compile header line; got {lines:?}"
        );
    }

    /// When the LLM adapter fails, the log must capture an `[err]` line and
    /// include (a tail of) the subprocess stderr so the operator can debug
    /// without re-running a long compile.
    #[test]
    fn compile_captures_stderr_tail_on_llm_failure() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        write_normalized_document(root, &test_doc("doc-fail", "# T\nB.")).expect("write");

        // 5 KB of junk > ERR_TAIL_BYTES, so we verify truncation kicks in.
        let big_stderr: String = "E".repeat(5 * 1024) + "TAIL_MARKER";
        let adapter = FailingAdapter {
            stderr_blob: big_stderr.clone(),
        };

        let sink = MemorySink::default();
        let options = CompileOptions {
            force: false,
            dry_run: false,
            progress: false,
            log_sink: Some(Arc::new(sink.clone())),
            reporter: None,
        };
        run_compile_with_llm(root, &options, Some(&adapter)).expect("compile succeeds (failures logged, not bubbled)");

        let lines = sink.snapshot();
        let err_line = lines
            .iter()
            .find(|l| l.contains("[err] source_summary: doc-fail"))
            .unwrap_or_else(|| panic!("missing [err] line; got {lines:?}"));

        // Tail marker at the *end* of stderr must survive truncation.
        assert!(
            err_line.contains("TAIL_MARKER"),
            "stderr tail must preserve the last bytes; got {err_line:?}"
        );
        // Truncation must have dropped bytes — line length is bounded.
        assert!(
            err_line.len() < big_stderr.len(),
            "err line should be shorter than the full stderr; got {} bytes",
            err_line.len()
        );
        // Sentinel confirming the tail helper ran (prefix).
        assert!(
            err_line.contains("..."),
            "truncated tail should be prefixed with '...'; got {err_line:?}"
        );
    }

    /// Guards the behavior driving the Debug impl: the `log_sink` field must
    /// render as `<LogSink>` (not leak internal state) and `None` must render
    /// as `None`. Keeps Debug output useful in tracing without leaking paths.
    #[test]
    fn compile_options_debug_redacts_log_sink() {
        let with_sink = CompileOptions {
            force: false,
            dry_run: false,
            progress: false,
            log_sink: Some(Arc::new(MemorySink::default())),
            reporter: None,
        };
        let rendered = format!("{with_sink:?}");
        assert!(rendered.contains("Some(\"<LogSink>\")"), "got {rendered}");

        let without = CompileOptions::default();
        let rendered = format!("{without:?}");
        assert!(rendered.contains("log_sink: None"), "got {rendered}");
    }

    /// bn-1op: second compile on an unchanged corpus must not re-run the
    /// merge LLM pass. The first compile writes candidates + records the
    /// fingerprint; the second must see a matching fingerprint and skip.
    #[test]
    fn concept_merge_skipped_when_candidates_unchanged() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        write_normalized_document(root, &test_doc("doc-a", "# Alpha\nBody A")).expect("write");

        let adapter = RecordingAdapter::default();
        let options = CompileOptions::default();

        run_compile_with_llm(root, &options, Some(&adapter)).expect("first compile");
        assert_eq!(
            adapter
                .merge_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1,
            "first compile must run merge exactly once"
        );

        let state = HashState::load_from_root(root).expect("load state");
        assert!(
            state.hashes.contains_key(CONCEPT_MERGE_FINGERPRINT_KEY),
            "first compile must persist merge fingerprint; got {:?}",
            state.hashes.keys().collect::<Vec<_>>()
        );

        let report = run_compile_with_llm(root, &options, Some(&adapter)).expect("second compile");
        assert_eq!(
            adapter
                .merge_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1,
            "second compile must skip merge (no candidate changes)"
        );
        let merge_entry = report
            .passes
            .iter()
            .find(|(n, _)| n == "concept_merge")
            .expect("concept_merge entry");
        match &merge_entry.1 {
            PassStatus::Skipped { reason } => {
                assert!(
                    reason.contains("no candidate changes"),
                    "unexpected skip reason: {reason}"
                );
            }
            other => panic!("expected Skipped, got {other:?}"),
        }
    }

    /// Adding a new candidate JSON between runs must invalidate the
    /// fingerprint so merge runs again.
    #[test]
    fn concept_merge_runs_when_new_candidate_added() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        write_normalized_document(root, &test_doc("doc-a", "# Alpha\nBody A")).expect("write");

        let adapter = RecordingAdapter::default();
        let options = CompileOptions::default();
        run_compile_with_llm(root, &options, Some(&adapter)).expect("first compile");
        assert_eq!(
            adapter.merge_calls.load(std::sync::atomic::Ordering::SeqCst),
            1
        );

        // Drop a fresh candidate file into the candidates dir.
        let candidates_dir = crate::concept_extraction::concept_candidates_dir(root);
        std::fs::write(
            candidates_dir.join("injected.json"),
            r#"[{"name":"Novelty","aliases":[],"definition_hint":null,"source_anchors":[]}]"#,
        )
        .expect("write injected candidate");

        run_compile_with_llm(root, &options, Some(&adapter)).expect("second compile");
        assert_eq!(
            adapter.merge_calls.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "adding a candidate file must trigger merge"
        );
    }

    /// Changing the merge prompt template between runs must invalidate the
    /// fingerprint so merge runs again.
    #[test]
    fn concept_merge_runs_when_template_changes() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        write_normalized_document(root, &test_doc("doc-a", "# Alpha\nBody A")).expect("write");

        let adapter = RecordingAdapter::default();
        let options = CompileOptions::default();
        run_compile_with_llm(root, &options, Some(&adapter)).expect("first compile");

        // Override the merge template under .kb/prompts/ (Template::load
        // checks the user override dir first).
        let overrides = prompts_dir(root);
        std::fs::create_dir_all(&overrides).expect("create prompts");
        std::fs::write(
            overrides.join("merge_concept_candidates.md"),
            "NEW MERGE TEMPLATE: {{candidates}}",
        )
        .expect("write template");

        run_compile_with_llm(root, &options, Some(&adapter)).expect("second compile");
        assert!(
            adapter
                .merge_calls
                .load(std::sync::atomic::Ordering::SeqCst)
                >= 2,
            "template change must trigger merge"
        );
    }

    #[test]
    fn stderr_tail_preserves_short_input() {
        let s = "short";
        assert_eq!(stderr_tail(s), "short");
    }

    #[test]
    fn stderr_tail_truncates_long_input() {
        let s = "x".repeat(ERR_TAIL_BYTES + 100);
        let out = stderr_tail(&s);
        assert!(out.starts_with("..."));
        assert!(out.len() <= ERR_TAIL_BYTES + 3);
    }

    // bn-6puc Part 1: derive_title's four-rung fallback.

    #[test]
    fn derive_title_prefers_h1_heading() {
        let title = derive_title("# Real Title\n\nBody.", Some("file-stem"), "src-abc");
        assert_eq!(title, "Real Title");
    }

    #[test]
    fn derive_title_prefers_frontmatter_title_over_h1() {
        let text = "---\ntitle: Explicit Title\n---\n# H1 heading\n";
        let title = derive_title(text, Some("file-stem"), "src-abc");
        assert_eq!(title, "Explicit Title");
    }

    #[test]
    fn derive_title_falls_back_to_file_stem_when_no_h1() {
        // bn-6puc: body has no H1 (starts with H2/H3), no frontmatter — the
        // filename stem is preferred over the src id.
        let text = "## Sub heading\n\nBody starts here with a sub head.";
        let title = derive_title(
            text,
            Some("2026-04-07-LiveRamp-USB-team-intro"),
            "src-1wz",
        );
        assert_eq!(title, "2026-04-07-LiveRamp-USB-team-intro");
    }

    #[test]
    fn derive_title_falls_back_to_id_when_no_stem() {
        // URL-ingested source: `stem_fallback` is `None`, so step 4 applies.
        let title = derive_title("## Just sub heads.", None, "src-1wz");
        assert_eq!(title, "src-1wz");
    }

    #[test]
    fn derive_title_skips_empty_stem() {
        let title = derive_title("## Just sub heads.", Some("   "), "src-1wz");
        assert_eq!(title, "src-1wz");
    }

    #[test]
    fn derive_title_ignores_empty_h1_lines() {
        // An empty `# ` line must not be returned as the title.
        let text = "# \n\n# Actual\n";
        let title = derive_title(text, Some("stem"), "src-abc");
        assert_eq!(title, "Actual");
    }

    // bn-6puc: file_stem_from_stable_location coverage.

    #[test]
    fn file_stem_returns_basename_for_local_path() {
        let stem = kb_core::file_stem_from_stable_location(
            "/home/bob/notes/2026-04-07-LiveRamp-USB-team-intro.md",
        );
        assert_eq!(stem.as_deref(), Some("2026-04-07-LiveRamp-USB-team-intro"));
    }

    #[test]
    fn file_stem_returns_none_for_url() {
        let stem = kb_core::file_stem_from_stable_location("https://example.com/a/b");
        assert_eq!(stem, None);
    }

    #[test]
    fn file_stem_handles_repo_source_fragment() {
        let stem = kb_core::file_stem_from_stable_location(
            "git+https://github.com/foo/bar.git#notes/intro.md",
        );
        assert_eq!(stem.as_deref(), Some("intro"));
    }

    // bn-6puc Part 4 integration: end-to-end title fallback and filename.
    // Ingest a note with no frontmatter title and no H1 → compile's source
    // page must carry `title: <stem>` and land at `src-<id>-<slug-of-stem>.md`.
    #[test]
    fn compile_uses_file_stem_title_and_slugged_path_when_body_has_no_h1() {
        use std::fs;

        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        let src_id = "src-stem";
        let stem = "2026-04-07-LiveRamp-USB-team-intro";
        // Write a fake ingest record so derive_title can pick up the stem.
        let inbox = root.join("raw/inbox").join(src_id);
        fs::create_dir_all(&inbox).expect("mkdir inbox");
        let sd = serde_json::json!({
            "stable_location": format!("/tmp/{stem}.md"),
        });
        fs::write(inbox.join("source_document.json"), sd.to_string())
            .expect("write source_document.json");

        // Body has no H1 (starts with H2) — forces the stem fallback.
        write_normalized_document(
            root,
            &test_doc(src_id, "## Sub heading\n\nBody text.\n"),
        )
        .expect("write normalized");

        // Per-document passes (including source-page emission) require an
        // adapter — use the in-test RecordingAdapter.
        let adapter = RecordingAdapter::default();
        run_compile_with_llm(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
                progress: false,
                log_sink: None,
                reporter: None,
            },
            Some(&adapter),
        )
        .expect("compile");

        // The written source page lives under the slugged form of the stem —
        // NOT under `src-stem-src-stem.md`.
        let expected =
            root.join("wiki/sources/src-stem-2026-04-07-liveramp-usb-team-intro.md");
        assert!(
            expected.is_file(),
            "expected slugged stem path at {}",
            expected.display(),
        );
        let body = fs::read_to_string(&expected).expect("read page");
        assert!(
            body.contains(&format!("title: {stem}")),
            "frontmatter title must be the stem, got: {body}",
        );
        // No double-id ugliness.
        assert!(
            !root.join("wiki/sources/src-stem-src-stem.md").exists(),
            "compile must never emit a double-id source page filename",
        );
    }
}
