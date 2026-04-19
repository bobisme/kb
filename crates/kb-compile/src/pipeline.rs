use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use kb_core::{
    hash_bytes, hash_many, read_normalized_document, save_build_record, slug_from_title,
};
use kb_llm::{ConceptCandidate, LlmAdapter};

use crate::{HashState, StaleNode, detect_stale};

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
    /// When true, print per-pass progress lines (start/end with elapsed time)
    /// to stderr. Meant to reassure humans that long LLM passes are alive.
    /// Should be disabled under `--json` so structured consumers see clean logs.
    pub progress: bool,
    /// Optional sink that receives the same progress lines written to stderr,
    /// plus `[err]` lines on LLM-call failures (with a tail of subprocess
    /// stderr embedded in the error). `None` disables log forwarding.
    ///
    /// The CLI typically supplies a sink backed by the active `JobRun` log
    /// file so `state/jobs/<id>.log` captures per-pass events instead of
    /// containing only the initial "job started" line.
    pub log_sink: Option<Arc<dyn LogSink>>,
}

impl std::fmt::Debug for CompileOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompileOptions")
            .field("force", &self.force)
            .field("dry_run", &self.dry_run)
            .field("progress", &self.progress)
            .field("log_sink", &self.log_sink.as_ref().map(|_| "<LogSink>"))
            .finish()
    }
}

/// Emit `message` to both stderr (if progress is enabled) and the log
/// sink (if one is configured). Centralised so every call site stays in
/// sync and adding a new destination (e.g. tracing) is a one-line change.
fn emit_event(options: &CompileOptions, message: &str) {
    if options.progress {
        eprintln!("{message}");
    }
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
    let normalized_dir = root.join("normalized");
    if !normalized_dir.exists() {
        return Ok(Vec::new());
    }

    let mut ids = Vec::new();
    for entry in std::fs::read_dir(&normalized_dir)
        .with_context(|| format!("scan normalized dir {}", normalized_dir.display()))?
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
    // per-document LLM passes below can iterate the stale doc IDs.
    let (stale_sources, stale_doc_ids, hash_state) = if total_sources == 0 {
        (0, Vec::new(), None)
    } else {
        let previous = HashState::load_from_root(root)?;
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
        (count, ids, Some(report.current_state))
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

    emit_event(
        options,
        &format!("compile: {total_sources} source(s), {stale_sources} stale"),
    );

    // Per-document LLM passes (only when an adapter is configured and we have stale docs).
    if let Some(adapter) = adapter {
        if !stale_doc_ids.is_empty() {
            match run_per_document_passes(root, &stale_doc_ids, adapter, options) {
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
        match run_concept_merge_from_state(root, adapter, options) {
            Ok(report) => {
                build_records_emitted += report.build_records_emitted;
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

    Ok(CompileReport {
        total_sources,
        stale_sources,
        build_records_emitted,
        passes,
    })
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
) -> Result<PerDocumentReport> {
    let mut source_pages_written = 0;
    let mut candidate_files_written = 0;
    let mut build_records_emitted = 0;

    for doc_id in stale_doc_ids {
        let doc = read_normalized_document(root, doc_id)
            .with_context(|| format!("read normalized document {doc_id}"))?;
        let title = derive_title(&doc.canonical_text, doc_id);
        let page_path = crate::source_page::source_page_path_for_id(doc_id);
        let page_abs_path = root.join(&page_path);
        let existing_markdown = std::fs::read_to_string(&page_abs_path).ok();
        let existing_body = existing_markdown
            .as_deref()
            .and_then(split_body_from_frontmatter);
        let page_id = format!("wiki-source-{}", slug_from_title(doc_id));

        emit_event(options, &format!("  [run] source_summary: {doc_id}..."));
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
                emit_event(
                    options,
                    &format!(
                        "  [err] source_summary: {doc_id} ({:.1}s) — {err_str}",
                        summary_started.elapsed().as_secs_f64(),
                    ),
                );
                continue;
            }
        };
        emit_event(
            options,
            &format!(
                "  [ok]  source_summary: {doc_id} ({:.1}s)",
                summary_started.elapsed().as_secs_f64(),
            ),
        );

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
        let page_input = crate::source_page::SourcePageInput {
            page_id: &page_id,
            title: &title,
            source_document_id: doc_id,
            source_revision_id: &doc.source_revision_id,
            generated_at,
            build_record_id: &summary_artifact.build_record.metadata.id,
            summary: &summary_artifact.summary,
            key_topics: &summary_artifact.key_headings,
            citations: &citations_display,
        };

        let page_artifact = crate::source_page::render_source_page(
            &page_input,
            existing_markdown.as_deref(),
        )?;
        kb_core::fs::atomic_write(&page_abs_path, page_artifact.markdown.as_bytes())
            .with_context(|| format!("write source page {}", page_abs_path.display()))?;
        source_pages_written += 1;

        let candidates_path = crate::concept_extraction::concept_candidates_path(root, doc_id);
        emit_event(options, &format!("  [run] concept_extraction: {doc_id}..."));
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
                emit_event(
                    options,
                    &format!(
                        "  [ok]  concept_extraction: {doc_id} ({:.1}s)",
                        extract_started.elapsed().as_secs_f64(),
                    ),
                );
            }
            Err(err) => {
                tracing::warn!("concept_extraction failed for {doc_id}: {err}");
                let err_str = stderr_tail(&err.to_string());
                emit_event(
                    options,
                    &format!(
                        "  [err] concept_extraction: {doc_id} ({:.1}s) — {err_str}",
                        extract_started.elapsed().as_secs_f64(),
                    ),
                );
            }
        }
    }

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
fn run_concept_merge_from_state(
    root: &Path,
    adapter: &(dyn LlmAdapter + '_),
    options: &CompileOptions,
) -> Result<MergeRunReport> {
    let candidates_dir = root.join(crate::concept_extraction::CONCEPT_CANDIDATES_DIR);
    let mut all_candidates: Vec<ConceptCandidate> = Vec::new();
    if candidates_dir.exists() {
        for entry in std::fs::read_dir(&candidates_dir)
            .with_context(|| format!("read {}", candidates_dir.display()))?
        {
            let entry = entry?;
            if entry.file_type()?.is_file()
                && entry.path().extension().and_then(|s| s.to_str()) == Some("json")
            {
                let text = std::fs::read_to_string(entry.path())
                    .with_context(|| format!("read {}", entry.path().display()))?;
                let mut batch: Vec<ConceptCandidate> = serde_json::from_str(&text)
                    .with_context(|| format!("parse {}", entry.path().display()))?;
                all_candidates.append(&mut batch);
            }
        }
    }

    if all_candidates.is_empty() {
        return Ok(MergeRunReport {
            pages_written: 0,
            reviews_written: 0,
            build_records_emitted: 0,
        });
    }

    emit_event(
        options,
        &format!(
            "  [run] concept_merge: {} candidate(s)...",
            all_candidates.len(),
        ),
    );
    let merge_started = Instant::now();
    let artifact = match crate::concept_merge::run_concept_merge_pass(
        adapter,
        all_candidates,
        root,
    ) {
        Ok(a) => a,
        Err(err) => {
            let err_str = stderr_tail(&err.to_string());
            emit_event(
                options,
                &format!(
                    "  [err] concept_merge ({:.1}s) — {err_str}",
                    merge_started.elapsed().as_secs_f64(),
                ),
            );
            return Err(anyhow::anyhow!("{err}"));
        }
    };
    emit_event(
        options,
        &format!(
            "  [ok]  concept_merge ({:.1}s)",
            merge_started.elapsed().as_secs_f64(),
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

/// Extract a human-readable title: first `# ` heading in canonical text, else `fallback_id`.
fn derive_title(canonical_text: &str, fallback_id: &str) -> String {
    for line in canonical_text.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("# ") {
            let candidate = rest.trim();
            if !candidate.is_empty() {
                return candidate.to_string();
            }
        }
    }
    fallback_id.to_string()
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
        std::fs::create_dir_all(root.join("state")).expect("create state dir");
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
            },
        )
        .expect("second compile — no changes");
        assert_eq!(report.stale_sources, 0);

        // Write a custom template override to simulate a template change
        let prompts_dir = root.join("prompts");
        std::fs::create_dir_all(&prompts_dir).expect("create prompts dir");
        std::fs::write(
            prompts_dir.join("summarize_document.md"),
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

        // Source pages should exist on disk.
        let page_a = root.join("wiki/sources/doc-a.md");
        assert!(page_a.exists(), "source page for doc-a missing");
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
        };
        let rendered = format!("{with_sink:?}");
        assert!(rendered.contains("Some(\"<LogSink>\")"), "got {rendered}");

        let without = CompileOptions::default();
        let rendered = format!("{without:?}");
        assert!(rendered.contains("log_sink: None"), "got {rendered}");
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
}
