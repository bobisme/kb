use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context, Result};
use kb_core::{hash_bytes, read_normalized_document};

use crate::{HashState, StaleNode, detect_stale};

#[derive(Debug, Clone)]
pub struct CompileOptions {
    pub force: bool,
    pub dry_run: bool,
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

fn build_input_nodes(root: &Path, doc_ids: &[String]) -> Result<Vec<StaleNode>> {
    let mut nodes = Vec::with_capacity(doc_ids.len());
    for id in doc_ids {
        let doc = read_normalized_document(root, id)
            .with_context(|| format!("read normalized document {id}"))?;
        nodes.push(StaleNode {
            id: format!("normalized/{id}"),
            dependencies: vec![],
            content_hash: hash_bytes(doc.canonical_text.as_bytes()).to_hex(),
            prompt_template_hash: None,
            model_version: None,
        });
    }
    Ok(nodes)
}

/// Run the full compile pipeline with incremental stale detection.
///
/// Stale detection gates per-document LLM passes (source summary, concept
/// extraction, concept merge). Batch passes that scan the wiki directory
/// (backlinks, lexical index, index pages) always run because they are cheap
/// and their inputs may have changed outside the normalized-doc pipeline.
///
/// # Errors
///
/// Returns an error when normalized documents cannot be read, passes fail,
/// or state files cannot be persisted.
#[allow(clippy::too_many_lines)]
pub fn run_compile(root: &Path, options: &CompileOptions) -> Result<CompileReport> {
    let doc_ids = discover_normalized_ids(root)?;
    let total_sources = doc_ids.len();

    // Stale detection on normalized documents
    let (stale_sources, hash_state) = if total_sources == 0 {
        (0, None)
    } else {
        let previous = HashState::load_from_root(root)?;
        let nodes = build_input_nodes(root, &doc_ids)?;

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
        (count, Some(report.current_state))
    };

    if options.dry_run {
        let passes = vec![
            (
                "backlinks".to_string(),
                PassStatus::DryRun {
                    would_process: vec!["wiki/concepts/*".to_string()],
                },
            ),
            (
                "lexical_index".to_string(),
                PassStatus::DryRun {
                    would_process: vec!["wiki/**/*.md".to_string()],
                },
            ),
            (
                "index_pages".to_string(),
                PassStatus::DryRun {
                    would_process: vec![
                        "wiki/index.md".to_string(),
                        "wiki/sources/index.md".to_string(),
                        "wiki/concepts/index.md".to_string(),
                    ],
                },
            ),
        ];

        return Ok(CompileReport {
            total_sources,
            stale_sources,
            build_records_emitted: 0,
            passes,
        });
    }

    let mut passes = Vec::new();

    // Pass: backlinks (no LLM — scans wiki links and updates concept pages)
    match crate::backlinks::run_backlinks_pass(root) {
        Ok(artifacts) => {
            let mut updated = 0;
            for artifact in &artifacts {
                if artifact.needs_update() {
                    kb_core::fs::atomic_write(
                        &artifact.path,
                        artifact.updated_markdown.as_bytes(),
                    )
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
        build_records_emitted: 0,
        passes,
    })
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
    fn compile_on_empty_kb_runs_batch_passes() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        let report = run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
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

        write_normalized_document(root, &test_doc("doc-1", "# Hello\nWorld"))
            .expect("write doc");
        write_normalized_document(root, &test_doc("doc-2", "# Second\nDocument"))
            .expect("write doc");

        let report = run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
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

        write_normalized_document(root, &test_doc("doc-1", "# Hello\nWorld"))
            .expect("write doc");

        run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
            },
        )
        .expect("first compile");

        let report = run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
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

        write_normalized_document(root, &test_doc("doc-1", "# Original"))
            .expect("write doc");

        run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
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

        write_normalized_document(root, &test_doc("doc-1", "# Hello"))
            .expect("write doc");

        run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
            },
        )
        .expect("first compile");

        let report = run_compile(
            root,
            &CompileOptions {
                force: true,
                dry_run: false,
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

        write_normalized_document(root, &test_doc("doc-1", "# Hello"))
            .expect("write doc");

        let report = run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: true,
            },
        )
        .expect("dry run");

        assert_eq!(report.total_sources, 1);
        assert_eq!(report.stale_sources, 1);
        assert_eq!(report.build_records_emitted, 0);

        for (_, status) in &report.passes {
            assert!(
                matches!(status, PassStatus::DryRun { .. }),
                "all passes should be dry-run"
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
    fn dry_run_matches_live_run_stale_count() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        setup_kb(root);

        write_normalized_document(root, &test_doc("doc-1", "# A"))
            .expect("write doc");
        write_normalized_document(root, &test_doc("doc-2", "# B"))
            .expect("write doc");

        let dry = run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: true,
            },
        )
        .expect("dry run");

        let live = run_compile(
            root,
            &CompileOptions {
                force: false,
                dry_run: false,
            },
        )
        .expect("live run");

        assert_eq!(dry.total_sources, live.total_sources);
        assert_eq!(dry.stale_sources, live.stale_sources);
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
}
