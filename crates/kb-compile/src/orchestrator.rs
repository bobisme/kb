use std::collections::BTreeSet;
use std::path::Path;

use anyhow::Result;

use kb_core::{BuildRecord, save_build_record};

#[derive(Debug, Clone)]
pub struct PassDecl {
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

pub struct PassContext<'a> {
    pub root: &'a Path,
    pub stale_nodes: &'a BTreeSet<String>,
}

pub struct PassOutput {
    pub build_records: Vec<BuildRecord>,
}

#[derive(Debug, Clone)]
pub enum PassOutcome {
    Executed { build_records: Vec<BuildRecord> },
    Failed { error: String },
    Skipped { reason: String },
    DryRun { would_process: Vec<String> },
}

pub trait Pass: Send + Sync {
    fn decl(&self) -> &PassDecl;

    /// Execute this pass against the stale nodes in `ctx`.
    ///
    /// # Errors
    ///
    /// Returns an error when the pass cannot complete (e.g. LLM failure, I/O error).
    fn execute(&self, ctx: &PassContext<'_>) -> Result<PassOutput>;
}

pub struct Orchestrator {
    passes: Vec<Box<dyn Pass>>,
}

#[derive(Debug)]
pub struct OrchestratorReport {
    pub outcomes: Vec<(String, PassOutcome)>,
}

impl OrchestratorReport {
    #[must_use]
    pub fn has_failures(&self) -> bool {
        self.outcomes
            .iter()
            .any(|(_, outcome)| matches!(outcome, PassOutcome::Failed { .. }))
    }

    #[must_use]
    pub fn executed_count(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|(_, outcome)| matches!(outcome, PassOutcome::Executed { .. }))
            .count()
    }

    #[must_use]
    pub fn all_build_records(&self) -> Vec<&BuildRecord> {
        self.outcomes
            .iter()
            .flat_map(|(_, outcome)| match outcome {
                PassOutcome::Executed { build_records } => build_records.iter().collect(),
                _ => Vec::new(),
            })
            .collect()
    }

    /// Persist all build records emitted by executed passes to `state/build_records/`.
    ///
    /// Records are written as `<id>.json` under the `build_records` directory. Existing
    /// records are overwritten (idempotent); records from previous runs are never deleted.
    ///
    /// # Errors
    /// Returns an error when any record cannot be written to disk.
    pub fn persist_build_records(&self, root: &Path) -> Result<()> {
        for record in self.all_build_records() {
            save_build_record(root, record)?;
        }
        Ok(())
    }

    #[must_use]
    pub fn render(&self) -> String {
        let mut lines = Vec::new();
        for (name, outcome) in &self.outcomes {
            match outcome {
                PassOutcome::Executed { build_records } => {
                    lines.push(format!(
                        "  [ok] {name} — {} record(s)",
                        build_records.len()
                    ));
                }
                PassOutcome::Failed { error } => {
                    lines.push(format!("  [FAIL] {name} — {error}"));
                }
                PassOutcome::Skipped { reason } => {
                    lines.push(format!("  [skip] {name} — {reason}"));
                }
                PassOutcome::DryRun { would_process } => {
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
        lines.join("\n")
    }
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self::new()
    }
}

impl Orchestrator {
    #[must_use]
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    pub fn register(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }

    #[must_use]
    pub fn pass_names(&self) -> Vec<&str> {
        self.passes.iter().map(|p| p.decl().name.as_str()).collect()
    }

    #[must_use]
    pub fn run(
        &self,
        root: &Path,
        stale: &BTreeSet<String>,
        dry_run: bool,
    ) -> OrchestratorReport {
        let mut outcomes = Vec::new();
        let mut failed_outputs: BTreeSet<String> = BTreeSet::new();

        for pass in &self.passes {
            let decl = pass.decl();
            let name = decl.name.clone();

            let blocked = decl.inputs.iter().any(|prefix| {
                failed_outputs
                    .iter()
                    .any(|f| f.starts_with(prefix.as_str()))
            });

            if blocked {
                for output_prefix in &decl.outputs {
                    for node in stale {
                        if node.starts_with(output_prefix.as_str()) {
                            failed_outputs.insert(node.clone());
                        }
                    }
                }
                outcomes.push((
                    name,
                    PassOutcome::Skipped {
                        reason: "upstream pass failed".to_string(),
                    },
                ));
                continue;
            }

            let relevant: Vec<String> = stale
                .iter()
                .filter(|node| {
                    decl.outputs
                        .iter()
                        .any(|prefix| node.starts_with(prefix.as_str()))
                })
                .cloned()
                .collect();

            if relevant.is_empty() {
                outcomes.push((
                    name,
                    PassOutcome::Skipped {
                        reason: "no stale nodes".to_string(),
                    },
                ));
                continue;
            }

            if dry_run {
                outcomes.push((
                    name,
                    PassOutcome::DryRun {
                        would_process: relevant,
                    },
                ));
                continue;
            }

            let relevant_set: BTreeSet<String> = relevant.into_iter().collect();
            let ctx = PassContext {
                root,
                stale_nodes: &relevant_set,
            };

            match pass.execute(&ctx) {
                Ok(output) => {
                    outcomes.push((
                        name,
                        PassOutcome::Executed {
                            build_records: output.build_records,
                        },
                    ));
                }
                Err(err) => {
                    for output_prefix in &decl.outputs {
                        for node in stale {
                            if node.starts_with(output_prefix.as_str()) {
                                failed_outputs.insert(node.clone());
                            }
                        }
                    }
                    outcomes.push((
                        name,
                        PassOutcome::Failed {
                            error: err.to_string(),
                        },
                    ));
                }
            }
        }

        OrchestratorReport { outcomes }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::bail;
    use kb_core::{EntityMetadata, Status};

    fn test_metadata(id: &str) -> EntityMetadata {
        EntityMetadata {
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
        }
    }

    fn test_record(id: &str, inputs: &[&str], outputs: &[&str]) -> BuildRecord {
        BuildRecord {
            metadata: test_metadata(id),
            pass_name: "test_pass".to_string(),
            input_ids: inputs.iter().map(|s| (*s).to_string()).collect(),
            output_ids: outputs.iter().map(|s| (*s).to_string()).collect(),
            manifest_hash: format!("hash-{id}"),
        }
    }

    struct SuccessPass {
        decl: PassDecl,
        records: Vec<BuildRecord>,
    }

    impl Pass for SuccessPass {
        fn decl(&self) -> &PassDecl {
            &self.decl
        }
        fn execute(&self, _ctx: &PassContext<'_>) -> Result<PassOutput> {
            Ok(PassOutput {
                build_records: self.records.clone(),
            })
        }
    }

    struct FailPass {
        decl: PassDecl,
    }

    impl Pass for FailPass {
        fn decl(&self) -> &PassDecl {
            &self.decl
        }
        fn execute(&self, _ctx: &PassContext<'_>) -> Result<PassOutput> {
            bail!("intentional failure in {}", self.decl.name)
        }
    }

    #[test]
    fn passes_execute_in_registration_order() {
        let mut orch = Orchestrator::new();
        let stale: BTreeSet<String> = [
            "normalized/a.json",
            "wiki/sources/a.md",
            "wiki/concepts/x.md",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();

        orch.register(Box::new(SuccessPass {
            decl: PassDecl {
                name: "normalize".to_string(),
                inputs: vec!["raw/".to_string()],
                outputs: vec!["normalized/".to_string()],
            },
            records: vec![test_record("norm-a", &["raw/a.md"], &["normalized/a.json"])],
        }));
        orch.register(Box::new(SuccessPass {
            decl: PassDecl {
                name: "source_pages".to_string(),
                inputs: vec!["normalized/".to_string()],
                outputs: vec!["wiki/sources/".to_string()],
            },
            records: vec![test_record(
                "src-a",
                &["normalized/a.json"],
                &["wiki/sources/a.md"],
            )],
        }));
        orch.register(Box::new(SuccessPass {
            decl: PassDecl {
                name: "concepts".to_string(),
                inputs: vec!["wiki/sources/".to_string()],
                outputs: vec!["wiki/concepts/".to_string()],
            },
            records: vec![test_record(
                "concept-x",
                &["wiki/sources/a.md"],
                &["wiki/concepts/x.md"],
            )],
        }));

        let report = orch.run(Path::new("/tmp/kb"), &stale, false);

        assert_eq!(report.outcomes.len(), 3);
        assert_eq!(report.outcomes[0].0, "normalize");
        assert_eq!(report.outcomes[1].0, "source_pages");
        assert_eq!(report.outcomes[2].0, "concepts");
        assert!(matches!(
            report.outcomes[0].1,
            PassOutcome::Executed { .. }
        ));
        assert!(matches!(
            report.outcomes[1].1,
            PassOutcome::Executed { .. }
        ));
        assert!(matches!(
            report.outcomes[2].1,
            PassOutcome::Executed { .. }
        ));
        assert_eq!(report.executed_count(), 3);
        assert!(!report.has_failures());
    }

    #[test]
    fn failure_skips_downstream_passes() {
        let mut orch = Orchestrator::new();
        let stale: BTreeSet<String> = [
            "normalized/a.json",
            "wiki/sources/a.md",
            "wiki/concepts/x.md",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();

        orch.register(Box::new(FailPass {
            decl: PassDecl {
                name: "normalize".to_string(),
                inputs: vec!["raw/".to_string()],
                outputs: vec!["normalized/".to_string()],
            },
        }));
        orch.register(Box::new(SuccessPass {
            decl: PassDecl {
                name: "source_pages".to_string(),
                inputs: vec!["normalized/".to_string()],
                outputs: vec!["wiki/sources/".to_string()],
            },
            records: vec![],
        }));
        orch.register(Box::new(SuccessPass {
            decl: PassDecl {
                name: "concepts".to_string(),
                inputs: vec!["wiki/sources/".to_string()],
                outputs: vec!["wiki/concepts/".to_string()],
            },
            records: vec![],
        }));

        let report = orch.run(Path::new("/tmp/kb"), &stale, false);

        assert!(report.has_failures());
        assert!(matches!(report.outcomes[0].1, PassOutcome::Failed { .. }));
        assert!(matches!(
            report.outcomes[1].1,
            PassOutcome::Skipped { ref reason } if reason == "upstream pass failed"
        ));
        assert!(matches!(
            report.outcomes[2].1,
            PassOutcome::Skipped { ref reason } if reason == "upstream pass failed"
        ));
    }

    #[test]
    fn failure_does_not_affect_unrelated_passes() {
        let mut orch = Orchestrator::new();
        let stale: BTreeSet<String> = ["normalized/a.json", "lint/report.json"]
            .iter()
            .map(|s| (*s).to_string())
            .collect();

        orch.register(Box::new(FailPass {
            decl: PassDecl {
                name: "normalize".to_string(),
                inputs: vec!["raw/".to_string()],
                outputs: vec!["normalized/".to_string()],
            },
        }));
        orch.register(Box::new(SuccessPass {
            decl: PassDecl {
                name: "lint".to_string(),
                inputs: vec!["sources/".to_string()],
                outputs: vec!["lint/".to_string()],
            },
            records: vec![test_record("lint-1", &["sources/a.md"], &["lint/report.json"])],
        }));

        let report = orch.run(Path::new("/tmp/kb"), &stale, false);

        assert!(matches!(report.outcomes[0].1, PassOutcome::Failed { .. }));
        assert!(matches!(
            report.outcomes[1].1,
            PassOutcome::Executed { .. }
        ));
    }

    #[test]
    fn build_records_emitted_per_pass() {
        let mut orch = Orchestrator::new();
        let stale: BTreeSet<String> = ["normalized/a.json", "normalized/b.json"]
            .iter()
            .map(|s| (*s).to_string())
            .collect();

        orch.register(Box::new(SuccessPass {
            decl: PassDecl {
                name: "normalize".to_string(),
                inputs: vec!["raw/".to_string()],
                outputs: vec!["normalized/".to_string()],
            },
            records: vec![
                test_record("norm-a", &["raw/a.md"], &["normalized/a.json"]),
                test_record("norm-b", &["raw/b.md"], &["normalized/b.json"]),
            ],
        }));

        let report = orch.run(Path::new("/tmp/kb"), &stale, false);

        let all_records = report.all_build_records();
        assert_eq!(all_records.len(), 2);
        assert_eq!(all_records[0].metadata.id, "norm-a");
        assert_eq!(all_records[1].metadata.id, "norm-b");
    }

    #[test]
    fn dry_run_lists_stale_without_executing() {
        let mut orch = Orchestrator::new();
        let stale: BTreeSet<String> = ["normalized/a.json", "wiki/sources/a.md"]
            .iter()
            .map(|s| (*s).to_string())
            .collect();

        orch.register(Box::new(SuccessPass {
            decl: PassDecl {
                name: "normalize".to_string(),
                inputs: vec!["raw/".to_string()],
                outputs: vec!["normalized/".to_string()],
            },
            records: vec![],
        }));
        orch.register(Box::new(SuccessPass {
            decl: PassDecl {
                name: "source_pages".to_string(),
                inputs: vec!["normalized/".to_string()],
                outputs: vec!["wiki/sources/".to_string()],
            },
            records: vec![],
        }));

        let report = orch.run(Path::new("/tmp/kb"), &stale, true);

        assert_eq!(report.executed_count(), 0);
        assert!(!report.has_failures());
        assert!(matches!(
            &report.outcomes[0].1,
            PassOutcome::DryRun { would_process } if would_process == &["normalized/a.json"]
        ));
        assert!(matches!(
            &report.outcomes[1].1,
            PassOutcome::DryRun { would_process } if would_process == &["wiki/sources/a.md"]
        ));
    }

    #[test]
    fn pass_with_no_stale_nodes_is_skipped() {
        let mut orch = Orchestrator::new();
        let stale: BTreeSet<String> =
            BTreeSet::from(["normalized/a.json".to_string()]);

        orch.register(Box::new(SuccessPass {
            decl: PassDecl {
                name: "normalize".to_string(),
                inputs: vec!["raw/".to_string()],
                outputs: vec!["normalized/".to_string()],
            },
            records: vec![],
        }));
        orch.register(Box::new(SuccessPass {
            decl: PassDecl {
                name: "concepts".to_string(),
                inputs: vec!["wiki/sources/".to_string()],
                outputs: vec!["wiki/concepts/".to_string()],
            },
            records: vec![],
        }));

        let report = orch.run(Path::new("/tmp/kb"), &stale, false);

        assert!(matches!(
            report.outcomes[0].1,
            PassOutcome::Executed { .. }
        ));
        assert!(matches!(
            report.outcomes[1].1,
            PassOutcome::Skipped { ref reason } if reason == "no stale nodes"
        ));
    }

    #[test]
    fn render_produces_human_readable_output() {
        let report = OrchestratorReport {
            outcomes: vec![
                (
                    "normalize".to_string(),
                    PassOutcome::Executed {
                        build_records: vec![test_record("norm-a", &["raw/a"], &["normalized/a"])],
                    },
                ),
                (
                    "concepts".to_string(),
                    PassOutcome::Skipped {
                        reason: "no stale nodes".to_string(),
                    },
                ),
            ],
        };

        let rendered = report.render();
        assert!(rendered.contains("[ok] normalize"));
        assert!(rendered.contains("[skip] concepts"));
    }

    #[test]
    fn persist_build_records_writes_to_build_records_dir() {
        use kb_core::load_build_record;
        use tempfile::tempdir;

        let dir = tempdir().expect("tempdir");
        let mut orch = Orchestrator::new();
        let stale: BTreeSet<String> = BTreeSet::from(["normalized/a.json".to_string()]);

        orch.register(Box::new(SuccessPass {
            decl: PassDecl {
                name: "normalize".to_string(),
                inputs: vec!["raw/".to_string()],
                outputs: vec!["normalized/".to_string()],
            },
            records: vec![test_record("norm-a", &["raw/a.md"], &["normalized/a.json"])],
        }));

        let report = orch.run(dir.path(), &stale, false);
        report
            .persist_build_records(dir.path())
            .expect("persist build records");

        let loaded = load_build_record(dir.path(), "norm-a")
            .expect("load")
            .expect("record present");
        assert_eq!(loaded.metadata.id, "norm-a");
        assert_eq!(loaded.pass_name, "test_pass");
        assert_eq!(loaded.input_ids, vec!["raw/a.md"]);
        assert_eq!(loaded.output_ids, vec!["normalized/a.json"]);
    }
}
