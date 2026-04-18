use std::fmt::Write as FmtWrite;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use kb_core::fs::atomic_write;
use kb_core::{BuildRecord, EntityMetadata, Status, hash_many, slug_from_title};
use kb_llm::{
    ConceptCandidate, LlmAdapter, LlmAdapterError, MergeConceptCandidatesRequest,
    MergeConceptCandidatesResponse, MergeGroup, ProvenanceRecord, SourceAnchor,
};

pub const WIKI_CONCEPTS_DIR: &str = "wiki/concepts";
pub const MERGE_REVIEWS_DIR: &str = "reviews/merges";

/// A canonical concept page produced by the merge pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConceptPage {
    /// URL-safe slug derived from the canonical name.
    pub slug: String,
    /// Destination path under `wiki/concepts/`.
    pub path: PathBuf,
    /// Full markdown content including YAML frontmatter.
    pub content: String,
    pub canonical_name: String,
    pub aliases: Vec<String>,
}

/// An uncertain merge queued for human review.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeReviewRecord {
    pub merge_id: String,
    /// Destination path under `reviews/merges/`.
    pub path: PathBuf,
    /// JSON payload to persist.
    pub content: String,
}

/// Output emitted by the concept merge pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConceptMergeArtifact {
    pub concept_pages: Vec<ConceptPage>,
    pub review_records: Vec<MergeReviewRecord>,
    pub build_record: BuildRecord,
}

/// Errors produced by the concept merge pass.
#[derive(Debug, thiserror::Error)]
pub enum ConceptMergeError {
    #[error("llm merge_concept_candidates failed: {0}")]
    Llm(#[from] LlmAdapterError),
    #[error("failed to serialize: {0}")]
    Serialize(String),
    #[error("failed to read system time: {0}")]
    Time(String),
}

/// Persist a concept page produced by [`run_concept_merge_pass`].
///
/// # Errors
///
/// Returns an error when the target file cannot be written atomically.
pub fn persist_concept_page(page: &ConceptPage) -> Result<()> {
    atomic_write(&page.path, page.content.as_bytes())
        .with_context(|| format!("write concept page {}", page.path.display()))
}

/// Persist a merge review record produced by [`run_concept_merge_pass`].
///
/// # Errors
///
/// Returns an error when the target file cannot be written atomically.
pub fn persist_merge_review(record: &MergeReviewRecord) -> Result<()> {
    atomic_write(&record.path, record.content.as_bytes())
        .with_context(|| format!("write merge review {}", record.path.display()))
}

/// Cluster concept candidates into canonical concept pages and review items.
///
/// Candidates are sorted by name before being sent to the LLM so that identical
/// inputs always produce identical outputs across runs.
///
/// Confident merge groups become `wiki/concepts/<slug>.md`; uncertain groups go to
/// `reviews/merges/<merge-id>.json` for human approval.
///
/// # Errors
///
/// Returns [`ConceptMergeError`] when the LLM call fails, the response cannot be
/// serialized, or the system clock cannot be read for build-record timestamps.
pub fn run_concept_merge_pass<A: LlmAdapter>(
    adapter: &A,
    mut candidates: Vec<ConceptCandidate>,
    root: &Path,
) -> Result<ConceptMergeArtifact, ConceptMergeError> {
    candidates.sort_unstable_by(|a, b| a.name.cmp(&b.name));

    let request = MergeConceptCandidatesRequest {
        candidates: candidates.clone(),
    };
    let (response, provenance) = adapter.merge_concept_candidates(request)?;

    let concept_pages = build_concept_pages(&response, root);
    let review_records =
        build_review_records(&response, root, &provenance).map_err(ConceptMergeError::Serialize)?;
    let build_record = build_record_for_merge(
        &candidates,
        &response,
        &concept_pages,
        &review_records,
        &provenance,
    )?;

    Ok(ConceptMergeArtifact {
        concept_pages,
        review_records,
        build_record,
    })
}

fn build_concept_pages(response: &MergeConceptCandidatesResponse, root: &Path) -> Vec<ConceptPage> {
    response
        .groups
        .iter()
        .filter(|g| g.confident)
        .map(|group| render_concept_page(group, root))
        .collect()
}

fn build_review_records(
    response: &MergeConceptCandidatesResponse,
    root: &Path,
    provenance: &ProvenanceRecord,
) -> Result<Vec<MergeReviewRecord>, String> {
    response
        .groups
        .iter()
        .filter(|g| !g.confident)
        .map(|group| render_review_record(group, root, provenance))
        .collect()
}

fn render_concept_page(group: &MergeGroup, root: &Path) -> ConceptPage {
    let slug = slug_from_title(&group.canonical_name);
    let concept_id = format!("concept:{slug}");
    let path = root.join(WIKI_CONCEPTS_DIR).join(format!("{slug}.md"));

    let source_anchors: Vec<&SourceAnchor> = group
        .members
        .iter()
        .flat_map(|m| m.source_anchors.iter())
        .collect();

    let definition_hint: Option<&str> = group
        .members
        .iter()
        .find_map(|m| m.definition_hint.as_deref());

    let mut content = format!("---\nid: {concept_id}\nname: {}\n", group.canonical_name);

    if !group.aliases.is_empty() {
        content.push_str("aliases:\n");
        for alias in &group.aliases {
            writeln!(content, "  - {alias}").expect("infallible write to String");
        }
    }

    if !source_anchors.is_empty() {
        content.push_str("sources:\n");
        for anchor in &source_anchors {
            if let Some(heading) = &anchor.heading_anchor {
                writeln!(content, "  - heading_anchor: {heading}").expect("infallible");
                if let Some(quote) = &anchor.quote {
                    let escaped = quote.replace('"', "\\\"");
                    writeln!(content, "    quote: \"{escaped}\"").expect("infallible");
                }
            } else if let Some(quote) = &anchor.quote {
                let escaped = quote.replace('"', "\\\"");
                writeln!(content, "  - quote: \"{escaped}\"").expect("infallible");
            }
        }
    }

    content.push_str("---\n\n");
    writeln!(content, "# {}", group.canonical_name).expect("infallible write to String");

    if let Some(hint) = definition_hint {
        content.push('\n');
        content.push_str(hint);
        content.push('\n');
    }

    ConceptPage {
        slug,
        path,
        content,
        canonical_name: group.canonical_name.clone(),
        aliases: group.aliases.clone(),
    }
}

fn render_review_record(
    group: &MergeGroup,
    root: &Path,
    provenance: &ProvenanceRecord,
) -> Result<MergeReviewRecord, String> {
    let slug = slug_from_title(&group.canonical_name);
    let merge_id = format!("merge:{slug}");
    let path = root
        .join(MERGE_REVIEWS_DIR)
        .join(format!("{merge_id}.json"));

    let affected_pages = vec![
        root.join(WIKI_CONCEPTS_DIR).join(format!("{slug}.md")),
    ];

    let payload = serde_json::json!({
        "id": merge_id,
        "kind": "concept_merge",
        "status": "pending",
        "proposed_destination": format!("wiki/concepts/{slug}.md"),
        "affected_pages": affected_pages,
        "canonical_name_proposed": group.canonical_name,
        "aliases_proposed": group.aliases,
        "rationale": group.rationale,
        "members": group.members,
        "provenance_model": provenance.model,
        "provenance_template": provenance.prompt_template_name,
    });

    let content = serde_json::to_string_pretty(&payload)
        .map_err(|err| format!("serialize merge review record: {err}"))?;

    Ok(MergeReviewRecord {
        merge_id,
        path,
        content,
    })
}

fn build_record_for_merge(
    candidates: &[ConceptCandidate],
    response: &MergeConceptCandidatesResponse,
    concept_pages: &[ConceptPage],
    review_records: &[MergeReviewRecord],
    provenance: &ProvenanceRecord,
) -> Result<BuildRecord, ConceptMergeError> {
    let now = unix_time_ms()?;
    let prompt_template_hash = provenance.prompt_template_hash.to_hex();
    let prompt_render_hash = provenance.prompt_render_hash.to_hex();

    let candidate_names_joined: String = candidates
        .iter()
        .map(|c| c.name.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    let group_count = response.groups.len().to_string();
    let confident_count = response
        .groups
        .iter()
        .filter(|g| g.confident)
        .count()
        .to_string();

    let manifest_hash = hash_many(&[
        candidate_names_joined.as_bytes(),
        b"\0",
        group_count.as_bytes(),
        b"\0",
        confident_count.as_bytes(),
        b"\0",
        prompt_render_hash.as_bytes(),
    ])
    .to_hex();

    let input_ids: Vec<String> = candidates.iter().map(|c| c.name.clone()).collect();

    let output_ids: Vec<String> = concept_pages
        .iter()
        .map(|p| p.path.display().to_string())
        .chain(review_records.iter().map(|r| r.path.display().to_string()))
        .collect();

    let output_paths: Vec<PathBuf> = concept_pages
        .iter()
        .map(|p| p.path.clone())
        .chain(review_records.iter().map(|r| r.path.clone()))
        .collect();

    Ok(BuildRecord {
        pass_name: "merge_concept_candidates".to_string(),
        metadata: EntityMetadata {
            id: "build:merge-concept-candidates".to_string(),
            created_at_millis: now,
            updated_at_millis: now,
            source_hashes: vec![candidate_names_joined, prompt_render_hash],
            model_version: Some(provenance.model.clone()),
            tool_version: Some(format!(
                "{}/{}",
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION")
            )),
            prompt_template_hash: Some(prompt_template_hash),
            dependencies: input_ids.clone(),
            output_paths,
            status: Status::Fresh,
        },
        input_ids,
        output_ids,
        manifest_hash,
    })
}

fn unix_time_ms() -> Result<u64, ConceptMergeError> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| ConceptMergeError::Time(err.to_string()))?;
    let millis = duration
        .as_millis()
        .try_into()
        .context("system time exceeded u64 millisecond range")
        .map_err(|err| ConceptMergeError::Time(err.to_string()))?;
    Ok(millis)
}

#[cfg(test)]
mod tests {
    use super::*;

    use kb_core::Hash;
    use kb_llm::{
        AnswerQuestionRequest, AnswerQuestionResponse, ExtractConceptsRequest,
        ExtractConceptsResponse, GenerateSlidesRequest, GenerateSlidesResponse, LlmAdapter,
        LlmAdapterError, MergeConceptCandidatesResponse, RunHealthCheckRequest,
        RunHealthCheckResponse, SourceAnchor, SummarizeDocumentRequest, SummarizeDocumentResponse,
        TokenUsage,
    };
    use tempfile::tempdir;

    #[derive(Debug)]
    struct FakeAdapter {
        response: MergeConceptCandidatesResponse,
        provenance: ProvenanceRecord,
    }

    impl LlmAdapter for FakeAdapter {
        fn summarize_document(
            &self,
            _request: SummarizeDocumentRequest,
        ) -> Result<(SummarizeDocumentResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!("unused in concept merge test")
        }

        fn extract_concepts(
            &self,
            _request: ExtractConceptsRequest,
        ) -> Result<(ExtractConceptsResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!("unused in concept merge test")
        }

        fn merge_concept_candidates(
            &self,
            _request: MergeConceptCandidatesRequest,
        ) -> Result<(MergeConceptCandidatesResponse, ProvenanceRecord), LlmAdapterError> {
            Ok((self.response.clone(), self.provenance.clone()))
        }

        fn answer_question(
            &self,
            _request: AnswerQuestionRequest,
        ) -> Result<(AnswerQuestionResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!("unused in concept merge test")
        }

        fn generate_slides(
            &self,
            _request: GenerateSlidesRequest,
        ) -> Result<(GenerateSlidesResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!("unused in concept merge test")
        }

        fn run_health_check(
            &self,
            _request: RunHealthCheckRequest,
        ) -> Result<(RunHealthCheckResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!("unused in concept merge test")
        }
    }

    fn provenance() -> ProvenanceRecord {
        ProvenanceRecord {
            harness: "opencode".to_string(),
            harness_version: None,
            model: "openai/gpt-5.4".to_string(),
            prompt_template_name: "merge_concept_candidates.md".to_string(),
            prompt_template_hash: Hash::from([9u8; 32]),
            prompt_render_hash: Hash::from([10u8; 32]),
            started_at: 10,
            ended_at: 20,
            latency_ms: 10,
            retries: 0,
            tokens: Some(TokenUsage {
                prompt_tokens: 100,
                completion_tokens: 50,
            }),
            cost_estimate: None,
        }
    }

    fn borrow_checker_candidate() -> ConceptCandidate {
        ConceptCandidate {
            name: "borrow checker".to_string(),
            aliases: vec!["borrowck".to_string()],
            definition_hint: Some("Validates references at compile time.".to_string()),
            source_anchors: vec![SourceAnchor {
                heading_anchor: Some("ownership".to_string()),
                quote: Some("The borrow checker validates references.".to_string()),
            }],
        }
    }

    fn borrowck_candidate() -> ConceptCandidate {
        ConceptCandidate {
            name: "borrowck".to_string(),
            aliases: vec![],
            definition_hint: None,
            source_anchors: vec![SourceAnchor {
                heading_anchor: None,
                quote: Some("borrowck prevents dangling pointers.".to_string()),
            }],
        }
    }

    fn uncertain_candidate() -> ConceptCandidate {
        ConceptCandidate {
            name: "lifetime elision".to_string(),
            aliases: vec![],
            definition_hint: Some("Rules for inferring lifetimes.".to_string()),
            source_anchors: vec![],
        }
    }

    #[test]
    fn confident_groups_become_concept_pages() {
        let dir = tempdir().expect("tempdir");
        let adapter = FakeAdapter {
            response: MergeConceptCandidatesResponse {
                groups: vec![MergeGroup {
                    canonical_name: "Borrow checker".to_string(),
                    aliases: vec!["borrowck".to_string()],
                    members: vec![borrow_checker_candidate(), borrowck_candidate()],
                    confident: true,
                    rationale: None,
                }],
            },
            provenance: provenance(),
        };

        let candidates = vec![borrow_checker_candidate(), borrowck_candidate()];
        let artifact =
            run_concept_merge_pass(&adapter, candidates, dir.path()).expect("run merge pass");

        assert_eq!(artifact.concept_pages.len(), 1);
        assert!(artifact.review_records.is_empty());

        let page = &artifact.concept_pages[0];
        assert_eq!(page.slug, "borrow-checker");
        assert_eq!(page.canonical_name, "Borrow checker");
        assert_eq!(page.aliases, vec!["borrowck"]);
        assert!(page.content.contains("id: concept:borrow-checker"));
        assert!(page.content.contains("name: Borrow checker"));
        assert!(page.content.contains("aliases:"));
        assert!(page.content.contains("- borrowck"));
        assert!(page.content.contains("# Borrow checker"));
        assert!(
            page.content
                .contains("Validates references at compile time.")
        );
        assert_eq!(
            page.path,
            dir.path().join("wiki/concepts/borrow-checker.md")
        );
    }

    #[test]
    fn uncertain_groups_become_review_records() {
        let dir = tempdir().expect("tempdir");
        let adapter = FakeAdapter {
            response: MergeConceptCandidatesResponse {
                groups: vec![MergeGroup {
                    canonical_name: "Lifetime elision".to_string(),
                    aliases: vec![],
                    members: vec![uncertain_candidate()],
                    confident: false,
                    rationale: Some(
                        "Ambiguous whether this is the same as lifetime rules.".to_string(),
                    ),
                }],
            },
            provenance: provenance(),
        };

        let artifact = run_concept_merge_pass(&adapter, vec![uncertain_candidate()], dir.path())
            .expect("run merge pass");

        assert!(artifact.concept_pages.is_empty());
        assert_eq!(artifact.review_records.len(), 1);

        let review = &artifact.review_records[0];
        assert_eq!(review.merge_id, "merge:lifetime-elision");
        assert_eq!(
            review.path,
            dir.path()
                .join("reviews/merges/merge:lifetime-elision.json")
        );
        assert!(review.content.contains("canonical_name_proposed"));
        assert!(review.content.contains("Lifetime elision"));
        assert!(review.content.contains("Ambiguous"));
    }

    #[test]
    fn candidates_sorted_before_sending_to_adapter() {
        let dir = tempdir().expect("tempdir");
        let adapter = FakeAdapter {
            response: MergeConceptCandidatesResponse { groups: vec![] },
            provenance: provenance(),
        };

        let candidates = vec![borrowck_candidate(), borrow_checker_candidate()];
        let artifact =
            run_concept_merge_pass(&adapter, candidates, dir.path()).expect("run merge pass");

        assert!(artifact.concept_pages.is_empty());
        assert!(artifact.review_records.is_empty());
        assert_eq!(artifact.build_record.pass_name, "merge_concept_candidates");
    }

    #[test]
    fn build_record_lists_concept_page_paths_as_outputs() {
        let dir = tempdir().expect("tempdir");
        let adapter = FakeAdapter {
            response: MergeConceptCandidatesResponse {
                groups: vec![
                    MergeGroup {
                        canonical_name: "Borrow checker".to_string(),
                        aliases: vec!["borrowck".to_string()],
                        members: vec![borrow_checker_candidate()],
                        confident: true,
                        rationale: None,
                    },
                    MergeGroup {
                        canonical_name: "Lifetime elision".to_string(),
                        aliases: vec![],
                        members: vec![uncertain_candidate()],
                        confident: false,
                        rationale: Some("Uncertain".to_string()),
                    },
                ],
            },
            provenance: provenance(),
        };

        let candidates = vec![borrow_checker_candidate(), uncertain_candidate()];
        let artifact =
            run_concept_merge_pass(&adapter, candidates, dir.path()).expect("run merge pass");

        let output_ids = &artifact.build_record.output_ids;
        assert!(
            output_ids.iter().any(|id| id.contains("borrow-checker.md")),
            "output_ids: {output_ids:?}"
        );
        assert!(
            output_ids
                .iter()
                .any(|id| id.contains("lifetime-elision.json")),
            "output_ids: {output_ids:?}"
        );
    }

    #[test]
    fn persist_concept_page_writes_file() {
        let dir = tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("wiki/concepts")).expect("create dir");

        let page = ConceptPage {
            slug: "borrow-checker".to_string(),
            path: dir.path().join("wiki/concepts/borrow-checker.md"),
            content: "---\nid: concept:borrow-checker\n---\n".to_string(),
            canonical_name: "Borrow checker".to_string(),
            aliases: vec![],
        };

        persist_concept_page(&page).expect("persist concept page");
        let written = std::fs::read_to_string(&page.path).expect("read file");
        assert_eq!(written, page.content);
    }

    #[test]
    fn parse_merge_concept_candidates_json_roundtrips() {
        use kb_llm::parse_merge_concept_candidates_json;

        let json = r#"{
            "groups": [
                {
                    "canonical_name": "Borrow checker",
                    "aliases": ["borrowck"],
                    "members": [],
                    "confident": true,
                    "rationale": null
                }
            ]
        }"#;

        let response = parse_merge_concept_candidates_json(json).expect("parse");
        assert_eq!(response.groups.len(), 1);
        assert_eq!(response.groups[0].canonical_name, "Borrow checker");
        assert!(response.groups[0].confident);
    }
}
