//! Integration test: concept categories end-to-end.
//!
//! Stubs the LLM adapter to return merge groups with explicit `category`
//! fields, then asserts:
//! 1. Each confident group's concept page frontmatter carries `category:`.
//! 2. `wiki/concepts/index.md`, once rendered from those pages, groups
//!    entries by category with an "Uncategorized" bucket at the end.
//!
//! This exercises the path from `MergeGroup.category` → concept page
//! frontmatter → `generate_indexes` category grouping, which is the
//! user-visible contract the bone (bn-eqx7) adds.

use std::path::Path;

use kb_compile::concept_merge::{
    persist_concept_page, run_concept_merge_pass, ConceptPage,
};
use kb_compile::index_page::generate_indexes;
use kb_llm::{
    AnswerQuestionRequest, AnswerQuestionResponse, ConceptCandidate, ExtractConceptsRequest,
    ExtractConceptsResponse, GenerateSlidesRequest, GenerateSlidesResponse, LlmAdapter,
    LlmAdapterError, MergeConceptCandidatesRequest, MergeConceptCandidatesResponse, MergeGroup,
    RunHealthCheckRequest, RunHealthCheckResponse, SourceAnchor, SummarizeDocumentRequest,
    SummarizeDocumentResponse, TokenUsage,
};
use kb_llm::ProvenanceRecord;
use kb_core::Hash;
use tempfile::tempdir;

/// Stub adapter that returns a canned merge response and nothing else.
/// All other calls are unreachable — the `concept_merge` pass only invokes
/// `merge_concept_candidates` (plus the optional `generate_concept_body`,
/// for which the trait provides a "not implemented" default).
struct StubMergeAdapter {
    response: MergeConceptCandidatesResponse,
}

impl LlmAdapter for StubMergeAdapter {
    fn summarize_document(
        &self,
        _request: SummarizeDocumentRequest,
    ) -> Result<(SummarizeDocumentResponse, ProvenanceRecord), LlmAdapterError> {
        unreachable!("summarize_document not exercised by concept merge pass")
    }

    fn extract_concepts(
        &self,
        _request: ExtractConceptsRequest,
    ) -> Result<(ExtractConceptsResponse, ProvenanceRecord), LlmAdapterError> {
        unreachable!("extract_concepts not exercised by concept merge pass")
    }

    fn merge_concept_candidates(
        &self,
        _request: MergeConceptCandidatesRequest,
    ) -> Result<(MergeConceptCandidatesResponse, ProvenanceRecord), LlmAdapterError> {
        Ok((self.response.clone(), provenance()))
    }

    fn answer_question(
        &self,
        _request: AnswerQuestionRequest,
    ) -> Result<(AnswerQuestionResponse, ProvenanceRecord), LlmAdapterError> {
        unreachable!("answer_question not exercised by concept merge pass")
    }

    fn generate_slides(
        &self,
        _request: GenerateSlidesRequest,
    ) -> Result<(GenerateSlidesResponse, ProvenanceRecord), LlmAdapterError> {
        unreachable!("generate_slides not exercised by concept merge pass")
    }

    fn run_health_check(
        &self,
        _request: RunHealthCheckRequest,
    ) -> Result<(RunHealthCheckResponse, ProvenanceRecord), LlmAdapterError> {
        unreachable!("run_health_check not exercised by concept merge pass")
    }
}

fn provenance() -> ProvenanceRecord {
    ProvenanceRecord {
        harness: "test".to_string(),
        harness_version: None,
        model: "stub".to_string(),
        prompt_template_name: "merge_concept_candidates.md".to_string(),
        prompt_template_hash: Hash::from([0u8; 32]),
        prompt_render_hash: Hash::from([0u8; 32]),
        started_at: 0,
        ended_at: 1,
        latency_ms: 1,
        retries: 0,
        tokens: Some(TokenUsage {
            prompt_tokens: 10,
            completion_tokens: 5,
        }),
        cost_estimate: None,
    }
}

fn candidate(name: &str, hint: &str) -> ConceptCandidate {
    ConceptCandidate {
        name: name.to_string(),
        aliases: vec![],
        definition_hint: Some(hint.to_string()),
        source_anchors: vec![SourceAnchor {
            heading_anchor: None,
            quote: Some(format!("{name}: {hint}")),
        }],
    }
}

fn group(canonical: &str, category: Option<&str>, member: ConceptCandidate) -> MergeGroup {
    MergeGroup {
        canonical_name: canonical.to_string(),
        aliases: vec![],
        category: category.map(ToString::to_string),
        members: vec![member],
        confident: true,
        rationale: None,
    }
}

/// Count the number of `## <heading>` occurrences after the first top-level
/// `# Concepts` line. Used to assert the index has the expected bucket count.
fn category_headings(index: &str) -> Vec<String> {
    index
        .lines()
        .filter_map(|l| l.strip_prefix("## ").map(ToString::to_string))
        .collect()
}

#[test]
fn concept_frontmatter_carries_category_after_merge() {
    // Two concepts in "async", one in "ownership", one with no category
    // (which must land under "Uncategorized").
    let dir = tempdir().expect("tempdir");
    let root = dir.path();

    let tokio = candidate("Tokio runtime", "A multi-threaded async runtime.");
    let state_machine = candidate("async fn state machine", "State machine compile form.");
    let borrow_checker = candidate(
        "Borrow checker",
        "Validates references at compile time.",
    );
    let generic = candidate("Ambient concept", "A concept without a home.");

    let response = MergeConceptCandidatesResponse {
        groups: vec![
            group("Tokio runtime", Some("async"), tokio.clone()),
            group(
                "async fn state machine",
                Some("async"),
                state_machine.clone(),
            ),
            group(
                "Borrow checker",
                Some("ownership"),
                borrow_checker.clone(),
            ),
            group("Ambient concept", None, generic.clone()),
        ],
    };

    let adapter = StubMergeAdapter { response };
    let candidates = vec![tokio, state_machine, borrow_checker, generic];
    let artifact = run_concept_merge_pass(&adapter, candidates, root).expect("merge pass");

    assert_eq!(artifact.concept_pages.len(), 4);

    // Persist pages to disk so generate_indexes can pick them up.
    for page in &artifact.concept_pages {
        persist_concept_page(page).expect("persist concept page");
    }

    // Frontmatter contract: category is emitted when present, omitted when
    // None. This is the "after recompile, concept frontmatter has a
    // category: field" acceptance check.
    let find_page = |name: &str| -> &ConceptPage {
        artifact
            .concept_pages
            .iter()
            .find(|p| p.canonical_name == name)
            .unwrap_or_else(|| panic!("missing page for {name}"))
    };
    assert!(find_page("Tokio runtime").content.contains("category: async"));
    assert!(find_page("async fn state machine")
        .content
        .contains("category: async"));
    assert!(find_page("Borrow checker")
        .content
        .contains("category: ownership"));
    assert!(
        !find_page("Ambient concept").content.contains("category:"),
        "uncategorized concept must not emit a category: field; got:\n{}",
        find_page("Ambient concept").content
    );

    assert_eq!(find_page("Tokio runtime").category.as_deref(), Some("async"));
    assert_eq!(find_page("Ambient concept").category, None);
}

#[test]
fn concepts_index_groups_by_category_with_uncategorized_last() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();

    // Write concept pages directly — this also proves the renderer reads
    // `category:` out of existing frontmatter (not just straight from a
    // MergeGroup), matching the real recompile flow.
    let write = |slug: &str, name: &str, category: Option<&str>| {
        use std::fmt::Write as _;
        let path = root.join(format!("wiki/concepts/{slug}.md"));
        std::fs::create_dir_all(path.parent().expect("parent")).expect("create dir");
        let mut fm = String::from("---\n");
        writeln!(fm, "id: concept:{slug}").expect("write");
        writeln!(fm, "name: {name}").expect("write");
        if let Some(cat) = category {
            writeln!(fm, "category: {cat}").expect("write");
        }
        fm.push_str("---\n\n");
        writeln!(fm, "# {name}").expect("write");
        std::fs::write(&path, fm).expect("write concept");
    };
    write("tokio-runtime", "Tokio runtime", Some("async"));
    write("async-fn-state-machine", "async fn state machine", Some("async"));
    write("borrow-checker", "Borrow checker", Some("ownership"));
    write("ambient-concept", "Ambient concept", None);
    // Even a one-member category should get its own heading.
    write("raft", "Raft", Some("consensus"));

    let artifacts = generate_indexes(root).expect("generate");
    let concepts_index = artifacts
        .iter()
        .find(|a| a.path.ends_with("wiki/concepts/index.md"))
        .expect("concepts index artifact present");
    let text = &concepts_index.content;

    // Header records total count + number of categories (3 real + 1 Uncat).
    assert!(text.contains("# Concepts"));
    assert!(
        text.contains("5 page(s) in 4 categories."),
        "header count mismatch; got:\n{text}"
    );

    // Headings present, in the expected order:
    //   async (a…), consensus, ownership (alphabetical), then Uncategorized last.
    let headings = category_headings(text);
    assert_eq!(
        headings,
        vec![
            "async".to_string(),
            "consensus".to_string(),
            "ownership".to_string(),
            "Uncategorized".to_string(),
        ],
        "unexpected heading order; full index:\n{text}"
    );

    // Concepts are listed under their assigned category.
    assert_in_section(text, "async", "Tokio runtime");
    assert_in_section(text, "async", "async fn state machine");
    assert_in_section(text, "ownership", "Borrow checker");
    assert_in_section(text, "consensus", "Raft");
    assert_in_section(text, "Uncategorized", "Ambient concept");
}

#[test]
fn concepts_index_renders_only_uncategorized_when_no_categories() {
    // Silent-fallback path: a knowledge base where no concepts have a
    // category yet must still render a valid index — just one bucket.
    let dir = tempdir().expect("tempdir");
    let root = dir.path();

    let write = |slug: &str, name: &str| {
        let path = root.join(format!("wiki/concepts/{slug}.md"));
        std::fs::create_dir_all(path.parent().expect("parent")).expect("create dir");
        std::fs::write(
            &path,
            format!("---\nid: concept:{slug}\nname: {name}\n---\n\n# {name}\n"),
        )
        .expect("write");
    };
    write("alpha", "Alpha");
    write("beta", "Beta");

    let artifacts = generate_indexes(root).expect("generate");
    let concepts_index = artifacts
        .iter()
        .find(|a| a.path.ends_with("wiki/concepts/index.md"))
        .expect("concepts index");

    assert!(concepts_index.content.contains("2 page(s) in 1 category."));
    let headings = category_headings(&concepts_index.content);
    assert_eq!(headings, vec!["Uncategorized".to_string()]);
    assert!(concepts_index.content.contains("[Alpha]"));
    assert!(concepts_index.content.contains("[Beta]"));
}

/// Assert that `entry` appears in the markdown block under `## heading`
/// (before the next `## ` heading). This pins the concept into its
/// category bucket rather than just anywhere in the file.
fn assert_in_section(index: &str, heading: &str, entry: &str) {
    let needle = format!("## {heading}");
    let start = index
        .find(&needle)
        .unwrap_or_else(|| panic!("heading {needle:?} not found in:\n{index}"));
    // Find the next "\n## " after this heading, if any.
    let rest = &index[start + needle.len()..];
    let end = rest.find("\n## ").map_or(rest.len(), |i| i);
    let section = &rest[..end];
    assert!(
        section.contains(&format!("[{entry}]")),
        "expected [{entry}] under heading '{heading}'; section was:\n{section}"
    );
}

// Reference the Path type so clippy doesn't complain about unused import in
// some feature configurations.
#[allow(dead_code)]
const _PATH_USE: fn(&Path) = |_| {};
