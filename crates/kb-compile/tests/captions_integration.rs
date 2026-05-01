//! Integration test: bn-2qda captions pass end-to-end through `run_compile_with_llm`.
//!
//! Wires up a stub vision-capable LLM adapter and confirms that:
//!   1. A wiki source page with `![](diagram.png)` gets a `kb:caption`
//!      managed region appended after the image.
//!   2. The caption text appears in the rendered lexical index so a
//!      subsequent `kb search` over the corpus would surface the page.
//!   3. Re-running the compile is a no-op for caption generation (the
//!      adapter is NOT called a second time when the cache file exists).

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use kb_compile::pipeline::{CompileOptions, run_compile_with_llm};
use kb_core::Hash;
use kb_llm::{
    AnswerQuestionRequest, AnswerQuestionResponse, ExtractConceptsRequest,
    ExtractConceptsResponse, GenerateSlidesRequest, GenerateSlidesResponse, LlmAdapter,
    LlmAdapterError, MergeConceptCandidatesRequest, MergeConceptCandidatesResponse,
    ProvenanceRecord, RunHealthCheckRequest, RunHealthCheckResponse, SummarizeDocumentRequest,
    SummarizeDocumentResponse, TokenUsage,
};
use tempfile::TempDir;

/// Stub adapter: implements only `caption_image`. Every other LLM call
/// returns a default empty response so the broader compile pipeline can run
/// to completion without reaching out to a real model.
struct StubVisionAdapter {
    caption: String,
    caption_calls: Mutex<usize>,
}

impl StubVisionAdapter {
    fn new(caption: &str) -> Self {
        Self {
            caption: caption.to_string(),
            caption_calls: Mutex::new(0),
        }
    }

    fn caption_call_count(&self) -> usize {
        *self.caption_calls.lock().unwrap()
    }
}

impl LlmAdapter for StubVisionAdapter {
    fn caption_image(
        &self,
        _path: &Path,
        _prompt: &str,
    ) -> Result<(String, ProvenanceRecord), LlmAdapterError> {
        *self.caption_calls.lock().unwrap() += 1;
        Ok((self.caption.clone(), provenance()))
    }

    fn summarize_document(
        &self,
        request: SummarizeDocumentRequest,
    ) -> Result<(SummarizeDocumentResponse, ProvenanceRecord), LlmAdapterError> {
        Ok((
            SummarizeDocumentResponse {
                summary: format!("Stub summary of {}", request.title),
            },
            provenance(),
        ))
    }

    fn extract_concepts(
        &self,
        _: ExtractConceptsRequest,
    ) -> Result<(ExtractConceptsResponse, ProvenanceRecord), LlmAdapterError> {
        Ok((
            ExtractConceptsResponse { concepts: vec![] },
            provenance(),
        ))
    }

    fn merge_concept_candidates(
        &self,
        _: MergeConceptCandidatesRequest,
    ) -> Result<(MergeConceptCandidatesResponse, ProvenanceRecord), LlmAdapterError> {
        Ok((
            MergeConceptCandidatesResponse { groups: vec![] },
            provenance(),
        ))
    }

    fn answer_question(
        &self,
        _: AnswerQuestionRequest,
    ) -> Result<(AnswerQuestionResponse, ProvenanceRecord), LlmAdapterError> {
        Ok((
            AnswerQuestionResponse {
                answer: String::new(),
                references: None,
            },
            provenance(),
        ))
    }

    fn generate_slides(
        &self,
        _: GenerateSlidesRequest,
    ) -> Result<(GenerateSlidesResponse, ProvenanceRecord), LlmAdapterError> {
        Ok((
            GenerateSlidesResponse {
                slides: String::new(),
                slide_count: 0,
            },
            provenance(),
        ))
    }

    fn run_health_check(
        &self,
        _: RunHealthCheckRequest,
    ) -> Result<(RunHealthCheckResponse, ProvenanceRecord), LlmAdapterError> {
        Ok((
            RunHealthCheckResponse {
                status: "healthy".to_string(),
                details: None,
            },
            provenance(),
        ))
    }
}

fn provenance() -> ProvenanceRecord {
    ProvenanceRecord {
        harness: "stub".to_string(),
        harness_version: None,
        model: "stub-vision".to_string(),
        prompt_template_name: "caption_image".to_string(),
        prompt_template_hash: Hash::from([0u8; 32]),
        prompt_render_hash: Hash::from([0u8; 32]),
        started_at: 1,
        ended_at: 2,
        latency_ms: 1,
        retries: 0,
        tokens: Some(TokenUsage {
            prompt_tokens: 10,
            completion_tokens: 10,
        }),
        cost_estimate: None,
    }
}

/// Set up a minimal kb root with one wiki source page that references a
/// real PNG byte-blob. Returns the temp dir holder + paths.
fn setup_kb_with_diagram() -> (TempDir, PathBuf, PathBuf) {
    let dir = TempDir::new().expect("tempdir");
    let root = dir.path().to_path_buf();
    std::fs::create_dir_all(root.join("wiki/sources")).unwrap();
    std::fs::create_dir_all(root.join(".kb/state")).unwrap();
    std::fs::create_dir_all(root.join(".kb/cache/captions")).unwrap();

    // Image: tiny but real bytes so the content hash is deterministic.
    let img_path = root.join("wiki/sources/diagram.png");
    std::fs::write(&img_path, b"\x89PNG\r\n\x1a\n test bytes for hash").unwrap();

    let page_path = root.join("wiki/sources/example.md");
    let page_body = "---\n\
page_id: wiki-source-example\n\
type: source\n\
---\n\
# Source\n\
<!-- kb:begin id=title -->\n\
Example source page\n\
<!-- kb:end id=title -->\n\
\n\
## Body\n\
\n\
![](diagram.png)\n\
\n\
Surrounding prose so the body isn't only the image ref.\n\
";
    std::fs::write(&page_path, page_body).unwrap();

    (dir, page_path, img_path)
}

#[test]
fn captions_pass_injects_block_and_appears_in_lexical_index() {
    let (dir, page_path, img_path) = setup_kb_with_diagram();
    let root = dir.path();

    let adapter = StubVisionAdapter::new(
        "An architecture diagram showing service A talking to service B.",
    );
    let options = CompileOptions {
        captions: kb_compile::captions::CaptionsConfig {
            enabled: true,
            allow_paths: vec!["wiki/".to_string()],
            prompt: kb_compile::captions::DEFAULT_PROMPT.to_string(),
        },
        ..CompileOptions::default()
    };

    let report = run_compile_with_llm(root, &options, Some(&adapter)).expect("compile");

    // Captions pass should have run.
    let caption_pass_status = report
        .passes
        .iter()
        .find(|(name, _)| name == "captions")
        .expect("captions pass missing from report");
    eprintln!("captions pass: {caption_pass_status:?}");

    // Verify the page now carries a `kb:caption` managed region.
    let updated = std::fs::read_to_string(&page_path).expect("read page");
    let img_bytes = std::fs::read(&img_path).expect("read image");
    let hash = kb_core::hash_bytes(&img_bytes).to_hex();
    assert!(
        updated.contains(&format!("<!-- kb:begin id=caption-{hash} -->")),
        "page must carry caption fence; got:\n{updated}"
    );
    assert!(
        updated.contains("An architecture diagram showing service A talking to service B."),
        "page must carry caption text; got:\n{updated}"
    );

    // Adapter was called exactly once (cache miss on the first compile).
    assert_eq!(adapter.caption_call_count(), 1);

    // Verify the lexical index picked up the caption text. The caption
    // sits inside a managed region in the page body, so the lexical index
    // — which indexes page bodies — should include it.
    let lex_path = kb_query::lexical_index_path(root);
    let lex_raw = std::fs::read_to_string(&lex_path).expect("read lexical index");
    assert!(
        lex_raw.contains("architecture diagram") || lex_raw.contains("service A"),
        "lexical index should carry caption tokens; got:\n{lex_raw}"
    );

    // Re-run: captions pass must be a no-op (cache hit).
    let report2 = run_compile_with_llm(root, &options, Some(&adapter)).expect("re-compile");
    let _ = report2;
    assert_eq!(
        adapter.caption_call_count(),
        1,
        "second compile must NOT re-call the vision adapter"
    );
}
