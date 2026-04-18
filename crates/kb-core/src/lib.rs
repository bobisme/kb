#![forbid(unsafe_code)]

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Shared status model for all domain entities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    #[default]
    Fresh,
    Stale,
    Failed,
    NeedsReview,
}

/// Canonical identifier type for all entities in the KB.
pub type EntityId = String;

/// A reusable hash representation for immutable content.
pub type ContentHash = String;

/// Common provenance metadata included on every persisted entity.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct EntityMetadata {
    /// Stable entity ID.
    pub id: EntityId,

    /// Creation time since the Unix epoch in milliseconds.
    pub created_at_millis: u64,

    /// Last update time since the Unix epoch in milliseconds.
    pub updated_at_millis: u64,

    /// One or more content hashes that support freshness and provenance checks.
    pub source_hashes: Vec<ContentHash>,

    /// Version of the model used to produce this entity, where applicable.
    pub model_version: Option<String>,

    /// Version of tooling used to produce this entity.
    pub tool_version: Option<String>,

    /// Hash of prompt template used, where applicable.
    pub prompt_template_hash: Option<ContentHash>,

    /// IDs of entities this item depends on.
    pub dependencies: Vec<EntityId>,

    /// Files produced or consumed by this entity.
    pub output_paths: Vec<PathBuf>,

    /// Entity freshness state.
    pub status: Status,
}

/// A source that has logical identity across refetches (e.g. a URL, file path, repo item).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceDocument {
    pub metadata: EntityMetadata,
    pub source_kind: SourceKind,
    pub stable_location: String,
    pub discovered_at_millis: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    File,
    Url,
    Repo,
    Image,
    Dataset,
    #[default]
    Other,
}

/// A single immutable fetched revision of a source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRevision {
    pub metadata: EntityMetadata,
    pub source_document_id: EntityId,
    pub fetched_revision_hash: ContentHash,
    pub fetched_path: PathBuf,
    pub fetched_size_bytes: u64,
    pub fetched_at_millis: u64,
}

/// A normalized representation derived from one source revision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizedDocument {
    pub metadata: EntityMetadata,
    pub source_revision_id: EntityId,
    pub canonical_text: String,
    pub normalized_assets: Vec<PathBuf>,
    pub heading_ids: Vec<String>,
}

/// A generated wiki page with section-level provenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WikiPage {
    pub metadata: EntityMetadata,
    pub title: String,
    pub source_revision_ids: Vec<EntityId>,
    pub source_sections: Vec<String>,
    pub generated_from: Option<EntityId>,
}

/// A canonical topic node with aliases/backlinks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Concept {
    pub metadata: EntityMetadata,
    pub name: String,
    pub aliases: Vec<String>,
    pub wiki_page_id: Option<EntityId>,
    pub backlinks: Vec<EntityId>,
}

/// A question asked by a user, with retrieval plan metadata persisted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Question {
    pub metadata: EntityMetadata,
    pub prompt: String,
    pub requested_format: String,
    pub retrieval_plan: String,
    pub token_budget: Option<u32>,
}

/// Any generated output artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Artifact {
    pub metadata: EntityMetadata,
    pub question_id: EntityId,
    pub artifact_kind: ArtifactKind,
    pub format: String,
    pub output_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Report,
    Figure,
    SlideDeck,
    AnswerNote,
    JsonSpec,
    Other,
}

/// A citation/claim locator linked to a specific source revision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Citation {
    pub metadata: EntityMetadata,
    pub source_revision_id: EntityId,
    pub claim_text: Option<String>,
    pub heading_anchor: Option<String>,
    pub line_span: Option<LineSpan>,
    pub char_span: Option<CharSpan>,
    pub page_number: Option<u32>,
    pub asset_ref: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineSpan {
    pub start_line: u32,
    pub end_line: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CharSpan {
    pub start_char: u32,
    pub end_char: u32,
}

/// Tracks which inputs produced which outputs and with which hashes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildRecord {
    pub metadata: EntityMetadata,
    pub input_ids: Vec<EntityId>,
    pub output_ids: Vec<EntityId>,
    pub manifest_hash: ContentHash,
}

/// One execution of compile/ask/lint/publish with logs and status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobRun {
    pub metadata: EntityMetadata,
    pub command: String,
    pub root_path: PathBuf,
    pub started_at_millis: u64,
    pub ended_at_millis: Option<u64>,
    pub status: JobRunStatus,
    pub log_path: Option<PathBuf>,
    pub affected_outputs: Vec<PathBuf>,
    pub pid: Option<u32>,
    pub exit_code: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobRunStatus {
    Running,
    Succeeded,
    Failed,
    Interrupted,
}

/// A pending machine-prepared item awaiting review.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReviewItem {
    pub metadata: EntityMetadata,
    pub target_entity_id: EntityId,
    pub action: ReviewAction,
    pub comment: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewAction {
    Promote,
    Merge,
    Canonicalize,
    Archive,
    Retry,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip<T>(entity: &T)
    where
        T: Serialize + for<'a> Deserialize<'a> + PartialEq + std::fmt::Debug,
    {
        let json = serde_json::to_string_pretty(entity).expect("serialize");
        let decoded: T = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(*entity, decoded);
    }

    fn metadata(id: &str) -> EntityMetadata {
        EntityMetadata {
            id: id.to_string(),
            created_at_millis: 1_700_000_000_000,
            updated_at_millis: 1_700_000_000_500,
            source_hashes: vec!["hash-abc-1".to_string()],
            model_version: Some("gpt-4o-mini".to_string()),
            tool_version: Some("kb-tool/0.1".to_string()),
            prompt_template_hash: Some("tmpl-hash-1".to_string()),
            dependencies: vec!["dep-id".to_string()],
            output_paths: vec![PathBuf::from("out/a.json")],
            status: Status::Fresh,
        }
    }

    #[test]
    fn domain_entities_round_trip() {
        round_trip(&SourceDocument {
            metadata: metadata("source-document-1"),
            source_kind: SourceKind::Url,
            stable_location: "https://example.com/doc".to_string(),
            discovered_at_millis: 1_700_000_000_100,
        });

        round_trip(&SourceRevision {
            metadata: metadata("source-revision-1"),
            source_document_id: "source-document-1".to_string(),
            fetched_revision_hash: "revhash-1".to_string(),
            fetched_path: PathBuf::from("raw/example.html"),
            fetched_size_bytes: 1234,
            fetched_at_millis: 1_700_000_000_110,
        });

        round_trip(&NormalizedDocument {
            metadata: metadata("normalized-doc-1"),
            source_revision_id: "source-revision-1".to_string(),
            canonical_text: "normalized body".to_string(),
            normalized_assets: vec![PathBuf::from("normalized/assets/example.txt")],
            heading_ids: vec!["intro".to_string(), "details".to_string()],
        });

        round_trip(&WikiPage {
            metadata: metadata("wiki-page-1"),
            title: "Example Page".to_string(),
            source_revision_ids: vec!["source-revision-1".to_string()],
            source_sections: vec!["intro".to_string(), "summary".to_string()],
            generated_from: Some("normalized-doc-1".to_string()),
        });

        round_trip(&Concept {
            metadata: metadata("concept-1"),
            name: "Example Concept".to_string(),
            aliases: vec!["example".to_string(), "sample".to_string()],
            wiki_page_id: Some("wiki-page-1".to_string()),
            backlinks: vec!["concept-2".to_string()],
        });

        round_trip(&Question {
            metadata: metadata("question-1"),
            prompt: "How does this work?".to_string(),
            requested_format: "markdown".to_string(),
            retrieval_plan: "top-k: 10".to_string(),
            token_budget: Some(1024),
        });

        round_trip(&Artifact {
            metadata: metadata("artifact-1"),
            question_id: "question-1".to_string(),
            artifact_kind: ArtifactKind::Report,
            format: "md".to_string(),
            output_path: PathBuf::from("outputs/artifact.md"),
        });

        round_trip(&Citation {
            metadata: metadata("citation-1"),
            source_revision_id: "source-revision-1".to_string(),
            claim_text: Some("Claim about source".to_string()),
            heading_anchor: Some("overview".to_string()),
            line_span: Some(LineSpan {
                start_line: 1,
                end_line: 3,
            }),
            char_span: Some(CharSpan {
                start_char: 10,
                end_char: 25,
            }),
            page_number: Some(2),
            asset_ref: Some(PathBuf::from("assets/figure-1.png")),
        });

        round_trip(&BuildRecord {
            metadata: metadata("build-record-1"),
            input_ids: vec!["normalized-doc-1".to_string()],
            output_ids: vec!["wiki-page-1".to_string()],
            manifest_hash: "build-manifest-hash".to_string(),
        });

        round_trip(&JobRun {
            metadata: metadata("job-run-1"),
            command: "kb compile".to_string(),
            root_path: PathBuf::from("/tmp/kb"),
            started_at_millis: 1_700_000_000_120,
            ended_at_millis: Some(1_700_000_000_150),
            status: JobRunStatus::Succeeded,
            log_path: Some(PathBuf::from("logs/job.log")),
            affected_outputs: vec![PathBuf::from("wiki/index.md")],
            pid: Some(12345),
            exit_code: Some(0),
        });

        round_trip(&ReviewItem {
            metadata: metadata("review-item-1"),
            target_entity_id: "artifact-1".to_string(),
            action: ReviewAction::Promote,
            comment: "Looks good for promotion".to_string(),
        });
    }
}

pub mod frontmatter;
pub mod fs;
pub mod hashing;
pub mod managed_region;
pub mod normalized;
pub mod source_identity;

pub use hashing::{Hash, hash_bytes, hash_file, hash_many};
pub use managed_region::{
    ManagedRegion, extract_managed_regions, rewrite_managed_region, slug_from_title,
};
pub use normalized::{read_normalized_document, write_normalized_document};
pub use source_identity::{
    SOURCE_DOCUMENT_ID_PREFIX, SOURCE_REVISION_ID_PREFIX, mint_source_document_id,
    mint_source_revision_id, normalize_file_stable_location, normalize_url_stable_location,
    source_document_id_for_file, source_document_id_for_url, source_revision_content_hash,
};
