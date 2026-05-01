//! Loader for `evals/golden.toml` — the per-kb golden Q/A set.
//!
//! Format:
//!
//! ```toml
//! [[query]]
//! id = "auth-paraphrase"
//! query = "how does authentication work?"
//! expected_sources = ["src-cred", "src-authn"]
//! expected_concepts = ["authentication"]
//! ```
//!
//! `expected_sources` are bare ids (without the `wiki/sources/` prefix or
//! the `.md` suffix); the runner matches them against retrieved
//! `item_id`s by substring. `expected_concepts` are matched against
//! `wiki/concepts/<slug>.md` item ids the same way.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// Top-level wrapper for `golden.toml` — a list of [`GoldenQuery`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenSet {
    /// One `[[query]]` table per golden Q/A entry. Required.
    #[serde(default)]
    pub query: Vec<GoldenQuery>,
}

/// A single golden query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenQuery {
    /// Stable id — used for diffs against baselines and for naming
    /// per-query rows in the markdown table.
    pub id: String,
    /// The natural-language query passed to `plan_retrieval_hybrid_with_backend`.
    pub query: String,
    /// Bare source ids (e.g. `src-cred`); matched via substring against
    /// retrieved `item_id`s.
    #[serde(default)]
    pub expected_sources: Vec<String>,
    /// Concept slugs; matched via substring against retrieved `item_id`s
    /// (which look like `wiki/concepts/<slug>.md`).
    #[serde(default)]
    pub expected_concepts: Vec<String>,
}

impl GoldenSet {
    /// Parse a `golden.toml` file at `path`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, the TOML is malformed,
    /// or any required field (`id` / `query`) is missing or empty.
    pub fn load(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("read golden set from {}", path.display()))?;
        let set = Self::from_toml_str(&raw)
            .with_context(|| format!("parse golden set at {}", path.display()))?;
        Ok(set)
    }

    /// Parse a `golden.toml` string. Validates that ids and queries are
    /// non-empty and that ids are unique within the set.
    ///
    /// # Errors
    ///
    /// Returns an error on TOML parse failures, empty fields, or
    /// duplicate query ids.
    pub fn from_toml_str(raw: &str) -> Result<Self> {
        let set: Self = toml::from_str(raw).context("parse golden.toml")?;
        if set.query.is_empty() {
            bail!("golden.toml must contain at least one [[query]] entry");
        }
        let mut seen = std::collections::HashSet::new();
        for q in &set.query {
            if q.id.trim().is_empty() {
                bail!("golden query is missing an `id` (or it is empty)");
            }
            if q.query.trim().is_empty() {
                bail!("golden query `{}` has an empty `query` field", q.id);
            }
            if !seen.insert(q.id.clone()) {
                bail!("duplicate golden query id: {}", q.id);
            }
            if q.expected_sources.is_empty() && q.expected_concepts.is_empty() {
                bail!(
                    "golden query `{}` has no expected_sources and no expected_concepts; \
                     at least one is required for scoring",
                    q.id
                );
            }
        }
        Ok(set)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_set() {
        let raw = r#"
[[query]]
id = "q1"
query = "what is foo?"
expected_sources = ["src-foo"]

[[query]]
id = "q2"
query = "tell me about bar"
expected_sources = ["src-bar"]
expected_concepts = ["bar"]
"#;
        let set = GoldenSet::from_toml_str(raw).expect("parse ok");
        assert_eq!(set.query.len(), 2);
        assert_eq!(set.query[0].id, "q1");
        assert_eq!(set.query[1].expected_concepts, vec!["bar".to_string()]);
    }

    #[test]
    fn round_trips() {
        let original = GoldenSet {
            query: vec![GoldenQuery {
                id: "q1".to_string(),
                query: "what is foo?".to_string(),
                expected_sources: vec!["src-foo".to_string()],
                expected_concepts: vec![],
            }],
        };
        let raw = toml::to_string(&original).expect("serialize");
        let parsed = GoldenSet::from_toml_str(&raw).expect("re-parse");
        assert_eq!(original, parsed);
    }

    #[test]
    fn rejects_empty_set() {
        let err = GoldenSet::from_toml_str("").expect_err("expected error");
        assert!(format!("{err}").contains("at least one"), "{err:#}");
    }

    #[test]
    fn rejects_empty_id() {
        let raw = r#"
[[query]]
id = ""
query = "what?"
expected_sources = ["s"]
"#;
        let err = GoldenSet::from_toml_str(raw).expect_err("expected error");
        assert!(format!("{err:#}").contains("id"), "{err:#}");
    }

    #[test]
    fn rejects_duplicate_ids() {
        let raw = r#"
[[query]]
id = "q1"
query = "a?"
expected_sources = ["s"]

[[query]]
id = "q1"
query = "b?"
expected_sources = ["s"]
"#;
        let err = GoldenSet::from_toml_str(raw).expect_err("expected error");
        assert!(format!("{err:#}").contains("duplicate"), "{err:#}");
    }

    #[test]
    fn rejects_query_without_expectations() {
        let raw = r#"
[[query]]
id = "q1"
query = "what?"
"#;
        let err = GoldenSet::from_toml_str(raw).expect_err("expected error");
        assert!(
            format!("{err:#}").contains("expected"),
            "expected scoring-target error, got {err:#}"
        );
    }

    #[test]
    fn rejects_malformed_toml() {
        let err = GoldenSet::from_toml_str("[[query\nid = ").expect_err("expected error");
        assert!(format!("{err:#}").to_lowercase().contains("parse"), "{err:#}");
    }
}
