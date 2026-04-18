use serde::{Deserialize, Serialize};

use kb_core::Hash;

/// Detailed token accounting reported by an LLM backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TokenUsage {
    /// Number of tokens in the input prompt.
    pub prompt_tokens: u32,
    /// Number of tokens generated in the completion.
    pub completion_tokens: u32,
}

impl TokenUsage {
    /// Total tokens consumed by the request.
    #[must_use]
    pub const fn total_tokens(&self) -> u32 {
        self.prompt_tokens + self.completion_tokens
    }
}

/// Provenance metadata for every LLM backend response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ProvenanceRecord {
    /// Harness/provider used to execute the request (e.g. "opencode", "claude").
    pub harness: String,

    /// Harness version if discoverable.
    pub harness_version: Option<String>,

    /// Model identifier used for the request.
    pub model: String,

    /// Template file name used for the prompt.
    pub prompt_template_name: String,

    /// Hash of the source prompt template.
    pub prompt_template_hash: Hash,

    /// Hash of the rendered prompt text (with variable interpolation).
    pub prompt_render_hash: Hash,

    /// Request start time in milliseconds since Unix epoch.
    pub started_at: u64,

    /// Request end time in milliseconds since Unix epoch.
    pub ended_at: u64,

    /// Total end-to-end latency in milliseconds.
    pub latency_ms: u64,

    /// Number of retry attempts performed by the adapter/runner.
    pub retries: u32,

    /// Token usage if reported by the harness, omitted as null in JSON when unavailable.
    pub tokens: Option<TokenUsage>,

    /// Optional cost estimate for the request, if available from the harness or policy.
    pub cost_estimate: Option<f64>,
}

/// Generic response wrapper for backend calls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct BackendResponse<Payload> {
    /// Backend-specific payload.
    pub payload: Payload,

    /// Execution provenance for the call that produced this payload.
    pub provenance: ProvenanceRecord,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_usage_totals_are_computed() {
        let usage = TokenUsage {
            prompt_tokens: 12,
            completion_tokens: 8,
        };

        assert_eq!(usage.total_tokens(), 20);
    }

    #[test]
    fn provenance_record_round_trips_with_missing_fields() {
        let hash = Hash::from([0u8; 32]);
        let record = ProvenanceRecord {
            harness: "opencode".to_string(),
            harness_version: None,
            model: "openai/gpt-5.4".to_string(),
            prompt_template_name: "ask.md".to_string(),
            prompt_template_hash: hash,
            prompt_render_hash: hash,
            started_at: 10,
            ended_at: 60,
            latency_ms: 50,
            retries: 2,
            tokens: None,
            cost_estimate: None,
        };

        let json = serde_json::to_string_pretty(&record).expect("serialize provenance record");
        assert!(json.contains("\"tokens\": null") || json.contains("\"tokens\":null"));
        assert!(
            json.contains("\"cost_estimate\": null") || json.contains("\"cost_estimate\":null")
        );

        let decoded: ProvenanceRecord =
            serde_json::from_str(&json).expect("deserialize provenance record");
        assert_eq!(decoded, record);
    }
}
