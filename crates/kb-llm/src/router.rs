use std::collections::HashSet;

/// A configured backend target for LLM requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Backend {
    /// Claude Code CLI backend.
    ClaudeCode,
    #[default]
    /// `opencode run` backend.
    Opencode,
    /// Optional `pi` backend (present when configured).
    Pi,
}

impl Backend {
    /// Human-readable backend name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude",
            Self::Opencode => "opencode",
            Self::Pi => "pi",
        }
    }
}

/// A backwards-compatible alias for the model-to-backend router.
pub type BackendRouter = Router;

/// Routes model requests to a backend based on model family.
#[derive(Debug, Clone)]
pub struct Router {
    default_backend: Backend,
    configured_backends: HashSet<Backend>,
}

impl Router {
    /// Create a router with a single default backend configured.
    #[must_use]
    pub fn new(default_backend: Backend) -> Self {
        Self::with_backends(default_backend, [default_backend])
    }

    /// Create a router with explicit configured backends and default fallback.
    #[must_use]
    pub fn with_backends(
        default_backend: Backend,
        configured_backends: impl IntoIterator<Item = Backend>,
    ) -> Self {
        let backends: HashSet<_> = configured_backends.into_iter().collect();
        let mut normalized = HashSet::with_capacity(backends.len() + 1);
        normalized.extend(backends);
        normalized.insert(default_backend);

        Self {
            default_backend,
            configured_backends: normalized,
        }
    }

    /// Return the configured default backend.
    #[must_use]
    pub const fn default_backend(&self) -> Backend {
        self.default_backend
    }

    /// Return the backend selected for the provided model identifier.
    #[must_use]
    pub fn route_model(&self, model: &str) -> Backend {
        if is_claude_model(model) {
            // Claude family models MUST go to Claude Code regardless of caller preference.
            if self.configured_backends.contains(&Backend::ClaudeCode) {
                Backend::ClaudeCode
            } else {
                self.default_backend
            }
        } else {
            // For non-Claude models, prefer pi when configured, otherwise default.
            if self.configured_backends.contains(&Backend::Pi) {
                Backend::Pi
            } else if self.configured_backends.contains(&self.default_backend) {
                self.default_backend
            } else {
                Backend::Opencode
            }
        }
    }

    /// Alias kept for callers expecting `backend_for_model` naming.
    #[must_use]
    pub fn backend_for_model(&self, model: &str) -> Backend {
        self.route_model(model)
    }
}

fn is_claude_model(model: &str) -> bool {
    let model = model.trim().to_ascii_lowercase();

    if model.is_empty() {
        return false;
    }

    model.starts_with("claude-") || model.starts_with("anthropic/") || model.starts_with("claude/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_claude_family_to_claude_code() {
        let router = Router::with_backends(
            Backend::Opencode,
            [Backend::ClaudeCode, Backend::Opencode, Backend::Pi],
        );

        assert_eq!(router.route_model("claude-opus-4-7"), Backend::ClaudeCode);
        assert_eq!(
            router.route_model("anthropic/claude-sonnet-4-6"),
            Backend::ClaudeCode
        );
        assert_eq!(router.route_model("claude/haiku-3"), Backend::ClaudeCode);
    }

    #[test]
    fn routes_non_claude_to_pi_when_configured() {
        let router = Router::with_backends(
            Backend::Opencode,
            [Backend::ClaudeCode, Backend::Opencode, Backend::Pi],
        );

        assert_eq!(router.route_model("openai/gpt-5.4"), Backend::Pi);
        assert_eq!(router.route_model("gemini-flash"), Backend::Pi);
        assert_eq!(
            router.route_model("openai/gpt-5.4"),
            router.backend_for_model("openai/gpt-5.4")
        );
    }

    #[test]
    fn routes_non_claude_to_opencode_when_pi_not_configured() {
        let router =
            Router::with_backends(Backend::Opencode, [Backend::ClaudeCode, Backend::Opencode]);

        assert_eq!(router.route_model("openai/gpt-5.4"), Backend::Opencode);
        assert_eq!(router.route_model("gemini-2.5-pro"), Backend::Opencode);
    }

    #[test]
    fn claude_requests_are_not_forced_to_non_claude_backend() {
        let router =
            Router::with_backends(Backend::Opencode, [Backend::ClaudeCode, Backend::Opencode]);

        assert_eq!(
            router.route_model("ANTHROPIC/CLAUDE-SONNET-4-6"),
            Backend::ClaudeCode
        );
    }
}
