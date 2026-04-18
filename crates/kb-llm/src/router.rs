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

/// Errors produced when a model cannot be routed to a backend.
#[derive(Debug, thiserror::Error)]
pub enum RouterError {
    /// Caller requested a Claude-family model but no `ClaudeCode` backend is configured.
    ///
    /// Anthropic's subscription terms forbid running Claude models through third-party
    /// harnesses, so the router refuses to fall back rather than silently violating those terms.
    #[error(
        "Claude-family model '{model}' requires the ClaudeCode backend, but it is not configured. \
         Add a [llm.runners.claude] section to kb.toml to enable it."
    )]
    ClaudeCodeRequired { model: String },
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
    ///
    /// # Errors
    ///
    /// Returns [`RouterError::ClaudeCodeRequired`] if a Claude-family model is requested
    /// but the `ClaudeCode` backend is not configured. Claude models may not route to
    /// any other backend (Anthropic's subscription terms); the router refuses to fall back silently.
    pub fn route_model(&self, model: &str) -> Result<Backend, RouterError> {
        if is_claude_model(model) {
            if self.configured_backends.contains(&Backend::ClaudeCode) {
                Ok(Backend::ClaudeCode)
            } else {
                Err(RouterError::ClaudeCodeRequired {
                    model: model.to_string(),
                })
            }
        } else if self.configured_backends.contains(&self.default_backend) {
            Ok(self.default_backend)
        } else {
            // Fall back to opencode only when the configured default is absent.
            Ok(Backend::Opencode)
        }
    }

    /// Alias kept for callers expecting `backend_for_model` naming.
    ///
    /// # Errors
    ///
    /// Same conditions as [`Router::route_model`].
    pub fn backend_for_model(&self, model: &str) -> Result<Backend, RouterError> {
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
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn routes_claude_family_to_claude_code() {
        let router = Router::with_backends(
            Backend::Opencode,
            [Backend::ClaudeCode, Backend::Opencode, Backend::Pi],
        );

        assert_eq!(
            router.route_model("claude-opus-4-7").unwrap(),
            Backend::ClaudeCode
        );
        assert_eq!(
            router.route_model("anthropic/claude-sonnet-4-6").unwrap(),
            Backend::ClaudeCode
        );
        assert_eq!(
            router.route_model("claude/haiku-3").unwrap(),
            Backend::ClaudeCode
        );
    }

    #[test]
    fn non_claude_models_use_configured_default() {
        let router = Router::with_backends(
            Backend::Opencode,
            [Backend::ClaudeCode, Backend::Opencode, Backend::Pi],
        );

        assert_eq!(
            router.route_model("openai/gpt-5.4").unwrap(),
            Backend::Opencode
        );
        assert_eq!(
            router.route_model("gemini-flash").unwrap(),
            Backend::Opencode
        );
        assert_eq!(
            router.route_model("openai/gpt-5.4").unwrap(),
            router.backend_for_model("openai/gpt-5.4").unwrap()
        );
    }

    #[test]
    fn non_claude_default_pi_when_configured() {
        let router = Router::with_backends(Backend::Pi, [Backend::ClaudeCode, Backend::Pi]);

        assert_eq!(router.route_model("openai/gpt-5.4").unwrap(), Backend::Pi);
    }

    #[test]
    fn routes_non_claude_to_opencode_when_default_missing() {
        let router = Router::with_backends(Backend::Opencode, [Backend::ClaudeCode]);

        assert_eq!(
            router.route_model("openai/gpt-5.4").unwrap(),
            Backend::Opencode
        );
    }

    #[test]
    fn claude_refuses_when_claude_code_not_configured() {
        let router = Router::with_backends(Backend::Opencode, [Backend::Opencode]);

        let err = router
            .route_model("claude-opus-4-7")
            .expect_err("must refuse claude without ClaudeCode backend");
        assert!(matches!(err, RouterError::ClaudeCodeRequired { .. }));
    }

    #[test]
    fn claude_requests_are_not_forced_to_non_claude_backend() {
        let router =
            Router::with_backends(Backend::Opencode, [Backend::ClaudeCode, Backend::Opencode]);

        assert_eq!(
            router.route_model("ANTHROPIC/CLAUDE-SONNET-4-6").unwrap(),
            Backend::ClaudeCode
        );
    }
}
