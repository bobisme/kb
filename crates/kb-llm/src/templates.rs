use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use regex::Regex;

use kb_core::{Hash, hash_bytes};

/// A loaded prompt template with metadata.
#[derive(Debug, Clone)]
pub struct Template {
    /// Name of the template file (e.g., "ask.md").
    pub name: String,
    /// Raw template content before variable substitution.
    pub content: String,
    /// Hash of the raw template content.
    pub template_hash: Hash,
}

impl Template {
    /// Load a template from a project path, falling back to bundled default.
    ///
    /// # Errors
    /// Returns an error if the template cannot be read or no bundled default is available.
    pub fn load(name: &str, project_root: Option<&Path>) -> Result<Self> {
        let content = if let Some(root) = project_root {
            let path = root.join("prompts").join(name);
            if path.exists() {
                fs::read_to_string(&path)
                    .with_context(|| format!("failed to read template {}", path.display()))?
            } else {
                Self::bundled(name)?
            }
        } else {
            Self::bundled(name)?
        };

        let template_hash = hash_bytes(content.as_bytes());

        Ok(Self {
            name: name.to_string(),
            content,
            template_hash,
        })
    }

    /// Retrieve a bundled default template by name.
    fn bundled(name: &str) -> Result<String> {
        match name {
            "ask.md" => Ok(include_str!("../prompts/ask.md").to_string()),
            "compile.md" => Ok(include_str!("../prompts/compile.md").to_string()),
            "summarize_document.md" => {
                Ok(include_str!("../prompts/summarize_document.md").to_string())
            }
            _ => Err(anyhow!(
                "template '{name}' not found in project and no bundled default available"
            )),
        }
    }

    /// Render template by substituting `{{var}}` placeholders with values from context.
    /// Missing variables are replaced with empty strings.
    ///
    /// # Errors
    /// Never fails; regex compilation is guaranteed to succeed.
    pub fn render(&self, context: &HashMap<String, String>) -> Result<RenderedTemplate> {
        let rendered = Self::substitute_variables(&self.content, context);
        let render_hash = hash_bytes(rendered.as_bytes());

        Ok(RenderedTemplate {
            content: rendered,
            render_hash,
        })
    }

    /// Replace `{{variable}}` patterns with values from context.
    /// Pattern: `{{word characters only}}`, e.g., `{{var_name}}`, `{{count}}`.
    fn substitute_variables(template: &str, context: &HashMap<String, String>) -> String {
        let re = Regex::new(r"\{\{(\w+)\}\}").expect("valid regex");
        re.replace_all(template, |caps: &regex::Captures| {
            let var_name = &caps[1];
            context.get(var_name).cloned().unwrap_or_default()
        })
        .to_string()
    }

    /// Load and prepend a persona to the rendered template.
    ///
    /// # Errors
    /// Returns an error if the persona file cannot be read or does not exist.
    pub fn with_persona(
        rendered: RenderedTemplate,
        persona_name: Option<&str>,
        project_root: Option<&Path>,
    ) -> Result<RenderedTemplate> {
        if let Some(name) = persona_name {
            let persona_content = Self::load_persona(name, project_root)?;
            let combined = format!("{}\n\n---\n\n{}", persona_content, rendered.content);
            let render_hash = hash_bytes(combined.as_bytes());

            Ok(RenderedTemplate {
                content: combined,
                render_hash,
            })
        } else {
            Ok(rendered)
        }
    }

    /// Load a persona file from project or bundled defaults.
    fn load_persona(name: &str, project_root: Option<&Path>) -> Result<String> {
        if let Some(root) = project_root {
            let path = root.join("prompts").join("personas").join(name);
            if path.exists() {
                return fs::read_to_string(&path)
                    .with_context(|| format!("failed to read persona {}", path.display()));
            }
        }

        match name {
            "researcher" => Ok(include_str!("../prompts/personas/researcher.md").to_string()),
            "analyst" => Ok(include_str!("../prompts/personas/analyst.md").to_string()),
            _ => Err(anyhow!(
                "persona '{name}' not found in project and no bundled default available"
            )),
        }
    }
}

/// A template after variable substitution.
#[derive(Debug, Clone)]
pub struct RenderedTemplate {
    /// Rendered content with variables substituted and persona prepended (if any).
    pub content: String,
    /// Hash of the fully rendered content.
    pub render_hash: Hash,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_substitution() {
        let template = Template {
            name: "test.md".to_string(),
            content: "Hello {{name}}, you have {{count}} messages.".to_string(),
            template_hash: hash_bytes(b"test"),
        };

        let mut context = HashMap::new();
        context.insert("name".to_string(), "Alice".to_string());
        context.insert("count".to_string(), "5".to_string());

        let rendered = template.render(&context).expect("render");
        assert_eq!(rendered.content, "Hello Alice, you have 5 messages.");
    }

    #[test]
    fn test_missing_variables_render_empty() {
        let template = Template {
            name: "test.md".to_string(),
            content: "Hello {{name}}, you have {{count}} messages.".to_string(),
            template_hash: hash_bytes(b"test"),
        };

        let context = HashMap::new();
        let rendered = template.render(&context).expect("render");
        assert_eq!(rendered.content, "Hello , you have  messages.");
    }

    #[test]
    fn test_template_hashing() {
        let content = "test content";
        let template = Template {
            name: "test.md".to_string(),
            content: content.to_string(),
            template_hash: hash_bytes(content.as_bytes()),
        };

        assert_eq!(template.template_hash, hash_bytes(content.as_bytes()));
    }

    #[test]
    fn test_render_hash_stability() {
        let template = Template {
            name: "test.md".to_string(),
            content: "Hello {{name}}".to_string(),
            template_hash: hash_bytes(b"test"),
        };

        let mut context = HashMap::new();
        context.insert("name".to_string(), "Bob".to_string());

        let rendered1 = template.render(&context).expect("render");
        let rendered2 = template.render(&context).expect("render");

        assert_eq!(rendered1.render_hash, rendered2.render_hash);
    }
}
