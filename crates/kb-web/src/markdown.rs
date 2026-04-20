//! Markdown → HTML via `pulldown_cmark`.
//!
//! We enable tables, footnotes, strikethrough, and task lists — enough to
//! render a kb wiki faithfully without needing a full markdown runtime.
//! The output is assumed to be rendered inside the wiki shell template, so
//! this function returns only the rendered `<article>`-style body.

use pulldown_cmark::{Options, Parser, html};

/// Render a markdown string into an HTML fragment.
#[must_use]
pub fn render(source: &str) -> String {
    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_FOOTNOTES);
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TASKLISTS);

    let parser = Parser::new_ext(source, options);
    let mut out = String::with_capacity(source.len() * 2);
    html::push_html(&mut out, parser);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_headings_and_emphasis() {
        let html = render("# Hello\n\n**world**\n");
        assert!(html.contains("<h1>Hello</h1>"));
        assert!(html.contains("<strong>world</strong>"));
    }

    #[test]
    fn renders_tables() {
        let html = render("| a | b |\n|---|---|\n| 1 | 2 |\n");
        assert!(html.contains("<table>"));
        assert!(html.contains("<td>1</td>"));
    }

    #[test]
    fn renders_code_blocks() {
        let html = render("```rust\nfn main() {}\n```\n");
        assert!(html.contains("<pre><code"));
        assert!(html.contains("fn main()"));
    }
}
