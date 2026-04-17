# Tiny Fixture Corpus

Small synthetic corpus for end-to-end CLI tests. Each file covers edge cases relevant to KB ingestion and compilation.

## Files

- **basic.md** - Simple document with standard Markdown formatting
- **unicode.md** - Tests unicode handling (emoji, accents, CJK characters)
- **anchors.md** - Document with heading anchors and cross-references
- **long-tokens.md** - Document with very long tokens (URLs, hashes, identifiers)
- **images.md** - Document with image references (linked to placeholder.png)
- **sample.html** - Simple HTML page for format diversity
- **placeholder.png** - Minimal PNG image (1x1 transparent pixel)

## Usage

These fixtures are used in snapshot tests to verify that:
- Files are correctly ingested into a KB
- Document metadata is extracted properly
- The CLI handles various file types without errors
