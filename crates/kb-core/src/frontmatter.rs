use std::fs;
use std::io;
use std::path::Path;

use serde_yaml::Mapping;

use crate::fs::atomic_write;

const FRONTMATTER_OPEN: &str = "---";

/// Reads a YAML frontmatter block from a markdown-like file.
///
/// Returns a tuple of `(frontmatter, body)`.
///
/// If no frontmatter block is present, returns an empty mapping and the full
/// file contents as body.
///
/// # Errors
/// Returns an error if reading from disk fails or if frontmatter content cannot be
/// parsed as YAML.
pub fn read_frontmatter(path: impl AsRef<Path>) -> io::Result<(Mapping, String)> {
    let content = fs::read_to_string(path)?;
    parse_frontmatter(&content)
}

/// Writes `frontmatter` and `body` as a YAML frontmatter document.
///
/// # Errors
/// Returns an error if YAML serialization fails or if writing to disk fails.
pub fn write_frontmatter(
    path: impl AsRef<Path>,
    frontmatter: &Mapping,
    body: impl AsRef<str>,
) -> io::Result<()> {
    let yaml = serde_yaml::to_string(frontmatter).map_err(io_error_invalid_data)?;

    let mut output = String::new();
    output.push_str(FRONTMATTER_OPEN);
    output.push('\n');
    output.push_str(&yaml);
    output.push_str(FRONTMATTER_OPEN);
    output.push('\n');
    output.push_str(body.as_ref());

    atomic_write(path, output.as_bytes())
}

fn parse_frontmatter(content: &str) -> io::Result<(Mapping, String)> {
    let mut lines = content.split_inclusive('\n');
    let Some(first_line) = lines.next() else {
        return Ok((Mapping::new(), String::new()));
    };

    if first_line != "---\n" && first_line != "---\r\n" && first_line != FRONTMATTER_OPEN {
        return Ok((Mapping::new(), content.to_string()));
    }

    let mut frontmatter_text = String::new();
    let mut offset = first_line.len();

    for line in lines {
        if line == "---\n" || line == "---\r\n" || line == FRONTMATTER_OPEN {
            let body = content[offset + line.len()..].to_string();
            let frontmatter =
                serde_yaml::from_str(&frontmatter_text).map_err(io_error_invalid_data)?;

            return Ok((frontmatter, body));
        }

        frontmatter_text.push_str(line);
        offset += line.len();
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "frontmatter block was not terminated with a closing --- line",
    ))
}

fn io_error_invalid_data<E: std::fmt::Display>(err: E) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err.to_string())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_yaml::Value;
    use std::fs;
    use tempfile::TempDir;

    fn mapping_from_pairs(pairs: &[(&str, Value)]) -> Mapping {
        let mut map = Mapping::new();
        for (key, value) in pairs {
            map.insert(Value::String((*key).to_string()), value.clone());
        }
        map
    }

    fn read_file(path: impl AsRef<Path>) -> io::Result<String> {
        fs::read_to_string(path).map_err(|err| io::Error::new(err.kind(), err.to_string()))
    }

    #[test]
    fn read_frontmatter_extracts_metadata_and_body() -> io::Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("with_frontmatter.md");

        let input = "---\n\
id: example\n\
status: fresh\n\
custom_note: keep-me\n\
---\n\
Body line\n\
second line\n";

        fs::write(&path, input)?;

        let (frontmatter, body) = read_frontmatter(&path)?;

        assert_eq!(
            frontmatter.get("id"),
            Some(&Value::String("example".into()))
        );
        assert_eq!(
            frontmatter.get("status"),
            Some(&Value::String("fresh".into()))
        );
        assert_eq!(
            frontmatter.get("custom_note"),
            Some(&Value::String("keep-me".into())),
        );
        assert_eq!(body, "Body line\nsecond line\n");

        Ok(())
    }

    #[test]
    fn unknown_keys_are_preserved_on_round_trip() -> io::Result<()> {
        let tmp = TempDir::new()?;
        let input_path = tmp.path().join("input.md");
        let output_path = tmp.path().join("output.md");

        let input = "---\n\
id: example\n\
legacy_key: unknown\n\
---\n\
Hello\n\
";
        fs::write(&input_path, input)?;

        let (frontmatter, body) = read_frontmatter(&input_path)?;
        write_frontmatter(&output_path, &frontmatter, body.clone())?;

        let (read_back, read_body) = read_frontmatter(&output_path)?;

        assert_eq!(frontmatter, read_back);
        assert_eq!(body, read_body);
        assert_eq!(
            read_back.get("legacy_key"),
            Some(&Value::String("unknown".into()))
        );
        Ok(())
    }

    #[test]
    fn frontmatter_without_markers_returns_whole_body() -> io::Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("plain.md");
        let input = "No frontmatter here\n";
        fs::write(&path, input)?;

        let (frontmatter, body) = read_frontmatter(&path)?;

        assert!(frontmatter.is_empty());
        assert_eq!(body, input);

        Ok(())
    }

    #[test]
    fn managed_fields_round_trip_is_stable() -> io::Result<()> {
        let tmp = TempDir::new()?;
        let input_path = tmp.path().join("managed-in.md");
        let output_path = tmp.path().join("managed-out.md");

        let input = "---\n\
type: wiki\n\
status: fresh\n\
id: wiki-1\n\
title: Example Title\n\
source_document_ids:\n\
  - doc-1\n\
source_revision_ids:\n\
  - rev-2\n\
generated_at: 1700000000\n\
generated_by: builder\n\
build_record_id: build-1\n\
---\n\
Managed markdown body\n";
        fs::write(&input_path, input)?;

        let (frontmatter, body) = read_frontmatter(&input_path)?;
        let expected = mapping_from_pairs(&[
            ("type", Value::String("wiki".into())),
            ("status", Value::String("fresh".into())),
            ("id", Value::String("wiki-1".into())),
            ("title", Value::String("Example Title".into())),
            (
                "source_document_ids",
                Value::Sequence(vec![Value::String("doc-1".into())]),
            ),
            (
                "source_revision_ids",
                Value::Sequence(vec![Value::String("rev-2".into())]),
            ),
            ("generated_at", Value::Number(1_700_000_000.into())),
            ("generated_by", Value::String("builder".into())),
            ("build_record_id", Value::String("build-1".into())),
        ]);

        let serialized = read_file(&input_path)?;
        assert!(serialized.contains("source_document_ids"));

        write_frontmatter(&output_path, &frontmatter, body.as_str())?;
        let rendered = read_file(&output_path)?;

        let (rendered_frontmatter, rendered_body) = read_frontmatter(&output_path)?;
        assert_eq!(rendered_body, body);
        assert_eq!(rendered_frontmatter, frontmatter);

        for pair in [
            ("type", "wiki"),
            ("status", "fresh"),
            ("id", "wiki-1"),
            ("title", "Example Title"),
            ("generated_by", "builder"),
            ("build_record_id", "build-1"),
        ] {
            assert!(rendered.contains(&format!("{}: {}", pair.0, pair.1)));
        }

        assert_eq!(rendered_frontmatter, expected);
        assert_eq!(frontmatter, expected);

        Ok(())
    }
}
