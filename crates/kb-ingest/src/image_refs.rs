//! Markdown image reference scanning, copying, and path rewriting for ingest.
//!
//! When a markdown file references a local image with `![alt](./pic.png)` the
//! ingest pass needs to:
//!
//! 1. Locate the referenced file (relative to the markdown's directory).
//! 2. Validate it — extension allow-list + a size cap — to avoid pulling in
//!    arbitrary binary payloads.
//! 3. Copy it into `normalized/<src>/assets/<basename>`, uniquifying the
//!    basename when two references in the same document collide.
//! 4. Rewrite the markdown body so the reference points at the new
//!    `assets/<basename>` location relative to the normalized source.
//!
//! URLs (http/https/data/mailto) and fragments are left untouched. Broken
//! references (relative paths that don't resolve on disk) are left untouched
//! with a warning so the ingest itself still succeeds.
//!
//! **Implementation note — staging dir.** `kb_core::write_normalized_document`
//! reads each path in `NormalizedDocument::normalized_assets` and writes to
//! `assets/<source_path.file_name()>`. That means if two source paths share a
//! basename (e.g. `a/pic.png` + `b/pic.png`) they'd collide at destination
//! even though we've already minted distinct target names. To work around
//! this without changing the kb-core API, we stage each asset into a
//! per-ingest temporary directory under the final uniquified basename, and
//! hand those staged paths to `normalized_assets`. `copy_and_validate_assets`
//! then reads each staged path by its (unique) filename and the destination
//! name matches.

use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;
use tracing::warn;

/// File extensions we'll copy alongside a markdown ingest. Anything outside
/// this allow-list is treated as "not an image we're willing to pull in" and
/// skipped with a warning. Hardcoded deliberately — bn-1geb's scope doesn't
/// include a user-facing config knob for this.
const ALLOWED_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "gif", "svg", "webp", "bmp"];

/// Per-asset size cap. Above this the reference is skipped with a warning.
/// Images this large are almost always an accident (video frames, raw photos)
/// and will bloat the normalized/ tree for every revision.
const MAX_ASSET_BYTES: u64 = 10 * 1024 * 1024;

/// Matches `![alt](path)` images in markdown.
///
/// - `alt` is captured on `[^\]]*` (group 1) so `](` inside alt text
///   terminates the match early. This is intentional: `CommonMark`'s full
///   escape-aware parsing is out of scope here and the worst failure mode is
///   a reference we fail to rewrite, which degrades gracefully to "broken
///   link in the normalized source".
/// - The path is captured as either group 2 (angle-bracket-wrapped,
///   `<path with spaces.png>`) or group 3 (plain, `path.png`). Callers take
///   whichever group matched — see [`captured_raw_path`].
/// - An optional `CommonMark` title (`"caption"`) after the path is tolerated
///   and discarded — we don't carry titles through v1 (bn-18qs).
///
/// Angle-bracket form matters for paths with whitespace. Obsidian emits
/// `![](<./Screenshot with spaces.png>)` when "Markdown links" is enabled on
/// filenames containing spaces; the plain form can't represent such paths
/// without producing ambiguous markdown.
static IMAGE_REF_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"!\[([^\]]*)\]\(\s*(?:<([^>]+)>|([^)\s"]+))(?:\s+"[^"]*")?\s*\)"#)
        .expect("hard-coded image regex is valid")
});

/// Extract the raw path from an [`IMAGE_REF_RE`] capture, trimming any
/// surrounding whitespace. Returns an empty string when neither form matched
/// (shouldn't happen for a successful whole-pattern match, but we're
/// defensive rather than panicking).
fn captured_raw_path<'a>(capture: &regex::Captures<'a>) -> &'a str {
    capture
        .get(2)
        .or_else(|| capture.get(3))
        .map_or("", |m| m.as_str().trim())
}

/// True if `path` contains any ASCII whitespace. When rewriting a reference
/// whose resolved basename contains whitespace we emit the angle-bracket
/// form (`<assets/My Screenshot.png>`) so `CommonMark` parsers treat the
/// URL as a single unambiguous token.
fn basename_needs_angle_wrap(basename: &str) -> bool {
    basename.chars().any(char::is_whitespace)
}

/// Result of scanning a markdown body and staging its image references.
pub struct CopiedImages {
    /// Paths to the staged image files (inside `staging_dir`), suitable for
    /// `NormalizedDocument::normalized_assets`. Each path's `file_name()` is
    /// the destination basename under `normalized/<src>/assets/`, which is
    /// what `kb_core::write_normalized_document` uses to compute its write
    /// target.
    pub normalized_assets: Vec<PathBuf>,
    /// Markdown body with every successfully-staged reference rewritten to
    /// `assets/<basename>`. References we skipped (URLs, missing files,
    /// oversized, unsupported extension) are left byte-identical.
    pub rewritten_markdown: String,
}

/// Scan `markdown_body` for local image references, copy each accepted
/// image into `staging_dir` under a unique basename, and return the staged
/// asset paths alongside a rewritten body.
///
/// `markdown_dir` is the parent directory of the source markdown file, used
/// to resolve relative references. `staging_dir` must exist and be writable;
/// `normalized_assets` in the returned struct will be paths under it.
///
/// # Errors
/// Returns an IO error if copying an accepted image into `staging_dir`
/// fails. Skip conditions (URL, missing file, oversized, wrong extension)
/// are not errors — they log a warning and leave the reference unchanged.
///
/// # Panics
/// Does not panic in practice. The `.expect` on `capture.get(0)` holds
/// because regex capture group 0 is always present when a match exists.
pub fn scan_and_stage(
    markdown_body: &str,
    markdown_dir: &Path,
    staging_dir: &Path,
) -> io::Result<CopiedImages> {
    let mut normalized_assets: Vec<PathBuf> = Vec::new();
    // Tracks the basenames we've already handed out to avoid collisions when
    // two distinct references would otherwise write to the same destination
    // (e.g. `a/pic.png` and `b/pic.png`).
    let mut used_basenames: HashSet<String> = HashSet::new();
    // Maps source absolute path -> target basename so that a single source
    // file referenced multiple times only gets copied once AND keeps the same
    // rewritten target.
    let mut path_to_basename: HashMap<PathBuf, String> = HashMap::new();

    // We rebuild the body by walking regex matches and splicing in the
    // rewritten substring whenever we decide to keep & copy the image.
    let mut rewritten = String::with_capacity(markdown_body.len());
    let mut cursor = 0usize;

    for capture in IMAGE_REF_RE.captures_iter(markdown_body) {
        let whole = capture.get(0).expect("regex always has group 0");
        // Emit everything between the previous match and this one verbatim.
        rewritten.push_str(&markdown_body[cursor..whole.start()]);
        cursor = whole.end();

        let alt = capture.get(1).map_or("", |m| m.as_str());
        let raw_path = captured_raw_path(&capture);

        match stage_reference(
            raw_path,
            markdown_dir,
            &mut used_basenames,
            &mut path_to_basename,
        ) {
            ReferenceOutcome::Staged {
                absolute_source,
                basename,
            } => {
                let staged_path = staging_dir.join(&basename);
                // Copy only once per (source, destination) pair. If this is
                // the first time we're seeing this source in the document,
                // copy it into the staging area under its uniquified name.
                if !normalized_assets
                    .iter()
                    .any(|existing: &PathBuf| existing == &staged_path)
                {
                    std::fs::copy(&absolute_source, &staged_path)?;
                    normalized_assets.push(staged_path);
                }
                rewritten.push_str("![");
                rewritten.push_str(alt);
                // Emit the angle-bracket form when the basename contains
                // whitespace so the rewritten markdown stays unambiguously
                // parseable. Plain form otherwise preserves existing test
                // output byte-for-byte (bn-18qs).
                if basename_needs_angle_wrap(&basename) {
                    rewritten.push_str("](<assets/");
                    rewritten.push_str(&basename);
                    rewritten.push_str(">)");
                } else {
                    rewritten.push_str("](assets/");
                    rewritten.push_str(&basename);
                    rewritten.push(')');
                }
            }
            ReferenceOutcome::Leave => {
                // Preserve the original matched substring unchanged.
                rewritten.push_str(whole.as_str());
            }
        }
    }

    rewritten.push_str(&markdown_body[cursor..]);

    Ok(CopiedImages {
        normalized_assets,
        rewritten_markdown: rewritten,
    })
}

enum ReferenceOutcome {
    /// Copy this asset and rewrite the reference. `absolute_source` is the
    /// resolved path on disk; `basename` is the destination filename (after
    /// collision handling).
    Staged {
        absolute_source: PathBuf,
        basename: String,
    },
    /// Leave the reference byte-identical — URL, broken, too large, wrong
    /// type, etc. A warning has already been logged for the skip cases that
    /// deserve one.
    Leave,
}

fn stage_reference(
    raw_path: &str,
    markdown_dir: &Path,
    used_basenames: &mut HashSet<String>,
    path_to_basename: &mut HashMap<PathBuf, String>,
) -> ReferenceOutcome {
    let trimmed = raw_path.trim();
    if trimmed.is_empty() {
        return ReferenceOutcome::Leave;
    }

    // External / non-local schemes. Lowercased prefix check to catch
    // `HTTPS://` as well, matching `crate::is_url`'s behaviour.
    let head_len = trimmed.len().min(8);
    let head = trimmed.get(..head_len).unwrap_or("").to_ascii_lowercase();
    if head.starts_with("http://")
        || head.starts_with("https://")
        || head.starts_with("data:")
        || head.starts_with("mailto:")
        || trimmed.starts_with('#')
    {
        return ReferenceOutcome::Leave;
    }

    // Already-rewritten references (`assets/foo.png`) are idempotent: if the
    // file happens to exist relative to markdown_dir we'll pick it up; if not
    // we leave it alone. Either branch is handled naturally below.

    let candidate = Path::new(trimmed);
    let absolute_candidate = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        markdown_dir.join(candidate)
    };

    // Canonicalize when possible — collapses `..` and symlinks so two
    // references to the same underlying file dedupe cleanly. Fall back to the
    // un-canonicalized path if the file doesn't exist (canonicalize fails on
    // missing paths).
    let resolved = std::fs::canonicalize(&absolute_candidate).unwrap_or(absolute_candidate);

    if !resolved.is_file() {
        warn!("referenced image not found: {}", raw_path);
        return ReferenceOutcome::Leave;
    }

    // If we've already staged this exact source in this document, reuse the
    // basename we minted the first time.
    if let Some(existing) = path_to_basename.get(&resolved) {
        return ReferenceOutcome::Staged {
            absolute_source: resolved.clone(),
            basename: existing.clone(),
        };
    }

    let extension = resolved
        .extension()
        .and_then(OsStr::to_str)
        .map(str::to_ascii_lowercase);
    match extension.as_deref() {
        Some(ext) if ALLOWED_EXTENSIONS.contains(&ext) => {}
        _ => {
            warn!(
                "skipping image reference with disallowed extension: {}",
                resolved.display()
            );
            return ReferenceOutcome::Leave;
        }
    }

    // Size check. `metadata` failure is exceedingly unlikely on a path we
    // just confirmed is_file, but treat it as a skip rather than abort.
    match std::fs::metadata(&resolved) {
        Ok(md) if md.len() > MAX_ASSET_BYTES => {
            warn!(
                "skipping oversized image (> {} bytes): {}",
                MAX_ASSET_BYTES,
                resolved.display()
            );
            return ReferenceOutcome::Leave;
        }
        Ok(_) => {}
        Err(err) => {
            warn!(
                "skipping image after metadata read failed ({}): {}",
                err,
                resolved.display()
            );
            return ReferenceOutcome::Leave;
        }
    }

    let basename = mint_basename(&resolved, used_basenames);
    used_basenames.insert(basename.clone());
    path_to_basename.insert(resolved.clone(), basename.clone());

    ReferenceOutcome::Staged {
        absolute_source: resolved,
        basename,
    }
}

/// Pick a destination basename for `source_path`, uniquifying on collision
/// against `used`.
///
/// Collision strategy: compute a short 3-char hash of the full absolute
/// source path and splice it in before the extension — e.g.
/// `pic.png` + collision → `pic-a7x.png`. The hash is deterministic in the
/// full source path, so re-ingesting the same document yields the same
/// mapping.
fn mint_basename(source_path: &Path, used: &HashSet<String>) -> String {
    let file_name = source_path
        .file_name()
        .and_then(OsStr::to_str)
        .map_or_else(|| "image.bin".to_string(), str::to_string);

    if !used.contains(&file_name) {
        return file_name;
    }

    let (stem, ext) = split_name(&file_name);
    let source_bytes = source_path.as_os_str().to_string_lossy();
    // terseid is already in the dependency graph via kb-core; its `hash`
    // returns a short alphanumeric string whose length we control. 3 chars
    // (~15 bits) is plenty to disambiguate within a single document and keeps
    // the suffix human-readable.
    let suffix = terseid::hash(source_bytes.as_bytes(), 3);
    let candidate = format_with_suffix(stem, &suffix, ext);
    if !used.contains(&candidate) {
        return candidate;
    }

    // Very unlikely in practice (would require 3-char hash collision on two
    // different source paths) but fall back to escalating lengths.
    for len in 4..=8 {
        let suffix = terseid::hash(source_bytes.as_bytes(), len);
        let candidate = format_with_suffix(stem, &suffix, ext);
        if !used.contains(&candidate) {
            return candidate;
        }
    }

    // Give up and append a numeric counter. We've exhausted 8-char hashes of
    // the same input (impossible without repeating inputs), so this branch is
    // for test harnesses that inject pre-seeded collisions.
    for n in 1..u32::MAX {
        let suffix = format!("{n}");
        let candidate = format_with_suffix(stem, &suffix, ext);
        if !used.contains(&candidate) {
            return candidate;
        }
    }
    file_name
}

fn split_name(file_name: &str) -> (&str, Option<&str>) {
    match file_name.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() => (stem, Some(ext)),
        // Dot-files (`.gitignore`) and extensionless names — treat the whole
        // thing as stem.
        _ => (file_name, None),
    }
}

fn format_with_suffix(stem: &str, suffix: &str, ext: Option<&str>) -> String {
    ext.map_or_else(
        || format!("{stem}-{suffix}"),
        |ext| format!("{stem}-{suffix}.{ext}"),
    )
}

/// Rewrite `assets/<basename>` references in `text` to `<prefix>/<basename>`.
///
/// Used by the compile pass: the normalized source references `assets/foo.png`
/// (relative to `normalized/<src>/`), but the wiki source page lives at
/// `wiki/sources/<src>.md` — a different depth — so any image reference we
/// carry through must be re-anchored to a path that resolves from the wiki
/// page's own location.
///
/// Only references whose path starts with `assets/` are rewritten. External
/// URLs, fragments, and references that have already been rewritten to some
/// other prefix are left alone.
///
/// # Panics
/// Does not panic in practice. The `.expect` on `capture.get(0)` holds
/// because regex capture group 0 is always present when a match exists.
#[must_use]
pub fn rewrite_asset_refs(text: &str, new_prefix: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0usize;

    for capture in IMAGE_REF_RE.captures_iter(text) {
        let whole = capture.get(0).expect("regex always has group 0");
        out.push_str(&text[cursor..whole.start()]);
        cursor = whole.end();

        let alt = capture.get(1).map_or("", |m| m.as_str());
        let path = captured_raw_path(&capture);

        if let Some(rest) = path.strip_prefix("assets/") {
            // Preserve angle-bracket form when the path contains whitespace,
            // both as an input signal (we wouldn't have matched otherwise
            // under the plain branch) and on output so the new ref stays
            // valid CommonMark (bn-18qs).
            let wrap = basename_needs_angle_wrap(rest);
            out.push_str("![");
            out.push_str(alt);
            if wrap {
                out.push_str("](<");
            } else {
                out.push_str("](");
            }
            out.push_str(new_prefix);
            if !new_prefix.ends_with('/') {
                out.push('/');
            }
            out.push_str(rest);
            if wrap {
                out.push_str(">)");
            } else {
                out.push(')');
            }
        } else {
            out.push_str(whole.as_str());
        }
    }

    out.push_str(&text[cursor..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn staging(dir: &TempDir) -> PathBuf {
        let p = dir.path().join("_staging");
        fs::create_dir_all(&p).expect("create staging dir");
        p
    }

    fn ends_with_png(name: &str) -> bool {
        Path::new(name)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("png"))
    }

    #[test]
    fn scans_and_stages_local_png() {
        let dir = TempDir::new().expect("tempdir");
        let png = dir.path().join("pic.png");
        fs::write(&png, b"\x89PNGfake").expect("write png");
        let stage = staging(&dir);

        let body = "![hello](./pic.png)\n";
        let out = scan_and_stage(body, dir.path(), &stage).expect("stage");

        assert_eq!(out.normalized_assets.len(), 1);
        assert!(out.normalized_assets[0].ends_with("pic.png"));
        assert!(
            out.normalized_assets[0].is_file(),
            "staged file must exist on disk"
        );
        assert_eq!(out.rewritten_markdown, "![hello](assets/pic.png)\n");
    }

    #[test]
    fn leaves_url_images_untouched() {
        let dir = TempDir::new().expect("tempdir");
        let body = "![ext](https://example.com/pic.png)\n![data](data:image/png;base64,AAAA)\n";
        let stage = staging(&dir);
        let out = scan_and_stage(body, dir.path(), &stage).expect("stage");
        assert!(out.normalized_assets.is_empty());
        assert_eq!(out.rewritten_markdown, body);
    }

    #[test]
    fn warns_and_leaves_missing_images() {
        let dir = TempDir::new().expect("tempdir");
        let body = "![missing](./nope.png)\n";
        let stage = staging(&dir);
        let out = scan_and_stage(body, dir.path(), &stage).expect("stage");
        assert!(out.normalized_assets.is_empty());
        assert_eq!(out.rewritten_markdown, body);
    }

    #[test]
    fn skips_unsupported_extension() {
        let dir = TempDir::new().expect("tempdir");
        let data = dir.path().join("data.csv");
        fs::write(&data, b"col1,col2\n1,2\n").expect("write csv");
        let stage = staging(&dir);
        let body = "![data](./data.csv)\n";
        let out = scan_and_stage(body, dir.path(), &stage).expect("stage");
        assert!(out.normalized_assets.is_empty());
        assert_eq!(out.rewritten_markdown, body);
    }

    #[test]
    fn collision_hashes_suffix() {
        let dir = TempDir::new().expect("tempdir");
        let sub_a = dir.path().join("a");
        let sub_b = dir.path().join("b");
        fs::create_dir_all(&sub_a).expect("mkdir a");
        fs::create_dir_all(&sub_b).expect("mkdir b");
        fs::write(sub_a.join("pic.png"), b"A").expect("write a/pic");
        fs::write(sub_b.join("pic.png"), b"B").expect("write b/pic");
        let stage = staging(&dir);

        let body = "![one](./a/pic.png)\n![two](./b/pic.png)\n";
        let out = scan_and_stage(body, dir.path(), &stage).expect("stage");

        assert_eq!(out.normalized_assets.len(), 2);
        // Both rewrites live under assets/ and carry .png.
        let mut lines_iter = out.rewritten_markdown.lines();
        let first = lines_iter.next().expect("line 0").to_string();
        let second = lines_iter.next().expect("line 1").to_string();
        assert!(first.contains("assets/"), "{first}");
        assert!(second.contains("assets/"), "{second}");
        let extract = |line: &str| -> String {
            let start = line.find("assets/").expect("prefix") + "assets/".len();
            let end = line[start..].find(')').expect("closing paren") + start;
            line[start..end].to_string()
        };
        let a = extract(&first);
        let b = extract(&second);
        assert_ne!(a, b, "basenames must differ after collision");
        assert!(ends_with_png(&a) && ends_with_png(&b));

        // The staged files exist on disk with matching unique basenames.
        for asset in &out.normalized_assets {
            assert!(asset.is_file(), "staged asset missing: {}", asset.display());
        }
    }

    #[test]
    fn dedups_same_source_referenced_twice() {
        let dir = TempDir::new().expect("tempdir");
        let png = dir.path().join("pic.png");
        fs::write(&png, b"X").expect("write png");
        let stage = staging(&dir);

        let body = "![first](./pic.png)\nsome text\n![again](./pic.png)\n";
        let out = scan_and_stage(body, dir.path(), &stage).expect("stage");
        assert_eq!(out.normalized_assets.len(), 1);
        assert!(
            out.rewritten_markdown
                .contains("![first](assets/pic.png)")
        );
        assert!(
            out.rewritten_markdown
                .contains("![again](assets/pic.png)")
        );
    }

    #[test]
    fn oversized_image_skipped() {
        let dir = TempDir::new().expect("tempdir");
        let big = dir.path().join("huge.png");
        let bytes = vec![0u8; usize::try_from(MAX_ASSET_BYTES).expect("fits in usize") + 1];
        fs::write(&big, &bytes).expect("write big png");
        let stage = staging(&dir);

        let body = "![big](./huge.png)\n";
        let out = scan_and_stage(body, dir.path(), &stage).expect("stage");
        assert!(out.normalized_assets.is_empty());
        assert_eq!(out.rewritten_markdown, body);
    }

    #[test]
    fn rewrite_asset_refs_retargets_prefix() {
        let input = "intro\n![fig](assets/foo.png) and ![url](https://x/y.png) and ![other](./foo.png)\n";
        let out = rewrite_asset_refs(input, "../../normalized/src-xyz/assets");
        assert!(
            out.contains("![fig](../../normalized/src-xyz/assets/foo.png)"),
            "got: {out}"
        );
        // Non-assets refs untouched.
        assert!(out.contains("![url](https://x/y.png)"));
        assert!(out.contains("![other](./foo.png)"));
    }

    #[test]
    fn rewrite_asset_refs_handles_trailing_slash_prefix() {
        let out = rewrite_asset_refs("![a](assets/b.png)", "/abs/");
        assert_eq!(out, "![a](/abs/b.png)");
    }

    // ---- bn-18qs: angle-bracket form + paths with spaces + titles ----

    /// Obsidian emits `![](<./Screenshot from YYYY-MM-DD.png>)` when a
    /// filename contains spaces. The v1 regex excluded whitespace entirely,
    /// so these refs never matched and the image was never staged.
    #[test]
    fn angle_bracket_path_with_spaces_is_staged_and_rewritten() {
        let dir = TempDir::new().expect("tempdir");
        let png = dir.path().join("screenshot with spaces.png");
        fs::write(&png, b"\x89PNGfake").expect("write png");
        let stage = staging(&dir);

        let body = "![](<./screenshot with spaces.png>)\n";
        let out = scan_and_stage(body, dir.path(), &stage).expect("stage");

        assert_eq!(out.normalized_assets.len(), 1, "ref should match + stage");
        assert!(
            out.normalized_assets[0].ends_with("screenshot with spaces.png"),
            "basename preserved: {:?}",
            out.normalized_assets[0],
        );
        assert!(out.normalized_assets[0].is_file());
        // Rewrite must preserve the angle-bracket form so the path-with-
        // spaces stays a single URL token under CommonMark.
        assert_eq!(
            out.rewritten_markdown,
            "![](<assets/screenshot with spaces.png>)\n",
            "got: {}",
            out.rewritten_markdown,
        );
    }

    /// `CommonMark` titles (`"caption"`) are tolerated but discarded.
    /// Without this the existing regex would fail to match
    /// `![x](a.png "caption")` because the closing paren isn't the next
    /// non-space char after the path.
    #[test]
    fn title_attribute_is_tolerated_and_discarded() {
        let dir = TempDir::new().expect("tempdir");
        let png = dir.path().join("a.png");
        fs::write(&png, b"\x89PNGfake").expect("write png");
        let stage = staging(&dir);

        let body = r#"![x](a.png "caption")"#;
        let out = scan_and_stage(body, dir.path(), &stage).expect("stage");

        assert_eq!(out.normalized_assets.len(), 1);
        assert!(out.normalized_assets[0].ends_with("a.png"));
        // No title in the rewritten ref — titles aren't carried through v1.
        assert_eq!(out.rewritten_markdown, "![x](assets/a.png)");
    }

    /// Title with the angle-bracket form too.
    #[test]
    fn angle_bracket_with_title_staged_title_discarded() {
        let dir = TempDir::new().expect("tempdir");
        let png = dir.path().join("a b.png");
        fs::write(&png, b"\x89PNGfake").expect("write png");
        let stage = staging(&dir);

        let body = r#"![x](<./a b.png> "caption")"#;
        let out = scan_and_stage(body, dir.path(), &stage).expect("stage");

        assert_eq!(out.normalized_assets.len(), 1);
        assert!(out.normalized_assets[0].ends_with("a b.png"));
        assert_eq!(out.rewritten_markdown, "![x](<assets/a b.png>)");
    }

    /// Missing / URL / data URI refs in angle-bracket form still skip.
    #[test]
    fn angle_bracket_url_and_data_still_skipped() {
        let dir = TempDir::new().expect("tempdir");
        let stage = staging(&dir);
        let body = "![](<https://example.com/x y.png>)\n![](<data:image/png;base64,AA>)\n";
        let out = scan_and_stage(body, dir.path(), &stage).expect("stage");
        assert!(out.normalized_assets.is_empty());
        // Skipped refs are preserved byte-identical.
        assert_eq!(out.rewritten_markdown, body);
    }

    #[test]
    fn angle_bracket_missing_file_leaves_ref_untouched() {
        let dir = TempDir::new().expect("tempdir");
        let stage = staging(&dir);
        let body = "![](<./does not exist.png>)\n";
        let out = scan_and_stage(body, dir.path(), &stage).expect("stage");
        assert!(out.normalized_assets.is_empty());
        assert_eq!(out.rewritten_markdown, body);
    }

    /// Rewrite emits plain form when basename has no whitespace — preserves
    /// existing callers' output byte-for-byte.
    #[test]
    fn plain_basename_still_rewrites_plain() {
        let dir = TempDir::new().expect("tempdir");
        let png = dir.path().join("pic.png");
        fs::write(&png, b"\x89PNGfake").expect("write png");
        let stage = staging(&dir);

        let body = "![hi](./pic.png)";
        let out = scan_and_stage(body, dir.path(), &stage).expect("stage");
        assert_eq!(out.rewritten_markdown, "![hi](assets/pic.png)");
    }

    /// `rewrite_asset_refs` (used by the ingest's own tests + the wiki-page
    /// prefix rewrite path) must preserve angle-bracket form on output when
    /// the basename has spaces.
    #[test]
    fn rewrite_asset_refs_preserves_angle_wrap_for_spaces() {
        let input = "![](<assets/My Screenshot.png>)\n![](assets/plain.png)\n";
        let out = rewrite_asset_refs(input, "../../normalized/src-xyz/assets");
        assert!(
            out.contains("![](<../../normalized/src-xyz/assets/My Screenshot.png>)"),
            "spaces path must stay angle-wrapped on rewrite: {out}",
        );
        assert!(
            out.contains("![](../../normalized/src-xyz/assets/plain.png)"),
            "plain path stays plain: {out}",
        );
    }
}
