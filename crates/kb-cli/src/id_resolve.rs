//! Short-id prefix resolution shared by `kb inspect` and `kb forget`.
//!
//! Users shouldn't have to type the full 6+ char terseid hash when a short
//! prefix uniquely identifies an entity. This module wraps `terseid::IdResolver`
//! around the KB's three id-bearing namespaces — source documents, concept
//! pages, and question outputs — and resolves a user-typed string in that
//! order (src → concept → question). The first resolver that returns a unique
//! match wins; an ambiguous match errors immediately with the candidate list
//! so the user can disambiguate with a longer prefix.
//!
//! Why this order? Source ids are the most common inspect / forget target
//! (users type them straight out of `kb ingest` output), so trying them first
//! short-circuits the cheapest case and avoids ambiguous-across-kinds errors
//! when a slug happens to overlap with a src hash. Concept slugs come next
//! because they're the second most common human-typed form; question ids are
//! mostly machine-emitted and rarely typed, so they sit last.
//!
//! The ids themselves are enumerated lazily from filesystem directory
//! listings rather than a pre-built index: `normalized/*/` for srcs,
//! `wiki/concepts/*.md` for concepts, and `outputs/questions/*/` for
//! questions. This keeps the helper honest for brand-new KBs (no index file
//! yet) and avoids a stale-index foot-gun — the filesystem is the source of
//! truth.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use kb_core::normalized_dir;
use terseid::{IdResolver, ResolverConfig, TerseIdError, find_matching_ids};

/// What a successful [`resolve`] call produces.
///
/// The kind is preserved so callers can branch on it (`inspect` dispatches
/// to different report builders per kind; `forget` only accepts srcs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedId {
    /// Full id as it appears on disk (e.g. `src-a7x3q9`, `shared-memory`,
    /// `q-a7x3`).
    pub id: String,
    /// Which namespace matched.
    pub kind: IdKind,
}

/// The three id-bearing namespaces this module knows about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdKind {
    /// `src-<hash>` — a source document. Enumerated from `normalized/*/`.
    Source,
    /// A concept page slug (arbitrary string, derived from `wiki/concepts/*.md`
    /// filenames). `terseid` is still a safe wrapper here: slugs without a
    /// dash won't collide with the `con` default prefix because exact-match
    /// wins first, and substring matching handles the bare-slug case.
    Concept,
    /// `q-<hash>` — a question artifact. Enumerated from `outputs/questions/*/`.
    Question,
}

/// Try to resolve `input` as a full id or unique prefix in the three
/// namespaces, in the order src → concept → question.
///
/// Semantics:
///   - an **exact** match in any namespace wins immediately (a full `src-a7x3q9`
///     returns before we even try concepts);
///   - a **unique prefix** match in the first namespace that finds one wins,
///     even if a later namespace would also match a different id (the user's
///     input is assumed to be in the higher-priority namespace when prefixed
///     with `src-` / etc., and unprefixed inputs are most often concept slugs
///     — see the ordering rationale in the module docstring);
///   - an **ambiguous** match in any namespace fails fast with the candidate
///     list, even if a later namespace would find a unique match. This is
///     deliberate: if the user typed `src-a7` and two srcs share that prefix,
///     they want a "longer prefix, please" error, not a silent fall-through to
///     a concept.
///
/// Returns `Err` with a readable message when:
///   - the prefix is ambiguous (lists the candidates, suggests a longer
///     prefix),
///   - the input matches nothing in any namespace (suggests checking `kb status`).
///
/// # Errors
///
/// Returns an error when directory listings fail, when the input is ambiguous
/// in the first namespace it appears in, or when no namespace contains a
/// match.
pub fn resolve(root: &Path, input: &str) -> Result<ResolvedId> {
    if input.trim().is_empty() {
        bail!("id resolve: input cannot be empty");
    }

    // Try src, then concept, then question. The first "found" or "ambiguous"
    // outcome short-circuits — we never fall through on ambiguity.
    if let Some(resolved) = try_resolve_kind(root, input, IdKind::Source)? {
        return Ok(resolved);
    }
    if let Some(resolved) = try_resolve_kind(root, input, IdKind::Concept)? {
        return Ok(resolved);
    }
    if let Some(resolved) = try_resolve_kind(root, input, IdKind::Question)? {
        return Ok(resolved);
    }

    bail!(
        "'{input}' was not found as a source, concept, or question id. \
         Run 'kb status' to list known entities."
    )
}

/// Run `IdResolver::resolve` against one namespace. Returns:
///   - `Ok(Some(ResolvedId))` on exact or unique-prefix match,
///   - `Ok(None)` when the namespace has no candidates or none match,
///   - `Err(...)` on ambiguity (bubbles up with the candidate list formatted)
///     or on an I/O failure enumerating ids.
fn try_resolve_kind(root: &Path, input: &str, kind: IdKind) -> Result<Option<ResolvedId>> {
    let known_ids = list_ids(root, kind)?;
    if known_ids.is_empty() {
        return Ok(None);
    }
    let prefix = default_prefix_for(kind);
    let config = ResolverConfig::new(prefix);
    let id_resolver = IdResolver::new(config);

    // `find_matching_ids` compares its needle against the hash portion of
    // each parsed id (the part after the `<prefix>-`). If the user typed a
    // prefixed input like `src-a7`, we strip the leading `src-` so the needle
    // (`a7`) actually matches inside the hash (`a7x3q9`). Inputs without the
    // structural prefix pass through unchanged.
    let known = &known_ids;
    let prefix_dash = format!("{prefix}-");
    match id_resolver.resolve(
        input,
        |id| known.iter().any(|k| k == id),
        |substr| {
            let needle = substr.strip_prefix(&prefix_dash).unwrap_or(substr);
            find_matching_ids(known, needle)
        },
    ) {
        Ok(hit) => Ok(Some(ResolvedId {
            id: hit.id,
            kind,
        })),
        Err(TerseIdError::AmbiguousId { partial, matches }) => {
            bail!(
                "'{partial}' is ambiguous; candidates: {}\n\
                 disambiguate by using a longer prefix.",
                matches.join(", ")
            );
        }
        // NotFound and any other terseid error (e.g. validation) we treat as
        // "not a match in this namespace" rather than aborting — a later
        // namespace might still succeed. The caller will surface a single
        // "not found" at the end if every namespace comes up empty.
        Err(_) => Ok(None),
    }
}

/// Default id prefix used by `ResolverConfig` for each namespace. This is
/// what `terseid` prepends when the user's input has no dash (e.g. `a7` → `src-a7`
/// under the src resolver). Chosen to match the on-disk id shape.
const fn default_prefix_for(kind: IdKind) -> &'static str {
    match kind {
        IdKind::Source => "src",
        // Concept slugs don't carry a structural prefix; "con" is a neutral
        // default that won't accidentally match a real slug since substring
        // resolution falls back to the raw input.
        IdKind::Concept => "con",
        IdKind::Question => "q",
    }
}

/// Enumerate the full ids on disk for a namespace. Filesystem-driven so brand-
/// new KBs and in-progress compiles don't depend on an index file.
///
/// # Errors
///
/// Returns an error only when a `read_dir` itself fails (permission denied,
/// etc.). A missing directory is treated as "no ids" — the helper is called
/// against fresh roots that may not have run ingest yet.
pub fn list_ids(root: &Path, kind: IdKind) -> Result<Vec<String>> {
    match kind {
        IdKind::Source => list_subdirs(&normalized_dir(root), |name| name.starts_with("src-")),
        IdKind::Concept => list_md_stems(&root.join("wiki/concepts")),
        IdKind::Question => list_subdirs(&root.join("outputs/questions"), |_| true),
    }
}

/// List immediate subdirectory names under `dir`, filtered by `keep`. Sorted
/// for deterministic behavior in error messages.
fn list_subdirs(dir: &Path, keep: impl Fn(&str) -> bool) -> Result<Vec<String>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in fs::read_dir(dir)
        .with_context(|| format!("read dir {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("read entry in {}", dir.display()))?;
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_dir() {
            continue;
        }
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        if !keep(&name) {
            continue;
        }
        out.push(name);
    }
    out.sort();
    Ok(out)
}

/// List the filename stems of `*.md` files directly under `dir`. Index pages
/// (`index.md`) are excluded so `inspect con-` doesn't prefer the auto-gen
/// landing page over a real concept.
fn list_md_stems(dir: &Path) -> Result<Vec<String>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in fs::read_dir(dir)
        .with_context(|| format!("read dir {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("read entry in {}", dir.display()))?;
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if stem == "index" {
            continue;
        }
        out.push(stem.to_string());
    }
    out.sort();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn mk_src(root: &Path, id: &str) {
        fs::create_dir_all(normalized_dir(root).join(id)).expect("mkdir normalized/<id>");
    }

    fn mk_concept(root: &Path, slug: &str) {
        let dir = root.join("wiki/concepts");
        fs::create_dir_all(&dir).expect("mkdir wiki/concepts");
        fs::write(dir.join(format!("{slug}.md")), "---\nid: x\n---\n").expect("write concept");
    }

    fn mk_question(root: &Path, id: &str) {
        fs::create_dir_all(root.join("outputs/questions").join(id))
            .expect("mkdir outputs/questions/<id>");
    }

    #[test]
    fn resolves_unique_src_prefix() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        mk_src(root, "src-a7x3q9");
        mk_src(root, "src-b2k1m5");

        let resolved = resolve(root, "src-a7").expect("resolve unique prefix");
        assert_eq!(resolved.id, "src-a7x3q9");
        assert_eq!(resolved.kind, IdKind::Source);
    }

    #[test]
    fn full_src_id_still_matches() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        mk_src(root, "src-a7x3q9");
        let resolved = resolve(root, "src-a7x3q9").expect("resolve exact");
        assert_eq!(resolved.id, "src-a7x3q9");
        assert_eq!(resolved.kind, IdKind::Source);
    }

    #[test]
    fn ambiguous_src_prefix_lists_candidates() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        mk_src(root, "src-a7x3q9");
        mk_src(root, "src-a7b2k1");

        let err = resolve(root, "src-a7").expect_err("must be ambiguous");
        let msg = err.to_string();
        assert!(msg.contains("ambiguous"), "expected ambiguity: {msg}");
        assert!(msg.contains("src-a7x3q9"), "candidates listed: {msg}");
        assert!(msg.contains("src-a7b2k1"), "candidates listed: {msg}");
        assert!(msg.contains("longer prefix"), "hint present: {msg}");
    }

    #[test]
    fn not_found_reports_clearly() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        mk_src(root, "src-a7x3q9");

        let err = resolve(root, "src-zz").expect_err("must be missing");
        assert!(
            err.to_string().contains("was not found"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn resolves_concept_slug_when_no_src_match() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        mk_src(root, "src-a7x3q9");
        mk_concept(root, "shared-memory");
        mk_concept(root, "state-machine");

        let resolved = resolve(root, "shared-memory").expect("resolve concept");
        assert_eq!(resolved.id, "shared-memory");
        assert_eq!(resolved.kind, IdKind::Concept);
    }

    #[test]
    fn resolves_question_id_when_nothing_else_matches() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        mk_question(root, "q-a7x3");

        let resolved = resolve(root, "q-a7").expect("resolve question prefix");
        assert_eq!(resolved.id, "q-a7x3");
        assert_eq!(resolved.kind, IdKind::Question);
    }

    #[test]
    fn src_wins_over_concept_on_prefix_collision() {
        // Deliberate overlap: a concept slug happens to start with "src-a".
        // The src resolver runs first and wins; the concept is ignored even
        // though it would also match a bare-input lookup.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        mk_src(root, "src-a7x3q9");
        mk_concept(root, "src-approach"); // unlikely but documented tie-break

        let resolved = resolve(root, "src-a").expect("src wins");
        assert_eq!(resolved.kind, IdKind::Source);
        assert_eq!(resolved.id, "src-a7x3q9");
    }

    #[test]
    fn empty_input_rejected() {
        let dir = tempdir().expect("tempdir");
        let err = resolve(dir.path(), "").expect_err("empty");
        assert!(err.to_string().contains("cannot be empty"));
    }

    #[test]
    fn list_ids_tolerates_missing_dirs() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        assert!(list_ids(root, IdKind::Source).expect("src").is_empty());
        assert!(list_ids(root, IdKind::Concept).expect("concept").is_empty());
        assert!(list_ids(root, IdKind::Question).expect("q").is_empty());
    }

    #[test]
    fn list_ids_excludes_concept_index() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        mk_concept(root, "real-concept");
        mk_concept(root, "index"); // auto-gen landing page
        let ids = list_ids(root, IdKind::Concept).expect("list");
        assert_eq!(ids, vec!["real-concept".to_string()]);
    }
}
