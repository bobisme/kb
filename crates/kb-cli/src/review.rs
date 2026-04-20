use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::Serialize;

use kb_compile::backlinks;
use kb_compile::concept_candidate::apply_concept_candidate;
use kb_compile::concept_merge::{AppliedMerge, WIKI_CONCEPTS_DIR, apply_concept_merge};
use kb_compile::imputed_fix::apply_imputed_fix;
use kb_compile::index_page;
use kb_compile::promotion::execute_promotion;
use kb_core::{
    ReviewItem, ReviewKind, ReviewStatus, list_review_items, load_review_item, save_review_item,
};
use kb_llm::LlmAdapter;

/// Prefix used by `kb-lint`'s duplicate-concepts check when it queues a
/// `concept_merge` review item. These items encode both concept ids in the
/// review id (`lint:duplicate-concepts:<a_id>:<b_id>`) and use
/// `reviews/merges/<slug>.json` as their `proposed_destination`. Auto-apply
/// has to derive the canonical concept page from the id instead of reading
/// `proposed_destination` (which points at the proposal sidecar, not a page).
const LINT_DUPLICATE_CONCEPTS_PREFIX: &str = "lint:duplicate-concepts:";

fn now_millis() -> Result<u64> {
    let millis = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    Ok(u64::try_from(millis)?)
}

const fn kind_label(kind: ReviewKind) -> &'static str {
    match kind {
        ReviewKind::Promotion => "promotion",
        ReviewKind::ConceptMerge => "concept_merge",
        ReviewKind::AliasMerge => "alias_merge",
        ReviewKind::Canonicalization => "canonicalization",
        ReviewKind::ConceptCandidate => "concept_candidate",
        ReviewKind::Contradiction => "contradiction",
        ReviewKind::ImputedFix => "imputed_fix",
    }
}

const fn status_label(status: ReviewStatus) -> &'static str {
    match status {
        ReviewStatus::Pending => "pending",
        ReviewStatus::Approved => "approved",
        ReviewStatus::Rejected => "rejected",
    }
}

#[derive(Serialize)]
struct ReviewListPayload {
    items: Vec<ReviewItemSummary>,
    counts: ReviewCounts,
}

#[derive(Serialize)]
struct ReviewItemSummary {
    id: String,
    kind: String,
    status: String,
    comment: String,
    target_entity_id: String,
    proposed_destination: Option<String>,
    created_at_millis: u64,
}

#[derive(Serialize)]
struct ReviewCounts {
    total: usize,
    pending: usize,
    approved: usize,
    rejected: usize,
    by_kind: Vec<KindCount>,
}

#[derive(Serialize)]
struct KindCount {
    kind: String,
    count: usize,
}

impl From<&ReviewItem> for ReviewItemSummary {
    fn from(item: &ReviewItem) -> Self {
        Self {
            id: item.metadata.id.clone(),
            kind: kind_label(item.kind).to_string(),
            status: status_label(item.status).to_string(),
            comment: item.comment.clone(),
            target_entity_id: item.target_entity_id.clone(),
            proposed_destination: item.proposed_destination.as_ref().map(|p| p.display().to_string()),
            created_at_millis: item.created_at_millis,
        }
    }
}

pub fn run_review_list(root: &Path, json: bool, emit_json: &dyn Fn(&str, serde_json::Value) -> Result<()>) -> Result<()> {
    let items = list_review_items(root)?;

    let pending_count = items.iter().filter(|i| i.status == ReviewStatus::Pending).count();
    let approved_count = items.iter().filter(|i| i.status == ReviewStatus::Approved).count();
    let rejected_count = items.iter().filter(|i| i.status == ReviewStatus::Rejected).count();

    let mut kind_counts = Vec::new();
    for kind in &[ReviewKind::Promotion, ReviewKind::ConceptMerge, ReviewKind::AliasMerge, ReviewKind::Canonicalization, ReviewKind::ConceptCandidate, ReviewKind::Contradiction, ReviewKind::ImputedFix] {
        let count = items.iter().filter(|i| i.kind == *kind && i.status == ReviewStatus::Pending).count();
        if count > 0 {
            kind_counts.push(KindCount {
                kind: kind_label(*kind).to_string(),
                count,
            });
        }
    }

    let summaries: Vec<ReviewItemSummary> = items.iter().map(ReviewItemSummary::from).collect();
    let payload = ReviewListPayload {
        counts: ReviewCounts {
            total: items.len(),
            pending: pending_count,
            approved: approved_count,
            rejected: rejected_count,
            by_kind: kind_counts,
        },
        items: summaries,
    };

    if json {
        emit_json("review.list", serde_json::to_value(&payload)?)?;
        return Ok(());
    }

    if pending_count == 0 {
        println!("No pending review items.");
        if approved_count > 0 || rejected_count > 0 {
            println!("({approved_count} approved, {rejected_count} rejected)");
        }
        return Ok(());
    }

    println!("{pending_count} pending review item(s):\n");
    for item in &items {
        if item.status != ReviewStatus::Pending {
            continue;
        }
        let dest = item
            .proposed_destination
            .as_ref()
            .map_or_else(|| "-".to_string(), |p| p.display().to_string());
        println!("  {} [{}] → {}", item.metadata.id, kind_label(item.kind), dest);
        if !item.comment.is_empty() {
            println!("    {}", item.comment);
        }
    }

    if approved_count > 0 || rejected_count > 0 {
        println!("\n({approved_count} approved, {rejected_count} rejected)");
    }

    Ok(())
}

/// Print the "On approve / On reject" narrative shown by
/// `run_review_show` for pending items. Split out so `run_review_show`
/// stays under clippy's function-length cap.
fn print_pending_kind_narrative(kind: ReviewKind) {
    match kind {
        ReviewKind::ConceptMerge => {
            println!(
                "On approve: the canonical page absorbs member aliases + sources \
                 and the subsumed member concept files are deleted."
            );
        }
        ReviewKind::Promotion => {
            println!("On approve: the promotion is executed and the wiki page written.");
        }
        ReviewKind::AliasMerge | ReviewKind::Canonicalization => {
            println!(
                "On approve: the decision is recorded; the corresponding edit \
                 must be made manually, then run 'kb compile'."
            );
        }
        ReviewKind::ConceptCandidate => {
            println!(
                "On approve: an LLM drafts wiki/concepts/<slug>.md from \
                 the source mentions and refreshes backlinks + indexes."
            );
        }
        ReviewKind::Contradiction => {
            println!(
                "On approve: the contradiction is marked acknowledged — \
                 no auto-fix is applied. The same concept + quote-set \
                 will be skipped on future runs so the LLM doesn't \
                 re-flag it."
            );
            println!(
                "On reject: the contradiction is marked 'intended nuance' \
                 — the same concept + quote-set is still suppressed on \
                 future runs, but the rejected status is preserved so you \
                 can tell apart deliberate nuance from acknowledged bugs."
            );
        }
        ReviewKind::ImputedFix => {
            println!(
                "On approve: the imputed draft is applied — for \
                 missing-concept gaps a new concept page is written; \
                 for thin-body gaps the existing page's body is \
                 rewritten (frontmatter and managed regions preserved)."
            );
            println!(
                "On reject: the draft is discarded and the same gap is \
                 skipped on future impute runs (acknowledged via \
                 fingerprint)."
            );
        }
    }
}

pub fn run_review_show(root: &Path, id: &str, json: bool, emit_json: &dyn Fn(&str, serde_json::Value) -> Result<()>) -> Result<()> {
    let item = load_review_item(root, id)?
        .with_context(|| format!("review item '{id}' not found"))?;

    if json {
        emit_json("review.show", serde_json::to_value(&item)?)?;
        return Ok(());
    }

    println!("Review: {}", item.metadata.id);
    println!("Kind:   {}", kind_label(item.kind));
    println!("Status: {}", status_label(item.status));
    if item.status == ReviewStatus::Pending {
        print_pending_kind_narrative(item.kind);
    }
    println!("Comment: {}", item.comment);
    println!("Target: {}", item.target_entity_id);
    if let Some(dest) = &item.proposed_destination {
        println!("Destination: {}", dest.display());
    }
    if !item.citations.is_empty() {
        println!("Citations: {}", item.citations.join(", "));
    }
    if !item.affected_pages.is_empty() {
        println!("Affected pages:");
        for page in &item.affected_pages {
            println!("  {}", page.display());
        }
    }

    if item.kind == ReviewKind::Promotion {
        if let Some(dest) = &item.proposed_destination {
            let dest_path = root.join(dest);
            if dest_path.exists() {
                println!("\n--- Existing page at {} ---", dest.display());
                let content = std::fs::read_to_string(&dest_path)
                    .with_context(|| format!("read existing page {}", dest_path.display()))?;
                let preview: String = content.lines().take(20).collect::<Vec<_>>().join("\n");
                println!("{preview}");
                if content.lines().count() > 20 {
                    println!("... ({} more lines)", content.lines().count() - 20);
                }
            }
        }

        for output_path in &item.metadata.output_paths {
            let full = root.join(output_path);
            if full.exists() && full.extension().is_some_and(|e| e == "md") {
                println!("\n--- Proposed content ({}) ---", output_path.display());
                let content = std::fs::read_to_string(&full)
                    .with_context(|| format!("read artifact {}", full.display()))?;
                let preview: String = content.lines().take(30).collect::<Vec<_>>().join("\n");
                println!("{preview}");
                if content.lines().count() > 30 {
                    println!("... ({} more lines)", content.lines().count() - 30);
                }
                break;
            }
        }
    }

    Ok(())
}

pub fn run_review_approve(
    root: &Path,
    id: &str,
    json: bool,
    emit_json: &dyn Fn(&str, serde_json::Value) -> Result<()>,
    adapter_factory: &dyn Fn() -> Result<Box<dyn LlmAdapter>>,
) -> Result<()> {
    let item = load_review_item(root, id)?
        .with_context(|| format!("review item '{id}' not found"))?;

    // `concept_merge` approvals execute the merge (see `apply_concept_merge`).
    // Re-approving an already-approved merge is a no-op that re-prints the
    // summary instead of erroring, so users who repeat the command or run it
    // after a crash don't get a scary failure. Every other kind still requires
    // a fresh Pending item.
    let allow_reapply = item.kind == ReviewKind::ConceptMerge
        && item.status == ReviewStatus::Approved;
    if item.status != ReviewStatus::Pending && !allow_reapply {
        bail!(
            "review item '{}' is {} — only pending items can be approved",
            id,
            status_label(item.status)
        );
    }

    let now = now_millis()?;

    match item.kind {
        ReviewKind::Promotion => {
            let result = execute_promotion(root, &item, now)
                .with_context(|| format!("execute promotion for review '{id}'"))?;

            // Refresh `wiki/index.md` + `wiki/questions/index.md` (and the
            // sources/concepts indexes) so the new question shows up in the
            // wiki entry points without needing a follow-up `kb compile`.
            // This is layout-only — no LLM passes, just file walks + atomic
            // writes. Failure here must not fail the promotion itself (the
            // page has already been written atomically), so we log a warning
            // and continue; mirrors the pattern in `kb forget` (bn-i5r).
            let index_pages_refreshed = refresh_wiki_indexes(root);

            if json {
                emit_json("review.approve", serde_json::json!({
                    "id": id,
                    "action": "approved",
                    "kind": "promotion",
                    "destination": result.destination.display().to_string(),
                    "build_record_id": result.build_record.metadata.id,
                    "index_pages_refreshed": index_pages_refreshed,
                }))?;
            } else {
                println!("Approved: {id}");
                println!("  Promoted to: {}", result.destination.display());
                println!("  Build record: {}", result.build_record.metadata.id);
                if index_pages_refreshed {
                    println!("  Wiki indexes refreshed.");
                }
            }
        }
        ReviewKind::ConceptMerge => {
            // `lint:duplicate-concepts:...` items have a different shape from the
            // merge-pass-produced `merge:<slug>` items: their
            // `proposed_destination` points at the proposal JSON sidecar, not a
            // concept page, so they can't be fed to `apply_concept_merge`
            // as-is. Rewrite the item in-memory to the shape apply expects (see
            // `prepare_lint_duplicate_concepts_merge`), then delegate.
            let apply_item = if id.starts_with(LINT_DUPLICATE_CONCEPTS_PREFIX) {
                prepare_lint_duplicate_concepts_merge(&item)
                    .with_context(|| format!("prepare lint-duplicate-concepts apply for '{id}'"))?
            } else {
                item.clone()
            };

            let applied = apply_concept_merge(root, &apply_item)
                .with_context(|| format!("apply concept merge for review '{id}'"))?;

            let prior_status = item.status;
            // Flip status on first apply; a re-apply leaves it approved. Persist
            // the original (on-disk) item, not the synthesized apply-item.
            if prior_status == ReviewStatus::Pending {
                let mut approved = item;
                approved.status = ReviewStatus::Approved;
                approved.metadata.updated_at_millis = now;
                save_review_item(root, &approved)
                    .with_context(|| format!("save approved review item '{id}'"))?;
            }

            emit_concept_merge_applied(id, &applied, prior_status, json, emit_json)?;
        }
        ReviewKind::ConceptCandidate => {
            approve_concept_candidate(
                root,
                id,
                item,
                now,
                json,
                emit_json,
                adapter_factory,
            )?;
        }
        ReviewKind::ImputedFix => {
            approve_imputed_fix(root, id, item, now, json, emit_json)?;
        }
        other => {
            // alias_merge and canonicalization remain decision-only for now: we
            // flip status so the queue moves on but the reviewer must make the
            // corresponding edit (rename a concept page, adjust aliases, etc.)
            // by hand. Be explicit about this rather than silently implying
            // success.
            let mut approved = item;
            approved.status = ReviewStatus::Approved;
            approved.metadata.updated_at_millis = now;
            save_review_item(root, &approved)
                .with_context(|| format!("save approved review item '{id}'"))?;

            if json {
                emit_json("review.approve", serde_json::json!({
                    "id": id,
                    "action": "approved",
                    "kind": kind_label(other),
                    "requires_manual_followup": true,
                }))?;
            } else {
                println!("Approved: {id} ({})", kind_label(other));
                println!(
                    "  note: the proposed change is not applied automatically — \
                     perform the edit manually, then run 'kb compile'."
                );
            }
        }
    }

    Ok(())
}

/// Apply a `concept_candidate` approval: build the LLM adapter lazily,
/// draft + write the concept page, flip the review item, and run the
/// best-effort backlinks + index refreshes. Split out of
/// `run_review_approve` so the match-body stays under clippy's line cap.
fn approve_concept_candidate(
    root: &Path,
    id: &str,
    item: kb_core::ReviewItem,
    now: u64,
    json: bool,
    emit_json: &dyn Fn(&str, serde_json::Value) -> Result<()>,
    adapter_factory: &dyn Fn() -> Result<Box<dyn LlmAdapter>>,
) -> Result<()> {
    // bn-lw06: draft a concept page via the LLM from the lint-flagged
    // candidate + the normalized source mentions, then refresh backlinks
    // + indexes inline (mirrors bn-i5r/bn-2zy pattern).
    let adapter = adapter_factory()
        .with_context(|| format!("build LLM adapter for review '{id}'"))?;
    let applied = apply_concept_candidate(adapter.as_ref(), root, &item)
        .with_context(|| format!("apply concept candidate for review '{id}'"))?;

    // Flip review item → approved before the layout refresh so even if
    // backlinks/index refresh hiccups, the queue moves on. The page is
    // already written atomically by apply_concept_candidate.
    let mut approved = item;
    approved.status = ReviewStatus::Approved;
    approved.metadata.updated_at_millis = now;
    save_review_item(root, &approved)
        .with_context(|| format!("save approved review item '{id}'"))?;

    // Best-effort refreshes: any failure is logged but does not propagate
    // — the concept page is already on disk and the review item is
    // flipped. Users can run `kb compile` to fully rebuild if needed.
    let backlinks_refreshed = refresh_backlinks(root);
    let index_pages_refreshed = refresh_wiki_indexes(root);

    if json {
        emit_json(
            "review.approve",
            serde_json::json!({
                "id": id,
                "action": "approved",
                "kind": "concept_candidate",
                "concept_path": applied.concept_path.display().to_string(),
                "canonical_name": applied.canonical_name,
                "aliases": applied.aliases,
                "category": applied.category,
                "source_document_ids": applied.source_document_ids,
                "backlinks_refreshed": backlinks_refreshed,
                "index_pages_refreshed": index_pages_refreshed,
            }),
        )?;
    } else {
        println!("Approved: {id} (concept_candidate)");
        println!("  Canonical: {}", applied.canonical_name);
        if !applied.aliases.is_empty() {
            println!("  Aliases: {}", applied.aliases.join(", "));
        }
        if let Some(cat) = &applied.category {
            println!("  Category: {cat}");
        }
        println!("  Page: {}", applied.concept_path.display());
        if !applied.source_document_ids.is_empty() {
            println!("  Sources: {}", applied.source_document_ids.join(", "));
        }
        if backlinks_refreshed {
            println!("  Backlinks refreshed.");
        }
        if index_pages_refreshed {
            println!("  Wiki indexes refreshed.");
        }
    }
    Ok(())
}

/// Apply an `ImputedFix` review item (bn-xt4o). For missing-concept gaps
/// this writes a new concept page; for thin-body gaps it rewrites the
/// existing page's body while preserving frontmatter + managed regions.
/// Best-effort backlink + index refreshes run inline so the new content
/// is visible in the wiki without a follow-up `kb compile`.
fn approve_imputed_fix(
    root: &Path,
    id: &str,
    item: kb_core::ReviewItem,
    now: u64,
    json: bool,
    emit_json: &dyn Fn(&str, serde_json::Value) -> Result<()>,
) -> Result<()> {
    let applied = apply_imputed_fix(root, &item)
        .with_context(|| format!("apply imputed fix for review '{id}'"))?;

    // Flip review item → approved before the layout refresh so the queue
    // moves on even if backlinks/index refresh fails.
    let mut approved = item;
    approved.status = ReviewStatus::Approved;
    approved.metadata.updated_at_millis = now;
    save_review_item(root, &approved)
        .with_context(|| format!("save approved review item '{id}'"))?;

    let backlinks_refreshed = refresh_backlinks(root);
    let index_pages_refreshed = refresh_wiki_indexes(root);

    if json {
        emit_json(
            "review.approve",
            serde_json::json!({
                "id": id,
                "action": "approved",
                "kind": "imputed_fix",
                "gap_kind": applied.gap_kind,
                "concept_path": applied.concept_path.display().to_string(),
                "concept_name": applied.concept_name,
                "created_new_page": applied.created_new_page,
                "cited_sources": applied.cited_sources,
                "backlinks_refreshed": backlinks_refreshed,
                "index_pages_refreshed": index_pages_refreshed,
            }),
        )?;
    } else {
        println!("Approved: {id} (imputed_fix)");
        println!("  Gap kind: {}", applied.gap_kind);
        println!("  Concept:  {}", applied.concept_name);
        let verb = if applied.created_new_page { "Wrote" } else { "Rewrote body of" };
        println!("  {verb}: {}", applied.concept_path.display());
        if !applied.cited_sources.is_empty() {
            println!("  Web sources cited:");
            for src in &applied.cited_sources {
                println!("    - {src}");
            }
        }
        if backlinks_refreshed {
            println!("  Backlinks refreshed.");
        }
        if index_pages_refreshed {
            println!("  Wiki indexes refreshed.");
        }
    }
    Ok(())
}

/// Regenerate the wiki's global + per-category index pages from current
/// on-disk state. Returns `true` on success, `false` (with a warning printed)
/// on any failure. Never propagates the error — index refresh is best-effort
/// layout maintenance that should not mask a successful promotion.
fn refresh_wiki_indexes(root: &Path) -> bool {
    match index_page::generate_indexes(root) {
        Ok(artifacts) => match index_page::persist_index_artifacts(&artifacts) {
            Ok(()) => true,
            Err(err) => {
                eprintln!(
                    "warning: wiki index refresh failed after approve: {err:#}"
                );
                false
            }
        },
        Err(err) => {
            eprintln!(
                "warning: wiki index refresh failed after approve: {err:#}"
            );
            false
        }
    }
}

/// Rebuild the `backlinks` managed region on every concept page from
/// current on-disk state. Returns `true` on success, `false` (with a
/// warning printed) on any failure. Never propagates the error — a new
/// concept page has already been written atomically, so a backlinks hiccup
/// shouldn't make approve look failed. Users can run `kb compile` for a
/// full rebuild.
fn refresh_backlinks(root: &Path) -> bool {
    match backlinks::run_backlinks_pass(root) {
        Ok(artifacts) => match backlinks::persist_backlinks_artifacts(&artifacts) {
            Ok(()) => true,
            Err(err) => {
                eprintln!(
                    "warning: backlinks refresh failed after approve: {err:#}"
                );
                false
            }
        },
        Err(err) => {
            eprintln!(
                "warning: backlinks refresh failed after approve: {err:#}"
            );
            false
        }
    }
}

/// Rewrite a `lint:duplicate-concepts:<a_id>:<b_id>` review item into the shape
/// `apply_concept_merge` expects.
///
/// The original item's `proposed_destination` points at the proposal JSON
/// sidecar (`reviews/merges/<slug>.json`) because `save_review_item` also uses
/// it as the item's output path. Apply needs a concept page path instead.
///
/// Parsing convention (mirrors how the lint emitter builds the id — see
/// `crates/kb-lint/src/lib.rs::build_review_item`): the two concept ids are
/// joined with `:` after the fixed prefix. The *second* id is treated as the
/// canonical page (the duplicate-detection comment is phrased "A is a
/// near-duplicate of B", so B is the surviving page). The *first* id is folded
/// into the canonical.
///
/// Concept ids have the shape `concept:<slug>` with slugs restricted to
/// `[a-z0-9-]+` (see `kb_core::slug_from_title`), so splitting on `:concept:`
/// is unambiguous. Each id's slug (the part after `concept:`) is also the file
/// stem of its `wiki/concepts/<slug>.md` page.
///
/// The synthesized item keeps the original id (used only for error messages
/// inside apply) but carries `proposed_destination` = canonical page path and
/// `dependencies` = \[`canonical_slug`, `merged_from_slug`\] — which
/// `apply_concept_merge` then slugs to find each member file. Because
/// `slug_from_title("raft-consensus") == "raft-consensus"`, the slug of the
/// canonical dependency matches the canonical page stem (so apply skips it)
/// and the slug of the merged-from dependency finds the right member file.
fn prepare_lint_duplicate_concepts_merge(item: &ReviewItem) -> Result<ReviewItem> {
    let id = &item.metadata.id;
    let payload = id
        .strip_prefix(LINT_DUPLICATE_CONCEPTS_PREFIX)
        .with_context(|| format!("review id '{id}' missing expected lint-duplicate prefix"))?;

    // `payload` is `<a_id>:<b_id>` with each id in the form `concept:<slug>`.
    // Split on the single `:concept:` separator that joins them.
    let (merged_from_id, canonical_id) = split_concept_pair(payload).with_context(|| {
        format!(
            "cannot parse pair of concept ids from review id '{id}' (expected \
             '{LINT_DUPLICATE_CONCEPTS_PREFIX}concept:<a>:concept:<b>')"
        )
    })?;

    let canonical_slug = canonical_id
        .strip_prefix("concept:")
        .with_context(|| format!("canonical id '{canonical_id}' does not start with 'concept:'"))?;
    let merged_from_slug = merged_from_id.strip_prefix("concept:").with_context(|| {
        format!("merged-from id '{merged_from_id}' does not start with 'concept:'")
    })?;

    if canonical_slug.is_empty() || merged_from_slug.is_empty() {
        bail!("review id '{id}' encodes an empty concept slug; cannot apply");
    }

    let canonical_rel = PathBuf::from(WIKI_CONCEPTS_DIR).join(format!("{canonical_slug}.md"));

    let mut synthesized = item.clone();
    synthesized.proposed_destination = Some(canonical_rel);
    // `apply_concept_merge` slugs each dependency with `slug_from_title`. Feed
    // it the bare slugs so they round-trip to the correct file stems (feeding
    // the raw ids like `concept:raft-consensus` would slug to
    // `concept-raft-consensus` and miss the file).
    synthesized.metadata.dependencies =
        vec![canonical_slug.to_string(), merged_from_slug.to_string()];

    Ok(synthesized)
}

/// Split a `concept:<a>:concept:<b>` string into `(concept:<a>, concept:<b>)`.
///
/// Returns `None` if the separator `:concept:` is missing.
fn split_concept_pair(payload: &str) -> Option<(&str, &str)> {
    // Skip the leading `concept:` so the first `:concept:` we find is the
    // separator between the two ids, not a prefix match on the opening id.
    let after_first = payload.strip_prefix("concept:")?;
    let sep = after_first.find(":concept:")?;
    let first_slug = &after_first[..sep];
    let second_with_prefix = &after_first[sep + 1..]; // drop the leading ':'
    // Reconstruct the first id with its `concept:` prefix restored.
    let first_id_end = "concept:".len() + first_slug.len();
    let first_id = &payload[..first_id_end];
    Some((first_id, second_with_prefix))
}

fn emit_concept_merge_applied(
    id: &str,
    applied: &AppliedMerge,
    prior_status: ReviewStatus,
    json: bool,
    emit_json: &dyn Fn(&str, serde_json::Value) -> Result<()>,
) -> Result<()> {
    let was_reapply = prior_status == ReviewStatus::Approved;

    if json {
        emit_json(
            "review.approve",
            serde_json::json!({
                "id": id,
                "action": if was_reapply { "reapplied" } else { "approved" },
                "kind": "concept_merge",
                "canonical_page": applied.canonical_path.display().to_string(),
                "removed_members": applied
                    .removed_members
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>(),
                "added_aliases": applied.added_aliases,
                "added_source_document_ids": applied.added_source_document_ids,
                "canonical_updated": applied.canonical_updated,
            }),
        )?;
        return Ok(());
    }

    if was_reapply {
        println!("Already approved: {id} (concept_merge) — nothing new to apply.");
    } else {
        println!("Approved: {id} (concept_merge)");
    }
    println!("  Canonical: {}", applied.canonical_path.display());
    if applied.removed_members.is_empty() {
        println!("  Removed members: (none)");
    } else {
        println!("  Removed members:");
        for rel in &applied.removed_members {
            println!("    - {}", rel.display());
        }
    }
    if !applied.added_aliases.is_empty() {
        println!("  Merged aliases: {}", applied.added_aliases.join(", "));
    }
    if !applied.added_source_document_ids.is_empty() {
        println!(
            "  Merged source_document_ids: {}",
            applied.added_source_document_ids.join(", ")
        );
    }
    if !applied.canonical_updated && applied.removed_members.is_empty() {
        println!("  (no changes — canonical already holds all member data)");
    }
    println!("  Next: run 'kb compile' to regenerate backlinks.");
    Ok(())
}

pub fn run_review_reject(root: &Path, id: &str, comment: Option<&str>, json: bool, emit_json: &dyn Fn(&str, serde_json::Value) -> Result<()>) -> Result<()> {
    let item = load_review_item(root, id)?
        .with_context(|| format!("review item '{id}' not found"))?;

    if item.status != ReviewStatus::Pending {
        bail!(
            "review item '{}' is {} — only pending items can be rejected",
            id,
            status_label(item.status)
        );
    }

    let now = now_millis()?;
    let mut rejected = item;
    rejected.status = ReviewStatus::Rejected;
    rejected.metadata.updated_at_millis = now;
    if let Some(c) = comment {
        if !rejected.comment.is_empty() {
            rejected.comment.push_str(" | ");
        }
        rejected.comment.push_str(c);
    }
    save_review_item(root, &rejected)
        .with_context(|| format!("save rejected review item '{id}'"))?;

    if json {
        emit_json("review.reject", serde_json::json!({
            "id": id,
            "action": "rejected",
            "kind": kind_label(rejected.kind),
        }))?;
    } else {
        println!("Rejected: {id}");
    }

    Ok(())
}
