use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::Serialize;

use kb_compile::concept_merge::{AppliedMerge, apply_concept_merge};
use kb_compile::promotion::execute_promotion;
use kb_core::{
    ReviewItem, ReviewKind, ReviewStatus, list_review_items, load_review_item, save_review_item,
};

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
    for kind in &[ReviewKind::Promotion, ReviewKind::ConceptMerge, ReviewKind::AliasMerge, ReviewKind::Canonicalization] {
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
        match item.kind {
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
        }
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

pub fn run_review_approve(root: &Path, id: &str, json: bool, emit_json: &dyn Fn(&str, serde_json::Value) -> Result<()>) -> Result<()> {
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

            if json {
                emit_json("review.approve", serde_json::json!({
                    "id": id,
                    "action": "approved",
                    "kind": "promotion",
                    "destination": result.destination.display().to_string(),
                    "build_record_id": result.build_record.metadata.id,
                }))?;
            } else {
                println!("Approved: {id}");
                println!("  Promoted to: {}", result.destination.display());
                println!("  Build record: {}", result.build_record.metadata.id);
            }
        }
        ReviewKind::ConceptMerge => {
            let applied = apply_concept_merge(root, &item)
                .with_context(|| format!("apply concept merge for review '{id}'"))?;

            let prior_status = item.status;
            // Flip status on first apply; a re-apply leaves it approved.
            if prior_status == ReviewStatus::Pending {
                let mut approved = item;
                approved.status = ReviewStatus::Approved;
                approved.metadata.updated_at_millis = now;
                save_review_item(root, &approved)
                    .with_context(|| format!("save approved review item '{id}'"))?;
            }

            emit_concept_merge_applied(id, &applied, prior_status, json, emit_json)?;
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
