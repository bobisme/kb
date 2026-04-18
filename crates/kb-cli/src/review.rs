use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::Serialize;

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

    if item.status != ReviewStatus::Pending {
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
        other => {
            // Non-Promotion approvals (concept_merge, alias_merge, canonicalization)
            // flip the status so the queue moves on, but v1 does not execute the
            // proposed change automatically — the reviewer is expected to make the
            // corresponding edit (rename a concept page, adjust aliases, etc.) by
            // hand. Be explicit about this rather than silently implying success.
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
