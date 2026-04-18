# Review and Promotion Workflow

Generated output is not automatically trusted. kb requires explicit review before
synthetic artifacts become durable wiki facts.

## ReviewItem

A `ReviewItem` is a pending action that requires user approval before it is executed.
Each `ReviewItem` has:
- `id` — stable identifier
- `kind` — one of `Promotion`, `ConceptMerge`, or `AliasCleanup`
- `target_entity_id` — the artifact or entity being promoted/merged
- `proposed_destination` — the target wiki path (for promotions)
- `citations` — list of source citation labels the artifact depends on
- `affected_pages` — wiki pages that would change if approved
- `status` — `Pending`, `Approved`, or `Rejected`
- `comment` — description of what the review item is doing

`ReviewItem` records live under `reviews/` in the KB root.

## Promotion Workflow

1. `kb ask --promote <question>` writes the answer artifact to `outputs/questions/<id>/`
   and creates a `ReviewItem` with `kind = Promotion` pointing to a proposed destination
   in `wiki/questions/`.
2. `kb review` lists all pending `ReviewItem` records for the user to inspect.
3. The user runs `kb review approve <id>` to accept a promotion. kb then writes the
   artifact through the managed-region contract into the destination wiki path.
4. The user runs `kb review reject <id>` to decline. The artifact remains in `outputs/`
   but is marked `Rejected` so kb does not re-suggest it.

## Concept Merge Workflow

When `kb compile` detects two candidate concepts that appear to refer to the same thing,
it creates a `ReviewItem` with `kind = ConceptMerge` proposing to merge them. The user
approves or rejects the merge via `kb review`.

## Alias Cleanup Workflow

If an alias on a concept page duplicates another concept's canonical title, a
`ReviewItem` with `kind = AliasCleanup` is created proposing to remove or redirect
the alias.

## Why Review Gates Matter

Without review gates, a confident but wrong LLM response could silently overwrite
durable wiki facts. The review queue is the mechanism that keeps synthetic outputs
separated from human-validated knowledge until a human explicitly promotes them.
