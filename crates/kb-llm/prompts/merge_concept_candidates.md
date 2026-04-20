# Concept Merge Request

You are clustering concept candidates extracted from multiple documents into canonical
concept entries for a knowledge base.

## Candidates

{{candidates_json}}

## Task

Group the candidates into canonical concepts. There are TWO kinds of grouping to perform,
and both should be applied:

### 1. Semantic identity (near-duplicates)

Two candidates refer to the same concept when they describe the same idea, even if they
use different names, phrasing, or abbreviations. Group these together.

### 2. Parent / child containment (sub-concept folding)

When one candidate names a MEMBER, INSTANCE, or SPECIFIC CASE of another candidate's
broader idea, fold the narrower candidate into the broader one instead of leaving them
as separate concepts. The narrower candidate's name becomes an alias of the parent, and
its `members` entry is preserved under the parent's group.

Examples of parent / child relationships to fold:

- `edit`, `drop`, `pick`, `squash`, `fixup`, `reword` are each MEMBERS of "git rebase
  todo actions" — fold them into that parent.
- "5-second busy timeout default" is a SPECIFIC CASE of "SQLite busy timeout" — fold it.
- "borrowck" and "borrow checker" are the same concept (semantic identity) — merge by
  identity, not by containment.
- "WAL mode on NFS" is a SPECIFIC FAILURE CASE of "SQLite WAL journaling" — fold it.

Rule of thumb for containment: if candidate X would reasonably be explained as "a kind of
Y" or "an example of Y" or "one of the Ys", prefer folding X into Y. The parent (broader)
concept becomes the canonical name; the child's name goes into `aliases`.

Do NOT fold sibling concepts together. Two distinct children of the same parent should
still merge with the PARENT, not with each other. If the parent is not present among the
candidates, emit ONE merged group named after the parent concept (e.g. "git rebase todo
actions") rather than leaving six separate children ungrouped.

### For each group, output:

- `canonical_name` — the clearest, most general form that covers every member. When
  folding children into a parent, this is the parent name.
- `aliases` — alternate names, abbreviations, shorthand, and folded child names. Do not
  repeat the canonical name in aliases.
- `category` — a short 1-3 word tag that places the concept in a high-level bucket used
  by the concepts index. Pick freely from a domain-relevant taxonomy; the system infers
  the taxonomy from whatever tags you emit across concepts. Examples:
    - Rust topic: `concurrency`, `async`, `macros`, `ownership`, `ffi`
    - Distributed systems: `consensus`, `storage`, `networking`, `observability`
  Rules:
    - Keep it short — a tag, not a description.
    - Prefer lowercase kebab-case or single words (e.g. `async`, `distributed-systems`).
    - Use flat category names only — do NOT include slashes.
    - If nothing obvious fits, set `category: null` and the concept will be placed under
      an "Uncategorized" bucket. Do not invent a category just to avoid null.
    - Reuse existing category names across concepts when they fit — do not spawn a new
      tag per concept.
- `members` — every original candidate object that belongs to the group, whether by
  semantic identity or by parent/child folding.
- `confident: true` when the grouping is unambiguous.
- `confident: false` when you are unsure — this routes the group to human review instead
  of being silently merged. Prefer routing to review when containment is plausible but
  not clear-cut.
- `rationale` — for uncertain groupings, briefly explain the ambiguity (e.g. "X could be
  a member of Y or an independent concept").

### 3. Definition hint (optional context only)

Each input candidate may carry a `definition_hint`. You may preserve or drop
these on the member objects as-is; do NOT rewrite them here. **The general
concept body is synthesized downstream by a separate dedicated call — don't
try to pick or merge one-liner definitions in this step.** Just get the
clustering right (canonical name, aliases, members, confidence).

Return only valid JSON in this exact shape — no other text before or after:
{
  "groups": [
    {
      "canonical_name": "Borrow checker",
      "aliases": ["borrowck"],
      "category": "ownership",
      "members": [],
      "confident": true,
      "rationale": null
    }
  ]
}
