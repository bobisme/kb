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
- `members` — every original candidate object that belongs to the group, whether by
  semantic identity or by parent/child folding.
- `confident: true` when the grouping is unambiguous.
- `confident: false` when you are unsure — this routes the group to human review instead
  of being silently merged. Prefer routing to review when containment is plausible but
  not clear-cut.
- `rationale` — for uncertain groupings, briefly explain the ambiguity (e.g. "X could be
  a member of Y or an independent concept").

### 3. Definition synthesis across variants

When multiple variants/aliases fold into a single canonical concept, the merged
concept's body definition (carried in any `definition_hint` on the canonical member,
or synthesized downstream) MUST describe the **most general reading** — the umbrella
that covers every folded variant — NOT the definition of the last-seen or most-specific
variant. Do not copy one variant's one-liner verbatim when that one-liner only
describes that variant.

Signals you have picked the wrong body:

- The body starts with "A <variant-name> variant…" where `<variant-name>` is one of
  the folded aliases rather than the canonical name.
- The body starts with "A specialization of…" or "An instance of…" framing.
- The body would be wrong if the reader only knew the canonical name and didn't know
  which specific variant was meant.

Worked example:

- Input candidates:
  - `{name: "Basic Paxos", definition_hint: "A two-phase consensus protocol for agreeing on a single value.", …}`
  - `{name: "Multi-Paxos", definition_hint: "An optimization of Paxos that amortizes leader election across a log of values.", …}`
  - `{name: "Cheap Paxos", definition_hint: "A Paxos variant that keeps auxiliary nodes out of the steady state and activates them only during failures.", …}`
- Canonical: `"Paxos"`, aliases: `["Basic Paxos", "Multi-Paxos", "Cheap Paxos"]`.
- ✓ correct canonical-member `definition_hint`: "A family of consensus algorithms
  for replicated logs, with variants including Basic Paxos, Multi-Paxos, and Cheap
  Paxos."
- ❌ wrong canonical-member `definition_hint`: "A Paxos variant that keeps auxiliary
  nodes out of the steady state and activates them only during failures." — that is
  Cheap Paxos specifically, not Paxos in general.

Return only valid JSON in this exact shape — no other text before or after:
{
  "groups": [
    {
      "canonical_name": "Borrow checker",
      "aliases": ["borrowck"],
      "members": [],
      "confident": true,
      "rationale": null
    }
  ]
}
