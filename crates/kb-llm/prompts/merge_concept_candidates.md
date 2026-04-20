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

**Structural framing requirement — the first sentence of the canonical
`definition_hint` MUST begin with the canonical name (NOT an alias) followed by
"is" or a similarly declarative verb.** Acceptable openings:

- `<Canonical> is a <category> ...`
- `<Canonical> is the <role> ...`
- `<Canonical> refers to <definition> ...`

The first sentence MUST NOT begin with:

- "A <alias> variant ...", "A <alias> safety property ...", or any phrase whose
  subject is one of the folded aliases rather than the canonical concept.
- "A specialization of ...", "An instance of ...", "A kind of ...", or any
  framing that positions the concept as a sub-case of something else.
- A verbatim lift from the richest candidate quote when that quote only
  describes one variant. Abstract to the umbrella form even when a single
  quote dominates the candidate pool.

**Multi-variant coverage requirement.** When the aliases span multiple
distinct variants, subtypes, or family members (typically ≥ 2 named
variants, or ≥ 3 aliases), the body MUST explicitly signal that scope by
either:

- Using one of the words "variants", "variant", "family", "includes", or
  "including" in the first two sentences, AND
- Naming at least two of the variants by name.

Signals you have picked the wrong body:

- The body starts with "A <variant-name> variant…" where `<variant-name>` is one of
  the folded aliases rather than the canonical name.
- The body starts with "A specialization of…" or "An instance of…" framing.
- The first sentence names a non-canonical alias as its grammatical subject.
- The body would be wrong if the reader only knew the canonical name and didn't know
  which specific variant was meant.

#### Worked examples

**Paxos (consensus-algorithm family)**

- Input candidates:
  - `{name: "Basic Paxos", definition_hint: "A two-phase consensus protocol for agreeing on a single value.", …}`
  - `{name: "Multi-Paxos", definition_hint: "An optimization of Paxos that amortizes leader election across a log of values.", …}`
  - `{name: "Cheap Paxos", definition_hint: "A Paxos variant that keeps auxiliary nodes out of the steady state and activates them only during failures.", …}`
- Canonical: `"Paxos"`, aliases: `["Basic Paxos", "Multi-Paxos", "Cheap Paxos"]`.
- ✓ correct canonical-member `definition_hint`: "Paxos is a family of
  consensus algorithms for replicated logs. Variants include Basic Paxos,
  Multi-Paxos, and Cheap Paxos."
- ❌ wrong canonical-member `definition_hint`: "A Paxos variant that keeps auxiliary
  nodes out of the steady state and activates them only during failures." — that is
  Cheap Paxos specifically, not Paxos in general.
- ❌ wrong: "Basic Paxos is a two-phase protocol for agreeing on a single
  value." — subject is an alias, not the canonical; ignores Multi-Paxos and
  Cheap Paxos.

**Quorum (multi-variant set-theoretic concept)**

- Input candidates:
  - `{name: "Majority quorum", definition_hint: "A quorum consisting of more than half the voting nodes.", …}`
  - `{name: "Byzantine quorum", definition_hint: "A quorum sized to tolerate f Byzantine failures; typically ⌈(n+f+1)/2⌉.", …}`
  - `{name: "Uniform quorum", definition_hint: "A quorum where every member has equal weight.", …}`
- Canonical: `"Quorum"`, aliases: `["Majority quorum", "Byzantine quorum", "Uniform quorum"]`.
- ✓ correct: "Quorum is a subset of nodes whose agreement is required for a
  distributed decision. Variants include majority, uniform, and Byzantine
  quorums."
- ❌ wrong: "Byzantine quorums are sized to tolerate f Byzantine failures..." —
  first sentence scopes to one variant and names an alias as the subject.

**Leader election (general process)**

- Input candidates:
  - `{name: "Raft leader election", definition_hint: "Randomized-timeout election using RequestVote RPCs.", …}`
  - `{name: "Zab leader election", definition_hint: "Fast leader election using epochs and proposals in ZooKeeper Atomic Broadcast.", …}`
  - `{name: "Bully algorithm", definition_hint: "A leader-election algorithm where the highest-id live process wins.", …}`
- Canonical: `"Leader election"`, aliases: `["Raft leader election", "Zab leader election", "Bully algorithm"]`.
- ✓ correct: "Leader election is the process by which distributed processes
  agree on a single coordinator. Variants include the Raft randomized-timeout
  scheme, Zab's epoch-based election, and the Bully algorithm."
- ❌ wrong: "A Zab leader election proceeds in phases using epochs..." —
  describes one variant only; subject is an alias.

**Raft (single consensus algorithm, not a family)**

- Input candidates all describe Raft (possibly different components):
  `{name: "Raft", definition_hint: "…log replication with a strong leader…"}`,
  `{name: "Leader Completeness", definition_hint: "A Raft safety property …"}`.
- Canonical: `"Raft"`, aliases may include `["Leader Completeness"]` if folded.
- ✓ correct: "Raft is a consensus algorithm for managing a replicated log
  with a strong leader."
- ❌ wrong: "A Raft safety property stating that any entry committed in a
  term must appear in the logs of all leaders elected in later terms." —
  subject is an alias (Leader Completeness), not Raft.

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
