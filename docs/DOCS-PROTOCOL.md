# XLOG Documentation Protocol

**A page is not "done" because it is accurate. It is done when an outside reader
succeeds without reading our source code, our commit history, or asking us.**

Docs are a product surface — often the first one. This protocol is the bar every
user-facing page must clear. It is enforced in review with three zones: 🟢 Green
(ships), 🟡 Yellow (blocked, fix first), 🔴 Red (reject, rewrite). A page's zone
is the zone of its **worst** section: one red section makes the whole page red.

---

## 1. Who we write for

The reader is a **competent engineer who did not build xlog**. Assume they know
Python/PyTorch, general logic-programming ideas, and roughly what a GPU is.
Assume they do **not** know:

- our internal names (`RIR`, `PIR`, `XGCF`, `EIR`, "Free Join", "factorized
  delta", dispatch-counter names),
- our release history ("the 0.9.2 line", "main-only", "unreleased beyond…"),
- our research lineage (FAEEL, Gelfond-1991, d-DNNF, "founded support"),
- anything that is only obvious after reading the source.

They are reading to **do something**. Every page earns its place by getting them
closer to a working result and a correct mental model.

---

## 2. The three user questions

Every feature page must answer these, in the reader's words, in this order:

1. **When do I use this?** (the problem it solves — not the mechanism)
2. **How do I do it?** (the smallest runnable example)
3. **How do I know it worked?** (the observable signal, in user terms)

If a page cannot answer all three, it is not user-facing yet.

---

## 🔴 3. RED ZONE — banned. Reject and rewrite.

If any of these appears, the section is not "rough" — it is **not yet written for
a reader**. Do not patch; rewrite from the reader's point of view. Each rule is
followed by a real example pulled from the current docs.

- **R1 — Notes addressed to the doc author, not the reader.**
  > 🔴 *"Do not describe them as available in the 0.9.2 release artifacts."*
  This is an instruction to us. The reader does not care. Convert to a reader
  fact ("Available from v0.10.0") or delete.

- **R2 — Opening with release status or internal plumbing before value.**
  A page must not open with a `<Warning>` about branch/version or a list of
  counters. The reader does not yet know what the feature *is* or why they want
  it. Value first; status and internals later.
  > 🔴 factorized-execution opens with a release `<Warning>` and a table whose
  > "user-visible signal" column is `wcoj_groupby_fusion_dispatch_count`.

- **R3 — An internal symbol as the headline concept.**
  Counter names, IR names, and kernel names are not concepts. Lead with what the
  thing *does*; the exact symbol goes in parentheses or the reference.
  > 🔴 *"User-visible signal: `free_join_dispatch_count`."* → 🟢 "You can confirm
  > it fired from `--stats` (the `free_join_dispatch` count)."

- **R4 — Implementer's-perspective prose as the user's explanation.**
  Contracts, "decline", "fallback parity", "opportunistic" describe how *we*
  reason about the code. The reader needs behavior and usage.
  > 🔴 *"The contract is opportunistic: a WCOJ decline must preserve the same row
  > set through the fallback path. A successful answer does not prove WCOJ fired;
  > use the executor counters."*

- **R5 — Undefined jargon or research terms dropped raw.**
  Any non-obvious term must be glossed in plain language on first use — or cut.
  > 🔴 "witness-multiplied recursive delta joins", "outcome mask", "founded
  > support", "FAEEL", "Gelfond-1991 semantics", "RIR bodies", "helper-split" —
  > all used with no plain-language meaning for a first-time reader.

- **R6 — A user-facing feature explained *only* on an Architecture/internals page.**
  If a user can turn it on, it needs a Guide/concept page written for users. An
  internals page may exist too, but it is not the user's explanation.
  > 🔴 WCOJ and factorized execution are performance features a user enables, yet
  > their only home is under "Architecture".

- **R7 — Sentences that stack three or more new concepts.**
  > 🔴 *"a small trained head whose straight-through-thresholded output gates the
  > candidate's eligibility, so the conjunct stays a derivation gate rather than
  > soft truth mass."* One sentence, four unexplained ideas. Split into: the
  > idea, then the mechanism, then the caveat.

- **R8 — No runnable example for a feature the reader is meant to use.**
  Concept-only prose for a usable feature is red. Show a minimal program and its
  expected output.

- **R9 — Dishonesty: an overclaim, or a real limit hidden.**
  State the bound plainly where it matters. A limit is not a footnote to bury; it
  is information the reader needs to succeed. (See G7.)

---

## 🟡 4. YELLOW ZONE — warning signs. Fix before merge.

Not banned, but a reviewer must resolve each before the page ships.

- **Y1 — "How it works" is present but "when to use it / how to confirm it" is missing.** (See §2.)
- **Y2 — Abstract description arrives before any concrete example.** Flip the order.
- **Y3 — Nominalized mechanism-speak where a subject + verb is clearer.**
  "materialization of the intermediate occurs" → "xlog builds the full
  intermediate table".
- **Y4 — The page starts friendly and degrades into expert-only prose with no on-ramp.** Add a one-line "what this section is for" before each advanced subsection.
- **Y5 — Version/branch status is woven into prose** instead of a single badge or one-line note.
- **Y6 — A second undefined term appears near a freshly defined one.** Gloss it too, or cut it.
- **Y7 — More than one screen of unbroken text** with no example, table, diagram, or callout.

---

## 🟢 5. GREEN ZONE — the bar. This is what "shining" means.

A page ships only when every section is green. Checklist:

- **G1 — Opens with reader value** (what you can do + why it matters) in the first
  two or three sentences — before any mechanism, status, or symbol.
- **G2 — Answers the three user questions** (When? How? How-do-I-know?) explicitly.
- **G3 — Shows a minimal runnable example with expected output** before generalizing.
- **G4 — Defines every non-obvious term in plain language at first use**, and links
  to the glossary.
- **G5 — Uses the reader's vocabulary.** Internal names appear only in parentheses
  or the reference, never as the headline idea.
- **G6 — One idea per paragraph; short, active sentences.**
- **G7 — States limits plainly, once, next to the usage** ("works for small graphs,
  up to ~6–7 events"), not as a wall of hedges.
- **G8 — Keeps maintainer concerns** (release gating, dispatch counters, internal
  contracts, kill switches) **in a clearly-labeled Diagnostics or Reference
  section**, not in the lead.
- **G9 — Is genre-correct** (see §7).

---

## 6. Required page anatomy (feature / concept pages)

1. **Value proposition** — one or two sentences: what you can do, why you'd want it.
2. **Smallest runnable example** — program + expected output.
3. **When to use this** — the problem it solves, in the reader's situation.
4. **How it works** — conceptual, jargon-glossed, one idea per paragraph.
5. **Confirm it worked** — the observable signal, described in user terms.
6. **Limits** — honest bounds, plain, in one place, next to the usage.
7. **See also / reference** — exact symbols, counters, env vars, internals links.

Reference pages and Architecture pages relax anatomy but never §3 (Red) — even a
reference must not stack four concepts in a sentence or leave a research term
unglossed.

---

## 7. Genres — write to the right one

| Genre | Job | Reader | Rule |
|---|---|---|---|
| **Guide / Concept** | Get me to a working result and a correct model. | User. | Full anatomy (§6). This is the user's home for a feature. |
| **Reference** | Exhaustive, terse, correct. | User, looking up a detail. | Complete and scannable; still no unglossed jargon. |
| **Architecture** | How it is built and why. | Contributor. | Label it "for contributors" at the top. It may be dense — but it is **never the only explanation** of a user-facing feature (R6). |

A performance feature a user can enable (WCOJ, factorized execution) needs a
**Guide/Concept** page, even if an Architecture page also exists.

---

## 8. The review gate

Score every section 🟢/🟡/🔴 and apply:

- **Any 🔴 → the page is RED.** Rewrite that section from the reader's POV.
- **Any 🟡 → fix before merge.**
- **Two tests, run on the opening:**
  - **Read-aloud test.** Read the first paragraph to someone who has never seen
    xlog. If they cannot say, in one breath, what the page lets them do, it is not
    green.
  - **No-source test.** Could the reader reach a working result without opening the
    repository or the source? If not, it is RED.

---

## 9. Worked rewrite: RED → GREEN

**🔴 Before** (actual WCOJ opening — implementer's POV, no example, symbol-led):

> Worst-case optimal join (WCOJ) support is XLOG's route family for multiway join
> shapes where a binary join chain can create avoidable intermediate blow-up. The
> runtime recognizes eligible RIR bodies, dispatches shape-specific CUDA kernels,
> and falls back to ordinary execution when a route is not applicable. The contract
> is opportunistic: a WCOJ decline must preserve the same row set through the
> fallback path. A successful answer does not prove WCOJ fired; use the executor
> counters.

**🟢 After** (value → when → how → confirm → limit):

> Some queries — counting triangles or cycles in a large graph — force an ordinary
> join to build a huge intermediate table before returning a tiny answer. XLOG's
> **worst-case-optimal join (WCOJ)** computes these patterns directly and keeps peak
> memory flat.
>
> **When to use it.** Triangle, 4-cycle, or clique patterns over large, skewed
> graphs (program-analysis or graph workloads).
>
> **How.** Run with `--wcoj`:
> ```bash
> xlog run triangles.xlog --wcoj --stats
> ```
> **Confirm it worked.** In the `--stats` output, `wcoj.triangle_dispatch` is `> 0`.
> If it is `0`, xlog used the ordinary join instead — same answer, no speedup — which
> usually means the rule shape wasn't recognized.
>
> **Limit.** WCOJ targets skewed graphs where the intermediate would blow up; on
> small or uniform inputs the ordinary join is already fine and xlog uses it.

Same facts. The reader can now *use* the feature.

---

## 10. Glossary discipline

Keep one glossary. The first use of any of these terms on a user-facing page must
link to it with a one-line plain gloss — or the term must be replaced with plain
language. Current terms that are used raw and **must** be glossed or removed:

`RIR`, `PIR`, `XGCF`, `EIR`, WCOJ, Free Join, factorized delta, recursive delta,
d-DNNF, weighted model counting, FAEEL, Gelfond-1991, founded support, world view,
witness-multiplied, outcome mask, dispatch counter, kill switch, semi-naive
fixpoint, stratification.

Example gloss: *"a **world view** — the set of models the program considers
possible at once"*, not just "world view".

---

*This document is itself held to the protocol: value first, plain language, one
idea per line, examples over abstraction. If it stops meeting its own bar, fix it.*
