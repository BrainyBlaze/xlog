#!/usr/bin/env python3
"""Source-B graded marginal readout — live probabilistic-axis numbers on the per-fork selection.

The graded WMC marginal is reachable as a HOST SCALAR via the loss surface:
``P(query) = exp(-nll_loss(query))`` — no CUDA-DLPack capsule, no rebuild, no
torch/cupy (ref scripts/repro/nb2_graded_marginal.py). This wires that readout
onto THIS lane's Source-B per-fork selection (``select_grounding_subset``): each
selected graded-pro fact is grounded as a ``p::atom`` and its consequence
marginal is read as a plain float, filling the ``marginal: None`` slot with a
real number.

Two marginals per selection — both are the vision's "probabilistic feature"
(uncertainty carried THROUGH logic, which argmax would collapse to 1.0):
  - per-fact: P(consequence of one grounded p::atom) via a wrapper-head rule ->
    preserves the graded confidence p (vs argmax-collapse 1.0; gap = discarded
    uncertainty).
  - combined: P(consequence fed by the selected graded neighbourhood) via
    noisy-OR -> REAL WMC (P = 1 - prod_i(1 - p_i)), proving the number is a
    genuine joint over the subset, not an echo of one input.

Honest boundary: the RULES that combine facts in the live loop come from the DTS
program's rule graph (not carried in the Source-B selection payload). Standalone
this exercises the readout on real selected facts with canonical wrapper-head /
noisy-OR derivations; the live rule graph substitutes the actual derivations.
Marginal VALUES are exact-WMC either way.

Run with no args for the self-test; pass a fact_table_dump path to select+read live.
"""
import math
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from p1_source_b_selection import select_grounding_subset

try:
    import pyxlog
    _HAVE_PYXLOG = True
except Exception as _exc:  # pragma: no cover - env-dependent
    _HAVE_PYXLOG = False
    _PYXLOG_ERR = _exc


def marginal(program_src, query):
    """Exact-WMC marginal P(query) as a host float, via nll_loss = -log P."""
    prog = pyxlog.Program.compile(program_src + f"\n?- {query}.\n")
    return math.exp(-prog.nll_loss(query))


def per_fact_marginal(rec):
    """P(consequence of this one grounded p::atom) through a wrapper-head rule.

    Carries the graded pre-collapse confidence p through to the derived
    consequence: P == p (preserved), where argmax-collapse would give 1.0.
    """
    pred, a0, a1, p = rec["pred_id"], rec["arg0"], rec["arg1"], rec["p"]
    src = f"{p} :: q{pred}({a0}, {a1}).\nd{pred}(X, Y) :- q{pred}(X, Y)."
    return marginal(src, f"d{pred}({a0}, {a1})")


def combined_marginal(recs):
    """Noisy-OR joint over the selected graded neighbourhood feeding one head.

    Real WMC: P(head) = 1 - prod_i (1 - p_i). Demonstrates the marginal is a
    genuine joint over the selected subset, not an echo of a single input.
    """
    facts, rules = [], []
    for i, r in enumerate(recs):
        facts.append(f"{r['p']} :: s{i}().")
        rules.append(f"hh() :- s{i}().")
    return marginal("\n".join(facts + rules), "hh()")


def enrich_selection_marginals(selection):
    """Fill each selected graded record's ``marginal`` via the nll_loss readout.

    Returns (enriched_records, combined_marginal). Mutates copies, not inputs.
    """
    enriched = []
    for rec in selection["graded"]:
        out = dict(rec)
        out["marginal"] = round(per_fact_marginal(rec), 6)
        enriched.append(out)
    combined = round(combined_marginal(selection["graded"]), 6) if enriched else None
    return enriched, combined


def marginals_from_sink(sink_record, *, k=32):
    """Full live path: a wired per-step sink payload -> per-fork graded marginals.

    Drives this lane's ``consume_source_b_sink`` (one bounded per-fork selection
    per clone-fork, carrying clone_id/resolution) and enriches each fork's
    selection with real graded WMC marginals via the host-scalar nll_loss path.
    This is the consumer end of the live probabilistic axis: it takes the exact
    ``{step_index, facts, fork_anchors}`` payload dts-dlm emits (runtime.py:6077)
    and returns numbers, keyed by the same clone_id the engine/routing index on.
    """
    from p1_source_b_selection import consume_source_b_sink

    out = []
    for sel in consume_source_b_sink(sink_record, k=k):
        enriched, combined = enrich_selection_marginals(sel)
        out.append({
            "step_index": sel.get("step_index"),
            "clone_id": sel.get("clone_id"),
            "resolution": sel.get("resolution"),
            "anchor_fact_id": sel.get("anchor_fact_id"),
            "combined_marginal": combined,
            "marginals": {r["fact_id"]: r["marginal"] for r in enriched},
        })
    return out


def _atom(pred_id, arg0, arg1):
    return f"q{pred_id}({arg0}, {arg1})"


def _rule_atom_str(atom):
    """One rule atom -> pyxlog: q{pred_id}(A, B). args are var-name strings
    (uppercase, shared across atoms = join) or int constants."""
    return f"q{atom['pred_id']}({', '.join(str(a) for a in atom['args'])})"


def rule_records_to_pyxlog(rules):
    """Map dts-dlm normalized rule records -> pyxlog rule strings.

    Rule record (runtime.py:_source_b_rule_graph_records):
      {rule_id, head:{pred_id,args}, body:[{pred_id,args}], state, weight, ...}
    args = variable-name strings (A,B,C; same name across atoms = join) or int
    constants; predicates map to q{pred_id} (matching the fact grounding). The
    producer already excludes single-premise rules; existential/body-only joins
    are emitted (the soft-WMC readout handles them), with tractability guarded by
    the bounded Source-B selection, not here.
    """
    return [
        f"{_rule_atom_str(r['head'])} :- {', '.join(_rule_atom_str(b) for b in r['body'])}."
        for r in rules
    ]


def _ground_rule_heads(rule, facts):
    """All ground head atoms of ``rule`` derivable from ``facts`` — a binding join
    over the body atoms (shared var names unify, constants must match). Standard
    Datalog grounding; handles chain/existential joins (a head var bound by a
    non-anchor body atom) that anchor-only unification misses.

    Fail-closed on arity: fact records are binary (arg0, arg1), so a body atom
    with arity != 2 cannot be grounded here — the rule is skipped entirely rather
    than silently mis-bound (zip against (arg0, arg1) would drop or misalign args).
    """
    if any(len(atom["args"]) != 2 for atom in rule["body"]):
        return set()
    by_pred = {}
    for f in facts:
        by_pred.setdefault(f["pred_id"], []).append((f["arg0"], f["arg1"]))
    body, heads = rule["body"], set()

    def rec(i, binding):
        if i == len(body):
            hargs = []
            for x in rule["head"]["args"]:
                if isinstance(x, str):
                    if x not in binding:
                        return  # head var unbound by body -> not groundable here
                    hargs.append(binding[x])
                else:
                    hargs.append(x)
            heads.add(f"q{rule['head']['pred_id']}({', '.join(str(a) for a in hargs)})")
            return
        atom = body[i]
        for c0, c1 in by_pred.get(atom["pred_id"], []):
            nb, ok = dict(binding), True
            for var, const in zip(atom["args"], (c0, c1)):
                if isinstance(var, str):
                    if nb.get(var, const) != const:
                        ok = False
                        break
                    nb[var] = const
                elif var != const:
                    ok = False
                    break
            if ok:
                rec(i + 1, nb)

    rec(0, {})
    return heads


def derive_consequence_queries(anchor_rec, neighbour_recs, rules, *, max_rounds=4):
    """Ground consequence atoms derivable from anchor + neighbours via the rules.

    Semi-naive fixpoint (bounded rounds): derived heads are fed back as facts so
    multi-hop/recursive rule chains surface transitive consequences as queries
    (pyxlog evaluates them regardless; this only decides WHICH queries to read).
    Which consequences actually DEPEND on the contested anchor is revealed by the
    per-clone delta (withdrawn anchor -> dependent heads collapse, independent
    heads unchanged), so no separate provenance tracking is needed.
    """
    facts = [anchor_rec] + list(neighbour_recs)
    queries: set[str] = set()
    for _ in range(max_rounds):
        new = set()
        for r in rules:
            new |= _ground_rule_heads(r, facts)
        fresh = new - queries
        if not fresh:
            break
        queries |= fresh
        for q in fresh:  # feed derived heads back as (binary) facts for hop N+1
            pred, args = q[1:].split("(", 1)
            parts = [a.strip() for a in args.rstrip(")").split(",")]
            if len(parts) == 2:
                facts.append({"pred_id": int(pred), "arg0": int(parts[0]),
                              "arg1": int(parts[1])})
    return sorted(queries)


def _clone_src(anchor_rec, neighbour_recs, rules, resolution):
    """Assemble one clone's program source: anchor@resolution + neighbours + rules.

    ``pro`` clone asserts the contested anchor with its pro mass; ``contra`` clone
    withdraws it (omitted), matching ``build_resolution_variant_row`` in the
    clone-seed lane. This src-assembly is THIS lane's job; the readout (WMC
    marginal) is the engine primitive's (xlog_marginal_readout, @xlog-claude).

    Live facts tables carry several fact_ids for the SAME ground atom (per-step
    re-admissions with fresh confidences), so neighbours are deduped by ground
    atom with a max-merge — matching dts `confidence_join` (pointwise max) — and
    the anchor's own ground atom is reserved for the resolution: a duplicate of
    it in the neighbour set would re-assert the withdrawn anchor in the contra
    clone (faking delta=0) and duplicate `p::atom` lines break the engine's
    WMC circuit compile (var-count mismatch).
    """
    anchor_key = (anchor_rec["pred_id"], anchor_rec["arg0"], anchor_rec["arg1"])
    merged = {}
    for n in neighbour_recs:
        key = (n["pred_id"], n["arg0"], n["arg1"])
        if key == anchor_key:
            continue
        merged[key] = max(merged.get(key, 0.0), n["p"])
    facts = []
    if resolution == "pro":
        facts.append(f"{anchor_rec['pro']} :: {_atom(*anchor_key)}.")
    facts += [f"{p} :: {_atom(*key)}." for key, p in sorted(merged.items())]
    return "\n".join(facts + list(rules))


def per_clone_consequence_marginals(anchor_rec, neighbour_recs, rules, query):
    """Per-clone consequence marginal — where the epistemic axis meets the probabilistic.

    Builds the two clone sources (anchor resolved pro vs contra, same neighbours +
    rules) and reads the graded consequence marginal per clone. A rule whose
    consequence depends on the anchor yields DIFFERENT marginals per clone ->
    per-clone divergence (|Δ| > 0), the coupling the fact+fork-anchor-only payload
    cannot show. ``rules`` is a list of pyxlog rule strings (from the live DTS rule
    graph via the mapper, or canonical here to prove the seam).

    Delegates the readout to the engine primitive ``xlog_marginal_readout``
    (@xlog-claude's lane) when importable; falls back to the local nll_loss readout
    standalone. A withdrawn premise floors P at ~1e-38, treated as 0.

    Returns {"pro": P_pro, "contra": P_contra, "delta": |P_pro - P_contra|}.
    """
    pro_src = _clone_src(anchor_rec, neighbour_recs, rules, "pro")
    contra_src = _clone_src(anchor_rec, neighbour_recs, rules, "contra")
    try:
        from xlog_marginal_readout import per_clone_marginals as _engine
        r = _engine(pro_src, contra_src, [query])[query]
        pro, contra = r["pro"], (0.0 if r["contra"] < 1e-9 else r["contra"])
    except ImportError:
        pro = marginal(pro_src, query)
        contra_raw = marginal(contra_src, query)
        contra = 0.0 if contra_raw < 1e-9 else contra_raw
    return {"pro": round(pro, 6), "contra": round(contra, 6),
            "delta": round(abs(pro - contra), 6)}


def per_clone_divergence_from_step(step, fork, *, k=32):
    """Live per-clone consequence divergence for one fork of a rule-graph step.

    step: {step_index, facts, fork_anchors, rules} (dts-dlm rule-graph sink).
    fork: one fork_anchor {fact_id, clone_id, resolution}.
    Selects the anchor's Source-B neighbourhood, maps the rule graph to pyxlog,
    grounds the anchor-reachable consequence queries, and reads per-clone
    {pro, contra, delta} for each via the engine readout primitive. delta > 0 marks
    a consequence whose marginal genuinely depends on the contested resolution —
    the epistemic∩probabilistic coupling. Returns None when the step has no rules
    or no anchor-reachable consequence.
    """
    rules = step.get("rules") or []
    if not rules:
        return None
    anchor = {f["fact_id"]: f for f in step["facts"]}.get(fork["fact_id"])
    if anchor is None:
        return None
    sel = select_grounding_subset(step["facts"], anchor_fact_id=fork["fact_id"], k=k)
    rule_strs = rule_records_to_pyxlog(rules)
    queries = derive_consequence_queries(anchor, sel["graded"], rules)
    if not queries:
        return None
    pro_src = _clone_src(anchor, sel["graded"], rule_strs, "pro")
    contra_src = _clone_src(anchor, sel["graded"], rule_strs, "contra")
    try:
        from xlog_marginal_readout import per_clone_marginals as _engine
        raw = _engine(pro_src, contra_src, queries)
    except ImportError:
        raw = {q: {"pro": marginal(pro_src, q), "contra": marginal(contra_src, q)}
               for q in queries}
    out = {}
    for q, v in raw.items():
        contra = 0.0 if v["contra"] < 1e-9 else v["contra"]
        out[q] = {"pro": round(v["pro"], 6), "contra": round(contra, 6),
                  "delta": round(abs(v["pro"] - contra), 6)}
    return {"step_index": step.get("step_index"), "clone_id": fork.get("clone_id"),
            "resolution": fork.get("resolution"), "anchor_fact_id": fork["fact_id"],
            "consequences": out,
            "coupled": {q: v for q, v in out.items() if v["delta"] > 1e-6}}


def _config_src(anchor_recs, resolutions, neighbour_recs, rules):
    """Program source for ONE clone-config: N anchors each resolved per config.

    Generalizes ``_clone_src`` from one anchor to a config over N contested
    anchors (resolution per anchor: ``pro`` = assert with pro mass, ``contra`` =
    withdraw). Neighbours are deduped by ground atom (max-merge = confidence_join)
    with ALL anchor atoms reserved for their resolutions.
    """
    anchor_keys = {(a["pred_id"], a["arg0"], a["arg1"]): a for a in anchor_recs}
    merged = {}
    for n in neighbour_recs:
        key = (n["pred_id"], n["arg0"], n["arg1"])
        if key in anchor_keys:
            continue
        merged[key] = max(merged.get(key, 0.0), n["p"])
    facts = []
    for a, res in zip(anchor_recs, resolutions):
        if res == "pro":
            facts.append(f"{a['pro']} :: {_atom(a['pred_id'], a['arg0'], a['arg1'])}.")
    facts += [f"{p} :: {_atom(*key)}." for key, p in sorted(merged.items())]
    return "\n".join(facts + list(rules))


def per_config_metastability_from_step(step, *, k=32, max_anchors=4):
    """Metastability observable: consequence-marginal landscape over clone-CONFIGS.

    Generalizes the per-fork measurement (1 anchor, 2 clones) to the joint
    clone-config space (N contested anchors, 2^N configs; N capped at
    ``max_anchors``, skipped anchors reported — no silent truncation). For each
    config: every anchor resolved pro/contra jointly, one bounded neighbourhood,
    live rules, consequence marginals via the engine readout primitive.

    Metastability = distinct quasi-stable world-states: configs whose marginal
    vectors form separated modes while carrying comparable plausibility. Reported
    per config: {config, plausibility (product over anchors: pro-mass if pro else
    contra-mass — the seed weight of that world), marginals}; plus the max
    pairwise L-inf distance between config marginal-vectors (the mode gap; with
    1 anchor this reduces exactly to the per-fork delta).
    """
    rules = step.get("rules") or []
    forks = step.get("fork_anchors") or []
    if not rules or not forks:
        return None
    by_id = {f["fact_id"]: f for f in step["facts"]}
    anchor_ids = sorted({fk["fact_id"] for fk in forks if fk["fact_id"] in by_id})
    skipped = anchor_ids[max_anchors:]
    anchor_ids = anchor_ids[:max_anchors]
    anchors = [by_id[i] for i in anchor_ids]
    if not anchors:
        return None
    # one bounded neighbourhood shared across configs (per-anchor budget k // N)
    seen, neighbours = set(), []
    for aid in anchor_ids:
        sel = select_grounding_subset(step["facts"], anchor_fact_id=aid,
                                      k=max(2, k // len(anchor_ids)))
        for g in sel["graded"]:
            if g["fact_id"] not in seen:
                seen.add(g["fact_id"])
                neighbours.append(g)
    rule_strs = rule_records_to_pyxlog(rules)
    queries = derive_consequence_queries(
        anchors[0], anchors[1:] + neighbours, rules)
    if not queries:
        return None
    configs = []
    for mask in range(2 ** len(anchors)):
        res = ["pro" if (mask >> i) & 1 == 0 else "contra"
               for i in range(len(anchors))]
        src = _config_src(anchors, res, neighbours, rule_strs)
        try:
            from xlog_marginal_readout import readout_marginals as _read
            raw = _read(src, queries)
        except ImportError:
            raw = {q: marginal(src, q) for q in queries}
        plaus = 1.0
        for a, r in zip(anchors, res):
            plaus *= a["pro"] if r == "pro" else a["contra"]
        configs.append({
            "config": {a["fact_id"]: r for a, r in zip(anchors, res)},
            "plausibility": round(plaus, 6),
            "marginals": {q: round(0.0 if v < 1e-9 else v, 6)
                          for q, v in raw.items()},
        })
    gap = 0.0
    for i in range(len(configs)):
        for j in range(i + 1, len(configs)):
            for q in queries:
                gap = max(gap, abs(configs[i]["marginals"][q]
                                   - configs[j]["marginals"][q]))
    return {"step_index": step.get("step_index"), "anchor_fact_ids": anchor_ids,
            "anchors_skipped": skipped, "n_configs": len(configs),
            "queries": queries, "configs": configs,
            "max_pairwise_gap": round(gap, 6)}


def fork_anchor_marginals(step, *, k=32):
    """STAGE-A writeback input: per-clone marginal of each fork anchor's OWN atom.

    Fills the routing records' reserved ``marginal: None`` slot (worldcloner
    contract: clone_id + target_mode = clone-seed lane; marginal = engine lane):
    for each fork anchor, P(anchor's ground atom | that clone's world) — the
    anchor asserted with pro mass in the ``pro`` clone, withdrawn in ``contra``.
    With a recursive rule graph the contra value is the re-support floor (the
    world pushing back on the withdrawal, e.g. 0.7143 live); without back-edges
    it collapses to 0. One compile per fork (each fork reads only its own
    resolution's world). Returns [{fact_id, clone_id, resolution, marginal}]
    ready for the STAGE-A fact-table writeback f(clone-marginals, plausibility).
    """
    rules = step.get("rules") or []
    forks = step.get("fork_anchors") or []
    by_id = {f["fact_id"]: f for f in step["facts"]}
    rule_strs = rule_records_to_pyxlog(rules)
    out = []
    for fork in forks:
        anchor = by_id.get(fork["fact_id"])
        if anchor is None:
            continue
        sel = select_grounding_subset(step["facts"], anchor_fact_id=fork["fact_id"], k=k)
        anchor_atom = _atom(anchor["pred_id"], anchor["arg0"], anchor["arg1"])
        src = _clone_src(anchor, sel["graded"], rule_strs, fork["resolution"])
        try:
            from xlog_marginal_readout import readout_marginals as _read
            p = _read(src, [anchor_atom])[anchor_atom]
        except ImportError:
            p = marginal(src, anchor_atom)
        out.append({"fact_id": fork["fact_id"], "clone_id": fork.get("clone_id"),
                    "resolution": fork.get("resolution"),
                    "marginal": round(0.0 if p < 1e-9 else p, 6)})
    return out


def marginals_from_per_step_dump(payloads, *, k=32):
    """Ingest a ``source_b_per_step.json`` list -> per-step graded marginals.

    Each element is one ``{step_index, facts, fork_anchors}`` record emitted by
    dts-dlm's ``--dump-source-b-per-step`` (runtime.py:6418). Runs
    ``marginals_from_sink`` on each, so the live handoff is turnkey:
    dumped file -> per-step, per-fork graded marginals keyed by clone_id.
    """
    return [
        {"step_index": rec.get("step_index"), "forks": marginals_from_sink(rec, k=k)}
        for rec in payloads
    ]


def _self_test():
    if not _HAVE_PYXLOG:
        print(f"SKIP: pyxlog unavailable ({_PYXLOG_ERR})")
        return 0
    # anchor = a hard-contested fact (Source-A fork point) over entities {1,2};
    # graded-pro neighbours share entity 1 (co-feed the anchor's derivations);
    # one non-neighbour graded fact; one near-committed (excluded from grounding).
    facts = [
        {"fact_id": 0, "pred_id": 5, "arg0": 1, "arg1": 2, "pro": 0.60, "contra": 0.30},  # anchor (contested)
        {"fact_id": 2, "pred_id": 7, "arg0": 1, "arg1": 9, "pro": 0.55, "contra": 0.0},   # neighbour (shares 1)
        {"fact_id": 5, "pred_id": 7, "arg0": 1, "arg1": 11, "pro": 0.40, "contra": 0.0},  # neighbour (shares 1)
        {"fact_id": 3, "pred_id": 7, "arg0": 7, "arg1": 8, "pro": 0.50, "contra": 0.0},   # non-neighbour graded
        {"fact_id": 4, "pred_id": 7, "arg0": 20, "arg1": 21, "pro": 0.99, "contra": 0.0}, # near-committed -> excluded
    ]
    sel = select_grounding_subset(facts, anchor_fact_id=0, k=4)
    assert sel["mode"] == "per_fork_anchor", sel
    enriched, combined = enrich_selection_marginals(sel)

    # every selected graded record now carries a real graded marginal (not None)
    assert enriched and all(r["marginal"] is not None for r in enriched), enriched
    # per-fact wrapper-head marginal preserves the graded p (uncertainty carried
    # through the logic), NOT collapsed to argmax 1.0
    by_id = {r["fact_id"]: r for r in enriched}
    assert abs(by_id[2]["marginal"] - 0.55) < 1e-3, by_id[2]
    assert abs(by_id[5]["marginal"] - 0.40) < 1e-3, by_id[5]
    assert 4 not in by_id, by_id  # near-committed never grounded

    # combined = real WMC noisy-OR over the selected graded subset (a genuine
    # joint, provably != any single input)
    ps = [r["p"] for r in sel["graded"]]
    analytic = 1.0
    for p in ps:
        analytic *= (1.0 - p)
    analytic = 1.0 - analytic
    assert abs(combined - analytic) < 1e-3, (combined, analytic)
    assert combined > max(ps) + 1e-6, (combined, ps)  # joint strictly exceeds any input

    print("selected graded ids:", [r["fact_id"] for r in enriched])
    print("per-fact graded marginals (p preserved through logic):",
          {r["fact_id"]: r["marginal"] for r in enriched})
    print(f"combined noisy-OR marginal over selection: {combined:.4f} "
          f"(analytic {analytic:.4f}); argmax-collapse would give 1.0")
    print(f"discarded-uncertainty gap on fact 2: {1.0 - by_id[2]['marginal']:.4f}")

    # FULL LIVE PATH: exact wired sink payload shape (dts-dlm runtime.py:6077) ->
    # consume_source_b_sink -> per-fork graded marginals, keyed by clone_id.
    sink = {
        "step_index": 3,
        "facts": facts,
        "fork_anchors": [{"fact_id": 0, "clone_id": 0, "resolution": "pro"},
                         {"fact_id": 0, "clone_id": 1, "resolution": "contra"}],
    }
    forks = marginals_from_sink(sink, k=4)
    assert len(forks) == 2, forks                                   # one per clone-fork
    assert {f["clone_id"] for f in forks} == {0, 1}, forks
    assert {f["resolution"] for f in forks} == {"pro", "contra"}, forks
    for f in forks:
        assert f["step_index"] == 3, f
        assert f["combined_marginal"] is not None, f
        assert all(m is not None for m in f["marginals"].values()), f
    print("live-path forks (clone_id -> combined graded marginal):",
          {f["clone_id"]: f["combined_marginal"] for f in forks})

    # DUMP INGEST: a source_b_per_step.json is a LIST of the above sink records;
    # marginals_from_per_step_dump iterates them (turnkey live handoff).
    dump = [dict(sink, step_index=3), dict(sink, step_index=4)]
    per_step = marginals_from_per_step_dump(dump, k=4)
    assert [s["step_index"] for s in per_step] == [3, 4], per_step
    assert all(len(s["forks"]) == 2 for s in per_step), per_step
    print("dump-ingest steps:", [s["step_index"] for s in per_step],
          "forks/step:", [len(s["forks"]) for s in per_step])

    # PER-CLONE DIVERGENCE (the coupling the fact-only payload cannot show): a rule
    # making the consequence depend on the contested anchor -> pro-clone and
    # contra-clone yield DIFFERENT consequence marginals (|Δ| > 0). Needs the live
    # DTS rule graph in the sink; canonical rule here proves the seam.
    anchor_pc = {"pred_id": 100, "arg0": 1, "arg1": 2, "pro": 0.6, "contra": 0.35}
    nbr_pc = [{"pred_id": 200, "arg0": 2, "arg1": 3, "p": 0.7}]
    div = per_clone_consequence_marginals(
        anchor_pc, nbr_pc, ["c(X, Z) :- q100(X, Y), q200(Y, Z)."], "c(1, 3)")
    assert abs(div["pro"] - 0.42) < 1e-3 and div["contra"] == 0.0, div
    assert div["delta"] > 0.4, div
    print("per-clone divergence (rule-graph coupling): pro=%.3f contra=%.3f delta=%.3f"
          % (div["pro"], div["contra"], div["delta"]))

    # LIVE per-clone divergence from a rule-graph STEP in dts-dlm sink format
    # (facts + fork_anchors + normalized rules): a chain-join rule couples the
    # consequence to the contested anchor -> pro/contra diverge, end-to-end.
    rg_step = {
        "step_index": 7,
        "facts": [
            {"fact_id": 35, "pred_id": 100, "arg0": 1, "arg1": 2, "pro": 0.6, "contra": 0.30},
            {"fact_id": 40, "pred_id": 200, "arg0": 2, "arg1": 3, "pro": 0.7, "contra": 0.0},
            {"fact_id": 41, "pred_id": 200, "arg0": 8, "arg1": 9, "pro": 0.5, "contra": 0.0},
        ],
        "fork_anchors": [{"fact_id": 35, "clone_id": 70, "resolution": "pro"},
                         {"fact_id": 35, "clone_id": 71, "resolution": "contra"}],
        "rules": [{"rule_id": 1, "head": {"pred_id": 300, "args": ["A", "C"]},
                   "body": [{"pred_id": 100, "args": ["A", "B"]},
                            {"pred_id": 200, "args": ["B", "C"]}],
                   "state": 1, "weight": 1.0, "head_pred_sort": 0}]
    }
    assert rule_records_to_pyxlog(rg_step["rules"]) == [
        "q300(A, C) :- q100(A, B), q200(B, C)."], rule_records_to_pyxlog(rg_step["rules"])
    pcd = per_clone_divergence_from_step(rg_step, rg_step["fork_anchors"][0], k=8)
    assert pcd is not None and "q300(1, 3)" in pcd["consequences"], pcd
    c = pcd["consequences"]["q300(1, 3)"]
    assert abs(c["pro"] - 0.42) < 1e-3 and c["contra"] == 0.0 and c["delta"] > 0.4, c
    assert "q300(1, 3)" in pcd["coupled"], pcd
    print("LIVE per-clone divergence (rule-graph step): q300(1,3) pro=%.3f contra=%.3f delta=%.3f (coupled)"
          % (c["pro"], c["contra"], c["delta"]))

    # METASTABILITY over clone-CONFIGS: 2 contested anchors -> 4 joint configs.
    # Consequence q500 depends on BOTH anchors (conjunction) -> its marginal
    # ranges over {p1*p2, 0} across configs; per-anchor consequences split the
    # landscape into distinct quasi-stable modes.
    meta_step = {
        "step_index": 9,
        "facts": [
            {"fact_id": 1, "pred_id": 100, "arg0": 1, "arg1": 2, "pro": 0.6, "contra": 0.30},
            {"fact_id": 2, "pred_id": 200, "arg0": 2, "arg1": 3, "pro": 0.8, "contra": 0.40},
            {"fact_id": 3, "pred_id": 300, "arg0": 1, "arg1": 3, "pro": 0.5, "contra": 0.0},
        ],
        "fork_anchors": [
            {"fact_id": 1, "clone_id": 2, "resolution": "pro"},
            {"fact_id": 1, "clone_id": 3, "resolution": "contra"},
            {"fact_id": 2, "clone_id": 4, "resolution": "pro"},
            {"fact_id": 2, "clone_id": 5, "resolution": "contra"},
        ],
        "rules": [{"rule_id": 1, "head": {"pred_id": 500, "args": ["A", "C"]},
                   "body": [{"pred_id": 100, "args": ["A", "B"]},
                            {"pred_id": 200, "args": ["B", "C"]}],
                   "state": 1, "weight": 1.0, "head_pred_sort": 0}],
    }
    meta = per_config_metastability_from_step(meta_step, k=4)
    assert meta["n_configs"] == 4 and meta["anchor_fact_ids"] == [1, 2], meta
    by_cfg = {tuple(sorted(c["config"].items())): c for c in meta["configs"]}
    both_pro = by_cfg[((1, "pro"), (2, "pro"))]
    assert abs(both_pro["marginals"]["q500(1, 3)"] - 0.6 * 0.8) < 1e-3, both_pro
    assert abs(both_pro["plausibility"] - 0.6 * 0.8) < 1e-6, both_pro
    for key, cfg in by_cfg.items():
        if key != ((1, "pro"), (2, "pro")):
            assert cfg["marginals"]["q500(1, 3)"] == 0.0, cfg  # conjunction collapses
    assert abs(meta["max_pairwise_gap"] - 0.48) < 1e-3, meta["max_pairwise_gap"]
    print("metastability landscape (2 anchors, 4 configs): q500 modes",
          sorted({c["marginals"]["q500(1, 3)"] for c in meta["configs"]}),
          "gap=%.3f" % meta["max_pairwise_gap"])
    # 1-anchor reduction == per-fork delta
    meta1 = per_config_metastability_from_step(rg_step, k=8)
    assert meta1["n_configs"] == 2, meta1
    assert abs(meta1["max_pairwise_gap"] - pcd["consequences"]["q300(1, 3)"]["delta"]) < 1e-6
    print("1-anchor reduction: max_pairwise_gap == per-fork delta =",
          meta1["max_pairwise_gap"])

    # STAGE-A writeback input: per-clone marginal of the anchor's OWN atom.
    # Acyclic graph: pro-clone = anchor's pro (base fact), contra-clone = 0
    # (no back-edge to re-support the withdrawn anchor).
    fam = fork_anchor_marginals(rg_step, k=8)
    by_clone = {r["clone_id"]: r for r in fam}
    assert abs(by_clone[70]["marginal"] - 0.6) < 1e-3, by_clone
    assert by_clone[71]["marginal"] == 0.0, by_clone
    assert by_clone[70]["resolution"] == "pro" and by_clone[71]["resolution"] == "contra"
    print("STAGE-A fork-anchor marginals (acyclic): pro-clone=%.3f contra-clone=%.3f"
          % (by_clone[70]["marginal"], by_clone[71]["marginal"]))
    print("SELF-TEST PASS")
    return 0


def main():
    if len(sys.argv) > 1:
        import json
        d = json.load(open(sys.argv[1]))
        k = int(sys.argv[2]) if len(sys.argv) > 2 else 32
        if not _HAVE_PYXLOG:
            print(f"SKIP marginals: pyxlog unavailable ({_PYXLOG_ERR})")
            return 0
        # A source_b_per_step.json is a LIST of per-step sink records; a
        # fact_table_dump is a dict with "facts". Dispatch on shape.
        if isinstance(d, list):
            print(json.dumps(marginals_from_per_step_dump(d, k=k), indent=2))
            return 0
        sel = select_grounding_subset(d["facts"], k=k)
        enriched, combined = enrich_selection_marginals(sel)
        print(json.dumps({
            "mode": sel["mode"], "anchor_fact_id": sel["anchor_fact_id"],
            "n_selected": len(enriched), "combined_marginal": combined,
            "marginals": {r["fact_id"]: r["marginal"] for r in enriched},
        }, indent=2))
        return 0
    return _self_test()


if __name__ == "__main__":
    sys.exit(main())
