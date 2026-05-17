# G39 W65 Sort-Label RCA

Date: 2026-05-17.
Branch: `feat/w65-sort-label-propagation-g39`.
Base: `feat/w3-bundle-integration @ c1689d70`.

## Summary

W6.5's "sort-map misses" flood is not caused by an existing xlog sort-label
field being dropped. Before this step, xlog had no per-column sort-label field:
`Schema` carried only column name/type pairs plus key columns, and pyxlog query
results exposed only `relation_name`, `columns`, `tensors`, `num_rows`, and
`is_true`.

This means DTS-DLM could not observe authoritative per-column sort metadata from
xlog at all. The xlog-side fix is therefore additive:

- `Schema::new(...)` now assigns one non-empty sort label per column.
- Schema projection/join helpers preserve labels when they already exist.
- Query result schemas label output columns from the query variables.
- `xlog_gpu::logic::LogicQueryResult` and pyxlog `LogicQueryResult` expose
  `sort_labels`.

## Root Cause

DTS-DLM's current warning is produced after xlog evaluation, inside
`src/dts_dlm/propagate/xlog_executor.py::enrich_support_sorts`. That function
builds sort metadata from typed DTS tables (`facts`, `rules`, and optional
`rule_binding`) and increments `_sort_default_count` when a raw triple from xlog
cannot be found in the typed `sort_map`.

The current DTS consumer path does not read any xlog sort-label metadata:

- Stage 4 materializes xlog query outputs from `eval_result.queries[qi].tensors`.
- It uses `qr.num_rows` as an empty-result guard.
- It does not reference `qr.columns` for sort resolution.
- It has no reference to `qr.sort_labels`.

Therefore, adding consumer-visible sort labels on the xlog/pyxlog API makes the
metadata available and testable, but it cannot by itself change
`_sort_default_count` in DTS-DLM. The warning count can only reach zero if either
the xlog tensor payload no longer contains any raw triple that DTS cannot
resolve from typed tables, or a future DTS-side consumer change uses
`qr.sort_labels` in `enrich_support_sorts`.

## Answer to G_W65 Questions

**Q_W65.1: Root cause of un-inferred sort labels on padding columns?**

xlog had no sort-label metadata layer to infer from. DTS-DLM's "padding" language
refers to raw support/usable triples that its post-hoc sort resolver cannot map
to typed facts. The existing pyxlog payload had tensors but no per-column labels
that could mark support head, rule id, witness predicate, or witness argument
columns as authoritative roles.

**Q_W65.2: Does fixing schema layer eliminate diagnostic?**

No, not with DTS-DLM source frozen. The schema-layer fix emits labels, but
`xlog_executor.py` does not consume them. The diagnostic is data-path-driven by
typed table lookup misses, not by xlog schema inspection.

**Q_W65.3: Does fix preserve existing schema call sites?**

Yes. `Schema::new(Vec<(String, ScalarType)>)` remains source-compatible and now
derives default non-empty labels from column names. Existing call sites do not
need an extra argument. Explicit labels are additive via `Schema::with_sort_labels`.

## Evidence

Implemented xlog-side certs:

- `crates/xlog-integration/tests/test_w65_sort_label.rs`
  - `w65_schema_new_assigns_non_default_sort_labels`
  - `w65_query_output_sort_labels_follow_query_variables`
  - `w65_pyxlog_logic_query_result_exposes_sort_labels`

Static DTS-DLM unchanged check:

```text
git -C /home/dev/projects/dts-dlm status --short -- src/dts_dlm/propagate/xlog_executor.py
git -C /home/dev/projects/dts-dlm diff -- src/dts_dlm/propagate/xlog_executor.py
```

Both commands produced no output before this RCA was written.

Static consumer-path check:

```text
rg -n "sort_labels" /home/dev/projects/dts-dlm/src/dts_dlm/propagate/xlog_executor.py
```

No matches. The current DTS consumer cannot be affected by a newly exposed
`qr.sort_labels` field without a DTS source change, which is out of bounds for
G_W65 under goal-039 lock 17.

## Follow-up Runtime Diagnosis

A branch-local pyxlog build was loaded through `/tmp/g39-pyxlog` and used for a
bounded DTS-DLM arm-C replay:

```text
PYTHONPATH=/tmp/g39-pyxlog:/home/dev/projects/dts-dlm/src
XLOG_CUBIN_DIR=.../target/release/build/xlog-cuda-43b482a33001fc07/out
python3 -m dts_dlm.pilots.m37c_prime_pilot \
  --run-id g39-w65-5doc-20260517-r1 \
  --eval-path /tmp/g39-w65-corpus-5.jsonl \
  --out-root /tmp/g39-w65-m37c-prime \
  --arms C
```

Result:

```text
RC=0
SORT_WARNINGS=11
Sort enrichment warning counts: 4 x 6 calls, 48 x 5 calls
Total default events: 264
```

An in-process diagnostic wrapper around
`dts_dlm.propagate.xlog_executor.enrich_support_sorts` classified every warning
event:

```text
binding_inactive: 264
binding_missing: 0
no_binding_missing: 0
usable_missing: 0
rule_id_missing_rows: 0
```

Representative row:

```text
support body_len=1
rule_id=4
head=(10006, 10022, 10022)
witness0=(10012, 10022, 10022)
head_arg_binding={0: (0, 0), 1: (1, 1)}
event=first_binding_inactive:h1->b1a1
```

So the live warning source is not an xlog sort-label propagation miss and not a
missing raw triple. DTS-DLM receives a `support_1` row for a rule whose
`rule_binding` says head argument 1 is sourced from body position 1. During
`normalize_support`, body position 1 is inactive for `support_1`, so
`enrich_support_sorts` uses the rule-derived default sort and increments the
warning counter.

The upstream source-generation reason is visible in DTS-DLM
`src/dts_dlm/propagate/xlog_executor.py`:

- Lines 836-854 group compiled rules by body length, but build shared
  `wmir_body_{pos}` maps containing all rules with that position.
- Lines 894-898 generate one `support_N` clause per body length.
- Lines 981-986 emit each `support_N` clause against the shared
  `wmir_body_{pos}` relations without a body-length membership guard for `RId`.

Consequently, `support_1` can match `wmir_body_0` rows belonging to 2-body rules
and emit partial unary support rows. DTS enrichment then sees a `body_len=1`
support row for a rule whose authoritative binding still references body
position 1, exactly producing the `binding_inactive` warnings above.

## Boundary

The remaining M_W65.1 failure is therefore outside xlog's Datalog runtime
semantics. pyxlog evaluates the source program it is given; filtering
`support_1` rows by whether `RId` also appears in `wmir_body_1` would make pyxlog
violate the source Datalog program. The minimal behavioral fix belongs at the
DTS-DLM AST/source-generation boundary: either emit per-body-length body-map
relations (for example `wmir_body_0_len_1`) or add an explicit body-length/RId
membership guard to each generated `support_N` clause.

Under goal-039 lock 17, DTS-DLM source mutation is out of bounds for G_W65.
