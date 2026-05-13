# W3.7 G4/S4.1 Hand-Split Helper Spike

Branch: `bench-spike/w37-helper-split-hand-g37`
Base: `feat/w33-hg-block-slice-prod` at `035b0713`

G4/S4.1 asked for a hand-written helper split on a CallGraphEdge-style deep join before implementing the AOT rewriter. G2/G3 produced preserved RED spike branches rather than production branches, so this spike is based directly on the G1 production branch. The measured algorithmic question is whether extracting the buried inner sub-pattern into `helper(d,f)` gives the outer rule a smaller top-level relation while preserving the final row set.

## Fixture

Rule shape:

```text
out(a,b,c,d,f) :-
  r_ab(a,b),
  r_bc(b,c),
  r_cd(c,d),
  r_de(d,e),
  r_ef(e,f),
  r_af(a,f).
```

Hand-split shape:

```text
helper(d,f) :- r_de(d,e), r_ef(e,f).
out(a,b,c,d,f) :-
  r_ab(a,b),
  r_bc(b,c),
  r_cd(c,d),
  helper(d,f),
  r_af(a,f).
```

The skew is buried at `d`: every outer `c` reaches `d=0`, and `d=0` fans out over many `e` values. Since `e` is not in the head, the helper relation projects and deduplicates to a small `(d,f)` set before the outer chain consumes it.

The timed split path uses a persisted helper buffer built during setup. This matches the non-recursive helper lifecycle in G4.Q4.3: the helper relation is materialized before the outer rule consumes it. The row-equality check builds the helper before accepting timing data.

See `fixture_shape.tsv`.

## Measurements

Command:

```sh
cargo bench -p xlog-integration --bench wcoj_w37_helper_split -- --output-format bencher
```

Median estimates were read from Criterion `new/estimates.json`.

| workload | unsplit median | hand-split median | speedup | row equality | metric verdict |
|---|---:|---:|---:|---|---|
| callgraph-inner-skew | 35.824 ms | 10.776 ms | 3.324x | PASS | M4.1 PASS |
| heapalloc-inner-skew | 484.894 ms | 10.134 ms | 47.849x | PASS | M4.2 PASS |

The raw table is in `measurements.tsv`.

## Verdict

M4.1 is green for the hand-split CallGraphEdge analog: the split plan is 3.324x faster than the unsplit plan with bit-exact row-set equality.

M4.2 is green for the HeapAllocHelper analog: the split plan is 47.849x faster than the unsplit plan with bit-exact row-set equality.

Next implementation input: the AOT pass should target the `d`-centered extraction pattern measured here and preserve the helper lifecycle used by this spike.

## Verification

- `cargo bench -p xlog-integration --bench wcoj_w37_helper_split --no-run` EXIT 0
- `cargo bench -p xlog-integration --bench wcoj_w37_helper_split -- --output-format bencher` EXIT 0
- Row equality: `callgraph-inner-skew` PASS, `heapalloc-inner-skew` PASS
