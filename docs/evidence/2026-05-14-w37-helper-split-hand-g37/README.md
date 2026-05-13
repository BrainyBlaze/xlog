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

See `fixture_shape.tsv`.

## Measurements

Command:

```sh
cargo bench -p xlog-integration --bench wcoj_w37_helper_split -- --output-format bencher
```

Median estimates were read from Criterion `new/estimates.json`.

| workload | unsplit median | hand-split median | speedup | row equality | metric verdict |
|---|---:|---:|---:|---|---|
| callgraph-inner-skew | 33.590 ms | 15.863 ms | 2.117x | PASS | M4.1 PASS |
| heapalloc-inner-skew | 441.322 ms | 54.646 ms | 8.075x | PASS | M4.2 RED |

The raw table is in `measurements.tsv`.

## Verdict

M4.1 is green for the hand-split CallGraphEdge analog: the split plan is 2.117x faster than the unsplit plan with bit-exact row-set equality.

M4.2 is not green in this spike: the HeapAllocHelper analog reaches 8.075x against the 10x target. The result still confirms the helper-splitting mechanism but does not close the HeapAlloc production metric.

Next implementation input: the AOT pass should first target the `d`-centered extraction pattern measured here, then broaden fixture coverage until the HeapAlloc-style target clears 10x.

## Verification

- `cargo bench -p xlog-integration --bench wcoj_w37_helper_split --no-run` EXIT 0
- `cargo bench -p xlog-integration --bench wcoj_w37_helper_split -- --output-format bencher` EXIT 0
- Row equality: `callgraph-inner-skew` PASS, `heapalloc-inner-skew` PASS
