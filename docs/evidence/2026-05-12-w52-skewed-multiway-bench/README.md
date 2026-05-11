# W5.2 Skewed Multiway Bench Evidence

Date: 2026-05-12

Branch: `feat/w52-skewed-multiway-bench`

Plan: `docs/plans/2026-05-11-w52-bench-plan.md`

Bench target: `crates/xlog-integration/benches/w52_skewed_multiway_bench.rs`

## Step 2 Skeleton

Step 2 adds the bench target registration, shared provider-direct CUDA setup,
2-column U32 upload helper, row-set download helper, Criterion configuration,
and this extraction runbook.

Compile gate:

```text
cargo bench -p xlog-integration --bench w52_skewed_multiway_bench --no-run
```

## LP-MULTI-RUN Extraction

Criterion overwrites `target/criterion/.../new/estimates.json` on each bench
invocation. Run this extraction immediately after every bench run and before
starting the next run.

```bash
extract_w52_medians() {
  run_id="$1"
  out="/tmp/w52_skewed_multiway_run${run_id}.tsv"
  group="target/criterion/w52_skewed_multiway"
  : > "${out}"
  count=0

  for gpu_json in "${group}"/gpu_wcoj/*/new/estimates.json; do
    [ -f "${gpu_json}" ] || continue
    cell="$(basename "$(dirname "$(dirname "${gpu_json}")")")"
    hash_json="${group}/hash_chain/${cell}/new/estimates.json"
    [ -f "${hash_json}" ] || continue
    gpu_ms="$(jq -r '.median.point_estimate/1000000' "${gpu_json}")"
    hash_ms="$(jq -r '.median.point_estimate/1000000' "${hash_json}")"
    awk \
      -v run="${run_id}" \
      -v cell="${cell}" \
      -v gpu="${gpu_ms}" \
      -v hash="${hash_ms}" \
      'BEGIN {
        ratio = hash / gpu;
        direction = ratio >= 1.0 ? "GPU" : "HASH";
        printf "%s\t%s\t%.4f\t%.4f\t%.4f\t%s\n",
          run, cell, gpu, hash, ratio, direction;
      }' >> "${out}"
    count=$((count + 1))
  done

  if [ "${count}" -eq 0 ]; then
    echo "no paired gpu_wcoj/hash_chain estimates found" >&2
    return 1
  fi

  cat "${out}"
}
```

Required execution pattern for every workload:

```bash
rm -f /tmp/w52_skewed_multiway_run{1,2,3}.tsv
for run in 1 2 3; do
  cargo bench -p xlog-integration --bench w52_skewed_multiway_bench
  extract_w52_medians "${run}"
  [ "${run}" -lt 3 ] && sleep 20
done
cat /tmp/w52_skewed_multiway_run{1,2,3}.tsv \
  > /tmp/w52_skewed_multiway_runs.tsv
```

Aggregation command for per-cell `min / median / max` ratios and direction
stability:

```bash
awk -F '\t' '
function sort3(a, b, c, arr) {
  arr[1] = a; arr[2] = b; arr[3] = c;
  for (i = 1; i <= 2; i++) {
    for (j = i + 1; j <= 3; j++) {
      if (arr[i] > arr[j]) {
        t = arr[i]; arr[i] = arr[j]; arr[j] = t;
      }
    }
  }
  return sprintf("%.4f\t%.4f\t%.4f", arr[1], arr[2], arr[3]);
}
{
  cell = $2;
  ratio[cell, $1] = $5;
  direction[cell, $1] = $6;
  cells[cell] = 1;
}
END {
  for (cell in cells) {
    split(sort3(ratio[cell, 1] + 0, ratio[cell, 2] + 0, ratio[cell, 3] + 0), a, "\t");
    if (direction[cell, 1] == direction[cell, 2] &&
        direction[cell, 2] == direction[cell, 3]) {
      stable = direction[cell, 1] " 3/3";
    } else {
      stable = "mixed";
    }
    printf "%s\t%.4fx\t%.4fx\t%.4fx\t%s\n",
      cell, a[1], a[2], a[3], stable;
  }
}' /tmp/w52_skewed_multiway_runs.tsv
```

## Paper Alignment

W5.2 evidence may claim P2 and P5 only. P1, P3, and P4 are not claimed here;
P3 histogram-guided launch balancing remains W3.3-owned.

## Step 3 - 4-Cycle Workload

Fixture: `hub_filtered`, matching the G3 spike shape. For each cell,
`E1(W,X)` and `E2(X,Y)` share one hub `X=0`, so the first binary join forms
`N*N` rows. `E3(Y,Z)` and `E4(Z,W)` filter the final answer back to `N`
cycles.

Cells: `N in {50, 250, 1000, 2000}`.

GPU WCOJ path:

```text
wcoj_layout_u32_recorded(E1)
wcoj_layout_u32_recorded(E2)
wcoj_layout_u32_recorded(E3)
wcoj_layout_u32_recorded(E4)
wcoj_4cycle_u32_recorded(E1, E2, E3, E4)
```

Binary hash path:

```text
E1(W,X) join E2(X,Y) on X
then join E3(Y,Z) on Y
then join E4(Z,W) on Z,W
then project WXYZ
```

Parity is asserted before timing for every cell. Raw extraction files for this
step are `/tmp/w52_skewed_multiway_step3_run{1,2,3}.tsv`.

Measured run pattern:

```text
cargo bench -p xlog-integration --bench w52_skewed_multiway_bench
extract medians immediately to /tmp/w52_skewed_multiway_step3_run1.tsv
sleep 20
cargo bench -p xlog-integration --bench w52_skewed_multiway_bench
extract medians immediately to /tmp/w52_skewed_multiway_step3_run2.tsv
sleep 20
cargo bench -p xlog-integration --bench w52_skewed_multiway_bench
extract medians immediately to /tmp/w52_skewed_multiway_step3_run3.tsv
```

All three bench invocations exited 0.

| Cell | Run 1 GPU ms | Run 1 Hash ms | Run 1 Ratio | Run 2 GPU ms | Run 2 Hash ms | Run 2 Ratio | Run 3 GPU ms | Run 3 Hash ms | Run 3 Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4cycle_N50 | 1.5954 | 10.9897 | 6.8884x | 1.5512 | 10.8857 | 7.0174x | 1.6090 | 11.2404 | 6.9861x |
| 4cycle_N250 | 2.2198 | 10.9828 | 4.9477x | 2.1166 | 11.1045 | 5.2464x | 2.1256 | 11.4953 | 5.4079x |
| 4cycle_N1000 | 5.0714 | 13.7850 | 2.7182x | 4.9477 | 13.7447 | 2.7780x | 4.9208 | 13.5968 | 2.7631x |
| 4cycle_N2000 | 9.2398 | 21.5434 | 2.3316x | 9.5465 | 20.1968 | 2.1156x | 9.3820 | 21.1509 | 2.2544x |

| Cell | Min Ratio | Median Ratio | Max Ratio | Direction Stability |
| --- | ---: | ---: | ---: | --- |
| 4cycle_N50 | 6.8884x | 6.9861x | 7.0174x | GPU 3/3 |
| 4cycle_N250 | 4.9477x | 5.2464x | 5.4079x | GPU 3/3 |
| 4cycle_N1000 | 2.7182x | 2.7631x | 2.7780x | GPU 3/3 |
| 4cycle_N2000 | 2.1156x | 2.2544x | 2.3316x | GPU 3/3 |

Direction flips: none. The 1.0x and 2.0x thresholds are at or below the
smallest tested cell (`N=50`) for this fixture; no binary-win crossover appears
within the tested 4-cycle range.

## Step 4 - 5-Clique Workload

Fixture: diagonal K5 in canonical lexicographic edge order:
`(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)`.
Each edge relation contains `(i, i)` for `i in 1..=N`; expected output is
`N` rows `(i, i, i, i, i)`.

Cells: `N in {10, 25, 50, 100}`.

GPU WCOJ path:

```text
wcoj_layout_sort_u32_recorded(E01)
wcoj_layout_sort_u32_recorded(E02)
wcoj_layout_sort_u32_recorded(E03)
wcoj_layout_sort_u32_recorded(E04)
wcoj_layout_sort_u32_recorded(E12)
wcoj_layout_sort_u32_recorded(E13)
wcoj_layout_sort_u32_recorded(E14)
wcoj_layout_sort_u32_recorded(E23)
wcoj_layout_sort_u32_recorded(E24)
wcoj_layout_sort_u32_recorded(E34)
wcoj_clique5_u32_recorded(E01, E02, E03, E04, E12, E13, E14, E23, E24, E34)
```

Binary hash path:

```text
E01 join E02 on V0
then join E03 on V0
then join E04 on V0
then join E12 on V1,V2
then join E13 on V1,V3
then join E14 on V1,V4
then join E23 on V2,V3
then join E24 on V2,V4
then join E34 on V3,V4
then project V0..V4
```

Parity and exact expected rows are asserted before timing for every cell. Raw
extraction files for this step are
`/tmp/w52_skewed_multiway_step4_run{1,2,3}.tsv`.

Measured run pattern:

```text
cargo bench -p xlog-integration --bench w52_skewed_multiway_bench
extract 5-clique medians immediately to /tmp/w52_skewed_multiway_step4_run1.tsv
sleep 20
cargo bench -p xlog-integration --bench w52_skewed_multiway_bench
extract 5-clique medians immediately to /tmp/w52_skewed_multiway_step4_run2.tsv
sleep 20
cargo bench -p xlog-integration --bench w52_skewed_multiway_bench
extract 5-clique medians immediately to /tmp/w52_skewed_multiway_step4_run3.tsv
```

All three bench invocations exited 0.

| Cell | Run 1 GPU ms | Run 1 Hash ms | Run 1 Ratio | Run 2 GPU ms | Run 2 Hash ms | Run 2 Ratio | Run 3 GPU ms | Run 3 Hash ms | Run 3 Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5clique_N10 | 40.3576 | 23.5284 | 0.5830x | 43.5682 | 23.7401 | 0.5449x | 41.9650 | 22.3792 | 0.5333x |
| 5clique_N25 | 43.1361 | 23.8343 | 0.5525x | 45.5823 | 24.0796 | 0.5283x | 42.7401 | 23.5001 | 0.5498x |
| 5clique_N50 | 44.0312 | 23.9602 | 0.5442x | 42.3305 | 24.2765 | 0.5735x | 43.6296 | 23.3934 | 0.5362x |
| 5clique_N100 | 45.3401 | 23.5539 | 0.5195x | 38.5873 | 22.9401 | 0.5945x | 48.9590 | 24.2123 | 0.4945x |

| Cell | Min Ratio | Median Ratio | Max Ratio | Direction Stability |
| --- | ---: | ---: | ---: | --- |
| 5clique_N10 | 0.5333x | 0.5449x | 0.5830x | HASH 3/3 |
| 5clique_N25 | 0.5283x | 0.5498x | 0.5525x | HASH 3/3 |
| 5clique_N50 | 0.5362x | 0.5442x | 0.5735x | HASH 3/3 |
| 5clique_N100 | 0.4945x | 0.5195x | 0.5945x | HASH 3/3 |

Direction flips: none. Unlike the 4-cycle fixture, the 5-clique diagonal
fixture is hash-favored across all tested cells; no GPU-win crossover appears
within the tested 5-clique range.

## Step 5 - Pivot-Heavy K5 Workload

Fixture:

```text
pivot5(P, A, B, C, D) :-
    pa(P, A), pb(P, B), pc(P, C), pd(P, D),
    ab(A, B), ac(A, C), ad(A, D),
    bc(B, C), bd(B, D), cd(C, D).
```

Incident pivot edges `pa/pb/pc/pd` contain `(0, i)` for `i in 1..=N`.
Leaf filter edges `ab/ac/ad/bc/bd/cd` contain `(i, i)` for `i in 1..=N`.
Expected output is exactly `N` rows `(0, i, i, i, i)`.

Cells: `N in {10, 20, 30, 40}`.

GPU WCOJ path: same clique5 provider-direct path as Step 4:

```text
wcoj_layout_sort_u32_recorded(PA/PB/PC/PD/AB/AC/AD/BC/BD/CD)
wcoj_clique5_u32_recorded(PA, PB, PC, PD, AB, AC, AD, BC, BD, CD)
```

Binary hash path, with pivot-incident joins first:

```text
PA join PB on P
then join PC on P
then join PD on P
then join AB on A,B
then join AC on A,C
then join AD on A,D
then join BC on B,C
then join BD on B,D
then join CD on C,D
then project PABCD
```

Parity and exact expected rows are asserted before timing for every cell. Raw
extraction files for this step are
`/tmp/w52_skewed_multiway_step5_run{1,2,3}.tsv`.

Measured run pattern:

```text
cargo bench -p xlog-integration --bench w52_skewed_multiway_bench
extract pivot5 medians immediately to /tmp/w52_skewed_multiway_step5_run1.tsv
sleep 20
cargo bench -p xlog-integration --bench w52_skewed_multiway_bench
extract pivot5 medians immediately to /tmp/w52_skewed_multiway_step5_run2.tsv
sleep 20
cargo bench -p xlog-integration --bench w52_skewed_multiway_bench
extract pivot5 medians immediately to /tmp/w52_skewed_multiway_step5_run3.tsv
```

All three bench invocations exited 0.

| Cell | Run 1 GPU ms | Run 1 Hash ms | Run 1 Ratio | Run 2 GPU ms | Run 2 Hash ms | Run 2 Ratio | Run 3 GPU ms | Run 3 Hash ms | Run 3 Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pivot5_N10 | 45.9577 | 24.8917 | 0.5416x | 44.6955 | 24.7172 | 0.5530x | 46.4044 | 25.3963 | 0.5473x |
| pivot5_N20 | 45.2255 | 26.8287 | 0.5932x | 45.7472 | 24.5435 | 0.5365x | 42.8277 | 25.6521 | 0.5990x |
| pivot5_N30 | 47.9274 | 36.7255 | 0.7663x | 48.8499 | 34.7033 | 0.7104x | 50.0308 | 39.5201 | 0.7899x |
| pivot5_N40 | 49.0965 | 44.6680 | 0.9098x | 52.3407 | 34.5357 | 0.6598x | 47.7345 | 41.4601 | 0.8686x |

| Cell | Min Ratio | Median Ratio | Max Ratio | Direction Stability |
| --- | ---: | ---: | ---: | --- |
| pivot5_N10 | 0.5416x | 0.5473x | 0.5530x | HASH 3/3 |
| pivot5_N20 | 0.5365x | 0.5932x | 0.5990x | HASH 3/3 |
| pivot5_N30 | 0.7104x | 0.7663x | 0.7899x | HASH 3/3 |
| pivot5_N40 | 0.6598x | 0.8686x | 0.9098x | HASH 3/3 |

Direction flips: none. The pivot-heavy K5 fixture remains hash-favored in all
tested cells, though `N=40` approaches parity in two of three runs. No GPU-win
crossover appears within the tested pivot-heavy range.
