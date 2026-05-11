# W5.2 Skewed Multiway Bench Evidence

Date: 2026-05-12

Branch: `feat/w52-skewed-multiway-bench`

Plan: `docs/plans/2026-05-11-w52-bench-plan.md`

Bench target: `crates/xlog-integration/benches/w52_skewed_multiway_bench.rs`

## Step 2 Skeleton

Step 2 adds the bench target registration, shared provider-direct CUDA setup,
2-column U32 upload helper, row-set download helper, Criterion configuration,
and this extraction runbook. Workload-specific 4-cycle, 5-clique, and
pivot-heavy K5 measurement tables are added in later commits.

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
