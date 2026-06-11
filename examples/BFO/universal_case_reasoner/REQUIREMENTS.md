# Production Requirements: BFO Universal Case Reasoner

## P0 Hard Gates

- `./validate.sh --strict --gpu-required` must be the authoritative gate.
- CUDA, PyTorch, `pyxlog`, and XLOG `nn/4` integration are mandatory.
- Production transfer cases must come from real Hugging Face datasets, with
  dataset ID, split, row index, and row hash provenance recorded for every
  prediction record.
- Root-cause truth must be read from externally supplied Hugging Face RCA,
  diagnosis, human reasoning, marked-log, or fault-diagnosis fields/artifacts.
  Mapping an ordinary task label such as defective/not-defective or late/not
  late into a root-cause label is forbidden.
- CPU-only semantic evaluation is forbidden.
- Mocked neural outputs are forbidden.
- Two-domain toy transfer is forbidden.
- Per-domain BFO core edits are forbidden.
- Hidden host hot loops are forbidden.
- Unsupported P0 requirements produce `FAIL` with blocker evidence.

## BFO Transfer Conformance

- At least 12 BFO upper categories must be used.
- At least 8 BFO relation families must be used.
- 100% of domain classes must map to BFO.
- Core BFO rule checksum must be identical for all domains.
- Domain adapters must contain domain facts and mappings only.
- Adapter/core rule ratio must be <= 0.25 per domain.
- 100% of invalid cross-domain fixtures must be rejected or inconsistent.

## Domain Coverage

At least five domains are mandatory. Recommended domains:

- AMR or scientific discovery incident.
- Clinical deterioration or medication/process incident.
- Manufacturing quality or equipment failure.
- Cybersecurity intrusion or access-control incident.
- Supply-chain, logistics, or lab operations incident.

Each domain must include root cause, failure chain, risk state, intervention,
and explanation fixtures.

## Neural Requirements

- A real CUDA PyTorch model must be registered and invoked through XLOG `nn/4`.
- Neural output must materially affect root-cause or intervention ranking.
- The validation must include neural-only, domain-symbolic, shared-symbolic, and
  neuro-symbolic ablations.
- Transfer-quality metrics must be recomputed from prediction, ablation, and
  invalid-fixture records by the validator; hand-authored summary metrics are
  insufficient.
- Neuro-symbolic performance must improve by at least 15% over the strongest
  baseline on the selected primary transfer metric.
- Fixed-seed results must be byte-identical across 5 runs.

## Transfer And Evolution

- At least one domain must be held out during rule evolution.
- Held-out root-cause F1 must be >= 0.90.
- Accepted intervention precision must be >= 0.95.
- Promoted rule quality on non-held-out domains: precision >= 0.98,
  recall >= 0.95, F1 >= 0.965.
- Learned rules may not mutate the BFO kernel.
- Every top-level claim must include a BFO-valid explanation.

## DILP Rule Induction

These gates distinguish robust XLOG/CUDA ranking transfer from an example-level
DILP evidence claim.

- DILP-001 XLOG_PROOF_PATHS: XLOG proof-path clauses must execute and provide
  proof candidates for learned rule selection.
- DILP-002 JOINT_TRAINING: neural predicate parameters and symbolic rule
  weights must train jointly on CUDA, with finite nonzero gradients.
- DILP-003 RULE_INVENTORY: learned rule inventories must cover every
  leave-one-domain-out fold, including selected clauses, clause weights,
  training domains, and held-out exclusion metadata.
- DILP-004 CLAUSE_ABLATIONS: learned-clause ablations must be reported from raw
  prediction records, and the full DILP model must reach macro F1 >= 0.90 while
  matching or beating every single-clause ablation.
- DILP-005 PROOF_GRADIENTS: proof-level support tensors must remain
  device-resident and receive finite nonzero gradients.
- DILP-006 HELDOUT_SAFE_INDUCTION: held-out cases and labels may not be used
  during rule induction; held-out labels are only used after prediction for
  metric recomputation.

## Robust Generalization

These gates distinguish the five-domain demo-transfer claim from a production
or scientific robust-generalization claim. Strict validation must fail any
evidence bundle that excludes a production domain from the aggregate.

- GEN-001 LEAVE_ONE_DOMAIN_OUT: run leave-one-domain-out evaluation over every
  production domain, not only the configured showcase holdout.
- GEN-002 MIN_HELDOUT_SIZE: every held-out domain must include at least 100 real
  Hugging Face cases with dataset ID, row hash, root-field hash, and root-text
  hash provenance.
- GEN-003 MACRO_TRANSFER: macro held-out root-cause F1 must be >= 0.90 and the
  minimum per-domain held-out F1 must be >= 0.85.
- GEN-004 NO_TEST_LABEL_CANDIDATES: held-out candidate spaces may not be
  constructed from the held-out test rows' true RCA/root/intervention text.
- GEN-005 FROZEN_MODEL_RULES: the BFO kernel, learned rules, neural
  architecture, thresholds, aliases, and scoring weights must be frozen before
  held-out evaluation starts.
- GEN-006 UNSEEN_DATASET_TRANSFER: at least one held-out evaluation must use a
  dataset family not used during rule evolution or feature design.
- GEN-007 STRONG_BASELINES: compare against neural-only, symbolic-only,
  domain-specific classifier, retrieval/RAG nearest-neighbor, majority/prior,
  and neuro-symbolic methods.
- GEN-008 STATISTICAL_CONFIDENCE: report bootstrap confidence intervals and
  paired significance tests.
- GEN-009 ADVERSARIAL_DOMAIN_SHIFT: evaluate noisy, sparse, paraphrased,
  missing-field, and distractor-candidate variants.
- GEN-010 GENERALIZATION_REPORT: the validator must recompute aggregate
  metrics from raw generalization prediction records and fail if any production
  domain is excluded.

## Public Benchmark Claim Boundary

- PUBLIC-SOTA-001 PUBLIC_BENCHMARK_STATE: strict validation must include a
  `public_benchmark_report`.
- External SOTA may be claimed only when runnable public benchmark adapters
  cover every required family, exact split/protocol/version hashes are recorded,
  baseline citations are present, and the public benchmark report status is
  `PASS`.
- If public benchmark adapters are not yet implemented, the report must mark
  `external_sota_claim: false`, include fail-closed blockers, and leave covered
  public benchmark families empty or partial. This can preserve a local
  production/generalization claim, but it is not an external SOTA claim.
- Required public benchmark families: AIOps RCA, clinical diagnosis,
  cross-domain ontology shift, cybersecurity intrusion, manufacturing
  equipment/fault, PHM fault, and root-cause AIOps.

## Device-Resident Execution

- Data-plane D2H transfers in the hot loop: 0.
- Data-plane H2D transfers after initial load: 0.
- Control-plane metadata per hot iteration: <= 4096 bytes.
- Transfer counters must be captured before and after the hot loop.
- Semantic column downloads or tensor materialization on host inside hot loops
  are immediate failures.

## Bundle Reuse

- Production evidence must include executable reuse probes for the merged
  v0.8.0, v0.8.5, and v0.8.6 bundle.
- v0.8.0 reuse must exercise pyxlog `LogicProgram.compile`, session
  evaluation, and relation-delta equivalence.
- v0.8.5 reuse must audit the language showcase feature coverage.
- v0.8.6 reuse must exercise `apply_relation_delta_batch`, relation callbacks,
  persistent join-index cache stats, and zero hot-loop transfer counters.

## Scale And Performance

Production profile minimums:

- Symbolic/BFO facts: >= 1,000,000.
- Neural observations: >= 100,000.
- Entities: >= 50,000.
- Staged delta updates: >= 10,000.
- p95 core indexed query latency: <= 50 ms.
- Strict validation wall time: <= 180 seconds unless a separate production
  profile runner records and justifies longer execution.
- Soak run: >= 30 minutes, GPU memory drift <= 2%, no unbounded relation growth.

## Evidence Schema

`validation_summary.json` must include:

- Branch and git SHA.
- Hardware and CUDA/PyTorch/XLOG runtime details.
- Commands executed.
- Metric values for every GQM question.
- PASS/FAIL for every P0 gate.
- Raw output paths.
- Explanation paths.
- Failure/blocker descriptions with exact requirement IDs.
- `showcase_metrics`, `generalization_report`, `dilp_report`, and
  `public_benchmark_report` as separate evidence namespaces. Legacy top-level
  `baseline_metrics`, top-level baseline uplift fields, or
  `computed_metrics.baseline_metrics` are invalid because
  `generalization_report.baseline_uplift` is the canonical baseline-uplift
  source.
