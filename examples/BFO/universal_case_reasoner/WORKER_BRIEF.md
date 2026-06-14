# Worker Brief: BFO Universal Case Reasoner

You own branch `feat/bfo-universal-case-reasoner` and only this path:

`examples/BFO/universal_case_reasoner/`

You are not alone in the repository. Other workers own separate worktrees and
separate `examples/BFO/*` projects. Do not revert or edit their changes. Do not
edit shared files, repo-wide runtime code, or common helper directories unless a
separate instruction explicitly authorizes that scope.

## Read First

- `ROADMAP.md`
- `docs/language-reference.md`
- `docs/architecture/language-v085.md`
- `docs/architecture/dilp-training.md`
- `docs/architecture/xlog-prob.md`
- `examples/neural/*`
- `examples/v080-dts/*`
- `examples/language-completeness/showcase/*`
- `examples/v086-runtime/*`
- `docs/evidence/2026-05-19-v086-consumers/validation_summary.json`
- `/home/dev/projects/Goal-Driven_Software_Development.pdf`
- `/home/dev/projects/GQM.pdf`

## Mission

Implement a BFO-governed cross-domain case reasoner. The same BFO kernel must
infer root causes, failure chains, risk states, interventions, and explanations
across at least five domains while proving zero-shot or few-shot transfer to a
held-out domain.

## Required First Implementation Step

Before adding implementation code, refine the local validation plan against
`GOALS.md`, `GQM.md`, and `REQUIREMENTS.md`. If a P0 requirement cannot be met
with current XLOG capabilities from this example directory, document the
production blocker and produce a failing validation summary. Do not weaken the
requirement.

This step has been completed for the current branch. Subsequent work should
preserve the strict evidence split: smoke runners may not satisfy production
transfer/profile/soak gates; only `evidence/production_transfer.json` with
`scope: "production"` may do that.

## Expected Deliverables

- Stable BFO kernel and five thin domain adapters.
- Neural bridge using real CUDA PyTorch and XLOG `nn/4`.
- Holdout protocol for transfer validation.
- Root-cause, failure-chain, risk, intervention, and explanation rules.
- Baseline ablations.
- Core mutation audit.
- Real Hugging Face production case provenance.
- Executable v0.8.0/v0.8.5/v0.8.6 bundle reuse evidence.
- Production transfer/profile/soak evidence.
- Strict validation script and machine-readable validation summary.

## Completion Standard

The branch is not complete until `./validate.sh --strict --gpu-required`
produces a full `validation_summary.json` and either passes every P0 gate or
fails with precise blocker evidence.
