# Component-Centric Roadmap Design

**Date:** 2026-01-14
**Status:** Approved for implementation

## Purpose
Rewrite `docs/ROADMAP.md` into a component-first, version-targeted format while preserving all existing roadmap items, constraints, and references.

## Constraints
- Do not delete any items from the existing roadmap.
- Roadmap must be strictly by component.
- Each component has two subsections only: Implemented and Planned.
- Every item must include an explicit version tag (e.g., [v0.2.0], [v0.3.x], [v0.4-0.5], [v0.6+]).

## Chosen Approach
Approach A (component-first): Each XLOG component is a top-level section with Implemented and Planned subsections. Cross-cutting content (metrics, risks, resources) is preserved by moving it into component sections rather than removing it.

## Component Order
1. Core Language & Compiler (xlog-logic)
2. Runtime & Execution (xlog-runtime)
3. GPU Backend & Kernels (xlog-cuda + kernels)
4. Optimizer & Stats (xlog-solve + xlog-stats)
5. Incremental Maintenance & Adaptive Indexing
6. Interop (Arrow/DLPack/CuDF)
7. Probabilistic Reasoning (xlog-prob)
8. Python Interop (xlog-gpu-py)
9. CUDA Certification & Validation
10. Epistemic Logic (xlog-elp)
11. Scaling & Distributed
12. Quality & Readiness
13. Reliability & Risk
14. Documentation & References

## Preservation Checklist
- P1/P2/P3/P4 items preserved under relevant components.
- Phase 4 deliverables preserved under xlog-prob + xlog-gpu-py + Interop.
- Phase 5 (xlog-elp) and Phase 6 (Scaling) preserved with prerequisites.
- Success metrics preserved under Quality & Readiness.
- Risk assessment preserved under Reliability & Risk.
- Resources list preserved under Documentation & References.
