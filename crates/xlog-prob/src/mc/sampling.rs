//! GPU evaluation loop and nonmonotone SCC handling.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use xlog_core::{Result, XlogError};
use xlog_cuda::{CudaBuffer, CudaKernelProvider};
use xlog_runtime::Executor;

use super::buffers;
use super::EvalStats;

pub(super) fn evaluate_program_gpu(
    provider: &Arc<CudaKernelProvider>,
    executor: &mut Executor,
    plan: &xlog_ir::ExecutionPlan,
    nonmonotone_sccs: &HashSet<usize>,
    max_nonmonotone_iterations: usize,
) -> Result<EvalStats> {
    let mut stats = EvalStats::default();

    if plan.strata.is_empty() {
        for (idx, scc) in plan.sccs.iter().enumerate() {
            let rules = plan
                .rules_by_scc
                .get(idx)
                .ok_or_else(|| XlogError::Execution(format!("Missing rules for SCC {}", idx)))?;
            if nonmonotone_sccs.contains(&idx) {
                stats.nonmonotone_sccs += 1;
                let (cycle, hit_limit) = execute_nonmonotone_scc_gpu(
                    provider,
                    executor,
                    &scc.predicates,
                    rules,
                    max_nonmonotone_iterations,
                )?;
                if cycle {
                    stats.nonmonotone_cycles += 1;
                }
                if hit_limit {
                    stats.nonmonotone_iteration_limit_hits += 1;
                }
            } else if scc.is_recursive {
                executor.execute_recursive_scc(rules)?;
            } else {
                executor.execute_non_recursive_scc(rules)?;
            }
        }
        return Ok(stats);
    }

    for stratum in &plan.strata {
        for &scc_id in &stratum.sccs {
            let scc_idx = scc_id as usize;
            let scc = plan.sccs.get(scc_idx).ok_or_else(|| {
                XlogError::Execution(format!("Missing SCC metadata for {}", scc_id))
            })?;
            let rules = plan
                .rules_by_scc
                .get(scc_idx)
                .ok_or_else(|| XlogError::Execution(format!("Missing rules for SCC {}", scc_id)))?;

            if nonmonotone_sccs.contains(&scc_idx) {
                stats.nonmonotone_sccs += 1;
                let (cycle, hit_limit) = execute_nonmonotone_scc_gpu(
                    provider,
                    executor,
                    &scc.predicates,
                    rules,
                    max_nonmonotone_iterations,
                )?;
                if cycle {
                    stats.nonmonotone_cycles += 1;
                }
                if hit_limit {
                    stats.nonmonotone_iteration_limit_hits += 1;
                }
            } else if scc.is_recursive {
                executor.execute_recursive_scc(rules)?;
            } else {
                executor.execute_non_recursive_scc(rules)?;
            }
        }
    }

    Ok(stats)
}

fn execute_nonmonotone_scc_gpu(
    provider: &Arc<CudaKernelProvider>,
    executor: &mut Executor,
    preds: &[String],
    rules: &[xlog_ir::CompiledRule],
    max_iters: usize,
) -> Result<(bool, bool)> {
    let base_state = snapshot_scc_state(provider, executor, preds)?;
    let mut history: Vec<HashMap<String, CudaBuffer>> = Vec::new();
    history.push(clone_state(provider, &base_state)?);
    let mut signatures: Vec<Vec<u64>> = vec![state_signature(&history[0], preds)];

    for _ in 0..max_iters {
        let mut next_state = clone_state(provider, &base_state)?;

        for rule in rules {
            let mut result = executor.execute_node(&rule.body)?;
            if result.is_empty() {
                continue;
            }
            result = buffers::dedup_relation(provider, &result)?;
            if result.is_empty() {
                continue;
            }
            if let Some(entry) = next_state.get_mut(&rule.head) {
                if entry.is_empty() {
                    *entry = result;
                } else {
                    let merged = provider.union_gpu(entry, &result)?;
                    *entry = merged;
                }
            } else {
                next_state.insert(rule.head.clone(), result);
            }
        }

        let current = history
            .last()
            .ok_or_else(|| XlogError::Execution("Missing current state".to_string()))?;
        if states_equal(provider, current, &next_state, preds)? {
            apply_state_to_store_move(executor, next_state);
            return Ok((false, false));
        }

        let sig = state_signature(&next_state, preds);
        for (idx, prev_sig) in signatures.iter().enumerate() {
            if *prev_sig != sig {
                continue;
            }
            let candidate = history.get(idx).ok_or_else(|| {
                XlogError::Execution("Nonmonotone history index out of range".to_string())
            })?;
            if states_equal(provider, candidate, &next_state, preds)? {
                let final_state = intersect_states_device(provider, &history[idx..], preds)?;
                apply_state_to_store_move(executor, final_state);
                return Ok((true, false));
            }
        }

        apply_state_to_store_move(executor, next_state);
        history.push(snapshot_scc_state(provider, executor, preds)?);
        signatures.push(sig);
    }

    let final_state = intersect_states_device(provider, &history, preds)?;
    apply_state_to_store_move(executor, final_state);
    Ok((false, true))
}

fn snapshot_scc_state(
    provider: &Arc<CudaKernelProvider>,
    executor: &Executor,
    preds: &[String],
) -> Result<HashMap<String, CudaBuffer>> {
    let mut state = HashMap::new();
    for pred in preds {
        let buf = executor
            .store()
            .get(pred)
            .ok_or_else(|| XlogError::Execution(format!("Missing relation {}", pred)))?;
        let cloned = if buf.is_empty() {
            provider.create_empty_buffer(buf.schema().clone())?
        } else {
            buffers::clone_buffer_device(provider, buf)?
        };
        state.insert(pred.clone(), cloned);
    }
    Ok(state)
}

fn clone_state(
    provider: &Arc<CudaKernelProvider>,
    state: &HashMap<String, CudaBuffer>,
) -> Result<HashMap<String, CudaBuffer>> {
    let mut out = HashMap::new();
    for (pred, buf) in state {
        let cloned = if buf.is_empty() {
            provider.create_empty_buffer(buf.schema().clone())?
        } else {
            buffers::clone_buffer_device(provider, buf)?
        };
        out.insert(pred.clone(), cloned);
    }
    Ok(out)
}

fn apply_state_to_store_move(executor: &mut Executor, state: HashMap<String, CudaBuffer>) {
    for (pred, buf) in state {
        executor.put_relation(&pred, buf);
    }
}

fn state_signature(state: &HashMap<String, CudaBuffer>, preds: &[String]) -> Vec<u64> {
    let mut sig = Vec::with_capacity(preds.len());
    for pred in preds {
        let rows = state.get(pred).map(|b| b.num_rows()).unwrap_or(0);
        sig.push(rows);
    }
    sig
}

fn states_equal(
    provider: &Arc<CudaKernelProvider>,
    a: &HashMap<String, CudaBuffer>,
    b: &HashMap<String, CudaBuffer>,
    preds: &[String],
) -> Result<bool> {
    for pred in preds {
        let buf_a = a
            .get(pred)
            .ok_or_else(|| XlogError::Execution(format!("Missing state {}", pred)))?;
        let buf_b = b
            .get(pred)
            .ok_or_else(|| XlogError::Execution(format!("Missing state {}", pred)))?;
        if !buffers_equal(provider, buf_a, buf_b)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn buffers_equal(
    provider: &Arc<CudaKernelProvider>,
    a: &CudaBuffer,
    b: &CudaBuffer,
) -> Result<bool> {
    if a.num_rows() != b.num_rows() {
        return Ok(false);
    }
    if a.is_empty() && b.is_empty() {
        return Ok(true);
    }

    let diff_ab = provider.diff_gpu(a, b)?;
    if !diff_ab.is_empty() {
        return Ok(false);
    }
    let diff_ba = provider.diff_gpu(b, a)?;
    Ok(diff_ba.is_empty())
}

fn intersect_states_device(
    provider: &Arc<CudaKernelProvider>,
    states: &[HashMap<String, CudaBuffer>],
    preds: &[String],
) -> Result<HashMap<String, CudaBuffer>> {
    let mut out: HashMap<String, CudaBuffer> = HashMap::new();
    let Some(first) = states.first() else {
        return Ok(out);
    };

    for pred in preds {
        let first_buf = first
            .get(pred)
            .ok_or_else(|| XlogError::Execution(format!("Missing state {}", pred)))?;
        let mut acc = if first_buf.is_empty() {
            provider.create_empty_buffer(first_buf.schema().clone())?
        } else {
            buffers::clone_buffer_device(provider, first_buf)?
        };
        for state in &states[1..] {
            let next = state
                .get(pred)
                .ok_or_else(|| XlogError::Execution(format!("Missing state {}", pred)))?;
            acc = buffer_intersection(provider, &acc, next)?;
            if acc.is_empty() {
                break;
            }
        }
        out.insert(pred.clone(), acc);
    }

    Ok(out)
}

fn buffer_intersection(
    provider: &Arc<CudaKernelProvider>,
    a: &CudaBuffer,
    b: &CudaBuffer,
) -> Result<CudaBuffer> {
    if a.is_empty() || b.is_empty() {
        return provider.create_empty_buffer(a.schema().clone());
    }
    let diff = provider.diff_gpu(a, b)?;
    if diff.is_empty() {
        return buffers::clone_buffer_device(provider, a);
    }
    provider.diff_gpu(a, &diff)
}
