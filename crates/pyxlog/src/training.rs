// Training infrastructure for pyxlog Python bindings.
//
// Contains the impl block for TrainingHistory (new/add_epoch/add_batch) and
// the train_model / train_model_tensor #[pyfunction]s.
//
// The EpochStats and TrainingHistory #[pyclass] struct definitions live in
// lib.rs because pyclass registration order matters for the pymodule macro.

use pyo3::prelude::*;
use super::{CompiledProgram, TrainingHistory};

/// Train a program for multiple epochs.
///
/// This is the main training entry point that runs the full training loop:
/// - For each epoch: shuffle queries (optional), process batches, record stats
/// - Supports learning rate scheduling via scheduler_step() after each epoch
///
/// # Arguments
/// * `program` - Compiled program with registered networks
/// * `queries` - Training queries (e.g., ["addition(0, 1, 5)", "addition(2, 3, 7)")
/// * `epochs` - Number of training epochs (default: 10)
/// * `batch_size` - Number of queries per batch (default: 32)
/// * `log_iter` - Log progress every N batches (default: 100)
/// * `shuffle` - Whether to shuffle queries each epoch (default: true)
///
/// # Returns
/// TrainingHistory with epoch and batch losses
#[pyfunction]
#[pyo3(signature = (program, queries, epochs=10, batch_size=32, log_iter=100, shuffle=true, max_grad_norm=None, val_queries=None, patience=None))]
pub fn train_model(
    py: Python<'_>,
    program: &mut CompiledProgram,
    queries: Vec<String>,
    epochs: usize,
    batch_size: usize,
    log_iter: usize,
    shuffle: bool,
    max_grad_norm: Option<f64>,
    val_queries: Option<Vec<String>>,
    patience: Option<usize>,
) -> PyResult<TrainingHistory> {
    use rand::seq::SliceRandom;
    use rand::thread_rng;
    use std::time::Instant;

    // Validate: val_queries and patience must both be present or both absent
    match (&val_queries, &patience) {
        (Some(_), None) | (None, Some(_)) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "val_queries and patience must both be provided for early stopping"
            ));
        }
        _ => {}
    }

    let mut history = TrainingHistory::new();
    let mut best_val_loss = f64::INFINITY;
    let mut epochs_without_improvement = 0usize;

    for epoch in 0..epochs {
        let mut epoch_queries = queries.clone();

        if shuffle {
            let mut rng = thread_rng();
            epoch_queries.shuffle(&mut rng);
        }

        let epoch_start = Instant::now();
        let stats =
            program.train_epoch_internal(py, &epoch_queries, batch_size, log_iter, max_grad_norm, &mut history)?;
        history.add_epoch(stats.avg_loss, epoch_start.elapsed().as_secs_f64());

        println!(
            "Epoch {}/{}: avg_loss={:.6}",
            epoch + 1, epochs, stats.avg_loss
        );
        use std::io::Write;
        let _ = std::io::stdout().flush();

        // Early stopping check
        if let (Some(ref val_q), Some(pat)) = (&val_queries, patience) {
            let val_loss = program.evaluate_loss(val_q.clone())?;
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
            }
            if epochs_without_improvement >= pat {
                history.stopped_early = true;
                break;
            }
        }
    }

    Ok(history)
}

/// GPU-native training loop — no per-query host synchronization.
///
/// Identical to train_model but uses forward_backward_tensor internally.
/// Loss stays on CUDA device; .item() called once per batch only.
#[pyfunction]
#[pyo3(signature = (program, queries, epochs=10, batch_size=32, log_iter=100, shuffle=true, max_grad_norm=None, val_queries=None, patience=None))]
pub fn train_model_tensor(
    py: Python<'_>,
    program: &mut CompiledProgram,
    queries: Vec<String>,
    epochs: usize,
    batch_size: usize,
    log_iter: usize,
    shuffle: bool,
    max_grad_norm: Option<f64>,
    val_queries: Option<Vec<String>>,
    patience: Option<usize>,
) -> PyResult<TrainingHistory> {
    use rand::seq::SliceRandom;
    use rand::thread_rng;
    use std::time::Instant;

    match (&val_queries, &patience) {
        (Some(_), None) | (None, Some(_)) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "val_queries and patience must both be provided for early stopping"
            ));
        }
        _ => {}
    }

    let mut history = TrainingHistory::new();
    let mut best_val_loss = f64::INFINITY;
    let mut epochs_without_improvement = 0usize;

    for epoch in 0..epochs {
        let mut epoch_queries = queries.clone();

        if shuffle {
            let mut rng = thread_rng();
            epoch_queries.shuffle(&mut rng);
        }

        let epoch_start = Instant::now();
        let stats = program.train_epoch_tensor_internal(
            py,
            &epoch_queries,
            batch_size,
            log_iter,
            max_grad_norm,
            &mut history,
        )?;
        history.add_epoch(stats.avg_loss, epoch_start.elapsed().as_secs_f64());

        println!(
            "Epoch {}/{}: avg_loss={:.6}",
            epoch + 1,
            epochs,
            stats.avg_loss
        );
        use std::io::Write;
        let _ = std::io::stdout().flush();

        if let (Some(ref val_q), Some(pat)) = (&val_queries, patience) {
            let val_loss = program.evaluate_loss(val_q.clone())?;
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
            }
            if epochs_without_improvement >= pat {
                history.stopped_early = true;
                break;
            }
        }
    }

    Ok(history)
}

impl TrainingHistory {
    pub(super) fn new() -> Self {
        Self {
            epoch_losses: Vec::new(),
            epoch_times: Vec::new(),
            batch_losses: Vec::new(),
            stopped_early: false,
        }
    }

    pub(super) fn add_epoch(&mut self, loss: f64, epoch_time_sec: f64) {
        self.epoch_losses.push(loss);
        self.epoch_times.push(epoch_time_sec);
    }

    pub(super) fn add_batch(&mut self, loss: f64) {
        self.batch_losses.push(loss);
    }
}
