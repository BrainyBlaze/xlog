//! Neural → Probability Bridge
//!
//! This module converts neural network outputs (softmax probability distributions)
//! into probabilistic logic constructs used by XLOG's inference engine.
//!
//! # Architecture
//!
//! Neural networks produce softmax outputs: `[p1, p2, ..., pn]` for n labels.
//! These are converted to:
//!
//! 1. **Annotated Disjunctions**: `p1::pred(X,l1); p2::pred(X,l2); ...`
//! 2. **Circuit Leaves**: Weighted leaf nodes for d-DNNF circuit evaluation
//! 3. **Log Probabilities**: For numerical stability in gradient computation
//!
//! # Example
//!
//! ```
//! use xlog_neural::bridge::{NeuralBridge, NeuralOutput};
//!
//! let output = NeuralOutput {
//!     values: vec![0.7, 0.2, 0.1],
//!     labels: vec!["a".to_string(), "b".to_string(), "c".to_string()],
//! };
//!
//! let bridge = NeuralBridge::new();
//! let probs = bridge.to_ad_probabilities(&output);
//! // probs[0] = ADProbability { probability: 0.7, label: "a" }
//! ```

/// Neural network output with probability distribution over labels.
#[derive(Debug, Clone)]
pub struct NeuralOutput {
    /// Softmax probability values (should sum to ~1.0)
    pub values: Vec<f64>,
    /// Corresponding label names (strings or integer representations)
    pub labels: Vec<String>,
}

impl NeuralOutput {
    /// Create output with string labels.
    pub fn new(values: Vec<f64>, labels: Vec<String>) -> Self {
        debug_assert_eq!(
            values.len(),
            labels.len(),
            "values and labels must have same length"
        );
        Self { values, labels }
    }

    /// Create output with integer labels converted to strings.
    pub fn with_integer_labels(values: Vec<f64>, labels: Vec<i64>) -> Self {
        Self {
            values,
            labels: labels.into_iter().map(|i| i.to_string()).collect(),
        }
    }

    /// Number of classes/labels.
    pub fn num_classes(&self) -> usize {
        self.values.len()
    }

    /// Get the argmax (most likely class).
    pub fn argmax(&self) -> Option<(usize, &str)> {
        self.values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| (idx, self.labels[idx].as_str()))
    }
}

/// Annotated disjunction probability component.
///
/// Represents one choice in an annotated disjunction: `probability::pred(X, label)`
#[derive(Debug, Clone)]
pub struct ADProbability {
    /// Probability weight (clamped to [epsilon, 1.0])
    pub probability: f64,
    /// Label value as string
    pub label: String,
}

/// Circuit leaf node for d-DNNF evaluation.
///
/// Each leaf corresponds to a probabilistic variable with a weight.
#[derive(Debug, Clone)]
pub struct CircuitLeaf {
    /// Variable ID in the circuit
    pub variable_id: usize,
    /// Weight for weighted model counting
    pub weight: f64,
}

/// Bridge for converting neural outputs to probabilistic constructs.
///
/// Handles numerical stability through epsilon clamping and normalization.
pub struct NeuralBridge {
    /// Minimum probability to prevent log(0)
    epsilon: f64,
}

impl NeuralBridge {
    /// Create a new bridge with default epsilon (1e-8).
    pub fn new() -> Self {
        Self { epsilon: 1e-8 }
    }

    /// Create a bridge with custom epsilon.
    pub fn with_epsilon(epsilon: f64) -> Self {
        debug_assert!(epsilon > 0.0, "epsilon must be positive");
        Self { epsilon }
    }

    /// Convert softmax output to annotated disjunction probabilities.
    ///
    /// Each probability is clamped to [epsilon, 1.0] for numerical stability.
    pub fn to_ad_probabilities(&self, output: &NeuralOutput) -> Vec<ADProbability> {
        output
            .values
            .iter()
            .zip(output.labels.iter())
            .map(|(&prob, label)| ADProbability {
                probability: prob.max(self.epsilon).min(1.0),
                label: label.clone(),
            })
            .collect()
    }

    /// Convert batch of neural outputs to circuit leaf weights.
    ///
    /// Returns a 2D structure: `leaves[sample_idx][label_idx]`
    pub fn batch_to_circuit_leaves(&self, outputs: &[NeuralOutput]) -> Vec<Vec<CircuitLeaf>> {
        outputs
            .iter()
            .map(|output| {
                output
                    .values
                    .iter()
                    .enumerate()
                    .map(|(i, &weight)| CircuitLeaf {
                        variable_id: i,
                        weight: weight.max(self.epsilon).min(1.0),
                    })
                    .collect()
            })
            .collect()
    }

    /// Convert to log probabilities for numerical stability.
    ///
    /// Log probabilities are used for:
    /// - Computing NLL loss: `-log(p_true)`
    /// - Avoiding underflow in product of many small probabilities
    pub fn to_log_probabilities(&self, output: &NeuralOutput) -> Vec<f64> {
        output
            .values
            .iter()
            .map(|&p| (p.max(self.epsilon)).ln())
            .collect()
    }

    /// Normalize probabilities to sum to 1.0.
    ///
    /// Useful when network outputs have small numerical errors.
    pub fn normalize(&self, output: &NeuralOutput) -> NeuralOutput {
        let sum: f64 = output.values.iter().sum();
        if sum.abs() < self.epsilon {
            // Avoid division by zero - uniform distribution
            let uniform = 1.0 / output.values.len() as f64;
            NeuralOutput {
                values: vec![uniform; output.values.len()],
                labels: output.labels.clone(),
            }
        } else {
            NeuralOutput {
                values: output.values.iter().map(|&v| v / sum).collect(),
                labels: output.labels.clone(),
            }
        }
    }

    /// Extract raw weights for gradient computation.
    ///
    /// These weights are passed to the backward pass to compute
    /// gradients w.r.t. the neural network parameters.
    pub fn extract_gradient_weights(&self, output: &NeuralOutput) -> Vec<f64> {
        output.values.clone()
    }

    /// Compute the probability of a specific label.
    pub fn probability_of(&self, output: &NeuralOutput, label: &str) -> Option<f64> {
        output
            .labels
            .iter()
            .position(|l| l == label)
            .map(|idx| output.values[idx].max(self.epsilon))
    }

    /// Compute the log probability of a specific label.
    pub fn log_probability_of(&self, output: &NeuralOutput, label: &str) -> Option<f64> {
        self.probability_of(output, label).map(|p| p.ln())
    }

    /// Create circuit leaves for a single sample with variable ID offset.
    ///
    /// Used when multiple samples share a circuit structure but have
    /// different variable ID ranges.
    pub fn to_circuit_leaves_with_offset(
        &self,
        output: &NeuralOutput,
        variable_offset: usize,
    ) -> Vec<CircuitLeaf> {
        output
            .values
            .iter()
            .enumerate()
            .map(|(i, &weight)| CircuitLeaf {
                variable_id: variable_offset + i,
                weight: weight.max(self.epsilon).min(1.0),
            })
            .collect()
    }
}

impl Default for NeuralBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_output_argmax() {
        let output = NeuralOutput {
            values: vec![0.1, 0.7, 0.2],
            labels: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        };

        let (idx, label) = output.argmax().unwrap();
        assert_eq!(idx, 1);
        assert_eq!(label, "b");
    }

    #[test]
    fn test_probability_of_label() {
        let output = NeuralOutput {
            values: vec![0.3, 0.5, 0.2],
            labels: vec!["cat".to_string(), "dog".to_string(), "bird".to_string()],
        };

        let bridge = NeuralBridge::new();

        assert!((bridge.probability_of(&output, "cat").unwrap() - 0.3).abs() < 1e-6);
        assert!((bridge.probability_of(&output, "dog").unwrap() - 0.5).abs() < 1e-6);
        assert!(bridge.probability_of(&output, "fish").is_none());
    }

    #[test]
    fn test_circuit_leaves_with_offset() {
        let output = NeuralOutput {
            values: vec![0.4, 0.6],
            labels: vec!["x".to_string(), "y".to_string()],
        };

        let bridge = NeuralBridge::new();
        let leaves = bridge.to_circuit_leaves_with_offset(&output, 100);

        assert_eq!(leaves[0].variable_id, 100);
        assert_eq!(leaves[1].variable_id, 101);
    }
}
