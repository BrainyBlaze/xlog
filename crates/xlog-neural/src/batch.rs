//! Batched Neural Evaluation
//!
//! This module provides infrastructure for grouping neural predicate calls
//! by network name, enabling efficient batched GPU evaluation.
//!
//! # Why Batching?
//!
//! In DeepProbLog-style programs, the same neural network may be called many times:
//!
//! ```text
//! nn(mnist_net, [X], Y, [0..9]) :: digit(X, Y).
//! addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
//! ```
//!
//! For a query like `addition(img1, img2, Z)`, we need to evaluate `mnist_net`
//! twice (once for each digit). Instead of two separate forward passes, we batch
//! them into a single `mnist_net([img1, img2])` call for GPU efficiency.
//!
//! # Usage
//!
//! ```
//! use xlog_neural::batch::{BatchCollector, NeuralCall};
//!
//! let mut collector = BatchCollector::new();
//!
//! // During proof search, collect neural calls
//! collector.add(NeuralCall::new("mnist", vec![0])); // digit(img[0], Y)
//! collector.add(NeuralCall::new("mnist", vec![1])); // digit(img[1], Y)
//!
//! // Group by network for batched evaluation
//! let batches = collector.collect();
//! let mnist_indices = collector.indices_for_network("mnist");
//! // mnist_indices = [0, 1] - evaluate both in one forward pass
//! ```

use std::collections::HashMap;

/// A single neural predicate call.
///
/// Records which network to call and which input indices to use
/// from the active tensor source.
#[derive(Debug, Clone)]
pub struct NeuralCall {
    /// Name of the neural network (must match registered network)
    pub network: String,
    /// Indices into the active tensor source
    pub input_indices: Vec<usize>,
}

impl NeuralCall {
    /// Create a new neural call.
    pub fn new(network: &str, input_indices: Vec<usize>) -> Self {
        Self {
            network: network.to_string(),
            input_indices,
        }
    }

    /// Create a call with a single input index.
    pub fn single(network: &str, index: usize) -> Self {
        Self::new(network, vec![index])
    }

    /// Number of inputs in this call.
    pub fn num_inputs(&self) -> usize {
        self.input_indices.len()
    }
}

/// Collects neural predicate calls for batched evaluation.
///
/// During proof search, calls are accumulated. Before neural evaluation,
/// they are grouped by network name for efficient batched forward passes.
pub struct BatchCollector {
    /// All collected calls in order
    calls: Vec<NeuralCall>,
}

impl BatchCollector {
    /// Create a new empty collector.
    pub fn new() -> Self {
        Self { calls: Vec::new() }
    }

    /// Create a collector with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            calls: Vec::with_capacity(capacity),
        }
    }

    /// Add a neural call to the collector.
    pub fn add(&mut self, call: NeuralCall) {
        self.calls.push(call);
    }

    /// Group calls by network name for batched evaluation.
    ///
    /// Returns a map from network name to list of calls for that network.
    pub fn collect(&self) -> HashMap<String, Vec<&NeuralCall>> {
        let mut batches: HashMap<String, Vec<&NeuralCall>> = HashMap::new();

        for call in &self.calls {
            batches.entry(call.network.clone()).or_default().push(call);
        }

        batches
    }

    /// Get all input indices for a specific network.
    ///
    /// These indices can be used to gather inputs from the tensor source
    /// into a batched tensor for the forward pass.
    pub fn indices_for_network(&self, network: &str) -> Vec<usize> {
        self.calls
            .iter()
            .filter(|c| c.network == network)
            .flat_map(|c| c.input_indices.iter().copied())
            .collect()
    }

    /// Get the names of all networks that have been called.
    pub fn network_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .calls
            .iter()
            .map(|c| c.network.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        names.sort();
        names
    }

    /// Get the number of calls for a specific network.
    pub fn call_count_for_network(&self, network: &str) -> usize {
        self.calls.iter().filter(|c| c.network == network).count()
    }

    /// Get the total number of input indices across all calls.
    pub fn total_input_count(&self) -> usize {
        self.calls.iter().map(|c| c.input_indices.len()).sum()
    }

    /// Total number of calls.
    pub fn len(&self) -> usize {
        self.calls.len()
    }

    /// Check if the collector is empty.
    pub fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }

    /// Clear all collected calls.
    pub fn clear(&mut self) {
        self.calls.clear();
    }

    /// Iterate over all calls.
    pub fn iter(&self) -> impl Iterator<Item = &NeuralCall> {
        self.calls.iter()
    }

    /// Take ownership of all calls.
    pub fn take(&mut self) -> Vec<NeuralCall> {
        std::mem::take(&mut self.calls)
    }
}

impl Default for BatchCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a batched neural evaluation.
///
/// Contains the outputs for all inputs processed in a single batch.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Network that produced these results
    pub network: String,
    /// Output probability distributions, one per input
    /// `outputs[i]` is the softmax output for the i-th input in the batch
    pub outputs: Vec<Vec<f64>>,
}

impl BatchResult {
    /// Create a new batch result.
    pub fn new(network: &str, outputs: Vec<Vec<f64>>) -> Self {
        Self {
            network: network.to_string(),
            outputs,
        }
    }

    /// Get output for a specific index in the batch.
    pub fn get_output(&self, index: usize) -> Option<&Vec<f64>> {
        self.outputs.get(index)
    }

    /// Number of outputs in this batch.
    pub fn len(&self) -> usize {
        self.outputs.len()
    }

    /// Check if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.outputs.is_empty()
    }

    /// Iterate over outputs.
    pub fn iter(&self) -> impl Iterator<Item = &Vec<f64>> {
        self.outputs.iter()
    }
}

/// Mapping from call index to batch result index.
///
/// When calls are batched, we need to track which result corresponds
/// to which original call for reconstructing per-call outputs.
#[derive(Debug, Clone)]
pub struct BatchMapping {
    /// For each original call index, the (batch_network, index_in_batch)
    mappings: Vec<(String, usize)>,
}

impl BatchMapping {
    /// Create a new mapping from a collector.
    pub fn from_collector(collector: &BatchCollector) -> Self {
        let mut mappings = Vec::with_capacity(collector.len());
        let mut network_counts: HashMap<String, usize> = HashMap::new();

        for call in collector.iter() {
            let idx = *network_counts.entry(call.network.clone()).or_insert(0);
            mappings.push((call.network.clone(), idx));
            *network_counts.get_mut(&call.network).unwrap() += 1;
        }

        Self { mappings }
    }

    /// Look up which batch result to use for a given call index.
    pub fn get(&self, call_index: usize) -> Option<(&str, usize)> {
        self.mappings
            .get(call_index)
            .map(|(net, idx)| (net.as_str(), *idx))
    }

    /// Number of mapped calls.
    pub fn len(&self) -> usize {
        self.mappings.len()
    }

    /// Check if the mapping is empty.
    pub fn is_empty(&self) -> bool {
        self.mappings.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_mapping() {
        let mut collector = BatchCollector::new();
        collector.add(NeuralCall::new("net1", vec![0]));
        collector.add(NeuralCall::new("net2", vec![1]));
        collector.add(NeuralCall::new("net1", vec![2]));

        let mapping = BatchMapping::from_collector(&collector);

        assert_eq!(mapping.get(0), Some(("net1", 0)));
        assert_eq!(mapping.get(1), Some(("net2", 0)));
        assert_eq!(mapping.get(2), Some(("net1", 1)));
    }

    #[test]
    fn test_neural_call_num_inputs() {
        let call = NeuralCall::new("test", vec![1, 2, 3, 4]);
        assert_eq!(call.num_inputs(), 4);
    }

    #[test]
    fn test_batch_collector_take() {
        let mut collector = BatchCollector::new();
        collector.add(NeuralCall::new("net", vec![0]));
        collector.add(NeuralCall::new("net", vec![1]));

        let calls = collector.take();
        assert_eq!(calls.len(), 2);
        assert!(collector.is_empty());
    }
}
