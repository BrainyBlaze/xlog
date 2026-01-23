//! Tests for batched neural evaluation
//!
//! BatchCollector groups neural predicate calls by network name,
//! enabling efficient batched GPU evaluation.

#![cfg(feature = "python")]

use xlog_neural::batch::{BatchCollector, NeuralCall};

#[test]
fn test_batch_collector_groups_by_network() {
    let mut collector = BatchCollector::new();

    // Add calls for different networks
    collector.add(NeuralCall::new("net1", vec![0]));
    collector.add(NeuralCall::new("net2", vec![1]));
    collector.add(NeuralCall::new("net1", vec![2]));

    let batches = collector.collect();

    assert_eq!(batches.len(), 2); // 2 networks
    assert_eq!(batches.get("net1").unwrap().len(), 2); // 2 calls
    assert_eq!(batches.get("net2").unwrap().len(), 1); // 1 call
}

#[test]
fn test_batch_collector_indices_for_network() {
    let mut collector = BatchCollector::new();

    collector.add(NeuralCall::new("mnist", vec![0, 1]));
    collector.add(NeuralCall::new("encoder", vec![10]));
    collector.add(NeuralCall::new("mnist", vec![2, 3]));

    let mnist_indices = collector.indices_for_network("mnist");
    assert_eq!(mnist_indices, vec![0, 1, 2, 3]);

    let encoder_indices = collector.indices_for_network("encoder");
    assert_eq!(encoder_indices, vec![10]);
}

#[test]
fn test_batch_collector_clear() {
    let mut collector = BatchCollector::new();

    collector.add(NeuralCall::new("net", vec![0]));
    collector.add(NeuralCall::new("net", vec![1]));

    assert_eq!(collector.len(), 2);

    collector.clear();

    assert_eq!(collector.len(), 0);
    assert!(collector.is_empty());
}

#[test]
fn test_batch_collector_network_names() {
    let mut collector = BatchCollector::new();

    collector.add(NeuralCall::new("alpha", vec![0]));
    collector.add(NeuralCall::new("beta", vec![1]));
    collector.add(NeuralCall::new("gamma", vec![2]));

    let names = collector.network_names();
    assert_eq!(names.len(), 3);
    assert!(names.contains(&"alpha".to_string()));
    assert!(names.contains(&"beta".to_string()));
    assert!(names.contains(&"gamma".to_string()));
}

#[test]
fn test_neural_call_creation() {
    let call = NeuralCall::new("test_net", vec![5, 10, 15]);

    assert_eq!(call.network, "test_net");
    assert_eq!(call.input_indices, vec![5, 10, 15]);
}

#[test]
fn test_neural_call_with_single_index() {
    let call = NeuralCall::single("net", 42);

    assert_eq!(call.network, "net");
    assert_eq!(call.input_indices, vec![42]);
}

#[test]
fn test_batch_collector_call_count_per_network() {
    let mut collector = BatchCollector::new();

    collector.add(NeuralCall::new("net1", vec![0]));
    collector.add(NeuralCall::new("net1", vec![1]));
    collector.add(NeuralCall::new("net1", vec![2]));
    collector.add(NeuralCall::new("net2", vec![3]));

    assert_eq!(collector.call_count_for_network("net1"), 3);
    assert_eq!(collector.call_count_for_network("net2"), 1);
    assert_eq!(collector.call_count_for_network("nonexistent"), 0);
}

#[test]
fn test_batch_collector_total_inputs() {
    let mut collector = BatchCollector::new();

    collector.add(NeuralCall::new("net", vec![0, 1])); // 2 inputs
    collector.add(NeuralCall::new("net", vec![2, 3, 4])); // 3 inputs

    assert_eq!(collector.total_input_count(), 5);
}

#[test]
fn test_batch_collector_iterate_calls() {
    let mut collector = BatchCollector::new();

    collector.add(NeuralCall::new("net1", vec![0]));
    collector.add(NeuralCall::new("net2", vec![1]));

    let calls: Vec<_> = collector.iter().collect();
    assert_eq!(calls.len(), 2);
}

#[test]
fn test_batch_result_creation() {
    use xlog_neural::batch::BatchResult;

    let result = BatchResult::new("test_net", vec![vec![0.7, 0.2, 0.1], vec![0.3, 0.5, 0.2]]);

    assert_eq!(result.network, "test_net");
    assert_eq!(result.outputs.len(), 2);
    assert_eq!(result.outputs[0].len(), 3);
}

#[test]
fn test_batch_result_get_output() {
    use xlog_neural::batch::BatchResult;

    let result = BatchResult::new("net", vec![vec![0.9, 0.1], vec![0.4, 0.6]]);

    assert_eq!(result.get_output(0), Some(&vec![0.9, 0.1]));
    assert_eq!(result.get_output(1), Some(&vec![0.4, 0.6]));
    assert_eq!(result.get_output(2), None);
}

#[test]
fn test_batch_collector_preserves_order() {
    let mut collector = BatchCollector::new();

    // Add in specific order
    collector.add(NeuralCall::new("net", vec![100]));
    collector.add(NeuralCall::new("net", vec![200]));
    collector.add(NeuralCall::new("net", vec![300]));

    let indices = collector.indices_for_network("net");
    assert_eq!(indices, vec![100, 200, 300]); // Order preserved
}
