//! Tests for the neural → probability bridge
//!
//! The bridge converts neural network outputs (softmax probabilities)
//! into probabilistic logic constructs (annotated disjunctions, circuit leaves).

use xlog_neural::bridge::{NeuralBridge, NeuralOutput};

#[test]
fn test_softmax_to_ad_probabilities() {
    // Network outputs softmax: [0.7, 0.2, 0.1] for labels [a, b, c]
    let output = NeuralOutput {
        values: vec![0.7, 0.2, 0.1],
        labels: vec!["a".to_string(), "b".to_string(), "c".to_string()],
    };

    let bridge = NeuralBridge::new();
    let probs = bridge.to_ad_probabilities(&output);

    // Should produce: 0.7::pred(X, a); 0.2::pred(X, b); 0.1::pred(X, c)
    assert_eq!(probs.len(), 3);
    assert!((probs[0].probability - 0.7).abs() < 1e-6);
    assert_eq!(probs[0].label, "a");
    assert!((probs[1].probability - 0.2).abs() < 1e-6);
    assert_eq!(probs[1].label, "b");
    assert!((probs[2].probability - 0.1).abs() < 1e-6);
    assert_eq!(probs[2].label, "c");
}

#[test]
fn test_batch_to_circuit_leaves() {
    // Batch of 2 samples, 3 labels each
    let outputs = vec![
        NeuralOutput {
            values: vec![0.7, 0.2, 0.1],
            labels: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        },
        NeuralOutput {
            values: vec![0.1, 0.3, 0.6],
            labels: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        },
    ];

    let bridge = NeuralBridge::new();
    let leaves = bridge.batch_to_circuit_leaves(&outputs);

    // Should produce leaf weights for circuit evaluation
    assert_eq!(leaves.len(), 2); // 2 samples
    assert_eq!(leaves[0].len(), 3); // 3 labels each
    assert_eq!(leaves[1].len(), 3);

    // Verify weights
    assert!((leaves[0][0].weight - 0.7).abs() < 1e-6);
    assert!((leaves[1][2].weight - 0.6).abs() < 1e-6);
}

#[test]
fn test_to_log_probabilities() {
    let output = NeuralOutput {
        values: vec![0.5, 0.3, 0.2],
        labels: vec!["x".to_string(), "y".to_string(), "z".to_string()],
    };

    let bridge = NeuralBridge::new();
    let log_probs = bridge.to_log_probabilities(&output);

    assert_eq!(log_probs.len(), 3);
    assert!((log_probs[0] - 0.5_f64.ln()).abs() < 1e-6);
    assert!((log_probs[1] - 0.3_f64.ln()).abs() < 1e-6);
    assert!((log_probs[2] - 0.2_f64.ln()).abs() < 1e-6);
}

#[test]
fn test_epsilon_clamping_prevents_log_zero() {
    // Zero probability should be clamped to epsilon
    let output = NeuralOutput {
        values: vec![1.0, 0.0, 0.0],
        labels: vec!["a".to_string(), "b".to_string(), "c".to_string()],
    };

    let bridge = NeuralBridge::new();
    let log_probs = bridge.to_log_probabilities(&output);

    // Should not be -inf due to epsilon clamping
    assert!(log_probs[1].is_finite());
    assert!(log_probs[2].is_finite());
}

#[test]
fn test_custom_epsilon() {
    let bridge = NeuralBridge::with_epsilon(1e-4);

    let output = NeuralOutput {
        values: vec![0.0],
        labels: vec!["x".to_string()],
    };

    let probs = bridge.to_ad_probabilities(&output);
    assert!((probs[0].probability - 1e-4).abs() < 1e-10);
}

#[test]
fn test_integer_labels() {
    let output = NeuralOutput::with_integer_labels(vec![0.7, 0.2, 0.1], vec![0, 1, 2]);

    let bridge = NeuralBridge::new();
    let probs = bridge.to_ad_probabilities(&output);

    assert_eq!(probs[0].label, "0");
    assert_eq!(probs[1].label, "1");
    assert_eq!(probs[2].label, "2");
}

#[test]
fn test_normalize_probabilities() {
    // Probabilities that don't quite sum to 1 (numerical error)
    let output = NeuralOutput {
        values: vec![0.33, 0.33, 0.33],
        labels: vec!["a".to_string(), "b".to_string(), "c".to_string()],
    };

    let bridge = NeuralBridge::new();
    let normalized = bridge.normalize(&output);

    let sum: f64 = normalized.values.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
}

#[test]
fn test_batch_size() {
    let outputs = vec![
        NeuralOutput {
            values: vec![0.5, 0.5],
            labels: vec!["a".to_string(), "b".to_string()],
        },
        NeuralOutput {
            values: vec![0.3, 0.7],
            labels: vec!["a".to_string(), "b".to_string()],
        },
        NeuralOutput {
            values: vec![0.9, 0.1],
            labels: vec!["a".to_string(), "b".to_string()],
        },
    ];

    let bridge = NeuralBridge::new();
    let leaves = bridge.batch_to_circuit_leaves(&outputs);

    assert_eq!(leaves.len(), 3);
}

#[test]
fn test_gradient_weights() {
    // Test that we can extract weights suitable for gradient computation
    let output = NeuralOutput {
        values: vec![0.8, 0.15, 0.05],
        labels: vec!["cat".to_string(), "dog".to_string(), "bird".to_string()],
    };

    let bridge = NeuralBridge::new();
    let weights = bridge.extract_gradient_weights(&output);

    assert_eq!(weights.len(), 3);
    assert!((weights[0] - 0.8).abs() < 1e-6);
    assert!((weights[1] - 0.15).abs() < 1e-6);
    assert!((weights[2] - 0.05).abs() < 1e-6);
}
