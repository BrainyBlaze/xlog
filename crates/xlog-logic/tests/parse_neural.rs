//! Tests for neural predicate parsing
//!
//! Neural predicates have the syntax:
//!   nn(network, [inputs], output, [labels]) :: pred(args).
//!   nn(network, [inputs], embedding) :: pred(args).

use xlog_logic::parse_program;

#[test]
fn test_parse_neural_predicate_with_labels() {
    let source = r#"nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y)."#;
    let program = parse_program(source).expect("should parse");
    assert_eq!(program.neural_predicates.len(), 1);

    let np = &program.neural_predicates[0];
    assert_eq!(np.network, "mnist_net");
    assert_eq!(np.inputs.len(), 1);
    assert_eq!(np.labels.as_ref().unwrap().len(), 10);
    assert_eq!(np.predicate.predicate, "digit");
    assert_eq!(np.predicate.terms.len(), 2);
}

#[test]
fn test_parse_neural_predicate_embedding() {
    let source = r#"nn(encoder, [Text], Embedding) :: encode(Text, Embedding)."#;
    let program = parse_program(source).expect("should parse");
    assert_eq!(program.neural_predicates.len(), 1);

    let np = &program.neural_predicates[0];
    assert_eq!(np.network, "encoder");
    assert!(np.labels.is_none());
}

#[test]
fn test_parse_neural_predicate_multiple_inputs() {
    let source =
        r#"nn(neural1, [I1, I2, Carry], O, [0,1,2,3,4,5,6,7,8,9]) :: result(I1, I2, Carry, O)."#;
    let program = parse_program(source).expect("should parse");
    assert_eq!(program.neural_predicates.len(), 1);

    let np = &program.neural_predicates[0];
    assert_eq!(np.network, "neural1");
    assert_eq!(np.inputs.len(), 3);
    assert_eq!(np.labels.as_ref().unwrap().len(), 10);
}

#[test]
fn test_parse_neural_predicate_symbol_labels() {
    let source = r#"nn(coin_net, [X], Y, [heads, tails]) :: coin(X, Y)."#;
    let program = parse_program(source).expect("should parse");
    assert_eq!(program.neural_predicates.len(), 1);

    let np = &program.neural_predicates[0];
    assert_eq!(np.labels.as_ref().unwrap().len(), 2);
}

#[test]
fn test_parse_multiple_neural_predicates() {
    let source = r#"
        nn(net1, [X], Y, [0,1]) :: digit1(X, Y).
        nn(net2, [X], Y, [0,1]) :: digit2(X, Y).
        addition(X, Y, Z) :- digit1(X, LeftDigit), digit2(Y, RightDigit), Z is LeftDigit + RightDigit.
    "#;
    let program = parse_program(source).expect("should parse");
    assert_eq!(program.neural_predicates.len(), 2);
    assert_eq!(program.rules.len(), 1);
}

#[test]
fn test_parse_neural_with_empty_inputs() {
    let source = r#"nn(global_net, [], Y, [a, b, c]) :: global_pred(Y)."#;
    let program = parse_program(source).expect("should parse");
    assert_eq!(program.neural_predicates.len(), 1);

    let np = &program.neural_predicates[0];
    assert_eq!(np.inputs.len(), 0);
}
