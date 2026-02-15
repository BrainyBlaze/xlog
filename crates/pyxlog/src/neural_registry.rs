use std::collections::{HashMap, HashSet};

use xlog_logic::ast::{Atom, NeuralLabel, Program, Term};

#[derive(Debug, Clone)]
pub struct NeuralPredicateInfo {
    pub predicate: String,
    pub network: String,
    pub predicate_terms: Vec<Term>,
    pub input_arity: usize,
    pub input_positions: Vec<usize>,
    pub output_position: usize,
    pub labels: Option<Vec<String>>,
    pub predicate_arity: usize,
}

#[derive(Debug, Default, Clone)]
pub struct NeuralPredicateRegistry {
    by_predicate: HashMap<String, Vec<NeuralPredicateInfo>>,
}

impl NeuralPredicateRegistry {
    pub fn from_ast(ast: &Program) -> Result<Self, String> {
        let mut registry = NeuralPredicateRegistry::default();
        for np in &ast.neural_predicates {
            let predicate = np.predicate.predicate.clone();
            let predicate_arity = np.predicate.terms.len();

            let declared_vars: HashSet<String> = np
                .inputs
                .iter()
                .chain(std::iter::once(&np.output))
                .cloned()
                .collect();
            if declared_vars.len() != np.inputs.len() + 1 {
                return Err(format!(
                    "Invalid nn/4 declaration for predicate '{}': output variable '{}' overlaps an input",
                    predicate, np.output
                ));
            }

            let mut seen_inputs = HashSet::new();
            for input in &np.inputs {
                if !seen_inputs.insert(input.as_str()) {
                    return Err(format!(
                        "Invalid nn/4 declaration for predicate '{}': input variable '{}' appears multiple times",
                        predicate, input
                    ));
                }
            }

            let mut var_positions: HashMap<String, usize> = HashMap::new();
            let mut seen_vars: HashSet<&str> = HashSet::new();
            for (idx, term) in np.predicate.terms.iter().enumerate() {
                match term {
                    Term::Variable(name) => {
                        if !declared_vars.contains(name) {
                            return Err(format!(
                                "Invalid nn/4 declaration for predicate '{}': variable '{}' is not declared in nn/4 inputs/output",
                                predicate, name
                            ));
                        }
                        if !seen_vars.insert(name.as_str()) {
                            return Err(format!(
                                "Invalid nn/4 declaration for predicate '{}': variable '{}' appears multiple times",
                                predicate, name
                            ));
                        }
                        var_positions.insert(name.clone(), idx);
                    }
                    Term::Anonymous => {
                        return Err(format!(
                            "Invalid nn/4 declaration for predicate '{}': anonymous variables are not allowed",
                            predicate
                        ));
                    }
                    Term::Aggregate(_) => {
                        return Err(format!(
                            "Invalid nn/4 declaration for predicate '{}': aggregate terms are not allowed",
                            predicate
                        ));
                    }
                    _ => {}
                }
            }

            if seen_vars.len() != declared_vars.len() {
                let missing: Vec<&str> = declared_vars
                    .iter()
                    .filter(|name| !seen_vars.contains(name.as_str()))
                    .map(|name| name.as_str())
                    .collect();
                return Err(format!(
                    "Invalid nn/4 declaration for predicate '{}': variables {:?} are not present in predicate args",
                    predicate, missing
                ));
            }

            let mut input_positions = Vec::with_capacity(np.inputs.len());
            for input in &np.inputs {
                let pos = *var_positions.get(input).ok_or_else(|| {
                    format!(
                        "Invalid nn/4 declaration for predicate '{}': input variable '{}' not found in predicate args",
                        predicate, input
                    )
                })?;
                input_positions.push(pos);
            }

            let output_position = *var_positions.get(&np.output).ok_or_else(|| {
                format!(
                    "Invalid nn/4 declaration for predicate '{}': output variable '{}' not found in predicate args",
                    predicate, np.output
                )
            })?;

            let labels = np
                .labels
                .clone()
                .map(|label_list| {
                    let labels = label_list
                        .into_iter()
                        .map(|l| match l {
                            NeuralLabel::Integer(i) => i.to_string(),
                            NeuralLabel::Symbol(s) => s,
                        })
                        .collect::<Vec<_>>();

                    let mut unique = HashSet::with_capacity(labels.len());
                    for label in &labels {
                        if !unique.insert(label.as_str()) {
                            return Err(format!(
                                "Invalid nn/4 declaration for predicate '{}': duplicate label '{}'",
                                predicate, label
                            ));
                        }
                    }

                    Ok(labels)
                })
                .transpose()?;

            registry
                .by_predicate
                .entry(predicate.clone())
                .or_default()
                .push(NeuralPredicateInfo {
                    predicate,
                    network: np.network.clone(),
                    predicate_terms: np.predicate.terms.clone(),
                    input_arity: np.inputs.len(),
                    input_positions,
                    output_position,
                    labels,
                    predicate_arity,
                });
        }
        Ok(registry)
    }

    pub fn get(&self, predicate: &str) -> Option<&Vec<NeuralPredicateInfo>> {
        self.by_predicate.get(predicate)
    }

    pub fn resolve_atom(&self, atom: &Atom) -> Result<&NeuralPredicateInfo, String> {
        let infos = self
            .by_predicate
            .get(&atom.predicate)
            .ok_or_else(|| format!("No nn/4 declaration for predicate '{}'", atom.predicate))?;

        let mut matches: Vec<&NeuralPredicateInfo> = Vec::new();
        'candidate_loop: for info in infos.iter() {
            if info.predicate_arity != atom.arity() {
                continue;
            }

            for (decl_term, query_term) in info.predicate_terms.iter().zip(atom.terms.iter()) {
                if !matches_decl_term(decl_term, query_term) {
                    continue 'candidate_loop;
                }
            }
            matches.push(info);
        }

        match matches.len() {
            0 => Err(format!(
                "No nn/4 declaration matches atom '{}' with arity {}",
                atom.predicate,
                atom.arity()
            )),
            1 => Ok(matches[0]),
            _ => Err(format!(
                "Ambiguous nn/4 declaration match for atom '{}' ({} matches)",
                atom.predicate,
                matches.len()
            )),
        }
    }
}

fn matches_decl_term(decl_term: &Term, query_term: &Term) -> bool {
    if matches!(decl_term, Term::Variable(_)) {
        return true;
    }

    decl_term == query_term
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlog_logic::parser;

    fn parse_neural_registry(source: &str) -> Result<NeuralPredicateRegistry, String> {
        let ast = parser::parse_program(source).map_err(|e| format!("{e}"))?;
        NeuralPredicateRegistry::from_ast(&ast)
    }

    #[test]
    fn test_coins_registry_two_declarations() {
        let registry = parse_neural_registry(
            "nn(net1, [X], Y, [heads, tails]) :: coin(1, X, Y).\n\
             nn(net2, [X], Y, [heads, tails]) :: coin(2, X, Y).",
        )
        .expect("registry should parse");

        let infos = registry.get("coin").expect("coin should exist");
        assert_eq!(infos.len(), 2);
        assert_eq!(infos[0].network, "net1");
        assert_eq!(infos[0].labels, Some(vec!["heads".into(), "tails".into()]));
        assert_eq!(infos[0].predicate_arity, 3);
        assert_eq!(infos[1].network, "net2");
        assert_eq!(infos[1].labels, Some(vec!["heads".into(), "tails".into()]));
        assert_eq!(infos[1].predicate_arity, 3);
    }

    #[test]
    fn test_coins_resolve_atom_constant_1() {
        let registry = parse_neural_registry(
            "nn(net1, [X], Y, [heads, tails]) :: coin(1, X, Y).\n\
             nn(net2, [X], Y, [heads, tails]) :: coin(2, X, Y).",
        )
        .expect("registry should parse");

        let atom = Atom {
            predicate: "coin".to_string(),
            terms: vec![
                Term::Integer(1),
                Term::Variable("X".into()),
                Term::Variable("Y".into()),
            ],
        };
        let info = registry.resolve_atom(&atom).expect("atom should resolve");
        assert_eq!(info.network, "net1");
    }

    #[test]
    fn test_coins_resolve_atom_constant_2() {
        let registry = parse_neural_registry(
            "nn(net1, [X], Y, [heads, tails]) :: coin(1, X, Y).\n\
             nn(net2, [X], Y, [heads, tails]) :: coin(2, X, Y).",
        )
        .expect("registry should parse");

        let atom = Atom {
            predicate: "coin".to_string(),
            terms: vec![
                Term::Integer(2),
                Term::Variable("X".into()),
                Term::Variable("Y".into()),
            ],
        };
        let info = registry.resolve_atom(&atom).expect("atom should resolve");
        assert_eq!(info.network, "net2");
    }

    #[test]
    fn test_coins_resolve_atom_variable_is_ambiguous() {
        let registry = parse_neural_registry(
            "nn(net1, [X, Z], Y, [heads, tails]) :: coin(X, Z, Y).\n\
             nn(net2, [X, Z], Y, [heads, tails]) :: coin(X, Z, Y).",
        )
        .expect("registry should parse");

        let atom = Atom {
            predicate: "coin".to_string(),
            terms: vec![
                Term::Variable("C".into()),
                Term::Variable("X".into()),
                Term::Variable("Y".into()),
            ],
        };
        let err = registry
            .resolve_atom(&atom)
            .expect_err("ambiguous should error");
        assert!(err.contains("Ambiguous"));
    }

    #[test]
    fn test_single_declaration_no_constant() {
        let registry = parse_neural_registry("nn(net, [X], Y, [a, b]) :: pred(X, Y).")
            .expect("registry should parse");

        let infos = registry.get("pred").expect("pred should exist");
        assert_eq!(infos.len(), 1);
        let atom = Atom {
            predicate: "pred".to_string(),
            terms: vec![Term::Variable("X".into()), Term::Variable("Y".into())],
        };
        let info = registry
            .resolve_atom(&atom)
            .expect("single declaration should resolve");
        assert_eq!(info.network, "net");
    }

    #[test]
    fn test_duplicate_label_rejected() {
        let err = parse_neural_registry("nn(net, [X], Y, [a, a]) :: pred(X, Y).")
            .expect_err("duplicate labels should be rejected");
        assert!(err.contains("duplicate label"));
    }

    #[test]
    fn test_output_overlaps_input_rejected() {
        let err = parse_neural_registry("nn(net, [X, X], Y, [a]) :: pred(X, Y).")
            .expect_err("repeated input should be rejected");
        assert!(err.contains("appears multiple times") || err.contains("overlaps an input"));
    }
}
