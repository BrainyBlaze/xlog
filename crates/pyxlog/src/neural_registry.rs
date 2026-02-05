use std::collections::HashMap;

use xlog_logic::ast::{NeuralLabel, Program};
use xlog_logic::parse_program;

#[derive(Debug, Clone)]
pub struct NeuralPredicateInfo {
    pub predicate: String,
    pub network: String,
    pub labels: Vec<String>,
}

#[derive(Debug, Default, Clone)]
pub struct NeuralPredicateRegistry {
    by_predicate: HashMap<String, NeuralPredicateInfo>,
}

impl NeuralPredicateRegistry {
    pub fn from_ast(ast: &Program) -> Self {
        let mut registry = NeuralPredicateRegistry::default();
        for np in &ast.neural_predicates {
            let labels = np
                .labels
                .clone()
                .unwrap_or_default()
                .into_iter()
                .map(|l| match l {
                    NeuralLabel::Integer(i) => i.to_string(),
                    NeuralLabel::Symbol(s) => s,
                })
                .collect::<Vec<_>>();

            registry.by_predicate.insert(
                np.predicate.predicate.clone(),
                NeuralPredicateInfo {
                    predicate: np.predicate.predicate.clone(),
                    network: np.network.clone(),
                    labels,
                },
            );
        }
        registry
    }

    pub fn from_source(source: &str) -> Result<Self, String> {
        let ast = parse_program(source).map_err(|e| e.to_string())?;
        Ok(Self::from_ast(&ast))
    }

    pub fn get(&self, predicate: &str) -> Option<&NeuralPredicateInfo> {
        self.by_predicate.get(predicate)
    }
}
