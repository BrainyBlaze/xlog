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
