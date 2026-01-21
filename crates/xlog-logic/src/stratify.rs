//! Stratification analysis for negation and aggregation

use crate::ast::{BodyLiteral, ProbEngine, Program};
use std::collections::{HashMap, HashSet};
use xlog_core::{Result, XlogError};

/// Dependency edge type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepType {
    Positive,
    Negative,
    Aggregate,
}

/// Dependency graph edge
#[derive(Debug, Clone)]
pub struct DepEdge {
    pub from: String,
    pub to: String,
    pub dep_type: DepType,
}

/// Dependency graph for stratification analysis
#[derive(Debug, Default)]
pub struct DependencyGraph {
    pub predicates: HashSet<String>,
    pub edges: Vec<DepEdge>,
}

impl DependencyGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_predicate(&mut self, name: String) {
        self.predicates.insert(name);
    }

    pub fn add_edge(&mut self, from: String, to: String, dep_type: DepType) {
        self.predicates.insert(from.clone());
        self.predicates.insert(to.clone());
        self.edges.push(DepEdge { from, to, dep_type });
    }

    pub fn outgoing(&self, pred: &str) -> Vec<&DepEdge> {
        self.edges.iter().filter(|e| e.from == pred).collect()
    }
}

/// Build dependency graph from program
pub fn build_dependency_graph(program: &Program) -> DependencyGraph {
    let mut graph = DependencyGraph::new();

    for rule in &program.rules {
        let head = &rule.head.predicate;
        graph.add_predicate(head.clone());

        for lit in &rule.body {
            match lit {
                BodyLiteral::Positive(atom) => {
                    graph.add_edge(head.clone(), atom.predicate.clone(), DepType::Positive);
                }
                BodyLiteral::Negated(atom) => {
                    graph.add_edge(head.clone(), atom.predicate.clone(), DepType::Negative);
                }
                BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) => {}
            }
        }

        if rule.has_aggregation() {
            for lit in &rule.body {
                if let BodyLiteral::Positive(atom) = lit {
                    graph.add_edge(head.clone(), atom.predicate.clone(), DepType::Aggregate);
                }
            }
        }
    }

    graph
}

/// Find strongly connected components using Tarjan's algorithm
/// Returns SCCs in reverse topological order (dependencies first)
fn find_sccs(graph: &DependencyGraph) -> Vec<Vec<String>> {
    let mut index_counter = 0;
    let mut stack = Vec::new();
    let mut indices: HashMap<String, usize> = HashMap::new();
    let mut lowlinks: HashMap<String, usize> = HashMap::new();
    let mut on_stack: HashSet<String> = HashSet::new();
    let mut sccs: Vec<Vec<String>> = Vec::new();

    #[allow(clippy::too_many_arguments)]
    fn strongconnect(
        v: &str,
        graph: &DependencyGraph,
        index_counter: &mut usize,
        stack: &mut Vec<String>,
        indices: &mut HashMap<String, usize>,
        lowlinks: &mut HashMap<String, usize>,
        on_stack: &mut HashSet<String>,
        sccs: &mut Vec<Vec<String>>,
    ) {
        indices.insert(v.to_string(), *index_counter);
        lowlinks.insert(v.to_string(), *index_counter);
        *index_counter += 1;
        stack.push(v.to_string());
        on_stack.insert(v.to_string());

        for edge in graph.outgoing(v) {
            let w = &edge.to;
            if !indices.contains_key(w) {
                strongconnect(
                    w,
                    graph,
                    index_counter,
                    stack,
                    indices,
                    lowlinks,
                    on_stack,
                    sccs,
                );
                let low_v = *lowlinks.get(v).unwrap();
                let low_w = *lowlinks.get(w).unwrap();
                lowlinks.insert(v.to_string(), low_v.min(low_w));
            } else if on_stack.contains(w) {
                let low_v = *lowlinks.get(v).unwrap();
                let idx_w = *indices.get(w).unwrap();
                lowlinks.insert(v.to_string(), low_v.min(idx_w));
            }
        }

        let low_v = *lowlinks.get(v).unwrap();
        let idx_v = *indices.get(v).unwrap();
        if low_v == idx_v {
            let mut scc = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack.remove(&w);
                scc.push(w.clone());
                if w == v {
                    break;
                }
            }
            sccs.push(scc);
        }
    }

    for pred in &graph.predicates {
        if !indices.contains_key(pred) {
            strongconnect(
                pred,
                graph,
                &mut index_counter,
                &mut stack,
                &mut indices,
                &mut lowlinks,
                &mut on_stack,
                &mut sccs,
            );
        }
    }

    sccs
}

/// Check for cycles through negation/aggregation in an SCC
fn check_scc_for_negation_cycle(scc: &[String], graph: &DependencyGraph) -> Option<Vec<String>> {
    if scc.len() == 1 {
        let pred = &scc[0];
        for edge in graph.outgoing(pred) {
            if edge.to == *pred && edge.dep_type != DepType::Positive {
                return Some(vec![pred.clone()]);
            }
        }
        return None;
    }

    let scc_set: HashSet<&str> = scc.iter().map(|s| s.as_str()).collect();
    for pred in scc {
        for edge in graph.outgoing(pred) {
            if scc_set.contains(edge.to.as_str()) && edge.dep_type != DepType::Positive {
                return Some(scc.to_vec());
            }
        }
    }
    None
}

/// Stratum assignment result
#[derive(Debug, Clone)]
pub struct Stratum {
    pub id: usize,
    pub predicates: Vec<String>,
}

/// Result of stratification analysis for probabilistic inference
#[derive(Debug, Clone)]
pub struct StratificationResult {
    /// SCCs in evaluation order (dependencies first)
    pub sccs: Vec<Vec<String>>,
    /// Indices of SCCs that have cycles through negation (non-monotone)
    pub non_monotone_sccs: HashSet<usize>,
    /// Stratum number for each predicate (if fully stratified)
    pub strata: HashMap<String, usize>,
}

/// Perform stratification analysis
pub fn stratify(program: &Program) -> Result<Vec<Stratum>> {
    let graph = build_dependency_graph(program);
    let sccs = find_sccs(&graph);

    for scc in &sccs {
        if let Some(cycle) = check_scc_for_negation_cycle(scc, &graph) {
            if program.is_probabilistic_profile() && program.prob_engine() != ProbEngine::Mc {
                return Err(XlogError::Compilation(format!(
                    "Non-monotone recursion detected (cycle through negation/aggregation involving {:?}); requires P3 (`prob_engine=mc`)",
                    cycle
                )));
            }

            if !program.is_probabilistic_profile() {
                return Err(XlogError::StratificationCycle(cycle));
            }
        }
    }

    let mut stratum_map: HashMap<String, usize> = HashMap::new();
    let mut max_stratum = 0;

    // Tarjan produces SCCs in reverse topological order.
    // Since edges go from "dependent" to "dependency" (head -> body predicate),
    // SCCs of dependencies come first. Process in order to ensure dependencies
    // are assigned strata before dependents.
    for scc in &sccs {
        let mut min_stratum = 0;
        for pred in scc {
            for edge in graph.outgoing(pred) {
                if let Some(&dep_stratum) = stratum_map.get(&edge.to) {
                    let required = match edge.dep_type {
                        DepType::Positive => dep_stratum,
                        DepType::Negative | DepType::Aggregate => dep_stratum + 1,
                    };
                    min_stratum = min_stratum.max(required);
                }
            }
        }
        for pred in scc {
            stratum_map.insert(pred.clone(), min_stratum);
        }
        max_stratum = max_stratum.max(min_stratum);
    }

    let mut strata: Vec<Stratum> = (0..=max_stratum)
        .map(|id| Stratum {
            id,
            predicates: vec![],
        })
        .collect();

    for (pred, stratum) in stratum_map {
        strata[stratum].predicates.push(pred);
    }

    strata.retain(|s| !s.predicates.is_empty());
    for (i, stratum) in strata.iter_mut().enumerate() {
        stratum.id = i;
    }

    Ok(strata)
}

/// Analyze stratification for probabilistic inference
/// Returns detailed information about SCCs and which ones are non-monotone
pub fn analyze_stratification(program: &Program) -> StratificationResult {
    let graph = build_dependency_graph(program);
    let sccs = find_sccs(&graph);

    let mut non_monotone_sccs: HashSet<usize> = HashSet::new();
    for (i, scc) in sccs.iter().enumerate() {
        if check_scc_for_negation_cycle(scc, &graph).is_some() {
            non_monotone_sccs.insert(i);
        }
    }

    // Compute strata for predicates in stratified SCCs
    let mut strata: HashMap<String, usize> = HashMap::new();
    let mut max_stratum = 0;

    for (scc_idx, scc) in sccs.iter().enumerate() {
        if non_monotone_sccs.contains(&scc_idx) {
            continue; // Skip non-monotone SCCs for stratum assignment
        }

        let mut min_stratum = 0;
        for pred in scc {
            for edge in graph.outgoing(pred) {
                if let Some(&dep_stratum) = strata.get(&edge.to) {
                    let required = match edge.dep_type {
                        DepType::Positive => dep_stratum,
                        DepType::Negative | DepType::Aggregate => dep_stratum + 1,
                    };
                    min_stratum = min_stratum.max(required);
                }
            }
        }
        for pred in scc {
            strata.insert(pred.clone(), min_stratum);
        }
        max_stratum = max_stratum.max(min_stratum);
    }

    StratificationResult {
        sccs,
        non_monotone_sccs,
        strata,
    }
}

/// Find SCCs for the lowering phase
/// Returns SCCs in reverse topological order (dependencies first)
pub fn find_sccs_for_lowering(graph: &DependencyGraph) -> Vec<Vec<String>> {
    find_sccs(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;

    fn create_tc_program() -> Program {
        let mut program = Program::new();
        program.rules.push(Rule {
            head: Atom {
                predicate: "edge".into(),
                terms: vec![Term::Integer(1), Term::Integer(2)],
            },
            body: vec![],
        });
        program.rules.push(Rule {
            head: Atom {
                predicate: "reach".into(),
                terms: vec![Term::Variable("X".into()), Term::Variable("Y".into())],
            },
            body: vec![BodyLiteral::Positive(Atom {
                predicate: "edge".into(),
                terms: vec![Term::Variable("X".into()), Term::Variable("Y".into())],
            })],
        });
        program.rules.push(Rule {
            head: Atom {
                predicate: "reach".into(),
                terms: vec![Term::Variable("X".into()), Term::Variable("Z".into())],
            },
            body: vec![
                BodyLiteral::Positive(Atom {
                    predicate: "reach".into(),
                    terms: vec![Term::Variable("X".into()), Term::Variable("Y".into())],
                }),
                BodyLiteral::Positive(Atom {
                    predicate: "edge".into(),
                    terms: vec![Term::Variable("Y".into()), Term::Variable("Z".into())],
                }),
            ],
        });
        program
    }

    fn create_isolated_program() -> Program {
        let mut program = Program::new();
        for i in 1..=3 {
            program.rules.push(Rule {
                head: Atom {
                    predicate: "node".into(),
                    terms: vec![Term::Integer(i)],
                },
                body: vec![],
            });
        }
        program.rules.push(Rule {
            head: Atom {
                predicate: "edge".into(),
                terms: vec![Term::Integer(1), Term::Integer(2)],
            },
            body: vec![],
        });
        program.rules.push(Rule {
            head: Atom {
                predicate: "isolated".into(),
                terms: vec![Term::Variable("X".into())],
            },
            body: vec![
                BodyLiteral::Positive(Atom {
                    predicate: "node".into(),
                    terms: vec![Term::Variable("X".into())],
                }),
                BodyLiteral::Negated(Atom {
                    predicate: "edge".into(),
                    terms: vec![Term::Variable("X".into()), Term::Variable("Y".into())],
                }),
            ],
        });
        program
    }

    fn create_unstratifiable_program() -> Program {
        let mut program = Program::new();
        program.rules.push(Rule {
            head: Atom {
                predicate: "p".into(),
                terms: vec![],
            },
            body: vec![BodyLiteral::Negated(Atom {
                predicate: "q".into(),
                terms: vec![],
            })],
        });
        program.rules.push(Rule {
            head: Atom {
                predicate: "q".into(),
                terms: vec![],
            },
            body: vec![BodyLiteral::Negated(Atom {
                predicate: "p".into(),
                terms: vec![],
            })],
        });
        program
    }

    #[test]
    fn test_stratify_simple() {
        let program = create_tc_program();
        let result = stratify(&program);
        assert!(result.is_ok(), "Stratification failed: {:?}", result.err());
    }

    #[test]
    fn test_stratify_with_negation() {
        let program = create_isolated_program();
        let result = stratify(&program);
        assert!(result.is_ok(), "Stratification failed: {:?}", result.err());
        let strata = result.unwrap();
        assert!(
            strata.len() >= 2,
            "Expected at least 2 strata, got {}",
            strata.len()
        );
    }

    #[test]
    fn test_stratify_cycle_through_negation() {
        let program = create_unstratifiable_program();
        let result = stratify(&program);
        assert!(result.is_err(), "Should fail with cycle through negation");
        if let Err(XlogError::StratificationCycle(preds)) = result {
            assert!(preds.contains(&"p".to_string()) || preds.contains(&"q".to_string()));
        }
    }

    #[test]
    fn test_stratify_probabilistic_non_monotone_requires_mc() {
        let mut program = create_unstratifiable_program();
        program.directives.prob_engine = Some(ProbEngine::ExactDdnnf);

        let result = stratify(&program);
        match result {
            Err(XlogError::Compilation(msg)) => {
                assert!(msg.contains("requires P3"), "msg={}", msg);
                assert!(msg.contains("prob_engine=mc"), "msg={}", msg);
            }
            other => panic!("Expected Compilation error, got: {:?}", other),
        }
    }

    #[test]
    fn test_stratify_probabilistic_non_monotone_allows_mc() {
        let mut program = create_unstratifiable_program();
        program.directives.prob_engine = Some(ProbEngine::Mc);

        let result = stratify(&program);
        assert!(
            result.is_ok(),
            "Expected mc to allow non-monotone recursion, got: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_dependency_graph_construction() {
        let program = create_tc_program();
        let graph = build_dependency_graph(&program);
        assert!(graph.predicates.contains("edge"));
        assert!(graph.predicates.contains("reach"));
        let reach_deps = graph.outgoing("reach");
        assert!(!reach_deps.is_empty());
    }

    #[test]
    fn test_analyze_stratification_detects_non_monotone() {
        let program = create_unstratifiable_program(); // p :- not q. q :- not p.
        let result = analyze_stratification(&program);

        assert!(!result.non_monotone_sccs.is_empty(), "Should detect non-monotone SCC");
        // The SCC containing p and q should be marked as non-monotone
        let has_non_monotone = result.sccs.iter().enumerate().any(|(i, scc)| {
            result.non_monotone_sccs.contains(&i) &&
            (scc.contains(&"p".to_string()) || scc.contains(&"q".to_string()))
        });
        assert!(has_non_monotone, "SCC with p/q should be non-monotone");
    }

    #[test]
    fn test_analyze_stratification_stratified_program() {
        let program = create_isolated_program(); // isolated(X) :- node(X), not edge(X, Y).
        let result = analyze_stratification(&program);

        assert!(result.non_monotone_sccs.is_empty(), "Stratified program has no non-monotone SCCs");
        assert!(result.strata.contains_key("isolated"), "isolated should have a stratum");
        assert!(result.strata.contains_key("edge"), "edge should have a stratum");

        // isolated depends negatively on edge, so isolated.stratum > edge.stratum
        let isolated_stratum = result.strata.get("isolated").unwrap();
        let edge_stratum = result.strata.get("edge").unwrap();
        assert!(isolated_stratum > edge_stratum, "isolated should be in higher stratum than edge");
    }
}
