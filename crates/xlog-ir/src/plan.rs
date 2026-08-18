//! Execution plan representation

use crate::metadata::RirMeta;
use crate::rir::RirNode;
use xlog_core::RelId;

/// Strongly Connected Component in the dependency graph
#[derive(Debug, Clone)]
pub struct Scc {
    /// Unique SCC identifier
    pub id: u32,
    /// Predicate names in this SCC
    pub predicates: Vec<String>,
    /// Whether this SCC contains recursion
    pub is_recursive: bool,
}

/// Stratum in stratified evaluation
#[derive(Debug, Clone)]
pub struct Stratum {
    /// Stratum number (0 = base)
    pub id: u32,
    /// SCCs in this stratum (topologically ordered)
    pub sccs: Vec<u32>,
}

/// Compiled rule ready for execution
#[derive(Debug, Clone)]
pub struct CompiledRule {
    /// Head predicate name
    pub head: String,
    /// RIR tree for rule body
    pub body: RirNode,
    /// Metadata for cost estimation
    pub meta: RirMeta,
}

/// Compiler-produced provenance for one desugared program query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedQueryRuleProvenance {
    /// Zero-based position of the query in the authored program.
    pub query_index: usize,
    /// Position of the generated rule's SCC in [`ExecutionPlan::rules_by_scc`].
    pub scc_index: usize,
    /// Position of the generated rule within its SCC.
    pub rule_index: usize,
}

/// Complete execution plan for a program
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// SCCs in dependency order
    pub sccs: Vec<Scc>,
    /// Strata for negation ordering
    pub strata: Vec<Stratum>,
    /// Compiled rules grouped by SCC
    pub rules_by_scc: Vec<Vec<CompiledRule>>,
    /// Exact compiled rule positions originating from authored program queries.
    /// Manually assembled plans leave this empty unless they explicitly model
    /// generated-query semantics.
    pub generated_query_rules: Vec<GeneratedQueryRuleProvenance>,
    /// Total estimated memory peak (bytes)
    pub est_memory_peak: u64,
    /// Relation arities known at lowering time (every predicate the
    /// lowerer assigned a RelId). Consumed by shape promoters that
    /// must size Scan leaves without schema access (general Free Join
    /// multiway promotion).
    pub rel_arities: std::collections::HashMap<RelId, usize>,
}

impl ExecutionPlan {
    /// Create a new execution plan from SCCs
    pub fn new(sccs: Vec<Scc>) -> Self {
        Self {
            sccs,
            strata: vec![],
            rules_by_scc: vec![],
            generated_query_rules: vec![],
            est_memory_peak: 0,
            rel_arities: std::collections::HashMap::new(),
        }
    }

    /// Add strata to the plan
    pub fn with_strata(mut self, strata: Vec<Stratum>) -> Self {
        self.strata = strata;
        self
    }

    /// Get the number of recursive SCCs
    pub fn recursive_scc_count(&self) -> usize {
        self.sccs.iter().filter(|s| s.is_recursive).count()
    }

    /// Return the dependency-closed subplan for a set of root SCC positions.
    ///
    /// `defining_sccs` maps derived relation IDs to the SCC that defines them.
    /// Relations absent from that map are treated as extensional inputs. The
    /// projection preserves source order, retains complete SCCs, and rewrites
    /// every positional index in the plan. `None` means the supplied plan or
    /// dependency proof is inconsistent, so callers must keep the original plan.
    pub fn dependency_closed_subplan(
        &self,
        root_sccs: &[usize],
        defining_sccs: &std::collections::HashMap<RelId, usize>,
    ) -> Option<Self> {
        let scc_count = self.sccs.len();
        if root_sccs.is_empty()
            || self.rules_by_scc.len() != scc_count
            || self
                .sccs
                .iter()
                .enumerate()
                .any(|(index, scc)| u32::try_from(index).ok() != Some(scc.id))
            || self
                .strata
                .iter()
                .enumerate()
                .any(|(index, stratum)| u32::try_from(index).ok() != Some(stratum.id))
            || root_sccs.iter().any(|root| *root >= scc_count)
            || defining_sccs.values().any(|scc| *scc >= scc_count)
        {
            return None;
        }

        let mut stratum_membership = vec![0_u8; scc_count];
        for stratum in &self.strata {
            for scc in &stratum.sccs {
                let scc_index = usize::try_from(*scc).ok()?;
                let membership = stratum_membership.get_mut(scc_index)?;
                *membership = membership.checked_add(1)?;
                if *membership != 1 {
                    return None;
                }
            }
        }
        if stratum_membership.iter().any(|membership| *membership != 1) {
            return None;
        }

        for query in &self.generated_query_rules {
            self.rules_by_scc
                .get(query.scc_index)?
                .get(query.rule_index)?;
        }

        let mut retained = std::collections::BTreeSet::new();
        let mut pending = root_sccs.to_vec();
        while let Some(scc_index) = pending.pop() {
            if !retained.insert(scc_index) {
                continue;
            }
            for rule in &self.rules_by_scc[scc_index] {
                for relation in rule.body.referenced_relations() {
                    let Some(dependency) = defining_sccs.get(&relation).copied() else {
                        continue;
                    };
                    if dependency >= scc_count {
                        return None;
                    }
                    if !retained.contains(&dependency) {
                        pending.push(dependency);
                    }
                }
            }
        }
        let mut remapped_sccs = vec![None; scc_count];
        let mut sccs = Vec::with_capacity(retained.len());
        let mut rules_by_scc = Vec::with_capacity(retained.len());
        for old_index in 0..scc_count {
            if !retained.contains(&old_index) {
                continue;
            }
            let new_index = u32::try_from(sccs.len()).ok()?;
            remapped_sccs[old_index] = Some(new_index);
            let mut scc = self.sccs[old_index].clone();
            scc.id = new_index;
            sccs.push(scc);
            rules_by_scc.push(self.rules_by_scc[old_index].clone());
        }

        let mut strata = Vec::new();
        for original in &self.strata {
            let mut remapped = Vec::new();
            for scc in &original.sccs {
                let old_index = usize::try_from(*scc).ok()?;
                if let Some(new_index) = remapped_sccs.get(old_index).copied().flatten() {
                    remapped.push(new_index);
                }
            }
            if remapped.is_empty() {
                continue;
            }
            strata.push(Stratum {
                id: u32::try_from(strata.len()).ok()?,
                sccs: remapped,
            });
        }

        let generated_query_rules = self
            .generated_query_rules
            .iter()
            .map(|query| {
                Some(GeneratedQueryRuleProvenance {
                    query_index: query.query_index,
                    scc_index: remapped_sccs[query.scc_index]? as usize,
                    rule_index: query.rule_index,
                })
            })
            .collect::<Option<Vec<_>>>()?;

        Some(Self {
            sccs,
            strata,
            rules_by_scc,
            generated_query_rules,
            est_memory_peak: self.est_memory_peak,
            rel_arities: self.rel_arities.clone(),
        })
    }

    /// Check if this plan has any recursion
    pub fn has_recursion(&self) -> bool {
        self.sccs.iter().any(|s| s.is_recursive)
    }
}

/// Builder for execution plans
#[derive(Debug, Default)]
pub struct PlanBuilder {
    sccs: Vec<Scc>,
    strata: Vec<Stratum>,
    rules: Vec<Vec<CompiledRule>>,
}

impl PlanBuilder {
    /// Create a new empty plan builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a strongly connected component to the plan.
    pub fn add_scc(&mut self, scc: Scc) -> &mut Self {
        self.sccs.push(scc);
        self.rules.push(vec![]);
        self
    }

    /// Add a compiled rule to the given SCC (by index).
    pub fn add_rule(&mut self, scc_id: u32, rule: CompiledRule) -> &mut Self {
        if let Some(rules) = self.rules.get_mut(scc_id as usize) {
            rules.push(rule);
        }
        self
    }

    /// Append a stratum to the plan.
    pub fn add_stratum(&mut self, stratum: Stratum) -> &mut Self {
        self.strata.push(stratum);
        self
    }

    /// Consume the builder and produce the final [`ExecutionPlan`].
    pub fn build(self) -> ExecutionPlan {
        ExecutionPlan {
            sccs: self.sccs,
            strata: self.strata,
            rules_by_scc: self.rules,
            generated_query_rules: vec![],
            est_memory_peak: 0,
            rel_arities: std::collections::HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scc_ordering() {
        let sccs = vec![
            Scc {
                id: 0,
                predicates: vec!["edge".into()],
                is_recursive: false,
            },
            Scc {
                id: 1,
                predicates: vec!["reach".into()],
                is_recursive: true,
            },
        ];
        let plan = ExecutionPlan::new(sccs);
        assert_eq!(plan.sccs.len(), 2);
        assert!(!plan.sccs[0].is_recursive);
        assert!(plan.sccs[1].is_recursive);
    }

    #[test]
    fn test_stratum_assignment() {
        let strata = [
            Stratum {
                id: 0,
                sccs: vec![0, 1],
            },
            Stratum {
                id: 1,
                sccs: vec![2],
            },
        ];
        assert_eq!(strata[0].sccs.len(), 2);
    }

    #[test]
    fn test_plan_builder() {
        let mut builder = PlanBuilder::new();
        builder.add_scc(Scc {
            id: 0,
            predicates: vec!["p".into()],
            is_recursive: false,
        });
        builder.add_stratum(Stratum {
            id: 0,
            sccs: vec![0],
        });
        let plan = builder.build();
        assert_eq!(plan.sccs.len(), 1);
        assert_eq!(plan.strata.len(), 1);
    }

    fn compiled_rule(head: &str, body: RirNode) -> CompiledRule {
        CompiledRule {
            head: head.into(),
            body,
            meta: RirMeta::default(),
        }
    }

    #[test]
    fn dependency_closed_subplan_keeps_whole_sccs_and_remaps_plan_indices() {
        let mut plan = ExecutionPlan {
            sccs: vec![
                Scc {
                    id: 0,
                    predicates: vec!["source".into()],
                    is_recursive: false,
                },
                Scc {
                    id: 1,
                    predicates: vec!["disconnected".into()],
                    is_recursive: false,
                },
                Scc {
                    id: 2,
                    predicates: vec!["reachable".into()],
                    is_recursive: true,
                },
                Scc {
                    id: 3,
                    predicates: vec!["audited".into()],
                    is_recursive: false,
                },
                Scc {
                    id: 4,
                    predicates: vec!["__xlog_constraint_0".into()],
                    is_recursive: false,
                },
                Scc {
                    id: 5,
                    predicates: vec!["__xlog_query_0".into()],
                    is_recursive: false,
                },
            ],
            strata: vec![
                Stratum {
                    id: 0,
                    sccs: vec![0],
                },
                Stratum {
                    id: 1,
                    sccs: vec![1],
                },
                Stratum {
                    id: 2,
                    sccs: vec![2, 3],
                },
                Stratum {
                    id: 3,
                    sccs: vec![4, 5],
                },
            ],
            rules_by_scc: vec![
                vec![compiled_rule("source", RirNode::Scan { rel: RelId(1) })],
                vec![compiled_rule(
                    "disconnected",
                    RirNode::Scan { rel: RelId(2) },
                )],
                vec![compiled_rule(
                    "reachable",
                    RirNode::Union {
                        inputs: vec![
                            RirNode::Scan { rel: RelId(10) },
                            RirNode::Scan { rel: RelId(12) },
                        ],
                    },
                )],
                vec![compiled_rule("audited", RirNode::Scan { rel: RelId(10) })],
                vec![compiled_rule(
                    "__xlog_constraint_0",
                    RirNode::Scan { rel: RelId(13) },
                )],
                vec![compiled_rule(
                    "__xlog_query_0",
                    RirNode::Scan { rel: RelId(12) },
                )],
            ],
            generated_query_rules: vec![GeneratedQueryRuleProvenance {
                query_index: 0,
                scc_index: 5,
                rule_index: 0,
            }],
            est_memory_peak: 123,
            rel_arities: [(RelId(10), 1), (RelId(12), 1)].into_iter().collect(),
        };
        let original_heads = plan
            .rules_by_scc
            .iter()
            .flatten()
            .map(|rule| rule.head.clone())
            .collect::<Vec<_>>();
        let defining_sccs = [
            (RelId(10), 0),
            (RelId(11), 1),
            (RelId(12), 2),
            (RelId(13), 3),
            (RelId(14), 4),
            (RelId(15), 5),
        ]
        .into_iter()
        .collect();

        let projected = plan
            .dependency_closed_subplan(&[4, 5], &defining_sccs)
            .expect("consistent dependency closure");

        assert_eq!(
            projected
                .rules_by_scc
                .iter()
                .flatten()
                .map(|rule| rule.head.as_str())
                .collect::<Vec<_>>(),
            vec![
                "source",
                "reachable",
                "audited",
                "__xlog_constraint_0",
                "__xlog_query_0",
            ]
        );
        assert_eq!(
            projected.sccs.iter().map(|scc| scc.id).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
        assert_eq!(
            projected
                .strata
                .iter()
                .map(|stratum| (stratum.id, stratum.sccs.clone()))
                .collect::<Vec<_>>(),
            vec![(0, vec![0]), (1, vec![1, 2]), (2, vec![3, 4])]
        );
        assert_eq!(
            projected.generated_query_rules,
            vec![GeneratedQueryRuleProvenance {
                query_index: 0,
                scc_index: 4,
                rule_index: 0,
            }]
        );
        assert_eq!(projected.est_memory_peak, 123);
        assert_eq!(projected.rel_arities, plan.rel_arities);
        assert_eq!(
            plan.rules_by_scc
                .iter()
                .flatten()
                .map(|rule| rule.head.clone())
                .collect::<Vec<_>>(),
            original_heads
        );

        let mut missing_unretained_schedule_entry = plan.clone();
        missing_unretained_schedule_entry.strata[1].sccs.clear();
        assert!(missing_unretained_schedule_entry
            .dependency_closed_subplan(&[4, 5], &defining_sccs)
            .is_none());

        plan.generated_query_rules[0].scc_index = 1;
        assert!(plan
            .dependency_closed_subplan(&[4, 5], &defining_sccs)
            .is_none());
    }

    #[test]
    fn dependency_closed_subplan_rejects_inconsistent_proof_without_pruning() {
        let plan = ExecutionPlan {
            sccs: vec![Scc {
                id: 0,
                predicates: vec!["output".into()],
                is_recursive: false,
            }],
            strata: vec![Stratum {
                id: 0,
                sccs: vec![0],
            }],
            rules_by_scc: vec![vec![compiled_rule(
                "output",
                RirNode::Scan { rel: RelId(1) },
            )]],
            generated_query_rules: vec![],
            est_memory_peak: 0,
            rel_arities: Default::default(),
        };

        assert!(plan
            .dependency_closed_subplan(&[], &Default::default())
            .is_none());
        assert!(plan
            .dependency_closed_subplan(&[1], &Default::default())
            .is_none());
        assert!(plan
            .dependency_closed_subplan(&[0], &[(RelId(1), 1)].into_iter().collect())
            .is_none());

        let mut invalid_scc_id = plan.clone();
        invalid_scc_id.sccs[0].id = 7;
        assert!(invalid_scc_id
            .dependency_closed_subplan(&[0], &Default::default())
            .is_none());

        let mut invalid_stratum = plan;
        invalid_stratum.strata[0].sccs[0] = 7;
        assert!(invalid_stratum
            .dependency_closed_subplan(&[0], &Default::default())
            .is_none());

        let mut invalid_stratum_id = invalid_scc_id;
        invalid_stratum_id.sccs[0].id = 0;
        invalid_stratum_id.strata[0].id = 7;
        assert!(invalid_stratum_id
            .dependency_closed_subplan(&[0], &Default::default())
            .is_none());
    }

    #[test]
    fn test_has_recursion() {
        let non_recursive = ExecutionPlan::new(vec![Scc {
            id: 0,
            predicates: vec!["p".into()],
            is_recursive: false,
        }]);
        assert!(!non_recursive.has_recursion());

        let recursive = ExecutionPlan::new(vec![Scc {
            id: 0,
            predicates: vec!["reach".into()],
            is_recursive: true,
        }]);
        assert!(recursive.has_recursion());
    }
}
