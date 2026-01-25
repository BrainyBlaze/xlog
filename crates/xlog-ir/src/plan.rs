//! Execution plan representation

use crate::metadata::RirMeta;
use crate::rir::RirNode;

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

/// Complete execution plan for a program
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// SCCs in dependency order
    pub sccs: Vec<Scc>,
    /// Strata for negation ordering
    pub strata: Vec<Stratum>,
    /// Compiled rules grouped by SCC
    pub rules_by_scc: Vec<Vec<CompiledRule>>,
    /// Total estimated memory peak (bytes)
    pub est_memory_peak: u64,
}

impl ExecutionPlan {
    /// Create a new execution plan from SCCs
    pub fn new(sccs: Vec<Scc>) -> Self {
        Self {
            sccs,
            strata: vec![],
            rules_by_scc: vec![],
            est_memory_peak: 0,
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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_scc(&mut self, scc: Scc) -> &mut Self {
        self.sccs.push(scc);
        self.rules.push(vec![]);
        self
    }

    pub fn add_rule(&mut self, scc_id: u32, rule: CompiledRule) -> &mut Self {
        if let Some(rules) = self.rules.get_mut(scc_id as usize) {
            rules.push(rule);
        }
        self
    }

    pub fn add_stratum(&mut self, stratum: Stratum) -> &mut Self {
        self.strata.push(stratum);
        self
    }

    pub fn build(self) -> ExecutionPlan {
        ExecutionPlan {
            sccs: self.sccs,
            strata: self.strata,
            rules_by_scc: self.rules,
            est_memory_peak: 0,
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
        let strata = vec![
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
