//! Joint constraint solver — skeleton: deterministic component
//! decomposition, pinned-width envelope, and typed fuel accounting.
//!
//! Decomposition, width bounding, and strategy selection are
//! cold-path setup over the constraint graph. Solve execution
//! (feasibility propagation, exact top-two/max-marginal dynamic
//! programming, branch-and-bound) is device-resident and lands with
//! the solve slice; nothing in this module emits solver outputs, so
//! no host path here can become a solving fallback.

/// Identity of the solver ABI and objective this module implements:
/// deterministic component decomposition, exact top-two/max-marginal
/// DP inside the pinned width envelope, exact branch-and-bound
/// within device fuel, typed exhaustion beyond it. Carrier schemas
/// and calibration artifacts bind to this identity; it changes
/// whenever the ABI or objective changes.
pub const SOLVER_ABI_IDENTITY: &str = "joint-solver/nary-device-components-dp-bb/5";

/// Typed solver errors. Beyond fuel the solve refuses with the
/// exact spent/limit literals — no partial emission, no
/// approximation, no host fallback.
#[derive(Debug, PartialEq, Eq)]
pub enum SolverError {
    /// The device fuel budget is exhausted.
    ResourceExhausted { fuel_spent: u64, fuel_limit: u64 },
}

impl std::fmt::Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverError::ResourceExhausted {
                fuel_spent,
                fuel_limit,
            } => write!(
                f,
                "solver fuel exhausted: spent {fuel_spent} of {fuel_limit} node expansions"
            ),
        }
    }
}

impl std::error::Error for SolverError {}

/// Saturating feasibility count for a component: none, exactly one,
/// or many satisfying assignments. Deliberately separate from score
/// ambiguity — a component can be uniquely feasible with an
/// ambiguous maximum, or plurally feasible with a unique maximum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Feasibility {
    None,
    One,
    Many,
}

/// Solve strategy for one component, selected by the width bound
/// against the pinned envelope: exact dynamic programming inside the
/// envelope, exact branch-and-bound (within fuel) outside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveStrategy {
    ExactDp,
    BranchAndBound,
}

/// One connected component of the constraint graph, in canonical
/// form: variables ascending, edges normalized (low, high) and
/// sorted, plus a deterministic elimination-order width bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Component {
    pub variables: Vec<u32>,
    pub edges: Vec<(u32, u32)>,
    pub width_bound: u32,
}

impl Component {
    /// Strategy under a pinned width envelope. The bound is an
    /// upper bound, so `ExactDp` selection is safe: true width can
    /// only be smaller.
    pub fn strategy(&self, pinned_width: u32) -> SolveStrategy {
        if self.width_bound <= pinned_width {
            SolveStrategy::ExactDp
        } else {
            SolveStrategy::BranchAndBound
        }
    }
}

/// Undirected constraint graph over entity variables. Self-loops
/// are meaningless for binary constraints and rejected at
/// construction; unconnected variables still form singleton
/// components so no variable can silently drop out of the solve.
pub struct ConstraintGraph {
    num_variables: u32,
    edges: Vec<(u32, u32)>,
}

impl ConstraintGraph {
    pub fn new(num_variables: u32, edges: impl IntoIterator<Item = (u32, u32)>) -> Self {
        let edges: Vec<(u32, u32)> = edges
            .into_iter()
            .map(|(a, b)| {
                assert!(
                    a < num_variables && b < num_variables,
                    "constraint edge ({a}, {b}) references a variable outside 0..{num_variables}"
                );
                assert!(a != b, "self-loop constraint edge on variable {a}");
                (a.min(b), a.max(b))
            })
            .collect();
        Self {
            num_variables,
            edges,
        }
    }

    /// Deterministic connected-component decomposition: union-find
    /// over the edges, components ordered by their minimum variable
    /// index, members ascending, edges normalized and sorted. The
    /// same graph decomposes identically regardless of edge input
    /// order.
    pub fn decompose(&self) -> Vec<Component> {
        let n = self.num_variables as usize;
        let mut parent: Vec<u32> = (0..self.num_variables).collect();

        fn find(parent: &mut [u32], x: u32) -> u32 {
            let mut root = x;
            while parent[root as usize] != root {
                root = parent[root as usize];
            }
            let mut cur = x;
            while parent[cur as usize] != root {
                let next = parent[cur as usize];
                parent[cur as usize] = root;
                cur = next;
            }
            root
        }

        for &(a, b) in &self.edges {
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            if ra != rb {
                // Deterministic union: smaller root wins, so every
                // component's representative is its minimum variable.
                let (lo, hi) = (ra.min(rb), ra.max(rb));
                parent[hi as usize] = lo;
            }
        }

        let mut members: Vec<Vec<u32>> = vec![Vec::new(); n];
        for v in 0..self.num_variables {
            let root = find(&mut parent, v);
            members[root as usize].push(v);
        }
        let mut component_edges: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n];
        for &(a, b) in &self.edges {
            let root = find(&mut parent, a);
            component_edges[root as usize].push((a, b));
        }

        (0..n)
            .filter(|&root| !members[root].is_empty())
            .map(|root| {
                let variables = members[root].clone();
                let mut edges = component_edges[root].clone();
                edges.sort_unstable();
                edges.dedup();
                let width_bound = elimination_width_bound(&variables, &edges);
                Component {
                    variables,
                    edges,
                    width_bound,
                }
            })
            .collect()
    }
}

/// Deterministic upper bound on the component's treewidth via
/// min-degree elimination (ties broken by variable index). An upper
/// bound is the safe direction for strategy selection: it can send
/// a narrow component to branch-and-bound, never a wide one to DP.
fn elimination_width_bound(variables: &[u32], edges: &[(u32, u32)]) -> u32 {
    use std::collections::{BTreeMap, BTreeSet};

    let mut adj: BTreeMap<u32, BTreeSet<u32>> =
        variables.iter().map(|&v| (v, BTreeSet::new())).collect();
    for &(a, b) in edges {
        adj.get_mut(&a).unwrap().insert(b);
        adj.get_mut(&b).unwrap().insert(a);
    }

    let mut width = 0u32;
    while !adj.is_empty() {
        // Min degree, then min index: fully deterministic.
        let (&v, _) = adj
            .iter()
            .min_by_key(|(idx, neigh)| (neigh.len(), **idx))
            .expect("non-empty adjacency");
        let neighbors: Vec<u32> = adj[&v].iter().copied().collect();
        width = width.max(neighbors.len() as u32);
        for &n in &neighbors {
            let set = adj.get_mut(&n).expect("neighbor present");
            set.remove(&v);
            for &m in &neighbors {
                if m != n {
                    set.insert(m);
                }
            }
        }
        adj.remove(&v);
    }
    width
}

/// Test-only oracle for device component discovery. Production
/// component membership is derived from carrier-owned arguments on
/// the device and never accepts this host CSR representation.
#[cfg(test)]
fn candidate_components(num_entities: u32, pairs: &[(u32, u32)]) -> (Vec<u32>, Vec<u32>) {
    let n = pairs.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();

    fn find(parent: &mut [u32], x: u32) -> u32 {
        let mut root = x;
        while parent[root as usize] != root {
            root = parent[root as usize];
        }
        let mut cur = x;
        while parent[cur as usize] != root {
            let next = parent[cur as usize];
            parent[cur as usize] = root;
            cur = next;
        }
        root
    }

    let mut entity_owner: Vec<Option<u32>> = vec![None; num_entities as usize];
    for (i, &(head, tail)) in pairs.iter().enumerate() {
        for entity in [head, tail] {
            assert!(
                entity < num_entities,
                "candidate {i} references entity {entity} outside 0..{num_entities}"
            );
            match entity_owner[entity as usize] {
                None => entity_owner[entity as usize] = Some(i as u32),
                Some(owner) => {
                    let (ra, rb) = (find(&mut parent, i as u32), find(&mut parent, owner));
                    if ra != rb {
                        let (lo, hi) = (ra.min(rb), ra.max(rb));
                        parent[hi as usize] = lo;
                    }
                }
            }
        }
    }

    let mut members: Vec<Vec<u32>> = vec![Vec::new(); n];
    for cand in 0..n as u32 {
        let root = find(&mut parent, cand);
        members[root as usize].push(cand);
    }
    let mut offsets = Vec::new();
    let mut indices = Vec::new();
    offsets.push(0u32);
    for group in members.into_iter().filter(|g| !g.is_empty()) {
        indices.extend_from_slice(&group);
        offsets.push(indices.len() as u32);
    }
    (offsets, indices)
}

/// Fuel accounting for node expansions. The production counter is
/// device-resident and read back once post-solve as bounded
/// metadata; this meter is the typed refusal seam both sides share.
/// Exhaustion saturates: once refused, every further charge refuses
/// with the same literals, so no caller can slip work past the
/// budget by retrying.
#[derive(Debug)]
pub struct FuelMeter {
    limit: u64,
    spent: u64,
}

impl FuelMeter {
    pub fn new(limit: u64) -> Self {
        Self { limit, spent: 0 }
    }

    pub fn spent(&self) -> u64 {
        self.spent
    }

    /// Unspent budget.
    pub fn remaining(&self) -> u64 {
        self.limit - self.spent
    }

    /// Refund expansions that a prior authorization charged but the
    /// device measurably did not spend. Callers refund at most
    /// `authorized - measured` for one completed solve; the meter
    /// saturates at zero rather than underflowing.
    pub fn refund(&mut self, expansions: u64) {
        self.spent = self.spent.saturating_sub(expansions);
    }

    /// Charge `expansions` node expansions. Refuses typed the
    /// moment the budget would be exceeded; the overflowing charge
    /// is not applied.
    pub fn charge(&mut self, expansions: u64) -> Result<(), SolverError> {
        let new_spent = self.spent.saturating_add(expansions);
        if new_spent > self.limit {
            return Err(SolverError::ResourceExhausted {
                fuel_spent: self.spent,
                fuel_limit: self.limit,
            });
        }
        self.spent = new_spent;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solver_abi_identity_names_nary_device_component_contract() {
        assert_eq!(
            SOLVER_ABI_IDENTITY,
            "joint-solver/nary-device-components-dp-bb/5"
        );
    }

    #[test]
    fn decomposition_is_deterministic_under_edge_order() {
        let edges = [(3, 1), (7, 5), (1, 0), (5, 6)];
        let mut reversed = edges;
        reversed.reverse();
        let a = ConstraintGraph::new(9, edges).decompose();
        let b = ConstraintGraph::new(9, reversed).decompose();
        assert_eq!(a, b, "edge input order must not change the decomposition");
    }

    #[test]
    fn every_variable_lands_in_exactly_one_component() {
        let graph = ConstraintGraph::new(6, [(0, 1), (4, 5)]);
        let components = graph.decompose();
        let mut seen: Vec<u32> = components
            .iter()
            .flat_map(|c| c.variables.iter().copied())
            .collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2, 3, 4, 5]);
        // Isolated variables 2 and 3 are singleton components, not
        // silently dropped from the solve.
        assert_eq!(components.len(), 4);
        assert!(components
            .iter()
            .any(|c| c.variables == vec![2] && c.edges.is_empty()));
    }

    #[test]
    fn components_are_canonical_and_ordered_by_minimum_variable() {
        let graph = ConstraintGraph::new(7, [(6, 4), (2, 0), (4, 5)]);
        let components = graph.decompose();
        let mins: Vec<u32> = components.iter().map(|c| c.variables[0]).collect();
        let mut sorted = mins.clone();
        sorted.sort_unstable();
        assert_eq!(mins, sorted, "components ordered by minimum variable");
        for c in &components {
            let mut vars = c.variables.clone();
            vars.sort_unstable();
            assert_eq!(vars, c.variables, "variables ascending");
            let mut edges = c.edges.clone();
            edges.sort_unstable();
            assert_eq!(edges, c.edges, "edges normalized and sorted");
            assert!(c.edges.iter().all(|(a, b)| a < b), "edges are (low, high)");
        }
    }

    #[test]
    fn width_bound_matches_known_graphs() {
        // Path 0-1-2-3: treewidth 1.
        let path = ConstraintGraph::new(4, [(0, 1), (1, 2), (2, 3)]).decompose();
        assert_eq!(path[0].width_bound, 1);
        // Star center 0: treewidth 1.
        let star = ConstraintGraph::new(5, [(0, 1), (0, 2), (0, 3), (0, 4)]).decompose();
        assert_eq!(star[0].width_bound, 1);
        // Complete graph K4: treewidth 3.
        let k4 =
            ConstraintGraph::new(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]).decompose();
        assert_eq!(k4[0].width_bound, 3);
    }

    #[test]
    fn strategy_splits_on_the_pinned_envelope() {
        let k4 =
            ConstraintGraph::new(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]).decompose();
        assert_eq!(k4[0].strategy(3), SolveStrategy::ExactDp);
        assert_eq!(k4[0].strategy(2), SolveStrategy::BranchAndBound);
    }

    #[test]
    fn candidate_components_join_on_shared_entities_deterministically() {
        // Candidates 0,1 share entity 1; candidate 2 is disjoint.
        let pairs = [(0, 1), (1, 2), (3, 4)];
        let (offsets, indices) = candidate_components(5, &pairs);
        assert_eq!(offsets, vec![0, 2, 3]);
        assert_eq!(indices, vec![0, 1, 2]);

        // Same graph with pairs listed in reverse candidate roles:
        // the grouping is identical because membership is by shared
        // entity, not by input order of the pair fields.
        let flipped = [(1, 0), (2, 1), (4, 3)];
        let (offsets_f, indices_f) = candidate_components(5, &flipped);
        assert_eq!((offsets_f, indices_f), (offsets, indices));

        // Every candidate lands exactly once.
        let (offsets, indices) = candidate_components(3, &[(0, 1), (1, 2), (0, 2)]);
        assert_eq!(offsets, vec![0, 3]);
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn fuel_refuses_typed_at_the_boundary_and_saturates() {
        let mut fuel = FuelMeter::new(10);
        fuel.charge(10).expect("exactly the budget is legal");
        let err = fuel.charge(1).expect_err("beyond fuel must refuse");
        assert_eq!(
            err,
            SolverError::ResourceExhausted {
                fuel_spent: 10,
                fuel_limit: 10
            }
        );
        // The refused charge was not applied, and refusal repeats
        // with identical literals — no retry can slip work through.
        assert_eq!(fuel.spent(), 10);
        assert_eq!(
            fuel.charge(1).expect_err("still refused"),
            SolverError::ResourceExhausted {
                fuel_spent: 10,
                fuel_limit: 10
            }
        );
    }
}
