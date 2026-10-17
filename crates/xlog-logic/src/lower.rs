//! Lowering from AST to IR
//!
//! This module transforms Datalog programs (AST) into the Relational IR (RIR)
//! representation for execution. The lowering process:
//!
//! 1. Infers schemas from facts and predicate declarations
//! 2. Tracks variable positions across atoms for join key computation
//! 3. Builds left-deep join trees for multi-atom rule bodies
//! 4. Handles negation via set difference (Diff) nodes
//! 5. Wraps recursive predicates in Fixpoint nodes
//! 6. Projects to match head variables

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use xlog_core::{symbol, AggOp as CoreAggOp, RelId, Result, ScalarType, Schema, XlogError};
use xlog_ir::{
    CompareOp, CompiledRule, ConstValue, ExecutionPlan, Expr, JoinType, PlanBuilder, ProjectExpr,
    RirMeta, RirNode, Scc, Stratum as IrStratum,
};

use crate::ast::{
    AggOp, ArithExpr, Atom, BodyLiteral, CompOp, Comparison, IsExpr, LearnableRule, Program, Rule,
    Term, TypeRef,
};
use crate::stratify::{build_dependency_graph, find_sccs_for_lowering, DepType};

struct JoinPlan<'a> {
    node: RirNode,
    leaf_order: Vec<&'a Atom>,
    leaf_order_idx: Vec<usize>,
    var_pos: HashMap<String, usize>,
    width: usize,
    est_rows: f64,
    total_cost: f64,
    allocation_receipt: Option<Arc<dyn Send + Sync>>,
}

struct LoweringAllocations<'a, 'b> {
    scope: Option<&'a mut crate::compile::CompilationAllocationScope<'b>>,
}

struct LoweringSymbolOutput {
    text: String,
    allocation_owner: Arc<dyn Send + Sync>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LowererAllocationKind {
    SchemaTable,
    SchemaKey,
    SchemaValue,
    RelationTable,
    RelationKey,
    Strata,
    CardinalityHints,
    EstimateTable,
    EstimateKey,
    ComponentTable,
    ComponentValue,
    ComponentIndexTable,
    ComponentIndexKey,
}

struct LowererAllocationRecord {
    kind: LowererAllocationKind,
    // String heap addresses identify existing stored keys without a second
    // allocating name map. They are compared only, never dereferenced; map
    // growth moves String headers but not their live UTF-8 allocations.
    identity: usize,
    receipt: Option<Arc<dyn Send + Sync>>,
    pending: bool,
}

#[derive(Default)]
struct LowererAllocationOwners {
    records: Vec<LowererAllocationRecord>,
    storage: Option<Arc<dyn Send + Sync>>,
}

struct PendingLowererAllocation {
    slot: usize,
    receipt: Arc<dyn Send + Sync>,
}

impl LowererAllocationOwners {
    fn reserve(
        &mut self,
        kind: LowererAllocationKind,
        identity: usize,
        bytes: u64,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<Option<PendingLowererAllocation>> {
        let Some(scope) = allocations.scope.as_deref_mut() else {
            return Ok(None);
        };
        let slot = self
            .records
            .iter()
            .position(|record| {
                record.kind == kind
                    && record.identity == identity
                    && record.receipt.is_some()
                    && !record.pending
            })
            .or_else(|| {
                self.records
                    .iter()
                    .position(|record| record.receipt.is_none() && !record.pending)
            });
        let slot = match slot {
            Some(slot) => slot,
            None => {
                if self.records.len() == self.records.capacity() {
                    let count = self.records.len().checked_add(1).ok_or_else(|| {
                        XlogError::allocation_extent_overflow("lowerer receipt storage")
                    })?;
                    let storage = scope.reserve_allocation(
                        crate::compile::push_vector_allocation_bytes::<LowererAllocationRecord>(
                            count,
                        )?,
                    )?;
                    self.records.reserve(1);
                    // The previous real vector buffer has now dropped.
                    self.storage = Some(storage);
                }
                self.records.push(LowererAllocationRecord {
                    kind,
                    identity,
                    receipt: None,
                    pending: false,
                });
                self.records.len() - 1
            }
        };
        let receipt = scope.reserve_allocation(bytes)?;
        if self.records[slot].receipt.is_none() {
            self.records[slot].kind = kind;
            self.records[slot].identity = identity;
        }
        self.records[slot].pending = true;
        Ok(Some(PendingLowererAllocation { slot, receipt }))
    }

    // Called only after the previous corresponding data has actually dropped.
    fn commit(&mut self, pending: Option<PendingLowererAllocation>, identity: usize) {
        if let Some(pending) = pending {
            self.records[pending.slot].identity = identity;
            self.records[pending.slot].receipt = Some(pending.receipt);
            self.records[pending.slot].pending = false;
        }
    }

    fn retire(&mut self, kind: LowererAllocationKind) {
        for record in &mut self.records {
            if record.kind == kind {
                record.receipt = None;
            }
        }
    }

    fn finish_attempt(&mut self) {
        // Pending guards belong to the completed stack frame, not this ledger.
        for record in &mut self.records {
            record.pending = false;
        }
    }
}

/// The original constant-data acquisition. Prepared storage borrows the live
/// leaf/atom receipt; empty data and ordinary unowned compilation need no receipt.
/// Both cases still acquire diagnostic text through the same lowering scope.
enum ConstantDataAllocation<'a> {
    Scope,
    Prepared(&'a Option<Arc<dyn Send + Sync>>),
}

fn group_proper_rules_with_owner<'p>(
    program: &'p Program,
    allocations: &mut LoweringAllocations<'_, '_>,
) -> Result<(
    HashMap<&'p str, Vec<&'p Rule>>,
    Option<Arc<dyn Send + Sync>>,
)> {
    match allocations.scope.as_deref_mut() {
        Some(scope) => {
            let mut stage = scope.child_scope();
            stage.prepare_output_owner()?;
            let rules = group_proper_rules_core(
                program,
                &mut LoweringAllocations {
                    scope: Some(&mut stage),
                },
            )?;
            let owner = stage.into_owner()?;
            // The lookup drops before the receipts that own its real buffers.
            Ok((rules, Some(owner)))
        }
        None => Ok((group_proper_rules_core(program, allocations)?, None)),
    }
}

fn group_proper_rules_core<'p>(
    program: &'p Program,
    allocations: &mut LoweringAllocations<'_, '_>,
) -> Result<HashMap<&'p str, Vec<&'p Rule>>> {
    let mut rules_by_pred: HashMap<&str, Vec<&Rule>> = HashMap::new();
    for rule in program.proper_rules() {
        let predicate = rule.head.predicate.as_str();
        if !rules_by_pred.contains_key(predicate) && rules_by_pred.len() == rules_by_pred.capacity()
        {
            allocations.hash::<&str, Vec<&Rule>>(rules_by_pred.len() + 1)?;
        }
        let (length, capacity) = rules_by_pred
            .get(predicate)
            .map(|rules| (rules.len(), rules.capacity()))
            .unwrap_or((0, 0));
        if length == capacity {
            allocations.vector::<&Rule>(length + 1)?;
        }
        // The authored AST remains live for the whole lowering operation.
        rules_by_pred.entry(predicate).or_default().push(rule);
    }
    Ok(rules_by_pred)
}

fn arithmetic_expression_nodes(expression: &ArithExpr) -> Result<usize> {
    let overflow = || XlogError::allocation_extent_overflow("arithmetic expression");
    let children = match expression {
        ArithExpr::Variable(_) | ArithExpr::Integer(_) | ArithExpr::Float(_) => 0,
        ArithExpr::Add(a, b)
        | ArithExpr::Sub(a, b)
        | ArithExpr::Mul(a, b)
        | ArithExpr::Div(a, b)
        | ArithExpr::Mod(a, b)
        | ArithExpr::Min(a, b)
        | ArithExpr::Max(a, b)
        | ArithExpr::Pow(a, b) => arithmetic_expression_nodes(a)?
            .checked_add(arithmetic_expression_nodes(b)?)
            .ok_or_else(overflow)?,
        ArithExpr::Abs(inner) | ArithExpr::Cast(inner, _) => arithmetic_expression_nodes(inner)?,
        ArithExpr::FuncCall { args, .. } => {
            let mut total = 0usize;
            for argument in args {
                total = total
                    .checked_add(arithmetic_expression_nodes(argument)?)
                    .ok_or_else(overflow)?;
            }
            total
        }
        ArithExpr::Conditional {
            cond_left,
            cond_right,
            then_expr,
            else_expr,
            ..
        } => arithmetic_expression_nodes(cond_left)?
            .checked_add(arithmetic_expression_nodes(cond_right)?)
            .and_then(|n| n.checked_add(arithmetic_expression_nodes(then_expr).ok()?))
            .and_then(|n| n.checked_add(arithmetic_expression_nodes(else_expr).ok()?))
            .ok_or_else(overflow)?,
    };
    children.checked_add(1).ok_or_else(overflow)
}

impl LoweringAllocations<'_, '_> {
    fn arithmetic_type(&mut self, expression: &ArithExpr) -> Result<()> {
        if self.scope.is_none() {
            return Ok(());
        }
        let nodes = arithmetic_expression_nodes(expression)?;
        self.vector::<ArithmeticTypeTask<'_>>(
            nodes
                .checked_mul(2)
                .ok_or_else(|| XlogError::allocation_extent_overflow("arithmetic type tasks"))?,
        )?;
        self.vector::<Option<ScalarType>>(nodes)?;
        Ok(())
    }

    fn acquire(&mut self, bytes: u64) -> Result<()> {
        self.acquire_receipt(bytes).map(|_| ())
    }

    fn acquire_receipt(&mut self, bytes: u64) -> Result<Option<Arc<dyn Send + Sync>>> {
        if bytes != 0 {
            if let Some(scope) = self.scope.as_deref_mut() {
                return scope.acquire(bytes).map(Some);
            }
        }
        Ok(None)
    }

    fn vector<T>(&mut self, count: usize) -> Result<()> {
        if self.scope.is_some() {
            self.acquire(crate::compile::push_vector_allocation_bytes::<T>(count)?)?;
        }
        Ok(())
    }

    fn hash<K, V>(&mut self, count: usize) -> Result<()> {
        if self.scope.is_some() {
            self.acquire(crate::compile::hash_allocation_bytes::<K, V>(count)?)?;
        }
        Ok(())
    }

    fn formatted(&mut self, arguments: std::fmt::Arguments<'_>) -> Result<()> {
        if self.scope.is_some() {
            self.acquire(crate::compile::formatted_text_allocation_bytes(arguments)?)?;
        }
        Ok(())
    }

    fn compilation_error(&mut self, arguments: std::fmt::Arguments<'_>) -> XlogError {
        match self.prepare_diagnostic(crate::compile::formatted_text_allocation_bytes(arguments)) {
            Ok(producer) => producer.finish(|| XlogError::Compilation(std::fmt::format(arguments))),
            Err(error) => error,
        }
    }

    fn unsafe_variable(&mut self, name: &str) -> XlogError {
        match self.prepare_diagnostic(Ok(name.len() as u64)) {
            Ok(producer) => producer.finish(|| XlogError::UnsafeVariable(name.to_owned())),
            Err(error) => error,
        }
    }

    fn prepare_diagnostic(
        &mut self,
        bytes: Result<u64>,
    ) -> Result<xlog_core::compilation_allocation::CompilationDiagnosticProducer> {
        match self.scope.as_deref_mut() {
            Some(scope) => crate::compile::CompilationAllocationScope::prepare_diagnostic(
                bytes,
                Some(&mut |bytes| scope.reserve_allocation(bytes)),
            ),
            None => crate::compile::CompilationAllocationScope::prepare_diagnostic(bytes, None),
        }
    }

    fn symbol(&mut self, id: u32) -> Result<String> {
        match self.scope.as_deref_mut() {
            Some(scope) => {
                let (text, receipt) = symbol::resolve_with_allocation_owner(id, &mut |bytes| {
                    scope.reserve_allocation(
                        u64::try_from(bytes)
                            .map_err(|_| XlogError::allocation_extent_overflow("symbol storage"))?,
                    )
                })?;
                let output = LoweringSymbolOutput {
                    text,
                    allocation_owner: receipt,
                };
                scope.retain(Arc::clone(&output.allocation_owner))?;
                Ok(output.text)
            }
            None => Ok(symbol::resolve(id)),
        }
    }

    fn schema(&mut self, schema: &Schema) -> Result<()> {
        if self.scope.is_some() {
            self.acquire(schema.owned_allocation_bytes()?)?;
        }
        Ok(())
    }

    fn generated_schema(&mut self, arity: usize) -> Result<()> {
        self.vector::<(String, ScalarType)>(arity)?;
        self.vector::<usize>(arity)?;
        self.vector::<String>(arity)?;
        for index in 0..arity {
            // Column text and the original Schema::new default sort label.
            self.formatted(format_args!("c{index}"))?;
            self.formatted(format_args!("c{index}"))?;
        }
        Ok(())
    }

    fn sort_labels(&mut self, terms: &[Term]) -> Result<()> {
        self.vector::<String>(terms.len())?;
        // Schema::with_sort_labels consumes the original label vector and
        // builds its canonical normalized vector.
        self.vector::<String>(terms.len())?;
        for (index, term) in terms.iter().enumerate() {
            match term {
                Term::Variable(name) if !name.trim().is_empty() => {
                    self.acquire(name.len() as u64)?
                }
                Term::Aggregate(aggregate) => {
                    self.formatted(format_args!("{:?}_{}", aggregate.op, aggregate.variable))?
                }
                Term::List(_) => self.formatted(format_args!("list{index}"))?,
                Term::Cons { .. } => self.formatted(format_args!("cons{index}"))?,
                Term::Compound { functor, .. } => self.acquire(functor.len() as u64)?,
                Term::PredRef(name) => self.acquire(name.len() as u64)?,
                _ => self.formatted(format_args!("c{index}"))?,
            }
        }
        Ok(())
    }

    fn temporary(&mut self, bytes: u64) -> Result<Option<Arc<dyn Send + Sync>>> {
        self.scope
            .as_deref_mut()
            .map(|scope| scope.reserve_allocation(bytes))
            .transpose()
    }

    fn finish<'a>(&mut self, plan: JoinPlan<'a>) -> Result<(RirNode, Vec<&'a Atom>)> {
        if let Some(receipt) = plan.allocation_receipt.as_ref() {
            self.scope
                .as_deref_mut()
                .expect("original owned join scope")
                .retain(Arc::clone(receipt))?;
        }
        Ok((plan.node, plan.leaf_order))
    }

    fn join_candidate<'a>(
        &mut self,
        left: &JoinPlan<'a>,
        right: &JoinPlan<'a>,
    ) -> Result<Option<Arc<dyn Send + Sync>>> {
        if self.scope.is_none() {
            return Ok(None);
        }
        let overflow = || XlogError::allocation_extent_overflow("join candidate");
        let count = left
            .leaf_order
            .len()
            .checked_add(right.leaf_order.len())
            .ok_or_else(overflow)?;
        let variables = left
            .var_pos
            .len()
            .checked_add(right.var_pos.len())
            .ok_or_else(overflow)?;
        let width = left.width.checked_add(right.width).ok_or_else(overflow)?;
        let mut bytes = left
            .node
            .owned_allocation_bytes()?
            .checked_add(right.node.owned_allocation_bytes()?)
            .ok_or_else(overflow)?;
        bytes = bytes
            .checked_add(4 * std::mem::size_of::<RirNode>() as u64)
            .and_then(|n| {
                n.checked_add(
                    crate::compile::push_vector_allocation_bytes::<ProjectExpr>(
                        width.checked_add(2)?,
                    )
                    .ok()?
                    .checked_mul(3)?,
                )
            })
            .and_then(|n| {
                n.checked_add(
                    crate::compile::push_vector_allocation_bytes::<usize>(variables)
                        .ok()?
                        .checked_mul(4)?,
                )
            })
            .and_then(|n| {
                n.checked_add(
                    crate::compile::push_vector_allocation_bytes::<&String>(variables).ok()?,
                )
            })
            .and_then(|n| {
                n.checked_add(crate::compile::push_vector_allocation_bytes::<&Atom>(count).ok()?)
            })
            .and_then(|n| {
                n.checked_add(crate::compile::push_vector_allocation_bytes::<usize>(count).ok()?)
            })
            .and_then(|n| {
                n.checked_add(
                    crate::compile::hash_allocation_bytes::<String, usize>(variables)
                        .ok()?
                        .checked_mul(2)?,
                )
            })
            .ok_or_else(overflow)?;
        for name in left.var_pos.keys().chain(right.var_pos.keys()) {
            bytes = bytes.checked_add(name.len() as u64).ok_or_else(overflow)?;
        }
        // Both orientations are charged independently in the original loop.
        // This cell covers this candidate's original subtree clones, key and
        // Cartesian projection buffers, leaf roster and variable table only.
        self.temporary(bytes)
    }
}

pub(crate) fn positive_task_lowering_allocation_bound(
    program: &Program,
    max_name: usize,
) -> Result<usize> {
    use std::mem::size_of;
    let overflow = || XlogError::allocation_extent_overflow("positive task lowering");
    let add = |a: usize, b: usize| a.checked_add(b).ok_or_else(overflow);
    let mul = |a: usize, b: usize| a.checked_mul(b).ok_or_else(overflow);
    let relations = add(program.predicates.len(), program.queries.len())?;
    let vector = |count: usize, width: usize| mul(add(mul(count, 2)?, 4)?, width);
    let table = |count: usize, width: usize| -> Result<usize> {
        if count == 0 {
            return Ok(0);
        }
        let buckets = add(mul(count, 8)?, 6)?
            .checked_div(7)
            .and_then(|n| n.max(8).checked_next_power_of_two())
            .ok_or_else(overflow)?;
        add(mul(buckets, add(width, 1)?)?, 16)
    };
    let mut plans = 0usize;
    let mut dp = 0usize;
    // Ground queries become one-body rules with a one-column constant head in
    // the original frontend. Facts contribute schemas but no executable tree.
    for (head, body, query) in program
        .rules
        .iter()
        .filter(|rule| !rule.is_fact())
        .map(|rule| (rule.head.terms.len(), rule.body.as_slice(), None))
        .chain(
            program
                .queries
                .iter()
                .map(|query| (1, &[][..], Some(&query.atom))),
        )
    {
        let count = if query.is_some() { 1 } else { body.len() };
        if !(1..=3).contains(&count) {
            return Err(XlogError::Compilation(
                "positive task lowering capacity requires one to three body atoms".into(),
            ));
        }
        let mut arities = [0usize; 3];
        let mut filters = [0usize; 3];
        let mut variable_counts = [0usize; 3];
        if let Some(query) = query {
            arities[0] = query.terms.len();
            filters[0] = arities[0];
        } else {
            for (index, literal) in body.iter().enumerate() {
                let BodyLiteral::Positive(atom) = literal else {
                    return Err(XlogError::Compilation(
                        "positive task lowering capacity requires positive body atoms".into(),
                    ));
                };
                arities[index] = atom.terms.len();
                for (position, term) in atom.terms.iter().enumerate() {
                    match term {
                        Term::Integer(_) => filters[index] = add(filters[index], 1)?,
                        Term::Variable(name) => {
                            variable_counts[index] = add(variable_counts[index], 1)?;
                            if atom.terms[..position].iter().any(|previous| matches!(previous, Term::Variable(previous) if previous == name)) {
                                filters[index] = add(filters[index], 1)?;
                            }
                        }
                        Term::Anonymous => {}
                        _ => {
                            return Err(XlogError::Compilation(
                                "positive task lowering capacity requires scalar body terms".into(),
                            ))
                        }
                    }
                }
            }
        }
        if arities[..count]
            .iter()
            .any(|arity| !(1..=2).contains(arity))
        {
            return Err(XlogError::Compilation(
                "positive task lowering capacity requires unary or binary atoms".into(),
            ));
        }
        let tree = |mask: usize| -> Result<usize> {
            let mut atoms = 0usize;
            let mut columns = 0usize;
            let mut comparisons = 0usize;
            let mut filtered_atoms = 0usize;
            for index in 0..count {
                if mask & (1 << index) != 0 {
                    atoms = add(atoms, 1)?;
                    columns = add(columns, arities[index])?;
                    comparisons = add(comparisons, filters[index])?;
                    filtered_atoms = add(filtered_atoms, usize::from(filters[index] != 0))?;
                }
            }
            let joins = atoms - 1;
            // Each Cartesian join owns two dummy-input projections, its join
            // and its output projection. Equijoins fit in the same geometry.
            let nodes = add(add(atoms, filtered_atoms)?, add(mul(joins, 4)?, 1)?)?;
            let width = add(columns, 2)?;
            let projections = vector(width.max(head), size_of::<ProjectExpr>())?;
            let keys = vector(columns, size_of::<usize>())?;
            let expressions = mul(
                add(mul(comparisons, 4)?, mul(joins, 2)?)?,
                size_of::<Expr>(),
            )?;
            add(
                mul(
                    nodes,
                    add(size_of::<RirNode>(), add(projections, mul(keys, 2)?)?)?,
                )?,
                expressions,
            )
        };
        let full_mask = (1usize << count) - 1;
        let full_tree = tree(full_mask)?;
        let columns = arities[..count]
            .iter()
            .try_fold(0usize, |n, arity| add(n, *arity))?;
        let variables = variable_counts[..count]
            .iter()
            .try_fold(0usize, |n, count| add(n, *count))?;
        let width = add(columns, 2)?;
        let schema = add(
            size_of::<Schema>(),
            add(
                vector(width, size_of::<(String, ScalarType)>())?,
                add(
                    vector(width, size_of::<String>())?,
                    add(
                        mul(mul(width, 2)?, max_name)?,
                        vector(width, size_of::<usize>())?,
                    )?,
                )?,
            )?,
        )?;
        let mut scratch = vector(1usize << count, size_of::<Option<JoinPlan<'_>>>())?;
        // Every nonempty subset has its own best tree, ordered leaf rosters
        // and variable-position table. Two oriented candidates can coexist
        // with these cells during replacement in the original DP loop.
        for mask in 1..=full_mask {
            let leaves = mask.count_ones() as usize;
            scratch = add(scratch, tree(mask)?)?;
            scratch = add(scratch, vector(leaves, size_of::<&Atom>())?)?;
            scratch = add(scratch, vector(leaves, size_of::<usize>())?)?;
            scratch = add(
                scratch,
                table(variables, add(size_of::<(String, usize)>(), max_name)?)?,
            )?;
        }
        scratch = add(scratch, mul(full_tree, 2)?)?;
        scratch = add(scratch, mul(vector(variables, size_of::<usize>())?, 4)?)?;
        scratch = add(scratch, vector(variables, size_of::<&String>())?)?;
        scratch = add(
            scratch,
            mul(
                table(variables, add(size_of::<(String, usize)>(), max_name)?)?,
                2,
            )?,
        )?;
        // Promotion retains the original fallback and adds owned input nodes,
        // coordinate mappings, lookup permutations and projection buffers.
        let promoted_tree = if count > 1 {
            [
                size_of::<RirNode>(),
                vector(count, size_of::<RirNode>())?,
                vector(count, size_of::<Vec<Option<u32>>>())?,
                mul(count, vector(columns, size_of::<Option<u32>>())?)?,
                vector(count, size_of::<xlog_ir::rir::LookupPerm>())?,
                vector(width.max(head), size_of::<ProjectExpr>())?,
                table(columns, size_of::<(usize, u32)>())?,
                vector(columns, size_of::<u32>())?,
                vector(count, size_of::<RelId>())?,
                vector(count, size_of::<usize>())?,
                vector(columns, size_of::<usize>())?,
                vector(columns, size_of::<(usize, usize)>())?,
            ]
            .into_iter()
            .try_fold(full_tree, add)?
        } else {
            full_tree
        };
        let compiled = add(
            promoted_tree,
            add(
                size_of::<CompiledRule>(),
                add(max_name, add(size_of::<RirMeta>(), schema)?)?,
            )?,
        )?;
        // The pipeline has distinct lowered, helper-split, optimized,
        // selectivity, promotion, clique-spec, provenance and returned owners.
        // Include the promoted shape in each owner so subsequent whole-plan
        // clones keep the same fallback and mapping buffers in this envelope.
        plans = add(plans, mul(compiled, 8)?)?;
        dp = dp.max(scratch);
    }
    let topology = add(
        mul(
            relations,
            add(size_of::<Scc>(), add(max_name, size_of::<String>())?)?,
        )?,
        add(
            mul(relations, size_of::<IrStratum>())?,
            mul(relations, size_of::<u32>())?,
        )?,
    )?;
    add(
        plans,
        add(dp, add(mul(topology, 4)?, size_of::<PlanBuilder>())?)?,
    )
}

#[derive(Clone, Copy)]
enum UserFunctionTypeEvidence {
    RequireExpansion,
    Defer,
}

#[derive(Clone, Copy)]
enum ArithmeticTypeOperation {
    Standard,
    Modulo,
    MinMax,
    Power,
}

enum ArithmeticTypeTask<'a> {
    Visit(&'a ArithExpr),
    FinishBinary(ArithmeticTypeOperation),
    FinishAbs,
    FinishCast(ScalarType),
    FinishFunctionArguments(usize),
    FinishConditional,
}

#[derive(Clone, Copy)]
enum ArithmeticExpressionOperation {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Min,
    Max,
    Pow,
}

enum ArithmeticExpressionTask<'a> {
    Visit(&'a ArithExpr),
    FinishBinary(ArithmeticExpressionOperation),
    FinishAbs,
    FinishCast(ScalarType),
    FinishConditional(CompOp),
}

fn resolve_pred_column_type(
    predicate: &str,
    index: usize,
    typ: &TypeRef,
    domains: &HashMap<String, ScalarType>,
    allocations: &mut LoweringAllocations<'_, '_>,
) -> Result<ScalarType> {
    match typ {
        TypeRef::Scalar(ty) => Ok(*ty),
        TypeRef::Domain(name) => domains.get(name).copied().ok_or_else(|| {
            allocations.compilation_error(format_args!(
                "v0.8.5 unknown domain alias '{}' in predicate '{}' column {}",
                name, predicate, index
            ))
        }),
        TypeRef::List(_) | TypeRef::Term | TypeRef::Compound | TypeRef::PredRef => {
            Ok(ScalarType::U64)
        }
    }
}

fn validate_lowerable_terms(
    program: &Program,
    allocations: &mut LoweringAllocations<'_, '_>,
) -> Result<()> {
    for rule in &program.rules {
        validate_atom_terms(&rule.head, "rule head", allocations)?;
        for lit in &rule.body {
            match lit {
                BodyLiteral::Positive(atom) => {
                    validate_atom_terms(atom, "positive body atom", allocations)?
                }
                BodyLiteral::Negated(atom) => {
                    validate_atom_terms(atom, "negated body atom", allocations)?
                }
                BodyLiteral::Epistemic(_) => {}
                BodyLiteral::Comparison(cmp) => {
                    validate_term_lowerable(&cmp.left, "comparison left operand", allocations)?;
                    validate_term_lowerable(&cmp.right, "comparison right operand", allocations)?;
                }
                BodyLiteral::IsExpr(_) => {}
                BodyLiteral::Univ(_) => {
                    return Err(allocations.compilation_error(format_args!(
                        "v0.8.5 meta error: univ literal was not normalized before lowering"
                    )));
                }
            }
        }
    }
    for constraint in &program.constraints {
        for lit in &constraint.body {
            match lit {
                BodyLiteral::Positive(atom) => {
                    validate_atom_terms(atom, "constraint body atom", allocations)?
                }
                BodyLiteral::Negated(atom) => {
                    validate_atom_terms(atom, "constraint negated body atom", allocations)?
                }
                BodyLiteral::Epistemic(_) => {}
                BodyLiteral::Comparison(cmp) => {
                    validate_term_lowerable(
                        &cmp.left,
                        "constraint comparison left operand",
                        allocations,
                    )?;
                    validate_term_lowerable(
                        &cmp.right,
                        "constraint comparison right operand",
                        allocations,
                    )?;
                }
                BodyLiteral::IsExpr(_) => {}
                BodyLiteral::Univ(_) => {
                    return Err(allocations.compilation_error(format_args!(
                        "v0.8.5 meta error: univ literal was not normalized before lowering"
                    )));
                }
            }
        }
    }
    for query in &program.queries {
        validate_atom_terms(&query.atom, "query atom", allocations)?;
    }
    for pf in &program.prob_facts {
        validate_atom_terms(&pf.atom, "probabilistic fact", allocations)?;
    }
    for ad in &program.annotated_disjunctions {
        for choice in &ad.choices {
            validate_atom_terms(&choice.atom, "annotated disjunction choice", allocations)?;
        }
    }
    for evidence in &program.evidence {
        validate_atom_terms(&evidence.atom, "evidence atom", allocations)?;
    }
    for query in &program.prob_queries {
        validate_atom_terms(&query.atom, "probabilistic query", allocations)?;
    }
    for neural in &program.neural_predicates {
        validate_atom_terms(&neural.predicate, "neural predicate", allocations)?;
    }
    for learnable in &program.learnable_rules {
        validate_atom_terms(&learnable.head, "learnable rule head", allocations)?;
        for lit in &learnable.body {
            if let BodyLiteral::Positive(atom) = lit {
                validate_atom_terms(atom, "learnable rule body", allocations)?;
            }
        }
    }
    Ok(())
}

/// Variables bound by the positive relational portion of a rule body.
///
/// The lowerer joins every positive atom before evaluating deterministic
/// arithmetic bindings, regardless of conjunct order.
pub(crate) fn positive_body_bound_variables(body: &[BodyLiteral]) -> BTreeSet<String> {
    body.iter()
        .filter_map(|literal| match literal {
            BodyLiteral::Positive(atom) => Some(atom),
            _ => None,
        })
        .flat_map(Atom::variables)
        .filter(|name| *name != "_")
        .map(str::to_string)
        .collect()
}

pub(crate) fn positive_body_bound_variables_with_allocation_owner(
    body: &[BodyLiteral],
    scope: &mut crate::compile::CompilationAllocationScope<'_>,
) -> Result<BTreeSet<String>> {
    let overflow = || XlogError::allocation_extent_overflow("bound variable set");
    let mut count = 0usize;
    let mut bytes = 0u64;
    for literal in body {
        let BodyLiteral::Positive(atom) = literal else {
            continue;
        };
        let mut atom_count = 0usize;
        let mut invalid = false;
        atom.visit_variables(&mut |name| {
            match atom_count.checked_add(1) {
                Some(value) => atom_count = value,
                None => invalid = true,
            }
            if name != "_" {
                match count.checked_add(1) {
                    Some(value) => count = value,
                    None => invalid = true,
                }
                match bytes.checked_add(name.len() as u64) {
                    Some(value) => bytes = value,
                    None => invalid = true,
                }
            }
        });
        if invalid {
            return Err(overflow());
        }
        bytes = bytes
            .checked_add(crate::compile::push_vector_allocation_bytes::<&str>(
                atom_count,
            )?)
            .ok_or_else(overflow)?;
    }
    // The canonical tree collect uses a sorted temporary string vector before
    // constructing its tree nodes. Duplicate names still create temporary strings.
    bytes = bytes
        .checked_add(crate::compile::push_vector_allocation_bytes::<String>(
            count,
        )?)
        .ok_or_else(overflow)?;
    for index in 0..count {
        bytes = bytes
            .checked_add(crate::compile::btree_insert_allocation_bytes::<String, ()>(
                index,
            )?)
            .ok_or_else(overflow)?;
    }
    scope.acquire(bytes)?;
    Ok(positive_body_bound_variables(body))
}

fn validate_is_expression_bindings(
    program: &Program,
    allocations: &mut LoweringAllocations<'_, '_>,
) -> Result<()> {
    for rule in program.proper_rules() {
        let mut bound = match allocations.scope.as_deref_mut() {
            Some(scope) => positive_body_bound_variables_with_allocation_owner(&rule.body, scope)?,
            None => positive_body_bound_variables(&rule.body),
        };
        for literal in &rule.body {
            let BodyLiteral::IsExpr(is_expr) = literal else {
                continue;
            };
            if bound.contains(&is_expr.target) {
                return Err(allocations.compilation_error(format_args!(
                    "Variable {} already bound; 'is' requires fresh variable",
                    is_expr.target
                )));
            }
            if allocations.scope.is_some() {
                let mut count = 0usize;
                let mut invalid = false;
                is_expr
                    .expr
                    .visit_variables(&mut |_| match count.checked_add(1) {
                        Some(value) => count = value,
                        None => invalid = true,
                    });
                if invalid {
                    return Err(XlogError::allocation_extent_overflow(
                        "arithmetic variables",
                    ));
                }
                allocations.vector::<&str>(count)?;
            }
            for variable in is_expr.expr.variables() {
                if !bound.contains(variable) {
                    return Err(allocations.compilation_error(format_args!(
                        "Variable {} used in arithmetic but not bound",
                        variable
                    )));
                }
            }
            if allocations.scope.is_some() {
                allocations.acquire(is_expr.target.len() as u64)?;
                allocations.acquire(
                    crate::compile::btree_insert_allocation_bytes::<String, ()>(bound.len())?,
                )?;
            }
            bound.insert(is_expr.target.clone());
        }
    }
    Ok(())
}

fn validate_atom_terms(
    atom: &Atom,
    context: &str,
    allocations: &mut LoweringAllocations<'_, '_>,
) -> Result<()> {
    for term in &atom.terms {
        validate_term_lowerable(term, context, allocations)?;
    }
    Ok(())
}

fn validate_term_lowerable(
    term: &Term,
    context: &str,
    allocations: &mut LoweringAllocations<'_, '_>,
) -> Result<()> {
    match term {
        Term::List(_) => Err(term_not_lowerable_error(context, "list", allocations)),
        Term::Cons { .. } => Err(term_not_lowerable_error(context, "cons", allocations)),
        Term::Compound { .. } => Err(term_not_lowerable_error(context, "compound", allocations)),
        Term::PredRef(_) => Err(term_not_lowerable_error(context, "predref", allocations)),
        Term::Variable(_)
        | Term::Anonymous
        | Term::Integer(_)
        | Term::Float(_)
        | Term::String(_)
        | Term::Symbol(_)
        | Term::Aggregate(_) => Ok(()),
    }
}

fn term_not_lowerable_error(
    context: &str,
    kind: &str,
    allocations: &mut LoweringAllocations<'_, '_>,
) -> XlogError {
    allocations.compilation_error(format_args!(
        "term form '{}' in {} is parsed but not lowerable by this execution path",
        kind, context
    ))
}

fn term_kind_for_lowering_error(term: &Term) -> &'static str {
    match term {
        Term::List(_) => "list",
        Term::Cons { .. } => "cons",
        Term::Compound { .. } => "compound",
        Term::PredRef(_) => "predref",
        Term::Variable(_)
        | Term::Anonymous
        | Term::Integer(_)
        | Term::Float(_)
        | Term::String(_)
        | Term::Symbol(_)
        | Term::Aggregate(_) => "term",
    }
}

/// Lowerer transforms AST programs into RIR execution plans.
pub struct Lowerer {
    /// Inferred or declared schemas for each predicate
    schemas: HashMap<String, Schema>,
    /// Stratification result (predicates grouped by strata)
    strata: Vec<Vec<String>>,
    /// Estimated cardinality per predicate (for join ordering)
    est_cardinality: HashMap<String, u64>,
    /// Optional cardinality hints per predicate (e.g., from runtime statistics).
    cardinality_hints: HashMap<String, u64>,
    /// Next available relation ID
    next_rel_id: u32,
    /// Mapping from predicate names to relation IDs
    rel_ids: HashMap<String, RelId>,
    /// SCCs for the program (from stratification)
    sccs: Vec<Scc>,
    /// Maximum active rules for TensorMaskedJoin (default 32)
    max_active_rules: usize,
    /// Predicate -> SCC id, rebuilt with `sccs` in `build_sccs`. A predicate
    /// belongs to exactly one SCC, so lookups replace the linear scan that
    /// made `find_scc_for_predicate` O(rules x SCCs).
    scc_by_pred: std::collections::HashMap<String, u32>,
    // Persistent state drops before its genuine original allocation receipts.
    allocation_owners: LowererAllocationOwners,
}

impl Default for Lowerer {
    fn default() -> Self {
        Self::new()
    }
}

impl Lowerer {
    /// Create a new lowerer instance
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
            strata: Vec::new(),
            est_cardinality: HashMap::new(),
            cardinality_hints: HashMap::new(),
            next_rel_id: 0,
            rel_ids: HashMap::new(),
            sccs: Vec::new(),
            max_active_rules: 32,
            scc_by_pred: std::collections::HashMap::new(),
            allocation_owners: LowererAllocationOwners::default(),
        }
    }

    /// Set the maximum active rules for TensorMaskedJoin.
    pub fn set_max_active_rules(&mut self, max: usize) {
        self.max_active_rules = max;
    }

    /// Set the stratification result for ordering
    pub(crate) fn set_strata(&mut self, strata: Vec<Vec<String>>) {
        self.strata = strata;
        self.allocation_owners.retire(LowererAllocationKind::Strata);
    }

    /// Set cardinality hints (typically sourced from runtime statistics snapshots).
    ///
    /// These hints are used by lowering-time join ordering when available.
    pub(crate) fn set_cardinality_hints(&mut self, hints: HashMap<String, u64>) {
        self.cardinality_hints = hints;
        self.allocation_owners
            .retire(LowererAllocationKind::CardinalityHints);
    }

    pub(crate) fn set_strata_with_allocation_owner(
        &mut self,
        strata: Vec<Vec<String>>,
        scope: &mut crate::compile::CompilationAllocationScope<'_>,
    ) -> Result<()> {
        self.allocation_owners.finish_attempt();
        let mut bytes = crate::compile::push_vector_allocation_bytes::<Vec<String>>(strata.len())?;
        for predicates in &strata {
            bytes = bytes
                .checked_add(crate::compile::push_vector_allocation_bytes::<String>(
                    predicates.len(),
                )?)
                .ok_or_else(|| XlogError::allocation_extent_overflow("lowerer strata"))?;
            for predicate in predicates {
                bytes = bytes
                    .checked_add(predicate.len() as u64)
                    .ok_or_else(|| XlogError::allocation_extent_overflow("lowerer strata"))?;
            }
        }
        let pending = self.allocation_owners.reserve(
            LowererAllocationKind::Strata,
            0,
            bytes,
            &mut LoweringAllocations { scope: Some(scope) },
        )?;
        self.strata = strata.clone();
        self.allocation_owners.commit(pending, 0);
        Ok(())
    }

    pub(crate) fn set_cardinality_hints_with_allocation_owner(
        &mut self,
        hints: HashMap<String, u64>,
        scope: &mut crate::compile::CompilationAllocationScope<'_>,
    ) -> Result<()> {
        self.allocation_owners.finish_attempt();
        let mut bytes = crate::compile::hash_allocation_bytes::<String, u64>(hints.len())?;
        for name in hints.keys() {
            bytes = bytes.checked_add(name.len() as u64).ok_or_else(|| {
                XlogError::allocation_extent_overflow("lowerer cardinality hints")
            })?;
        }
        let pending = self.allocation_owners.reserve(
            LowererAllocationKind::CardinalityHints,
            0,
            bytes,
            &mut LoweringAllocations { scope: Some(scope) },
        )?;
        // Copy live entries with the original hasher, not retired bucket capacity.
        let mut retained = HashMap::with_capacity_and_hasher(hints.len(), hints.hasher().clone());
        for (name, hint) in &hints {
            retained.insert(name.clone(), *hint);
        }
        self.cardinality_hints = retained;
        self.allocation_owners.commit(pending, 0);
        Ok(())
    }

    fn insert_schema_with_owner(
        &mut self,
        name: &str,
        schema: Schema,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<()> {
        if allocations.scope.is_none() {
            if let Some((key, _)) = self.schemas.get_key_value(name) {
                let identity = key.as_ptr() as usize;
                self.schemas.insert(name.to_owned(), schema);
                for record in &mut self.allocation_owners.records {
                    if record.kind == LowererAllocationKind::SchemaValue
                        && record.identity == identity
                    {
                        record.receipt = None;
                    }
                }
            } else {
                self.schemas.insert(name.to_owned(), schema);
            }
            return Ok(());
        }
        let identity = self
            .schemas
            .get_key_value(name)
            .map(|(key, _)| key.as_ptr() as usize);
        let table = if identity.is_none() && self.schemas.len() == self.schemas.capacity() {
            self.allocation_owners.reserve(
                LowererAllocationKind::SchemaTable,
                0,
                crate::compile::hash_allocation_bytes::<String, Schema>(self.schemas.len() + 1)?,
                allocations,
            )?
        } else {
            None
        };
        let key = if identity.is_none() {
            self.allocation_owners.reserve(
                LowererAllocationKind::SchemaKey,
                name.as_ptr() as usize,
                name.len() as u64,
                allocations,
            )?
        } else {
            // HashMap::insert drops the newly supplied duplicate key.
            allocations.acquire(name.len() as u64)?;
            None
        };
        let value = self.allocation_owners.reserve(
            LowererAllocationKind::SchemaValue,
            identity.unwrap_or(name.as_ptr() as usize),
            schema.owned_allocation_bytes()?,
            allocations,
        )?;
        // Schema construction also has discarded inference scratch. The one
        // guarded copy isolates the actual surviving state from that scratch.
        self.schemas.insert(name.to_owned(), schema.clone());
        let identity = self
            .schemas
            .get_key_value(name)
            .expect("inserted schema")
            .0
            .as_ptr() as usize;
        self.allocation_owners.commit(table, 0);
        self.allocation_owners.commit(key, identity);
        self.allocation_owners.commit(value, identity);
        Ok(())
    }

    /// Get the mapping from predicate names to relation IDs
    pub fn rel_ids(&self) -> &HashMap<String, RelId> {
        &self.rel_ids
    }

    /// Get the inferred schemas for predicates
    pub fn schemas(&self) -> &HashMap<String, Schema> {
        &self.schemas
    }

    pub(crate) fn create_helper_relation(&mut self, schema: Schema) -> (String, RelId) {
        let name = format!("__kclique_helper_{}", self.next_rel_id);
        let rel_id = self.get_or_create_rel_id(&name);
        self.schemas.insert(name.clone(), schema);
        (name, rel_id)
    }

    pub(crate) fn create_helper_relation_with_allocation_owner(
        &mut self,
        schema: Schema,
        scope: &mut crate::compile::CompilationAllocationScope<'_>,
    ) -> Result<(String, RelId)> {
        self.allocation_owners.finish_attempt();
        let mut allocations = LoweringAllocations { scope: Some(scope) };
        allocations.formatted(format_args!("__kclique_helper_{}", self.next_rel_id))?;
        let name = format!("__kclique_helper_{}", self.next_rel_id);
        let rel_id = self.get_or_create_rel_id_with_owner(&name, &mut allocations)?;
        self.insert_schema_with_owner(&name, schema, &mut allocations)?;
        Ok((name, rel_id))
    }

    /// Get or allocate a relation ID for a predicate
    fn get_or_create_rel_id(&mut self, name: &str) -> RelId {
        if let Some(&id) = self.rel_ids.get(name) {
            id
        } else {
            let id = RelId(self.next_rel_id);
            self.next_rel_id += 1;
            self.rel_ids.insert(name.to_string(), id);
            id
        }
    }

    fn get_or_create_rel_id_with_owner(
        &mut self,
        name: &str,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<RelId> {
        if !self.rel_ids.contains_key(name) {
            let table = if self.rel_ids.len() == self.rel_ids.capacity() {
                self.allocation_owners.reserve(
                    LowererAllocationKind::RelationTable,
                    0,
                    crate::compile::hash_allocation_bytes::<String, RelId>(self.rel_ids.len() + 1)?,
                    allocations,
                )?
            } else {
                None
            };
            let key = self.allocation_owners.reserve(
                LowererAllocationKind::RelationKey,
                name.as_ptr() as usize,
                name.len() as u64,
                allocations,
            )?;
            let id = self.get_or_create_rel_id(name);
            let identity = self
                .rel_ids
                .get_key_value(name)
                .expect("inserted relation id")
                .0
                .as_ptr() as usize;
            self.allocation_owners.commit(table, 0);
            self.allocation_owners.commit(key, identity);
            return Ok(id);
        }
        Ok(self.get_or_create_rel_id(name))
    }

    /// Reject rules whose variables draw incompatible column types from
    /// the predicate schemas they touch. The executor requires exact
    /// schema equality when relations meet, so any conflict tolerated
    /// here would surface later as an internal kernel schema error
    /// instead of a source-level diagnostic.
    fn validate_rule_types(&self, program: &Program) -> Result<()> {
        self.validate_rule_types_with_owner(program, &mut LoweringAllocations { scope: None })
    }

    fn validate_rule_types_with_owner(
        &self,
        program: &Program,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<()> {
        allocations.hash::<&str, ()>(program.predicates.len())?;
        let declared_predicates = program
            .predicates
            .iter()
            .map(|declaration| declaration.name.as_str())
            .collect::<HashSet<_>>();
        for rule in &program.rules {
            let var_types = self.infer_rule_variable_types_with_owner(
                rule,
                |atom, index| {
                    self.schemas
                        .get(&atom.predicate)
                        .and_then(|schema| schema.column_type(index))
                },
                UserFunctionTypeEvidence::RequireExpansion,
                allocations,
            )?;
            let Some(head_schema) = self.schemas.get(&rule.head.predicate) else {
                continue;
            };
            for (j, term) in rule.head.terms.iter().enumerate() {
                let Some((_, head_ty)) = head_schema.columns.get(j) else {
                    continue;
                };
                match term {
                    Term::Variable(name) => {
                        let Some((body_ty, source)) = var_types.get(name) else {
                            continue;
                        };
                        if body_ty != head_ty {
                            return Err(allocations.compilation_error(format_args!(
                                "Type mismatch in rule for '{}': variable {} is {:?} \
                                 (from {}) but {} declares {:?} at position {}",
                                rule.head.predicate,
                                name,
                                body_ty,
                                source,
                                rule.head.predicate,
                                head_ty,
                                j
                            )));
                        }
                    }
                    Term::Anonymous => {}
                    Term::Aggregate(aggregate) => {
                        let aggregate_ty = Self::infer_aggregate_result_type(
                            rule,
                            aggregate,
                            &var_types,
                            allocations,
                        )?
                        .ok_or_else(|| allocations.unsafe_variable(&aggregate.variable))?;
                        if aggregate_ty != *head_ty {
                            return Err(allocations.compilation_error(format_args!(
                                "Type mismatch in rule for '{}': aggregate {:?} over {} produces \
                                 {:?}, but the predicate schema requires {:?} at position {}",
                                rule.head.predicate,
                                aggregate.op,
                                aggregate.variable,
                                aggregate_ty,
                                head_ty,
                                j
                            )));
                        }
                    }
                    Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {
                        if declared_predicates.contains(rule.head.predicate.as_str()) {
                            term_to_typed_const_value_with_owner(term, *head_ty, allocations)
                                .map_err(|error| {
                                    if allocations.scope.is_some()
                                        && matches!(
                                            error.unowned(),
                                            XlogError::AllocationRefused { .. }
                                        )
                                    {
                                        return error;
                                    }
                                    allocations.compilation_error(format_args!(
                                    "Type mismatch in rule for '{}': head term at position {} is \
                                     not compatible with {:?}: {}",
                                    rule.head.predicate, j, head_ty, error
                                ))
                                })?;
                        } else if term.inferred_scalar_type() != *head_ty {
                            return Err(allocations.compilation_error(format_args!(
                                "Type mismatch in rule for '{}': undeclared head term at position \
                                 {} has inferred type {:?}, but another clause requires {:?}",
                                rule.head.predicate,
                                j,
                                term.inferred_scalar_type(),
                                head_ty
                            )));
                        }
                    }
                    _ if term.inferred_scalar_type() != *head_ty => {
                        return Err(allocations.compilation_error(format_args!(
                            "Type mismatch in rule for '{}': head term at position {} has type \
                             {:?}, but the predicate schema requires {:?}",
                            rule.head.predicate,
                            j,
                            term.inferred_scalar_type(),
                            head_ty
                        )));
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    fn infer_aggregate_result_type(
        rule: &Rule,
        aggregate: &crate::ast::AggExpr,
        variable_types: &HashMap<String, (ScalarType, String)>,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<Option<ScalarType>> {
        let Some((input_type, source)) = variable_types.get(&aggregate.variable) else {
            return Ok(aggregate.input_independent_result_type());
        };
        aggregate
            .result_type_for_input(*input_type)
            .map(Some)
            .ok_or_else(|| {
                let required = match aggregate.op {
                    AggOp::Count => "any scalar input",
                    AggOp::Sum | AggOp::Min | AggOp::Max => "U32 or U64 input",
                    AggOp::LogSumExp => "F64 input",
                };
                allocations.compilation_error(format_args!(
                    "Unsupported aggregate input in rule for '{}': {:?}({}) receives {:?} from \
                     {}, but the execution provider requires {}",
                    rule.head.predicate,
                    aggregate.op,
                    aggregate.variable,
                    input_type,
                    source,
                    required
                ))
            })
    }

    fn record_rule_variable_type(
        rule: &Rule,
        variable_types: &mut HashMap<String, (ScalarType, String)>,
        variable: &str,
        typ: ScalarType,
        source: String,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<()> {
        match variable_types.get(variable) {
            Some((existing, existing_source)) if *existing != typ => Err(allocations
                .compilation_error(format_args!(
                    "Type mismatch in rule for '{}': variable {} is {:?} (from {}) but {:?} \
                     is required by {}",
                    rule.head.predicate, variable, existing, existing_source, typ, source
                ))),
            Some(_) => Ok(()),
            None => {
                variable_types.insert(variable.to_string(), (typ, source));
                Ok(())
            }
        }
    }

    /// Collect all type evidence available inside a rule.
    ///
    /// Ordinary body atoms are considered together because lowering joins them
    /// before evaluating arithmetic bindings. Arithmetic bindings are then
    /// processed in source order so chained `is` expressions can propagate their
    /// result types. Unknown inputs defer an arithmetic result until a later
    /// schema-inference iteration; known incompatible evidence is rejected here.
    fn infer_rule_variable_types_with_user_functions<F>(
        &self,
        rule: &Rule,
        column_type: F,
        user_functions: UserFunctionTypeEvidence,
    ) -> Result<HashMap<String, (ScalarType, String)>>
    where
        F: FnMut(&Atom, usize) -> Option<ScalarType>,
    {
        self.infer_rule_variable_types_with_owner(
            rule,
            column_type,
            user_functions,
            &mut LoweringAllocations { scope: None },
        )
    }

    fn infer_rule_variable_types_with_owner<F>(
        &self,
        rule: &Rule,
        mut column_type: F,
        user_functions: UserFunctionTypeEvidence,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<HashMap<String, (ScalarType, String)>>
    where
        F: FnMut(&Atom, usize) -> Option<ScalarType>,
    {
        let mut variable_types = HashMap::new();

        for literal in &rule.body {
            let atom = match literal {
                BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => atom,
                BodyLiteral::Epistemic(_)
                | BodyLiteral::Comparison(_)
                | BodyLiteral::IsExpr(_)
                | BodyLiteral::Univ(_) => continue,
            };
            for (index, term) in atom.terms.iter().enumerate() {
                let Term::Variable(variable) = term else {
                    continue;
                };
                let Some(typ) = column_type(atom, index) else {
                    continue;
                };
                allocations.formatted(format_args!("{} position {}", atom.predicate, index))?;
                if !variable_types.contains_key(variable) {
                    allocations.acquire(variable.len() as u64)?;
                    if variable_types.len() == variable_types.capacity() {
                        allocations
                            .hash::<String, (ScalarType, String)>(variable_types.len() + 1)?;
                    }
                }
                Self::record_rule_variable_type(
                    rule,
                    &mut variable_types,
                    variable,
                    typ,
                    format!("{} position {}", atom.predicate, index),
                    allocations,
                )?;
            }
        }

        for literal in &rule.body {
            let BodyLiteral::IsExpr(is_expr) = literal else {
                continue;
            };
            allocations.arithmetic_type(&is_expr.expr)?;
            let result_type = Self::infer_arith_type_from_known_variables(
                &is_expr.expr,
                &|variable| variable_types.get(variable).map(|(typ, _)| *typ),
                user_functions,
                allocations,
            )?;
            if let Some(result_type) = result_type {
                allocations.acquire("an arithmetic binding".len() as u64)?;
                if !variable_types.contains_key(&is_expr.target) {
                    allocations.acquire(is_expr.target.len() as u64)?;
                    if variable_types.len() == variable_types.capacity() {
                        allocations
                            .hash::<String, (ScalarType, String)>(variable_types.len() + 1)?;
                    }
                }
                Self::record_rule_variable_type(
                    rule,
                    &mut variable_types,
                    &is_expr.target,
                    result_type,
                    "an arithmetic binding".to_string(),
                    allocations,
                )?;
            }
        }

        Ok(variable_types)
    }

    pub(crate) fn infer_rule_variable_types<F>(
        &self,
        rule: &Rule,
        column_type: F,
    ) -> Result<HashMap<String, (ScalarType, String)>>
    where
        F: FnMut(&Atom, usize) -> Option<ScalarType>,
    {
        self.infer_rule_variable_types_with_user_functions(
            rule,
            column_type,
            UserFunctionTypeEvidence::RequireExpansion,
        )
    }

    pub(crate) fn infer_rule_variable_types_with_allocation_owner<F>(
        &self,
        rule: &Rule,
        column_type: F,
        scope: &mut crate::compile::CompilationAllocationScope<'_>,
    ) -> Result<HashMap<String, (ScalarType, String)>>
    where
        F: FnMut(&Atom, usize) -> Option<ScalarType>,
    {
        self.infer_rule_variable_types_with_owner(
            rule,
            column_type,
            UserFunctionTypeEvidence::RequireExpansion,
            &mut LoweringAllocations { scope: Some(scope) },
        )
    }

    fn infer_rule_head_column_types_with_user_functions<F>(
        &self,
        rule: &Rule,
        column_type: F,
        user_functions: UserFunctionTypeEvidence,
    ) -> Result<Vec<Option<ScalarType>>>
    where
        F: FnMut(&Atom, usize) -> Option<ScalarType>,
    {
        self.infer_rule_head_column_types_with_owner(
            rule,
            column_type,
            user_functions,
            &mut LoweringAllocations { scope: None },
        )
    }

    fn infer_rule_head_column_types_with_owner<F>(
        &self,
        rule: &Rule,
        column_type: F,
        user_functions: UserFunctionTypeEvidence,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<Vec<Option<ScalarType>>>
    where
        F: FnMut(&Atom, usize) -> Option<ScalarType>,
    {
        let variable_types = self.infer_rule_variable_types_with_owner(
            rule,
            column_type,
            user_functions,
            allocations,
        )?;
        allocations.vector::<Option<ScalarType>>(rule.head.terms.len())?;
        rule.head
            .terms
            .iter()
            .map(|term| match term {
                Term::Variable(name) => Ok(variable_types.get(name).map(|(typ, _)| *typ)),
                Term::Aggregate(aggregate) => {
                    Self::infer_aggregate_result_type(rule, aggregate, &variable_types, allocations)
                }
                Term::Anonymous => Ok(None),
                _ => Ok(Some(term.inferred_scalar_type())),
            })
            .collect()
    }

    /// Infer each rule-head column from the same body, arithmetic, and aggregate
    /// evidence used by schema inference during lowering.
    ///
    /// A `None` column has no statically known evidence yet. This path requires
    /// user-defined function calls to have been expanded.
    pub(crate) fn infer_rule_head_column_types<F>(
        &self,
        rule: &Rule,
        column_type: F,
    ) -> Result<Vec<Option<ScalarType>>>
    where
        F: FnMut(&Atom, usize) -> Option<ScalarType>,
    {
        self.infer_rule_head_column_types_with_user_functions(
            rule,
            column_type,
            UserFunctionTypeEvidence::RequireExpansion,
        )
    }

    /// Infer rule-head columns before user-defined functions have been expanded.
    /// Function-call results remain unknown, while independent body, arithmetic,
    /// aggregate, and head-term evidence is still validated and propagated.
    pub(crate) fn infer_rule_head_column_types_before_function_expansion<F>(
        &self,
        rule: &Rule,
        column_type: F,
    ) -> Result<Vec<Option<ScalarType>>>
    where
        F: FnMut(&Atom, usize) -> Option<ScalarType>,
    {
        self.infer_rule_head_column_types_with_user_functions(
            rule,
            column_type,
            UserFunctionTypeEvidence::Defer,
        )
    }

    /// Infer schemas from facts and predicate declarations
    pub(crate) fn infer_schemas(&mut self, program: &Program) -> Result<()> {
        self.infer_schemas_with_owner(program, &mut LoweringAllocations { scope: None })
    }

    pub(crate) fn infer_schemas_with_allocation_owner(
        &mut self,
        program: &Program,
        scope: &mut crate::compile::CompilationAllocationScope<'_>,
    ) -> Result<()> {
        let mut stage = scope.child_scope();
        let result = self.infer_schemas_with_owner(
            program,
            &mut LoweringAllocations {
                scope: Some(&mut stage),
            },
        );
        self.allocation_owners.finish_attempt();
        crate::compile::finish_compiler_stage(result, stage)
    }

    fn infer_schemas_with_owner(
        &mut self,
        program: &Program,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<()> {
        self.allocation_owners.finish_attempt();
        allocations.hash::<String, ScalarType>(program.domains.len())?;
        for domain in &program.domains {
            allocations.acquire(domain.name.len() as u64)?;
        }
        let domains: HashMap<String, ScalarType> = program
            .domains
            .iter()
            .map(|domain| (domain.name.clone(), domain.typ))
            .collect();

        // First, use explicit predicate declarations
        for pred_decl in &program.predicates {
            if allocations.scope.is_some() {
                allocations.vector::<crate::ast::PredColumn>(pred_decl.arity())?;
                if pred_decl.columns.is_empty() {
                    for typ in &pred_decl.types {
                        allocations.acquire(typ.owned_allocation_bytes()?)?;
                    }
                } else {
                    for column in &pred_decl.columns {
                        allocations.acquire(column.typ.owned_allocation_bytes()?)?;
                        if let Some(name) = &column.name {
                            allocations.acquire(name.len() as u64)?;
                        }
                    }
                }
                allocations.vector::<(String, ScalarType)>(pred_decl.arity())?;
                allocations.vector::<usize>(pred_decl.arity())?;
                allocations.vector::<String>(pred_decl.arity())?;
                for (index, column) in pred_decl.columns.iter().enumerate() {
                    match &column.name {
                        Some(name) if !name.trim().is_empty() => {
                            allocations.acquire((name.len() as u64).checked_mul(2).ok_or_else(
                                || XlogError::allocation_extent_overflow("schema text"),
                            )?)?
                        }
                        Some(name) => {
                            allocations.acquire(name.len() as u64)?;
                            allocations.formatted(format_args!("col{index}"))?;
                        }
                        None => {
                            allocations.formatted(format_args!("c{index}"))?;
                            allocations.formatted(format_args!("c{index}"))?;
                        }
                    }
                }
                if pred_decl.columns.is_empty() {
                    for index in 0..pred_decl.arity() {
                        allocations.formatted(format_args!("c{index}"))?;
                        allocations.formatted(format_args!("c{index}"))?;
                    }
                }
            }
            let declared_columns = pred_decl.schema_columns();
            let columns: Vec<(String, ScalarType)> = declared_columns
                .iter()
                .enumerate()
                .map(|(i, col)| {
                    let name = col.name.clone().unwrap_or_else(|| format!("c{}", i));
                    resolve_pred_column_type(&pred_decl.name, i, &col.typ, &domains, allocations)
                        .map(|ty| (name, ty))
                })
                .collect::<Result<Vec<_>>>()?;
            self.insert_schema_with_owner(&pred_decl.name, Schema::new(columns), allocations)?;
        }

        // Then, infer from facts (if no declaration exists)
        for rule in program.facts() {
            let pred = &rule.head.predicate;
            if !self.schemas.contains_key(pred) {
                allocations.generated_schema(rule.head.terms.len())?;
                let columns: Vec<(String, ScalarType)> = rule
                    .head
                    .terms
                    .iter()
                    .enumerate()
                    .map(|(i, term)| {
                        let ty = term.inferred_scalar_type();
                        (format!("c{}", i), ty)
                    })
                    .collect();
                self.insert_schema_with_owner(pred, Schema::new(columns), allocations)?;
            }
        }

        // Probabilistic facts and annotated-disjunction choices are also
        // ground schema evidence. Register them before the body-only fallback
        // so a rule variable does not default an otherwise typed predicate to
        // U64 merely because its facts live in a probabilistic AST collection.
        for pf in &program.prob_facts {
            let pred = &pf.atom.predicate;
            if self.schemas.contains_key(pred) {
                continue;
            }
            allocations.generated_schema(pf.atom.terms.len())?;
            let columns: Vec<(String, ScalarType)> = pf
                .atom
                .terms
                .iter()
                .enumerate()
                .map(|(i, term)| (format!("c{}", i), term.inferred_scalar_type()))
                .collect();
            self.insert_schema_with_owner(pred, Schema::new(columns), allocations)?;
        }

        for ad in &program.annotated_disjunctions {
            for choice in &ad.choices {
                let pred = &choice.atom.predicate;
                if self.schemas.contains_key(pred) {
                    continue;
                }
                allocations.generated_schema(choice.atom.terms.len())?;
                let columns: Vec<(String, ScalarType)> = choice
                    .atom
                    .terms
                    .iter()
                    .enumerate()
                    .map(|(i, term)| (format!("c{}", i), term.inferred_scalar_type()))
                    .collect();
                self.insert_schema_with_owner(pred, Schema::new(columns), allocations)?;
            }
        }

        // Infer schemas for extensional predicates that occur only in rule bodies
        // before propagating rule-head types. This lets aggregates and ordinary
        // head variables consume the same body-column types that execution will
        // use, while leaving derived predicates to the fixed-point pass below.
        allocations.hash::<&str, ()>(program.rules.len())?;
        let derived_predicates = program
            .rules
            .iter()
            .filter(|rule| !rule.body.is_empty())
            .map(|rule| rule.head.predicate.as_str())
            .collect::<HashSet<_>>();
        for rule in &program.rules {
            for lit in &rule.body {
                let atom = match lit {
                    BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => atom,
                    BodyLiteral::Epistemic(_)
                    | BodyLiteral::Comparison(_)
                    | BodyLiteral::IsExpr(_)
                    | BodyLiteral::Univ(_) => continue,
                };
                let pred = &atom.predicate;
                if self.schemas.contains_key(pred) || derived_predicates.contains(pred.as_str()) {
                    continue;
                }
                allocations.generated_schema(atom.terms.len())?;
                allocations.sort_labels(&atom.terms)?;
                let columns: Vec<(String, ScalarType)> = atom
                    .terms
                    .iter()
                    .enumerate()
                    .map(|(i, term)| (format!("c{}", i), term.inferred_scalar_type()))
                    .collect();
                let schema = Schema::new(columns)
                    .with_sort_labels(sort_labels_from_terms(&atom.terms))
                    .expect("body sort labels match inferred schema arity");
                self.insert_schema_with_owner(pred, schema, allocations)?;
            }
        }

        // Propagate body-derived rule-head types to a fixed point before
        // defaulting variables that have no type anchor. Columns converge
        // independently so an unresolved sibling cannot hide known evidence.
        let mut inferred_rule_columns: HashMap<String, Vec<Option<ScalarType>>> = HashMap::new();
        for rule in &program.rules {
            if self.schemas.contains_key(&rule.head.predicate) {
                continue;
            }
            allocations.acquire(rule.head.predicate.len() as u64)?;
            if !inferred_rule_columns.contains_key(&rule.head.predicate) {
                if inferred_rule_columns.len() == inferred_rule_columns.capacity() {
                    allocations
                        .hash::<String, Vec<Option<ScalarType>>>(inferred_rule_columns.len() + 1)?;
                }
                allocations.vector::<Option<ScalarType>>(rule.head.terms.len())?;
            }
            inferred_rule_columns
                .entry(rule.head.predicate.clone())
                .or_insert_with(|| vec![None; rule.head.terms.len()]);
        }

        loop {
            let mut changed = false;
            for rule in &program.rules {
                let pred = &rule.head.predicate;
                if self.schemas.contains_key(pred) {
                    continue;
                }

                let Some(current_columns) = inferred_rule_columns.get(pred) else {
                    continue;
                };
                if current_columns.len() != rule.head.terms.len() {
                    continue;
                }

                let resolved_columns = self.infer_rule_head_column_types_with_owner(
                    rule,
                    |atom, index| {
                        self.schemas
                            .get(&atom.predicate)
                            .and_then(|schema| schema.column_type(index))
                            .or_else(|| {
                                inferred_rule_columns
                                    .get(&atom.predicate)
                                    .and_then(|columns| columns.get(index))
                                    .copied()
                                    .flatten()
                            })
                    },
                    UserFunctionTypeEvidence::RequireExpansion,
                    allocations,
                )?;

                let columns = inferred_rule_columns
                    .get_mut(pred)
                    .expect("rule-head inference entry exists");
                for (index, (column, resolved)) in
                    columns.iter_mut().zip(resolved_columns).enumerate()
                {
                    match (*column, resolved) {
                        (None, Some(resolved)) => {
                            *column = Some(resolved);
                            changed = true;
                        }
                        (Some(existing), Some(resolved)) if existing != resolved => {
                            return Err(allocations.compilation_error(format_args!(
                                "Conflicting inferred schema for predicate '{}': column {} is \
                                 {:?} in one rule and {:?} in another",
                                pred,
                                index + 1,
                                existing,
                                resolved
                            )));
                        }
                        (None, None) | (Some(_), None) | (Some(_), Some(_)) => {}
                    }
                }
            }
            if !changed {
                break;
            }
        }
        for rule in &program.rules {
            let pred = &rule.head.predicate;
            if !self.schemas.contains_key(pred) {
                allocations.generated_schema(rule.head.terms.len())?;
                allocations.sort_labels(&rule.head.terms)?;
                let inferred_columns = inferred_rule_columns
                    .get(pred)
                    .expect("undeclared rule head has an inference entry");
                let columns = rule
                    .head
                    .terms
                    .iter()
                    .zip(inferred_columns)
                    .enumerate()
                    .map(|(index, (term, inferred))| {
                        (
                            format!("c{index}"),
                            inferred.unwrap_or_else(|| term.inferred_scalar_type()),
                        )
                    })
                    .collect();
                let schema = Schema::new(columns)
                    .with_sort_labels(sort_labels_from_terms(&rule.head.terms))
                    .expect("rule head sort labels match inferred schema arity");
                self.insert_schema_with_owner(pred, schema, allocations)?;
            }
        }

        Ok(())
    }

    /// Infer predicate schemas and validate the type flow within every rule.
    ///
    /// This applies the schema/type contract shared by execution routes without
    /// preprocessing, stratifying, validating unrelated term forms, or building
    /// an execution plan. The inferred schemas remain available through
    /// [`Lowerer::schemas`].
    pub fn infer_and_validate_schemas(&mut self, program: &Program) -> Result<()> {
        self.infer_schemas(program)?;
        self.validate_rule_types(program)
    }

    fn infer_cardinalities_with_owner(
        &mut self,
        program: &Program,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<()> {
        let fact_count = program.facts().count();
        let _counts_owner = match allocations.scope.as_deref_mut() {
            Some(scope) => Some(scope.reserve_allocation(
                crate::compile::hash_allocation_bytes::<&str, u64>(fact_count)?,
            )?),
            None => None,
        };
        self.est_cardinality.clear();
        self.allocation_owners
            .retire(LowererAllocationKind::EstimateKey);

        let mut fact_counts: HashMap<&str, u64> = HashMap::with_capacity(fact_count);
        for fact in program.facts() {
            *fact_counts.entry(fact.head.predicate.as_str()).or_insert(0) += 1;
        }

        for pred in self.schemas.keys() {
            let est = self
                .cardinality_hints
                .get(pred)
                .copied()
                .or_else(|| fact_counts.get(pred.as_str()).copied())
                .unwrap_or(1000)
                .max(1);
            let table = if self.est_cardinality.len() == self.est_cardinality.capacity() {
                self.allocation_owners.reserve(
                    LowererAllocationKind::EstimateTable,
                    0,
                    crate::compile::hash_allocation_bytes::<String, u64>(
                        self.est_cardinality.len() + 1,
                    )?,
                    allocations,
                )?
            } else {
                None
            };
            let key = self.allocation_owners.reserve(
                LowererAllocationKind::EstimateKey,
                pred.as_ptr() as usize,
                pred.len() as u64,
                allocations,
            )?;
            self.est_cardinality.insert(pred.clone(), est);
            let identity = self
                .est_cardinality
                .get_key_value(pred)
                .expect("inserted cardinality")
                .0
                .as_ptr() as usize;
            self.allocation_owners.commit(table, 0);
            self.allocation_owners.commit(key, identity);
        }
        Ok(())
    }

    /// Build SCCs from the dependency graph
    fn build_sccs(&mut self, program: &Program) {
        self.build_sccs_with_owner(program, &mut LoweringAllocations { scope: None })
            .expect("ordinary SCC construction has no allocation callback");
    }

    fn build_sccs_with_owner(
        &mut self,
        program: &Program,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<()> {
        match allocations.scope.as_deref_mut() {
            Some(scope) => {
                let mut stage = scope.child_scope();
                let result = self.build_sccs_core(
                    program,
                    &mut LoweringAllocations {
                        scope: Some(&mut stage),
                    },
                );
                self.allocation_owners.finish_attempt();
                crate::compile::finish_compiler_stage(result, stage)
            }
            None => self.build_sccs_core(program, allocations),
        }
    }

    fn build_sccs_core(
        &mut self,
        program: &Program,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<()> {
        let graph = match allocations.scope.as_deref_mut() {
            Some(scope) => {
                crate::stratify::build_dependency_graph_with_allocation_owner(program, scope)?
            }
            None => build_dependency_graph(program),
        };
        let scc_groups = match allocations.scope.as_deref_mut() {
            Some(scope) => {
                crate::stratify::find_sccs_for_lowering_with_allocation_owner(&graph, scope)?
            }
            None => find_sccs_for_lowering(&graph),
        };

        self.sccs.clear();
        self.scc_by_pred.clear();
        self.allocation_owners
            .retire(LowererAllocationKind::ComponentValue);
        self.allocation_owners
            .retire(LowererAllocationKind::ComponentIndexKey);
        for (id, predicates) in scc_groups.iter().enumerate() {
            let mut bytes =
                crate::compile::push_vector_allocation_bytes::<String>(predicates.len())?;
            for predicate in predicates {
                bytes = bytes.checked_add(predicate.len() as u64).ok_or_else(|| {
                    XlogError::allocation_extent_overflow("strongly connected component names")
                })?;
            }
            let value = self.allocation_owners.reserve(
                LowererAllocationKind::ComponentValue,
                id,
                bytes,
                allocations,
            )?;
            allocations.vector::<&crate::stratify::DepEdge>(graph.outgoing_len(&predicates[0]))?;
            let table = if self.sccs.len() == self.sccs.capacity() {
                self.allocation_owners.reserve(
                    LowererAllocationKind::ComponentTable,
                    0,
                    crate::compile::push_vector_allocation_bytes::<Scc>(self.sccs.len() + 1)?,
                    allocations,
                )?
            } else {
                None
            };
            // An SCC is recursive if it has more than one predicate
            // or if a single predicate depends on itself positively
            let is_recursive = if predicates.len() > 1 {
                true
            } else {
                let pred = &predicates[0];
                graph
                    .outgoing(pred)
                    .iter()
                    .any(|e| e.to == *pred && e.dep_type == DepType::Positive)
            };

            for p in predicates {
                let index_table = if self.scc_by_pred.len() == self.scc_by_pred.capacity() {
                    self.allocation_owners.reserve(
                        LowererAllocationKind::ComponentIndexTable,
                        0,
                        crate::compile::hash_allocation_bytes::<String, u32>(
                            self.scc_by_pred.len() + 1,
                        )?,
                        allocations,
                    )?
                } else {
                    None
                };
                let key = self.allocation_owners.reserve(
                    LowererAllocationKind::ComponentIndexKey,
                    p.as_ptr() as usize,
                    p.len() as u64,
                    allocations,
                )?;
                self.scc_by_pred.insert(p.clone(), id as u32);
                let identity = self
                    .scc_by_pred
                    .get_key_value(p)
                    .expect("inserted component index")
                    .0
                    .as_ptr() as usize;
                self.allocation_owners.commit(index_table, 0);
                self.allocation_owners.commit(key, identity);
            }
            self.sccs.push(Scc {
                id: id as u32,
                predicates: predicates.clone(),
                is_recursive,
            });
            self.allocation_owners.commit(table, 0);
            self.allocation_owners.commit(value, id);
        }
        Ok(())
    }

    fn prepare_program_for_lowering(&mut self, program: &Program) -> Result<()> {
        self.prepare_program_for_lowering_with_owner(
            program,
            &mut LoweringAllocations { scope: None },
        )
    }

    fn prepare_program_for_lowering_with_owner(
        &mut self,
        program: &Program,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<()> {
        match allocations.scope.as_deref_mut() {
            Some(scope) => {
                let mut stage = scope.child_scope();
                let result = self.prepare_program_for_lowering_core(
                    program,
                    &mut LoweringAllocations {
                        scope: Some(&mut stage),
                    },
                );
                self.allocation_owners.finish_attempt();
                crate::compile::finish_compiler_stage(result, stage)
            }
            None => self.prepare_program_for_lowering_core(program, allocations),
        }
    }

    fn prepare_program_for_lowering_core(
        &mut self,
        program: &Program,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<()> {
        validate_lowerable_terms(program, allocations)?;
        validate_is_expression_bindings(program, allocations)?;
        self.infer_schemas_with_owner(program, allocations)?;
        self.validate_rule_types_with_owner(program, allocations)?;
        self.infer_cardinalities_with_owner(program, allocations)?;

        // Pre-allocate RelIds for declared predicates so schema-only programs
        // can populate relation stores before any facts or executable rules
        // mention those relations. This keeps ILP candidate generation and
        // runtime relation upload aligned with declared schemas.
        for pred_decl in &program.predicates {
            self.get_or_create_rel_id_with_owner(&pred_decl.name, allocations)?;
        }
        // Facts are grouped and materialized directly into the relation store.
        // They still need stable relation IDs even when no declaration or rule
        // mentions their predicates.
        for fact in program.facts() {
            self.get_or_create_rel_id_with_owner(&fact.head.predicate, allocations)?;
        }

        Ok(())
    }

    /// Validate every source-level contract enforced while lowering, without
    /// requiring a stratification or constructing an execution plan.
    ///
    /// Epistemic preparation uses this after replacing modal literals with their
    /// validation-only ordinary counterparts. It intentionally exercises the same
    /// schema inference, rule type checks, constant conversion, arithmetic ordering,
    /// negation lowering, and head projection as production lowering.
    pub(crate) fn validate_program_without_plan(&mut self, program: &Program) -> Result<()> {
        self.validate_program_without_plan_with_owner(
            program,
            &mut LoweringAllocations { scope: None },
        )
    }

    pub(crate) fn validate_program_without_plan_with_allocation_owner(
        &mut self,
        program: &Program,
        scope: &mut crate::compile::CompilationAllocationScope<'_>,
    ) -> Result<()> {
        let result = self.validate_program_without_plan_with_owner(
            program,
            &mut LoweringAllocations { scope: Some(scope) },
        );
        self.allocation_owners.finish_attempt();
        result
    }

    fn validate_program_without_plan_with_owner(
        &mut self,
        program: &Program,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<()> {
        self.prepare_program_for_lowering_with_owner(program, allocations)?;

        for rule in program.proper_rules() {
            self.lower_rule_with_owner(rule, allocations)?;
        }

        // Match the relation allocation and validation performed by `lower_program`
        // for learnable rules as well. These rules cannot contain modal literals, but
        // they may share declarations and schemas with the authored program.
        for learnable in &program.learnable_rules {
            self.get_or_create_rel_id_with_owner(&learnable.head.predicate, allocations)?;
            for lit in &learnable.body {
                if let BodyLiteral::Positive(atom) = lit {
                    self.get_or_create_rel_id_with_owner(&atom.predicate, allocations)?;
                }
            }
        }
        for learnable in &program.learnable_rules {
            self.lower_learnable_rule_with_owner(learnable, allocations)?;
        }

        Ok(())
    }

    /// Lower an entire program to an execution plan
    pub fn lower_program(&mut self, program: &Program) -> Result<ExecutionPlan> {
        self.lower_program_with_owner(program, &mut LoweringAllocations { scope: None })
    }

    pub fn lower_program_with_allocation_owner(
        &mut self,
        program: &Program,
        scope: &mut crate::compile::CompilationAllocationScope<'_>,
    ) -> Result<ExecutionPlan> {
        let result =
            self.lower_program_with_owner(program, &mut LoweringAllocations { scope: Some(scope) });
        self.allocation_owners.finish_attempt();
        result
    }

    pub(crate) fn lower_program_guarded_with_allocation_owner(
        &mut self,
        program: &Program,
        scope: &mut crate::compile::CompilationAllocationScope<'_>,
    ) -> Result<crate::compile::CompilationOutput<ExecutionPlan>> {
        let mut stage = scope.child_scope();
        let plan = self.lower_program_with_allocation_owner(program, &mut stage)?;
        // Lowering moves rule buffers into the plan while its collector also
        // owns discarded lookup and join scratch. Preserve only an acquired
        // copy of the actual output after those original buffers have dropped.
        let allocation_owner = stage.reserve_allocation(plan.owned_allocation_bytes()?)?;
        let value = plan.clone();
        drop(plan);
        drop(stage);
        Ok(crate::compile::CompilationOutput {
            value,
            allocation_owner,
        })
    }

    fn lower_program_with_owner(
        &mut self,
        program: &Program,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<ExecutionPlan> {
        self.prepare_program_for_lowering_with_owner(program, allocations)?;

        // Build SCCs
        self.build_sccs_with_owner(program, allocations)?;

        // Build execution plan
        let mut builder = PlanBuilder::new();
        allocations.vector::<Scc>(self.sccs.len())?;
        allocations.vector::<Vec<CompiledRule>>(self.sccs.len())?;
        allocations.vector::<IrStratum>(self.strata.len())?;
        for scc in &self.sccs {
            let count = program
                .proper_rules()
                .filter(|rule| scc.predicates.contains(&rule.head.predicate))
                .count()
                .checked_add(
                    program
                        .learnable_rules
                        .iter()
                        .filter(|rule| scc.predicates.contains(&rule.head.predicate))
                        .count(),
                )
                .ok_or_else(|| XlogError::allocation_extent_overflow("compiled rules"))?;
            allocations.vector::<CompiledRule>(count)?;
        }

        // Add SCCs to the builder
        for scc in &self.sccs {
            allocations.vector::<String>(scc.predicates.len())?;
            for name in &scc.predicates {
                allocations.acquire(name.len() as u64)?;
            }
            builder.add_scc(scc.clone());
        }

        // Build strata from our strata field
        for (id, preds) in self.strata.iter().enumerate() {
            let _predicate_set_owner = match allocations.scope.as_deref_mut() {
                Some(scope) => Some(scope.reserve_allocation(
                    crate::compile::hash_allocation_bytes::<&str, ()>(preds.len())?,
                )?),
                None => None,
            };
            allocations.vector::<u32>(self.sccs.len())?;
            // Find which SCCs belong to this stratum
            let pred_set: std::collections::HashSet<&str> =
                preds.iter().map(|s| s.as_str()).collect();
            let scc_ids: Vec<u32> = self
                .sccs
                .iter()
                .filter(|scc| scc.predicates.iter().any(|p| pred_set.contains(p.as_str())))
                .map(|scc| scc.id)
                .collect();

            if !scc_ids.is_empty() {
                builder.add_stratum(IrStratum {
                    id: id as u32,
                    sccs: scc_ids,
                });
            }
        }

        {
            let rules_by_pred = group_proper_rules_with_owner(program, allocations)?;
            // Preserve lexicographic head order and authored order within each
            // group. The lookup and sorting buffers retire before learnable
            // rules; the builder owns only the actual compiled rule outputs.
            let _sort_owner = match allocations.scope.as_deref_mut() {
                Some(scope) => Some(scope.reserve_allocation(
                    crate::compile::push_vector_allocation_bytes::<&str>(rules_by_pred.0.len())?,
                )?),
                None => None,
            };
            let mut preds_sorted: Vec<&str> = rules_by_pred.0.keys().copied().collect();
            preds_sorted.sort();
            for pred in preds_sorted {
                let rules = &rules_by_pred.0[pred];
                let scc_id = self.find_scc_for_predicate(pred);

                for rule in rules {
                    let body = self.lower_rule_with_owner(rule, allocations)?;
                    if let Some(schema) = self.schemas.get(pred) {
                        allocations.schema(schema)?;
                    }
                    let meta = self.create_meta_for_predicate(pred);
                    allocations.acquire(pred.len() as u64)?;

                    builder.add_rule(
                        scc_id,
                        CompiledRule {
                            head: pred.to_owned(),
                            body,
                            meta,
                        },
                    );
                }
            }
        }

        // Lower learnable rules into tensor-masked joins.
        // Pre-allocate RelIds for ALL learnable predicates (heads + bodies)
        // so every lower_learnable_rule snapshot is complete.
        for learnable in &program.learnable_rules {
            self.get_or_create_rel_id_with_owner(&learnable.head.predicate, allocations)?;
            for lit in &learnable.body {
                if let BodyLiteral::Positive(atom) = lit {
                    self.get_or_create_rel_id_with_owner(&atom.predicate, allocations)?;
                }
            }
        }
        for learnable in &program.learnable_rules {
            let head_pred = &learnable.head.predicate;
            let scc_id = self.find_scc_for_predicate(head_pred);
            let body = self.lower_learnable_rule_with_owner(learnable, allocations)?;
            if let Some(schema) = self.schemas.get(head_pred) {
                allocations.schema(schema)?;
            }
            let meta = self.create_meta_for_predicate(head_pred);
            allocations.acquire(head_pred.len() as u64)?;
            builder.add_rule(
                scc_id,
                CompiledRule {
                    head: head_pred.clone(),
                    body,
                    meta,
                },
            );
        }

        let mut plan = builder.build();
        // Record relation arities for downstream generic multiway shape
        // promoters that size Scan leaves from these values.
        // One pre-pass over the AST covers every predicate the lowerer
        // assigned a RelId: rule heads, positive/negated body atoms,
        // and facts.
        for rule in program.proper_rules() {
            if let Some(&id) = self.rel_ids.get(&rule.head.predicate) {
                if !plan.rel_arities.contains_key(&id)
                    && plan.rel_arities.len() == plan.rel_arities.capacity()
                {
                    allocations.hash::<RelId, usize>(plan.rel_arities.len() + 1)?;
                }
                plan.rel_arities.insert(id, rule.head.terms.len());
            }
            for lit in &rule.body {
                let atom = match lit {
                    BodyLiteral::Positive(a) | BodyLiteral::Negated(a) => a,
                    _ => continue,
                };
                if let Some(&id) = self.rel_ids.get(&atom.predicate) {
                    if !plan.rel_arities.contains_key(&id)
                        && plan.rel_arities.len() == plan.rel_arities.capacity()
                    {
                        allocations.hash::<RelId, usize>(plan.rel_arities.len() + 1)?;
                    }
                    plan.rel_arities.insert(id, atom.terms.len());
                }
            }
        }
        for fact in program.facts() {
            if let Some(&id) = self.rel_ids.get(&fact.head.predicate) {
                if !plan.rel_arities.contains_key(&id)
                    && plan.rel_arities.len() == plan.rel_arities.capacity()
                {
                    allocations.hash::<RelId, usize>(plan.rel_arities.len() + 1)?;
                }
                plan.rel_arities.insert(id, fact.head.terms.len());
            }
        }
        Ok(plan)
    }

    /// Find the SCC ID for a predicate
    fn find_scc_for_predicate(&self, pred: &str) -> u32 {
        self.scc_by_pred.get(pred).copied().unwrap_or(0)
    }

    /// Create metadata for a predicate
    fn create_meta_for_predicate(&self, pred: &str) -> RirMeta {
        let schema = self
            .schemas
            .get(pred)
            .cloned()
            .unwrap_or_else(|| Schema::new(vec![]));
        RirMeta::with_schema(schema)
    }

    /// Lower a learnable rule template into a TensorMaskedJoin node.
    /// Validates that the body has exactly two positive atoms.
    /// Sorts rel_index by RelId for deterministic tensor dimension mapping.
    /// Uses get_or_create_rel_id for heads so head-only predicates are handled.
    fn lower_learnable_rule(&mut self, rule: &LearnableRule) -> Result<RirNode> {
        self.lower_learnable_rule_with_owner(rule, &mut LoweringAllocations { scope: None })
    }

    fn lower_learnable_rule_with_owner(
        &mut self,
        rule: &LearnableRule,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<RirNode> {
        // Validate body shape before indexing fixed body positions.
        if rule.body.len() != 2 {
            return Err(allocations.compilation_error(format_args!(
                "learnable rule '{}' requires exactly 2 body literals, got {}",
                rule.mask_name,
                rule.body.len()
            )));
        }
        for (idx, lit) in rule.body.iter().enumerate() {
            match lit {
                BodyLiteral::Positive(_) => {}
                _ => {
                    return Err(allocations.compilation_error(format_args!(
                        "learnable rule '{}' body[{}]: only positive atoms allowed",
                        rule.mask_name, idx
                    )));
                }
            }
        }

        // Sort by RelId for deterministic tensor dimension mapping.
        allocations.vector::<(RelId, String)>(self.rel_ids.len())?;
        for name in self.rel_ids.keys() {
            allocations.acquire(name.len() as u64)?;
        }
        allocations.vector::<usize>(
            rule.body[0]
                .atom()
                .unwrap()
                .terms
                .len()
                .checked_mul(rule.body[1].atom().unwrap().terms.len())
                .ok_or_else(|| XlogError::allocation_extent_overflow("learnable join keys"))?,
        )?;
        allocations.vector::<usize>(
            rule.body[0]
                .atom()
                .unwrap()
                .terms
                .len()
                .checked_mul(rule.body[1].atom().unwrap().terms.len())
                .ok_or_else(|| XlogError::allocation_extent_overflow("learnable join keys"))?,
        )?;
        let mut rel_index: Vec<(RelId, String)> = self
            .rel_ids()
            .iter()
            .map(|(name, id)| (*id, name.clone()))
            .collect();
        rel_index.sort_by_key(|(id, _)| id.0);
        let schema_size = rel_index.len();

        let (left_keys, right_keys) =
            self.extract_template_join_keys(&rule.body[0], &rule.body[1], allocations)?;

        allocations.acquire(rule.head.predicate.len() as u64)?;
        let head_rel_name = rule.head.predicate.clone();
        // Allocate lazily because head-only predicates may not have a RelId yet.
        let head_rel_id = self.get_or_create_rel_id_with_owner(&head_rel_name, allocations)?;

        // Compute head projection: map head variables to join result columns.
        // Join result layout: [left_col_0..left_col_n, right_col_0..right_col_m].
        let left_atom = rule.body[0].atom().unwrap();
        let right_atom = rule.body[1].atom().unwrap();
        let left_arity = left_atom.terms.len();

        // Build variable -> first-occurrence column mapping over joined result
        let mut var_to_col: HashMap<String, usize> = HashMap::new();
        for (i, term) in left_atom.terms.iter().enumerate() {
            if let Some(name) = term.variable_name() {
                allocations.acquire(name.len() as u64)?;
                if !var_to_col.contains_key(name) && var_to_col.len() == var_to_col.capacity() {
                    allocations.hash::<String, usize>(var_to_col.len() + 1)?;
                }
                var_to_col.entry(name.to_string()).or_insert(i);
            }
        }
        for (i, term) in right_atom.terms.iter().enumerate() {
            if let Some(name) = term.variable_name() {
                allocations.acquire(name.len() as u64)?;
                if !var_to_col.contains_key(name) && var_to_col.len() == var_to_col.capacity() {
                    allocations.hash::<String, usize>(var_to_col.len() + 1)?;
                }
                var_to_col.entry(name.to_string()).or_insert(left_arity + i);
            }
        }

        allocations.vector::<usize>(rule.head.terms.len())?;
        let mut head_projection: Vec<usize> = Vec::new();
        for term in &rule.head.terms {
            if let Some(name) = term.variable_name() {
                let col = var_to_col.get(name).ok_or_else(|| {
                    allocations.compilation_error(format_args!(
                        "Learnable rule head variable '{}' not found in body atoms \
                         ({}, {}). All head variables must appear in the body.",
                        name, left_atom.predicate, right_atom.predicate,
                    ))
                })?;
                head_projection.push(*col);
            } else {
                return Err(allocations.compilation_error(format_args!(
                    "Learnable rule head must contain only variables, \
                     found constant {:?} in head of '{}'",
                    term, head_rel_name,
                )));
            }
        }

        // Infer schema for head predicate from the learnable rule if not already set.
        // The head's column types come from the projected join columns.
        if !self.schemas.contains_key(&head_rel_name) {
            allocations.generated_schema(head_projection.len())?;
            let columns: Vec<(String, ScalarType)> = head_projection
                .iter()
                .enumerate()
                .map(|(i, &col)| {
                    // Determine the type from left or right atom's schema
                    let ty = if col < left_arity {
                        self.schemas
                            .get(&left_atom.predicate)
                            .and_then(|s| s.column_type(col))
                            .unwrap_or(ScalarType::U32)
                    } else {
                        self.schemas
                            .get(&right_atom.predicate)
                            .and_then(|s| s.column_type(col - left_arity))
                            .unwrap_or(ScalarType::U32)
                    };
                    (format!("c{}", i), ty)
                })
                .collect();
            self.insert_schema_with_owner(&head_rel_name, Schema::new(columns), allocations)?;
        }

        allocations.acquire(rule.mask_name.len() as u64)?;
        Ok(RirNode::TensorMaskedJoin {
            mask_name: rule.mask_name.clone(),
            schema_size,
            left_keys,
            right_keys,
            rel_index,
            head_rel_name,
            head_rel_id,
            max_active_rules: self.max_active_rules,
            head_projection,
        })
    }

    /// Extract join keys from two body literals' shared variables.
    /// For `b1(X, Z), b2(Z, Y)`, the shared variable Z gives left_keys=[1], right_keys=[0].
    fn extract_template_join_keys(
        &self,
        left: &BodyLiteral,
        right: &BodyLiteral,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<(Vec<usize>, Vec<usize>)> {
        let left_atom = left.atom().ok_or_else(|| {
            allocations.compilation_error(format_args!("Learnable body[0] is not an atom"))
        })?;
        let right_atom = right.atom().ok_or_else(|| {
            allocations.compilation_error(format_args!("Learnable body[1] is not an atom"))
        })?;

        let mut left_keys = Vec::new();
        let mut right_keys = Vec::new();

        for (li, lt) in left_atom.terms.iter().enumerate() {
            if let Some(lname) = lt.variable_name() {
                for (ri, rt) in right_atom.terms.iter().enumerate() {
                    if let Some(rname) = rt.variable_name() {
                        if lname == rname {
                            left_keys.push(li);
                            right_keys.push(ri);
                        }
                    }
                }
            }
        }

        Ok((left_keys, right_keys))
    }

    /// Lower a single rule to an RIR node
    fn lower_rule(&mut self, rule: &Rule) -> Result<RirNode> {
        self.lower_rule_with_owner(rule, &mut LoweringAllocations { scope: None })
    }

    fn lower_rule_with_owner(
        &mut self,
        rule: &Rule,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<RirNode> {
        if let Some(lit) = rule.body.iter().find_map(|lit| match lit {
            BodyLiteral::Epistemic(lit) => Some(lit),
            _ => None,
        }) {
            allocations.acquire("RIR lowering boundary".len() as u64)?;
            allocations.formatted(format_args!(
                "{:?} {}({})",
                lit.op,
                lit.atom.predicate,
                lit.atom.arity()
            ))?;
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "RIR lowering boundary".to_string(),
                context: format!("{:?} {}({})", lit.op, lit.atom.predicate, lit.atom.arity()),
            });
        }

        // Split body literals.
        allocations.vector::<&Atom>(rule.body.len())?;
        allocations.vector::<&Atom>(rule.body.len())?;
        allocations.vector::<&Comparison>(rule.body.len())?;
        allocations.vector::<&IsExpr>(rule.body.len())?;
        let (positive_atoms, negated_atoms, comparisons, is_exprs) =
            Self::split_body_literals(&rule.body);

        // Allocate RelIds for all body predicates in source order so join planning
        // does not influence identifier assignment.
        for lit in &rule.body {
            match lit {
                BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => {
                    self.get_or_create_rel_id_with_owner(&atom.predicate, allocations)?;
                }
                BodyLiteral::Epistemic(_)
                | BodyLiteral::Comparison(_)
                | BodyLiteral::IsExpr(_)
                | BodyLiteral::Univ(_) => {}
            }
        }

        // Plan positive atoms (join tree shape + leaf order).
        //
        // Rules with no positive atoms are legal for nullary/ground heads in our
        // probabilistic profiles (e.g. `q() :- not p().`). Lower them by seeding
        // the body with a unit relation ({()}) and applying filters/negations.
        let (positive_root, leaf_order) = if positive_atoms.is_empty() {
            (RirNode::Unit, Vec::new())
        } else {
            self.plan_positive_atoms(&positive_atoms, allocations)?
        };

        // Build variable environment from the planned leaf order (matches join output layout:
        // left subtree columns then right subtree columns).
        let mut var_env = VariableEnv::new();
        let mut current_col = 0;
        for atom in &leaf_order {
            let schema = self.schemas.get(&atom.predicate);
            for (i, term) in atom.terms.iter().enumerate() {
                if let Term::Variable(name) = term {
                    if name == "_" {
                        continue;
                    }
                    allocations.acquire(
                        (name.len() as u64)
                            .checked_mul(2)
                            .and_then(|n| n.checked_add(atom.predicate.len() as u64))
                            .ok_or_else(|| {
                                XlogError::allocation_extent_overflow("variable environment")
                            })?,
                    )?;
                    if !var_env.occurrences.contains_key(name)
                        && var_env.occurrences.len() == var_env.occurrences.capacity()
                    {
                        allocations.hash::<String, Vec<(String, usize, usize)>>(
                            var_env.occurrences.len() + 1,
                        )?;
                    }
                    let (length, capacity) = var_env
                        .occurrences
                        .get(name)
                        .map(|v| (v.len(), v.capacity()))
                        .unwrap_or((0, 0));
                    if length == capacity {
                        allocations.vector::<(String, usize, usize)>(length + 1)?;
                    }
                    var_env.add_occurrence(name, atom.predicate.clone(), i, current_col + i);
                    // Also record the type for this variable (first occurrence wins)
                    if !var_env.types.contains_key(name) {
                        allocations.acquire(name.len() as u64)?;
                        if var_env.types.len() == var_env.types.capacity() {
                            allocations.hash::<String, ScalarType>(var_env.types.len() + 1)?;
                        }
                        let typ = schema
                            .and_then(|s| s.column_type(i))
                            .unwrap_or(ScalarType::I64); // Default to I64 for arithmetic
                        var_env.types.insert(name.to_string(), typ);
                    }
                }
            }
            current_col += atom.terms.len();
        }
        var_env.total_cols = current_col;

        // Lower the body starting from the planned positive join root.
        let body_node = self.lower_body_parts_with_owner(
            positive_root,
            &negated_atoms,
            &comparisons,
            &is_exprs,
            &mut var_env,
            allocations,
        )?;

        if rule.has_aggregation() {
            return self.lower_aggregate_rule_with_owner(
                &rule.head,
                body_node,
                &var_env,
                allocations,
            );
        }

        // Project to head terms (variables and constants).
        let projection_exprs =
            self.compute_head_projection_with_owner(&rule.head, &var_env, allocations)?;

        if Self::is_identity_projection(&projection_exprs, var_env.column_count()) {
            Ok(body_node)
        } else {
            allocations.acquire(std::mem::size_of::<RirNode>() as u64)?;
            Ok(RirNode::Project {
                input: Box::new(body_node),
                columns: projection_exprs,
            })
        }
    }

    fn split_body_literals(
        body: &[BodyLiteral],
    ) -> (Vec<&Atom>, Vec<&Atom>, Vec<&Comparison>, Vec<&IsExpr>) {
        let mut positive_atoms: Vec<&Atom> = Vec::new();
        let mut negated_atoms: Vec<&Atom> = Vec::new();
        let mut comparisons: Vec<&Comparison> = Vec::new();
        let mut is_exprs: Vec<&IsExpr> = Vec::new();

        for lit in body {
            match lit {
                BodyLiteral::Positive(atom) => positive_atoms.push(atom),
                BodyLiteral::Negated(atom) => negated_atoms.push(atom),
                BodyLiteral::Epistemic(_) => {}
                BodyLiteral::Comparison(cmp) => comparisons.push(cmp),
                BodyLiteral::IsExpr(is_expr) => is_exprs.push(is_expr),
                BodyLiteral::Univ(_) => {}
            }
        }

        (positive_atoms, negated_atoms, comparisons, is_exprs)
    }

    fn atom_vars(atom: &Atom) -> std::collections::HashSet<String> {
        atom.terms
            .iter()
            .flat_map(|t| t.variables().into_iter())
            .filter(|name| *name != "_")
            .map(ToOwned::to_owned)
            .collect()
    }

    fn estimate_atom_rows(&self, atom: &Atom) -> f64 {
        let base = self
            .est_cardinality
            .get(&atom.predicate)
            .copied()
            .unwrap_or(1000)
            .max(1) as f64;

        let const_count = atom
            .terms
            .iter()
            .filter(|term| {
                matches!(
                    term,
                    Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_)
                )
            })
            .count();

        // Equality constants are usually selective; use a conservative default.
        let selectivity = 0.1_f64.powi(const_count as i32);
        (base * selectivity).max(1.0)
    }

    fn build_cartesian_join(
        &self,
        left: RirNode,
        right: RirNode,
        left_width: usize,
        right_width: usize,
    ) -> RirNode {
        // Implement cross join by appending a constant key column to both inputs and joining on it,
        // then projecting away the constant columns.
        let left_const_col =
            ProjectExpr::Computed(Expr::Const(ConstValue::U32(0)), ScalarType::U32);
        let right_const_col =
            ProjectExpr::Computed(Expr::Const(ConstValue::U32(0)), ScalarType::U32);

        let mut left_cols: Vec<ProjectExpr> = (0..left_width).map(ProjectExpr::Column).collect();
        left_cols.push(left_const_col);
        let left_aug = RirNode::Project {
            input: Box::new(left),
            columns: left_cols,
        };

        let mut right_cols: Vec<ProjectExpr> = (0..right_width).map(ProjectExpr::Column).collect();
        right_cols.push(right_const_col);
        let right_aug = RirNode::Project {
            input: Box::new(right),
            columns: right_cols,
        };

        let joined = RirNode::Join {
            left: Box::new(left_aug),
            right: Box::new(right_aug),
            left_keys: vec![left_width],
            right_keys: vec![right_width],
            join_type: JoinType::Inner,
        };

        let mut keep: Vec<ProjectExpr> = Vec::with_capacity(left_width + right_width);
        keep.extend((0..left_width).map(ProjectExpr::Column));
        let right_start = left_width + 1;
        keep.extend((right_start..right_start + right_width).map(ProjectExpr::Column));

        RirNode::Project {
            input: Box::new(joined),
            columns: keep,
        }
    }

    fn make_leaf_plan<'a>(
        &mut self,
        atom: &'a Atom,
        orig_idx: usize,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<JoinPlan<'a>> {
        let allocation_receipt = if allocations.scope.is_some() {
            let overflow = || XlogError::allocation_extent_overflow("join leaf");
            let count = atom.terms.len();
            let mut bytes = crate::compile::hash_allocation_bytes::<String, usize>(count)?;
            bytes = bytes
                .checked_add(crate::compile::push_vector_allocation_bytes::<Expr>(count)?)
                .and_then(|n| {
                    n.checked_add((count as u64).checked_mul(
                        (std::mem::size_of::<RirNode>() + 3 * std::mem::size_of::<Expr>()) as u64,
                    )?)
                })
                .and_then(|n| {
                    n.checked_add(
                        std::mem::size_of::<&Atom>() as u64 + std::mem::size_of::<usize>() as u64,
                    )
                })
                .ok_or_else(overflow)?;
            for term in &atom.terms {
                bytes = bytes
                    .checked_add(term.owned_allocation_bytes()?)
                    .ok_or_else(overflow)?;
                match term {
                    Term::String(text) => {
                        bytes = bytes.checked_add(text.len() as u64).ok_or_else(overflow)?
                    }
                    Term::Symbol(id) => {
                        bytes = bytes
                            .checked_add(
                                (symbol::resolved_allocation_bytes(*id)? as u64)
                                    .checked_mul(2)
                                    .ok_or_else(overflow)?,
                            )
                            .ok_or_else(overflow)?
                    }
                    _ => {}
                }
            }
            allocations.temporary(bytes)?
        } else {
            None
        };
        let rel_id = self.get_or_create_rel_id_with_owner(&atom.predicate, allocations)?;
        let scan = RirNode::Scan { rel: rel_id };
        let node = self.apply_constant_filters(scan, atom, 0, allocations, &allocation_receipt)?;

        let mut var_pos: HashMap<String, usize> = HashMap::new();
        for (i, term) in atom.terms.iter().enumerate() {
            if let Term::Variable(name) = term {
                if name != "_" {
                    var_pos.entry(name.clone()).or_insert(i);
                }
            }
        }

        let est_rows = self.estimate_atom_rows(atom);
        Ok(JoinPlan {
            node,
            leaf_order: vec![atom],
            leaf_order_idx: vec![orig_idx],
            var_pos,
            width: atom.terms.len(),
            est_rows,
            total_cost: est_rows,
            allocation_receipt,
        })
    }

    fn join_plans<'a>(
        &self,
        left: &JoinPlan<'a>,
        right: &JoinPlan<'a>,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<JoinPlan<'a>> {
        let allocation_receipt = allocations.join_candidate(left, right)?;
        let shared_vars: Vec<&String> = left
            .var_pos
            .keys()
            .filter(|v| right.var_pos.contains_key(*v))
            .collect();

        let node = if shared_vars.is_empty() {
            self.build_cartesian_join(
                left.node.clone(),
                right.node.clone(),
                left.width,
                right.width,
            )
        } else {
            let mut key_pairs: Vec<(usize, usize)> = shared_vars
                .iter()
                .filter_map(|v| {
                    Some((
                        left.var_pos.get(*v).copied()?,
                        right.var_pos.get(*v).copied()?,
                    ))
                })
                .collect();
            key_pairs.sort_unstable();

            let (left_keys, right_keys): (Vec<usize>, Vec<usize>) = key_pairs.into_iter().unzip();

            RirNode::Join {
                left: Box::new(left.node.clone()),
                right: Box::new(right.node.clone()),
                left_keys,
                right_keys,
                join_type: JoinType::Inner,
            }
        };

        let mut leaf_order = left.leaf_order.clone();
        leaf_order.extend(right.leaf_order.iter().copied());

        let mut leaf_order_idx = left.leaf_order_idx.clone();
        leaf_order_idx.extend_from_slice(&right.leaf_order_idx);

        let mut var_pos = left.var_pos.clone();
        for (var, pos) in &right.var_pos {
            var_pos.entry(var.clone()).or_insert(left.width + *pos);
        }

        let shared = shared_vars.len();
        let mut selectivity = if shared == 0 {
            1.0
        } else {
            0.1_f64.powi(shared as i32)
        };
        if shared == 0 {
            // Penalize cartesian joins strongly.
            selectivity *= 1.0e6;
        }

        let output_rows = (left.est_rows * right.est_rows * selectivity).max(1.0);

        // Hash join cost is sensitive to which side is build (right) and probe (left).
        let build_cost = right.est_rows;
        let probe_cost = left.est_rows * 0.5;
        let total_cost = left.total_cost + right.total_cost + build_cost + probe_cost + output_rows;

        Ok(JoinPlan {
            node,
            leaf_order,
            leaf_order_idx,
            var_pos,
            width: left.width + right.width,
            est_rows: output_rows,
            total_cost,
            allocation_receipt,
        })
    }

    fn plan_positive_atoms_bushy<'a>(
        &mut self,
        atoms: &[&'a Atom],
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<(RirNode, Vec<&'a Atom>)> {
        let n = atoms.len();
        if n == 0 {
            return Err(allocations.compilation_error(format_args!("Empty rule body")));
        }
        if n == 1 {
            let plan = self.make_leaf_plan(atoms[0], 0, allocations)?;
            return allocations.finish(plan);
        }

        let size = 1usize << n;
        let _table_receipt = if allocations.scope.is_some() {
            allocations.temporary(crate::compile::push_vector_allocation_bytes::<
                Option<JoinPlan<'a>>,
            >(size)?)?
        } else {
            None
        };
        let mut best: Vec<Option<JoinPlan<'a>>> = (0..size).map(|_| None).collect();

        for (i, atom) in atoms.iter().enumerate() {
            best[1usize << i] = Some(self.make_leaf_plan(atom, i, allocations)?);
        }

        fn lex_lt(a: &[usize], b: &[usize]) -> bool {
            for (ai, bi) in a.iter().zip(b.iter()) {
                if ai != bi {
                    return ai < bi;
                }
            }
            a.len() < b.len()
        }

        for mask in 1..size {
            if mask.count_ones() <= 1 {
                continue;
            }

            let mut best_for_mask: Option<JoinPlan<'a>> = None;

            let mut sub = (mask - 1) & mask;
            while sub > 0 {
                let a = sub;
                let b = mask ^ a;
                if b == 0 {
                    sub = (sub - 1) & mask;
                    continue;
                }

                let (Some(plan_a), Some(plan_b)) = (&best[a], &best[b]) else {
                    sub = (sub - 1) & mask;
                    continue;
                };

                // Consider both orientations: A ⋈ B and B ⋈ A.
                for (left, right) in [(plan_a, plan_b), (plan_b, plan_a)] {
                    let cand = self.join_plans(left, right, allocations)?;
                    let replace = match &best_for_mask {
                        None => true,
                        Some(current) => {
                            if cand.total_cost < current.total_cost {
                                true
                            } else if (cand.total_cost - current.total_cost).abs() < 1e-9 {
                                lex_lt(&cand.leaf_order_idx, &current.leaf_order_idx)
                            } else {
                                false
                            }
                        }
                    };

                    if replace {
                        best_for_mask = Some(cand);
                    }
                }

                sub = (sub - 1) & mask;
            }

            best[mask] = best_for_mask;
        }

        let full_mask = size - 1;
        if let Some(plan) = best[full_mask].take() {
            return allocations.finish(plan);
        }

        // Should be unreachable, but fall back to greedy ordering.
        let ordered = self.order_positive_atoms_greedy(atoms, allocations)?;
        let mut dummy_env = VariableEnv::new();
        let node = self.build_join_tree(&ordered, &mut dummy_env, allocations)?;
        Ok((node, ordered))
    }

    fn plan_positive_atoms<'a>(
        &mut self,
        atoms: &[&'a Atom],
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<(RirNode, Vec<&'a Atom>)> {
        if atoms.len() <= 1 {
            if atoms.is_empty() {
                return Err(allocations.compilation_error(format_args!("Empty rule body")));
            }
            let plan = self.make_leaf_plan(atoms[0], 0, allocations)?;
            return allocations.finish(plan);
        }

        const MAX_BUSHY_DP_ATOMS: usize = 10;
        if atoms.len() <= MAX_BUSHY_DP_ATOMS {
            return self.plan_positive_atoms_bushy(atoms, allocations);
        }

        // Greedy bushy join planning for large rules (scales beyond exponential DP).
        self.plan_positive_atoms_bushy_greedy(atoms, allocations)
    }

    fn plan_positive_atoms_bushy_greedy<'a>(
        &mut self,
        atoms: &[&'a Atom],
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<(RirNode, Vec<&'a Atom>)> {
        if atoms.is_empty() {
            return Err(allocations.compilation_error(format_args!("Empty rule body")));
        }

        fn lex_lt(a: &[usize], b: &[usize]) -> bool {
            for (ai, bi) in a.iter().zip(b.iter()) {
                if ai != bi {
                    return ai < bi;
                }
            }
            a.len() < b.len()
        }

        let _table_receipt = if allocations.scope.is_some() {
            allocations.temporary(
                crate::compile::push_vector_allocation_bytes::<JoinPlan<'a>>(atoms.len())?,
            )?
        } else {
            None
        };
        let mut plans: Vec<JoinPlan<'a>> = Vec::with_capacity(atoms.len());
        for (idx, atom) in atoms.iter().enumerate() {
            plans.push(self.make_leaf_plan(atom, idx, allocations)?);
        }

        while plans.len() > 1 {
            let mut best_pair: Option<(usize, usize, JoinPlan<'a>)> = None;

            for i in 0..plans.len() {
                for j in (i + 1)..plans.len() {
                    let a = &plans[i];
                    let b = &plans[j];

                    let cand_ab = self.join_plans(a, b, allocations)?;
                    let cand_ba = self.join_plans(b, a, allocations)?;

                    let cand = if cand_ab.total_cost < cand_ba.total_cost
                        || (cand_ab.total_cost - cand_ba.total_cost).abs() < 1e-9
                            && lex_lt(&cand_ab.leaf_order_idx, &cand_ba.leaf_order_idx)
                    {
                        cand_ab
                    } else {
                        cand_ba
                    };

                    let replace = match &best_pair {
                        None => true,
                        Some((_bi, _bj, best)) => {
                            if cand.total_cost < best.total_cost {
                                true
                            } else if (cand.total_cost - best.total_cost).abs() < 1e-9 {
                                lex_lt(&cand.leaf_order_idx, &best.leaf_order_idx)
                            } else {
                                false
                            }
                        }
                    };

                    if replace {
                        best_pair = Some((i, j, cand));
                    }
                }
            }

            let Some((i, j, joined)) = best_pair else {
                break;
            };

            // Remove joined inputs from the plan list and replace with the join.
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            plans.remove(b);
            plans.remove(a);
            plans.push(joined);
        }

        let plan = plans
            .pop()
            .ok_or_else(|| allocations.compilation_error(format_args!("Join planning failed")))?;
        allocations.finish(plan)
    }

    fn order_positive_atoms_greedy<'a>(
        &self,
        atoms: &[&'a Atom],
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<Vec<&'a Atom>> {
        allocations.vector::<(usize, &Atom)>(atoms.len())?;
        allocations.vector::<&Atom>(atoms.len())?;
        let mut total_variables = 0usize;
        let mut variable_bytes = 0u64;
        let mut comparator_bytes = 0u64;
        if allocations.scope.is_some() {
            let overflow = || XlogError::allocation_extent_overflow("greedy join variables");
            for atom in atoms {
                let mut count = 0usize;
                let mut bytes = 0u64;
                let mut invalid = false;
                atom.visit_variables(&mut |name| {
                    match count.checked_add(1) {
                        Some(value) => count = value,
                        None => invalid = true,
                    }
                    if name != "_" {
                        match bytes.checked_add(name.len() as u64) {
                            Some(value) => bytes = value,
                            None => invalid = true,
                        }
                    }
                });
                if invalid {
                    return Err(overflow());
                }
                total_variables = total_variables.checked_add(count).ok_or_else(overflow)?;
                variable_bytes = variable_bytes.checked_add(bytes).ok_or_else(overflow)?;
                let atom_bytes = crate::compile::hash_allocation_bytes::<String, ()>(count)?
                    .checked_add(crate::compile::push_vector_allocation_bytes::<&str>(count)?)
                    .and_then(|value| value.checked_add(bytes))
                    .ok_or_else(overflow)?;
                comparator_bytes = comparator_bytes.max(atom_bytes);
            }
            allocations.hash::<String, ()>(total_variables)?;
            allocations.acquire(variable_bytes)?;
            comparator_bytes = comparator_bytes.checked_mul(2).ok_or_else(overflow)?;
        }
        let _comparator_receipt = allocations.temporary(comparator_bytes)?;
        let mut remaining: Vec<(usize, &Atom)> = atoms.iter().copied().enumerate().collect();
        let mut ordered: Vec<&Atom> = Vec::with_capacity(atoms.len());
        let mut bound_vars: HashSet<String> = HashSet::new();

        while !remaining.is_empty() {
            let pick_idx = if ordered.is_empty() {
                remaining
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let (ai, aa) = **a;
                        let (bi, bb) = **b;
                        self.estimate_atom_rows(aa)
                            .partial_cmp(&self.estimate_atom_rows(bb))
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then(ai.cmp(&bi))
                    })
                    .map(|(idx, _)| idx)
                    .unwrap()
            } else {
                remaining
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let (ai, aa) = **a;
                        let (bi, bb) = **b;

                        let a_vars = Self::atom_vars(aa);
                        let b_vars = Self::atom_vars(bb);

                        let a_shared = a_vars.intersection(&bound_vars).count();
                        let b_shared = b_vars.intersection(&bound_vars).count();

                        let a_score = if a_shared == 0 {
                            self.estimate_atom_rows(aa) * 1.0e12
                        } else {
                            self.estimate_atom_rows(aa) / a_shared as f64
                        };
                        let b_score = if b_shared == 0 {
                            self.estimate_atom_rows(bb) * 1.0e12
                        } else {
                            self.estimate_atom_rows(bb) / b_shared as f64
                        };

                        a_score
                            .partial_cmp(&b_score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then(ai.cmp(&bi))
                    })
                    .map(|(idx, _)| idx)
                    .unwrap()
            };

            let (_orig_idx, atom) = remaining.remove(pick_idx);
            ordered.push(atom);
            bound_vars.extend(Self::atom_vars(atom));
        }

        Ok(ordered)
    }

    fn lower_body_parts(
        &mut self,
        positive_root: RirNode,
        negated_atoms: &[&Atom],
        comparisons: &[&Comparison],
        is_exprs: &[&IsExpr],
        var_env: &mut VariableEnv,
    ) -> Result<RirNode> {
        self.lower_body_parts_with_owner(
            positive_root,
            negated_atoms,
            comparisons,
            is_exprs,
            var_env,
            &mut LoweringAllocations { scope: None },
        )
    }

    fn lower_body_parts_with_owner(
        &mut self,
        positive_root: RirNode,
        negated_atoms: &[&Atom],
        comparisons: &[&Comparison],
        is_exprs: &[&IsExpr],
        var_env: &mut VariableEnv,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<RirNode> {
        let mut result = positive_root;

        // Apply comparisons as filters.
        for cmp in comparisons {
            result = self.apply_comparison_with_owner(result, cmp, var_env, allocations)?;
        }

        // Apply is-expressions (must be after atoms that bind the input variables).
        for is_expr in is_exprs {
            result = self.lower_is_expr_with_owner(is_expr, result, var_env, allocations)?;
        }

        // Handle negated atoms via Diff / semi-join.
        for neg_atom in negated_atoms {
            result = self.apply_negation_with_owner(result, neg_atom, var_env, allocations)?;
        }

        Ok(result)
    }

    /// Build a left-deep join tree from positive atoms
    fn build_join_tree(
        &mut self,
        atoms: &[&Atom],
        var_env: &mut VariableEnv,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<RirNode> {
        if atoms.is_empty() {
            return Err(allocations.compilation_error(format_args!("Empty rule body")));
        }

        // Start with the first atom as a scan
        let first_atom = atoms[0];
        let rel_id = self.get_or_create_rel_id_with_owner(&first_atom.predicate, allocations)?;
        let mut result = RirNode::Scan { rel: rel_id };
        let variable_count = atoms.iter().try_fold(0usize, |count, atom| {
            count
                .checked_add(atom.terms.len())
                .ok_or_else(|| XlogError::allocation_extent_overflow("join variables"))
        })?;
        allocations.vector::<(String, usize)>(variable_count)?;
        for atom in atoms {
            for term in &atom.terms {
                if let Term::Variable(name) = term {
                    allocations.acquire(name.len() as u64)?;
                }
            }
        }
        let mut result_vars = self.collect_atom_vars(first_atom);
        let mut result_width = first_atom.terms.len();

        // Apply constant filters if any
        let constant_data = self.acquire_atom_filters(first_atom, allocations)?;
        result = self.apply_constant_filters(result, first_atom, 0, allocations, &constant_data)?;

        // Join with remaining atoms (left-deep)
        for atom in atoms.iter().skip(1) {
            let right_rel_id =
                self.get_or_create_rel_id_with_owner(&atom.predicate, allocations)?;
            let right_scan = RirNode::Scan { rel: right_rel_id };

            // Apply constant filters to the right side
            let constant_data = self.acquire_atom_filters(atom, allocations)?;
            let right_filtered =
                self.apply_constant_filters(right_scan, atom, 0, allocations, &constant_data)?;

            // Compute join keys based on shared variables
            allocations.vector::<usize>(atom.terms.len())?;
            allocations.vector::<usize>(atom.terms.len())?;
            allocations.acquire(2 * std::mem::size_of::<RirNode>() as u64)?;
            let (left_keys, right_keys) = self.compute_join_keys(&result_vars, atom, result_width);

            if left_keys.is_empty() {
                // Cartesian product (no shared variables)
                result = RirNode::Join {
                    left: Box::new(result),
                    right: Box::new(right_filtered),
                    left_keys: vec![],
                    right_keys: vec![],
                    join_type: JoinType::Inner,
                };
            } else {
                result = RirNode::Join {
                    left: Box::new(result),
                    right: Box::new(right_filtered),
                    left_keys,
                    right_keys,
                    join_type: JoinType::Inner,
                };
            }

            // Update result vars for the next iteration
            for (i, term) in atom.terms.iter().enumerate() {
                if let Term::Variable(name) = term {
                    result_vars.push((name.clone(), result_width + i));
                }
            }
            result_width += atom.terms.len();
        }

        // Update var_env with final positions
        var_env.total_cols = result_width;

        Ok(result)
    }

    /// Collect variable names and their positions within an atom
    fn collect_atom_vars(&self, atom: &Atom) -> Vec<(String, usize)> {
        atom.terms
            .iter()
            .enumerate()
            .filter_map(|(i, term)| {
                if let Term::Variable(name) = term {
                    Some((name.clone(), i))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Compute join keys between the current result and a new atom
    fn compute_join_keys(
        &self,
        left_vars: &[(String, usize)],
        right_atom: &Atom,
        _left_width: usize,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left_keys = Vec::new();
        let mut right_keys = Vec::new();

        for (right_idx, term) in right_atom.terms.iter().enumerate() {
            if let Term::Variable(name) = term {
                // Find if this variable exists in the left side
                for (left_name, left_idx) in left_vars {
                    if left_name == name {
                        left_keys.push(*left_idx);
                        right_keys.push(right_idx);
                        break; // Only use first occurrence for join key
                    }
                }
            }
        }

        (left_keys, right_keys)
    }

    /// Apply constant filters for an atom
    fn acquire_atom_filters(
        &self,
        atom: &Atom,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<Option<Arc<dyn Send + Sync>>> {
        allocations.hash::<&str, usize>(atom.terms.len())?;
        allocations.vector::<Expr>(atom.terms.len())?;
        allocations.acquire(std::mem::size_of::<RirNode>() as u64)?;
        let mut text_bytes = 0u64;
        for term in &atom.terms {
            allocations.acquire(2 * std::mem::size_of::<Expr>() as u64)?;
            let bytes = match term {
                Term::String(text) => text.len() as u64,
                Term::Symbol(id) if allocations.scope.is_some() => {
                    symbol::resolved_allocation_bytes(*id)? as u64
                }
                _ => 0,
            };
            text_bytes = text_bytes
                .checked_add(bytes)
                .ok_or_else(|| XlogError::allocation_extent_overflow("constant filter text"))?;
        }
        // The same scope retains this original receipt after the temporary
        // borrow ends. It covers precisely the existing constant text copies.
        allocations.acquire_receipt(text_bytes)
    }

    fn apply_constant_filters(
        &self,
        input: RirNode,
        atom: &Atom,
        _base_col: usize,
        allocations: &mut LoweringAllocations<'_, '_>,
        constant_data: &Option<Arc<dyn Send + Sync>>,
    ) -> Result<RirNode> {
        let mut filters = Vec::new();
        let mut first_var_col: HashMap<&str, usize> = HashMap::new();
        let schema = self.schemas.get(&atom.predicate).ok_or_else(|| {
            allocations.compilation_error(format_args!(
                "Missing schema for predicate {}",
                atom.predicate
            ))
        })?;

        for (i, term) in atom.terms.iter().enumerate() {
            if let Term::Variable(name) = term {
                if name != "_" {
                    if let Some(&first) = first_var_col.get(name.as_str()) {
                        filters.push(Expr::Compare {
                            left: Box::new(Expr::Column(first)),
                            op: CompareOp::Eq,
                            right: Box::new(Expr::Column(i)),
                        });
                    } else {
                        first_var_col.insert(name.as_str(), i);
                    }
                }
            }

            let col_type = schema.column_type(i).ok_or_else(|| {
                allocations.compilation_error(format_args!(
                    "Missing column type for {} column {}",
                    atom.predicate, i
                ))
            })?;
            if let Some(const_val) = term_to_typed_const_value_core(
                term,
                col_type,
                allocations,
                ConstantDataAllocation::Prepared(constant_data),
            )? {
                filters.push(Expr::Compare {
                    left: Box::new(Expr::Column(i)),
                    op: CompareOp::Eq,
                    right: Box::new(Expr::Const(const_val)),
                });
            }
        }

        if filters.is_empty() {
            Ok(input)
        } else {
            let predicate = if filters.len() == 1 {
                filters.pop().unwrap()
            } else {
                Expr::And(filters)
            };

            Ok(RirNode::Filter {
                input: Box::new(input),
                predicate,
            })
        }
    }

    /// Apply a comparison as a filter
    fn apply_comparison(
        &self,
        input: RirNode,
        cmp: &Comparison,
        var_env: &VariableEnv,
    ) -> Result<RirNode> {
        self.apply_comparison_with_owner(
            input,
            cmp,
            var_env,
            &mut LoweringAllocations { scope: None },
        )
    }

    fn apply_comparison_with_owner(
        &self,
        input: RirNode,
        cmp: &Comparison,
        var_env: &VariableEnv,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<RirNode> {
        allocations
            .acquire((std::mem::size_of::<RirNode>() + 2 * std::mem::size_of::<Expr>()) as u64)?;
        let (left_expr, right_expr) = match (&cmp.left, &cmp.right) {
            (Term::Variable(name), term) => {
                let col = var_env.get_column(name).ok_or_else(|| {
                    allocations.compilation_error(format_args!(
                        "Variable {} not found in environment",
                        name
                    ))
                })?;
                let typ = var_env.get_type(name).ok_or_else(|| {
                    allocations
                        .compilation_error(format_args!("Missing type for variable {}", name))
                })?;
                if let Some(const_val) =
                    term_to_typed_const_value_with_owner(term, typ, allocations)?
                {
                    (Expr::Column(col), Expr::Const(const_val))
                } else {
                    (
                        self.term_to_expr_with_owner(&cmp.left, var_env, allocations)?,
                        self.term_to_expr_with_owner(&cmp.right, var_env, allocations)?,
                    )
                }
            }
            (term, Term::Variable(name)) => {
                let col = var_env.get_column(name).ok_or_else(|| {
                    allocations.compilation_error(format_args!(
                        "Variable {} not found in environment",
                        name
                    ))
                })?;
                let typ = var_env.get_type(name).ok_or_else(|| {
                    allocations
                        .compilation_error(format_args!("Missing type for variable {}", name))
                })?;
                if let Some(const_val) =
                    term_to_typed_const_value_with_owner(term, typ, allocations)?
                {
                    (Expr::Const(const_val), Expr::Column(col))
                } else {
                    (
                        self.term_to_expr_with_owner(&cmp.left, var_env, allocations)?,
                        self.term_to_expr_with_owner(&cmp.right, var_env, allocations)?,
                    )
                }
            }
            _ => (
                self.term_to_expr_with_owner(&cmp.left, var_env, allocations)?,
                self.term_to_expr_with_owner(&cmp.right, var_env, allocations)?,
            ),
        };

        let op = match cmp.op {
            CompOp::Eq => CompareOp::Eq,
            CompOp::Ne => CompareOp::Ne,
            CompOp::Lt => CompareOp::Lt,
            CompOp::Le => CompareOp::Le,
            CompOp::Gt => CompareOp::Gt,
            CompOp::Ge => CompareOp::Ge,
        };

        Ok(RirNode::Filter {
            input: Box::new(input),
            predicate: Expr::Compare {
                left: Box::new(left_expr),
                op,
                right: Box::new(right_expr),
            },
        })
    }

    /// Convert a term to an expression
    fn term_to_expr(&self, term: &Term, var_env: &VariableEnv) -> Result<Expr> {
        self.term_to_expr_with_owner(term, var_env, &mut LoweringAllocations { scope: None })
    }

    fn term_to_expr_with_owner(
        &self,
        term: &Term,
        var_env: &VariableEnv,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<Expr> {
        match term {
            Term::Variable(name) => {
                if let Some(col) = var_env.get_column(name) {
                    Ok(Expr::Column(col))
                } else {
                    Err(allocations.compilation_error(format_args!(
                        "Variable {} not found in environment",
                        name
                    )))
                }
            }
            Term::Anonymous => Err(allocations.compilation_error(format_args!(
                "Anonymous wildcard '_' not allowed in comparisons"
            ))),
            Term::Integer(i) => Ok(Expr::Const(ConstValue::I64(*i))),
            Term::Float(f) => Ok(Expr::Const(ConstValue::F64(*f))),
            Term::String(s) => {
                allocations.acquire(s.len() as u64)?;
                Ok(Expr::Const(ConstValue::Symbol(s.clone())))
            }
            Term::Symbol(id) => Ok(Expr::Const(ConstValue::Symbol(allocations.symbol(*id)?))),
            Term::Aggregate(_) => Err(allocations
                .compilation_error(format_args!("Aggregates not allowed in comparisons"))),
            Term::List(_) | Term::Cons { .. } | Term::Compound { .. } | Term::PredRef(_) => {
                Err(term_not_lowerable_error(
                    "comparison",
                    term_kind_for_lowering_error(term),
                    allocations,
                ))
            }
        }
    }

    /// Apply negation via set difference
    fn apply_negation(
        &mut self,
        input: RirNode,
        neg_atom: &Atom,
        var_env: &VariableEnv,
    ) -> Result<RirNode> {
        self.apply_negation_with_owner(
            input,
            neg_atom,
            var_env,
            &mut LoweringAllocations { scope: None },
        )
    }

    fn apply_negation_with_owner(
        &mut self,
        input: RirNode,
        neg_atom: &Atom,
        var_env: &VariableEnv,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<RirNode> {
        let rel_id = self.get_or_create_rel_id_with_owner(&neg_atom.predicate, allocations)?;
        let constant_data = self.acquire_atom_filters(neg_atom, allocations)?;
        let neg_scan = RirNode::Scan { rel: rel_id };

        // Apply constant filters to the negated atom
        let neg_filtered =
            self.apply_constant_filters(neg_scan, neg_atom, 0, allocations, &constant_data)?;

        // Find which columns from the input correspond to variables in the negated atom
        allocations.vector::<usize>(neg_atom.terms.len())?;
        allocations.vector::<usize>(neg_atom.terms.len())?;
        let mut input_cols = Vec::new();
        let mut neg_cols = Vec::new();

        for (neg_idx, term) in neg_atom.terms.iter().enumerate() {
            if let Term::Variable(name) = term {
                if let Some(col) = var_env.get_column(name) {
                    input_cols.push(col);
                    neg_cols.push(neg_idx);
                }
            }
        }

        if input_cols.is_empty() {
            // A negated atom with no shared variables is a Boolean existence gate over
            // the entire positive input, not a tuple difference. Give both sides the
            // same synthetic key, anti-join on that key, then remove it. If the
            // negated atom has any matching row, every input row is rejected; if it is
            // empty, the anti-join returns the input unchanged. This also preserves
            // arbitrary input schemas, including the zero-arity unit used by
            // negation-only rules.
            let input_width = var_env.column_count();
            allocations.vector::<ProjectExpr>(
                input_width
                    .checked_add(1)
                    .ok_or_else(|| XlogError::allocation_extent_overflow("negation projection"))?,
            )?;
            allocations.vector::<ProjectExpr>(input_width)?;
            allocations.vector::<ProjectExpr>(1)?;
            allocations.vector::<usize>(1)?;
            allocations.vector::<usize>(1)?;
            allocations.acquire(4 * std::mem::size_of::<RirNode>() as u64)?;
            let join_key =
                || ProjectExpr::Computed(Expr::Const(ConstValue::U32(0)), ScalarType::U32);

            let mut keyed_input_columns: Vec<ProjectExpr> =
                (0..input_width).map(ProjectExpr::Column).collect();
            keyed_input_columns.push(join_key());
            let keyed_input = RirNode::Project {
                input: Box::new(input),
                columns: keyed_input_columns,
            };
            let keyed_negation = RirNode::Project {
                input: Box::new(neg_filtered),
                columns: vec![join_key()],
            };
            let gated_input = RirNode::Join {
                left: Box::new(keyed_input),
                right: Box::new(keyed_negation),
                left_keys: vec![input_width],
                right_keys: vec![0],
                join_type: JoinType::Anti,
            };

            Ok(RirNode::Project {
                input: Box::new(gated_input),
                columns: (0..input_width).map(ProjectExpr::Column).collect(),
            })
        } else {
            allocations.vector::<ProjectExpr>(neg_cols.len())?;
            allocations.vector::<ProjectExpr>(input_cols.len())?;
            allocations.vector::<usize>(input_cols.len())?;
            allocations.vector::<usize>(input_cols.len())?;
            allocations.acquire(4 * std::mem::size_of::<RirNode>() as u64)?;
            if allocations.scope.is_some() {
                allocations.acquire(input.owned_allocation_bytes()?)?;
            }
            // Project the negated atom to only the shared variable columns
            let neg_projected = if neg_cols.len() < neg_atom.terms.len() {
                let neg_proj_exprs: Vec<ProjectExpr> =
                    neg_cols.iter().map(|&c| ProjectExpr::Column(c)).collect();
                RirNode::Project {
                    input: Box::new(neg_filtered),
                    columns: neg_proj_exprs,
                }
            } else {
                neg_filtered
            };

            // Project input to matching columns for the diff, then diff
            // Actually, for proper anti-join semantics we need to be careful.
            // The Diff operation subtracts matching tuples.
            // We need to project input to the shared columns, diff, then rejoin.

            // Simpler approach: project input to shared columns, diff with negated,
            // then rejoin with original
            let input_proj_exprs: Vec<ProjectExpr> =
                input_cols.iter().map(|&c| ProjectExpr::Column(c)).collect();
            let input_projected = RirNode::Project {
                input: Box::new(input.clone()),
                columns: input_proj_exprs,
            };

            // The Diff gives us the keys that should be kept
            let kept_keys = RirNode::Diff {
                left: Box::new(input_projected),
                right: Box::new(neg_projected),
            };

            // Join back with original input to get full tuples
            // This effectively filters the input to only rows where the key
            // is not in the negated relation
            Ok(RirNode::Join {
                left: Box::new(input),
                right: Box::new(kept_keys),
                left_keys: input_cols.clone(),
                right_keys: (0..input_cols.len()).collect(),
                join_type: JoinType::Semi,
            })
        }
    }

    fn is_identity_projection(proj: &[ProjectExpr], input_cols: usize) -> bool {
        if proj.len() != input_cols {
            return false;
        }
        proj.iter()
            .enumerate()
            .all(|(i, e)| matches!(e, ProjectExpr::Column(c) if *c == i))
    }

    /// Build a projection list that matches the rule head term order.
    ///
    /// For non-aggregate rules this supports:
    /// - Variables (column passthrough)
    /// - Constants (computed constant columns)
    fn compute_head_projection(
        &self,
        head: &Atom,
        var_env: &VariableEnv,
    ) -> Result<Vec<ProjectExpr>> {
        self.compute_head_projection_with_owner(
            head,
            var_env,
            &mut LoweringAllocations { scope: None },
        )
    }

    fn compute_head_projection_with_owner(
        &self,
        head: &Atom,
        var_env: &VariableEnv,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<Vec<ProjectExpr>> {
        allocations.vector::<ProjectExpr>(head.terms.len())?;
        let mut cols = Vec::with_capacity(head.terms.len());

        for (index, term) in head.terms.iter().enumerate() {
            match term {
                Term::Variable(name) => {
                    let col = var_env
                        .get_column(name)
                        .ok_or_else(|| allocations.unsafe_variable(name))?;
                    cols.push(ProjectExpr::Column(col));
                }
                Term::Anonymous => {
                    return Err(allocations.compilation_error(format_args!(
                        "Anonymous wildcard '_' not allowed in rule head"
                    )));
                }
                Term::Aggregate(_) => {
                    return Err(allocations.compilation_error(format_args!(
                        "Aggregate term in non-aggregate rule head"
                    )));
                }
                Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {
                    let typ = self
                        .schemas
                        .get(&head.predicate)
                        .and_then(|schema| schema.column_type(index))
                        .ok_or_else(|| {
                            allocations.compilation_error(format_args!(
                                "Missing schema type for '{}' column {}",
                                head.predicate, index
                            ))
                        })?;
                    let value = term_to_typed_const_value_with_owner(term, typ, allocations)?
                        .ok_or_else(|| {
                            allocations.compilation_error(format_args!("Expected constant term"))
                        })?;
                    cols.push(ProjectExpr::Computed(Expr::Const(value), typ));
                }
                Term::List(_) | Term::Cons { .. } | Term::Compound { .. } | Term::PredRef(_) => {
                    return Err(term_not_lowerable_error(
                        "rule head projection",
                        term_kind_for_lowering_error(term),
                        allocations,
                    ));
                }
            }
        }

        Ok(cols)
    }

    /// Lower an aggregate rule head into `GroupBy` + final projection.
    fn lower_aggregate_rule(
        &mut self,
        head: &Atom,
        body: RirNode,
        var_env: &VariableEnv,
    ) -> Result<RirNode> {
        self.lower_aggregate_rule_with_owner(
            head,
            body,
            var_env,
            &mut LoweringAllocations { scope: None },
        )
    }

    fn lower_aggregate_rule_with_owner(
        &mut self,
        head: &Atom,
        body: RirNode,
        var_env: &VariableEnv,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<RirNode> {
        let arity = head.terms.len();
        allocations.vector::<String>(arity)?;
        allocations.vector::<String>(arity)?;
        allocations.vector::<(AggOp, String)>(arity)?;
        allocations.hash::<String, usize>(arity)?;
        allocations.hash::<String, usize>(arity)?;
        allocations.hash::<(AggOp, String), usize>(arity)?;
        allocations.vector::<usize>(arity)?;
        allocations.vector::<usize>(arity)?;
        for term in &head.terms {
            let extent = match term {
                Term::Variable(name) => (name.len() as u64).checked_mul(2),
                Term::Aggregate(aggregate) => (aggregate.variable.len() as u64).checked_mul(5),
                _ => Some(0),
            }
            .ok_or_else(|| XlogError::allocation_extent_overflow("aggregate variables"))?;
            allocations.acquire(extent)?;
        }
        // Collect unique group keys in head order.
        let mut key_vars: Vec<String> = Vec::new();
        let mut key_var_to_pos: HashMap<String, usize> = HashMap::new();
        let mut key_src_cols: Vec<usize> = Vec::new();

        // Collect unique aggregate specs (op, var) in head order.
        let mut agg_specs: Vec<(AggOp, String)> = Vec::new();
        let mut agg_to_pos: HashMap<(AggOp, String), usize> = HashMap::new();
        let mut value_vars: Vec<String> = Vec::new();
        let mut value_var_to_pos: HashMap<String, usize> = HashMap::new();
        let mut value_src_cols: Vec<usize> = Vec::new();

        for term in &head.terms {
            match term {
                Term::Variable(name) => {
                    if !key_var_to_pos.contains_key(name) {
                        let col = var_env
                            .get_column(name)
                            .ok_or_else(|| allocations.unsafe_variable(name))?;
                        let pos = key_vars.len();
                        key_vars.push(name.clone());
                        key_var_to_pos.insert(name.clone(), pos);
                        key_src_cols.push(col);
                    }
                }
                Term::Aggregate(agg) => {
                    let key = (agg.op, agg.variable.clone());
                    if let std::collections::hash_map::Entry::Vacant(entry) = agg_to_pos.entry(key)
                    {
                        // Ensure the aggregated variable is bound.
                        let col = var_env
                            .get_column(&agg.variable)
                            .ok_or_else(|| allocations.unsafe_variable(&agg.variable))?;

                        // Ensure the value variable exists in the groupby input.
                        let value_pos = *value_var_to_pos
                            .entry(agg.variable.clone())
                            .or_insert_with(|| {
                                let p = value_vars.len();
                                value_vars.push(agg.variable.clone());
                                value_src_cols.push(col);
                                p
                            });

                        let agg_pos = agg_specs.len();
                        agg_specs.push((agg.op, agg.variable.clone()));
                        entry.insert(agg_pos);

                        // Keep clippy happy about unused value_pos in insert_with closure.
                        let _ = value_pos;
                    }
                }
                Term::Anonymous => {
                    return Err(allocations.compilation_error(format_args!(
                        "Anonymous wildcard '_' not allowed in rule head"
                    )));
                }
                Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {
                    // Constants are allowed in the head; they are projected after aggregation.
                }
                Term::List(_) | Term::Cons { .. } | Term::Compound { .. } | Term::PredRef(_) => {
                    return Err(term_not_lowerable_error(
                        "aggregate rule head",
                        term_kind_for_lowering_error(term),
                        allocations,
                    ));
                }
            }
        }

        if agg_specs.is_empty() {
            return Err(allocations.compilation_error(format_args!(
                "Rule marked as aggregate but no aggregate terms found"
            )));
        }

        // Build groupby input: [keys..., values...]. For global aggregates (no keys),
        // synthesize a constant key column so GroupBy is well-defined.
        allocations.vector::<ProjectExpr>(
            key_src_cols
                .len()
                .max(1)
                .checked_add(value_src_cols.len())
                .ok_or_else(|| XlogError::allocation_extent_overflow("aggregate projection"))?,
        )?;
        allocations.vector::<usize>(key_src_cols.len().max(1))?;
        allocations.vector::<(usize, CoreAggOp)>(agg_specs.len())?;
        allocations.vector::<ProjectExpr>(head.terms.len())?;
        allocations.acquire(3 * std::mem::size_of::<RirNode>() as u64)?;
        let mut group_input_cols: Vec<ProjectExpr> = Vec::new();
        let mut key_cols: Vec<usize> = Vec::new();

        if key_src_cols.is_empty() {
            group_input_cols.push(ProjectExpr::Computed(
                Expr::Const(ConstValue::U32(0)),
                ScalarType::U32,
            ));
            key_cols.push(0);
        } else {
            for (i, &col) in key_src_cols.iter().enumerate() {
                group_input_cols.push(ProjectExpr::Column(col));
                key_cols.push(i);
            }
        }

        let value_offset = group_input_cols.len();
        for &col in &value_src_cols {
            group_input_cols.push(ProjectExpr::Column(col));
        }

        let group_input = RirNode::Project {
            input: Box::new(body),
            columns: group_input_cols,
        };

        // Build multi-aggregation spec list (value_col indices are in the group_input schema).
        let mut aggs: Vec<(usize, CoreAggOp)> = Vec::with_capacity(agg_specs.len());
        for (op, var) in &agg_specs {
            let value_pos = *value_var_to_pos
                .get(var)
                .ok_or_else(|| allocations.unsafe_variable(var))?;
            let value_col = value_offset + value_pos;
            aggs.push((value_col, convert_agg_op(op)));
        }

        let groupby = RirNode::GroupBy {
            input: Box::new(group_input),
            key_cols,
            aggs,
        };

        // Final projection to match head term order:
        // - variables map to group key columns
        // - aggregates map to groupby output agg columns (after keys)
        // - constants are computed columns
        let key_count = if key_src_cols.is_empty() {
            1
        } else {
            key_vars.len()
        };

        let mut final_proj: Vec<ProjectExpr> = Vec::with_capacity(head.terms.len());
        for (index, term) in head.terms.iter().enumerate() {
            match term {
                Term::Variable(name) => {
                    let idx = if key_src_cols.is_empty() {
                        // Global aggregates have no key vars in the output; binding a variable in the head
                        // is a semantic error because it would be unbound.
                        return Err(allocations.unsafe_variable(name));
                    } else {
                        *key_var_to_pos
                            .get(name)
                            .ok_or_else(|| allocations.unsafe_variable(name))?
                    };
                    final_proj.push(ProjectExpr::Column(idx));
                }
                Term::Aggregate(agg) => {
                    let pos = *agg_to_pos
                        .get(&(agg.op, agg.variable.clone()))
                        .ok_or_else(|| allocations.unsafe_variable(&agg.variable))?;
                    final_proj.push(ProjectExpr::Column(key_count + pos));
                }
                Term::Anonymous => {
                    return Err(allocations.compilation_error(format_args!(
                        "Anonymous wildcard '_' not allowed in rule head"
                    )));
                }
                Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_) => {
                    let typ = self
                        .schemas
                        .get(&head.predicate)
                        .and_then(|schema| schema.column_type(index))
                        .ok_or_else(|| {
                            allocations.compilation_error(format_args!(
                                "Missing schema type for '{}' column {}",
                                head.predicate, index
                            ))
                        })?;
                    let value = term_to_typed_const_value_with_owner(term, typ, allocations)?
                        .ok_or_else(|| {
                            allocations.compilation_error(format_args!("Expected constant term"))
                        })?;
                    final_proj.push(ProjectExpr::Computed(Expr::Const(value), typ));
                }
                Term::List(_) | Term::Cons { .. } | Term::Compound { .. } | Term::PredRef(_) => {
                    return Err(term_not_lowerable_error(
                        "aggregate rule projection",
                        term_kind_for_lowering_error(term),
                        allocations,
                    ));
                }
            }
        }

        if final_proj.is_empty() {
            return Err(allocations.compilation_error(format_args!(
                "Aggregate rule produced empty head projection"
            )));
        }

        Ok(RirNode::Project {
            input: Box::new(groupby),
            columns: final_proj,
        })
    }

    /// Infer an arithmetic result when every type needed for that result is
    /// currently known. An explicit cast fixes its result type independently of
    /// the operand, which lets schema inference propagate the target type before
    /// all upstream variable schemas have converged.
    fn infer_arith_type_from_known_variables<F>(
        expr: &ArithExpr,
        variable_type: &F,
        user_functions: UserFunctionTypeEvidence,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<Option<ScalarType>>
    where
        F: Fn(&str) -> Option<ScalarType>,
    {
        let mut tasks = vec![ArithmeticTypeTask::Visit(expr)];
        let mut values: Vec<Option<ScalarType>> = Vec::new();
        while let Some(task) = tasks.pop() {
            match task {
                ArithmeticTypeTask::Visit(expression) => match expression {
                    ArithExpr::Variable(name) => values.push(variable_type(name)),
                    ArithExpr::Integer(_) => values.push(Some(ScalarType::I64)),
                    ArithExpr::Float(_) => values.push(Some(ScalarType::F64)),
                    ArithExpr::Add(left, right)
                    | ArithExpr::Sub(left, right)
                    | ArithExpr::Mul(left, right)
                    | ArithExpr::Div(left, right) => {
                        tasks.push(ArithmeticTypeTask::FinishBinary(
                            ArithmeticTypeOperation::Standard,
                        ));
                        tasks.push(ArithmeticTypeTask::Visit(right));
                        tasks.push(ArithmeticTypeTask::Visit(left));
                    }
                    ArithExpr::Mod(left, right) => {
                        tasks.push(ArithmeticTypeTask::FinishBinary(
                            ArithmeticTypeOperation::Modulo,
                        ));
                        tasks.push(ArithmeticTypeTask::Visit(right));
                        tasks.push(ArithmeticTypeTask::Visit(left));
                    }
                    ArithExpr::Min(left, right) | ArithExpr::Max(left, right) => {
                        tasks.push(ArithmeticTypeTask::FinishBinary(
                            ArithmeticTypeOperation::MinMax,
                        ));
                        tasks.push(ArithmeticTypeTask::Visit(right));
                        tasks.push(ArithmeticTypeTask::Visit(left));
                    }
                    ArithExpr::Pow(left, right) => {
                        tasks.push(ArithmeticTypeTask::FinishBinary(
                            ArithmeticTypeOperation::Power,
                        ));
                        tasks.push(ArithmeticTypeTask::Visit(right));
                        tasks.push(ArithmeticTypeTask::Visit(left));
                    }
                    ArithExpr::Abs(inner) => {
                        tasks.push(ArithmeticTypeTask::FinishAbs);
                        tasks.push(ArithmeticTypeTask::Visit(inner));
                    }
                    ArithExpr::Cast(inner, target) => {
                        tasks.push(ArithmeticTypeTask::FinishCast(*target));
                        tasks.push(ArithmeticTypeTask::Visit(inner));
                    }
                    ArithExpr::FuncCall { name, args } => match user_functions {
                        UserFunctionTypeEvidence::RequireExpansion => {
                            return Err(allocations.compilation_error(format_args!(
                                "User-defined function '{}' must be inlined before lowering",
                                name
                            )));
                        }
                        UserFunctionTypeEvidence::Defer => {
                            tasks.push(ArithmeticTypeTask::FinishFunctionArguments(args.len()));
                            tasks.extend(args.iter().rev().map(ArithmeticTypeTask::Visit));
                        }
                    },
                    ArithExpr::Conditional {
                        then_expr,
                        else_expr,
                        ..
                    } => {
                        tasks.push(ArithmeticTypeTask::FinishConditional);
                        tasks.push(ArithmeticTypeTask::Visit(else_expr));
                        tasks.push(ArithmeticTypeTask::Visit(then_expr));
                    }
                },
                ArithmeticTypeTask::FinishBinary(operation) => {
                    let right = values.pop().expect("right arithmetic type is inferred");
                    let left = values.pop().expect("left arithmetic type is inferred");
                    values.push(Self::finish_binary_arithmetic_type(
                        operation,
                        left,
                        right,
                        allocations,
                    )?);
                }
                ArithmeticTypeTask::FinishAbs => {
                    let inferred = values.pop().expect("absolute-value type is inferred");
                    if let Some(typ) = inferred {
                        if !Self::is_numeric_type(&typ) {
                            return Err(allocations.compilation_error(format_args!(
                                "abs requires numeric type, got {:?}",
                                typ
                            )));
                        }
                    }
                    values.push(inferred);
                }
                ArithmeticTypeTask::FinishCast(target) => {
                    values.pop().expect("cast operand type is inferred");
                    values.push(Some(target));
                }
                ArithmeticTypeTask::FinishFunctionArguments(argument_count) => {
                    let start = values
                        .len()
                        .checked_sub(argument_count)
                        .expect("function argument types are inferred");
                    values.truncate(start);
                    values.push(None);
                }
                ArithmeticTypeTask::FinishConditional => {
                    let else_type = values.pop().expect("else branch type is inferred");
                    let then_type = values.pop().expect("then branch type is inferred");
                    let inferred = match (then_type, else_type) {
                        (Some(then_type), Some(else_type)) => {
                            if then_type != else_type {
                                return Err(allocations.compilation_error(format_args!(
                                    "Conditional branches have different types: {:?} vs {:?}",
                                    then_type, else_type
                                )));
                            }
                            Some(then_type)
                        }
                        _ => None,
                    };
                    values.push(inferred);
                }
            }
        }
        let inferred = values
            .pop()
            .expect("arithmetic expression produces one inferred type");
        debug_assert!(values.is_empty());
        Ok(inferred)
    }

    fn finish_binary_arithmetic_type(
        operation: ArithmeticTypeOperation,
        left: Option<ScalarType>,
        right: Option<ScalarType>,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<Option<ScalarType>> {
        let matching_numeric_type = |context: &str,
                                     allocations: &mut LoweringAllocations<'_, '_>|
         -> Result<Option<ScalarType>> {
            if let (Some(left), Some(right)) = (left, right) {
                if left != right {
                    return Err(allocations.compilation_error(format_args!(
                        "Type mismatch in {context}: {:?} vs {:?}",
                        left, right
                    )));
                }
            }
            for typ in [left, right].into_iter().flatten() {
                if !Self::is_numeric_type(&typ) {
                    return Err(allocations.compilation_error(format_args!(
                        "{context} requires numeric type, got {:?}",
                        typ
                    )));
                }
            }
            Ok(match (left, right) {
                (Some(left), Some(_)) => Some(left),
                _ => None,
            })
        };

        match operation {
            ArithmeticTypeOperation::Standard => {
                if let (Some(left), Some(right)) = (left, right) {
                    if left != right {
                        return Err(allocations.compilation_error(format_args!(
                            "Type mismatch in arithmetic: {:?} vs {:?}. Use cast() for conversion.",
                            left, right
                        )));
                    }
                }
                for typ in [left, right].into_iter().flatten() {
                    if !Self::is_numeric_type(&typ) {
                        return Err(allocations.compilation_error(format_args!(
                            "Arithmetic requires numeric type, got {:?}",
                            typ
                        )));
                    }
                }
                Ok(match (left, right) {
                    (Some(left), Some(_)) => Some(left),
                    _ => None,
                })
            }
            ArithmeticTypeOperation::Modulo => {
                if let (Some(left), Some(right)) = (left, right) {
                    if left != right {
                        return Err(allocations.compilation_error(format_args!(
                            "Type mismatch in mod: {:?} vs {:?}",
                            left, right
                        )));
                    }
                }
                for typ in [left, right].into_iter().flatten() {
                    if matches!(typ, ScalarType::F32 | ScalarType::F64) {
                        return Err(allocations.compilation_error(format_args!(
                            "Modulo (%) not supported for floating point"
                        )));
                    }
                    if !Self::is_numeric_type(&typ) {
                        return Err(allocations.compilation_error(format_args!(
                            "Modulo (%) requires integer operands, got {:?}",
                            typ
                        )));
                    }
                }
                Ok(match (left, right) {
                    (Some(left), Some(_)) => Some(left),
                    _ => None,
                })
            }
            ArithmeticTypeOperation::MinMax => matching_numeric_type("min/max", allocations),
            ArithmeticTypeOperation::Power => {
                if let Some(invalid) = [left, right]
                    .into_iter()
                    .flatten()
                    .find(|typ| !Self::is_numeric_type(typ))
                {
                    let detail = match (left, right) {
                        (Some(left), Some(right)) => {
                            allocations.formatted(format_args!("{:?} and {:?}", left, right))?;
                            format!("{:?} and {:?}", left, right)
                        }
                        _ => {
                            allocations.formatted(format_args!("{:?}", invalid))?;
                            format!("{:?}", invalid)
                        }
                    };
                    return Err(allocations.compilation_error(format_args!(
                        "pow requires numeric operands, got {detail}"
                    )));
                }
                Ok((left.is_some() && right.is_some()).then_some(ScalarType::F64))
            }
        }
    }
    /// Infer the result type of an arithmetic expression (strict same-type).
    pub(crate) fn infer_arith_type(
        &self,
        expr: &ArithExpr,
        var_env: &VariableEnv,
    ) -> Result<ScalarType> {
        self.infer_arith_type_with_owner(expr, var_env, &mut LoweringAllocations { scope: None })
    }

    fn infer_arith_type_with_owner(
        &self,
        expr: &ArithExpr,
        var_env: &VariableEnv,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<ScalarType> {
        if let Some(typ) = Self::infer_arith_type_from_known_variables(
            expr,
            &|variable| var_env.get_type(variable),
            UserFunctionTypeEvidence::RequireExpansion,
            allocations,
        )? {
            return Ok(typ);
        }

        if allocations.scope.is_some() {
            let mut count = 0usize;
            let mut invalid = false;
            expr.visit_variables(&mut |_| match count.checked_add(1) {
                Some(value) => count = value,
                None => invalid = true,
            });
            if invalid {
                return Err(XlogError::allocation_extent_overflow(
                    "arithmetic diagnostic variables",
                ));
            }
            allocations.vector::<&str>(count)?;
        }
        let variable = expr
            .variables()
            .into_iter()
            .find(|variable| var_env.get_type(variable).is_none())
            .unwrap_or("<unknown>");
        Err(allocations
            .compilation_error(format_args!("Unknown variable {} in arithmetic", variable)))
    }

    fn is_numeric_type(t: &ScalarType) -> bool {
        matches!(
            t,
            ScalarType::I32
                | ScalarType::I64
                | ScalarType::U32
                | ScalarType::U64
                | ScalarType::F32
                | ScalarType::F64
        )
    }

    /// Convert ArithExpr to IR Expr
    fn arith_to_expr(
        &self,
        arith: &ArithExpr,
        var_env: &VariableEnv,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<Expr> {
        let mut tasks = vec![ArithmeticExpressionTask::Visit(arith)];
        let mut values = Vec::new();
        while let Some(task) = tasks.pop() {
            match task {
                ArithmeticExpressionTask::Visit(expression) => match expression {
                    ArithExpr::Variable(name) => {
                        let column = var_env.get_column(name).ok_or_else(|| {
                            allocations.compilation_error(format_args!(
                                "Variable {} not bound before use in arithmetic",
                                name
                            ))
                        })?;
                        values.push(Expr::Column(column));
                    }
                    ArithExpr::Integer(value) => {
                        values.push(Expr::Const(ConstValue::I64(*value)));
                    }
                    ArithExpr::Float(value) => {
                        values.push(Expr::Const(ConstValue::F64(*value)));
                    }
                    ArithExpr::Add(left, right) => Self::schedule_arithmetic_expression(
                        &mut tasks,
                        left,
                        right,
                        ArithmeticExpressionOperation::Add,
                    ),
                    ArithExpr::Sub(left, right) => Self::schedule_arithmetic_expression(
                        &mut tasks,
                        left,
                        right,
                        ArithmeticExpressionOperation::Sub,
                    ),
                    ArithExpr::Mul(left, right) => Self::schedule_arithmetic_expression(
                        &mut tasks,
                        left,
                        right,
                        ArithmeticExpressionOperation::Mul,
                    ),
                    ArithExpr::Div(left, right) => Self::schedule_arithmetic_expression(
                        &mut tasks,
                        left,
                        right,
                        ArithmeticExpressionOperation::Div,
                    ),
                    ArithExpr::Mod(left, right) => Self::schedule_arithmetic_expression(
                        &mut tasks,
                        left,
                        right,
                        ArithmeticExpressionOperation::Mod,
                    ),
                    ArithExpr::Min(left, right) => Self::schedule_arithmetic_expression(
                        &mut tasks,
                        left,
                        right,
                        ArithmeticExpressionOperation::Min,
                    ),
                    ArithExpr::Max(left, right) => Self::schedule_arithmetic_expression(
                        &mut tasks,
                        left,
                        right,
                        ArithmeticExpressionOperation::Max,
                    ),
                    ArithExpr::Pow(left, right) => Self::schedule_arithmetic_expression(
                        &mut tasks,
                        left,
                        right,
                        ArithmeticExpressionOperation::Pow,
                    ),
                    ArithExpr::Abs(inner) => {
                        tasks.push(ArithmeticExpressionTask::FinishAbs);
                        tasks.push(ArithmeticExpressionTask::Visit(inner));
                    }
                    ArithExpr::Cast(inner, target) => {
                        tasks.push(ArithmeticExpressionTask::FinishCast(*target));
                        tasks.push(ArithmeticExpressionTask::Visit(inner));
                    }
                    ArithExpr::FuncCall { name, .. } => {
                        return Err(allocations.compilation_error(format_args!(
                            "User-defined function '{}' must be inlined before lowering",
                            name
                        )));
                    }
                    ArithExpr::Conditional {
                        cond_left,
                        cond_op,
                        cond_right,
                        then_expr,
                        else_expr,
                    } => {
                        tasks.push(ArithmeticExpressionTask::FinishConditional(*cond_op));
                        tasks.push(ArithmeticExpressionTask::Visit(else_expr));
                        tasks.push(ArithmeticExpressionTask::Visit(then_expr));
                        tasks.push(ArithmeticExpressionTask::Visit(cond_right));
                        tasks.push(ArithmeticExpressionTask::Visit(cond_left));
                    }
                },
                ArithmeticExpressionTask::FinishBinary(operation) => {
                    let right = values
                        .pop()
                        .expect("right arithmetic expression is lowered");
                    let left = values.pop().expect("left arithmetic expression is lowered");
                    values.push(match operation {
                        ArithmeticExpressionOperation::Add => {
                            Expr::Add(Box::new(left), Box::new(right))
                        }
                        ArithmeticExpressionOperation::Sub => {
                            Expr::Sub(Box::new(left), Box::new(right))
                        }
                        ArithmeticExpressionOperation::Mul => {
                            Expr::Mul(Box::new(left), Box::new(right))
                        }
                        ArithmeticExpressionOperation::Div => {
                            Expr::Div(Box::new(left), Box::new(right))
                        }
                        ArithmeticExpressionOperation::Mod => {
                            Expr::Mod(Box::new(left), Box::new(right))
                        }
                        ArithmeticExpressionOperation::Min => {
                            Expr::Min(Box::new(left), Box::new(right))
                        }
                        ArithmeticExpressionOperation::Max => {
                            Expr::Max(Box::new(left), Box::new(right))
                        }
                        ArithmeticExpressionOperation::Pow => {
                            Expr::Pow(Box::new(left), Box::new(right))
                        }
                    });
                }
                ArithmeticExpressionTask::FinishAbs => {
                    let inner = values.pop().expect("absolute-value expression is lowered");
                    values.push(Expr::Abs(Box::new(inner)));
                }
                ArithmeticExpressionTask::FinishCast(target) => {
                    let inner = values.pop().expect("cast expression is lowered");
                    values.push(Expr::Cast(Box::new(inner), target));
                }
                ArithmeticExpressionTask::FinishConditional(op) => {
                    let else_expr = values.pop().expect("else expression is lowered");
                    let then_expr = values.pop().expect("then expression is lowered");
                    let right = values.pop().expect("condition right side is lowered");
                    let left = values.pop().expect("condition left side is lowered");
                    let op = match op {
                        CompOp::Eq => CompareOp::Eq,
                        CompOp::Ne => CompareOp::Ne,
                        CompOp::Lt => CompareOp::Lt,
                        CompOp::Le => CompareOp::Le,
                        CompOp::Gt => CompareOp::Gt,
                        CompOp::Ge => CompareOp::Ge,
                    };
                    values.push(Expr::Conditional {
                        condition: Box::new(Expr::Compare {
                            left: Box::new(left),
                            op,
                            right: Box::new(right),
                        }),
                        then_expr: Box::new(then_expr),
                        else_expr: Box::new(else_expr),
                    });
                }
            }
        }
        let expression = values
            .pop()
            .expect("arithmetic expression produces one lowered value");
        debug_assert!(values.is_empty());
        Ok(expression)
    }

    fn schedule_arithmetic_expression<'a>(
        tasks: &mut Vec<ArithmeticExpressionTask<'a>>,
        left: &'a ArithExpr,
        right: &'a ArithExpr,
        operation: ArithmeticExpressionOperation,
    ) {
        tasks.push(ArithmeticExpressionTask::FinishBinary(operation));
        tasks.push(ArithmeticExpressionTask::Visit(right));
        tasks.push(ArithmeticExpressionTask::Visit(left));
    }
    /// Lower an is-expression to a Project node with computed column
    fn lower_is_expr(
        &mut self,
        is_expr: &IsExpr,
        input: RirNode,
        var_env: &mut VariableEnv,
    ) -> Result<RirNode> {
        self.lower_is_expr_with_owner(
            is_expr,
            input,
            var_env,
            &mut LoweringAllocations { scope: None },
        )
    }

    fn lower_is_expr_with_owner(
        &mut self,
        is_expr: &IsExpr,
        input: RirNode,
        var_env: &mut VariableEnv,
        allocations: &mut LoweringAllocations<'_, '_>,
    ) -> Result<RirNode> {
        allocations.arithmetic_type(&is_expr.expr)?;
        if allocations.scope.is_some() {
            let nodes = arithmetic_expression_nodes(&is_expr.expr)?;
            allocations.vector::<ArithmeticExpressionTask<'_>>(
                nodes
                    .checked_mul(2)
                    .ok_or_else(|| XlogError::allocation_extent_overflow("arithmetic IR tasks"))?,
            )?;
            allocations.vector::<Expr>(nodes)?;
            allocations.acquire(
                (nodes as u64)
                    .checked_mul(2 * std::mem::size_of::<Expr>() as u64)
                    .ok_or_else(|| XlogError::allocation_extent_overflow("arithmetic IR"))?,
            )?;
        }
        // Binding freshness and source order are validated once for the full
        // program before schema inference can preempt their diagnostics.
        let result_type = self.infer_arith_type_with_owner(&is_expr.expr, var_env, allocations)?;

        // Convert expression to IR.
        let ir_expr = self.arith_to_expr(&is_expr.expr, var_env, allocations)?;

        // Build projection: pass through all existing columns + add computed column.
        let num_cols = var_env.column_count();
        allocations.vector::<ProjectExpr>(
            num_cols
                .checked_add(1)
                .ok_or_else(|| XlogError::allocation_extent_overflow("arithmetic projection"))?,
        )?;
        allocations.acquire(std::mem::size_of::<RirNode>() as u64)?;
        let mut proj_exprs: Vec<ProjectExpr> = (0..num_cols).map(ProjectExpr::Column).collect();
        proj_exprs.push(ProjectExpr::Computed(ir_expr, result_type));

        // Bind the new variable.
        allocations.acquire(
            (is_expr.target.len() as u64)
                .checked_mul(2)
                .ok_or_else(|| XlogError::allocation_extent_overflow("arithmetic variables"))?,
        )?;
        if var_env.types.len() == var_env.types.capacity() {
            allocations.hash::<String, ScalarType>(var_env.types.len() + 1)?;
        }
        if var_env.occurrences.len() == var_env.occurrences.capacity() {
            allocations
                .hash::<String, Vec<(String, usize, usize)>>(var_env.occurrences.len() + 1)?;
        }
        allocations.vector::<(String, usize, usize)>(1)?;
        var_env.bind(&is_expr.target, num_cols, result_type);

        Ok(RirNode::Project {
            input: Box::new(input),
            columns: proj_exprs,
        })
    }
}

/// Track variable occurrences and column positions
pub(crate) struct VariableEnv {
    /// Maps variable name to list of (predicate, position in atom, global column)
    occurrences: HashMap<String, Vec<(String, usize, usize)>>,
    /// Total columns in current result
    total_cols: usize,
    /// Maps variable name to its type (for type inference)
    types: HashMap<String, ScalarType>,
}

impl VariableEnv {
    fn new() -> Self {
        Self {
            occurrences: HashMap::new(),
            total_cols: 0,
            types: HashMap::new(),
        }
    }

    fn add_occurrence(&mut self, var: &str, pred: String, atom_pos: usize, global_col: usize) {
        self.occurrences
            .entry(var.to_string())
            .or_default()
            .push((pred, atom_pos, global_col));
    }

    fn get_column(&self, var: &str) -> Option<usize> {
        self.occurrences
            .get(var)
            .and_then(|occs| occs.first())
            .map(|(_, _, col)| *col)
    }

    /// Bind a variable to a column with a specific type (for type inference)
    fn bind(&mut self, name: &str, column: usize, typ: ScalarType) {
        self.types.insert(name.to_string(), typ);
        // Also add occurrence for column lookup
        self.occurrences
            .entry(name.to_string())
            .or_default()
            .push(("".to_string(), 0, column));
        // Update total_cols to account for the new computed column
        // This is critical for chained is-expressions where each adds a column
        if column >= self.total_cols {
            self.total_cols = column + 1;
        }
    }

    /// Get the type of a bound variable
    fn get_type(&self, name: &str) -> Option<ScalarType> {
        self.types.get(name).copied()
    }

    /// Get the current column count (for adding new computed columns)
    fn column_count(&self) -> usize {
        self.total_cols
    }
}

fn sort_labels_from_terms(terms: &[Term]) -> Vec<String> {
    terms
        .iter()
        .enumerate()
        .map(|(idx, term)| match term {
            Term::Variable(name) if !name.trim().is_empty() => name.clone(),
            Term::Aggregate(agg) => format!("{:?}_{}", agg.op, agg.variable),
            Term::List(_) => format!("list{}", idx),
            Term::Cons { .. } => format!("cons{}", idx),
            Term::Compound { functor, .. } => functor.clone(),
            Term::PredRef(name) => name.clone(),
            _ => format!("c{}", idx),
        })
        .collect()
}

pub(crate) fn term_to_typed_const_value(
    term: &Term,
    expected: ScalarType,
) -> Result<Option<ConstValue>> {
    term_to_typed_const_value_with_owner(term, expected, &mut LoweringAllocations { scope: None })
}

fn term_to_typed_const_value_with_owner(
    term: &Term,
    expected: ScalarType,
    allocations: &mut LoweringAllocations<'_, '_>,
) -> Result<Option<ConstValue>> {
    term_to_typed_const_value_core(term, expected, allocations, ConstantDataAllocation::Scope)
}

fn term_to_typed_const_value_core(
    term: &Term,
    expected: ScalarType,
    allocations: &mut LoweringAllocations<'_, '_>,
    data: ConstantDataAllocation<'_>,
) -> Result<Option<ConstValue>> {
    let const_val = match term {
        Term::Integer(i) => match expected {
            ScalarType::U32 => {
                if *i >= 0 && *i <= u32::MAX as i64 {
                    ConstValue::U32(*i as u32)
                } else {
                    return Err(allocations.compilation_error(format_args!(
                        "Integer literal {} out of range for {:?}",
                        i, expected
                    )));
                }
            }
            ScalarType::U64 => {
                if *i >= 0 {
                    ConstValue::U64(*i as u64)
                } else {
                    return Err(allocations.compilation_error(format_args!(
                        "Integer literal {} out of range for {:?}",
                        i, expected
                    )));
                }
            }
            ScalarType::I32 => {
                if *i >= i32::MIN as i64 && *i <= i32::MAX as i64 {
                    ConstValue::I32(*i as i32)
                } else {
                    return Err(allocations.compilation_error(format_args!(
                        "Integer literal {} out of range for {:?}",
                        i, expected
                    )));
                }
            }
            ScalarType::I64 => ConstValue::I64(*i),
            ScalarType::F32 => {
                let value = *i as f64;
                if value < f32::MIN as f64 || value > f32::MAX as f64 {
                    return Err(allocations.compilation_error(format_args!(
                        "Integer literal {} out of range for {:?}",
                        i, expected
                    )));
                }
                ConstValue::F32(value as f32)
            }
            ScalarType::F64 => ConstValue::F64(*i as f64),
            ScalarType::Bool => {
                if *i == 0 || *i == 1 {
                    ConstValue::Bool(*i == 1)
                } else {
                    return Err(allocations.compilation_error(format_args!(
                        "Integer literal {} not valid for {:?}",
                        i, expected
                    )));
                }
            }
            ScalarType::Symbol => {
                return Err(allocations.compilation_error(format_args!(
                    "Integer literal {} not valid for {:?}",
                    i, expected
                )));
            }
        },
        Term::Float(f) => match expected {
            ScalarType::F32 => {
                if !f.is_finite() {
                    return Err(allocations.compilation_error(format_args!(
                        "Float literal {} not valid for {:?}",
                        f, expected
                    )));
                }
                if *f < f32::MIN as f64 || *f > f32::MAX as f64 {
                    return Err(allocations.compilation_error(format_args!(
                        "Float literal {} out of range for {:?}",
                        f, expected
                    )));
                }
                ConstValue::F32(*f as f32)
            }
            ScalarType::F64 => ConstValue::F64(*f),
            ScalarType::U32
            | ScalarType::U64
            | ScalarType::I32
            | ScalarType::I64
            | ScalarType::Bool
            | ScalarType::Symbol => {
                return Err(allocations.compilation_error(format_args!(
                    "Float literal {} not valid for {:?}",
                    f, expected
                )));
            }
        },
        Term::String(s) => {
            if expected == ScalarType::Symbol {
                match data {
                    ConstantDataAllocation::Scope => allocations.acquire(s.len() as u64)?,
                    ConstantDataAllocation::Prepared(_receipt) => {}
                }
                ConstValue::Symbol(s.clone())
            } else {
                return Err(allocations.compilation_error(format_args!(
                    "String literal {} not valid for {:?}",
                    s, expected
                )));
            }
        }
        Term::Symbol(id) => {
            let value = match data {
                ConstantDataAllocation::Scope => allocations.symbol(*id)?,
                // This private borrow names the actual leaf/atom allocation
                // already acquired above; it creates no replacement receipt.
                ConstantDataAllocation::Prepared(_receipt) => symbol::resolve(*id),
            };
            match expected {
                ScalarType::Symbol => ConstValue::Symbol(value),
                ScalarType::Bool if matches!(value.as_str(), "true" | "false") => {
                    ConstValue::Bool(value == "true")
                }
                _ => {
                    return Err(allocations.compilation_error(format_args!(
                        "Symbol literal {} not valid for {:?}",
                        value, expected
                    )));
                }
            }
        }
        Term::Variable(_)
        | Term::Anonymous
        | Term::Aggregate(_)
        | Term::List(_)
        | Term::Cons { .. }
        | Term::Compound { .. }
        | Term::PredRef(_) => return Ok(None),
    };

    Ok(Some(const_val))
}

/// Convert AST AggOp to core AggOp
fn convert_agg_op(op: &AggOp) -> CoreAggOp {
    match op {
        AggOp::Count => CoreAggOp::Count,
        AggOp::Sum => CoreAggOp::Sum,
        AggOp::Min => CoreAggOp::Min,
        AggOp::Max => CoreAggOp::Max,
        AggOp::LogSumExp => CoreAggOp::LogSumExp,
    }
}

// Export the find_sccs_for_lowering function from stratify
// We need to add this to the stratify module

#[cfg(test)]
mod arith_type_tests {
    use super::*;
    use crate::ast::ArithExpr;

    #[test]
    fn test_arith_type_inference_same_type() {
        // X + Y where both are i64 should succeed and return i64
        let lowerer = Lowerer::new();
        let mut var_env = VariableEnv::new();
        var_env.bind("X", 0, ScalarType::I64);
        var_env.bind("Y", 1, ScalarType::I64);

        let expr = ArithExpr::Add(
            Box::new(ArithExpr::Variable("X".to_string())),
            Box::new(ArithExpr::Variable("Y".to_string())),
        );
        let result = lowerer.infer_arith_type(&expr, &var_env);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ScalarType::I64);
    }

    #[test]
    fn test_arith_type_inference_mismatch() {
        // X + Y where X is i64 and Y is f64 should fail
        let lowerer = Lowerer::new();
        let mut var_env = VariableEnv::new();
        var_env.bind("X", 0, ScalarType::I64);
        var_env.bind("Y", 1, ScalarType::F64);

        let expr = ArithExpr::Add(
            Box::new(ArithExpr::Variable("X".to_string())),
            Box::new(ArithExpr::Variable("Y".to_string())),
        );
        let result = lowerer.infer_arith_type(&expr, &var_env);
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;

    fn pred_decl(name: &str, types: Vec<ScalarType>) -> PredDecl {
        let type_refs: Vec<TypeRef> = types.into_iter().map(TypeRef::Scalar).collect();
        let columns = type_refs
            .iter()
            .cloned()
            .map(|typ| PredColumn { name: None, typ })
            .collect();
        PredDecl {
            name: name.to_string(),
            types: type_refs,
            columns,
            is_private: false,
        }
    }

    /// Helper to create a simple edge atom
    fn edge_atom(x: &str, y: &str) -> Atom {
        Atom {
            predicate: "edge".to_string(),
            terms: vec![Term::Variable(x.to_string()), Term::Variable(y.to_string())],
        }
    }

    /// Helper to create a reach atom
    fn reach_atom(x: &str, y: &str) -> Atom {
        Atom {
            predicate: "reach".to_string(),
            terms: vec![Term::Variable(x.to_string()), Term::Variable(y.to_string())],
        }
    }

    /// Helper to create a node atom
    fn node_atom(x: &str) -> Atom {
        Atom {
            predicate: "node".to_string(),
            terms: vec![Term::Variable(x.to_string())],
        }
    }

    #[test]
    fn test_lowerer_new() {
        let lowerer = Lowerer::new();
        assert!(lowerer.schemas.is_empty());
        assert!(lowerer.strata.is_empty());
        assert_eq!(lowerer.next_rel_id, 0);
    }

    #[test]
    fn test_get_or_create_rel_id() {
        let mut lowerer = Lowerer::new();
        let id1 = lowerer.get_or_create_rel_id("edge");
        let id2 = lowerer.get_or_create_rel_id("reach");
        let id3 = lowerer.get_or_create_rel_id("edge");

        assert_eq!(id1, RelId(0));
        assert_eq!(id2, RelId(1));
        assert_eq!(id3, RelId(0)); // Same as id1
    }

    #[test]
    fn test_infer_schemas_from_facts() {
        let mut program = Program::new();
        program.rules.push(Rule {
            head: Atom {
                predicate: "edge".to_string(),
                terms: vec![Term::Integer(1), Term::Integer(2)],
            },
            body: vec![],
        });

        let mut lowerer = Lowerer::new();
        lowerer.infer_schemas(&program).unwrap();

        assert!(lowerer.schemas.contains_key("edge"));
        let schema = lowerer.schemas.get("edge").unwrap();
        assert_eq!(schema.arity(), 2);
    }

    #[test]
    fn test_infer_schemas_propagates_through_reversed_rule_chains() {
        let mut program = Program::new();
        program.rules.push(Rule {
            head: Atom {
                predicate: "shared".to_string(),
                terms: vec![Term::Variable("X".to_string())],
            },
            body: vec![BodyLiteral::Positive(Atom {
                predicate: "intermediate".to_string(),
                terms: vec![Term::Variable("X".to_string())],
            })],
        });
        program.rules.push(Rule {
            head: Atom {
                predicate: "intermediate".to_string(),
                terms: vec![Term::Variable("X".to_string())],
            },
            body: vec![BodyLiteral::Positive(Atom {
                predicate: "source".to_string(),
                terms: vec![Term::Variable("X".to_string())],
            })],
        });
        program.rules.push(Rule {
            head: Atom {
                predicate: "source".to_string(),
                terms: vec![Term::Symbol(symbol::intern("value"))],
            },
            body: vec![],
        });

        let mut lowerer = Lowerer::new();
        lowerer.infer_schemas(&program).unwrap();

        assert_eq!(
            lowerer
                .schemas
                .get("shared")
                .and_then(|schema| schema.column_type(0)),
            Some(ScalarType::Symbol)
        );
    }

    #[test]
    fn test_lower_simple_rule() {
        // reach(X, Y) :- edge(X, Y).
        let rule = Rule {
            head: reach_atom("X", "Y"),
            body: vec![BodyLiteral::Positive(edge_atom("X", "Y"))],
        };

        let mut lowerer = Lowerer::new();
        lowerer.schemas.insert(
            "edge".to_string(),
            Schema::new(vec![
                ("c0".to_string(), ScalarType::U32),
                ("c1".to_string(), ScalarType::U32),
            ]),
        );

        let result = lowerer.lower_rule(&rule);
        assert!(result.is_ok());

        let node = result.unwrap();
        // Should be just a scan (no projection needed since columns match)
        assert!(matches!(node, RirNode::Scan { .. }));
    }

    #[test]
    fn test_lower_join_rule() {
        // reach(X, Z) :- reach(X, Y), edge(Y, Z).
        let rule = Rule {
            head: Atom {
                predicate: "reach".to_string(),
                terms: vec![
                    Term::Variable("X".to_string()),
                    Term::Variable("Z".to_string()),
                ],
            },
            body: vec![
                BodyLiteral::Positive(reach_atom("X", "Y")),
                BodyLiteral::Positive(edge_atom("Y", "Z")),
            ],
        };

        let mut lowerer = Lowerer::new();
        lowerer.schemas.insert(
            "reach".to_string(),
            Schema::new(vec![
                ("c0".to_string(), ScalarType::U32),
                ("c1".to_string(), ScalarType::U32),
            ]),
        );
        lowerer.schemas.insert(
            "edge".to_string(),
            Schema::new(vec![
                ("c0".to_string(), ScalarType::U32),
                ("c1".to_string(), ScalarType::U32),
            ]),
        );

        let result = lowerer.lower_rule(&rule);
        assert!(result.is_ok());

        let node = result.unwrap();
        // Should be Project(Join(Scan, Scan))
        if let RirNode::Project { input, columns } = node {
            // X from reach (col 0), Z from edge (col 3)
            assert_eq!(
                columns,
                vec![ProjectExpr::Column(0), ProjectExpr::Column(3)]
            );
            assert!(matches!(*input, RirNode::Join { .. }));
            if let RirNode::Join {
                left_keys,
                right_keys,
                ..
            } = *input
            {
                assert_eq!(left_keys, vec![1]); // Y in reach (position 1)
                assert_eq!(right_keys, vec![0]); // Y in edge (position 0)
            }
        } else {
            panic!("Expected Project node");
        }
    }

    #[test]
    fn test_join_order_prefers_smaller_relation() {
        // out(X) :- big(X), small(X).
        let rule = Rule {
            head: Atom {
                predicate: "out".to_string(),
                terms: vec![Term::Variable("X".to_string())],
            },
            body: vec![
                BodyLiteral::Positive(Atom {
                    predicate: "big".to_string(),
                    terms: vec![Term::Variable("X".to_string())],
                }),
                BodyLiteral::Positive(Atom {
                    predicate: "small".to_string(),
                    terms: vec![Term::Variable("X".to_string())],
                }),
            ],
        };

        let mut lowerer = Lowerer::new();
        lowerer.schemas.insert(
            "big".to_string(),
            Schema::new(vec![("c0".to_string(), ScalarType::U32)]),
        );
        lowerer.schemas.insert(
            "small".to_string(),
            Schema::new(vec![("c0".to_string(), ScalarType::U32)]),
        );

        // Ensure stable RelIds independent of join order.
        let big_id = lowerer.get_or_create_rel_id("big");
        let small_id = lowerer.get_or_create_rel_id("small");
        assert_eq!(big_id, RelId(0));
        assert_eq!(small_id, RelId(1));

        // Prefer scanning the smaller relation first.
        lowerer.est_cardinality.insert("big".to_string(), 10_000);
        lowerer.est_cardinality.insert("small".to_string(), 10);

        let node = lowerer.lower_rule(&rule).unwrap();
        let join = match node {
            RirNode::Project { input, .. } => *input,
            other => other,
        };

        match join {
            RirNode::Join { left, right, .. } => {
                // Prefer building the hash table on the smaller relation (right/build side).
                assert!(matches!(*left, RirNode::Scan { rel } if rel == big_id));
                assert!(matches!(*right, RirNode::Scan { rel } if rel == small_id));
            }
            other => panic!("Expected Join node, got {:?}", other),
        }
    }

    #[test]
    fn test_lower_negation() {
        // isolated(X) :- node(X), not edge(X, _).
        let rule = Rule {
            head: Atom {
                predicate: "isolated".to_string(),
                terms: vec![Term::Variable("X".to_string())],
            },
            body: vec![
                BodyLiteral::Positive(node_atom("X")),
                BodyLiteral::Negated(Atom {
                    predicate: "edge".to_string(),
                    terms: vec![
                        Term::Variable("X".to_string()),
                        Term::Variable("_".to_string()),
                    ],
                }),
            ],
        };

        let mut lowerer = Lowerer::new();
        lowerer.schemas.insert(
            "node".to_string(),
            Schema::new(vec![("c0".to_string(), ScalarType::U32)]),
        );
        lowerer.schemas.insert(
            "edge".to_string(),
            Schema::new(vec![
                ("c0".to_string(), ScalarType::U32),
                ("c1".to_string(), ScalarType::U32),
            ]),
        );

        let result = lowerer.lower_rule(&rule);
        assert!(result.is_ok());

        // The result should involve a Diff or semi-join for negation
        let node = result.unwrap();
        // Verify the structure contains the negation handling
        fn contains_diff_or_semi(node: &RirNode) -> bool {
            match node {
                RirNode::Diff { .. } => true,
                RirNode::Join {
                    join_type: JoinType::Semi,
                    ..
                } => true,
                RirNode::Join { left, right, .. } => {
                    contains_diff_or_semi(left) || contains_diff_or_semi(right)
                }
                RirNode::Project { input, .. } => contains_diff_or_semi(input),
                RirNode::Filter { input, .. } => contains_diff_or_semi(input),
                _ => false,
            }
        }
        assert!(contains_diff_or_semi(&node));
    }

    #[test]
    fn test_lower_ground_negation_preserves_the_positive_input_schema() {
        // ok(X) :- x(X), not p(3).
        let rule = Rule {
            head: Atom {
                predicate: "ok".to_string(),
                terms: vec![Term::Variable("X".to_string())],
            },
            body: vec![
                BodyLiteral::Positive(Atom {
                    predicate: "x".to_string(),
                    terms: vec![Term::Variable("X".to_string())],
                }),
                BodyLiteral::Negated(Atom {
                    predicate: "p".to_string(),
                    terms: vec![Term::Integer(3)],
                }),
            ],
        };

        let mut lowerer = Lowerer::new();
        for predicate in ["ok", "x", "p"] {
            lowerer.schemas.insert(
                predicate.to_string(),
                Schema::new(vec![("c0".to_string(), ScalarType::U32)]),
            );
        }

        let node = lowerer.lower_rule(&rule).expect("lower ground negation");

        let RirNode::Project { input, columns } = node else {
            panic!("ground negation must project away its internal existence key");
        };
        assert_eq!(columns, vec![ProjectExpr::Column(0)]);
        assert!(matches!(
            *input,
            RirNode::Join {
                left_keys,
                right_keys,
                join_type: JoinType::Anti,
                ..
            } if left_keys == vec![1] && right_keys == vec![0]
        ));
    }

    #[test]
    fn test_lower_comparison() {
        // greater(X, Y) :- pair(X, Y), X > Y.
        let rule = Rule {
            head: Atom {
                predicate: "greater".to_string(),
                terms: vec![
                    Term::Variable("X".to_string()),
                    Term::Variable("Y".to_string()),
                ],
            },
            body: vec![
                BodyLiteral::Positive(Atom {
                    predicate: "pair".to_string(),
                    terms: vec![
                        Term::Variable("X".to_string()),
                        Term::Variable("Y".to_string()),
                    ],
                }),
                BodyLiteral::Comparison(Comparison {
                    left: Term::Variable("X".to_string()),
                    op: CompOp::Gt,
                    right: Term::Variable("Y".to_string()),
                }),
            ],
        };

        let mut lowerer = Lowerer::new();
        lowerer.schemas.insert(
            "pair".to_string(),
            Schema::new(vec![
                ("c0".to_string(), ScalarType::U32),
                ("c1".to_string(), ScalarType::U32),
            ]),
        );

        let result = lowerer.lower_rule(&rule);
        assert!(result.is_ok());

        let node = result.unwrap();
        // Should contain a Filter node
        fn contains_filter(node: &RirNode) -> bool {
            match node {
                RirNode::Filter { .. } => true,
                RirNode::Project { input, .. } => contains_filter(input),
                RirNode::Join { left, right, .. } => {
                    contains_filter(left) || contains_filter(right)
                }
                _ => false,
            }
        }
        assert!(contains_filter(&node));
    }

    #[test]
    fn test_lower_constant_filter() {
        // specific_edge(Y) :- edge(1, Y).
        let rule = Rule {
            head: Atom {
                predicate: "specific_edge".to_string(),
                terms: vec![Term::Variable("Y".to_string())],
            },
            body: vec![BodyLiteral::Positive(Atom {
                predicate: "edge".to_string(),
                terms: vec![Term::Integer(1), Term::Variable("Y".to_string())],
            })],
        };

        let mut lowerer = Lowerer::new();
        lowerer.schemas.insert(
            "edge".to_string(),
            Schema::new(vec![
                ("c0".to_string(), ScalarType::U32),
                ("c1".to_string(), ScalarType::U32),
            ]),
        );

        let result = lowerer.lower_rule(&rule);
        assert!(result.is_ok());

        let node = result.unwrap();
        // Should contain a Filter for the constant 1
        fn has_const_filter(node: &RirNode) -> bool {
            match node {
                RirNode::Filter {
                    predicate: Expr::Compare { right, .. },
                    ..
                } => matches!(**right, Expr::Const(_)),
                RirNode::Project { input, .. } => has_const_filter(input),
                _ => false,
            }
        }
        assert!(has_const_filter(&node));
    }

    #[test]
    fn test_lower_repeated_variable_filter() {
        // self_loop(X) :- edge(X, X).
        let rule = Rule {
            head: Atom {
                predicate: "self_loop".to_string(),
                terms: vec![Term::Variable("X".to_string())],
            },
            body: vec![BodyLiteral::Positive(Atom {
                predicate: "edge".to_string(),
                terms: vec![
                    Term::Variable("X".to_string()),
                    Term::Variable("X".to_string()),
                ],
            })],
        };

        let mut lowerer = Lowerer::new();
        lowerer.schemas.insert(
            "edge".to_string(),
            Schema::new(vec![
                ("c0".to_string(), ScalarType::U32),
                ("c1".to_string(), ScalarType::U32),
            ]),
        );

        let node = lowerer.lower_rule(&rule).expect("lower_rule failed");

        fn has_col_eq_filter(node: &RirNode) -> bool {
            match node {
                RirNode::Filter { predicate, .. } => match predicate {
                    Expr::Compare {
                        left,
                        op: CompareOp::Eq,
                        right,
                    } => {
                        matches!((&**left, &**right), (Expr::Column(0), Expr::Column(1)))
                            || matches!((&**left, &**right), (Expr::Column(1), Expr::Column(0)))
                    }
                    Expr::And(exprs) => exprs.iter().any(|e| match e {
                        Expr::Compare {
                            left,
                            op: CompareOp::Eq,
                            right,
                        } => {
                            matches!((&**left, &**right), (Expr::Column(0), Expr::Column(1)))
                                || matches!((&**left, &**right), (Expr::Column(1), Expr::Column(0)))
                        }
                        _ => false,
                    }),
                    _ => false,
                },
                RirNode::Project { input, .. } => has_col_eq_filter(input),
                _ => false,
            }
        }

        assert!(has_col_eq_filter(&node));
    }

    #[test]
    fn test_lower_program_simple() {
        let mut program = Program::new();

        // edge(1, 2).
        program.rules.push(Rule {
            head: Atom {
                predicate: "edge".to_string(),
                terms: vec![Term::Integer(1), Term::Integer(2)],
            },
            body: vec![],
        });

        // reach(X, Y) :- edge(X, Y).
        program.rules.push(Rule {
            head: reach_atom("X", "Y"),
            body: vec![BodyLiteral::Positive(edge_atom("X", "Y"))],
        });

        let mut lowerer = Lowerer::new();
        lowerer.set_strata(vec![vec!["edge".to_string()], vec!["reach".to_string()]]);

        let result = lowerer.lower_program(&program);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(!plan.sccs.is_empty());
    }

    #[test]
    fn facts_are_metadata_and_not_executable_rules() {
        let mut program = Program::new();
        for value in [1, 2, 2] {
            program.rules.push(Rule {
                head: Atom {
                    predicate: "base".to_string(),
                    terms: vec![Term::Integer(value)],
                },
                body: vec![],
            });
        }

        let mut lowerer = Lowerer::new();
        lowerer.set_strata(vec![vec!["base".to_string()]]);

        let plan = lowerer.lower_program(&program).unwrap();

        assert_eq!(
            plan.rules_by_scc.iter().map(Vec::len).sum::<usize>(),
            0,
            "facts are materialized by the relation loader, not executed as rules"
        );
        assert_eq!(lowerer.schemas.get("base").unwrap().arity(), 1);
        assert_eq!(lowerer.est_cardinality.get("base"), Some(&3));
        assert!(lowerer
            .sccs
            .iter()
            .any(|scc| scc.predicates.iter().any(|predicate| predicate == "base")));

        let base_id = lowerer.rel_ids().get("base").copied().unwrap();
        assert_eq!(plan.rel_arities.get(&base_id), Some(&1));
    }

    #[test]
    fn test_variable_env() {
        let mut env = VariableEnv::new();
        env.add_occurrence("X", "edge".to_string(), 0, 0);
        env.add_occurrence("Y", "edge".to_string(), 1, 1);
        env.add_occurrence("Y", "node".to_string(), 0, 2);

        assert_eq!(env.get_column("X"), Some(0));
        assert_eq!(env.get_column("Y"), Some(1)); // First occurrence
        assert_eq!(env.get_column("Z"), None);
    }

    #[test]
    fn test_inferred_scalar_type() {
        assert_eq!(
            Term::Variable("X".to_string()).inferred_scalar_type(),
            ScalarType::U64
        );
        assert_eq!(Term::Integer(42).inferred_scalar_type(), ScalarType::U32);
        assert_eq!(
            Term::Integer(i64::MAX).inferred_scalar_type(),
            ScalarType::I64
        );
        assert_eq!(Term::Float(3.25).inferred_scalar_type(), ScalarType::F64);
        assert_eq!(
            Term::Symbol(symbol::intern("foo")).inferred_scalar_type(),
            ScalarType::Symbol
        );
    }

    #[test]
    fn test_convert_agg_op() {
        assert_eq!(convert_agg_op(&AggOp::Count), CoreAggOp::Count);
        assert_eq!(convert_agg_op(&AggOp::Sum), CoreAggOp::Sum);
        assert_eq!(convert_agg_op(&AggOp::Min), CoreAggOp::Min);
        assert_eq!(convert_agg_op(&AggOp::Max), CoreAggOp::Max);
        assert_eq!(convert_agg_op(&AggOp::LogSumExp), CoreAggOp::LogSumExp);
    }

    #[test]
    fn test_variable_env_bind_updates_total_cols() {
        // Test that bind() properly updates total_cols for chained is-expressions
        let mut env = VariableEnv::new();
        env.total_cols = 2; // Simulate 2 columns from atoms

        // Bind first computed variable at column 2
        env.bind("A", 2, ScalarType::I64);
        assert_eq!(
            env.column_count(),
            3,
            "total_cols should be 3 after first bind"
        );
        assert_eq!(env.get_column("A"), Some(2));

        // Bind second computed variable at column 3
        env.bind("B", 3, ScalarType::I64);
        assert_eq!(
            env.column_count(),
            4,
            "total_cols should be 4 after second bind"
        );
        assert_eq!(env.get_column("B"), Some(3));
    }

    #[test]
    fn test_lower_chained_is_expressions() {
        // result(A, B) :- input(X, Y), A is X + Y, B is A * 2.
        // This tests that chained is-expressions correctly update column indices
        let rule = Rule {
            head: Atom {
                predicate: "result".to_string(),
                terms: vec![
                    Term::Variable("A".to_string()),
                    Term::Variable("B".to_string()),
                ],
            },
            body: vec![
                BodyLiteral::Positive(Atom {
                    predicate: "input".to_string(),
                    terms: vec![
                        Term::Variable("X".to_string()),
                        Term::Variable("Y".to_string()),
                    ],
                }),
                BodyLiteral::IsExpr(IsExpr {
                    target: "A".to_string(),
                    expr: ArithExpr::Add(
                        Box::new(ArithExpr::Variable("X".to_string())),
                        Box::new(ArithExpr::Variable("Y".to_string())),
                    ),
                }),
                BodyLiteral::IsExpr(IsExpr {
                    target: "B".to_string(),
                    expr: ArithExpr::Mul(
                        Box::new(ArithExpr::Variable("A".to_string())),
                        Box::new(ArithExpr::Integer(2)),
                    ),
                }),
            ],
        };

        let mut lowerer = Lowerer::new();
        lowerer.schemas.insert(
            "input".to_string(),
            Schema::new(vec![
                ("c0".to_string(), ScalarType::I64),
                ("c1".to_string(), ScalarType::I64),
            ]),
        );

        let result = lowerer.lower_rule(&rule);
        assert!(
            result.is_ok(),
            "Lowering chained is-expressions should succeed: {:?}",
            result.err()
        );

        let node = result.unwrap();

        // The structure should be:
        // Project([col 2, col 3]) <-- final projection for A, B
        //   Project([col 0, col 1, col 2, A*2]) <-- second is-expr adds B at col 3
        //     Project([col 0, col 1, X+Y]) <-- first is-expr adds A at col 2
        //       Scan(input)

        // Verify we have nested Project nodes
        fn count_projects(node: &RirNode) -> usize {
            match node {
                RirNode::Project { input, .. } => 1 + count_projects(input),
                _ => 0,
            }
        }

        // We expect 3 Project nodes: 2 for is-expressions + 1 for final head projection
        let project_count = count_projects(&node);
        assert!(
            project_count >= 2,
            "Expected at least 2 Project nodes for chained is-exprs, got {}",
            project_count
        );

        // Verify the final projection references columns 2 and 3 (A and B)
        if let RirNode::Project { columns, .. } = &node {
            assert_eq!(columns.len(), 2, "Head has 2 variables");
            // A should be at column 2, B at column 3
            assert_eq!(columns[0], ProjectExpr::Column(2), "A should be column 2");
            assert_eq!(columns[1], ProjectExpr::Column(3), "B should be column 3");
        } else {
            panic!("Expected top-level Project node");
        }
    }

    #[test]
    fn test_u64_comparison_type_from_pred_decl() {
        // Test that u64 type from pred decl is preserved in comparison lowering
        let mut program = Program::new();

        // pred count_data(symbol, u64).
        program.predicates.push(pred_decl(
            "count_data",
            vec![ScalarType::Symbol, ScalarType::U64],
        ));

        // count_data(alice, 5).
        program.rules.push(Rule {
            head: Atom {
                predicate: "count_data".to_string(),
                terms: vec![
                    Term::Symbol(xlog_core::symbol::intern("alice")),
                    Term::Integer(5),
                ],
            },
            body: vec![],
        });

        // pred big_count(symbol, u64).
        program.predicates.push(pred_decl(
            "big_count",
            vec![ScalarType::Symbol, ScalarType::U64],
        ));

        // big_count(Name, Count) :- count_data(Name, Count), Count >= 3.
        program.rules.push(Rule {
            head: Atom {
                predicate: "big_count".to_string(),
                terms: vec![
                    Term::Variable("Name".to_string()),
                    Term::Variable("Count".to_string()),
                ],
            },
            body: vec![
                BodyLiteral::Positive(Atom {
                    predicate: "count_data".to_string(),
                    terms: vec![
                        Term::Variable("Name".to_string()),
                        Term::Variable("Count".to_string()),
                    ],
                }),
                BodyLiteral::Comparison(Comparison {
                    left: Term::Variable("Count".to_string()),
                    op: CompOp::Ge,
                    right: Term::Integer(3),
                }),
            ],
        });

        let mut lowerer = Lowerer::new();
        lowerer.infer_schemas(&program).unwrap();

        // Verify schema has correct types
        let schema = lowerer
            .schemas
            .get("count_data")
            .expect("schema for count_data");
        assert_eq!(
            schema.column_type(0),
            Some(ScalarType::Symbol),
            "First column should be Symbol"
        );
        assert_eq!(
            schema.column_type(1),
            Some(ScalarType::U64),
            "Second column should be U64"
        );

        // Now test lowering the rule with comparison
        lowerer.set_strata(vec![
            vec!["count_data".to_string()],
            vec!["big_count".to_string()],
        ]);
        lowerer.build_sccs(&program);

        let rule = &program.rules[1]; // big_count rule
        let result = lowerer.lower_rule(rule);
        assert!(
            result.is_ok(),
            "Lowering should succeed: {:?}",
            result.err()
        );

        // Check that the filter has the correct constant type
        fn find_compare_const(node: &RirNode) -> Option<&ConstValue> {
            match node {
                RirNode::Filter { predicate, input } => {
                    if let Expr::Compare { right, .. } = predicate {
                        if let Expr::Const(val) = right.as_ref() {
                            return Some(val);
                        }
                    }
                    find_compare_const(input)
                }
                RirNode::Project { input, .. } => find_compare_const(input),
                RirNode::Join { left, right, .. } => {
                    find_compare_const(left).or_else(|| find_compare_const(right))
                }
                _ => None,
            }
        }

        let node = result.unwrap();
        let const_val = find_compare_const(&node);
        assert!(const_val.is_some(), "Should find a constant in comparison");

        // The constant should be U64(3), not I64(3)
        match const_val.unwrap() {
            ConstValue::U64(v) => assert_eq!(*v, 3, "Value should be 3"),
            other => panic!("Expected U64(3), got {:?}", other),
        }
    }

    #[test]
    fn test_u64_comparison_with_aggregation() {
        use crate::ast::AggExpr;

        // Test aggregation + comparison case
        let mut program = Program::new();

        // pred reports_to(symbol, symbol).
        program.predicates.push(pred_decl(
            "reports_to",
            vec![ScalarType::Symbol, ScalarType::Symbol],
        ));

        // reports_to facts
        program.rules.push(Rule {
            head: Atom {
                predicate: "reports_to".to_string(),
                terms: vec![
                    Term::Symbol(xlog_core::symbol::intern("alice")),
                    Term::Symbol(xlog_core::symbol::intern("bob")),
                ],
            },
            body: vec![],
        });
        program.rules.push(Rule {
            head: Atom {
                predicate: "reports_to".to_string(),
                terms: vec![
                    Term::Symbol(xlog_core::symbol::intern("carol")),
                    Term::Symbol(xlog_core::symbol::intern("bob")),
                ],
            },
            body: vec![],
        });

        // pred direct_count(symbol, u64).
        program.predicates.push(pred_decl(
            "direct_count",
            vec![ScalarType::Symbol, ScalarType::U64],
        ));

        // direct_count(Mgr, count(Emp)) :- reports_to(Emp, Mgr).
        program.rules.push(Rule {
            head: Atom {
                predicate: "direct_count".to_string(),
                terms: vec![
                    Term::Variable("Mgr".to_string()),
                    Term::Aggregate(AggExpr {
                        op: AggOp::Count,
                        variable: "Emp".to_string(),
                    }),
                ],
            },
            body: vec![BodyLiteral::Positive(Atom {
                predicate: "reports_to".to_string(),
                terms: vec![
                    Term::Variable("Emp".to_string()),
                    Term::Variable("Mgr".to_string()),
                ],
            })],
        });

        // pred big_manager(symbol, u64).
        program.predicates.push(pred_decl(
            "big_manager",
            vec![ScalarType::Symbol, ScalarType::U64],
        ));

        // big_manager(Mgr, Count) :- direct_count(Mgr, Count), Count >= 2.
        program.rules.push(Rule {
            head: Atom {
                predicate: "big_manager".to_string(),
                terms: vec![
                    Term::Variable("Mgr".to_string()),
                    Term::Variable("Count".to_string()),
                ],
            },
            body: vec![
                BodyLiteral::Positive(Atom {
                    predicate: "direct_count".to_string(),
                    terms: vec![
                        Term::Variable("Mgr".to_string()),
                        Term::Variable("Count".to_string()),
                    ],
                }),
                BodyLiteral::Comparison(Comparison {
                    left: Term::Variable("Count".to_string()),
                    op: CompOp::Ge,
                    right: Term::Integer(2),
                }),
            ],
        });

        let mut lowerer = Lowerer::new();
        lowerer.infer_schemas(&program).unwrap();

        // Verify schema has correct types
        let schema = lowerer
            .schemas
            .get("direct_count")
            .expect("schema for direct_count");
        assert_eq!(
            schema.column_type(0),
            Some(ScalarType::Symbol),
            "First column should be Symbol"
        );
        assert_eq!(
            schema.column_type(1),
            Some(ScalarType::U64),
            "Second column should be U64"
        );

        lowerer.set_strata(vec![
            vec!["reports_to".to_string()],
            vec!["direct_count".to_string()],
            vec!["big_manager".to_string()],
        ]);
        lowerer.build_sccs(&program);

        // Lower the big_manager rule (index 3: after 2 facts + aggregation rule)
        let big_manager_rule = &program.rules[3];
        let result = lowerer.lower_rule(big_manager_rule);
        assert!(
            result.is_ok(),
            "Lowering should succeed: {:?}",
            result.err()
        );

        // Check that the filter has the correct constant type
        fn find_compare_const(node: &RirNode) -> Option<&ConstValue> {
            match node {
                RirNode::Filter { predicate, input } => {
                    if let Expr::Compare { right, .. } = predicate {
                        if let Expr::Const(val) = right.as_ref() {
                            return Some(val);
                        }
                    }
                    find_compare_const(input)
                }
                RirNode::Project { input, .. } => find_compare_const(input),
                RirNode::Join { left, right, .. } => {
                    find_compare_const(left).or_else(|| find_compare_const(right))
                }
                _ => None,
            }
        }

        let node = result.unwrap();
        let const_val = find_compare_const(&node);
        assert!(const_val.is_some(), "Should find a constant in comparison");

        // The constant should be U64(2), not I64(2)
        match const_val.unwrap() {
            ConstValue::U64(v) => assert_eq!(*v, 2, "Value should be 2"),
            other => panic!("Expected U64(2), got {:?}", other),
        }
    }

    #[test]
    fn declared_head_constant_is_projected_with_the_schema_type() {
        let program = crate::parse_program(
            r#"
            pred real(f64).
            seed().
            real(1) :- seed().
        "#,
        )
        .expect("parse typed head-constant fixture");
        let mut lowerer = Lowerer::new();
        lowerer
            .infer_schemas(&program)
            .expect("infer declared schemas");
        lowerer
            .validate_rule_types(&program)
            .expect("validate supported literal conversion");

        let node = lowerer
            .lower_rule(&program.rules[1])
            .expect("lower typed head constant");
        let RirNode::Project { columns, .. } = node else {
            panic!("constant rule head should lower to a projection");
        };
        assert_eq!(
            columns,
            vec![ProjectExpr::Computed(
                Expr::Const(ConstValue::F64(1.0)),
                ScalarType::F64,
            )]
        );
    }
}

#[cfg(test)]
mod cross_predicate_type_tests {
    use super::*;

    // Regression for the paper's deliberately ill-typed example: the head
    // declaration constrains bridge column 0 to symbol while the body
    // constrains the same variable to u32 through connected/node. The
    // mismatch must surface as a compilation error naming the predicate
    // and argument positions, never as a kernel schema dump.
    const ILL_TYPED_BRIDGE: &str = r#"
pred node(u32, symbol).
pred connected(u32, u32).
pred bridge(symbol, u32).

node(1, "alice").
connected(1, 2).

bridge(A, B) :- connected(A, B), node(A, _).

?- bridge(W, X).
"#;

    #[test]
    fn cross_predicate_schema_mismatch_is_a_compilation_error() {
        let program = crate::parse_program(ILL_TYPED_BRIDGE).expect("example parses");
        let mut lowerer = Lowerer::new();
        let err = lowerer
            .lower_program(&program)
            .expect_err("ill-typed program must not lower");
        let msg = err.to_string();
        assert!(
            matches!(err, xlog_core::XlogError::Compilation(_)),
            "expected a compilation error, got: {msg}"
        );
        assert!(
            msg.contains("bridge"),
            "must name the head predicate: {msg}"
        );
        assert!(
            msg.contains("connected") || msg.contains("node"),
            "must name the conflicting body predicate: {msg}"
        );
        assert!(msg.contains('A'), "must name the variable: {msg}");
        assert!(msg.contains("position 0"), "must name the position: {msg}");
    }

    #[test]
    fn well_typed_cross_predicate_rule_still_lowers() {
        let source = r#"
pred node(u32, symbol).
pred connected(u32, u32).
pred bridge(u32, u32).

node(1, "alice").
connected(1, 2).

bridge(A, B) :- connected(A, B), node(A, _).

?- bridge(W, X).
"#;
        let program = crate::parse_program(source).expect("example parses");
        let mut lowerer = Lowerer::new();
        lowerer
            .lower_program(&program)
            .expect("well-typed program lowers");
    }

    #[test]
    fn probabilistic_fact_schema_precedes_body_only_variable_defaults() {
        let source = r#"
0.5::possible(0).
observed(0).
accepted(X) :- possible(X), observed(X).
?- accepted(X).
"#;
        let program = crate::parse_program(source).expect("probabilistic join parses");
        let mut lowerer = Lowerer::new();
        lowerer
            .lower_program(&program)
            .expect("probabilistic facts provide body predicate schemas");
        assert_eq!(
            lowerer
                .schemas()
                .get("possible")
                .expect("possible schema")
                .column_type(0),
            Some(ScalarType::U32)
        );
    }

    #[test]
    fn annotated_disjunction_schema_precedes_body_only_variable_defaults() {
        let source = r#"
0.5::possible(0); 0.5::possible(1).
observed(0).
accepted(X) :- possible(X), observed(X).
?- accepted(X).
"#;
        let program = crate::parse_program(source).expect("annotated-disjunction join parses");
        let mut lowerer = Lowerer::new();
        lowerer
            .lower_program(&program)
            .expect("annotated-disjunction choices provide body predicate schemas");
        assert_eq!(
            lowerer
                .schemas()
                .get("possible")
                .expect("possible schema")
                .column_type(0),
            Some(ScalarType::U32)
        );
    }

    #[test]
    fn body_body_schema_conflict_is_a_compilation_error() {
        // The same variable drawing incompatible types from two body atoms
        // must be rejected even when the head is consistent with one side.
        let source = r#"
pred label(symbol).
pred count(u32).
pred out(u32).

label("x").
count(1).

out(A) :- count(A), label(A).

?- out(W).
"#;
        let program = crate::parse_program(source).expect("example parses");
        let mut lowerer = Lowerer::new();
        let err = lowerer
            .lower_program(&program)
            .expect_err("body-body conflict must not lower");
        assert!(
            matches!(err, xlog_core::XlogError::Compilation(_)),
            "expected a compilation error, got: {err}"
        );
    }
}
