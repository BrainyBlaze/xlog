//! Safe meta-predicate normalization for the language-completeness surface.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use xlog_core::{symbol, Result, ScalarType, XlogError};

use crate::ast::{
    Atom, BodyLiteral, Comparison, PredColumn, PredDecl, Program, Rule, Term, TypeRef, Univ,
};
use crate::program_custody::{
    ColumnCustody, PayloadCustody, PredicateCustody, ProgramAllocationCustody, RetainedProgram,
    RetainedVectorCustody,
};

const TERM_ID_TYPE: ScalarType = ScalarType::U64;

/// Normalize safe meta-predicates into finite helper relations.
///
/// Accepted forms are static and finite. Runtime-variable predicate names,
/// dynamic database mutation, unrestricted calls, and non-finite collection are
/// rejected before lowering.
pub fn normalize_meta_builtins(program: &Program) -> Result<Program> {
    normalize_meta_builtins_owned(program.clone())
}

/// [`normalize_meta_builtins`] taking the program by value.
///
/// Avoids cloning the whole AST: facts whose predicate has no term-valued
/// columns are moved through untouched (the pass is the identity on them, see
/// `MetaNormalizer::fact_is_identity`), everything else is rebuilt as before.
pub fn normalize_meta_builtins_owned(program: Program) -> Result<Program> {
    normalize_meta_builtins_with_owner(program, None)
}

/// Normalize through the same finite meta core with original allocation
/// receipts acquired before owned output is created.
pub fn normalize_meta_builtins_owned_with_allocation_owner(
    program: Program,
    allocations: &mut crate::compile::CompilationAllocationScope<'_>,
) -> Result<Program> {
    normalize_meta_builtins_with_owner(program, Some(allocations))
}

fn normalize_meta_builtins_with_owner(
    program: Program,
    reserve: Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> Result<Program> {
    let mut program = RetainedProgram {
        value: program,
        custody: ProgramAllocationCustody::default(),
    };
    let mut normalizer = MetaNormalizer::new(&program.value, reserve)?;
    normalizer.prepare_raw_custody(&mut program)?;
    let program = normalizer.normalize_program(program)?;
    normalizer.retain_raw_program(program)
}

/// Move original compiler custody through the same finite normalizer while
/// newly constructed payloads retain their original birth-time receipts.
pub(crate) fn normalize_meta_builtins_retained(
    program: RetainedProgram,
    scope: &mut crate::compile::CompilationAllocationScope<'_>,
) -> Result<RetainedProgram> {
    let mut normalizer = MetaNormalizer::new(&program.value, Some(scope))?;
    normalizer.normalize_program(program)
}

struct RetainedMetaValue<T> {
    value: T,
    custody: PayloadCustody,
}

struct MetaProgramAllocation {
    _custody: ProgramAllocationCustody,
    _metadata_owner: Arc<dyn Send + Sync>,
}

struct RetainedRawMetaProgram {
    value: Program,
    owner: Arc<MetaProgramAllocation>,
}

fn meta_storage_bytes<T>(count: usize) -> Result<u64> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| XlogError::allocation_extent_overflow("meta vector storage"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpectedTerm {
    Any,
    Compound,
    PredRef,
}

#[derive(Debug, Clone)]
enum RegisteredTerm {
    Scalar,
    PredRef,
    Compound {
        functor: String,
        args: Vec<u64>,
        parts: Vec<u64>,
    },
}

#[derive(Clone)]
struct TermRecord {
    id: u64,
    kind: RegisteredTerm,
    _allocation_owners: [Option<Arc<dyn Send + Sync>>; 2],
}

struct RegisteredTermId {
    id: u64,
    _key_allocation: Option<Arc<dyn Send + Sync>>,
}

struct RetainedSourceFacts {
    facts: Vec<Atom>,
    _allocation_owner: Option<Arc<dyn Send + Sync>>,
}

struct RetainedTermKey {
    text: String,
    _allocation_owner: Option<Arc<dyn Send + Sync>>,
}

struct RetainedBindings {
    values: HashMap<String, Term>,
    _allocation_owner: Option<Arc<dyn Send + Sync>>,
}

struct RetainedTermRow {
    values: Vec<Term>,
    custody: PayloadCustody,
    storage_owner: Option<Arc<dyn Send + Sync>>,
}

struct RetainedTermValue {
    value: Term,
    custody: PayloadCustody,
}

struct RetainedTermRows {
    rows: Vec<RetainedTermRow>,
    _allocation_owner: Option<Arc<dyn Send + Sync>>,
}

struct RetainedFindallGroup {
    terms: RetainedTermRow,
    values: Vec<RetainedTermValue>,
    _key_owner: Option<Arc<dyn Send + Sync>>,
    values_storage: Option<Arc<dyn Send + Sync>>,
}

struct RetainedRegisteredTerm {
    key: RetainedTermKey,
    kind: RegisteredTerm,
    allocation_owners: [Option<Arc<dyn Send + Sync>>; 2],
}

struct RetainedTypeRef {
    value: TypeRef,
    custody: PayloadCustody,
}

impl RetainedTypeRef {
    fn scalar(typ: ScalarType) -> Self {
        Self {
            value: TypeRef::Scalar(typ),
            custody: PayloadCustody::default(),
        }
    }

    fn term() -> Self {
        Self {
            value: TypeRef::Term,
            custody: PayloadCustody::default(),
        }
    }

    fn clone_with_owner(
        typ: &TypeRef,
        scope: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
    ) -> Result<Self> {
        let custody = match scope.as_deref_mut() {
            Some(scope) => PayloadCustody::for_type_clone(typ, scope)?,
            None => PayloadCustody::default(),
        };
        Ok(Self {
            value: typ.clone(),
            custody,
        })
    }

    fn into_list(
        mut self,
        scope: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
    ) -> Result<Self> {
        if let Some(scope) = scope.as_deref_mut() {
            self.custody
                .reserve_unit(std::mem::size_of::<TypeRef>() as u64, scope)?;
        }
        Ok(Self {
            value: TypeRef::List(Box::new(self.value)),
            custody: self.custody,
        })
    }
}

struct RetainedHelperColumns {
    value: Vec<PredColumn>,
    custody: Option<RetainedVectorCustody<ColumnCustody>>,
}

struct RegisteredHelperDeclaration {
    columns: RetainedHelperColumns,
    key_owner: Option<Arc<dyn Send + Sync>>,
}

struct RetainedHelperDeclarations {
    values: BTreeMap<String, RegisteredHelperDeclaration>,
    storage: PayloadCustody,
}

struct RetainedHelperEmission {
    value: PredDecl,
    custody: PredicateCustody,
}

struct MetaNormalizer<'a, 'pool> {
    domains: HashMap<String, ScalarType>,
    /// Declared column types per predicate (index = column position).
    pred_columns: HashMap<String, Vec<TypeRef>>,
    /// Predicates with at least one term-valued (`term`/`compound`/`predref`)
    /// or `list<T>` column: only their atoms can change under value
    /// normalization.
    term_valued_preds: HashSet<String>,
    /// Source facts by predicate — consulted by `findall`/`maplist`
    /// expansion only, so it is built only when a meta predicate occurs.
    source_facts: HashMap<String, Vec<Atom>>,
    derived_predicates: HashSet<String>,
    term_ids: HashMap<String, RegisteredTermId>,
    term_records: Vec<TermRecord>,
    next_term_id: u64,
    helper_decls: BTreeMap<String, RegisteredHelperDeclaration>,
    helper_facts: Vec<Rule>,
    helper_facts_custody: RetainedVectorCustody<PayloadCustody>,
    helper_fact_keys: HashMap<String, Option<Arc<dyn Send + Sync>>>,
    findall_counter: usize,
    maplist_counter: usize,
    allocation_owner: Option<&'a mut crate::compile::CompilationAllocationScope<'pool>>,
    _term_ids_storage: Option<Arc<dyn Send + Sync>>,
    _term_records_storage: Option<Arc<dyn Send + Sync>>,
    _helper_fact_keys_storage: Option<Arc<dyn Send + Sync>>,
    helper_decls_storage: PayloadCustody,
    _environment_owner: Option<Arc<dyn Send + Sync>>,
    capture_stack: Vec<Option<PayloadCustody>>,
    active_capture: Option<PayloadCustody>,
    _capture_stack_storage: Option<Arc<dyn Send + Sync>>,
}

impl<'a, 'pool> MetaNormalizer<'a, 'pool> {
    /// Keep suspended parent receipts in the context until their partial data
    /// has unwound; a successful constructor returns its data before custody.
    fn capture<T>(
        &mut self,
        build: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<RetainedMetaValue<T>> {
        if self.capture_stack.len() == self.capture_stack.capacity() {
            let count = self
                .capture_stack
                .len()
                .checked_add(1)
                .and_then(|count| count.max(4).checked_next_power_of_two())
                .ok_or_else(|| XlogError::allocation_extent_overflow("meta constructor stack"))?;
            let owner =
                self.reserve_temporary(meta_storage_bytes::<Option<PayloadCustody>>(count)?)?;
            self.capture_stack
                .reserve_exact(count - self.capture_stack.len());
            self._capture_stack_storage = owner;
        }
        self.capture_stack.push(self.active_capture.take());
        self.active_capture = Some(PayloadCustody::default());
        let result = build(self);
        let custody = self
            .active_capture
            .take()
            .expect("active meta constructor custody");
        self.active_capture = self
            .capture_stack
            .pop()
            .expect("original meta constructor custody");
        result.map(|value| RetainedMetaValue { value, custody })
    }

    fn append_custody(
        &mut self,
        into: &mut PayloadCustody,
        from: &mut PayloadCustody,
    ) -> Result<()> {
        if let Some(scope) = self.allocation_owner.as_deref_mut() {
            into.append_from(from, scope)?;
        }
        Ok(())
    }

    fn adopt_custody(&mut self, from: &mut PayloadCustody) -> Result<()> {
        if let Some(scope) = self.allocation_owner.as_deref_mut() {
            self.active_capture
                .as_mut()
                .expect("active meta output custody")
                .append_from(from, scope)?;
        }
        Ok(())
    }

    fn prepare_raw_custody(&mut self, program: &mut RetainedProgram) -> Result<()> {
        let Some(scope) = self.allocation_owner.as_deref_mut() else {
            return Ok(());
        };
        macro_rules! prepare {
            ($field:ident) => {{
                program
                    .custody
                    .$field
                    .prepare_elements(program.value.$field.len(), scope)?;
                program
                    .custody
                    .$field
                    .elements
                    .resize_with(program.value.$field.len(), Default::default);
            }};
        }
        prepare!(rules);
        prepare!(constraints);
        prepare!(queries);
        prepare!(prob_facts);
        prepare!(evidence);
        prepare!(prob_queries);
        prepare!(neural_predicates);
        prepare!(learnable_rules);
        prepare!(predicates);
        prepare!(annotated_disjunctions);
        for (value, custody) in program
            .value
            .annotated_disjunctions
            .iter()
            .zip(&mut program.custody.annotated_disjunctions.elements)
        {
            custody
                .choices
                .prepare_elements(value.choices.len(), scope)?;
            custody
                .choices
                .elements
                .resize_with(value.choices.len(), Default::default);
        }
        Ok(())
    }

    fn retain_raw_program(&mut self, program: RetainedProgram) -> Result<Program> {
        let Some(scope) = self.allocation_owner.as_deref_mut() else {
            return Ok(program.value);
        };
        let bytes = std::mem::size_of::<MetaProgramAllocation>()
            .checked_add(2 * std::mem::size_of::<usize>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| XlogError::allocation_extent_overflow("meta program custody"))?;
        let metadata_owner = scope.reserve_allocation(bytes)?;
        let output = RetainedRawMetaProgram {
            value: program.value,
            owner: Arc::new(MetaProgramAllocation {
                _custody: program.custody,
                _metadata_owner: metadata_owner,
            }),
        };
        scope.retain(Arc::clone(&output.owner) as Arc<dyn Send + Sync>)?;
        Ok(output.value)
    }

    fn reserve_temporary(&mut self, bytes: u64) -> Result<Option<Arc<dyn Send + Sync>>> {
        if bytes == 0 {
            return Ok(None);
        }
        self.allocation_owner
            .as_deref_mut()
            .map(|scope| scope.reserve_allocation(bytes))
            .transpose()
    }

    fn retain_temporary(&mut self, owner: Option<Arc<dyn Send + Sync>>) -> Result<()> {
        if let Some(owner) = owner {
            let scope = self
                .allocation_owner
                .as_deref_mut()
                .expect("original meta allocation scope");
            self.active_capture
                .as_mut()
                .expect("active meta output custody")
                .retain_original(&owner, scope)?;
        }
        Ok(())
    }

    fn reserve_binding_map(
        &mut self,
        body: &[BodyLiteral],
    ) -> Result<(usize, Option<Arc<dyn Send + Sync>>)> {
        if self.allocation_owner.is_none() {
            return Ok((0, None));
        }
        let mut variables = Some((0usize, 0u64));
        for literal in body {
            literal.visit_variables(&mut |name| {
                variables = variables.and_then(|(count, bytes)| {
                    Some((count.checked_add(1)?, bytes.checked_add(name.len() as u64)?))
                });
            });
        }
        let (count, text) = variables
            .ok_or_else(|| XlogError::allocation_extent_overflow("finite meta binding"))?;
        let bytes = crate::compile::hash_allocation_bytes::<String, ScalarType>(count)?
            .checked_add(text)
            .ok_or_else(|| XlogError::allocation_extent_overflow("finite meta binding"))?;
        Ok((count, self.reserve_temporary(bytes)?))
    }

    fn acquire(&mut self, bytes: u64) -> Result<()> {
        if bytes != 0 {
            if let Some(reserve) = self.allocation_owner.as_deref_mut() {
                self.active_capture
                    .as_mut()
                    .expect("active meta output custody")
                    .reserve_unit(bytes, reserve)?;
            }
        }
        Ok(())
    }

    fn acquire_vector<T>(&mut self, count: usize) -> Result<()> {
        if self.allocation_owner.is_some() {
            self.acquire(meta_storage_bytes::<T>(count)?)?;
        }
        Ok(())
    }

    fn acquire_term_clone(&mut self, term: &Term) -> Result<()> {
        if let Some(scope) = self.allocation_owner.as_deref_mut() {
            let mut original = PayloadCustody::for_term_clone(term, scope)?;
            self.active_capture
                .as_mut()
                .expect("active meta output custody")
                .append_from(&mut original, scope)?;
        }
        Ok(())
    }

    fn acquire_literal_clone(&mut self, literal: &BodyLiteral) -> Result<()> {
        if let Some(scope) = self.allocation_owner.as_deref_mut() {
            let mut original = PayloadCustody::for_literal_clone(literal, scope)?;
            self.active_capture
                .as_mut()
                .expect("active meta output custody")
                .append_from(&mut original, scope)?;
        }
        Ok(())
    }

    fn intern_symbol(&mut self, text: &str) -> Result<u32> {
        match self.allocation_owner.as_deref_mut() {
            Some(reserve) => symbol::intern_with_allocation_owner(text, &mut |bytes| {
                reserve.reserve_allocation(
                    u64::try_from(bytes)
                        .map_err(|_| XlogError::allocation_extent_overflow("meta symbol"))?,
                )
            }),
            None => Ok(symbol::intern(text)),
        }
    }

    fn acquire_format(&mut self, arguments: std::fmt::Arguments<'_>) -> Result<()> {
        if self.allocation_owner.is_some() {
            self.acquire(crate::compile::formatted_text_allocation_bytes(arguments)?)?;
        }
        Ok(())
    }

    fn term_keys_equal(&mut self, left: &Term, right: &Term) -> Result<bool> {
        term_keys_equal(left, right, &mut self.allocation_owner)
    }

    fn reserve_hash_growth<K, V>(
        &mut self,
        length: usize,
        capacity: usize,
    ) -> Result<Option<Arc<dyn Send + Sync>>> {
        if self.allocation_owner.is_some() && length == capacity {
            let count = length
                .checked_add(1)
                .ok_or_else(|| XlogError::allocation_extent_overflow("meta table"))?;
            return self.reserve_temporary(crate::compile::hash_allocation_bytes::<K, V>(count)?);
        }
        Ok(None)
    }

    fn reserve_vector_growth<T>(
        &mut self,
        length: usize,
        capacity: usize,
    ) -> Result<Option<Arc<dyn Send + Sync>>> {
        if self.allocation_owner.is_some() && length == capacity {
            let count = length
                .checked_add(1)
                .ok_or_else(|| XlogError::allocation_extent_overflow("meta vector"))?;
            return self.reserve_temporary(xlog_core::push_vector_storage_bytes::<T>(count)?);
        }
        Ok(None)
    }

    fn helper_columns(&mut self, count: usize) -> Result<RetainedHelperColumns> {
        let custody = match self.allocation_owner.as_deref_mut() {
            Some(scope) => Some(RetainedVectorCustody::for_capacity::<PredColumn>(
                count, scope,
            )?),
            None => None,
        };
        Ok(RetainedHelperColumns {
            value: Vec::with_capacity(count),
            custody,
        })
    }

    fn push_helper_column(
        &mut self,
        columns: &mut RetainedHelperColumns,
        name: &str,
        typ: RetainedTypeRef,
        lowercase_name: bool,
    ) -> Result<()> {
        let name_custody = match self.allocation_owner.as_deref_mut() {
            Some(scope) => PayloadCustody::for_text_clone(name, scope)?,
            None => PayloadCustody::default(),
        };
        let column = PredColumn {
            name: Some(if lowercase_name {
                name.to_ascii_lowercase()
            } else {
                name.to_string()
            }),
            typ: typ.value,
        };
        columns.value.push(column);
        if let Some(custody) = columns.custody.as_mut() {
            custody.elements.push(ColumnCustody {
                name: name_custody,
                typ: typ.custody,
            });
        }
        Ok(())
    }

    fn scalar_columns(&mut self, columns: &[(&str, ScalarType)]) -> Result<RetainedHelperColumns> {
        let mut retained = self.helper_columns(columns.len())?;
        for (name, typ) in columns {
            self.push_helper_column(&mut retained, name, RetainedTypeRef::scalar(*typ), false)?;
        }
        Ok(retained)
    }

    fn univ_columns(&mut self) -> Result<RetainedHelperColumns> {
        let mut columns = self.helper_columns(2)?;
        self.push_helper_column(
            &mut columns,
            "term_id",
            RetainedTypeRef::scalar(TERM_ID_TYPE),
            false,
        )?;
        let parts = RetainedTypeRef::term().into_list(&mut self.allocation_owner)?;
        self.push_helper_column(&mut columns, "parts", parts, false)?;
        Ok(columns)
    }

    fn cloned_column_type(
        &mut self,
        predicate: &str,
        index: usize,
    ) -> Result<Option<RetainedTypeRef>> {
        Self::column_type(&self.pred_columns, predicate, index)
            .map(|typ| RetainedTypeRef::clone_with_owner(typ, &mut self.allocation_owner))
            .transpose()
    }

    fn cloned_source_facts(&mut self, predicate: &str) -> Result<RetainedSourceFacts> {
        let mut bytes = 0u64;
        if self.allocation_owner.is_some() {
            for fact in self.source_facts.get(predicate).into_iter().flatten() {
                bytes = bytes
                    .checked_add(fact.owned_allocation_bytes()?)
                    .ok_or_else(|| XlogError::allocation_extent_overflow("finite fact clone"))?;
            }
        }
        let allocation_owner = self.reserve_temporary(bytes)?;
        let facts = self
            .source_facts
            .get(predicate)
            .cloned()
            .unwrap_or_default();
        Ok(RetainedSourceFacts {
            facts,
            _allocation_owner: allocation_owner,
        })
    }

    fn new(
        program: &Program,
        mut allocation_owner: Option<&'a mut crate::compile::CompilationAllocationScope<'pool>>,
    ) -> Result<Self> {
        use crate::compile::hash_allocation_bytes;
        let overflow = || XlogError::allocation_extent_overflow("finite meta environment");
        let add = |left: u64, right: u64| left.checked_add(right).ok_or_else(overflow);
        let vector = |count: usize, width: usize| {
            count
                .checked_mul(width)
                .and_then(|n| u64::try_from(n).ok())
                .ok_or_else(overflow)
        };
        let needs_source_facts = program_uses_meta_predicates(program);
        let fact_count = program.rules.iter().filter(|rule| rule.is_fact()).count();
        let derived_count = program.rules.len() - fact_count;
        let _rosters_owner = if let Some(reserve) = allocation_owner.as_deref_mut() {
            let mut bytes = hash_allocation_bytes::<&str, ScalarType>(program.domains.len())?;
            bytes = add(
                bytes,
                hash_allocation_bytes::<&str, (&PredDecl, bool)>(program.predicates.len())?,
            )?;
            bytes = add(bytes, hash_allocation_bytes::<&str, ()>(derived_count)?)?;
            if needs_source_facts {
                bytes = add(bytes, hash_allocation_bytes::<&str, usize>(fact_count)?)?;
            }
            if bytes == 0 {
                None
            } else {
                Some(reserve.reserve_allocation(bytes)?)
            }
        } else {
            None
        };
        let mut domain_roster: HashMap<&str, ScalarType> =
            HashMap::with_capacity(program.domains.len());
        for domain in &program.domains {
            domain_roster.insert(domain.name.as_str(), domain.typ);
        }
        let mut predicate_roster: HashMap<&str, (&PredDecl, bool)> =
            HashMap::with_capacity(program.predicates.len());
        for pred in &program.predicates {
            let term_valued = |typ: &TypeRef| {
                matches!(
                    typ,
                    TypeRef::Term | TypeRef::Compound | TypeRef::PredRef | TypeRef::List(_)
                )
            };
            let has_terms = if pred.columns.is_empty() {
                pred.types.iter().any(term_valued)
            } else {
                pred.columns.iter().any(|column| term_valued(&column.typ))
            };
            predicate_roster
                .entry(pred.name.as_str())
                .and_modify(|(last, any_terms)| {
                    *last = pred;
                    *any_terms |= has_terms;
                })
                .or_insert((pred, has_terms));
        }
        let mut fact_roster: HashMap<&str, usize> =
            HashMap::with_capacity(if needs_source_facts { fact_count } else { 0 });
        let mut derived_roster: HashSet<&str> = HashSet::with_capacity(derived_count);
        for rule in &program.rules {
            if rule.is_fact() {
                if needs_source_facts {
                    let count = fact_roster.entry(rule.head.predicate.as_str()).or_default();
                    *count = count.checked_add(1).ok_or_else(overflow)?;
                }
            } else {
                derived_roster.insert(rule.head.predicate.as_str());
            }
        }
        let term_predicates = predicate_roster
            .values()
            .filter(|(_, has_terms)| *has_terms)
            .count();
        let environment_owner = if let Some(reserve) = allocation_owner.as_deref_mut() {
            let mut bytes = hash_allocation_bytes::<String, ScalarType>(domain_roster.len())?;
            bytes = add(
                bytes,
                hash_allocation_bytes::<String, Vec<TypeRef>>(predicate_roster.len())?,
            )?;
            bytes = add(bytes, hash_allocation_bytes::<String, ()>(term_predicates)?)?;
            bytes = add(
                bytes,
                hash_allocation_bytes::<String, ()>(derived_roster.len())?,
            )?;
            bytes = add(
                bytes,
                hash_allocation_bytes::<String, Vec<Atom>>(fact_roster.len())?,
            )?;
            for name in domain_roster
                .keys()
                .chain(derived_roster.iter())
                .chain(fact_roster.keys())
            {
                bytes = add(bytes, name.len() as u64)?;
            }
            for (name, (pred, has_terms)) in &predicate_roster {
                bytes = add(bytes, name.len() as u64)?;
                if *has_terms {
                    bytes = add(bytes, name.len() as u64)?;
                }
                let count = if pred.columns.is_empty() {
                    pred.types.len()
                } else {
                    pred.columns.len()
                };
                bytes = add(bytes, vector(count, std::mem::size_of::<TypeRef>())?)?;
                if pred.columns.is_empty() {
                    for typ in &pred.types {
                        bytes = add(
                            bytes,
                            typ.owned_allocation_bytes()? - std::mem::size_of::<TypeRef>() as u64,
                        )?;
                    }
                } else {
                    for column in &pred.columns {
                        bytes = add(
                            bytes,
                            column.typ.owned_allocation_bytes()?
                                - std::mem::size_of::<TypeRef>() as u64,
                        )?;
                    }
                }
            }
            for count in fact_roster.values() {
                bytes = add(bytes, vector(*count, std::mem::size_of::<Atom>())?)?;
            }
            if needs_source_facts {
                for rule in program.rules.iter().filter(|rule| rule.is_fact()) {
                    bytes = add(
                        bytes,
                        rule.head.owned_allocation_bytes()? - std::mem::size_of::<Atom>() as u64,
                    )?;
                }
            }
            if bytes == 0 {
                None
            } else {
                Some(reserve.reserve_allocation(bytes)?)
            }
        } else {
            None
        };
        let mut domains: HashMap<String, ScalarType> = HashMap::with_capacity(domain_roster.len());
        for (name, typ) in domain_roster {
            domains.insert(name.to_string(), typ);
        }
        let mut pred_columns: HashMap<String, Vec<TypeRef>> =
            HashMap::with_capacity(predicate_roster.len());
        let mut term_valued_preds: HashSet<String> = HashSet::with_capacity(term_predicates);
        for (name, (pred, has_terms)) in predicate_roster {
            let count = if pred.columns.is_empty() {
                pred.types.len()
            } else {
                pred.columns.len()
            };
            let _schema_owner = if let Some(reserve) = allocation_owner.as_deref_mut() {
                let mut bytes = vector(count, std::mem::size_of::<PredColumn>())?;
                for column in &pred.columns {
                    if let Some(name) = &column.name {
                        bytes = add(bytes, name.len() as u64)?;
                    }
                }
                if bytes == 0 {
                    None
                } else {
                    Some(reserve.reserve_allocation(bytes)?)
                }
            } else {
                None
            };
            let schema = pred.schema_columns();
            // This destination has its own original allocation. Type payloads
            // move under the context receipt; the schema vector cannot become
            // the surviving type vector through an in-place collection.
            let mut columns = Vec::with_capacity(count);
            for column in schema {
                columns.push(column.typ);
            }
            if has_terms {
                term_valued_preds.insert(name.to_string());
            }
            pred_columns.insert(name.to_string(), columns);
        }
        let mut source_facts: HashMap<String, Vec<Atom>> =
            HashMap::with_capacity(fact_roster.len());
        for (name, count) in fact_roster {
            source_facts.insert(name.to_string(), Vec::with_capacity(count));
        }
        if needs_source_facts {
            for rule in program.rules.iter().filter(|rule| rule.is_fact()) {
                source_facts
                    .get_mut(rule.head.predicate.as_str())
                    .expect("original finite source fact roster")
                    .push(rule.head.clone());
            }
        }
        let mut derived_predicates: HashSet<String> = HashSet::with_capacity(derived_roster.len());
        for name in derived_roster {
            derived_predicates.insert(name.to_string());
        }

        Ok(Self {
            domains,
            pred_columns,
            term_valued_preds,
            source_facts,
            derived_predicates,
            term_ids: HashMap::new(),
            term_records: Vec::new(),
            next_term_id: 1,
            helper_decls: BTreeMap::new(),
            helper_facts: Vec::new(),
            helper_facts_custody: RetainedVectorCustody::default(),
            helper_fact_keys: HashMap::new(),
            findall_counter: 0,
            maplist_counter: 0,
            allocation_owner,
            _term_ids_storage: None,
            _term_records_storage: None,
            _helper_fact_keys_storage: None,
            helper_decls_storage: PayloadCustody::default(),
            _environment_owner: environment_owner,
            capture_stack: Vec::new(),
            active_capture: None,
            _capture_stack_storage: None,
        })
    }

    /// Declared column type of `pred` at position `idx`, if any.
    fn column_type<'columns>(
        columns: &'columns HashMap<String, Vec<TypeRef>>,
        pred: &str,
        idx: usize,
    ) -> Option<&'columns TypeRef> {
        columns.get(pred).and_then(|cols| cols.get(idx))
    }

    /// True when value normalization is the identity on this fact: its
    /// predicate has no term-valued/list column, so every term goes through
    /// `normalize_untyped_value`, which rebuilds the term unchanged. Such facts
    /// are moved through without being rebuilt (they dominate ILP/EDB inputs).
    fn fact_is_identity(&self, rule: &Rule) -> bool {
        rule.body.is_empty() && !self.term_valued_preds.contains(&rule.head.predicate)
    }

    fn normalize_program(&mut self, program: RetainedProgram) -> Result<RetainedProgram> {
        let mut out = program;
        for (index, rule) in out.value.rules.iter_mut().enumerate() {
            if !self.fact_is_identity(rule) {
                let replacement = self.capture(|normalizer| normalizer.normalize_rule(rule))?;
                *rule = replacement.value;
                if self.allocation_owner.is_some() {
                    out.custody.rules.elements[index] = replacement.custody;
                }
            }
        }
        for (index, constraint) in out.value.constraints.iter_mut().enumerate() {
            let (capacity, _bound_owner) = self.reserve_binding_map(&constraint.body)?;
            let mut bound = HashMap::with_capacity(capacity);
            let replacement =
                self.capture(|normalizer| normalizer.normalize_body(&constraint.body, &mut bound))?;
            constraint.body = replacement.value;
            if self.allocation_owner.is_some() {
                out.custody.constraints.elements[index] = replacement.custody;
            }
        }
        for (index, query) in out.value.queries.iter_mut().enumerate() {
            let replacement =
                self.capture(|normalizer| normalizer.normalize_atom_values(&query.atom))?;
            query.atom = replacement.value;
            if self.allocation_owner.is_some() {
                out.custody.queries.elements[index] = replacement.custody;
            }
        }

        for (index, pf) in out.value.prob_facts.iter_mut().enumerate() {
            let replacement =
                self.capture(|normalizer| normalizer.normalize_atom_values(&pf.atom))?;
            pf.atom = replacement.value;
            if self.allocation_owner.is_some() {
                out.custody.prob_facts.elements[index] = replacement.custody;
            }
        }
        for (index, ad) in out.value.annotated_disjunctions.iter_mut().enumerate() {
            for (choice_index, choice) in ad.choices.iter_mut().enumerate() {
                let replacement =
                    self.capture(|normalizer| normalizer.normalize_atom_values(&choice.atom))?;
                choice.atom = replacement.value;
                if self.allocation_owner.is_some() {
                    out.custody.annotated_disjunctions.elements[index]
                        .choices
                        .elements[choice_index] = replacement.custody;
                }
            }
        }
        for (index, evidence) in out.value.evidence.iter_mut().enumerate() {
            let replacement =
                self.capture(|normalizer| normalizer.normalize_atom_values(&evidence.atom))?;
            evidence.atom = replacement.value;
            if self.allocation_owner.is_some() {
                out.custody.evidence.elements[index] = replacement.custody;
            }
        }
        for (index, query) in out.value.prob_queries.iter_mut().enumerate() {
            let replacement =
                self.capture(|normalizer| normalizer.normalize_atom_values(&query.atom))?;
            query.atom = replacement.value;
            if self.allocation_owner.is_some() {
                out.custody.prob_queries.elements[index] = replacement.custody;
            }
        }
        for (index, neural) in out.value.neural_predicates.iter_mut().enumerate() {
            let replacement =
                self.capture(|normalizer| normalizer.normalize_atom_values(&neural.predicate))?;
            neural.predicate = replacement.value;
            if self.allocation_owner.is_some() {
                out.custody.neural_predicates.elements[index].predicate = replacement.custody;
            }
        }
        for (index, learnable) in out.value.learnable_rules.iter_mut().enumerate() {
            let (capacity, _bound_owner) = self.reserve_binding_map(&learnable.body)?;
            let mut bound = HashMap::with_capacity(capacity);
            let head =
                self.capture(|normalizer| normalizer.normalize_atom_values(&learnable.head))?;
            learnable.head = head.value;
            if self.allocation_owner.is_some() {
                out.custody.learnable_rules.elements[index].head = head.custody;
            }
            let body =
                self.capture(|normalizer| normalizer.normalize_body(&learnable.body, &mut bound))?;
            learnable.body = body.value;
            if self.allocation_owner.is_some() {
                out.custody.learnable_rules.elements[index].body = body.custody;
            }
        }

        self.append_helpers(&mut out)?;
        lower_meta_type_refs_in_predicates(&mut out.value);
        Ok(out)
    }

    fn normalize_rule(&mut self, rule: &Rule) -> Result<Rule> {
        let head = self.normalize_atom_values(&rule.head)?;
        let (capacity, _bound_owner) = self.reserve_binding_map(&rule.body)?;
        let mut bound = HashMap::with_capacity(capacity);
        let body = self.normalize_body(&rule.body, &mut bound)?;
        Ok(Rule { head, body })
    }

    fn normalize_body(
        &mut self,
        body: &[BodyLiteral],
        bound: &mut HashMap<String, ScalarType>,
    ) -> Result<Vec<BodyLiteral>> {
        let mut storage_owner = None;
        let mut out = Vec::new();
        for lit in body {
            match lit {
                BodyLiteral::Positive(atom) => {
                    let expanded = self.normalize_positive_atom(atom, bound)?;
                    if let Some(lit) = expanded {
                        self.record_bound_from_literal(&lit, bound);
                        self.push_body_literal(&mut out, &mut storage_owner, lit)?;
                    }
                }
                BodyLiteral::Negated(atom) => {
                    if is_meta_predicate(&atom.predicate)
                        || is_blocked_dynamic_predicate(&atom.predicate)
                    {
                        return Err(meta_error(format_args!("{}", "safe meta-predicates are not supported under negation in the finite meta normalization subset"), &mut self.allocation_owner));
                    }
                    let literal = BodyLiteral::Negated(self.normalize_atom_values(atom)?);
                    self.push_body_literal(&mut out, &mut storage_owner, literal)?;
                }
                BodyLiteral::Comparison(cmp) => {
                    let literal = BodyLiteral::Comparison(Comparison {
                        op: cmp.op,
                        left: self.normalize_untyped_value(&cmp.left)?,
                        right: self.normalize_untyped_value(&cmp.right)?,
                    });
                    self.push_body_literal(&mut out, &mut storage_owner, literal)?;
                }
                BodyLiteral::Epistemic(epistemic) => {
                    self.acquire_literal_clone(lit)?;
                    self.push_body_literal(
                        &mut out,
                        &mut storage_owner,
                        BodyLiteral::Epistemic(epistemic.clone()),
                    )?;
                }
                BodyLiteral::IsExpr(is_expr) => {
                    self.acquire_literal_clone(lit)?;
                    self.push_body_literal(
                        &mut out,
                        &mut storage_owner,
                        BodyLiteral::IsExpr(is_expr.clone()),
                    )?;
                }
                BodyLiteral::Univ(univ) => {
                    let lit = self.expand_univ(univ, bound)?;
                    self.record_bound_from_literal(&lit, bound);
                    self.push_body_literal(&mut out, &mut storage_owner, lit)?;
                }
            }
        }
        if let Some(owner) = &storage_owner {
            self.retain_temporary(Some(Arc::clone(owner)))?;
        }
        Ok(out)
    }

    fn push_body_literal(
        &mut self,
        out: &mut Vec<BodyLiteral>,
        storage_owner: &mut Option<Arc<dyn Send + Sync>>,
        literal: BodyLiteral,
    ) -> Result<()> {
        let next_owner = self.reserve_vector_growth::<BodyLiteral>(out.len(), out.capacity())?;
        out.push(literal);
        if let Some(next_owner) = next_owner {
            *storage_owner = Some(next_owner);
        }
        Ok(())
    }

    fn normalize_positive_atom(
        &mut self,
        atom: &Atom,
        bound: &HashMap<String, ScalarType>,
    ) -> Result<Option<BodyLiteral>> {
        if is_blocked_dynamic_predicate(&atom.predicate) {
            return Err(dynamic_predicate_error(
                &atom.predicate,
                &mut self.allocation_owner,
            ));
        }

        match atom.predicate.as_str() {
            "ground" => self.expand_ground_like(atom, bound, MetaTruthKind::Ground),
            "var" => self.expand_ground_like(atom, bound, MetaTruthKind::Var),
            "nonvar" => self.expand_ground_like(atom, bound, MetaTruthKind::NonVar),
            "functor" => self.expand_functor(atom, bound),
            "findall" => self.expand_findall(atom, bound),
            "maplist" => self.expand_maplist(atom),
            _ => Ok(Some(BodyLiteral::Positive(
                self.normalize_atom_values(atom)?,
            ))),
        }
    }

    fn normalize_atom_values(&mut self, atom: &Atom) -> Result<Atom> {
        self.acquire_vector::<Term>(atom.terms.len())?;
        self.acquire(atom.predicate.len() as u64)?;
        let mut terms = Vec::with_capacity(atom.terms.len());
        for (idx, term) in atom.terms.iter().enumerate() {
            let expected = self.cloned_column_type(&atom.predicate, idx)?;
            terms.push(self.normalize_value_term(term, expected.as_ref().map(|typ| &typ.value))?);
        }
        Ok(Atom {
            predicate: atom.predicate.clone(),
            terms,
        })
    }

    fn normalize_value_term(&mut self, term: &Term, expected: Option<&TypeRef>) -> Result<Term> {
        self.normalize_value_term_with_receipt(term, expected, None)
    }

    fn normalize_value_term_retained(
        &mut self,
        term: &Term,
        expected: Option<&TypeRef>,
    ) -> Result<RetainedTermValue> {
        let value = self.capture(|normalizer| normalizer.normalize_value_term(term, expected))?;
        Ok(RetainedTermValue {
            value: value.value,
            custody: value.custody,
        })
    }

    fn normalize_value_term_with_receipt(
        &mut self,
        term: &Term,
        expected: Option<&TypeRef>,
        temporary_receipt: Option<&Arc<dyn Send + Sync>>,
    ) -> Result<Term> {
        match expected {
            Some(TypeRef::Term) if term.is_any_variable() => {
                if temporary_receipt.is_none() {
                    self.acquire_term_clone(term)?;
                }
                Ok(term.clone())
            }
            Some(TypeRef::Term) => Ok(Term::Integer(
                self.register_term(term, ExpectedTerm::Any)? as i128
            )),
            Some(TypeRef::Compound) if term.is_any_variable() => {
                if temporary_receipt.is_none() {
                    self.acquire_term_clone(term)?;
                }
                Ok(term.clone())
            }
            Some(TypeRef::Compound) => Ok(Term::Integer(
                self.register_term(term, ExpectedTerm::Compound)? as i128,
            )),
            Some(TypeRef::PredRef) if term.is_any_variable() => {
                if temporary_receipt.is_none() {
                    self.acquire_term_clone(term)?;
                }
                Ok(term.clone())
            }
            Some(TypeRef::PredRef) => Ok(Term::Integer(
                self.register_term(term, ExpectedTerm::PredRef)? as i128,
            )),
            Some(TypeRef::List(inner)) => self.normalize_list_value(term, inner, temporary_receipt),
            Some(TypeRef::Scalar(_)) | Some(TypeRef::Domain(_)) | None => {
                self.normalize_untyped_value_with_receipt(term, temporary_receipt)
            }
        }
    }

    fn normalize_untyped_value(&mut self, term: &Term) -> Result<Term> {
        self.normalize_untyped_value_with_receipt(term, None)
    }

    fn normalize_untyped_value_with_receipt(
        &mut self,
        term: &Term,
        temporary_receipt: Option<&Arc<dyn Send + Sync>>,
    ) -> Result<Term> {
        match term {
            Term::List(items) => {
                if temporary_receipt.is_none() {
                    self.acquire_vector::<Term>(items.len())?;
                }
                let mut values = Vec::with_capacity(items.len());
                for item in items {
                    values
                        .push(self.normalize_untyped_value_with_receipt(item, temporary_receipt)?);
                }
                Ok(Term::List(values))
            }
            Term::Cons { head, tail } => {
                if temporary_receipt.is_none() {
                    self.acquire(std::mem::size_of::<Term>() as u64)?;
                    self.acquire(std::mem::size_of::<Term>() as u64)?;
                }
                Ok(Term::Cons {
                    head: Box::new(
                        self.normalize_untyped_value_with_receipt(head, temporary_receipt)?,
                    ),
                    tail: Box::new(
                        self.normalize_untyped_value_with_receipt(tail, temporary_receipt)?,
                    ),
                })
            }
            _ => {
                if temporary_receipt.is_none() {
                    self.acquire_term_clone(term)?;
                }
                Ok(term.clone())
            }
        }
    }

    fn normalize_list_value(
        &mut self,
        term: &Term,
        inner: &TypeRef,
        temporary_receipt: Option<&Arc<dyn Send + Sync>>,
    ) -> Result<Term> {
        match term {
            Term::List(items) => {
                if temporary_receipt.is_none() {
                    self.acquire_vector::<Term>(items.len())?;
                }
                let mut values = Vec::with_capacity(items.len());
                for item in items {
                    values.push(self.normalize_value_term_with_receipt(
                        item,
                        Some(inner),
                        temporary_receipt,
                    )?);
                }
                Ok(Term::List(values))
            }
            Term::Cons { head, tail } => {
                if temporary_receipt.is_none() {
                    self.acquire(std::mem::size_of::<Term>() as u64)?;
                    self.acquire(std::mem::size_of::<Term>() as u64)?;
                }
                Ok(Term::Cons {
                    head: Box::new(self.normalize_value_term_with_receipt(
                        head,
                        Some(inner),
                        temporary_receipt,
                    )?),
                    tail: Box::new(self.normalize_list_value(tail, inner, temporary_receipt)?),
                })
            }
            Term::Variable(_) | Term::Anonymous | Term::Integer(_) => {
                if temporary_receipt.is_none() {
                    self.acquire_term_clone(term)?;
                }
                Ok(term.clone())
            }
            _ => Err(meta_error(
                format_args!(
                    "{}",
                    "list<T> column expects a finite list literal, list id, or variable"
                ),
                &mut self.allocation_owner,
            )),
        }
    }

    fn expand_ground_like(
        &mut self,
        atom: &Atom,
        bound: &HashMap<String, ScalarType>,
        kind: MetaTruthKind,
    ) -> Result<Option<BodyLiteral>> {
        require_arity(atom, 1, &mut self.allocation_owner)?;
        let truth = match kind {
            MetaTruthKind::Ground => self.is_ground(&atom.terms[0], bound),
            MetaTruthKind::Var => self.is_var(&atom.terms[0], bound),
            MetaTruthKind::NonVar => self.is_nonvar(&atom.terms[0], bound),
        };
        if truth {
            Ok(None)
        } else {
            Ok(Some(BodyLiteral::Positive(self.fail_atom()?)))
        }
    }

    fn expand_functor(
        &mut self,
        atom: &Atom,
        bound: &HashMap<String, ScalarType>,
    ) -> Result<Option<BodyLiteral>> {
        require_arity(atom, 3, &mut self.allocation_owner)?;
        let term = self.term_id_or_bound_variable(&atom.terms[0], bound, "functor/3")?;
        let name = self.normalize_symbol_argument(&atom.terms[1], "functor/3 name")?;
        let arity = self.normalize_u32_argument(&atom.terms[2], "functor/3 arity")?;
        let columns = self.scalar_columns(&[
            ("term_id", TERM_ID_TYPE),
            ("name", ScalarType::Symbol),
            ("arity", ScalarType::U32),
        ])?;
        self.register_helper_decl(functor_pred(), columns)?;
        self.acquire(functor_pred().len() as u64)?;
        self.acquire_vector::<Term>(3)?;
        Ok(Some(BodyLiteral::Positive(Atom {
            predicate: functor_pred().to_string(),
            terms: vec![term, name, arity],
        })))
    }

    fn expand_univ(
        &mut self,
        univ: &Univ,
        bound: &HashMap<String, ScalarType>,
    ) -> Result<BodyLiteral> {
        let term = self.term_id_or_bound_variable(&univ.term, bound, "univ")?;
        let parts = self.normalize_univ_parts(&univ.parts)?;
        let columns = self.univ_columns()?;
        self.register_helper_decl(univ_pred(), columns)?;
        self.acquire(univ_pred().len() as u64)?;
        self.acquire_vector::<Term>(2)?;
        Ok(BodyLiteral::Positive(Atom {
            predicate: univ_pred().to_string(),
            terms: vec![term, parts],
        }))
    }

    fn expand_findall(
        &mut self,
        atom: &Atom,
        bound: &HashMap<String, ScalarType>,
    ) -> Result<Option<BodyLiteral>> {
        require_arity(atom, 3, &mut self.allocation_owner)?;
        let template = &atom.terms[0];
        let _goal_owner = if self.allocation_owner.is_some() {
            self.reserve_temporary(
                atom.terms[1].owned_allocation_bytes()? - std::mem::size_of::<Term>() as u64,
            )?
        } else {
            None
        };
        let goal = compound_goal_atom(&atom.terms[1], "findall/3", &mut self.allocation_owner)?;
        self.acquire_term_clone(&atom.terms[2])?;
        let out_list = atom.terms[2].clone();
        if self.derived_predicates.contains(&goal.predicate) {
            return Err(meta_error(format_args!("{}", "findall/3 is limited to finite source facts; derived goals are reserved for a later aggregate-backed collection path"), &mut self.allocation_owner));
        }

        let _variables_owner = if self.allocation_owner.is_some() {
            let mut variables = Some((0usize, 0u64));
            template.visit_variables(&mut |name| {
                variables = variables
                    .and_then(|(n, b)| Some((n.checked_add(1)?, b.checked_add(name.len() as u64)?)))
            });
            let (count, text) = variables
                .ok_or_else(|| XlogError::allocation_extent_overflow("finite variable"))?;
            let mut bytes = crate::compile::push_vector_allocation_bytes::<&str>(count)?
                .checked_add(crate::compile::hash_allocation_bytes::<String, ()>(count)?)
                .and_then(|bytes| bytes.checked_add(text))
                .ok_or_else(|| XlogError::allocation_extent_overflow("finite variable"))?;
            bytes = bytes
                .checked_add(crate::compile::push_vector_allocation_bytes::<(
                    usize,
                    String,
                )>(goal.terms.len())?)
                .ok_or_else(|| XlogError::allocation_extent_overflow("finite variable"))?;
            let group_text = goal
                .terms
                .iter()
                .filter_map(|term| {
                    if let Term::Variable(name) = term {
                        Some(name.len() as u64)
                    } else {
                        None
                    }
                })
                .try_fold(0u64, |total, n| {
                    total
                        .checked_add(n)
                        .ok_or_else(|| XlogError::allocation_extent_overflow("finite variable"))
                })?;
            bytes = bytes
                .checked_add(group_text)
                .ok_or_else(|| XlogError::allocation_extent_overflow("finite variable"))?;
            self.reserve_temporary(bytes)?
        } else {
            None
        };
        let template_vars: HashSet<String> = template
            .variables()
            .into_iter()
            .map(str::to_string)
            .collect();
        let mut group_positions = Vec::new();
        for (idx, term) in goal.terms.iter().enumerate() {
            if let Term::Variable(name) = term {
                if bound.contains_key(name) {
                    group_positions.push((idx, name.clone()));
                } else if !template_vars.contains(name) {
                    return Err(meta_error(format_args!("{}", "unsafe findall/3 goal variable is neither bound before findall nor collected by the template"), &mut self.allocation_owner));
                }
            }
        }

        let elem_type_ref = self.infer_template_type(template, &goal)?;
        let findall_counter = self.findall_counter;
        self.acquire_format(format_args!("__xlog_meta_findall_{}", findall_counter))?;
        let helper = format!("__xlog_meta_findall_{}", findall_counter);
        self.findall_counter += 1;

        let mut columns = self.helper_columns(
            group_positions
                .len()
                .checked_add(1)
                .ok_or_else(|| XlogError::allocation_extent_overflow("finite findall columns"))?,
        )?;
        for (idx, name) in &group_positions {
            let typ =
                self.cloned_column_type(&goal.predicate, *idx)?
                    .unwrap_or(RetainedTypeRef::scalar(
                        bound.get(name).copied().unwrap_or(ScalarType::U64),
                    ));
            self.push_helper_column(&mut columns, name, typ, true)?;
        }
        let results_type =
            RetainedTypeRef::clone_with_owner(&elem_type_ref.value, &mut self.allocation_owner)?
                .into_list(&mut self.allocation_owner)?;
        self.push_helper_column(&mut columns, "results", results_type, false)?;
        self.register_helper_decl(&helper, columns)?;

        let _groups_owner = if self.allocation_owner.is_some() {
            let count = self
                .source_facts
                .get(&goal.predicate)
                .map_or(0, Vec::len)
                .max(1);
            let mut bytes = 0u64;
            for length in 0..count {
                bytes = bytes
                    .checked_add(xlog_core::btree_insert_allocation_bytes::<
                        String,
                        RetainedFindallGroup,
                    >(length)?)
                    .ok_or_else(|| {
                        XlogError::allocation_extent_overflow("finite findall groups")
                    })?;
            }
            self.reserve_temporary(bytes)?
        } else {
            None
        };
        let mut groups: BTreeMap<String, RetainedFindallGroup> = BTreeMap::new();
        let mut source_facts = self.cloned_source_facts(&goal.predicate)?;
        for fact in std::mem::take(&mut source_facts.facts) {
            if fact.terms.len() != goal.terms.len() {
                continue;
            }
            let Some(bindings) = self.match_goal_fact(&goal, &fact)? else {
                continue;
            };
            let storage_owner =
                self.reserve_temporary(meta_storage_bytes::<Term>(group_positions.len())?)?;
            let group_terms = self.capture(|normalizer| {
                let mut values = Vec::with_capacity(group_positions.len());
                for (idx, _) in &group_positions {
                    normalizer.acquire_term_clone(&fact.terms[*idx])?;
                    values.push(fact.terms[*idx].clone());
                }
                Ok(values)
            })?;
            let group_terms = RetainedTermRow {
                values: group_terms.value,
                custody: group_terms.custody,
                storage_owner,
            };
            let _keys_owner = if self.allocation_owner.is_some() {
                self.reserve_temporary(crate::compile::push_vector_allocation_bytes::<
                    RetainedTermKey,
                >(group_terms.values.len())?)?
            } else {
                None
            };
            let keys = group_terms
                .values
                .iter()
                .map(|term| retained_term_key(term, &mut self.allocation_owner))
                .collect::<Result<Vec<_>>>()?;
            let key = retained_joined_keys(&keys, "|", &mut self.allocation_owner)?;
            let template_value =
                self.substitute_template(template, &bindings.values, &elem_type_ref.value)?;
            let (length, capacity) = groups
                .get(&key.text)
                .map(|group| (group.values.len(), group.values.capacity()))
                .unwrap_or((0, 0));
            let values_storage =
                self.reserve_vector_growth::<RetainedTermValue>(length, capacity)?;
            let group = groups
                .entry(key.text)
                .or_insert_with(|| RetainedFindallGroup {
                    terms: group_terms,
                    values: Vec::new(),
                    _key_owner: key._allocation_owner,
                    values_storage: None,
                });
            group.values.push(template_value);
            if values_storage.is_some() {
                group.values_storage = values_storage;
            }
        }
        drop(source_facts);
        drop(elem_type_ref);

        if groups.is_empty() && group_positions.is_empty() {
            groups.insert(
                String::new(),
                RetainedFindallGroup {
                    terms: RetainedTermRow {
                        values: Vec::new(),
                        custody: PayloadCustody::default(),
                        storage_owner: None,
                    },
                    values: Vec::new(),
                    _key_owner: None,
                    values_storage: None,
                },
            );
        }

        for (_, mut group) in groups {
            let list = self.capture(|normalizer| {
                normalizer.acquire_vector::<Term>(group.values.len())?;
                let mut values = Vec::with_capacity(group.values.len());
                for mut value in std::mem::take(&mut group.values) {
                    normalizer.adopt_custody(&mut value.custody)?;
                    values.push(value.value);
                }
                Ok(Term::List(values))
            })?;
            if self.allocation_owner.is_some()
                && group.terms.values.len() == group.terms.values.capacity()
            {
                let count =
                    group.terms.values.len().checked_add(1).ok_or_else(|| {
                        XlogError::allocation_extent_overflow("finite findall row")
                    })?;
                let owner = self.reserve_temporary(meta_storage_bytes::<Term>(count)?)?;
                group.terms.values.reserve_exact(1);
                group.terms.storage_owner = owner;
            }
            let mut list = list;
            self.append_custody(&mut group.terms.custody, &mut list.custody)?;
            group.terms.values.push(list.value);
            self.add_helper_fact_retained(&helper, group.terms)?;
        }
        drop(_groups_owner);

        let count = group_positions
            .len()
            .checked_add(1)
            .ok_or_else(|| XlogError::allocation_extent_overflow("finite findall output"))?;
        self.acquire_vector::<Term>(count)?;
        for (_, name) in &group_positions {
            self.acquire(name.len() as u64)?;
        }
        let mut terms = Vec::with_capacity(count);
        for (_, name) in &group_positions {
            terms.push(Term::Variable(name.clone()));
        }
        terms.push(out_list);
        Ok(Some(BodyLiteral::Positive(Atom {
            predicate: helper,
            terms,
        })))
    }

    fn expand_maplist(&mut self, atom: &Atom) -> Result<Option<BodyLiteral>> {
        if atom.terms.len() != 2 && atom.terms.len() != 3 {
            return Err(meta_error(
                format_args!(
                    "maplist expects unary or binary form, got {} arguments",
                    atom.terms.len()
                ),
                &mut self.allocation_owner,
            ));
        }
        let pred = match &atom.terms[0] {
            Term::Symbol(id) => retained_symbol(*id, &mut self.allocation_owner)?,
            Term::String(name) | Term::PredRef(name) => {
                let owner = self.reserve_temporary(name.len() as u64)?;
                RetainedTermKey {
                    text: name.clone(),
                    _allocation_owner: owner,
                }
            }
            term => RetainedTermKey {
                text: static_pred_name(term, &mut self.allocation_owner)?,
                _allocation_owner: None,
            },
        };
        let pred_name = &pred.text;
        if self.derived_predicates.contains(pred_name) {
            return Err(meta_error(
                format_args!(
                    "{}",
                    "maplist is limited to finite source facts or literal empty lists"
                ),
                &mut self.allocation_owner,
            ));
        }
        let input_items =
            finite_list_items(&atom.terms[1], "maplist input", &mut self.allocation_owner)?;
        let input_type = self
            .cloned_column_type(pred_name, 0)?
            .unwrap_or_else(|| infer_type_ref_from_terms(input_items));
        let _input_owner = if self.allocation_owner.is_some() {
            let mut bytes =
                crate::compile::push_vector_allocation_bytes::<Term>(input_items.len())?;
            for item in input_items {
                bytes = bytes
                    .checked_add(normalized_value_allocation_bytes(
                        item,
                        Some(&input_type.value),
                    )?)
                    .ok_or_else(|| XlogError::allocation_extent_overflow("finite maplist input"))?;
            }
            self.reserve_temporary(bytes)?
        } else {
            None
        };
        let input_list = Term::List(
            input_items
                .iter()
                .map(|item| {
                    self.normalize_value_term_with_receipt(
                        item,
                        Some(&input_type.value),
                        _input_owner.as_ref(),
                    )
                })
                .collect::<Result<Vec<_>>>()?,
        );
        let maplist_counter = self.maplist_counter;
        self.acquire_format(format_args!(
            "__xlog_meta_maplist_{}_{}",
            pred_name, maplist_counter
        ))?;
        let helper = format!("__xlog_meta_maplist_{}_{}", pred_name, maplist_counter);
        self.maplist_counter += 1;

        if atom.terms.len() == 2 {
            let mut columns = self.helper_columns(1)?;
            let input_type = input_type.into_list(&mut self.allocation_owner)?;
            self.push_helper_column(&mut columns, "input", input_type, false)?;
            self.register_helper_decl(&helper, columns)?;
            if self.maplist_unary_holds(pred_name, &input_list)? {
                let terms = self.capture(|normalizer| {
                    normalizer.acquire_vector::<Term>(1)?;
                    normalizer.acquire_term_clone(&input_list)?;
                    Ok(vec![input_list.clone()])
                })?;
                self.add_helper_fact(&helper, terms)?;
            }
            drop(input_list);
            drop(_input_owner);
            self.acquire_vector::<Term>(1)?;
            self.acquire_term_clone(&atom.terms[1])?;
            return Ok(Some(BodyLiteral::Positive(Atom {
                predicate: helper,
                terms: vec![atom.terms[1].clone()],
            })));
        }

        let output_type = self
            .cloned_column_type(pred_name, 1)?
            .unwrap_or(RetainedTypeRef::scalar(ScalarType::U64));
        let mut columns = self.helper_columns(2)?;
        let input_type = input_type.into_list(&mut self.allocation_owner)?;
        self.push_helper_column(&mut columns, "input", input_type, false)?;
        let helper_output_type =
            RetainedTypeRef::clone_with_owner(&output_type.value, &mut self.allocation_owner)?
                .into_list(&mut self.allocation_owner)?;
        self.push_helper_column(&mut columns, "output", helper_output_type, false)?;
        self.register_helper_decl(&helper, columns)?;
        let mut outputs =
            self.maplist_binary_outputs(pred_name, &input_list, &output_type.value)?;
        drop(output_type);
        for mut output in std::mem::take(&mut outputs.rows) {
            if let (Some(scope), Some(owner)) = (
                self.allocation_owner.as_deref_mut(),
                output.storage_owner.as_ref(),
            ) {
                output.custody.retain_original(owner, scope)?;
            }
            let mut terms = self.capture(|normalizer| {
                normalizer.acquire_vector::<Term>(2)?;
                normalizer.acquire_term_clone(&input_list)?;
                let mut values = Vec::with_capacity(2);
                values.push(input_list.clone());
                Ok(values)
            })?;
            self.append_custody(&mut terms.custody, &mut output.custody)?;
            terms.value.push(Term::List(output.values));
            self.add_helper_fact(&helper, terms)?;
        }
        drop(outputs);
        drop(input_list);
        drop(_input_owner);
        self.acquire_vector::<Term>(2)?;
        self.acquire_term_clone(&atom.terms[1])?;
        self.acquire_term_clone(&atom.terms[2])?;
        Ok(Some(BodyLiteral::Positive(Atom {
            predicate: helper,
            terms: vec![atom.terms[1].clone(), atom.terms[2].clone()],
        })))
    }

    fn maplist_unary_holds(&mut self, pred: &str, input: &Term) -> Result<bool> {
        let Term::List(items) = input else {
            return Err(meta_error(
                format_args!("{}", "maplist input did not normalize to a finite list"),
                &mut self.allocation_owner,
            ));
        };
        let facts = self.cloned_source_facts(pred)?;
        for item in items {
            let mut present = false;
            for fact in &facts.facts {
                if fact.terms.len() == 1 && self.term_keys_equal(&fact.terms[0], item)? {
                    present = true;
                    break;
                }
            }
            if !present {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn maplist_binary_outputs(
        &mut self,
        pred: &str,
        input: &Term,
        output_type: &TypeRef,
    ) -> Result<RetainedTermRows> {
        let Term::List(items) = input else {
            return Err(meta_error(
                format_args!("{}", "maplist input did not normalize to a finite list"),
                &mut self.allocation_owner,
            ));
        };
        if items.is_empty() {
            let allocation_owner =
                self.reserve_temporary(std::mem::size_of::<RetainedTermRow>() as u64)?;
            return Ok(RetainedTermRows {
                rows: vec![RetainedTermRow {
                    values: Vec::new(),
                    custody: PayloadCustody::default(),
                    storage_owner: None,
                }],
                _allocation_owner: allocation_owner,
            });
        }
        let facts = self.cloned_source_facts(pred)?;
        let _choices_owner = if self.allocation_owner.is_some() {
            let outer = crate::compile::push_vector_allocation_bytes::<Vec<RetainedTermValue>>(
                items.len(),
            )?;
            let inner = crate::compile::push_vector_allocation_bytes::<RetainedTermValue>(
                facts.facts.len(),
            )?;
            let bytes = inner
                .checked_mul(items.len() as u64)
                .and_then(|bytes| bytes.checked_add(outer))
                .ok_or_else(|| XlogError::allocation_extent_overflow("finite maplist choices"))?;
            self.reserve_temporary(bytes)?
        } else {
            None
        };
        let mut choices: Vec<Vec<RetainedTermValue>> = Vec::new();
        for item in items {
            let mut outs = Vec::new();
            for fact in &facts.facts {
                if fact.terms.len() == 2 && self.term_keys_equal(&fact.terms[0], item)? {
                    outs.push(
                        self.normalize_value_term_retained(&fact.terms[1], Some(output_type))?,
                    );
                }
            }
            if outs.is_empty() {
                return Ok(RetainedTermRows {
                    rows: Vec::new(),
                    _allocation_owner: None,
                });
            }
            choices.push(outs);
        }
        cartesian_product(&choices, &mut self.allocation_owner)
    }

    fn match_goal_fact(&mut self, goal: &Atom, fact: &Atom) -> Result<Option<RetainedBindings>> {
        let Some(scope) = self.allocation_owner.as_deref_mut() else {
            return Ok(match_goal_fact_core(goal, fact, &mut None)?.map(|values| {
                RetainedBindings {
                    values,
                    _allocation_owner: None,
                }
            }));
        };
        let mut stage = scope.child_scope();
        let Some(original) = match_goal_fact_core(goal, fact, &mut Some(&mut stage))? else {
            return Ok(None);
        };
        let mut bytes = crate::compile::hash_allocation_bytes::<String, Term>(original.len())?;
        for (name, value) in &original {
            let value_bytes = value.owned_allocation_bytes()?;
            bytes = bytes
                .checked_add(name.len() as u64)
                .and_then(|n| n.checked_add(value_bytes))
                .and_then(|n| n.checked_sub(std::mem::size_of::<Term>() as u64))
                .ok_or_else(|| XlogError::allocation_extent_overflow("finite binding clone"))?;
        }
        let allocation_owner = stage.reserve_allocation(bytes)?;
        let mut values =
            HashMap::with_capacity_and_hasher(original.len(), original.hasher().clone());
        for (name, value) in &original {
            values.insert(name.clone(), value.clone());
        }
        drop(original);
        drop(stage);
        Ok(Some(RetainedBindings {
            values,
            _allocation_owner: Some(allocation_owner),
        }))
    }

    fn substitute_template(
        &mut self,
        template: &Term,
        bindings: &HashMap<String, Term>,
        elem_type: &TypeRef,
    ) -> Result<RetainedTermValue> {
        let construction_owner = if self.allocation_owner.is_some() {
            // Completed children and one active child's constructed/normalized
            // copies can coexist. The checked shape includes vector growth;
            // two extents cover those concurrent original scratch allocations.
            let bytes = template_construction_allocation_bytes(template, bindings, elem_type)?
                .checked_mul(2)
                .ok_or_else(|| {
                    XlogError::allocation_extent_overflow("finite template construction")
                })?;
            self.reserve_temporary(bytes)?
        } else {
            None
        };
        let value = self.substitute_template_value(
            template,
            bindings,
            elem_type,
            construction_owner.as_ref(),
        )?;
        // Only this final value survives in the helper fact. Nested normalized
        // values and the constructed template remain under the scratch receipt;
        // registrations made by either normalization keep their original scope.
        let output = self.normalize_value_term_retained(&value, Some(elem_type));
        drop(value);
        drop(construction_owner);
        output
    }

    fn substitute_template_nested(
        &mut self,
        template: &Term,
        bindings: &HashMap<String, Term>,
        elem_type: &TypeRef,
        temporary_receipt: Option<&Arc<dyn Send + Sync>>,
    ) -> Result<Term> {
        let value =
            self.substitute_template_value(template, bindings, elem_type, temporary_receipt)?;
        self.normalize_value_term_with_receipt(&value, Some(elem_type), temporary_receipt)
    }

    fn substitute_template_value(
        &mut self,
        template: &Term,
        bindings: &HashMap<String, Term>,
        elem_type: &TypeRef,
        temporary_receipt: Option<&Arc<dyn Send + Sync>>,
    ) -> Result<Term> {
        let value = match template {
            Term::Variable(name) => {
                let value = bindings.get(name).ok_or_else(|| {
                    meta_error(
                        format_args!(
                            "findall/3 template variable '{}' is not bound by goal",
                            name
                        ),
                        &mut self.allocation_owner,
                    )
                })?;
                if temporary_receipt.is_none() {
                    self.acquire_term_clone(value)?;
                }
                value.clone()
            }
            Term::Anonymous => {
                return Err(meta_error(
                    format_args!("{}", "findall/3 template cannot be the anonymous wildcard"),
                    &mut self.allocation_owner,
                ))
            }
            Term::List(items) => {
                if temporary_receipt.is_none() {
                    self.acquire_vector::<Term>(items.len())?;
                }
                Term::List(
                    items
                        .iter()
                        .map(|item| {
                            self.substitute_template_nested(
                                item,
                                bindings,
                                elem_type,
                                temporary_receipt,
                            )
                        })
                        .collect::<Result<Vec<_>>>()?,
                )
            }
            Term::Compound { functor, args } => {
                if temporary_receipt.is_none() {
                    self.acquire(functor.len() as u64)?;
                    self.acquire_vector::<Term>(args.len())?;
                }
                Term::Compound {
                    functor: functor.clone(),
                    args: args
                        .iter()
                        .map(|arg| {
                            self.substitute_template_nested(
                                arg,
                                bindings,
                                elem_type,
                                temporary_receipt,
                            )
                        })
                        .collect::<Result<Vec<_>>>()?,
                }
            }
            _ => {
                if temporary_receipt.is_none() {
                    self.acquire_term_clone(template)?;
                }
                template.clone()
            }
        };
        Ok(value)
    }

    fn infer_template_type(&mut self, template: &Term, goal: &Atom) -> Result<RetainedTypeRef> {
        match template {
            Term::Variable(name) => {
                for (idx, term) in goal.terms.iter().enumerate() {
                    if matches!(term, Term::Variable(var) if var == name) {
                        return Ok(self
                            .cloned_column_type(&goal.predicate, idx)?
                            .unwrap_or(RetainedTypeRef::scalar(ScalarType::U64)));
                    }
                }
                Err(meta_error(
                    format_args!(
                        "findall/3 template variable '{}' is not bound by the goal",
                        name
                    ),
                    &mut self.allocation_owner,
                ))
            }
            Term::Integer(_) => Ok(RetainedTypeRef::scalar(template.inferred_scalar_type())),
            Term::Float(_) => Ok(RetainedTypeRef::scalar(ScalarType::F64)),
            Term::String(_) | Term::Symbol(_) => Ok(RetainedTypeRef::scalar(ScalarType::Symbol)),
            Term::Compound { .. } | Term::PredRef(_) | Term::List(_) | Term::Cons { .. } => {
                Ok(RetainedTypeRef::term())
            }
            Term::Anonymous | Term::Aggregate(_) => Err(meta_error(
                format_args!(
                    "{}",
                    "findall/3 template must be a finite scalar or term expression"
                ),
                &mut self.allocation_owner,
            )),
        }
    }

    fn term_id_or_bound_variable(
        &mut self,
        term: &Term,
        bound: &HashMap<String, ScalarType>,
        context: &str,
    ) -> Result<Term> {
        match term {
            Term::Variable(name) if bound.contains_key(name) => {
                self.acquire_term_clone(term)?;
                Ok(term.clone())
            }
            Term::Variable(_) | Term::Anonymous => Err(meta_error(
                format_args!("{context} requires a known finite term before runtime inspection"),
                &mut self.allocation_owner,
            )),
            _ => Ok(Term::Integer(
                self.register_term(term, ExpectedTerm::Any)? as i128
            )),
        }
    }

    fn normalize_univ_parts(&mut self, term: &Term) -> Result<Term> {
        match term {
            Term::Variable(_) | Term::Anonymous => {
                self.acquire_term_clone(term)?;
                Ok(term.clone())
            }
            Term::List(items) => {
                self.acquire_vector::<Term>(items.len())?;
                let mut values = Vec::with_capacity(items.len());
                for item in items {
                    values.push(Term::Integer(
                        self.register_term(item, ExpectedTerm::Any)? as i128
                    ));
                }
                Ok(Term::List(values))
            }
            _ => Err(meta_error(
                format_args!(
                    "{}",
                    "univ parts must be a finite list literal or a list variable"
                ),
                &mut self.allocation_owner,
            )),
        }
    }

    fn normalize_symbol_argument(&mut self, term: &Term, context: &str) -> Result<Term> {
        match term {
            Term::Variable(_) | Term::Anonymous | Term::Symbol(_) | Term::String(_) => {
                self.acquire_term_clone(term)?;
                Ok(term.clone())
            }
            _ => Err(meta_error(
                format_args!("{context} must be a symbol or variable"),
                &mut self.allocation_owner,
            )),
        }
    }

    fn normalize_u32_argument(&mut self, term: &Term, context: &str) -> Result<Term> {
        match term {
            Term::Variable(_) | Term::Anonymous => {
                self.acquire_term_clone(term)?;
                Ok(term.clone())
            }
            Term::Integer(value) if *value >= 0 && *value <= u32::MAX as i128 => {
                self.acquire_term_clone(term)?;
                Ok(term.clone())
            }
            _ => Err(meta_error(
                format_args!("{context} must be a u32 integer or variable"),
                &mut self.allocation_owner,
            )),
        }
    }

    fn is_ground(&self, term: &Term, bound: &HashMap<String, ScalarType>) -> bool {
        match term {
            Term::Variable(name) => bound.contains_key(name),
            Term::Anonymous => false,
            Term::List(items) => items.iter().all(|item| self.is_ground(item, bound)),
            Term::Cons { head, tail } => self.is_ground(head, bound) && self.is_ground(tail, bound),
            Term::Compound { args, .. } => args.iter().all(|arg| self.is_ground(arg, bound)),
            Term::Aggregate(_) => false,
            Term::Integer(_)
            | Term::Float(_)
            | Term::String(_)
            | Term::Symbol(_)
            | Term::PredRef(_) => true,
        }
    }

    fn is_var(&self, term: &Term, bound: &HashMap<String, ScalarType>) -> bool {
        match term {
            Term::Variable(name) => !bound.contains_key(name),
            Term::Anonymous => true,
            _ => false,
        }
    }

    fn is_nonvar(&self, term: &Term, bound: &HashMap<String, ScalarType>) -> bool {
        !self.is_var(term, bound)
    }

    fn record_bound_from_literal(
        &self,
        lit: &BodyLiteral,
        bound: &mut HashMap<String, ScalarType>,
    ) {
        let BodyLiteral::Positive(atom) = lit else {
            return;
        };
        for (idx, term) in atom.terms.iter().enumerate() {
            if let Term::Variable(name) = term {
                let typ = Self::column_type(&self.pred_columns, &atom.predicate, idx)
                    .and_then(|typ| self.storage_type_for_type_ref(typ))
                    .unwrap_or(ScalarType::U64);
                bound.insert(name.clone(), typ);
            }
        }
    }

    fn register_term(&mut self, term: &Term, expected: ExpectedTerm) -> Result<u64> {
        let registered = match term {
            Term::Integer(value) if expected != ExpectedTerm::PredRef => {
                RetainedRegisteredTerm {
                    key: retained_format(format_args!("i:{value}"), &mut self.allocation_owner)?,
                    kind: RegisteredTerm::Scalar,
                    allocation_owners: [None, None],
                }
            }
            Term::Float(value) if expected != ExpectedTerm::PredRef => {
                RetainedRegisteredTerm {
                    key: retained_format(format_args!("f:{}", value.to_bits()), &mut self.allocation_owner)?,
                    kind: RegisteredTerm::Scalar,
                    allocation_owners: [None, None],
                }
            }
            Term::String(value) if expected != ExpectedTerm::PredRef => {
                RetainedRegisteredTerm {
                    key: retained_format(format_args!("s:{value}"), &mut self.allocation_owner)?,
                    kind: RegisteredTerm::Scalar,
                    allocation_owners: [None, None],
                }
            }
            Term::Symbol(id) if expected == ExpectedTerm::PredRef => {
                let name = retained_symbol(*id, &mut self.allocation_owner)?;
                RetainedRegisteredTerm {
                    key: retained_format(format_args!("predref:{}", name.text), &mut self.allocation_owner)?,
                    kind: RegisteredTerm::PredRef,
                    allocation_owners: [None, None],
                }
            }
            Term::String(name) if expected == ExpectedTerm::PredRef => {
                RetainedRegisteredTerm {
                    key: retained_format(format_args!("predref:{name}"), &mut self.allocation_owner)?,
                    kind: RegisteredTerm::PredRef,
                    allocation_owners: [None, None],
                }
            }
            Term::PredRef(name) => {
                RetainedRegisteredTerm {
                    key: retained_format(format_args!("predref:{name}"), &mut self.allocation_owner)?,
                    kind: RegisteredTerm::PredRef,
                    allocation_owners: [None, None],
                }
            }
            Term::Symbol(id) => {
                let name = retained_symbol(*id, &mut self.allocation_owner)?;
                RetainedRegisteredTerm {
                    key: retained_format(format_args!("sym:{}", name.text), &mut self.allocation_owner)?,
                    kind: RegisteredTerm::Scalar,
                    allocation_owners: [None, None],
                }
            }
            Term::Compound { functor, args } if expected != ExpectedTerm::PredRef => {
                let arguments_owner = if self.allocation_owner.is_some() {
                    self.reserve_temporary((args.len() as u64)
                        .checked_mul(std::mem::size_of::<u64>() as u64)
                        .ok_or_else(|| XlogError::allocation_extent_overflow("meta term arguments"))?)?
                } else { None };
                let mut arg_ids = Vec::with_capacity(args.len());
                for arg in args {
                    arg_ids.push(self.register_term(arg, ExpectedTerm::Any)?);
                }
                let symbol_id = self.intern_symbol(functor)?;
                let functor_id = self.register_term(&Term::Symbol(symbol_id), ExpectedTerm::Any)?;
                let parts_owner = if self.allocation_owner.is_some() {
                    let count = args.len().checked_add(1).ok_or_else(|| {
                        XlogError::allocation_extent_overflow("meta term parts")
                    })?;
                    let bytes = crate::compile::push_vector_allocation_bytes::<u64>(count)?
                        .checked_add(functor.len() as u64)
                        .ok_or_else(|| XlogError::allocation_extent_overflow("meta term parts"))?;
                    self.reserve_temporary(bytes)?
                } else { None };
                let mut parts = vec![functor_id];
                parts.extend(arg_ids.iter().copied());
                let _joined_owner = if self.allocation_owner.is_some() {
                    let overflow = || XlogError::allocation_extent_overflow("meta term text");
                    let mut bytes = crate::compile::push_vector_allocation_bytes::<String>(arg_ids.len())?;
                    let mut joined = arg_ids.len().saturating_sub(1) as u64;
                    for id in &arg_ids {
                        bytes = bytes.checked_add(crate::compile::formatted_text_allocation_bytes(format_args!("{id}"))?)
                            .ok_or_else(overflow)?;
                        joined = joined.checked_add(xlog_core::error::formatted_text_extent(format_args!("{id}"))?)
                            .ok_or_else(overflow)?;
                    }
                    self.reserve_temporary(bytes.checked_add(joined).ok_or_else(overflow)?)?
                } else { None };
                let joined = join_ids(&arg_ids);
                RetainedRegisteredTerm {
                    key: retained_format(format_args!("compound:{functor}({joined})"), &mut self.allocation_owner)?,
                    kind: RegisteredTerm::Compound {
                        functor: functor.clone(),
                        args: arg_ids,
                        parts,
                    },
                    allocation_owners: [arguments_owner, parts_owner],
                }
            }
            Term::List(_) | Term::Cons { .. } => {
                return Err(meta_error(format_args!("{}", "finite list terms inside term values are reserved for a later term-value encoding path"), &mut self.allocation_owner))
            }
            Term::Variable(_) | Term::Anonymous => {
                return Err(meta_error(format_args!("{}", "finite term registration requires a ground source term"), &mut self.allocation_owner))
            }
            Term::Aggregate(_) => {
                return Err(meta_error(format_args!("{}", "aggregate terms cannot be registered as finite meta terms"), &mut self.allocation_owner))
            }
            Term::Compound { .. } => {
                return Err(meta_error(format_args!("{}", "predref columns require a static predicate reference, not a compound term"), &mut self.allocation_owner))
            }
            Term::Integer(_) | Term::Float(_) | Term::String(_) => {
                return Err(meta_error(format_args!("{}", "predref columns require a static predicate reference"), &mut self.allocation_owner))
            }
        };

        if expected == ExpectedTerm::Compound
            && !matches!(registered.kind, RegisteredTerm::Compound { .. })
        {
            return Err(meta_error(
                format_args!("{}", "compound column requires a finite compound term"),
                &mut self.allocation_owner,
            ));
        }
        if expected == ExpectedTerm::PredRef && !matches!(registered.kind, RegisteredTerm::PredRef)
        {
            return Err(meta_error(
                format_args!("{}", "predref column requires a static predicate reference"),
                &mut self.allocation_owner,
            ));
        }

        if let Some(id) = self.term_ids.get(&registered.key.text) {
            return Ok(id.id);
        }
        let id = self.next_term_id;
        self.next_term_id += 1;
        let term_ids_storage = self.reserve_hash_growth::<String, RegisteredTermId>(
            self.term_ids.len(),
            self.term_ids.capacity(),
        )?;
        let term_records_storage = self.reserve_vector_growth::<TermRecord>(
            self.term_records.len(),
            self.term_records.capacity(),
        )?;
        self.term_ids.insert(
            registered.key.text,
            RegisteredTermId {
                id,
                _key_allocation: registered.key._allocation_owner,
            },
        );
        if let Some(storage) = term_ids_storage {
            self._term_ids_storage = Some(storage);
        }
        self.term_records.push(TermRecord {
            id,
            kind: registered.kind,
            _allocation_owners: registered.allocation_owners,
        });
        if let Some(storage) = term_records_storage {
            self._term_records_storage = Some(storage);
        }
        Ok(id)
    }

    fn append_helpers(&mut self, program: &mut RetainedProgram) -> Result<()> {
        self.append_term_helper_facts()?;

        let existing_owner = if self.allocation_owner.is_some() {
            let overflow = || XlogError::allocation_extent_overflow("meta helper output");
            let mut bytes = crate::compile::hash_allocation_bytes::<String, ()>(
                program.value.predicates.len(),
            )?;
            for predicate in &program.value.predicates {
                bytes = bytes
                    .checked_add(predicate.name.len() as u64)
                    .ok_or_else(overflow)?;
            }
            self.reserve_temporary(bytes)?
        } else {
            None
        };
        let existing: HashSet<String> = program
            .value
            .predicates
            .iter()
            .map(|pred| pred.name.clone())
            .collect();

        if self.allocation_owner.is_some() {
            let overflow = || XlogError::allocation_extent_overflow("meta helper output");
            let additions = self
                .helper_decls
                .keys()
                .filter(|name| !existing.contains(*name))
                .count();
            let scope = self
                .allocation_owner
                .as_deref_mut()
                .expect("original helper allocation scope");
            program
                .custody
                .predicates
                .prepare_elements(additions, scope)?;
            program
                .custody
                .rules
                .prepare_elements(self.helper_facts.len(), scope)?;
            let predicates = program
                .value
                .predicates
                .len()
                .checked_add(additions)
                .ok_or_else(overflow)?;
            if predicates > program.value.predicates.capacity() {
                let owner =
                    scope.reserve_allocation(meta_storage_bytes::<PredDecl>(predicates)?)?;
                program.value.predicates.reserve_exact(additions);
                program.custody.predicates.backing_owner = Some(owner);
            }
            let rules = program
                .value
                .rules
                .len()
                .checked_add(self.helper_facts.len())
                .ok_or_else(overflow)?;
            if rules > program.value.rules.capacity() {
                let owner = scope.reserve_allocation(meta_storage_bytes::<Rule>(rules)?)?;
                program.value.rules.reserve_exact(self.helper_facts.len());
                program.custody.rules.backing_owner = Some(owner);
            }
        }

        let declarations = RetainedHelperDeclarations {
            values: std::mem::take(&mut self.helper_decls),
            storage: std::mem::take(&mut self.helper_decls_storage),
        };
        for (name, mut declaration) in declarations.values {
            let name = RetainedTermKey {
                text: name,
                _allocation_owner: declaration.key_owner.take(),
            };
            if existing.contains(&name.text) {
                continue;
            }
            let emitted = self.helper_declaration(&name.text, declaration.columns)?;
            program.value.predicates.push(emitted.value);
            if self.allocation_owner.is_some() {
                program.custody.predicates.elements.push(emitted.custody);
            }
        }
        drop(declarations.storage);
        drop(existing);
        drop(existing_owner);
        let custody = std::mem::take(&mut self.helper_facts_custody);
        let payloads = custody.elements.into_iter();
        let values = std::mem::take(&mut self.helper_facts).into_iter();
        if self.allocation_owner.is_some() {
            for (value, payload) in values.zip(payloads) {
                program.value.rules.push(value);
                program.custody.rules.elements.push(payload);
            }
        } else {
            program.value.rules.extend(values);
        }
        Ok(())
    }

    fn append_term_helper_facts(&mut self) -> Result<()> {
        let columns = self.scalar_columns(&[
            ("term_id", TERM_ID_TYPE),
            ("name", ScalarType::Symbol),
            ("arity", ScalarType::U32),
        ])?;
        self.register_helper_decl(functor_pred(), columns)?;
        let columns = self.univ_columns()?;
        self.register_helper_decl(univ_pred(), columns)?;
        let columns = self.scalar_columns(&[
            ("term_id", TERM_ID_TYPE),
            ("idx", ScalarType::U32),
            ("arg_id", TERM_ID_TYPE),
        ])?;
        self.register_helper_decl(arg_pred(), columns)?;

        let _records_storage = self._term_records_storage.take();
        for record in std::mem::take(&mut self.term_records) {
            if let RegisteredTerm::Compound {
                functor,
                args,
                parts,
            } = record.kind
            {
                let functor_id = self.intern_symbol(&functor)?;
                let terms = self.capture(|normalizer| {
                    normalizer.acquire_vector::<Term>(3)?;
                    Ok(vec![
                        Term::Integer(record.id as i128),
                        Term::Symbol(functor_id),
                        Term::Integer(args.len() as i128),
                    ])
                })?;
                self.add_helper_fact(functor_pred(), terms)?;
                let terms = self.capture(|normalizer| {
                    normalizer.acquire_vector::<Term>(2)?;
                    normalizer.acquire_vector::<Term>(parts.len())?;
                    let mut values = Vec::with_capacity(parts.len());
                    for id in parts {
                        values.push(Term::Integer(id as i128));
                    }
                    Ok(vec![Term::Integer(record.id as i128), Term::List(values)])
                })?;
                self.add_helper_fact(univ_pred(), terms)?;
                for (idx, arg_id) in args.into_iter().enumerate() {
                    let terms = self.capture(|normalizer| {
                        normalizer.acquire_vector::<Term>(3)?;
                        Ok(vec![
                            Term::Integer(record.id as i128),
                            Term::Integer(idx as i128),
                            Term::Integer(arg_id as i128),
                        ])
                    })?;
                    self.add_helper_fact(arg_pred(), terms)?;
                }
            }
        }
        Ok(())
    }

    fn register_helper_decl(&mut self, name: &str, columns: RetainedHelperColumns) -> Result<()> {
        if self.helper_decls.contains_key(name) {
            return Ok(());
        }
        let key_owner = self.reserve_temporary(name.len() as u64)?;
        if let Some(scope) = self.allocation_owner.as_deref_mut() {
            self.helper_decls_storage.reserve_unit(
                xlog_core::btree_insert_allocation_bytes::<String, RegisteredHelperDeclaration>(
                    self.helper_decls.len(),
                )?,
                scope,
            )?;
        }
        self.helper_decls.insert(
            name.to_string(),
            RegisteredHelperDeclaration { columns, key_owner },
        );
        Ok(())
    }

    fn helper_declaration(
        &mut self,
        name: &str,
        mut columns: RetainedHelperColumns,
    ) -> Result<RetainedHelperEmission> {
        let name_custody = match self.allocation_owner.as_deref_mut() {
            Some(scope) => PayloadCustody::for_text_clone(name, scope)?,
            None => PayloadCustody::default(),
        };
        let mut types_custody: Option<RetainedVectorCustody<PayloadCustody>> =
            match self.allocation_owner.as_deref_mut() {
                Some(scope) => Some(RetainedVectorCustody::for_capacity::<TypeRef>(
                    columns.value.len(),
                    scope,
                )?),
                None => None,
            };
        let mut types = Vec::with_capacity(columns.value.len());
        for column in &columns.value {
            let typ = RetainedTypeRef::clone_with_owner(&column.typ, &mut self.allocation_owner)?;
            types.push(typ.value);
            if let Some(custody) = types_custody.as_mut() {
                custody.elements.push(typ.custody);
            }
        }
        let custody = if self.allocation_owner.is_some() {
            PredicateCustody {
                name: name_custody,
                types: types_custody.take().expect("original helper type custody"),
                columns: columns
                    .custody
                    .take()
                    .expect("original helper column custody"),
            }
        } else {
            PredicateCustody::default()
        };
        Ok(RetainedHelperEmission {
            value: PredDecl {
                name: name.to_string(),
                types,
                columns: columns.value,
                is_private: true,
            },
            custody,
        })
    }

    fn add_helper_fact(&mut self, pred: &str, terms: RetainedMetaValue<Vec<Term>>) -> Result<()> {
        self.add_helper_fact_retained(
            pred,
            RetainedTermRow {
                values: terms.value,
                custody: terms.custody,
                storage_owner: None,
            },
        )
    }

    fn add_helper_fact_retained(&mut self, pred: &str, mut terms: RetainedTermRow) -> Result<()> {
        let key = retained_format(
            format_args!("{pred}:{:?}", terms.values),
            &mut self.allocation_owner,
        )?;
        if self.helper_fact_keys.contains_key(&key.text) {
            return Ok(());
        }
        let key_storage = self.reserve_hash_growth::<String, Option<Arc<dyn Send + Sync>>>(
            self.helper_fact_keys.len(),
            self.helper_fact_keys.capacity(),
        )?;
        self.helper_fact_keys
            .insert(key.text, key._allocation_owner);
        if let Some(storage) = key_storage {
            self._helper_fact_keys_storage = Some(storage);
        }
        let mut name = self.capture(|normalizer| {
            normalizer.acquire(pred.len() as u64)?;
            Ok(pred.to_string())
        })?;
        self.append_custody(&mut terms.custody, &mut name.custody)?;
        if let (Some(scope), Some(owner)) = (
            self.allocation_owner.as_deref_mut(),
            terms.storage_owner.as_ref(),
        ) {
            terms.custody.retain_original(owner, scope)?;
        }
        if let Some(scope) = self.allocation_owner.as_deref_mut() {
            self.helper_facts_custody.prepare_elements(1, scope)?;
        }
        let facts_storage = self
            .reserve_vector_growth::<Rule>(self.helper_facts.len(), self.helper_facts.capacity())?;
        self.helper_facts.push(Rule {
            head: Atom {
                predicate: name.value,
                terms: terms.values,
            },
            body: vec![],
        });
        if self.allocation_owner.is_some() {
            self.helper_facts_custody.elements.push(terms.custody);
        }
        if let Some(storage) = facts_storage {
            self.helper_facts_custody.backing_owner = Some(storage);
        }
        Ok(())
    }

    fn fail_atom(&mut self) -> Result<Atom> {
        let columns = self.scalar_columns(&[("flag", ScalarType::U32)])?;
        self.register_helper_decl(fail_pred(), columns)?;
        self.acquire(fail_pred().len() as u64)?;
        self.acquire_vector::<Term>(1)?;
        Ok(Atom {
            predicate: fail_pred().to_string(),
            terms: vec![Term::Integer(1)],
        })
    }

    fn storage_type_for_type_ref(&self, typ: &TypeRef) -> Option<ScalarType> {
        match typ {
            TypeRef::Scalar(ty) => Some(*ty),
            TypeRef::Domain(name) => self.domains.get(name).copied(),
            TypeRef::List(_) | TypeRef::Term | TypeRef::Compound | TypeRef::PredRef => {
                Some(TERM_ID_TYPE)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum MetaTruthKind {
    Ground,
    Var,
    NonVar,
}

fn require_arity(
    atom: &Atom,
    expected: usize,
    owner: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> Result<()> {
    if atom.terms.len() == expected {
        Ok(())
    } else {
        Err(meta_error(
            format_args!(
                "{} expects {} arguments, got {}",
                atom.predicate,
                expected,
                atom.terms.len()
            ),
            owner,
        ))
    }
}

fn compound_goal_atom(
    term: &Term,
    context: &str,
    owner: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> Result<Atom> {
    match term {
        Term::Compound { functor, args } => Ok(Atom {
            predicate: functor.clone(),
            terms: args.clone(),
        }),
        _ => Err(meta_error(
            format_args!("{context} goal must be a finite atom expression"),
            owner,
        )),
    }
}

fn finite_list_items<'a>(
    term: &'a Term,
    context: &str,
    owner: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> Result<&'a [Term]> {
    match term {
        Term::List(items) => Ok(items),
        _ => Err(meta_error(
            format_args!("{context} requires a finite list literal in the safe meta subset"),
            owner,
        )),
    }
}

fn static_pred_name(
    term: &Term,
    owner: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> Result<String> {
    match term {
        Term::Symbol(id) => Ok(symbol::resolve(*id)),
        Term::String(name) | Term::PredRef(name) => Ok(name.clone()),
        Term::Variable(_) | Term::Anonymous => Err(meta_error(format_args!("{}", "maplist requires a static predicate reference; runtime-variable predicate names are rejected"), owner)),
        _ => Err(meta_error(format_args!("{}", "maplist predicate argument must be a static predicate reference"), owner)),
    }
}

/// Return the statically referenced predicate used by a safe meta-predicate.
///
/// Invalid or dynamic forms are diagnosed by the normalizer itself. This
/// helper exposes only references that the accepted finite forms resolve.
pub(crate) fn static_meta_predicate_dependency(atom: &Atom) -> Option<String> {
    match atom.predicate.as_str() {
        "findall" => atom
            .terms
            .get(1)
            .and_then(|term| compound_goal_atom(term, "findall/3", &mut None).ok())
            .map(|goal| goal.predicate),
        "maplist" => atom
            .terms
            .first()
            .and_then(|term| static_pred_name(term, &mut None).ok()),
        _ => None,
    }
}

/// Heap storage of the same normalized value, excluding independent registry
/// allocations. Push-built list buffers include their reached growth storage.
fn normalized_value_allocation_bytes(term: &Term, expected: Option<&TypeRef>) -> Result<u64> {
    let overflow = || XlogError::allocation_extent_overflow("finite meta value");
    let add = |left: u64, right: u64| left.checked_add(right).ok_or_else(overflow);
    let clone_bytes = |term: &Term| {
        term.owned_allocation_bytes()
            .map(|bytes| bytes - std::mem::size_of::<Term>() as u64)
    };
    let inner = match expected {
        Some(TypeRef::Term | TypeRef::Compound | TypeRef::PredRef) => {
            return if term.is_any_variable() {
                clone_bytes(term)
            } else {
                Ok(0)
            };
        }
        Some(TypeRef::List(inner)) => Some(inner.as_ref()),
        _ => None,
    };
    match term {
        Term::List(items) => {
            let mut bytes = crate::compile::push_vector_allocation_bytes::<Term>(items.len())?;
            for item in items {
                bytes = add(bytes, normalized_value_allocation_bytes(item, inner)?)?;
            }
            Ok(bytes)
        }
        Term::Cons { head, tail } => {
            let bytes = add(
                2 * std::mem::size_of::<Term>() as u64,
                normalized_value_allocation_bytes(head, inner)?,
            )?;
            add(
                bytes,
                normalized_value_allocation_bytes(
                    tail,
                    if inner.is_some() { expected } else { None },
                )?,
            )
        }
        Term::Variable(_) | Term::Anonymous | Term::Integer(_) => clone_bytes(term),
        _ if inner.is_some() => Ok(0),
        _ => clone_bytes(term),
    }
}

/// Borrowed construction geometry for the original recursive substitution.
/// Missing bindings and invalid nodes allocate no value; the shared producer
/// remains responsible for their unchanged classified diagnostics.
fn template_construction_allocation_bytes(
    template: &Term,
    bindings: &HashMap<String, Term>,
    elem_type: &TypeRef,
) -> Result<u64> {
    let overflow = || XlogError::allocation_extent_overflow("finite template construction");
    let add = |left: u64, right: u64| left.checked_add(right).ok_or_else(overflow);
    match template {
        Term::Variable(name) => match bindings.get(name) {
            Some(value) => add(
                value.owned_allocation_bytes()? - std::mem::size_of::<Term>() as u64,
                normalized_value_allocation_bytes(value, Some(elem_type))?,
            ),
            None => Ok(0),
        },
        Term::Anonymous => Ok(0),
        Term::List(items) => {
            let mut bytes = crate::compile::push_vector_allocation_bytes::<Term>(items.len())?;
            for item in items {
                bytes = add(
                    bytes,
                    template_construction_allocation_bytes(item, bindings, elem_type)?,
                )?;
            }
            Ok(bytes)
        }
        Term::Compound { functor, args } => {
            let mut bytes = add(
                functor.len() as u64,
                crate::compile::push_vector_allocation_bytes::<Term>(args.len())?,
            )?;
            for arg in args {
                bytes = add(
                    bytes,
                    template_construction_allocation_bytes(arg, bindings, elem_type)?,
                )?;
            }
            Ok(bytes)
        }
        _ => add(
            template.owned_allocation_bytes()? - std::mem::size_of::<Term>() as u64,
            normalized_value_allocation_bytes(template, Some(elem_type))?,
        ),
    }
}

fn infer_type_ref_from_terms(items: &[Term]) -> RetainedTypeRef {
    if let Some(first) = items.first() {
        match first {
            Term::Integer(_) => RetainedTypeRef::scalar(first.inferred_scalar_type()),
            Term::Float(_) => RetainedTypeRef::scalar(ScalarType::F64),
            Term::String(_) | Term::Symbol(_) => RetainedTypeRef::scalar(ScalarType::Symbol),
            Term::Compound { .. } | Term::PredRef(_) | Term::List(_) | Term::Cons { .. } => {
                RetainedTypeRef::term()
            }
            Term::Variable(_) | Term::Anonymous | Term::Aggregate(_) => {
                RetainedTypeRef::scalar(ScalarType::U64)
            }
        }
    } else {
        RetainedTypeRef::scalar(ScalarType::U64)
    }
}

fn cartesian_product(
    choices: &[Vec<RetainedTermValue>],
    allocate: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> Result<RetainedTermRows> {
    let rows_owner = allocate
        .as_deref_mut()
        .map(|scope| scope.reserve_allocation(std::mem::size_of::<RetainedTermRow>() as u64))
        .transpose()?;
    let mut rows = RetainedTermRows {
        rows: vec![RetainedTermRow {
            values: Vec::new(),
            custody: PayloadCustody::default(),
            storage_owner: None,
        }],
        _allocation_owner: rows_owner,
    };
    for choice in choices {
        let next_rows = rows
            .rows
            .len()
            .checked_mul(choice.len())
            .ok_or_else(|| XlogError::allocation_extent_overflow("finite meta product"))?;
        let next_owner = if let Some(allocate) = allocate.as_deref_mut() {
            Some(allocate.reserve_allocation(meta_storage_bytes::<RetainedTermRow>(next_rows)?)?)
        } else {
            None
        };
        let mut next = RetainedTermRows {
            rows: Vec::with_capacity(next_rows),
            _allocation_owner: next_owner,
        };
        for row in &rows.rows {
            for value in choice {
                let count =
                    row.values.len().checked_add(1).ok_or_else(|| {
                        XlogError::allocation_extent_overflow("finite meta product")
                    })?;
                let mut custody = PayloadCustody::default();
                if let Some(scope) = allocate.as_deref_mut() {
                    custody.reserve_unit(meta_storage_bytes::<Term>(count)?, scope)?;
                }
                let mut next_row = RetainedTermRow {
                    values: Vec::with_capacity(count),
                    custody,
                    storage_owner: None,
                };
                for term in row.values.iter().chain(std::iter::once(&value.value)) {
                    if let Some(scope) = allocate.as_deref_mut() {
                        let mut original = PayloadCustody::for_term_clone(term, scope)?;
                        next_row.custody.append_from(&mut original, scope)?;
                    }
                    next_row.values.push(term.clone());
                }
                next.rows.push(next_row);
            }
        }
        // The old batch and each of its row receipts retire together. The new
        // batch was acquired before any clone while all original rows lived.
        rows = next;
    }
    Ok(rows)
}

fn joined_text_bytes(items: &[String], separator: &str) -> Result<u64> {
    let overflow = || XlogError::allocation_extent_overflow("meta joined text");
    let mut bytes = items
        .len()
        .saturating_sub(1)
        .checked_mul(separator.len())
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(overflow)?;
    for item in items {
        bytes = bytes.checked_add(item.len() as u64).ok_or_else(overflow)?;
    }
    Ok(bytes)
}

fn format_owned(
    arguments: std::fmt::Arguments<'_>,
    owner: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> Result<String> {
    if let Some(reserve) = owner.as_deref_mut() {
        reserve.acquire(crate::compile::formatted_text_allocation_bytes(arguments)?)?;
    }
    Ok(format!("{arguments}"))
}

fn retained_format(
    arguments: std::fmt::Arguments<'_>,
    owner: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> Result<RetainedTermKey> {
    let allocation_owner = owner
        .as_deref_mut()
        .map(|scope| {
            scope.reserve_allocation(crate::compile::formatted_text_allocation_bytes(arguments)?)
        })
        .transpose()?;
    Ok(RetainedTermKey {
        text: format!("{arguments}"),
        _allocation_owner: allocation_owner,
    })
}

fn retained_symbol(
    id: u32,
    owner: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> Result<RetainedTermKey> {
    match owner.as_deref_mut() {
        Some(scope) => {
            let (text, allocation_owner) =
                symbol::resolve_with_allocation_owner(id, &mut |bytes| {
                    scope.reserve_allocation(
                        u64::try_from(bytes)
                            .map_err(|_| XlogError::allocation_extent_overflow("meta symbol"))?,
                    )
                })?;
            Ok(RetainedTermKey {
                text,
                _allocation_owner: Some(allocation_owner),
            })
        }
        None => Ok(RetainedTermKey {
            text: symbol::resolve(id),
            _allocation_owner: None,
        }),
    }
}

fn retained_joined_keys(
    keys: &[RetainedTermKey],
    separator: &str,
    owner: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> Result<RetainedTermKey> {
    let overflow = || XlogError::allocation_extent_overflow("finite joined keys");
    let mut length = keys
        .len()
        .saturating_sub(1)
        .checked_mul(separator.len())
        .ok_or_else(overflow)?;
    for key in keys {
        length = length.checked_add(key.text.len()).ok_or_else(overflow)?;
    }
    let allocation_owner = owner
        .as_deref_mut()
        .map(|scope| scope.reserve_allocation(u64::try_from(length).map_err(|_| overflow())?))
        .transpose()?;
    let mut text = String::with_capacity(length);
    for (index, key) in keys.iter().enumerate() {
        if index != 0 {
            text.push_str(separator);
        }
        text.push_str(&key.text);
    }
    Ok(RetainedTermKey {
        text,
        _allocation_owner: allocation_owner,
    })
}

fn match_goal_fact_core(
    goal: &Atom,
    fact: &Atom,
    owner: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> Result<Option<HashMap<String, Term>>> {
    if owner.is_some() {
        let mut bytes = crate::compile::hash_allocation_bytes::<String, Term>(goal.terms.len())?;
        for (goal_term, fact_term) in goal.terms.iter().zip(&fact.terms) {
            if let Term::Variable(name) = goal_term {
                let term_bytes = fact_term.owned_allocation_bytes()?;
                bytes = bytes
                    .checked_add(name.len() as u64)
                    .and_then(|n| n.checked_add(term_bytes))
                    .ok_or_else(|| XlogError::allocation_extent_overflow("finite binding"))?;
            }
        }
        owner
            .as_deref_mut()
            .expect("original matching scope")
            .acquire(bytes)?;
    }
    let mut bindings = HashMap::with_capacity(if owner.is_some() { goal.terms.len() } else { 0 });
    for (goal_term, fact_term) in goal.terms.iter().zip(&fact.terms) {
        match goal_term {
            Term::Variable(name) => {
                if let Some(existing) = bindings.get(name) {
                    if !term_keys_equal(existing, fact_term, owner)? {
                        return Ok(None);
                    }
                } else {
                    bindings.insert(name.clone(), fact_term.clone());
                }
            }
            Term::Anonymous => {}
            _ => {
                if !term_keys_equal(goal_term, fact_term, owner)? {
                    return Ok(None);
                }
            }
        }
    }
    Ok(Some(bindings))
}

fn retained_term_key(
    term: &Term,
    owner: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> Result<RetainedTermKey> {
    let Some(scope) = owner.as_deref_mut() else {
        return Ok(RetainedTermKey {
            text: term_match_key(term, &mut None)?,
            _allocation_owner: None,
        });
    };
    let mut stage = scope.child_scope();
    let original = term_match_key(term, &mut Some(&mut stage))?;
    let allocation_owner = stage.reserve_allocation(original.len() as u64)?;
    let text = original.clone();
    drop(original);
    drop(stage);
    Ok(RetainedTermKey {
        text,
        _allocation_owner: Some(allocation_owner),
    })
}

fn term_keys_equal(
    left: &Term,
    right: &Term,
    owner: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> Result<bool> {
    let left = retained_term_key(left, owner)?;
    let right = retained_term_key(right, owner)?;
    Ok(left.text == right.text)
}

fn term_match_key(
    term: &Term,
    owner: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> Result<String> {
    match term {
        Term::Variable(name) => format_owned(format_args!("var:{name}"), owner),
        Term::Anonymous => format_owned(format_args!("_"), owner),
        Term::Integer(value) => format_owned(format_args!("i:{value}"), owner),
        Term::Float(value) => format_owned(format_args!("f:{}", value.to_bits()), owner),
        Term::String(value) => format_owned(format_args!("s:{value}"), owner),
        Term::Symbol(id) => {
            let text =
                match owner.as_deref_mut() {
                    Some(reserve) => {
                        symbol::resolve_with_allocation_owner(*id, &mut |bytes| {
                            reserve.acquire(u64::try_from(bytes).map_err(|_| {
                                XlogError::allocation_extent_overflow("meta symbol")
                            })?)
                        })?
                        .0
                    }
                    None => symbol::resolve(*id),
                };
            format_owned(format_args!("sym:{text}"), owner)
        }
        Term::List(items) => {
            if let Some(reserve) = owner.as_deref_mut() {
                reserve.acquire(
                    (items.len() as u64)
                        .checked_mul(std::mem::size_of::<String>() as u64)
                        .ok_or_else(|| XlogError::allocation_extent_overflow("meta key vector"))?,
                )?;
            }
            let keys = items
                .iter()
                .map(|item| term_match_key(item, owner))
                .collect::<Result<Vec<_>>>()?;
            if let Some(reserve) = owner.as_deref_mut() {
                reserve.acquire(joined_text_bytes(&keys, ",")?)?;
            }
            let joined = keys.join(",");
            format_owned(format_args!("list:[{joined}]"), owner)
        }
        Term::Cons { head, tail } => {
            let head = term_match_key(head, owner)?;
            let tail = term_match_key(tail, owner)?;
            format_owned(format_args!("cons:{head}|{tail}"), owner)
        }
        Term::Compound { functor, args } => {
            if let Some(reserve) = owner.as_deref_mut() {
                reserve.acquire(
                    (args.len() as u64)
                        .checked_mul(std::mem::size_of::<String>() as u64)
                        .ok_or_else(|| XlogError::allocation_extent_overflow("meta key vector"))?,
                )?;
            }
            let keys = args
                .iter()
                .map(|argument| term_match_key(argument, owner))
                .collect::<Result<Vec<_>>>()?;
            if let Some(reserve) = owner.as_deref_mut() {
                reserve.acquire(joined_text_bytes(&keys, ",")?)?;
            }
            let joined = keys.join(",");
            format_owned(format_args!("compound:{functor}({joined})"), owner)
        }
        Term::PredRef(name) => format_owned(format_args!("predref:{name}"), owner),
        Term::Aggregate(agg) => {
            format_owned(format_args!("agg:{:?}:{}", agg.op, agg.variable), owner)
        }
    }
}

fn lower_meta_type_refs_in_predicates(program: &mut Program) {
    for pred in &mut program.predicates {
        for typ in &mut pred.types {
            lower_meta_type_ref(typ);
        }
        for col in &mut pred.columns {
            lower_meta_type_ref(&mut col.typ);
        }
    }
}

fn lower_meta_type_ref(typ: &mut TypeRef) {
    match typ {
        TypeRef::List(inner) => lower_meta_type_ref(inner),
        TypeRef::Term | TypeRef::Compound | TypeRef::PredRef => {
            *typ = TypeRef::Scalar(TERM_ID_TYPE);
        }
        TypeRef::Scalar(_) | TypeRef::Domain(_) => {}
    }
}

fn join_ids(ids: &[u64]) -> String {
    ids.iter().map(u64::to_string).collect::<Vec<_>>().join(",")
}

/// Does any body literal name a meta predicate? Only then can `findall`/
/// `maplist` expansion consult the source facts.
fn program_uses_meta_predicates(program: &Program) -> bool {
    let body_uses = |body: &[BodyLiteral]| {
        body.iter().any(|lit| match lit {
            BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => {
                is_meta_predicate(&atom.predicate)
            }
            _ => false,
        })
    };
    program.rules.iter().any(|rule| body_uses(&rule.body))
        || program
            .constraints
            .iter()
            .any(|constraint| body_uses(&constraint.body))
        || program
            .learnable_rules
            .iter()
            .any(|learnable| body_uses(&learnable.body))
}

fn is_meta_predicate(name: &str) -> bool {
    matches!(
        name,
        "ground" | "var" | "nonvar" | "functor" | "findall" | "maplist"
    )
}

fn is_blocked_dynamic_predicate(name: &str) -> bool {
    matches!(name, "call" | "assert" | "asserta" | "assertz" | "retract")
}

fn dynamic_predicate_error(
    name: &str,
    owner: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> XlogError {
    match name {
        "call" => meta_error(
            format_args!("{}", "dynamic call/N is outside the safe meta subset"),
            owner,
        ),
        "assert" | "asserta" | "assertz" | "retract" => meta_error(
            format_args!(
                "{}",
                "dynamic database mutation is outside the safe meta subset"
            ),
            owner,
        ),
        _ => meta_error(
            format_args!("{}", "unsupported dynamic meta predicate"),
            owner,
        ),
    }
}

fn functor_pred() -> &'static str {
    "__xlog_meta_functor"
}

fn univ_pred() -> &'static str {
    "__xlog_meta_univ"
}

fn arg_pred() -> &'static str {
    "__xlog_meta_arg"
}

fn fail_pred() -> &'static str {
    "__xlog_meta_fail"
}

fn meta_error(
    message: std::fmt::Arguments<'_>,
    owner: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> XlogError {
    let bytes = xlog_core::error::formatted_text_allocation_bytes(format_args!(
        "meta normalization error: {message}"
    ));
    let producer = match owner.as_deref_mut() {
        Some(scope) => xlog_core::CompilationAllocationScope::prepare_diagnostic(
            bytes,
            Some(&mut |bytes| scope.reserve_allocation(bytes)),
        ),
        None => xlog_core::CompilationAllocationScope::prepare_diagnostic(bytes, None),
    };
    match producer {
        Ok(producer) => producer
            .finish(|| XlogError::Compilation(format!("meta normalization error: {message}"))),
        Err(error) => error,
    }
}
