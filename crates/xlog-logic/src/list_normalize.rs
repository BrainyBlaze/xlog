//! Finite list normalization for the language-completeness surface.

use std::collections::HashMap;
use std::sync::Arc;

use xlog_core::{Result, ScalarType, XlogError};

use crate::ast::{
    Atom, BodyLiteral, Comparison, PredColumn, PredDecl, Program, Query, Rule, Term, TypeRef,
};
use crate::program_custody::{
    ColumnCustody, PayloadCustody, PredicateCustody, ProgramAllocationCustody, RetainedProgram,
    RetainedVectorCustody,
};

const LIST_ID_TYPE: ScalarType = ScalarType::U64;

/// Normalize finite list syntax into scalar list identifiers plus helper relations.
///
/// The normalized program uses ordinary XLOG relations:
///
/// - `__xlog_list_len(list_id, len)`
/// - `__xlog_list_item_<type>(list_id, index, value)`
/// - `__xlog_list_cons_<type>(list_id, head, tail_id)`
/// - selected built-in helper relations such as `append`, `sort`, `msort`, and `list_to_set`
///
/// This keeps accepted list programs on the existing relational lowering/runtime path.
pub fn normalize_list_builtins(program: &Program) -> Result<Program> {
    normalize_list_builtins_owned(program.clone())
}

/// [`normalize_list_builtins`] taking the program by value.
///
/// Avoids cloning the whole AST: facts without list literals are moved through
/// untouched (the pass is the identity on them, see
/// `ListNormalizer::fact_is_identity`), everything else is rebuilt as before.
pub fn normalize_list_builtins_owned(program: Program) -> Result<Program> {
    normalize_list_builtins_with_owner(program, None)
}

/// Run the same finite-list normalizer with original allocation custody.
pub fn normalize_list_builtins_owned_with_allocation_owner(
    program: Program,
    allocations: &mut crate::compile::CompilationAllocationScope<'_>,
) -> Result<Program> {
    normalize_list_builtins_with_owner(program, Some(allocations))
}

fn normalize_list_builtins_with_owner(
    program: Program,
    allocate: Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> Result<Program> {
    let mut normalizer = ListNormalizer::new(&program, allocate)?;
    let output = normalizer.normalize_program(
        RetainedProgram {
            value: program,
            custody: ProgramAllocationCustody::default(),
        },
        false,
    )?;
    normalizer.release_raw_output(output)
}

pub(crate) fn normalize_list_builtins_retained(
    program: RetainedProgram,
    allocations: &mut crate::compile::CompilationAllocationScope<'_>,
) -> Result<RetainedProgram> {
    let mut normalizer = ListNormalizer::new(&program.value, Some(allocations))?;
    normalizer.normalize_program(program, true)
}

struct RetainedOutput<T> {
    value: T,
    custody: PayloadCustody,
    storage_owner: Option<Arc<dyn Send + Sync>>,
}

struct TemporaryOutputs<T> {
    values: Vec<RetainedOutput<T>>,
    storage_owner: Option<Arc<dyn Send + Sync>>,
}

struct OriginalRules {
    values: std::vec::IntoIter<Rule>,
    payloads: std::vec::IntoIter<PayloadCustody>,
    _backing_owner: Option<Arc<dyn Send + Sync>>,
    _metadata_owner: Option<Arc<dyn Send + Sync>>,
}

struct RetainedRuleBuffer {
    values: Vec<Rule>,
    custody: RetainedVectorCustody<PayloadCustody>,
}

struct RawProgramOwner {
    _custody: ProgramAllocationCustody,
    _metadata_owner: Arc<dyn Send + Sync>,
}

struct RawProgramOutput {
    value: Program,
    owner: Arc<RawProgramOwner>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ListOp {
    Append,
    Sort,
    MSort,
    ListToSet,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum LiteralKey {
    U32(u32),
    U64(u64),
    I32(i32),
    I64(i64),
    F32(u32),
    F64(u64),
    Bool(bool),
    Symbol(String),
}

struct RetainedLiteralKey {
    value: LiteralKey,
    allocation_owner: Option<Arc<dyn Send + Sync>>,
}

#[derive(Debug, Clone)]
struct ListDef {
    id: u64,
    elem_type: ScalarType,
    items: Vec<Term>,
    keys: Vec<LiteralKey>,
}

struct RetainedListDef {
    value: ListDef,
    _items_owner: Option<Arc<dyn Send + Sync>>,
    _keys_owner: Option<Arc<dyn Send + Sync>>,
    _key_allocations: Vec<Arc<dyn Send + Sync>>,
    _key_allocation_storage: Option<Arc<dyn Send + Sync>>,
}

struct RetainedListId {
    id: u64,
    _key_owner: Option<Arc<dyn Send + Sync>>,
}

struct TemporaryListItems {
    items: Vec<Term>,
    storage_owner: Option<Arc<dyn Send + Sync>>,
    payload_owners: [Option<Arc<dyn Send + Sync>>; 2],
}

struct RetainedHelperDeclaration {
    value: PredDecl,
    custody: PredicateCustody,
}

struct RetainedHelperPredicates {
    values: HashMap<String, Option<Arc<dyn Send + Sync>>>,
    _storage_owner: Option<Arc<dyn Send + Sync>>,
}

#[derive(Debug, Clone)]
struct ListArg {
    term: Term,
    id: Option<u64>,
    elem_type: ScalarType,
    is_literal: bool,
}

#[derive(Debug, Clone, Copy)]
struct ListColumn {
    elem_type: ScalarType,
}

struct ListNormalizer<'a, 'pool> {
    /// Declared `list<T>` columns per predicate: `(column index, column)`.
    list_columns: HashMap<String, Vec<(usize, ListColumn)>>,
    lists_by_key: HashMap<(ScalarType, Vec<LiteralKey>), RetainedListId>,
    lists: Vec<RetainedListDef>,
    next_list_id: u64,
    helper_preds: HashMap<String, Option<Arc<dyn Send + Sync>>>,
    operation_rows: HashMap<(ListOp, ScalarType, Vec<u64>), Option<Arc<dyn Send + Sync>>>,
    fresh_counter: usize,
    allocation_owner: Option<&'a mut crate::compile::CompilationAllocationScope<'pool>>,
    _lists_by_key_owner: Option<Arc<dyn Send + Sync>>,
    _lists_owner: Option<Arc<dyn Send + Sync>>,
    _helper_preds_owner: Option<Arc<dyn Send + Sync>>,
    _operation_rows_owner: Option<Arc<dyn Send + Sync>>,
    _environment_owner: Option<Arc<dyn Send + Sync>>,
}

impl<'a, 'pool> ListNormalizer<'a, 'pool> {
    fn reserve_temporary(&mut self, bytes: u64) -> Result<Option<Arc<dyn Send + Sync>>> {
        if bytes == 0 {
            return Ok(None);
        }
        self.allocation_owner
            .as_deref_mut()
            .map(|scope| scope.reserve_allocation(bytes))
            .transpose()
    }

    fn acquire(&mut self, payload: &mut PayloadCustody, bytes: u64) -> Result<()> {
        if let Some(scope) = self.allocation_owner.as_deref_mut() {
            payload.reserve_unit(bytes, scope)?;
        }
        Ok(())
    }

    fn acquire_vector<T>(&mut self, payload: &mut PayloadCustody, count: usize) -> Result<()> {
        let bytes = count
            .checked_mul(std::mem::size_of::<T>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| XlogError::allocation_extent_overflow("list output storage"))?;
        self.acquire(payload, bytes)
    }

    fn acquire_term(&mut self, payload: &mut PayloadCustody, term: &Term) -> Result<()> {
        if let Some(scope) = self.allocation_owner.as_deref_mut() {
            let mut original = PayloadCustody::for_term_clone(term, scope)?;
            payload.append_from(&mut original, scope)?;
        }
        Ok(())
    }

    fn acquire_literal(
        &mut self,
        payload: &mut PayloadCustody,
        literal: &BodyLiteral,
    ) -> Result<()> {
        if let Some(scope) = self.allocation_owner.as_deref_mut() {
            let mut original = PayloadCustody::for_literal_clone(literal, scope)?;
            payload.append_from(&mut original, scope)?;
        }
        Ok(())
    }

    fn absorb<T>(
        &mut self,
        payload: &mut PayloadCustody,
        output: &mut RetainedOutput<T>,
    ) -> Result<()> {
        if let Some(scope) = self.allocation_owner.as_deref_mut() {
            if let Some(owner) = output.storage_owner.as_ref() {
                payload.retain_original(owner, scope)?;
            }
            payload.append_from(&mut output.custody, scope)?;
        }
        output.storage_owner = None;
        Ok(())
    }

    fn reserve_output_vector<T>(
        &mut self,
        vector: &mut Vec<T>,
        owner: &mut Option<Arc<dyn Send + Sync>>,
        additional: usize,
    ) -> Result<()> {
        if self.allocation_owner.is_some() && vector.capacity() - vector.len() < additional {
            let bytes = vector
                .len()
                .checked_add(additional)
                .and_then(|count| count.checked_mul(std::mem::size_of::<T>()))
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or_else(|| XlogError::allocation_extent_overflow("list output growth"))?;
            let next_owner = self.reserve_temporary(bytes)?;
            vector.reserve_exact(additional);
            *owner = next_owner;
        }
        Ok(())
    }

    fn prepare_program_vector<T, C>(
        &mut self,
        values: &mut Vec<T>,
        custody: &mut RetainedVectorCustody<C>,
        additional: usize,
    ) -> Result<()> {
        if let Some(scope) = self.allocation_owner.as_deref_mut() {
            custody.prepare_elements(additional, scope)?;
        } else {
            custody.elements.reserve(additional);
            values.reserve(additional);
        }
        self.reserve_output_vector(values, &mut custody.backing_owner, additional)
    }

    fn push_temporary<T>(
        &mut self,
        values: &mut Vec<RetainedOutput<T>>,
        owner: &mut Option<Arc<dyn Send + Sync>>,
        value: RetainedOutput<T>,
    ) -> Result<()> {
        let next_owner = if values.len() == values.capacity() {
            let count = values
                .len()
                .checked_add(1)
                .ok_or_else(|| XlogError::allocation_extent_overflow("list expansion roster"))?;
            self.reserve_temporary(xlog_core::push_vector_storage_bytes::<RetainedOutput<T>>(
                count,
            )?)?
        } else {
            None
        };
        values.push(value);
        if next_owner.is_some() {
            *owner = next_owner;
        }
        Ok(())
    }

    fn push_rule(
        &mut self,
        values: &mut Vec<Rule>,
        custody: &mut RetainedVectorCustody<PayloadCustody>,
        mut rule: RetainedOutput<Rule>,
    ) -> Result<()> {
        self.prepare_program_vector(values, custody, 1)?;
        if let (Some(scope), Some(owner)) = (
            self.allocation_owner.as_deref_mut(),
            rule.storage_owner.as_ref(),
        ) {
            rule.custody.retain_original(owner, scope)?;
        }
        values.push(rule.value);
        custody.elements.push(rule.custody);
        Ok(())
    }

    fn release_raw_output(&mut self, output: RetainedProgram) -> Result<Program> {
        let Some(scope) = self.allocation_owner.as_deref_mut() else {
            return Ok(output.value);
        };
        let metadata = std::mem::size_of::<RawProgramOwner>()
            .checked_add(2 * std::mem::size_of::<usize>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| XlogError::allocation_extent_overflow("list output owner"))?;
        let metadata_owner = scope.reserve_allocation(metadata)?;
        let output = RawProgramOutput {
            value: output.value,
            owner: Arc::new(RawProgramOwner {
                _custody: output.custody,
                _metadata_owner: metadata_owner,
            }),
        };
        scope.retain(Arc::clone(&output.owner) as Arc<dyn Send + Sync>)?;
        Ok(output.value)
    }

    fn items_clone_bytes(items: &[Term]) -> Result<u64> {
        items.iter().try_fold(0u64, |n, item| {
            n.checked_add(item.owned_allocation_bytes()?)
                .ok_or_else(|| XlogError::allocation_extent_overflow("list clone"))
        })
    }

    fn keys_clone_bytes(keys: &[LiteralKey]) -> Result<u64> {
        let mut bytes = keys
            .len()
            .checked_mul(std::mem::size_of::<LiteralKey>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| XlogError::allocation_extent_overflow("list key clone"))?;
        for key in keys {
            if let LiteralKey::Symbol(text) = key {
                bytes = bytes
                    .checked_add(text.len() as u64)
                    .ok_or_else(|| XlogError::allocation_extent_overflow("list key"))?;
            }
        }
        Ok(bytes)
    }

    fn reserve_list_clone(&mut self, id: u64) -> Result<Option<Arc<dyn Send + Sync>>> {
        if self.allocation_owner.is_some() {
            let list = Self::list_by_id(&self.lists, id, &mut self.allocation_owner)?;
            let bytes = Self::items_clone_bytes(&list.items)?
                .checked_add(Self::keys_clone_bytes(&list.keys)?)
                .ok_or_else(|| XlogError::allocation_extent_overflow("list clone"))?;
            return self.reserve_temporary(bytes);
        }
        Ok(None)
    }

    fn clone_list_items(&mut self, id: u64) -> Result<TemporaryListItems> {
        let (storage_bytes, payload_bytes) = if self.allocation_owner.is_some() {
            let list = Self::list_by_id(&self.lists, id, &mut self.allocation_owner)?;
            let storage_bytes = list
                .items
                .len()
                .checked_mul(std::mem::size_of::<Term>())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or_else(|| XlogError::allocation_extent_overflow("list input storage"))?;
            (
                storage_bytes,
                Self::items_clone_bytes(&list.items)? - storage_bytes,
            )
        } else {
            (0, 0)
        };
        let storage_owner = self.reserve_temporary(storage_bytes)?;
        let payload_owner = self.reserve_temporary(payload_bytes)?;
        let items = Self::list_by_id(&self.lists, id, &mut self.allocation_owner)?
            .items
            .clone();
        Ok(TemporaryListItems {
            items,
            storage_owner,
            payload_owners: [payload_owner, None],
        })
    }

    fn extend_list_items(
        &mut self,
        items: &mut TemporaryListItems,
        mut tail: TemporaryListItems,
    ) -> Result<()> {
        if self.allocation_owner.is_some()
            && items.items.capacity() - items.items.len() < tail.items.len()
        {
            let bytes = items
                .items
                .len()
                .checked_add(tail.items.len())
                .and_then(|count| count.checked_mul(std::mem::size_of::<Term>()))
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or_else(|| XlogError::allocation_extent_overflow("list input growth"))?;
            let next_owner = self.reserve_temporary(bytes)?;
            items.items.reserve_exact(tail.items.len());
            items.storage_owner = next_owner;
        }
        items.payload_owners[1] = tail.payload_owners[0].take();
        items.items.extend(std::mem::take(&mut tail.items));
        drop(tail);
        Ok(())
    }

    fn reserve_lists_clone(&mut self) -> Result<Option<Arc<dyn Send + Sync>>> {
        if self.allocation_owner.is_some() {
            let mut bytes = self
                .lists
                .len()
                .checked_mul(std::mem::size_of::<ListDef>())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or_else(|| XlogError::allocation_extent_overflow("list snapshot"))?;
            for list in &self.lists {
                let list = &list.value;
                let keys_bytes = Self::keys_clone_bytes(&list.keys)?;
                bytes = bytes
                    .checked_add(Self::items_clone_bytes(&list.items)?)
                    .and_then(|n| n.checked_add(keys_bytes))
                    .ok_or_else(|| XlogError::allocation_extent_overflow("list roster"))?;
            }
            return self.reserve_temporary(bytes);
        }
        Ok(None)
    }

    fn reserve_hash_growth<K, V>(
        &mut self,
        length: usize,
        capacity: usize,
    ) -> Result<Option<Arc<dyn Send + Sync>>> {
        if self.allocation_owner.is_some() && length == capacity {
            let bytes = crate::compile::hash_allocation_bytes::<K, V>(
                length
                    .checked_add(1)
                    .ok_or_else(|| XlogError::allocation_extent_overflow("list table"))?,
            )?;
            return self.reserve_temporary(bytes);
        }
        Ok(None)
    }

    fn insert_helper_pred(&mut self, pred: &str) -> Result<()> {
        if self.helper_preds.contains_key(pred) {
            return Ok(());
        }
        let payload_owner = self.reserve_temporary(pred.len() as u64)?;
        let owner = self.reserve_hash_growth::<String, Option<Arc<dyn Send + Sync>>>(
            self.helper_preds.len(),
            self.helper_preds.capacity(),
        )?;
        self.helper_preds.insert(pred.to_string(), payload_owner);
        if owner.is_some() {
            self._helper_preds_owner = owner;
        }
        Ok(())
    }

    fn bind_list_type(
        &mut self,
        types: &mut HashMap<String, ScalarType>,
        name: &str,
        typ: ScalarType,
    ) -> Result<()> {
        types.insert(name.to_string(), typ);
        Ok(())
    }

    fn reserve_list_bindings(
        &mut self,
        mut visit: impl FnMut(&mut dyn FnMut(&str)),
    ) -> Result<(usize, Option<Arc<dyn Send + Sync>>)> {
        if self.allocation_owner.is_none() {
            return Ok((0, None));
        }
        let mut extent = Some((0usize, 0u64));
        visit(&mut |name| {
            extent = extent.and_then(|(count, bytes)| {
                Some((count.checked_add(1)?, bytes.checked_add(name.len() as u64)?))
            });
        });
        let (count, text) = extent
            .ok_or_else(|| XlogError::allocation_extent_overflow("list variable bindings"))?;
        // Each original variable can bind once in argument normalization and
        // once again in its enclosing list helper. The map has at most the
        // original variable occurrence count; generated helper variables are
        // output terms, never keys in this local type environment.
        let bytes =
            crate::compile::hash_allocation_bytes::<String, ScalarType>(count)?
                .checked_add(text.checked_mul(2).ok_or_else(|| {
                    XlogError::allocation_extent_overflow("list variable bindings")
                })?)
                .ok_or_else(|| XlogError::allocation_extent_overflow("list variable bindings"))?;
        Ok((count, self.reserve_temporary(bytes)?))
    }

    fn acquire_format(
        &mut self,
        payload: &mut PayloadCustody,
        args: std::fmt::Arguments<'_>,
    ) -> Result<()> {
        if self.allocation_owner.is_some() {
            self.acquire(
                payload,
                crate::compile::formatted_text_allocation_bytes(args)?,
            )?;
        }
        Ok(())
    }

    fn new(
        program: &Program,
        mut allocation_owner: Option<&'a mut crate::compile::CompilationAllocationScope<'pool>>,
    ) -> Result<Self> {
        use crate::compile::hash_allocation_bytes;
        let overflow = || XlogError::allocation_extent_overflow("list declaration");
        let add = |left: u64, right: u64| left.checked_add(right).ok_or_else(overflow);
        let vector = |count: usize, width: usize| {
            count
                .checked_mul(width)
                .and_then(|n| u64::try_from(n).ok())
                .ok_or_else(overflow)
        };
        let _rosters_owner = if let Some(allocate) = allocation_owner.as_deref_mut() {
            let bytes = add(
                hash_allocation_bytes::<&str, ScalarType>(program.domains.len())?,
                hash_allocation_bytes::<&str, usize>(program.predicates.len())?,
            )?;
            if bytes == 0 {
                None
            } else {
                Some(allocate.reserve_allocation(bytes)?)
            }
        } else {
            None
        };
        let mut domain_roster: HashMap<&str, ScalarType> =
            HashMap::with_capacity(program.domains.len());
        for domain in &program.domains {
            domain_roster.insert(domain.name.as_str(), domain.typ);
        }
        let mut column_counts: HashMap<&str, usize> =
            HashMap::with_capacity(program.predicates.len());
        for pred in &program.predicates {
            let count = if pred.columns.is_empty() {
                pred.types
                    .iter()
                    .filter(|typ| matches!(typ, TypeRef::List(_)))
                    .count()
            } else {
                pred.columns
                    .iter()
                    .filter(|column| matches!(&column.typ, TypeRef::List(_)))
                    .count()
            };
            if count != 0 {
                let total = column_counts.entry(pred.name.as_str()).or_default();
                *total = total.checked_add(count).ok_or_else(overflow)?;
            }
        }
        let _domains_owner = if let Some(allocate) = allocation_owner.as_deref_mut() {
            let mut bytes = hash_allocation_bytes::<String, ScalarType>(domain_roster.len())?;
            for name in domain_roster.keys() {
                bytes = add(bytes, name.len() as u64)?;
            }
            if bytes == 0 {
                None
            } else {
                Some(allocate.reserve_allocation(bytes)?)
            }
        } else {
            None
        };
        let mut domains: HashMap<String, ScalarType> = HashMap::with_capacity(domain_roster.len());
        for (name, typ) in domain_roster {
            domains.insert(name.to_string(), typ);
        }
        let environment_owner = if let Some(allocate) = allocation_owner.as_deref_mut() {
            let mut bytes =
                hash_allocation_bytes::<String, Vec<(usize, ListColumn)>>(column_counts.len())?;
            for (name, count) in &column_counts {
                bytes = add(bytes, name.len() as u64)?;
                bytes = add(
                    bytes,
                    vector(*count, std::mem::size_of::<(usize, ListColumn)>())?,
                )?;
            }
            if bytes == 0 {
                None
            } else {
                Some(allocate.reserve_allocation(bytes)?)
            }
        } else {
            None
        };
        let mut list_columns: HashMap<String, Vec<(usize, ListColumn)>> =
            HashMap::with_capacity(column_counts.len());
        for (name, count) in column_counts {
            list_columns.insert(name.to_string(), Vec::with_capacity(count));
        }
        for pred in &program.predicates {
            let _schema_owner = if let Some(allocate) = allocation_owner.as_deref_mut() {
                let count = if pred.columns.is_empty() {
                    pred.types.len()
                } else {
                    pred.columns.len()
                };
                let mut bytes = vector(count, std::mem::size_of::<PredColumn>())?;
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
                        if let Some(name) = &column.name {
                            bytes = add(bytes, name.len() as u64)?;
                        }
                    }
                }
                if bytes == 0 {
                    None
                } else {
                    Some(allocate.reserve_allocation(bytes)?)
                }
            } else {
                None
            };
            let schema = pred.schema_columns();
            for (idx, col) in schema.iter().enumerate() {
                if let TypeRef::List(inner) = &col.typ {
                    let elem_type =
                        Self::storage_type_for_type_ref(inner, &domains, &mut allocation_owner)?;
                    list_columns
                        .get_mut(pred.name.as_str())
                        .expect("original list column roster")
                        .push((idx, ListColumn { elem_type }));
                }
            }
        }

        Ok(Self {
            list_columns,
            lists_by_key: HashMap::new(),
            lists: Vec::new(),
            next_list_id: 1,
            helper_preds: HashMap::new(),
            operation_rows: HashMap::new(),
            fresh_counter: 0,
            allocation_owner,
            _lists_by_key_owner: None,
            _lists_owner: None,
            _helper_preds_owner: None,
            _operation_rows_owner: None,
            _environment_owner: environment_owner,
        })
    }

    /// True when list normalization is the identity on this fact: with an
    /// empty body only the head is rewritten, and `normalize_value_term` only
    /// touches `Term::List`/`Term::Cons` (everything else is cloned as is).
    /// Such facts are moved through without being rebuilt.
    fn fact_is_identity(rule: &Rule) -> bool {
        rule.body.is_empty()
            && !rule
                .head
                .terms
                .iter()
                .any(|term| matches!(term, Term::List(_) | Term::Cons { .. }))
    }

    fn normalize_program(
        &mut self,
        mut out: RetainedProgram,
        has_input_custody: bool,
    ) -> Result<RetainedProgram> {
        let count = out.value.rules.len();
        // Raw entries leave original input allocations with their caller.
        // Only their ownership roster is constructed here; typed entries
        // move the real original per-record receipts through unchanged.
        if !has_input_custody {
            if let Some(scope) = self.allocation_owner.as_deref_mut() {
                out.custody.rules.prepare_elements(count, scope)?;
            }
            out.custody
                .rules
                .elements
                .resize_with(count, PayloadCustody::default);
        }
        let next_custody = match self.allocation_owner.as_deref_mut() {
            Some(scope) => RetainedVectorCustody::for_capacity::<Rule>(count, scope)?,
            None => RetainedVectorCustody {
                elements: Vec::with_capacity(count),
                ..RetainedVectorCustody::default()
            },
        };
        let mut rules = RetainedRuleBuffer {
            values: Vec::with_capacity(count),
            custody: next_custody,
        };
        let original_custody = std::mem::take(&mut out.custody.rules);
        let mut original = OriginalRules {
            values: std::mem::take(&mut out.value.rules).into_iter(),
            payloads: original_custody.elements.into_iter(),
            _backing_owner: original_custody.backing_owner,
            _metadata_owner: original_custody.elements_storage_owner,
        };
        while let Some(value) = original.values.next() {
            let rule = RetainedOutput {
                value,
                custody: original
                    .payloads
                    .next()
                    .expect("original list rule custody"),
                storage_owner: None,
            };
            if Self::fact_is_identity(&rule.value) {
                self.push_rule(&mut rules.values, &mut rules.custody, rule)?;
            } else {
                let expanded = self.normalize_rule(&rule.value)?;
                for expanded_rule in expanded.values {
                    self.push_rule(&mut rules.values, &mut rules.custody, expanded_rule)?;
                }
                drop(expanded.storage_owner);
                drop(rule);
            }
        }
        drop(original);
        out.value.rules = rules.values;
        out.custody.rules = rules.custody;

        if !has_input_custody && !out.value.constraints.is_empty() {
            if let Some(scope) = self.allocation_owner.as_deref_mut() {
                out.custody
                    .constraints
                    .prepare_elements(out.value.constraints.len(), scope)?;
            }
            out.custody
                .constraints
                .elements
                .resize_with(out.value.constraints.len(), PayloadCustody::default);
        }
        for (constraint, custody) in out
            .value
            .constraints
            .iter_mut()
            .zip(&mut out.custody.constraints.elements)
        {
            let mut normalized = self.normalize_body(&constraint.body)?;
            if let (Some(scope), Some(owner)) = (
                self.allocation_owner.as_deref_mut(),
                normalized.storage_owner.as_ref(),
            ) {
                normalized.custody.retain_original(owner, scope)?;
            }
            constraint.body = normalized.value;
            *custody = normalized.custody;
        }
        if !has_input_custody && !out.value.queries.is_empty() {
            if let Some(scope) = self.allocation_owner.as_deref_mut() {
                out.custody
                    .queries
                    .prepare_elements(out.value.queries.len(), scope)?;
            }
            out.custody
                .queries
                .elements
                .resize_with(out.value.queries.len(), PayloadCustody::default);
        }
        for (query, custody) in out
            .value
            .queries
            .iter_mut()
            .zip(&mut out.custody.queries.elements)
        {
            let mut normalized = self.normalize_query(query)?;
            if let (Some(scope), Some(owner)) = (
                self.allocation_owner.as_deref_mut(),
                normalized.storage_owner.as_ref(),
            ) {
                normalized.custody.retain_original(owner, scope)?;
            }
            *query = normalized.value;
            *custody = normalized.custody;
        }

        self.append_helper_declarations_and_facts(&mut out)?;
        Ok(out)
    }

    fn normalize_query(&mut self, query: &Query) -> Result<RetainedOutput<Query>> {
        let (capacity, _bindings_owner) =
            self.reserve_list_bindings(|visit| query.atom.visit_variables(visit))?;
        let mut var_list_types = HashMap::with_capacity(capacity);
        let mut atoms = self.expand_atom(&query.atom, false, &mut var_list_types)?;
        let mut selected = atoms.values.drain(..).next();
        drop(atoms);
        let atom = match selected.take() {
            Some(atom) => atom,
            None => {
                let custody = match self.allocation_owner.as_deref_mut() {
                    Some(scope) => PayloadCustody::for_atom_clone(&query.atom, scope)?,
                    None => PayloadCustody::default(),
                };
                RetainedOutput {
                    value: query.atom.clone(),
                    custody,
                    storage_owner: None,
                }
            }
        };
        Ok(RetainedOutput {
            value: Query { atom: atom.value },
            custody: atom.custody,
            storage_owner: atom.storage_owner,
        })
    }

    fn normalize_rule(&mut self, rule: &Rule) -> Result<TemporaryOutputs<Rule>> {
        let (capacity, _bindings_owner) = self.reserve_list_bindings(|visit| {
            for literal in &rule.body {
                literal.visit_variables(visit);
            }
        })?;
        let mut var_list_types = HashMap::with_capacity(capacity);
        let mut head = self.normalize_head_atom(&rule.head)?;
        let mut body = self.normalize_body_with_env(&rule.body, &mut var_list_types)?;
        self.absorb(&mut head.custody, &mut body)?;
        let storage_owner =
            self.reserve_temporary(std::mem::size_of::<RetainedOutput<Rule>>() as u64)?;
        Ok(TemporaryOutputs {
            values: vec![RetainedOutput {
                value: Rule {
                    head: head.value,
                    body: body.value,
                },
                custody: head.custody,
                storage_owner: None,
            }],
            storage_owner,
        })
    }

    fn normalize_body(&mut self, body: &[BodyLiteral]) -> Result<RetainedOutput<Vec<BodyLiteral>>> {
        let (capacity, _bindings_owner) = self.reserve_list_bindings(|visit| {
            for literal in body {
                literal.visit_variables(visit);
            }
        })?;
        let mut var_list_types = HashMap::with_capacity(capacity);
        self.normalize_body_with_env(body, &mut var_list_types)
    }

    fn normalize_body_with_env(
        &mut self,
        body: &[BodyLiteral],
        var_list_types: &mut HashMap<String, ScalarType>,
    ) -> Result<RetainedOutput<Vec<BodyLiteral>>> {
        let mut payload = PayloadCustody::default();
        let mut storage_owner = None;
        let mut out = Vec::new();
        for lit in body {
            self.reserve_output_vector(&mut out, &mut storage_owner, 1)?;
            match lit {
                BodyLiteral::Positive(atom) => {
                    let expanded = self.expand_atom(atom, false, var_list_types)?;
                    for mut atom in expanded.values {
                        self.reserve_output_vector(&mut out, &mut storage_owner, 1)?;
                        self.absorb(&mut payload, &mut atom)?;
                        out.push(BodyLiteral::Positive(atom.value));
                    }
                    drop(expanded.storage_owner);
                }
                BodyLiteral::Negated(atom) => {
                    let mut atoms = self.expand_atom(atom, true, var_list_types)?;
                    if atoms.values.len() != 1 {
                        return Err(list_error(
                            format_args!(
                                "{}",
                                "negated list pattern expansion would require multiple atoms"
                            ),
                            &mut self.allocation_owner,
                        ));
                    }
                    let mut atom = atoms.values.pop().unwrap();
                    self.absorb(&mut payload, &mut atom)?;
                    out.push(BodyLiteral::Negated(atom.value));
                }
                BodyLiteral::Comparison(cmp) => {
                    out.push(BodyLiteral::Comparison(Comparison {
                        op: cmp.op,
                        left: self.normalize_value_term(&cmp.left, None, &mut payload)?,
                        right: self.normalize_value_term(&cmp.right, None, &mut payload)?,
                    }));
                }
                BodyLiteral::Epistemic(epistemic) => {
                    self.acquire_literal(&mut payload, lit)?;
                    out.push(BodyLiteral::Epistemic(epistemic.clone()));
                }
                BodyLiteral::IsExpr(is_expr) => {
                    self.acquire_literal(&mut payload, lit)?;
                    out.push(BodyLiteral::IsExpr(is_expr.clone()));
                }
                BodyLiteral::Univ(_) => {
                    return Err(list_error(format_args!("{}", "univ literals must be normalized by the meta-term pass before list normalization"), &mut self.allocation_owner));
                }
            }
        }
        Ok(RetainedOutput {
            value: out,
            custody: payload,
            storage_owner,
        })
    }

    /// Element type of the declared `list<T>` column `idx` of `pred`, if any.
    fn list_column_elem(&self, pred: &str, idx: usize) -> Option<ScalarType> {
        self.list_columns
            .get(pred)
            .and_then(|cols| cols.iter().find(|(i, _)| *i == idx))
            .map(|(_, col)| col.elem_type)
    }

    fn normalize_head_atom(&mut self, atom: &Atom) -> Result<RetainedOutput<Atom>> {
        let mut payload = PayloadCustody::default();
        self.acquire_vector::<Term>(&mut payload, atom.terms.len())?;
        self.acquire(&mut payload, atom.predicate.len() as u64)?;
        let mut terms = Vec::with_capacity(atom.terms.len());
        for (idx, term) in atom.terms.iter().enumerate() {
            let expected = self.list_column_elem(&atom.predicate, idx);
            terms.push(self.normalize_value_term(term, expected, &mut payload)?);
        }
        Ok(RetainedOutput {
            value: Atom {
                predicate: atom.predicate.clone(),
                terms,
            },
            custody: payload,
            storage_owner: None,
        })
    }

    fn single_atom(&mut self, atom: RetainedOutput<Atom>) -> Result<TemporaryOutputs<Atom>> {
        let storage_owner =
            self.reserve_temporary(std::mem::size_of::<RetainedOutput<Atom>>() as u64)?;
        Ok(TemporaryOutputs {
            values: vec![atom],
            storage_owner,
        })
    }

    fn expand_atom(
        &mut self,
        atom: &Atom,
        negated: bool,
        var_list_types: &mut HashMap<String, ScalarType>,
    ) -> Result<TemporaryOutputs<Atom>> {
        if let Some(atoms) = self.expand_list_builtin(atom, var_list_types)? {
            return Ok(atoms);
        }
        if is_reserved_pair_helper(atom) {
            return Err(list_error(format_args!("{}", "pair helpers are reserved for meta-term normalization and are not implemented in list normalization"), &mut self.allocation_owner));
        }

        let mut payload = PayloadCustody::default();
        self.acquire_vector::<Term>(&mut payload, atom.terms.len())?;
        self.acquire(&mut payload, atom.predicate.len() as u64)?;
        let mut terms = Vec::with_capacity(atom.terms.len());
        let mut extra_storage = None;
        let mut extra_atoms = Vec::new();
        for (idx, term) in atom.terms.iter().enumerate() {
            let expected = self.list_column_elem(&atom.predicate, idx);
            match term {
                Term::Variable(name) => {
                    if let Some(elem_type) = expected {
                        self.bind_list_type(var_list_types, name, elem_type)?;
                    }
                    self.acquire_term(&mut payload, term)?;
                    terms.push(term.clone());
                }
                Term::Cons { head, tail } => {
                    if negated {
                        return Err(list_error(format_args!("{}", "cons patterns in negated atoms are unsupported before a positive binder"), &mut self.allocation_owner));
                    }
                    let elem_type = expected.ok_or_else(|| {
                        list_error(
                            format_args!("{}", "cons pattern requires a declared list<T> column"),
                            &mut self.allocation_owner,
                        )
                    })?;
                    let mut extra_payload = PayloadCustody::default();
                    let list_var = self.fresh_var("list_cons", &mut extra_payload)?;
                    let head_term = self
                        .normalize_scalar_term(head, Some(elem_type), &mut extra_payload)?
                        .1;
                    let tail_arg = self.normalize_list_arg(
                        tail,
                        Some(elem_type),
                        var_list_types,
                        &mut extra_payload,
                    )?;
                    if let Term::Variable(name) = &tail_arg.term {
                        self.bind_list_type(var_list_types, name, elem_type)?;
                    }
                    self.acquire(&mut payload, list_var.len() as u64)?;
                    self.acquire_vector::<Term>(&mut extra_payload, 3)?;
                    self.acquire_format(
                        &mut extra_payload,
                        format_args!("__xlog_list_cons_{}", scalar_suffix(elem_type)),
                    )?;
                    terms.push(Term::Variable(list_var.clone()));
                    self.push_temporary(
                        &mut extra_atoms,
                        &mut extra_storage,
                        RetainedOutput {
                            value: Atom {
                                predicate: cons_pred(elem_type),
                                terms: vec![Term::Variable(list_var), head_term, tail_arg.term],
                            },
                            custody: extra_payload,
                            storage_owner: None,
                        },
                    )?;
                }
                _ => {
                    terms.push(self.normalize_value_term(term, expected, &mut payload)?);
                }
            }
        }

        let mut atoms = self.single_atom(RetainedOutput {
            value: Atom {
                predicate: atom.predicate.clone(),
                terms,
            },
            custody: payload,
            storage_owner: None,
        })?;
        self.reserve_output_vector(
            &mut atoms.values,
            &mut atoms.storage_owner,
            extra_atoms.len(),
        )?;
        atoms.values.extend(extra_atoms);
        drop(extra_storage);
        Ok(atoms)
    }

    fn expand_list_builtin(
        &mut self,
        atom: &Atom,
        var_list_types: &mut HashMap<String, ScalarType>,
    ) -> Result<Option<TemporaryOutputs<Atom>>> {
        let pred = atom.predicate.as_str();
        let mut payload = PayloadCustody::default();
        match pred {
            "is_list" => {
                require_arity(atom, 1, &mut self.allocation_owner)?;
                let arg =
                    self.normalize_list_arg(&atom.terms[0], None, var_list_types, &mut payload)?;
                self.acquire_vector::<Term>(&mut payload, 2)?;
                self.acquire(&mut payload, len_pred().len() as u64)?;
                Ok(Some(self.single_atom(RetainedOutput {
                    value: Atom {
                        predicate: len_pred().to_string(),
                        terms: vec![arg.term, Term::Anonymous],
                    },
                    custody: payload,
                    storage_owner: None,
                })?))
            }
            "member" | "memberchk" => {
                require_arity(atom, 2, &mut self.allocation_owner)?;
                let list =
                    self.normalize_list_arg(&atom.terms[1], None, var_list_types, &mut payload)?;
                let value = self
                    .normalize_scalar_term(&atom.terms[0], Some(list.elem_type), &mut payload)?
                    .1;
                self.acquire_vector::<Term>(&mut payload, 3)?;
                self.acquire_format(
                    &mut payload,
                    format_args!("__xlog_list_item_{}", scalar_suffix(list.elem_type)),
                )?;
                Ok(Some(self.single_atom(RetainedOutput {
                    value: Atom {
                        predicate: item_pred(list.elem_type),
                        terms: vec![list.term, Term::Anonymous, value],
                    },
                    custody: payload,
                    storage_owner: None,
                })?))
            }
            "length" => {
                require_arity(atom, 2, &mut self.allocation_owner)?;
                let list =
                    self.normalize_list_arg(&atom.terms[0], None, var_list_types, &mut payload)?;
                let len = self
                    .normalize_scalar_term(&atom.terms[1], Some(ScalarType::U32), &mut payload)?
                    .1;
                self.acquire_vector::<Term>(&mut payload, 2)?;
                self.acquire(&mut payload, len_pred().len() as u64)?;
                Ok(Some(self.single_atom(RetainedOutput {
                    value: Atom {
                        predicate: len_pred().to_string(),
                        terms: vec![list.term, len],
                    },
                    custody: payload,
                    storage_owner: None,
                })?))
            }
            "nth" => {
                require_arity(atom, 3, &mut self.allocation_owner)?;
                let idx = self
                    .normalize_scalar_term(&atom.terms[0], Some(ScalarType::U32), &mut payload)?
                    .1;
                let list =
                    self.normalize_list_arg(&atom.terms[1], None, var_list_types, &mut payload)?;
                let value = self
                    .normalize_scalar_term(&atom.terms[2], Some(list.elem_type), &mut payload)?
                    .1;
                self.acquire_vector::<Term>(&mut payload, 3)?;
                self.acquire_format(
                    &mut payload,
                    format_args!("__xlog_list_item_{}", scalar_suffix(list.elem_type)),
                )?;
                Ok(Some(self.single_atom(RetainedOutput {
                    value: Atom {
                        predicate: item_pred(list.elem_type),
                        terms: vec![list.term, idx, value],
                    },
                    custody: payload,
                    storage_owner: None,
                })?))
            }
            "append" => {
                require_arity(atom, 3, &mut self.allocation_owner)?;
                if matches!(atom.terms[2], Term::List(_) | Term::Cons { .. })
                    && !matches!(atom.terms[0], Term::List(_) | Term::Cons { .. })
                    && !matches!(atom.terms[1], Term::List(_) | Term::Cons { .. })
                {
                    return Err(list_error(format_args!("{}", "unbounded append/3 generation is unsupported during list normalization; bind the first two lists to finite literals"), &mut self.allocation_owner));
                }
                let left =
                    self.normalize_list_arg(&atom.terms[0], None, var_list_types, &mut payload)?;
                let right = self.normalize_list_arg(
                    &atom.terms[1],
                    Some(left.elem_type),
                    var_list_types,
                    &mut payload,
                )?;
                if left.elem_type != right.elem_type {
                    return Err(list_error(
                        format_args!(
                            "{}",
                            "append/3 list arguments must have the same element type"
                        ),
                        &mut self.allocation_owner,
                    ));
                }
                let out = self.normalize_list_arg(
                    &atom.terms[2],
                    Some(left.elem_type),
                    var_list_types,
                    &mut payload,
                )?;
                if !left.is_literal || !right.is_literal {
                    return Err(list_error(format_args!("{}", "unbounded append/3 generation is unsupported during list normalization; bind the first two lists to finite literals"), &mut self.allocation_owner));
                }
                let left_id = left.id.expect("literal list has id");
                let right_id = right.id.expect("literal list has id");
                let out_id = match out.id {
                    Some(id) => id,
                    None => self.concat_lists(left_id, right_id)?,
                };
                let key_owner = self.reserve_temporary(3 * std::mem::size_of::<u64>() as u64)?;
                let key = (
                    ListOp::Append,
                    left.elem_type,
                    vec![left_id, right_id, out_id],
                );
                if !self.operation_rows.contains_key(&key) {
                    let operation_rows_owner = self
                        .reserve_hash_growth::<(ListOp, ScalarType, Vec<u64>), Option<Arc<dyn Send + Sync>>>(
                            self.operation_rows.len(),
                            self.operation_rows.capacity(),
                        )?;
                    self.operation_rows.insert(key, key_owner);
                    if operation_rows_owner.is_some() {
                        self._operation_rows_owner = operation_rows_owner;
                    }
                } else {
                    drop(key);
                    drop(key_owner);
                }
                if let Term::Variable(name) = &out.term {
                    self.bind_list_type(var_list_types, name, left.elem_type)?;
                }
                self.acquire_vector::<Term>(&mut payload, 3)?;
                self.acquire_format(
                    &mut payload,
                    format_args!("__xlog_list_append_{}", scalar_suffix(left.elem_type)),
                )?;
                Ok(Some(self.single_atom(RetainedOutput {
                    value: Atom {
                        predicate: append_pred(left.elem_type),
                        terms: vec![left.term, right.term, out.term],
                    },
                    custody: payload,
                    storage_owner: None,
                })?))
            }
            "sort" | "msort" | "list_to_set" => {
                require_arity(atom, 2, &mut self.allocation_owner)?;
                let input =
                    self.normalize_list_arg(&atom.terms[0], None, var_list_types, &mut payload)?;
                if !input.is_literal {
                    return Err(list_error(format_args!("{}", "sort/msort/list_to_set require a finite list literal during list normalization"), &mut self.allocation_owner));
                }
                let op = match pred {
                    "sort" => ListOp::Sort,
                    "msort" => ListOp::MSort,
                    "list_to_set" => ListOp::ListToSet,
                    _ => unreachable!(),
                };
                let out = self.normalize_list_arg(
                    &atom.terms[1],
                    Some(input.elem_type),
                    var_list_types,
                    &mut payload,
                )?;
                let input_id = input.id.expect("literal list has id");
                let out_id = match out.id {
                    Some(id) => id,
                    None => self.derived_ordered_list(input_id, op)?,
                };
                let key_owner = self.reserve_temporary(2 * std::mem::size_of::<u64>() as u64)?;
                let key = (op, input.elem_type, vec![input_id, out_id]);
                if !self.operation_rows.contains_key(&key) {
                    let operation_rows_owner = self
                        .reserve_hash_growth::<(ListOp, ScalarType, Vec<u64>), Option<Arc<dyn Send + Sync>>>(
                            self.operation_rows.len(),
                            self.operation_rows.capacity(),
                        )?;
                    self.operation_rows.insert(key, key_owner);
                    if operation_rows_owner.is_some() {
                        self._operation_rows_owner = operation_rows_owner;
                    }
                } else {
                    drop(key);
                    drop(key_owner);
                }
                if let Term::Variable(name) = &out.term {
                    self.bind_list_type(var_list_types, name, input.elem_type)?;
                }
                self.acquire_vector::<Term>(&mut payload, 2)?;
                let spelling = match op {
                    ListOp::Sort => "__xlog_list_sort_",
                    ListOp::MSort => "__xlog_list_msort_",
                    ListOp::ListToSet => "__xlog_list_to_set_",
                    ListOp::Append => unreachable!(),
                };
                self.acquire_format(
                    &mut payload,
                    format_args!("{}{}", spelling, scalar_suffix(input.elem_type)),
                )?;
                let helper = match op {
                    ListOp::Sort => sort_pred(input.elem_type),
                    ListOp::MSort => msort_pred(input.elem_type),
                    ListOp::ListToSet => list_to_set_pred(input.elem_type),
                    ListOp::Append => unreachable!(),
                };
                Ok(Some(self.single_atom(RetainedOutput {
                    value: Atom {
                        predicate: helper,
                        terms: vec![input.term, out.term],
                    },
                    custody: payload,
                    storage_owner: None,
                })?))
            }
            _ => Ok(None),
        }
    }

    fn normalize_value_term(
        &mut self,
        term: &Term,
        expected_list_elem: Option<ScalarType>,
        payload: &mut PayloadCustody,
    ) -> Result<Term> {
        match term {
            Term::List(_) => Ok(self
                .normalize_list_arg_with_fresh_bindings(term, expected_list_elem, payload)?
                .term),
            Term::Cons { head, tail } => {
                let elem_type = expected_list_elem.ok_or_else(|| {
                    list_error(
                        format_args!("{}", "cons literal requires a declared list<T> context"),
                        &mut self.allocation_owner,
                    )
                })?;
                let storage_owner = self.reserve_temporary(std::mem::size_of::<Term>() as u64)?;
                let payload_owner = if let Term::String(text) | Term::Variable(text) = &**head {
                    self.reserve_temporary(text.len() as u64)?
                } else {
                    None
                };
                let mut items = TemporaryListItems {
                    items: vec![
                        self.normalize_scalar_term_with_receipt(
                            head,
                            Some(elem_type),
                            storage_owner.as_ref(),
                            None,
                        )?
                        .1,
                    ],
                    storage_owner,
                    payload_owners: [payload_owner, None],
                };
                let tail_arg =
                    self.normalize_list_arg_with_fresh_bindings(tail, Some(elem_type), payload)?;
                let tail_id = tail_arg.id.ok_or_else(|| {
                    list_error(
                        format_args!("{}", "cons literal tail must be a finite list literal"),
                        &mut self.allocation_owner,
                    )
                })?;
                let tail_items = self.clone_list_items(tail_id)?;
                self.extend_list_items(&mut items, tail_items)?;
                let id = self.register_list_terms(items.items, Some(elem_type))?;
                Ok(Term::Integer(id as i128))
            }
            _ => {
                self.acquire_term(payload, term)?;
                Ok(term.clone())
            }
        }
    }

    fn normalize_list_arg_with_fresh_bindings(
        &mut self,
        term: &Term,
        expected_elem_type: Option<ScalarType>,
        payload: &mut PayloadCustody,
    ) -> Result<ListArg> {
        let (capacity, _bindings_owner) =
            self.reserve_list_bindings(|visit| term.visit_variables(visit))?;
        let mut bindings = HashMap::with_capacity(capacity);
        self.normalize_list_arg(term, expected_elem_type, &mut bindings, payload)
    }

    fn normalize_list_arg(
        &mut self,
        term: &Term,
        expected_elem_type: Option<ScalarType>,
        var_list_types: &mut HashMap<String, ScalarType>,
        payload: &mut PayloadCustody,
    ) -> Result<ListArg> {
        match term {
            Term::List(items) => {
                let _items_owner = if self.allocation_owner.is_some() {
                    self.reserve_temporary(Self::items_clone_bytes(items)?)?
                } else {
                    None
                };
                let id = self.register_list_terms(items.clone(), expected_elem_type)?;
                let elem_type =
                    Self::list_by_id(&self.lists, id, &mut self.allocation_owner)?.elem_type;
                Ok(ListArg {
                    term: Term::Integer(id as i128),
                    id: Some(id),
                    elem_type,
                    is_literal: true,
                })
            }
            Term::Cons { .. } => {
                let value = self.normalize_value_term(term, expected_elem_type, payload)?;
                let id = match value {
                    Term::Integer(id) if id >= 0 => id as u64,
                    _ => {
                        return Err(list_error(
                            format_args!(
                                "{}",
                                "cons list argument must normalize to a finite list id"
                            ),
                            &mut self.allocation_owner,
                        ));
                    }
                };
                let elem_type =
                    Self::list_by_id(&self.lists, id, &mut self.allocation_owner)?.elem_type;
                Ok(ListArg {
                    term: Term::Integer(id as i128),
                    id: Some(id),
                    elem_type,
                    is_literal: true,
                })
            }
            Term::Integer(id) if *id > 0 => {
                let elem_type = expected_elem_type
                    .or_else(|| find_list_by_id(&self.lists, *id as u64).map(|list| list.elem_type))
                    .ok_or_else(|| {
                        list_error(
                            format_args!("{}", "list id requires a known list<T> type"),
                            &mut self.allocation_owner,
                        )
                    })?;
                Ok(ListArg {
                    term: term.clone(),
                    id: Some(*id as u64),
                    elem_type,
                    is_literal: false,
                })
            }
            Term::Variable(name) => {
                let elem_type = expected_elem_type
                    .or_else(|| var_list_types.get(name).copied())
                    .ok_or_else(|| {
                        list_error(
                            format_args!(
                                "{}",
                                "list built-in requires a finite literal or known list<T> variable"
                            ),
                            &mut self.allocation_owner,
                        )
                    })?;
                self.bind_list_type(var_list_types, name, elem_type)?;
                self.acquire_term(payload, term)?;
                Ok(ListArg {
                    term: term.clone(),
                    id: None,
                    elem_type,
                    is_literal: false,
                })
            }
            Term::Anonymous => {
                let elem_type = expected_elem_type.ok_or_else(|| {
                    list_error(
                        format_args!(
                            "{}",
                            "anonymous list argument requires a known list<T> type"
                        ),
                        &mut self.allocation_owner,
                    )
                })?;
                Ok(ListArg {
                    term: term.clone(),
                    id: None,
                    elem_type,
                    is_literal: false,
                })
            }
            _ => Err(list_error(
                format_args!(
                    "{}",
                    "expected finite list literal or known list<T> variable"
                ),
                &mut self.allocation_owner,
            )),
        }
    }

    fn normalize_scalar_term(
        &mut self,
        term: &Term,
        expected: Option<ScalarType>,
        payload: &mut PayloadCustody,
    ) -> Result<(ScalarType, Term)> {
        self.normalize_scalar_term_with_receipt(term, expected, None, Some(payload))
    }

    fn normalize_scalar_term_with_receipt(
        &mut self,
        term: &Term,
        expected: Option<ScalarType>,
        temporary_receipt: Option<&Arc<dyn Send + Sync>>,
        mut payload: Option<&mut PayloadCustody>,
    ) -> Result<(ScalarType, Term)> {
        match term {
            Term::Variable(_) | Term::Anonymous => {
                let typ = expected.ok_or_else(|| {
                    list_error(format_args!("{}", "list element variable requires a known list<T> element type"), &mut self.allocation_owner)
                })?;
                if temporary_receipt.is_none() && self.allocation_owner.is_some() {
                    self.acquire_term(payload.as_deref_mut().expect("original scalar output custody"), term)?;
                }
                Ok((typ, term.clone()))
            }
            Term::Integer(i) => match expected {
                Some(ScalarType::U32) => {
                    let value = u32::try_from(*i)
                        .map_err(|_| list_error(format_args!("{}", "integer list element is out of range for u32"), &mut self.allocation_owner))?;
                    Ok((ScalarType::U32, Term::Integer(value as i128)))
                }
                Some(ScalarType::U64) => {
                    let value = u64::try_from(*i)
                        .map_err(|_| list_error(format_args!("{}", "integer list element is out of range for u64"), &mut self.allocation_owner))?;
                    Ok((ScalarType::U64, Term::Integer(value as i128)))
                }
                Some(ScalarType::I32) => {
                    let value = i32::try_from(*i)
                        .map_err(|_| list_error(format_args!("{}", "integer list element is out of range for i32"), &mut self.allocation_owner))?;
                    Ok((ScalarType::I32, Term::Integer(value as i128)))
                }
                Some(ScalarType::I64) => {
                    let value = i64::try_from(*i).map_err(|_| list_error(
                        format_args!("integer list element is out of range for i64"), &mut self.allocation_owner,
                    ))?;
                    Ok((ScalarType::I64, Term::Integer(i128::from(value))))
                }
                None => {
                    let typ = term.inferred_scalar_type();
                    self.normalize_scalar_term_with_receipt(term, Some(typ), temporary_receipt, payload)
                },
                Some(ScalarType::F32) => Ok((ScalarType::F32, Term::Float(*i as f64))),
                Some(ScalarType::F64) => Ok((ScalarType::F64, Term::Float(*i as f64))),
                Some(ScalarType::Bool) if *i == 0 || *i == 1 => {
                    Ok((ScalarType::Bool, Term::Integer(*i)))
                }
                Some(ScalarType::Bool) => Err(list_error(format_args!("{}", "bool list elements must be 0 or 1"), &mut self.allocation_owner)),
                Some(ScalarType::Symbol) => {
                    Err(list_error(format_args!("{}", "integer list element is not valid for symbol"), &mut self.allocation_owner))
                }
            },
            Term::Float(f) => match expected {
                Some(ScalarType::F32) => Ok((ScalarType::F32, Term::Float(*f))),
                Some(ScalarType::F64) | None => Ok((ScalarType::F64, Term::Float(*f))),
                Some(_) => Err(list_error(format_args!("{}", "float list element is not valid for expected type"), &mut self.allocation_owner)),
            },
            Term::String(s) => {
                if expected.is_none() || expected == Some(ScalarType::Symbol) {
                    if temporary_receipt.is_none() && self.allocation_owner.is_some() {
                        self.acquire_term(payload.as_deref_mut().expect("original scalar output custody"), term)?;
                    }
                    Ok((ScalarType::Symbol, Term::String(s.clone())))
                } else {
                    Err(list_error(format_args!("{}", "string list element is only valid for symbol lists"), &mut self.allocation_owner))
                }
            }
            Term::Symbol(id) => {
                if expected.is_none() || expected == Some(ScalarType::Symbol) {
                    Ok((ScalarType::Symbol, Term::Symbol(*id)))
                } else {
                    Err(list_error(format_args!("{}", "symbol list element is not valid for expected type"), &mut self.allocation_owner))
                }
            }
            Term::List(_) | Term::Cons { .. } => {
                let mut nested_payload = PayloadCustody::default();
                let id = self
                    .normalize_list_arg_with_fresh_bindings(term, None, &mut nested_payload)?
                    .id
                    .expect("literal list has id");
                let typ = expected.unwrap_or(LIST_ID_TYPE);
                if typ != LIST_ID_TYPE {
                    return Err(list_error(format_args!("{}", "nested list element requires list<list<T>> or list<u64> context"), &mut self.allocation_owner));
                }
                Ok((LIST_ID_TYPE, Term::Integer(id as i128)))
            }
            Term::Compound { .. } | Term::PredRef(_) | Term::Aggregate(_) => Err(list_error(format_args!("{}", "compound, predref, and aggregate list elements require meta-term normalization support"), &mut self.allocation_owner)),
        }
    }

    fn register_list_terms(
        &mut self,
        items: Vec<Term>,
        expected_elem_type: Option<ScalarType>,
    ) -> Result<u64> {
        let normalized_owner = if self.allocation_owner.is_some() {
            let overflow = || XlogError::allocation_extent_overflow("normalized list values");
            let mut bytes = (items.len() as u64)
                .checked_mul(std::mem::size_of::<Term>() as u64)
                .ok_or_else(overflow)?;
            for item in &items {
                // Scalar normalization copies only these original strings.
                // Nested finite lists produce an integer ID; their actual
                // registry allocations keep their independent original owners.
                if let Term::String(text) | Term::Variable(text) = item {
                    bytes = bytes.checked_add(text.len() as u64).ok_or_else(overflow)?;
                }
            }
            self.reserve_temporary(bytes)?
        } else {
            None
        };
        let _keys_vector_owner = if self.allocation_owner.is_some() {
            let bytes = items
                .len()
                .checked_mul(std::mem::size_of::<RetainedLiteralKey>())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or_else(|| XlogError::allocation_extent_overflow("list key values"))?;
            self.reserve_temporary(bytes)?
        } else {
            None
        };
        let mut elem_type = expected_elem_type;
        let mut normalized = Vec::with_capacity(items.len());
        let mut keys = Vec::with_capacity(items.len());

        for item in &items {
            let (item_type, term) = match self.normalize_scalar_term_with_receipt(
                item,
                elem_type,
                normalized_owner.as_ref(),
                None,
            ) {
                Ok(value) => value,
                Err(err) if matches!(err.unowned(), XlogError::AllocationRefused { .. }) => {
                    return Err(err)
                }
                Err(err) if elem_type.is_some() => {
                    return Err(list_error(
                        format_args!("heterogeneous list literal: {}", err),
                        &mut self.allocation_owner,
                    ));
                }
                Err(err) => return Err(err),
            };
            if let Some(existing) = elem_type {
                if existing != item_type {
                    return Err(list_error(
                        format_args!(
                            "{}",
                            "heterogeneous list literals require a declared finite term type"
                        ),
                        &mut self.allocation_owner,
                    ));
                }
            } else {
                elem_type = Some(item_type);
            }
            keys.push(literal_key_with_owner(
                &term,
                item_type,
                &mut self.allocation_owner,
            )?);
            normalized.push(term);
        }

        let elem_type = elem_type.unwrap_or(LIST_ID_TYPE);
        let key_owner = if self.allocation_owner.is_some() {
            let mut bytes = keys
                .len()
                .checked_mul(std::mem::size_of::<LiteralKey>())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or_else(|| XlogError::allocation_extent_overflow("list lookup key"))?;
            for key in &keys {
                if let LiteralKey::Symbol(text) = &key.value {
                    bytes = bytes
                        .checked_add(text.len() as u64)
                        .ok_or_else(|| XlogError::allocation_extent_overflow("list lookup key"))?;
                }
            }
            self.reserve_temporary(bytes)?
        } else {
            None
        };
        let key = (
            elem_type,
            keys.iter().map(|key| key.value.clone()).collect::<Vec<_>>(),
        );
        if let Some(id) = self.lists_by_key.get(&key) {
            return Ok(id.id);
        }

        let lists_by_key_owner = self
            .reserve_hash_growth::<(ScalarType, Vec<LiteralKey>), RetainedListId>(
                self.lists_by_key.len(),
                self.lists_by_key.capacity(),
            )?;
        let lists_owner =
            if self.allocation_owner.is_some() && self.lists.len() == self.lists.capacity() {
                let count = self
                    .lists
                    .len()
                    .checked_add(1)
                    .ok_or_else(|| XlogError::allocation_extent_overflow("list registry"))?;
                self.reserve_temporary(xlog_core::push_vector_storage_bytes::<RetainedListDef>(
                    count,
                )?)?
            } else {
                None
            };
        let keys_owner = self.reserve_temporary(
            u64::try_from(
                keys.len()
                    .checked_mul(std::mem::size_of::<LiteralKey>())
                    .ok_or_else(|| XlogError::allocation_extent_overflow("retained list keys"))?,
            )
            .map_err(|_| XlogError::allocation_extent_overflow("retained list keys"))?,
        )?;
        let receipt_count = keys
            .iter()
            .filter(|key| key.allocation_owner.is_some())
            .count();
        let key_allocation_storage = self.reserve_temporary(
            u64::try_from(
                receipt_count
                    .checked_mul(std::mem::size_of::<Arc<dyn Send + Sync>>())
                    .ok_or_else(|| XlogError::allocation_extent_overflow("list key receipts"))?,
            )
            .map_err(|_| XlogError::allocation_extent_overflow("list key receipts"))?,
        )?;
        let mut key_allocations = Vec::with_capacity(receipt_count);
        let mut retained_keys = Vec::with_capacity(keys.len());
        for key in keys {
            if let Some(owner) = key.allocation_owner {
                key_allocations.push(owner);
            }
            retained_keys.push(key.value);
        }
        drop(_keys_vector_owner);
        let id = self.next_list_id;
        self.next_list_id += 1;
        self.lists_by_key.insert(
            key,
            RetainedListId {
                id,
                _key_owner: key_owner,
            },
        );
        if lists_by_key_owner.is_some() {
            self._lists_by_key_owner = lists_by_key_owner;
        }
        self.lists.push(RetainedListDef {
            value: ListDef {
                id,
                elem_type,
                items: normalized,
                keys: retained_keys,
            },
            _items_owner: normalized_owner,
            _keys_owner: keys_owner,
            _key_allocations: key_allocations,
            _key_allocation_storage: key_allocation_storage,
        });
        if lists_owner.is_some() {
            self._lists_owner = lists_owner;
        }
        Ok(id)
    }

    fn concat_lists(&mut self, left_id: u64, right_id: u64) -> Result<u64> {
        let mut items = self.clone_list_items(left_id)?;
        let left_type =
            Self::list_by_id(&self.lists, left_id, &mut self.allocation_owner)?.elem_type;
        let right_items = self.clone_list_items(right_id)?;
        let right_type =
            Self::list_by_id(&self.lists, right_id, &mut self.allocation_owner)?.elem_type;
        if left_type != right_type {
            return Err(list_error(
                format_args!(
                    "{}",
                    "append/3 list arguments must have the same element type"
                ),
                &mut self.allocation_owner,
            ));
        }
        self.extend_list_items(&mut items, right_items)?;
        self.register_list_terms(items.items, Some(right_type))
    }

    fn derived_ordered_list(&mut self, input_id: u64, op: ListOp) -> Result<u64> {
        let _input_owner = self.reserve_list_clone(input_id)?;
        let sorting_bytes = if self.allocation_owner.is_some() {
            let input = Self::list_by_id(&self.lists, input_id, &mut self.allocation_owner)?;
            let mut formatted = 0u64;
            for key in &input.keys {
                formatted = formatted.max(crate::compile::formatted_text_allocation_bytes(
                    format_args!("{key:?}"),
                )?);
            }
            let count = input.items.len();
            let bytes = crate::compile::push_vector_allocation_bytes::<(LiteralKey, Term)>(count)?
                .checked_mul(2)
                .and_then(|n| n.checked_add(formatted.checked_mul(2)?))
                .ok_or_else(|| XlogError::allocation_extent_overflow("list sorting"))?;
            bytes
                .checked_add(crate::compile::push_vector_allocation_bytes::<Term>(count)?)
                .ok_or_else(|| XlogError::allocation_extent_overflow("list sorting"))?
        } else {
            0
        };
        let _sorting_owner = self.reserve_temporary(sorting_bytes)?;
        let input = Self::list_by_id(&self.lists, input_id, &mut self.allocation_owner)?.clone();
        let mut pairs: Vec<(LiteralKey, Term)> = input.keys.into_iter().zip(input.items).collect();
        pairs.sort_by(|(a, _), (b, _)| format!("{a:?}").cmp(&format!("{b:?}")));
        if matches!(op, ListOp::Sort | ListOp::ListToSet) {
            pairs.dedup_by(|(a, _), (b, _)| a == b);
        }
        let items = pairs.into_iter().map(|(_, term)| term).collect();
        self.register_list_terms(items, Some(input.elem_type))
    }

    fn append_helper_declarations_and_facts(
        &mut self,
        program: &mut RetainedProgram,
    ) -> Result<()> {
        self.ensure_tail_lists()?;
        let mut facts = self.operation_rows.len();
        for list in &self.lists {
            facts = facts
                .checked_add(1)
                .and_then(|n| n.checked_add(list.value.items.len()))
                .and_then(|n| n.checked_add(usize::from(!list.value.items.is_empty())))
                .ok_or_else(|| XlogError::allocation_extent_overflow("list helper"))?;
        }
        self.prepare_program_vector(&mut program.value.rules, &mut program.custody.rules, facts)?;

        let _lists_owner = self.reserve_lists_clone()?;
        let mut helper_rules = TemporaryOutputs {
            values: Vec::new(),
            storage_owner: None,
        };
        for list in self
            .lists
            .iter()
            .map(|list| list.value.clone())
            .collect::<Vec<_>>()
        {
            self.insert_helper_pred(len_pred())?;
            let rule = self.fact_output(
                len_pred(),
                RetainedOutput {
                    value: [
                        Term::Integer(list.id as i128),
                        Term::Integer(list.items.len() as i128),
                    ],
                    custody: PayloadCustody::default(),
                    storage_owner: None,
                },
            )?;
            self.push_temporary(
                &mut helper_rules.values,
                &mut helper_rules.storage_owner,
                rule,
            )?;

            let _item_name_owner =
                self.reserve_temporary(crate::compile::formatted_text_allocation_bytes(
                    format_args!("__xlog_list_item_{}", scalar_suffix(list.elem_type)),
                )?)?;
            let item_name = item_pred(list.elem_type);
            self.insert_helper_pred(&item_name)?;
            for (idx, item) in list.items.iter().enumerate() {
                let mut payload = PayloadCustody::default();
                self.acquire_term(&mut payload, item)?;
                let rule = self.fact_output(
                    &item_name,
                    RetainedOutput {
                        value: [
                            Term::Integer(list.id as i128),
                            Term::Integer(idx as i128),
                            item.clone(),
                        ],
                        custody: payload,
                        storage_owner: None,
                    },
                )?;
                self.push_temporary(
                    &mut helper_rules.values,
                    &mut helper_rules.storage_owner,
                    rule,
                )?;
            }
            drop(item_name);
            drop(_item_name_owner);

            if let Some((head, tail_id)) = self.cons_fact_parts(list.id)? {
                let _cons_name_owner =
                    self.reserve_temporary(crate::compile::formatted_text_allocation_bytes(
                        format_args!("__xlog_list_cons_{}", scalar_suffix(list.elem_type)),
                    )?)?;
                let cons_name = cons_pred(list.elem_type);
                self.insert_helper_pred(&cons_name)?;
                let rule = self.fact_output(
                    &cons_name,
                    RetainedOutput {
                        value: [
                            Term::Integer(list.id as i128),
                            head.value,
                            Term::Integer(tail_id as i128),
                        ],
                        custody: head.custody,
                        storage_owner: None,
                    },
                )?;
                self.push_temporary(
                    &mut helper_rules.values,
                    &mut helper_rules.storage_owner,
                    rule,
                )?;
            }
        }
        drop(_lists_owner);

        // Deterministic emission order keeps relation numbering stable.
        let operation_rows_owner = if self.allocation_owner.is_some() {
            let overflow = || XlogError::allocation_extent_overflow("list operation snapshot");
            let mut bytes = self
                .operation_rows
                .len()
                .checked_mul(std::mem::size_of::<(ListOp, ScalarType, Vec<u64>)>())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or_else(overflow)?;
            for (_, _, ids) in self.operation_rows.keys() {
                bytes = ids
                    .len()
                    .checked_mul(std::mem::size_of::<u64>())
                    .and_then(|ids| u64::try_from(ids).ok())
                    .and_then(|ids| bytes.checked_add(ids))
                    .ok_or_else(overflow)?;
            }
            self.reserve_temporary(bytes)?
        } else {
            None
        };
        let mut operation_rows = Vec::with_capacity(self.operation_rows.len());
        operation_rows.extend(self.operation_rows.keys().cloned());
        operation_rows.sort_unstable_by(|a, b| {
            (a.0 as u8, a.1 as u8, &a.2).cmp(&(b.0 as u8, b.1 as u8, &b.2))
        });
        for (op, elem_type, ids) in operation_rows {
            let _pred_owner =
                self.reserve_temporary(crate::compile::formatted_text_allocation_bytes(
                    format_args!("__xlog_list_to_set_{}", scalar_suffix(elem_type)),
                )?)?;
            let pred = match op {
                ListOp::Append => append_pred(elem_type),
                ListOp::Sort => sort_pred(elem_type),
                ListOp::MSort => msort_pred(elem_type),
                ListOp::ListToSet => list_to_set_pred(elem_type),
            };
            self.insert_helper_pred(&pred)?;
            let mut payload = PayloadCustody::default();
            self.acquire(&mut payload, pred.len() as u64)?;
            self.acquire_vector::<Term>(&mut payload, ids.len())?;
            let mut terms = Vec::with_capacity(ids.len());
            terms.extend(ids.into_iter().map(|id| Term::Integer(id as i128)));
            let rule = RetainedOutput {
                value: fact(&pred, terms),
                custody: payload,
                storage_owner: None,
            };
            self.push_temporary(
                &mut helper_rules.values,
                &mut helper_rules.storage_owner,
                rule,
            )?;
        }
        drop(operation_rows_owner);

        self.append_helper_pred_decls(program)?;
        for rule in helper_rules.values {
            self.push_rule(&mut program.value.rules, &mut program.custody.rules, rule)?;
        }
        drop(helper_rules.storage_owner);
        Ok(())
    }

    fn fact_output<const N: usize>(
        &mut self,
        pred: &str,
        mut terms: RetainedOutput<[Term; N]>,
    ) -> Result<RetainedOutput<Rule>> {
        self.acquire(&mut terms.custody, pred.len() as u64)?;
        self.acquire_vector::<Term>(&mut terms.custody, N)?;
        let mut values = Vec::with_capacity(N);
        values.extend(terms.value);
        Ok(RetainedOutput {
            value: fact(pred, values),
            custody: terms.custody,
            storage_owner: None,
        })
    }

    fn ensure_tail_lists(&mut self) -> Result<()> {
        loop {
            let before = self.lists.len();
            let _lists_owner = self.reserve_lists_clone()?;
            for list in self
                .lists
                .iter()
                .map(|list| list.value.clone())
                .collect::<Vec<_>>()
            {
                if list.items.is_empty() {
                    continue;
                }
                let _tail_owner = if self.allocation_owner.is_some() {
                    self.reserve_temporary(Self::items_clone_bytes(&list.items[1..])?)?
                } else {
                    None
                };
                self.register_list_terms(list.items[1..].to_vec(), Some(list.elem_type))?;
            }
            if self.lists.len() == before {
                break;
            }
        }
        Ok(())
    }

    fn cons_fact_parts(&mut self, list_id: u64) -> Result<Option<(RetainedOutput<Term>, u64)>> {
        let mut tail_owner = None;
        if self.allocation_owner.is_some() {
            let list = Self::list_by_id(&self.lists, list_id, &mut self.allocation_owner)?;
            if !list.items.is_empty() {
                let tail_bytes = Self::keys_clone_bytes(&list.keys[1..])?;
                tail_owner = self.reserve_temporary(tail_bytes)?;
            }
        }
        let list = Self::list_by_id(&self.lists, list_id, &mut self.allocation_owner)?;
        let Some(head) = list.items.first() else {
            return Ok(None);
        };
        let custody = match self.allocation_owner.as_deref_mut() {
            Some(scope) => PayloadCustody::for_term_clone(head, scope)?,
            None => PayloadCustody::default(),
        };
        let tail_key = (list.elem_type, list.keys[1..].to_vec());
        let tail_id = self
            .lists_by_key
            .get(&tail_key)
            .map(|entry| entry.id)
            .ok_or_else(|| {
                list_error(
                    format_args!("{}", "missing normalized list tail"),
                    &mut self.allocation_owner,
                )
            })?;
        drop(tail_key);
        drop(tail_owner);
        Ok(Some((
            RetainedOutput {
                value: head.clone(),
                custody,
                storage_owner: None,
            },
            tail_id,
        )))
    }

    fn append_helper_pred_decls(&mut self, program: &mut RetainedProgram) -> Result<()> {
        let helper_table = RetainedHelperPredicates {
            values: std::mem::take(&mut self.helper_preds),
            _storage_owner: self._helper_preds_owner.take(),
        };
        let count = program
            .value
            .predicates
            .len()
            .checked_add(helper_table.values.len())
            .ok_or_else(|| XlogError::allocation_extent_overflow("list declaration names"))?;
        let _existing_owner = if self.allocation_owner.is_some() {
            self.reserve_temporary(crate::compile::hash_allocation_bytes::<
                String,
                Option<Arc<dyn Send + Sync>>,
            >(count)?)?
        } else {
            None
        };
        let mut existing = HashMap::with_capacity(count);
        for predicate in &program.value.predicates {
            if existing.contains_key(&predicate.name) {
                continue;
            }
            let owner = self.reserve_temporary(predicate.name.len() as u64)?;
            existing.insert(predicate.name.clone(), owner);
        }
        let additions = helper_table
            .values
            .keys()
            .filter(|name| !existing.contains_key(*name))
            .count();
        self.prepare_program_vector(
            &mut program.value.predicates,
            &mut program.custody.predicates,
            additions,
        )?;

        // Borrowed names are sorted without cloning the private registry.
        let _helper_preds_owner = if self.allocation_owner.is_some() {
            let bytes = helper_table
                .values
                .len()
                .checked_mul(std::mem::size_of::<&String>())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or_else(|| XlogError::allocation_extent_overflow("list declaration roster"))?;
            self.reserve_temporary(bytes)?
        } else {
            None
        };
        let mut helper_preds = Vec::with_capacity(helper_table.values.len());
        helper_preds.extend(helper_table.values.keys());
        helper_preds.sort_unstable();
        for pred in helper_preds {
            if existing.contains_key(pred) {
                continue;
            }
            let name_owner = self.reserve_temporary(pred.len() as u64)?;
            let existing_name = pred.clone();
            let declaration = self.helper_declaration(pred)?;
            program.value.predicates.push(declaration.value);
            program
                .custody
                .predicates
                .elements
                .push(declaration.custody);
            existing.insert(existing_name, name_owner);
        }
        drop(_helper_preds_owner);
        drop(existing);
        drop(_existing_owner);
        drop(helper_table);
        Ok(())
    }

    fn helper_declaration(&mut self, pred: &str) -> Result<RetainedHelperDeclaration> {
        let schema = helper_column_schema(pred);
        let count = schema.iter().flatten().count();
        let name = match self.allocation_owner.as_deref_mut() {
            Some(scope) => PayloadCustody::for_text_clone(pred, scope)?,
            None => PayloadCustody::default(),
        };
        let mut column_custody = match self.allocation_owner.as_deref_mut() {
            Some(scope) => RetainedVectorCustody::for_capacity::<PredColumn>(count, scope)?,
            None => RetainedVectorCustody::default(),
        };
        let mut columns = Vec::with_capacity(count);
        for (column_name, typ) in schema.into_iter().flatten() {
            let name = match self.allocation_owner.as_deref_mut() {
                Some(scope) => PayloadCustody::for_text_clone(column_name, scope)?,
                None => PayloadCustody::default(),
            };
            columns.push(PredColumn {
                name: Some(column_name.to_string()),
                typ: TypeRef::Scalar(typ),
            });
            column_custody.elements.push(ColumnCustody {
                name,
                typ: PayloadCustody::default(),
            });
        }
        let mut type_custody = match self.allocation_owner.as_deref_mut() {
            Some(scope) => RetainedVectorCustody::for_capacity::<TypeRef>(count, scope)?,
            None => RetainedVectorCustody::default(),
        };
        let mut types = Vec::with_capacity(count);
        for column in &columns {
            types.push(column.typ.clone());
            type_custody.elements.push(PayloadCustody::default());
        }
        Ok(RetainedHelperDeclaration {
            value: PredDecl {
                name: pred.to_string(),
                types,
                columns,
                is_private: true,
            },
            custody: PredicateCustody {
                name,
                types: type_custody,
                columns: column_custody,
            },
        })
    }

    fn list_by_id<'list>(
        lists: &'list [RetainedListDef],
        id: u64,
        owner: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
    ) -> Result<&'list ListDef> {
        find_list_by_id(lists, id).ok_or_else(|| {
            list_error(
                format_args!("{}", "unknown normalized list id"),
                &mut *owner,
            )
        })
    }

    fn fresh_var(&mut self, prefix: &str, payload: &mut PayloadCustody) -> Result<String> {
        let uppercase_owner = self.reserve_temporary(prefix.len() as u64)?;
        let uppercase = prefix.to_ascii_uppercase();
        let fresh_counter = self.fresh_counter;
        self.acquire_format(payload, format_args!("__XLOG_{prefix}_{}", fresh_counter))?;
        let var = format!("__XLOG_{}_{}", uppercase, fresh_counter);
        drop(uppercase);
        drop(uppercase_owner);
        self.fresh_counter += 1;
        Ok(var)
    }

    fn storage_type_for_type_ref(
        typ: &TypeRef,
        domains: &HashMap<String, ScalarType>,
        owner: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
    ) -> Result<ScalarType> {
        match typ {
            TypeRef::Scalar(ty) => Ok(*ty),
            TypeRef::Domain(name) => domains.get(name).copied().ok_or_else(|| {
                list_error(
                    format_args!("unknown domain alias '{}' in list<T>", name),
                    &mut *owner,
                )
            }),
            TypeRef::List(_) | TypeRef::Term | TypeRef::Compound | TypeRef::PredRef => {
                Ok(LIST_ID_TYPE)
            }
        }
    }
}

fn fact(pred: &str, terms: Vec<Term>) -> Rule {
    Rule {
        head: Atom {
            predicate: pred.to_string(),
            terms,
        },
        body: vec![],
    }
}

fn require_arity(
    atom: &Atom,
    expected: usize,
    owner: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> Result<()> {
    if atom.terms.len() == expected {
        Ok(())
    } else {
        Err(list_error(
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

fn literal_key_with_owner(
    term: &Term,
    typ: ScalarType,
    allocate: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> Result<RetainedLiteralKey> {
    let mut allocation_owner = None;
    let value = match (typ, term) {
        (ScalarType::U32, Term::Integer(v)) => Ok(LiteralKey::U32(*v as u32)),
        (ScalarType::U64, Term::Integer(v)) => Ok(LiteralKey::U64(*v as u64)),
        (ScalarType::I32, Term::Integer(v)) => Ok(LiteralKey::I32(*v as i32)),
        (ScalarType::I64, Term::Integer(v)) => Ok(LiteralKey::I64(*v as i64)),
        (ScalarType::F32, Term::Float(v)) => Ok(LiteralKey::F32((*v as f32).to_bits())),
        (ScalarType::F64, Term::Float(v)) => Ok(LiteralKey::F64(v.to_bits())),
        (ScalarType::Bool, Term::Integer(v)) => Ok(LiteralKey::Bool(*v != 0)),
        (ScalarType::Symbol, Term::String(s)) => {
            if let Some(allocate) = allocate.as_deref_mut() {
                allocation_owner = Some(allocate.reserve_allocation(s.len() as u64)?);
            }
            Ok(LiteralKey::Symbol(s.clone()))
        }
        (ScalarType::Symbol, Term::Symbol(id)) => {
            let text = if let Some(allocate) = allocate.as_deref_mut() {
                let mut reserve = |bytes: usize| {
                    allocate.reserve_allocation(
                        u64::try_from(bytes)
                            .map_err(|_| XlogError::allocation_extent_overflow("list symbol"))?,
                    )
                };
                let (text, owner) =
                    xlog_core::symbol::resolve_with_allocation_owner(*id, &mut reserve)?;
                allocation_owner = Some(owner);
                text
            } else {
                xlog_core::symbol::resolve(*id)
            };
            Ok(LiteralKey::Symbol(text))
        }
        _ => Err(list_error(
            format_args!(
                "{}",
                "list literal element did not normalize to expected scalar type"
            ),
            allocate,
        )),
    }?;
    Ok(RetainedLiteralKey {
        value,
        allocation_owner,
    })
}

fn helper_column_schema(pred: &str) -> [Option<(&'static str, ScalarType)>; 3] {
    if pred == len_pred() {
        return [
            Some(("list_id", LIST_ID_TYPE)),
            Some(("len", ScalarType::U32)),
            None,
        ];
    }
    if let Some(elem_type) = helper_elem_type(pred, "__xlog_list_item_") {
        return [
            Some(("list_id", LIST_ID_TYPE)),
            Some(("idx", ScalarType::U32)),
            Some(("value", elem_type)),
        ];
    }
    if let Some(elem_type) = helper_elem_type(pred, "__xlog_list_cons_") {
        return [
            Some(("list_id", LIST_ID_TYPE)),
            Some(("head", elem_type)),
            Some(("tail_id", LIST_ID_TYPE)),
        ];
    }
    if helper_elem_type(pred, "__xlog_list_append_").is_some() {
        return [
            Some(("left_id", LIST_ID_TYPE)),
            Some(("right_id", LIST_ID_TYPE)),
            Some(("out_id", LIST_ID_TYPE)),
        ];
    }
    [
        Some(("input_id", LIST_ID_TYPE)),
        Some(("out_id", LIST_ID_TYPE)),
        None,
    ]
}

fn helper_elem_type(pred: &str, prefix: &str) -> Option<ScalarType> {
    pred.strip_prefix(prefix).and_then(scalar_type_from_suffix)
}

fn scalar_type_from_suffix(s: &str) -> Option<ScalarType> {
    match s {
        "u32" => Some(ScalarType::U32),
        "u64" => Some(ScalarType::U64),
        "i32" => Some(ScalarType::I32),
        "i64" => Some(ScalarType::I64),
        "f32" => Some(ScalarType::F32),
        "f64" => Some(ScalarType::F64),
        "bool" => Some(ScalarType::Bool),
        "symbol" => Some(ScalarType::Symbol),
        _ => None,
    }
}

fn scalar_suffix(typ: ScalarType) -> &'static str {
    match typ {
        ScalarType::U32 => "u32",
        ScalarType::U64 => "u64",
        ScalarType::I32 => "i32",
        ScalarType::I64 => "i64",
        ScalarType::F32 => "f32",
        ScalarType::F64 => "f64",
        ScalarType::Bool => "bool",
        ScalarType::Symbol => "symbol",
    }
}

fn len_pred() -> &'static str {
    "__xlog_list_len"
}

fn item_pred(typ: ScalarType) -> String {
    format!("__xlog_list_item_{}", scalar_suffix(typ))
}

fn cons_pred(typ: ScalarType) -> String {
    format!("__xlog_list_cons_{}", scalar_suffix(typ))
}

fn append_pred(typ: ScalarType) -> String {
    format!("__xlog_list_append_{}", scalar_suffix(typ))
}

fn sort_pred(typ: ScalarType) -> String {
    format!("__xlog_list_sort_{}", scalar_suffix(typ))
}

fn msort_pred(typ: ScalarType) -> String {
    format!("__xlog_list_msort_{}", scalar_suffix(typ))
}

fn list_to_set_pred(typ: ScalarType) -> String {
    format!("__xlog_list_to_set_{}", scalar_suffix(typ))
}

fn is_reserved_pair_helper(atom: &Atom) -> bool {
    matches!(
        (atom.predicate.as_str(), atom.terms.len()),
        ("pair", 3) | ("pairs", 2) | ("zip", 3) | ("zip_with_index", 2) | ("enumerate", 2)
    )
}

fn find_list_by_id(lists: &[RetainedListDef], id: u64) -> Option<&ListDef> {
    lists
        .iter()
        .find(|list| list.value.id == id)
        .map(|list| &list.value)
}

fn list_error(
    message: std::fmt::Arguments<'_>,
    owner: &mut Option<&mut crate::compile::CompilationAllocationScope<'_>>,
) -> XlogError {
    let bytes = xlog_core::error::formatted_text_allocation_bytes(format_args!(
        "list normalization error: {message}"
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
            .finish(|| XlogError::Compilation(format!("list normalization error: {message}"))),
        Err(error) => error,
    }
}
