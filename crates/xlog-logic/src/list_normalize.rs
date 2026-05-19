//! Finite list normalization for the v0.8.5 language surface.

use std::collections::{HashMap, HashSet};

use xlog_core::{Result, ScalarType, XlogError};

use crate::ast::{
    Atom, BodyLiteral, Comparison, DomainDecl, PredColumn, PredDecl, Program, Query, Rule, Term,
    TypeRef,
};

const LIST_ID_TYPE: ScalarType = ScalarType::U64;

/// Normalize finite v0.8.5 list syntax into scalar list identifiers plus helper relations.
///
/// The normalized program uses ordinary XLOG relations:
///
/// - `__xlog_list_len(list_id, len)`
/// - `__xlog_list_item_<type>(list_id, index, value)`
/// - `__xlog_list_cons_<type>(list_id, head, tail_id)`
/// - selected built-in helper relations such as `append`, `sort`, `msort`, and `list_to_set`
///
/// This keeps accepted list programs on the existing relational lowering/runtime path.
pub fn normalize_v085_lists(program: &Program) -> Result<Program> {
    let mut normalizer = ListNormalizer::new(program)?;
    normalizer.normalize_program(program)
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

#[derive(Debug, Clone)]
struct ListDef {
    id: u64,
    elem_type: ScalarType,
    items: Vec<Term>,
    keys: Vec<LiteralKey>,
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

struct ListNormalizer {
    list_columns: HashMap<(String, usize), ListColumn>,
    lists_by_key: HashMap<(ScalarType, Vec<LiteralKey>), u64>,
    lists: Vec<ListDef>,
    next_list_id: u64,
    helper_preds: HashSet<String>,
    operation_rows: HashSet<(ListOp, ScalarType, Vec<u64>)>,
    fresh_counter: usize,
}

impl ListNormalizer {
    fn new(program: &Program) -> Result<Self> {
        let domains: HashMap<String, ScalarType> = program
            .domains
            .iter()
            .map(|DomainDecl { name, typ }| (name.clone(), *typ))
            .collect();

        let mut list_columns = HashMap::new();
        for pred in &program.predicates {
            let columns = if pred.columns.is_empty() {
                pred.types
                    .iter()
                    .cloned()
                    .map(|typ| PredColumn { name: None, typ })
                    .collect()
            } else {
                pred.columns.clone()
            };
            for (idx, col) in columns.iter().enumerate() {
                if let TypeRef::List(inner) = &col.typ {
                    let elem_type = Self::storage_type_for_type_ref(inner, &domains)?;
                    list_columns.insert((pred.name.clone(), idx), ListColumn { elem_type });
                }
            }
        }

        Ok(Self {
            list_columns,
            lists_by_key: HashMap::new(),
            lists: Vec::new(),
            next_list_id: 1,
            helper_preds: HashSet::new(),
            operation_rows: HashSet::new(),
            fresh_counter: 0,
        })
    }

    fn normalize_program(&mut self, program: &Program) -> Result<Program> {
        let mut out = program.clone();
        out.rules = program
            .rules
            .iter()
            .map(|rule| self.normalize_rule(rule))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        out.constraints = program
            .constraints
            .iter()
            .map(|constraint| {
                let body = self.normalize_body(&constraint.body)?;
                Ok(crate::ast::Constraint { body })
            })
            .collect::<Result<Vec<_>>>()?;
        out.queries = program
            .queries
            .iter()
            .map(|query| self.normalize_query(query))
            .collect::<Result<Vec<_>>>()?;

        self.append_helper_declarations_and_facts(&mut out)?;
        Ok(out)
    }

    fn normalize_query(&mut self, query: &Query) -> Result<Query> {
        let mut var_list_types = HashMap::new();
        let atoms = self.expand_atom(&query.atom, false, &mut var_list_types)?;
        let atom = atoms
            .into_iter()
            .next()
            .unwrap_or_else(|| query.atom.clone());
        Ok(Query { atom })
    }

    fn normalize_rule(&mut self, rule: &Rule) -> Result<Vec<Rule>> {
        let mut var_list_types = HashMap::new();
        let head = self.normalize_head_atom(&rule.head)?;
        let body = self.normalize_body_with_env(&rule.body, &mut var_list_types)?;
        Ok(vec![Rule { head, body }])
    }

    fn normalize_body(&mut self, body: &[BodyLiteral]) -> Result<Vec<BodyLiteral>> {
        let mut var_list_types = HashMap::new();
        self.normalize_body_with_env(body, &mut var_list_types)
    }

    fn normalize_body_with_env(
        &mut self,
        body: &[BodyLiteral],
        var_list_types: &mut HashMap<String, ScalarType>,
    ) -> Result<Vec<BodyLiteral>> {
        let mut out = Vec::new();
        for lit in body {
            match lit {
                BodyLiteral::Positive(atom) => {
                    for atom in self.expand_atom(atom, false, var_list_types)? {
                        out.push(BodyLiteral::Positive(atom));
                    }
                }
                BodyLiteral::Negated(atom) => {
                    let atoms = self.expand_atom(atom, true, var_list_types)?;
                    if atoms.len() != 1 {
                        return Err(list_error(
                            "negated list pattern expansion would require multiple atoms",
                        ));
                    }
                    out.push(BodyLiteral::Negated(atoms.into_iter().next().unwrap()));
                }
                BodyLiteral::Comparison(cmp) => {
                    out.push(BodyLiteral::Comparison(Comparison {
                        op: cmp.op,
                        left: self.normalize_value_term(&cmp.left, None)?,
                        right: self.normalize_value_term(&cmp.right, None)?,
                    }));
                }
                BodyLiteral::IsExpr(is_expr) => out.push(BodyLiteral::IsExpr(is_expr.clone())),
                BodyLiteral::Univ(_) => {
                    return Err(list_error(
                        "univ literals must be normalized by G085_META before list normalization",
                    ));
                }
            }
        }
        Ok(out)
    }

    fn normalize_head_atom(&mut self, atom: &Atom) -> Result<Atom> {
        let mut terms = Vec::with_capacity(atom.terms.len());
        for (idx, term) in atom.terms.iter().enumerate() {
            let expected = self
                .list_columns
                .get(&(atom.predicate.clone(), idx))
                .map(|col| col.elem_type);
            terms.push(self.normalize_value_term(term, expected)?);
        }
        Ok(Atom {
            predicate: atom.predicate.clone(),
            terms,
        })
    }

    fn expand_atom(
        &mut self,
        atom: &Atom,
        negated: bool,
        var_list_types: &mut HashMap<String, ScalarType>,
    ) -> Result<Vec<Atom>> {
        if let Some(atoms) = self.expand_list_builtin(atom, var_list_types)? {
            return Ok(atoms);
        }
        if is_reserved_pair_helper(atom) {
            return Err(list_error(
                "pair helpers are reserved for G085_META and are not implemented in G085_LIST",
            ));
        }

        let mut terms = Vec::with_capacity(atom.terms.len());
        let mut extra_atoms = Vec::new();
        for (idx, term) in atom.terms.iter().enumerate() {
            let expected = self
                .list_columns
                .get(&(atom.predicate.clone(), idx))
                .map(|col| col.elem_type);
            match term {
                Term::Variable(name) => {
                    if let Some(elem_type) = expected {
                        var_list_types.insert(name.clone(), elem_type);
                    }
                    terms.push(term.clone());
                }
                Term::Cons { head, tail } => {
                    if negated {
                        return Err(list_error(
                            "cons patterns in negated atoms are unsupported before a positive binder",
                        ));
                    }
                    let elem_type = expected.ok_or_else(|| {
                        list_error("cons pattern requires a declared list<T> column")
                    })?;
                    let list_var = self.fresh_var("list_cons");
                    let head_term = self.normalize_scalar_term(head, Some(elem_type))?.1;
                    let tail_arg =
                        self.normalize_list_arg(tail, Some(elem_type), var_list_types)?;
                    if let Term::Variable(name) = &tail_arg.term {
                        var_list_types.insert(name.clone(), elem_type);
                    }
                    terms.push(Term::Variable(list_var.clone()));
                    extra_atoms.push(Atom {
                        predicate: cons_pred(elem_type),
                        terms: vec![Term::Variable(list_var), head_term, tail_arg.term],
                    });
                }
                _ => {
                    let normalized = self.normalize_value_term(term, expected)?;
                    terms.push(normalized);
                }
            }
        }

        let mut atoms = vec![Atom {
            predicate: atom.predicate.clone(),
            terms,
        }];
        atoms.extend(extra_atoms);
        Ok(atoms)
    }

    fn expand_list_builtin(
        &mut self,
        atom: &Atom,
        var_list_types: &mut HashMap<String, ScalarType>,
    ) -> Result<Option<Vec<Atom>>> {
        let pred = atom.predicate.as_str();
        match pred {
            "is_list" => {
                require_arity(atom, 1)?;
                let arg = self.normalize_list_arg(&atom.terms[0], None, var_list_types)?;
                Ok(Some(vec![Atom {
                    predicate: len_pred().to_string(),
                    terms: vec![arg.term, Term::Anonymous],
                }]))
            }
            "member" | "memberchk" => {
                require_arity(atom, 2)?;
                let list = self.normalize_list_arg(&atom.terms[1], None, var_list_types)?;
                let value = self
                    .normalize_scalar_term(&atom.terms[0], Some(list.elem_type))?
                    .1;
                Ok(Some(vec![Atom {
                    predicate: item_pred(list.elem_type),
                    terms: vec![list.term, Term::Anonymous, value],
                }]))
            }
            "length" => {
                require_arity(atom, 2)?;
                let list = self.normalize_list_arg(&atom.terms[0], None, var_list_types)?;
                let len = self
                    .normalize_scalar_term(&atom.terms[1], Some(ScalarType::U32))?
                    .1;
                Ok(Some(vec![Atom {
                    predicate: len_pred().to_string(),
                    terms: vec![list.term, len],
                }]))
            }
            "nth" => {
                require_arity(atom, 3)?;
                let idx = self
                    .normalize_scalar_term(&atom.terms[0], Some(ScalarType::U32))?
                    .1;
                let list = self.normalize_list_arg(&atom.terms[1], None, var_list_types)?;
                let value = self
                    .normalize_scalar_term(&atom.terms[2], Some(list.elem_type))?
                    .1;
                Ok(Some(vec![Atom {
                    predicate: item_pred(list.elem_type),
                    terms: vec![list.term, idx, value],
                }]))
            }
            "append" => {
                require_arity(atom, 3)?;
                if matches!(atom.terms[2], Term::List(_) | Term::Cons { .. })
                    && !matches!(atom.terms[0], Term::List(_) | Term::Cons { .. })
                    && !matches!(atom.terms[1], Term::List(_) | Term::Cons { .. })
                {
                    return Err(list_error(
                        "unbounded append/3 generation is unsupported; bind the first two lists to finite literals in G085_LIST",
                    ));
                }
                let left = self.normalize_list_arg(&atom.terms[0], None, var_list_types)?;
                let right =
                    self.normalize_list_arg(&atom.terms[1], Some(left.elem_type), var_list_types)?;
                if left.elem_type != right.elem_type {
                    return Err(list_error(
                        "append/3 list arguments must have the same element type",
                    ));
                }
                let out =
                    self.normalize_list_arg(&atom.terms[2], Some(left.elem_type), var_list_types)?;
                if !left.is_literal || !right.is_literal {
                    return Err(list_error(
                        "unbounded append/3 generation is unsupported; bind the first two lists to finite literals in G085_LIST",
                    ));
                }
                let left_id = left.id.expect("literal list has id");
                let right_id = right.id.expect("literal list has id");
                let out_id = match out.id {
                    Some(id) => id,
                    None => self.concat_lists(left_id, right_id)?,
                };
                self.operation_rows.insert((
                    ListOp::Append,
                    left.elem_type,
                    vec![left_id, right_id, out_id],
                ));
                if let Term::Variable(name) = &out.term {
                    var_list_types.insert(name.clone(), left.elem_type);
                }
                Ok(Some(vec![Atom {
                    predicate: append_pred(left.elem_type),
                    terms: vec![left.term, right.term, out.term],
                }]))
            }
            "sort" | "msort" | "list_to_set" => {
                require_arity(atom, 2)?;
                let input = self.normalize_list_arg(&atom.terms[0], None, var_list_types)?;
                if !input.is_literal {
                    return Err(list_error(
                        "sort/msort/list_to_set require a finite list literal in G085_LIST",
                    ));
                }
                let op = match pred {
                    "sort" => ListOp::Sort,
                    "msort" => ListOp::MSort,
                    "list_to_set" => ListOp::ListToSet,
                    _ => unreachable!(),
                };
                let out =
                    self.normalize_list_arg(&atom.terms[1], Some(input.elem_type), var_list_types)?;
                let input_id = input.id.expect("literal list has id");
                let out_id = match out.id {
                    Some(id) => id,
                    None => self.derived_ordered_list(input_id, op)?,
                };
                self.operation_rows
                    .insert((op, input.elem_type, vec![input_id, out_id]));
                if let Term::Variable(name) = &out.term {
                    var_list_types.insert(name.clone(), input.elem_type);
                }
                let helper = match op {
                    ListOp::Sort => sort_pred(input.elem_type),
                    ListOp::MSort => msort_pred(input.elem_type),
                    ListOp::ListToSet => list_to_set_pred(input.elem_type),
                    ListOp::Append => unreachable!(),
                };
                Ok(Some(vec![Atom {
                    predicate: helper,
                    terms: vec![input.term, out.term],
                }]))
            }
            _ => Ok(None),
        }
    }

    fn normalize_value_term(
        &mut self,
        term: &Term,
        expected_list_elem: Option<ScalarType>,
    ) -> Result<Term> {
        match term {
            Term::List(_) => Ok(self
                .normalize_list_arg(term, expected_list_elem, &mut HashMap::new())?
                .term),
            Term::Cons { head, tail } => {
                let elem_type = expected_list_elem.ok_or_else(|| {
                    list_error("cons literal requires a declared list<T> context")
                })?;
                let mut items = vec![self.normalize_scalar_term(head, Some(elem_type))?.1];
                let tail_arg =
                    self.normalize_list_arg(tail, Some(elem_type), &mut HashMap::new())?;
                let tail_id = tail_arg
                    .id
                    .ok_or_else(|| list_error("cons literal tail must be a finite list literal"))?;
                let tail_items = self.list_by_id(tail_id)?.items.clone();
                items.extend(tail_items);
                let id = self.register_list_terms(items, Some(elem_type))?;
                Ok(Term::Integer(id as i64))
            }
            _ => Ok(term.clone()),
        }
    }

    fn normalize_list_arg(
        &mut self,
        term: &Term,
        expected_elem_type: Option<ScalarType>,
        var_list_types: &mut HashMap<String, ScalarType>,
    ) -> Result<ListArg> {
        match term {
            Term::List(items) => {
                let id = self.register_list_terms(items.clone(), expected_elem_type)?;
                let elem_type = self.list_by_id(id)?.elem_type;
                Ok(ListArg {
                    term: Term::Integer(id as i64),
                    id: Some(id),
                    elem_type,
                    is_literal: true,
                })
            }
            Term::Cons { .. } => {
                let value = self.normalize_value_term(term, expected_elem_type)?;
                let id = match value {
                    Term::Integer(id) if id >= 0 => id as u64,
                    _ => {
                        return Err(list_error(
                            "cons list argument must normalize to a finite list id",
                        ));
                    }
                };
                let elem_type = self.list_by_id(id)?.elem_type;
                Ok(ListArg {
                    term: Term::Integer(id as i64),
                    id: Some(id),
                    elem_type,
                    is_literal: true,
                })
            }
            Term::Integer(id) if *id > 0 => {
                let elem_type = expected_elem_type
                    .or_else(|| self.list_by_id(*id as u64).ok().map(|list| list.elem_type))
                    .ok_or_else(|| list_error("list id requires a known list<T> type"))?;
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
                            "list built-in requires a finite literal or known list<T> variable",
                        )
                    })?;
                var_list_types.insert(name.clone(), elem_type);
                Ok(ListArg {
                    term: term.clone(),
                    id: None,
                    elem_type,
                    is_literal: false,
                })
            }
            Term::Anonymous => {
                let elem_type = expected_elem_type.ok_or_else(|| {
                    list_error("anonymous list argument requires a known list<T> type")
                })?;
                Ok(ListArg {
                    term: term.clone(),
                    id: None,
                    elem_type,
                    is_literal: false,
                })
            }
            _ => Err(list_error(
                "expected finite list literal or known list<T> variable",
            )),
        }
    }

    fn normalize_scalar_term(
        &mut self,
        term: &Term,
        expected: Option<ScalarType>,
    ) -> Result<(ScalarType, Term)> {
        match term {
            Term::Variable(_) | Term::Anonymous => {
                let typ = expected.ok_or_else(|| {
                    list_error("list element variable requires a known list<T> element type")
                })?;
                Ok((typ, term.clone()))
            }
            Term::Integer(i) => match expected {
                Some(ScalarType::U32) => {
                    let value = u32::try_from(*i)
                        .map_err(|_| list_error("integer list element is out of range for u32"))?;
                    Ok((ScalarType::U32, Term::Integer(value as i64)))
                }
                Some(ScalarType::U64) => {
                    let value = u64::try_from(*i)
                        .map_err(|_| list_error("integer list element is out of range for u64"))?;
                    Ok((ScalarType::U64, Term::Integer(value as i64)))
                }
                Some(ScalarType::I32) => {
                    let value = i32::try_from(*i)
                        .map_err(|_| list_error("integer list element is out of range for i32"))?;
                    Ok((ScalarType::I32, Term::Integer(value as i64)))
                }
                Some(ScalarType::I64) | None if *i < 0 || *i > u32::MAX as i64 => {
                    Ok((ScalarType::I64, Term::Integer(*i)))
                }
                Some(ScalarType::I64) => Ok((ScalarType::I64, Term::Integer(*i))),
                None => Ok((ScalarType::U32, Term::Integer(*i))),
                Some(ScalarType::F32) => Ok((ScalarType::F32, Term::Float(*i as f64))),
                Some(ScalarType::F64) => Ok((ScalarType::F64, Term::Float(*i as f64))),
                Some(ScalarType::Bool) if *i == 0 || *i == 1 => {
                    Ok((ScalarType::Bool, Term::Integer(*i)))
                }
                Some(ScalarType::Bool) => Err(list_error("bool list elements must be 0 or 1")),
                Some(ScalarType::Symbol) => {
                    Err(list_error("integer list element is not valid for symbol"))
                }
            },
            Term::Float(f) => match expected {
                Some(ScalarType::F32) => Ok((ScalarType::F32, Term::Float(*f))),
                Some(ScalarType::F64) | None => Ok((ScalarType::F64, Term::Float(*f))),
                Some(_) => Err(list_error(
                    "float list element is not valid for expected type",
                )),
            },
            Term::String(s) => {
                if expected.is_none() || expected == Some(ScalarType::Symbol) {
                    Ok((ScalarType::Symbol, Term::String(s.clone())))
                } else {
                    Err(list_error(
                        "string list element is only valid for symbol lists",
                    ))
                }
            }
            Term::Symbol(id) => {
                if expected.is_none() || expected == Some(ScalarType::Symbol) {
                    Ok((ScalarType::Symbol, Term::Symbol(*id)))
                } else {
                    Err(list_error(
                        "symbol list element is not valid for expected type",
                    ))
                }
            }
            Term::List(_) | Term::Cons { .. } => {
                let id = self
                    .normalize_list_arg(term, None, &mut HashMap::new())?
                    .id
                    .expect("literal list has id");
                let typ = expected.unwrap_or(LIST_ID_TYPE);
                if typ != LIST_ID_TYPE {
                    return Err(list_error(
                        "nested list element requires list<list<T>> or list<u64> context",
                    ));
                }
                Ok((LIST_ID_TYPE, Term::Integer(id as i64)))
            }
            Term::Compound { .. } | Term::PredRef(_) | Term::Aggregate(_) => Err(list_error(
                "compound, predref, and aggregate list elements require later G085_META support",
            )),
        }
    }

    fn register_list_terms(
        &mut self,
        items: Vec<Term>,
        expected_elem_type: Option<ScalarType>,
    ) -> Result<u64> {
        let mut elem_type = expected_elem_type;
        let mut normalized = Vec::with_capacity(items.len());
        let mut keys = Vec::with_capacity(items.len());

        for item in &items {
            let (item_type, term) = match self.normalize_scalar_term(item, elem_type) {
                Ok(value) => value,
                Err(err) if elem_type.is_some() => {
                    return Err(list_error(format!("heterogeneous list literal: {}", err)));
                }
                Err(err) => return Err(err),
            };
            if let Some(existing) = elem_type {
                if existing != item_type {
                    return Err(list_error(
                        "heterogeneous list literals require a declared finite term type",
                    ));
                }
            } else {
                elem_type = Some(item_type);
            }
            keys.push(literal_key(&term, item_type)?);
            normalized.push(term);
        }

        let elem_type = elem_type.unwrap_or(LIST_ID_TYPE);
        let key = (elem_type, keys.clone());
        if let Some(id) = self.lists_by_key.get(&key) {
            return Ok(*id);
        }

        let id = self.next_list_id;
        self.next_list_id += 1;
        self.lists_by_key.insert(key, id);
        self.lists.push(ListDef {
            id,
            elem_type,
            items: normalized,
            keys,
        });
        Ok(id)
    }

    fn concat_lists(&mut self, left_id: u64, right_id: u64) -> Result<u64> {
        let left = self.list_by_id(left_id)?.clone();
        let right = self.list_by_id(right_id)?.clone();
        if left.elem_type != right.elem_type {
            return Err(list_error(
                "append/3 list arguments must have the same element type",
            ));
        }
        let mut items = left.items;
        items.extend(right.items);
        self.register_list_terms(items, Some(right.elem_type))
    }

    fn derived_ordered_list(&mut self, input_id: u64, op: ListOp) -> Result<u64> {
        let input = self.list_by_id(input_id)?.clone();
        let mut pairs: Vec<(LiteralKey, Term)> = input.keys.into_iter().zip(input.items).collect();
        pairs.sort_by(|(a, _), (b, _)| format!("{a:?}").cmp(&format!("{b:?}")));
        if matches!(op, ListOp::Sort | ListOp::ListToSet) {
            pairs.dedup_by(|(a, _), (b, _)| a == b);
        }
        let items = pairs.into_iter().map(|(_, term)| term).collect();
        self.register_list_terms(items, Some(input.elem_type))
    }

    fn append_helper_declarations_and_facts(&mut self, program: &mut Program) -> Result<()> {
        self.ensure_tail_lists()?;

        let mut helper_rules = Vec::new();
        for list in self.lists.clone() {
            self.helper_preds.insert(len_pred().to_string());
            helper_rules.push(fact(
                len_pred(),
                vec![
                    Term::Integer(list.id as i64),
                    Term::Integer(list.items.len() as i64),
                ],
            ));

            let item_name = item_pred(list.elem_type);
            self.helper_preds.insert(item_name.clone());
            for (idx, item) in list.items.iter().enumerate() {
                helper_rules.push(fact(
                    &item_name,
                    vec![
                        Term::Integer(list.id as i64),
                        Term::Integer(idx as i64),
                        item.clone(),
                    ],
                ));
            }

            if let Some((head, tail_id)) = self.cons_fact_parts(list.id)? {
                let cons_name = cons_pred(list.elem_type);
                self.helper_preds.insert(cons_name.clone());
                helper_rules.push(fact(
                    &cons_name,
                    vec![
                        Term::Integer(list.id as i64),
                        head,
                        Term::Integer(tail_id as i64),
                    ],
                ));
            }
        }

        for (op, elem_type, ids) in self.operation_rows.clone() {
            let pred = match op {
                ListOp::Append => append_pred(elem_type),
                ListOp::Sort => sort_pred(elem_type),
                ListOp::MSort => msort_pred(elem_type),
                ListOp::ListToSet => list_to_set_pred(elem_type),
            };
            self.helper_preds.insert(pred.clone());
            helper_rules.push(fact(
                &pred,
                ids.into_iter().map(|id| Term::Integer(id as i64)).collect(),
            ));
        }

        self.append_helper_pred_decls(program);
        program.rules.extend(helper_rules);
        Ok(())
    }

    fn ensure_tail_lists(&mut self) -> Result<()> {
        loop {
            let before = self.lists.len();
            for list in self.lists.clone() {
                if list.items.is_empty() {
                    continue;
                }
                self.register_list_terms(list.items[1..].to_vec(), Some(list.elem_type))?;
            }
            if self.lists.len() == before {
                break;
            }
        }
        Ok(())
    }

    fn cons_fact_parts(&self, list_id: u64) -> Result<Option<(Term, u64)>> {
        let list = self.list_by_id(list_id)?;
        let Some(head) = list.items.first() else {
            return Ok(None);
        };
        let tail_key = (list.elem_type, list.keys[1..].to_vec());
        let tail_id = self
            .lists_by_key
            .get(&tail_key)
            .copied()
            .ok_or_else(|| list_error("missing normalized list tail"))?;
        Ok(Some((head.clone(), tail_id)))
    }

    fn append_helper_pred_decls(&self, program: &mut Program) {
        let mut existing: HashSet<String> = program
            .predicates
            .iter()
            .map(|pred| pred.name.clone())
            .collect();

        for pred in &self.helper_preds {
            if existing.contains(pred) {
                continue;
            }
            let columns = helper_columns(pred);
            program.predicates.push(PredDecl {
                name: pred.clone(),
                types: columns.iter().map(|col| col.typ.clone()).collect(),
                columns,
                is_private: true,
            });
            existing.insert(pred.clone());
        }
    }

    fn list_by_id(&self, id: u64) -> Result<&ListDef> {
        self.lists
            .iter()
            .find(|list| list.id == id)
            .ok_or_else(|| list_error("unknown normalized list id"))
    }

    fn fresh_var(&mut self, prefix: &str) -> String {
        let var = format!(
            "__XLOG_{}_{}",
            prefix.to_ascii_uppercase(),
            self.fresh_counter
        );
        self.fresh_counter += 1;
        var
    }

    fn storage_type_for_type_ref(
        typ: &TypeRef,
        domains: &HashMap<String, ScalarType>,
    ) -> Result<ScalarType> {
        match typ {
            TypeRef::Scalar(ty) => Ok(*ty),
            TypeRef::Domain(name) => domains
                .get(name)
                .copied()
                .ok_or_else(|| list_error(format!("unknown domain alias '{}' in list<T>", name))),
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

fn require_arity(atom: &Atom, expected: usize) -> Result<()> {
    if atom.terms.len() == expected {
        Ok(())
    } else {
        Err(list_error(format!(
            "{} expects {} arguments, got {}",
            atom.predicate,
            expected,
            atom.terms.len()
        )))
    }
}

fn literal_key(term: &Term, typ: ScalarType) -> Result<LiteralKey> {
    match (typ, term) {
        (ScalarType::U32, Term::Integer(v)) => Ok(LiteralKey::U32(*v as u32)),
        (ScalarType::U64, Term::Integer(v)) => Ok(LiteralKey::U64(*v as u64)),
        (ScalarType::I32, Term::Integer(v)) => Ok(LiteralKey::I32(*v as i32)),
        (ScalarType::I64, Term::Integer(v)) => Ok(LiteralKey::I64(*v)),
        (ScalarType::F32, Term::Float(v)) => Ok(LiteralKey::F32((*v as f32).to_bits())),
        (ScalarType::F64, Term::Float(v)) => Ok(LiteralKey::F64(v.to_bits())),
        (ScalarType::Bool, Term::Integer(v)) => Ok(LiteralKey::Bool(*v != 0)),
        (ScalarType::Symbol, Term::String(s)) => Ok(LiteralKey::Symbol(s.clone())),
        (ScalarType::Symbol, Term::Symbol(id)) => {
            Ok(LiteralKey::Symbol(xlog_core::symbol::resolve(*id)))
        }
        _ => Err(list_error(
            "list literal element did not normalize to expected scalar type",
        )),
    }
}

fn helper_columns(pred: &str) -> Vec<PredColumn> {
    let scalar = |name: &str, typ| PredColumn {
        name: Some(name.to_string()),
        typ: TypeRef::Scalar(typ),
    };
    if pred == len_pred() {
        return vec![
            scalar("list_id", LIST_ID_TYPE),
            scalar("len", ScalarType::U32),
        ];
    }
    if let Some(elem_type) = helper_elem_type(pred, "__xlog_list_item_") {
        return vec![
            scalar("list_id", LIST_ID_TYPE),
            scalar("idx", ScalarType::U32),
            scalar("value", elem_type),
        ];
    }
    if let Some(elem_type) = helper_elem_type(pred, "__xlog_list_cons_") {
        return vec![
            scalar("list_id", LIST_ID_TYPE),
            scalar("head", elem_type),
            scalar("tail_id", LIST_ID_TYPE),
        ];
    }
    if helper_elem_type(pred, "__xlog_list_append_").is_some() {
        return vec![
            scalar("left_id", LIST_ID_TYPE),
            scalar("right_id", LIST_ID_TYPE),
            scalar("out_id", LIST_ID_TYPE),
        ];
    }
    vec![
        scalar("input_id", LIST_ID_TYPE),
        scalar("out_id", LIST_ID_TYPE),
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

fn list_error(message: impl Into<String>) -> XlogError {
    XlogError::Compilation(format!("v0.8.5 list error: {}", message.into()))
}
