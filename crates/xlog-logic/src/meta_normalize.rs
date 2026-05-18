//! Safe meta-predicate normalization for the v0.8.5 language surface.

use std::collections::{BTreeMap, HashMap, HashSet};

use xlog_core::{symbol, Result, ScalarType, XlogError};

use crate::ast::{
    Atom, BodyLiteral, Comparison, Constraint, DomainDecl, PredColumn, PredDecl, Program, Query,
    Rule, Term, TypeRef, Univ,
};

const TERM_ID_TYPE: ScalarType = ScalarType::U64;

/// Normalize safe v0.8.5 meta-predicates into finite helper relations.
///
/// Accepted forms are static and finite. Runtime-variable predicate names,
/// dynamic database mutation, unrestricted calls, and non-finite collection are
/// rejected before lowering.
pub fn normalize_v085_meta(program: &Program) -> Result<Program> {
    let mut normalizer = MetaNormalizer::new(program)?;
    normalizer.normalize_program(program)
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

#[derive(Debug, Clone)]
struct TermRecord {
    id: u64,
    kind: RegisteredTerm,
}

struct MetaNormalizer {
    domains: HashMap<String, ScalarType>,
    pred_columns: HashMap<(String, usize), TypeRef>,
    source_facts: HashMap<String, Vec<Atom>>,
    derived_predicates: HashSet<String>,
    term_ids: HashMap<String, u64>,
    term_records: Vec<TermRecord>,
    next_term_id: u64,
    helper_decls: BTreeMap<String, Vec<PredColumn>>,
    helper_facts: Vec<Rule>,
    helper_fact_keys: HashSet<String>,
    findall_counter: usize,
    maplist_counter: usize,
}

impl MetaNormalizer {
    fn new(program: &Program) -> Result<Self> {
        let domains: HashMap<String, ScalarType> = program
            .domains
            .iter()
            .map(|DomainDecl { name, typ }| (name.clone(), *typ))
            .collect();

        let mut pred_columns = HashMap::new();
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
            for (idx, col) in columns.into_iter().enumerate() {
                pred_columns.insert((pred.name.clone(), idx), col.typ);
            }
        }

        let mut source_facts: HashMap<String, Vec<Atom>> = HashMap::new();
        let mut derived_predicates = HashSet::new();
        for rule in &program.rules {
            if rule.is_fact() {
                source_facts
                    .entry(rule.head.predicate.clone())
                    .or_default()
                    .push(rule.head.clone());
            } else {
                derived_predicates.insert(rule.head.predicate.clone());
            }
        }

        Ok(Self {
            domains,
            pred_columns,
            source_facts,
            derived_predicates,
            term_ids: HashMap::new(),
            term_records: Vec::new(),
            next_term_id: 1,
            helper_decls: BTreeMap::new(),
            helper_facts: Vec::new(),
            helper_fact_keys: HashSet::new(),
            findall_counter: 0,
            maplist_counter: 0,
        })
    }

    fn normalize_program(&mut self, program: &Program) -> Result<Program> {
        let mut out = program.clone();
        out.rules = program
            .rules
            .iter()
            .map(|rule| self.normalize_rule(rule))
            .collect::<Result<Vec<_>>>()?;
        out.constraints = program
            .constraints
            .iter()
            .map(|constraint| {
                let mut bound = HashMap::new();
                Ok(Constraint {
                    body: self.normalize_body(&constraint.body, &mut bound)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        out.queries = program
            .queries
            .iter()
            .map(|query| {
                Ok(Query {
                    atom: self.normalize_atom_values(&query.atom)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        out.prob_facts = program
            .prob_facts
            .iter()
            .map(|pf| {
                let mut pf = pf.clone();
                pf.atom = self.normalize_atom_values(&pf.atom)?;
                Ok(pf)
            })
            .collect::<Result<Vec<_>>>()?;
        out.annotated_disjunctions = program
            .annotated_disjunctions
            .iter()
            .map(|ad| {
                let mut ad = ad.clone();
                for choice in &mut ad.choices {
                    choice.atom = self.normalize_atom_values(&choice.atom)?;
                }
                Ok(ad)
            })
            .collect::<Result<Vec<_>>>()?;
        out.evidence = program
            .evidence
            .iter()
            .map(|evidence| {
                let mut evidence = evidence.clone();
                evidence.atom = self.normalize_atom_values(&evidence.atom)?;
                Ok(evidence)
            })
            .collect::<Result<Vec<_>>>()?;
        out.prob_queries = program
            .prob_queries
            .iter()
            .map(|query| {
                let mut query = query.clone();
                query.atom = self.normalize_atom_values(&query.atom)?;
                Ok(query)
            })
            .collect::<Result<Vec<_>>>()?;
        out.neural_predicates = program
            .neural_predicates
            .iter()
            .map(|neural| {
                let mut neural = neural.clone();
                neural.predicate = self.normalize_atom_values(&neural.predicate)?;
                Ok(neural)
            })
            .collect::<Result<Vec<_>>>()?;
        out.learnable_rules = program
            .learnable_rules
            .iter()
            .map(|learnable| {
                let mut learnable = learnable.clone();
                let mut bound = HashMap::new();
                learnable.head = self.normalize_atom_values(&learnable.head)?;
                learnable.body = self.normalize_body(&learnable.body, &mut bound)?;
                Ok(learnable)
            })
            .collect::<Result<Vec<_>>>()?;

        self.append_helpers(&mut out);
        lower_meta_type_refs_in_predicates(&mut out);
        Ok(out)
    }

    fn normalize_rule(&mut self, rule: &Rule) -> Result<Rule> {
        let head = self.normalize_atom_values(&rule.head)?;
        let mut bound = HashMap::new();
        let body = self.normalize_body(&rule.body, &mut bound)?;
        Ok(Rule { head, body })
    }

    fn normalize_body(
        &mut self,
        body: &[BodyLiteral],
        bound: &mut HashMap<String, ScalarType>,
    ) -> Result<Vec<BodyLiteral>> {
        let mut out = Vec::new();
        for lit in body {
            match lit {
                BodyLiteral::Positive(atom) => {
                    let expanded = self.normalize_positive_atom(atom, bound)?;
                    for lit in expanded {
                        self.record_bound_from_literal(&lit, bound);
                        out.push(lit);
                    }
                }
                BodyLiteral::Negated(atom) => {
                    if is_meta_predicate(&atom.predicate)
                        || is_blocked_dynamic_predicate(&atom.predicate)
                    {
                        return Err(meta_error(
                            "safe meta-predicates are not supported under negation in G085_META",
                        ));
                    }
                    out.push(BodyLiteral::Negated(self.normalize_atom_values(atom)?));
                }
                BodyLiteral::Comparison(cmp) => out.push(BodyLiteral::Comparison(Comparison {
                    op: cmp.op,
                    left: self.normalize_untyped_value(&cmp.left)?,
                    right: self.normalize_untyped_value(&cmp.right)?,
                })),
                BodyLiteral::IsExpr(is_expr) => out.push(BodyLiteral::IsExpr(is_expr.clone())),
                BodyLiteral::Univ(univ) => {
                    let lit = self.expand_univ(univ, bound)?;
                    self.record_bound_from_literal(&lit, bound);
                    out.push(lit);
                }
            }
        }
        Ok(out)
    }

    fn normalize_positive_atom(
        &mut self,
        atom: &Atom,
        bound: &HashMap<String, ScalarType>,
    ) -> Result<Vec<BodyLiteral>> {
        if is_blocked_dynamic_predicate(&atom.predicate) {
            return Err(dynamic_predicate_error(&atom.predicate));
        }

        match atom.predicate.as_str() {
            "ground" => self.expand_ground_like(atom, bound, MetaTruthKind::Ground),
            "var" => self.expand_ground_like(atom, bound, MetaTruthKind::Var),
            "nonvar" => self.expand_ground_like(atom, bound, MetaTruthKind::NonVar),
            "functor" => self.expand_functor(atom, bound),
            "findall" => self.expand_findall(atom, bound),
            "maplist" => self.expand_maplist(atom),
            _ => Ok(vec![BodyLiteral::Positive(
                self.normalize_atom_values(atom)?,
            )]),
        }
    }

    fn normalize_atom_values(&mut self, atom: &Atom) -> Result<Atom> {
        let mut terms = Vec::with_capacity(atom.terms.len());
        for (idx, term) in atom.terms.iter().enumerate() {
            let expected = self
                .pred_columns
                .get(&(atom.predicate.clone(), idx))
                .cloned();
            terms.push(self.normalize_value_term(term, expected.as_ref())?);
        }
        Ok(Atom {
            predicate: atom.predicate.clone(),
            terms,
        })
    }

    fn normalize_value_term(&mut self, term: &Term, expected: Option<&TypeRef>) -> Result<Term> {
        match expected {
            Some(TypeRef::Term) if term.is_any_variable() => Ok(term.clone()),
            Some(TypeRef::Term) => Ok(Term::Integer(
                self.register_term(term, ExpectedTerm::Any)? as i64
            )),
            Some(TypeRef::Compound) if term.is_any_variable() => Ok(term.clone()),
            Some(TypeRef::Compound) => Ok(Term::Integer(
                self.register_term(term, ExpectedTerm::Compound)? as i64,
            )),
            Some(TypeRef::PredRef) if term.is_any_variable() => Ok(term.clone()),
            Some(TypeRef::PredRef) => Ok(Term::Integer(
                self.register_term(term, ExpectedTerm::PredRef)? as i64,
            )),
            Some(TypeRef::List(inner)) => self.normalize_list_value(term, inner),
            Some(TypeRef::Scalar(_)) | Some(TypeRef::Domain(_)) | None => {
                self.normalize_untyped_value(term)
            }
        }
    }

    fn normalize_untyped_value(&mut self, term: &Term) -> Result<Term> {
        match term {
            Term::List(items) => Ok(Term::List(
                items
                    .iter()
                    .map(|item| self.normalize_untyped_value(item))
                    .collect::<Result<Vec<_>>>()?,
            )),
            Term::Cons { head, tail } => Ok(Term::Cons {
                head: Box::new(self.normalize_untyped_value(head)?),
                tail: Box::new(self.normalize_untyped_value(tail)?),
            }),
            _ => Ok(term.clone()),
        }
    }

    fn normalize_list_value(&mut self, term: &Term, inner: &TypeRef) -> Result<Term> {
        match term {
            Term::List(items) => Ok(Term::List(
                items
                    .iter()
                    .map(|item| self.normalize_value_term(item, Some(inner)))
                    .collect::<Result<Vec<_>>>()?,
            )),
            Term::Cons { head, tail } => Ok(Term::Cons {
                head: Box::new(self.normalize_value_term(head, Some(inner))?),
                tail: Box::new(self.normalize_list_value(tail, inner)?),
            }),
            Term::Variable(_) | Term::Anonymous | Term::Integer(_) => Ok(term.clone()),
            _ => Err(meta_error(
                "list<T> column expects a finite list literal, list id, or variable",
            )),
        }
    }

    fn expand_ground_like(
        &mut self,
        atom: &Atom,
        bound: &HashMap<String, ScalarType>,
        kind: MetaTruthKind,
    ) -> Result<Vec<BodyLiteral>> {
        require_arity(atom, 1)?;
        let truth = match kind {
            MetaTruthKind::Ground => self.is_ground(&atom.terms[0], bound),
            MetaTruthKind::Var => self.is_var(&atom.terms[0], bound),
            MetaTruthKind::NonVar => self.is_nonvar(&atom.terms[0], bound),
        };
        if truth {
            Ok(Vec::new())
        } else {
            Ok(vec![BodyLiteral::Positive(self.fail_atom())])
        }
    }

    fn expand_functor(
        &mut self,
        atom: &Atom,
        bound: &HashMap<String, ScalarType>,
    ) -> Result<Vec<BodyLiteral>> {
        require_arity(atom, 3)?;
        let term = self.term_id_or_bound_variable(&atom.terms[0], bound, "functor/3")?;
        let name = self.normalize_symbol_argument(&atom.terms[1], "functor/3 name")?;
        let arity = self.normalize_u32_argument(&atom.terms[2], "functor/3 arity")?;
        self.register_helper_decl(
            functor_pred(),
            vec![
                scalar_col("term_id", TERM_ID_TYPE),
                scalar_col("name", ScalarType::Symbol),
                scalar_col("arity", ScalarType::U32),
            ],
        );
        Ok(vec![BodyLiteral::Positive(Atom {
            predicate: functor_pred().to_string(),
            terms: vec![term, name, arity],
        })])
    }

    fn expand_univ(
        &mut self,
        univ: &Univ,
        bound: &HashMap<String, ScalarType>,
    ) -> Result<BodyLiteral> {
        let term = self.term_id_or_bound_variable(&univ.term, bound, "univ")?;
        let parts = self.normalize_univ_parts(&univ.parts)?;
        self.register_helper_decl(
            univ_pred(),
            vec![
                scalar_col("term_id", TERM_ID_TYPE),
                PredColumn {
                    name: Some("parts".to_string()),
                    typ: TypeRef::List(Box::new(TypeRef::Term)),
                },
            ],
        );
        Ok(BodyLiteral::Positive(Atom {
            predicate: univ_pred().to_string(),
            terms: vec![term, parts],
        }))
    }

    fn expand_findall(
        &mut self,
        atom: &Atom,
        bound: &HashMap<String, ScalarType>,
    ) -> Result<Vec<BodyLiteral>> {
        require_arity(atom, 3)?;
        let template = &atom.terms[0];
        let goal = compound_goal_atom(&atom.terms[1], "findall/3")?;
        let out_list = atom.terms[2].clone();
        if self.derived_predicates.contains(&goal.predicate) {
            return Err(meta_error(
                "findall/3 in G085_META is limited to finite source facts; derived goals are reserved for a later aggregate-backed collection path",
            ));
        }

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
                    return Err(meta_error(
                        "unsafe findall/3 goal variable is neither bound before findall nor collected by the template",
                    ));
                }
            }
        }

        let elem_type_ref = self.infer_template_type(template, &goal)?;
        let helper = format!("__xlog_meta_findall_{}", self.findall_counter);
        self.findall_counter += 1;

        let mut columns = Vec::new();
        for (idx, name) in &group_positions {
            columns.push(PredColumn {
                name: Some(name.to_ascii_lowercase()),
                typ: self
                    .pred_columns
                    .get(&(goal.predicate.clone(), *idx))
                    .cloned()
                    .unwrap_or(TypeRef::Scalar(
                        bound.get(name).copied().unwrap_or(ScalarType::U64),
                    )),
            });
        }
        columns.push(PredColumn {
            name: Some("results".to_string()),
            typ: TypeRef::List(Box::new(elem_type_ref.clone())),
        });
        self.register_helper_decl(&helper, columns);

        let mut groups: BTreeMap<String, (Vec<Term>, Vec<Term>)> = BTreeMap::new();
        for fact in self
            .source_facts
            .get(&goal.predicate)
            .cloned()
            .unwrap_or_default()
        {
            if fact.terms.len() != goal.terms.len() {
                continue;
            }
            let Some(bindings) = self.match_goal_fact(&goal, &fact)? else {
                continue;
            };
            let group_terms: Vec<Term> = group_positions
                .iter()
                .map(|(idx, _)| fact.terms[*idx].clone())
                .collect();
            let key = group_terms
                .iter()
                .map(term_match_key)
                .collect::<Vec<_>>()
                .join("|");
            let template_value = self.substitute_template(template, &bindings, &elem_type_ref)?;
            groups
                .entry(key)
                .or_insert_with(|| (group_terms, Vec::new()))
                .1
                .push(template_value);
        }

        if groups.is_empty() && group_positions.is_empty() {
            groups.insert(String::new(), (Vec::new(), Vec::new()));
        }

        for (_, (mut group_terms, values)) in groups {
            let list = Term::List(values);
            group_terms.push(list);
            self.add_helper_fact(&helper, group_terms);
        }

        let mut terms: Vec<Term> = group_positions
            .iter()
            .map(|(_, name)| Term::Variable(name.clone()))
            .collect();
        terms.push(out_list);
        Ok(vec![BodyLiteral::Positive(Atom {
            predicate: helper,
            terms,
        })])
    }

    fn expand_maplist(&mut self, atom: &Atom) -> Result<Vec<BodyLiteral>> {
        if atom.terms.len() != 2 && atom.terms.len() != 3 {
            return Err(meta_error(format!(
                "maplist expects unary or binary form, got {} arguments",
                atom.terms.len()
            )));
        }
        let pred = static_pred_name(&atom.terms[0])?;
        if self.derived_predicates.contains(&pred) {
            return Err(meta_error(
                "maplist in G085_META is limited to finite source facts or literal empty lists",
            ));
        }
        let input_items = finite_list_items(&atom.terms[1], "maplist input")?;
        let input_type = self
            .pred_columns
            .get(&(pred.clone(), 0))
            .cloned()
            .unwrap_or_else(|| infer_type_ref_from_terms(input_items));
        let input_list = Term::List(
            input_items
                .iter()
                .map(|item| self.normalize_value_term(item, Some(&input_type)))
                .collect::<Result<Vec<_>>>()?,
        );
        let helper = format!("__xlog_meta_maplist_{}_{}", pred, self.maplist_counter);
        self.maplist_counter += 1;

        if atom.terms.len() == 2 {
            self.register_helper_decl(
                &helper,
                vec![PredColumn {
                    name: Some("input".to_string()),
                    typ: TypeRef::List(Box::new(input_type)),
                }],
            );
            if self.maplist_unary_holds(&pred, &input_list)? {
                self.add_helper_fact(&helper, vec![input_list.clone()]);
            }
            return Ok(vec![BodyLiteral::Positive(Atom {
                predicate: helper,
                terms: vec![atom.terms[1].clone()],
            })]);
        }

        let output_type = self
            .pred_columns
            .get(&(pred.clone(), 1))
            .cloned()
            .unwrap_or(TypeRef::Scalar(ScalarType::U64));
        self.register_helper_decl(
            &helper,
            vec![
                PredColumn {
                    name: Some("input".to_string()),
                    typ: TypeRef::List(Box::new(input_type)),
                },
                PredColumn {
                    name: Some("output".to_string()),
                    typ: TypeRef::List(Box::new(output_type.clone())),
                },
            ],
        );
        for output in self.maplist_binary_outputs(&pred, &input_list, &output_type)? {
            self.add_helper_fact(&helper, vec![input_list.clone(), Term::List(output)]);
        }
        Ok(vec![BodyLiteral::Positive(Atom {
            predicate: helper,
            terms: vec![atom.terms[1].clone(), atom.terms[2].clone()],
        })])
    }

    fn maplist_unary_holds(&self, pred: &str, input: &Term) -> Result<bool> {
        let Term::List(items) = input else {
            return Err(meta_error(
                "maplist input did not normalize to a finite list",
            ));
        };
        let facts = self.source_facts.get(pred).cloned().unwrap_or_default();
        Ok(items.iter().all(|item| {
            facts.iter().any(|fact| {
                fact.terms.len() == 1 && term_match_key(&fact.terms[0]) == term_match_key(item)
            })
        }))
    }

    fn maplist_binary_outputs(
        &mut self,
        pred: &str,
        input: &Term,
        output_type: &TypeRef,
    ) -> Result<Vec<Vec<Term>>> {
        let Term::List(items) = input else {
            return Err(meta_error(
                "maplist input did not normalize to a finite list",
            ));
        };
        if items.is_empty() {
            return Ok(vec![Vec::new()]);
        }
        let facts = self.source_facts.get(pred).cloned().unwrap_or_default();
        let mut choices: Vec<Vec<Term>> = Vec::new();
        for item in items {
            let mut outs = Vec::new();
            for fact in &facts {
                if fact.terms.len() == 2 && term_match_key(&fact.terms[0]) == term_match_key(item) {
                    outs.push(self.normalize_value_term(&fact.terms[1], Some(output_type))?);
                }
            }
            if outs.is_empty() {
                return Ok(Vec::new());
            }
            choices.push(outs);
        }
        Ok(cartesian_product(&choices))
    }

    fn match_goal_fact(&self, goal: &Atom, fact: &Atom) -> Result<Option<HashMap<String, Term>>> {
        let mut bindings = HashMap::new();
        for (goal_term, fact_term) in goal.terms.iter().zip(&fact.terms) {
            match goal_term {
                Term::Variable(name) => {
                    if let Some(existing) = bindings.get(name) {
                        if term_match_key(existing) != term_match_key(fact_term) {
                            return Ok(None);
                        }
                    } else {
                        bindings.insert(name.clone(), fact_term.clone());
                    }
                }
                Term::Anonymous => {}
                _ => {
                    if term_match_key(goal_term) != term_match_key(fact_term) {
                        return Ok(None);
                    }
                }
            }
        }
        Ok(Some(bindings))
    }

    fn substitute_template(
        &mut self,
        template: &Term,
        bindings: &HashMap<String, Term>,
        elem_type: &TypeRef,
    ) -> Result<Term> {
        let value = match template {
            Term::Variable(name) => bindings.get(name).cloned().ok_or_else(|| {
                meta_error(format!(
                    "findall/3 template variable '{}' is not bound by goal",
                    name
                ))
            })?,
            Term::Anonymous => {
                return Err(meta_error(
                    "findall/3 template cannot be the anonymous wildcard",
                ))
            }
            Term::List(items) => Term::List(
                items
                    .iter()
                    .map(|item| self.substitute_template(item, bindings, elem_type))
                    .collect::<Result<Vec<_>>>()?,
            ),
            Term::Compound { functor, args } => Term::Compound {
                functor: functor.clone(),
                args: args
                    .iter()
                    .map(|arg| self.substitute_template(arg, bindings, elem_type))
                    .collect::<Result<Vec<_>>>()?,
            },
            _ => template.clone(),
        };
        self.normalize_value_term(&value, Some(elem_type))
    }

    fn infer_template_type(&self, template: &Term, goal: &Atom) -> Result<TypeRef> {
        match template {
            Term::Variable(name) => {
                for (idx, term) in goal.terms.iter().enumerate() {
                    if matches!(term, Term::Variable(var) if var == name) {
                        return Ok(self
                            .pred_columns
                            .get(&(goal.predicate.clone(), idx))
                            .cloned()
                            .unwrap_or(TypeRef::Scalar(ScalarType::U64)));
                    }
                }
                Err(meta_error(format!(
                    "findall/3 template variable '{}' is not bound by the goal",
                    name
                )))
            }
            Term::Integer(_) => Ok(TypeRef::Scalar(ScalarType::U32)),
            Term::Float(_) => Ok(TypeRef::Scalar(ScalarType::F64)),
            Term::String(_) | Term::Symbol(_) => Ok(TypeRef::Scalar(ScalarType::Symbol)),
            Term::Compound { .. } | Term::PredRef(_) | Term::List(_) | Term::Cons { .. } => {
                Ok(TypeRef::Term)
            }
            Term::Anonymous | Term::Aggregate(_) => Err(meta_error(
                "findall/3 template must be a finite scalar or term expression",
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
            Term::Variable(name) if bound.contains_key(name) => Ok(term.clone()),
            Term::Variable(_) | Term::Anonymous => Err(meta_error(format!(
                "{context} requires a known finite term before runtime inspection"
            ))),
            _ => Ok(Term::Integer(
                self.register_term(term, ExpectedTerm::Any)? as i64
            )),
        }
    }

    fn normalize_univ_parts(&mut self, term: &Term) -> Result<Term> {
        match term {
            Term::Variable(_) | Term::Anonymous => Ok(term.clone()),
            Term::List(items) => Ok(Term::List(
                items
                    .iter()
                    .map(|item| {
                        Ok(Term::Integer(
                            self.register_term(item, ExpectedTerm::Any)? as i64
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?,
            )),
            _ => Err(meta_error(
                "univ parts must be a finite list literal or a list variable",
            )),
        }
    }

    fn normalize_symbol_argument(&self, term: &Term, context: &str) -> Result<Term> {
        match term {
            Term::Variable(_) | Term::Anonymous => Ok(term.clone()),
            Term::Symbol(_) | Term::String(_) => Ok(term.clone()),
            _ => Err(meta_error(format!(
                "{context} must be a symbol or variable"
            ))),
        }
    }

    fn normalize_u32_argument(&self, term: &Term, context: &str) -> Result<Term> {
        match term {
            Term::Variable(_) | Term::Anonymous => Ok(term.clone()),
            Term::Integer(value) if *value >= 0 && *value <= u32::MAX as i64 => Ok(term.clone()),
            _ => Err(meta_error(format!(
                "{context} must be a u32 integer or variable"
            ))),
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
                let typ = self
                    .pred_columns
                    .get(&(atom.predicate.clone(), idx))
                    .and_then(|typ| self.storage_type_for_type_ref(typ).ok())
                    .unwrap_or(ScalarType::U64);
                bound.insert(name.clone(), typ);
            }
        }
    }

    fn register_term(&mut self, term: &Term, expected: ExpectedTerm) -> Result<u64> {
        let (key, kind) = match term {
            Term::Integer(value) if expected != ExpectedTerm::PredRef => {
                (format!("i:{value}"), RegisteredTerm::Scalar)
            }
            Term::Float(value) if expected != ExpectedTerm::PredRef => {
                (format!("f:{}", value.to_bits()), RegisteredTerm::Scalar)
            }
            Term::String(value) if expected != ExpectedTerm::PredRef => {
                (format!("s:{value}"), RegisteredTerm::Scalar)
            }
            Term::Symbol(id) if expected == ExpectedTerm::PredRef => {
                let name = symbol::resolve(*id);
                (format!("predref:{name}"), RegisteredTerm::PredRef)
            }
            Term::String(name) if expected == ExpectedTerm::PredRef => {
                (format!("predref:{name}"), RegisteredTerm::PredRef)
            }
            Term::PredRef(name) => (format!("predref:{name}"), RegisteredTerm::PredRef),
            Term::Symbol(id) => {
                let name = symbol::resolve(*id);
                (format!("sym:{name}"), RegisteredTerm::Scalar)
            }
            Term::Compound { functor, args } if expected != ExpectedTerm::PredRef => {
                let mut arg_ids = Vec::with_capacity(args.len());
                for arg in args {
                    arg_ids.push(self.register_term(arg, ExpectedTerm::Any)?);
                }
                let functor_id =
                    self.register_term(&Term::Symbol(symbol::intern(functor)), ExpectedTerm::Any)?;
                let mut parts = vec![functor_id];
                parts.extend(arg_ids.iter().copied());
                (
                    format!("compound:{functor}({})", join_ids(&arg_ids)),
                    RegisteredTerm::Compound {
                        functor: functor.clone(),
                        args: arg_ids,
                        parts,
                    },
                )
            }
            Term::List(_) | Term::Cons { .. } => {
                return Err(meta_error(
                    "finite list terms inside term values are reserved for later G085 nodes",
                ))
            }
            Term::Variable(_) | Term::Anonymous => {
                return Err(meta_error(
                    "finite term registration requires a ground source term",
                ))
            }
            Term::Aggregate(_) => {
                return Err(meta_error(
                    "aggregate terms cannot be registered as finite meta terms",
                ))
            }
            Term::Compound { .. } => {
                return Err(meta_error(
                    "predref columns require a static predicate reference, not a compound term",
                ))
            }
            Term::Integer(_) | Term::Float(_) | Term::String(_) => {
                return Err(meta_error(
                    "predref columns require a static predicate reference",
                ))
            }
        };

        if expected == ExpectedTerm::Compound && !matches!(kind, RegisteredTerm::Compound { .. }) {
            return Err(meta_error(
                "compound column requires a finite compound term",
            ));
        }
        if expected == ExpectedTerm::PredRef && !matches!(kind, RegisteredTerm::PredRef) {
            return Err(meta_error(
                "predref column requires a static predicate reference",
            ));
        }

        if let Some(id) = self.term_ids.get(&key) {
            return Ok(*id);
        }
        let id = self.next_term_id;
        self.next_term_id += 1;
        self.term_ids.insert(key, id);
        self.term_records.push(TermRecord { id, kind });
        Ok(id)
    }

    fn append_helpers(&mut self, program: &mut Program) {
        self.append_term_helper_facts();

        let existing: HashSet<String> = program
            .predicates
            .iter()
            .map(|pred| pred.name.clone())
            .collect();
        for (name, columns) in &self.helper_decls {
            if existing.contains(name) {
                continue;
            }
            program.predicates.push(PredDecl {
                name: name.clone(),
                types: columns.iter().map(|col| col.typ.clone()).collect(),
                columns: columns.clone(),
                is_private: true,
            });
        }
        program.rules.extend(self.helper_facts.clone());
    }

    fn append_term_helper_facts(&mut self) {
        self.register_helper_decl(
            functor_pred(),
            vec![
                scalar_col("term_id", TERM_ID_TYPE),
                scalar_col("name", ScalarType::Symbol),
                scalar_col("arity", ScalarType::U32),
            ],
        );
        self.register_helper_decl(
            univ_pred(),
            vec![
                scalar_col("term_id", TERM_ID_TYPE),
                PredColumn {
                    name: Some("parts".to_string()),
                    typ: TypeRef::List(Box::new(TypeRef::Term)),
                },
            ],
        );
        self.register_helper_decl(
            arg_pred(),
            vec![
                scalar_col("term_id", TERM_ID_TYPE),
                scalar_col("idx", ScalarType::U32),
                scalar_col("arg_id", TERM_ID_TYPE),
            ],
        );

        for record in self.term_records.clone() {
            if let RegisteredTerm::Compound {
                functor,
                args,
                parts,
            } = record.kind
            {
                self.add_helper_fact(
                    functor_pred(),
                    vec![
                        Term::Integer(record.id as i64),
                        Term::Symbol(symbol::intern(&functor)),
                        Term::Integer(args.len() as i64),
                    ],
                );
                self.add_helper_fact(
                    univ_pred(),
                    vec![
                        Term::Integer(record.id as i64),
                        Term::List(
                            parts
                                .into_iter()
                                .map(|id| Term::Integer(id as i64))
                                .collect(),
                        ),
                    ],
                );
                for (idx, arg_id) in args.into_iter().enumerate() {
                    self.add_helper_fact(
                        arg_pred(),
                        vec![
                            Term::Integer(record.id as i64),
                            Term::Integer(idx as i64),
                            Term::Integer(arg_id as i64),
                        ],
                    );
                }
            }
        }
    }

    fn register_helper_decl(&mut self, name: &str, columns: Vec<PredColumn>) {
        self.helper_decls.entry(name.to_string()).or_insert(columns);
    }

    fn add_helper_fact(&mut self, pred: &str, terms: Vec<Term>) {
        let key = format!("{pred}:{:?}", terms);
        if !self.helper_fact_keys.insert(key) {
            return;
        }
        self.helper_facts.push(Rule {
            head: Atom {
                predicate: pred.to_string(),
                terms,
            },
            body: vec![],
        });
    }

    fn fail_atom(&mut self) -> Atom {
        self.register_helper_decl(fail_pred(), vec![scalar_col("flag", ScalarType::U32)]);
        Atom {
            predicate: fail_pred().to_string(),
            terms: vec![Term::Integer(1)],
        }
    }

    fn storage_type_for_type_ref(&self, typ: &TypeRef) -> Result<ScalarType> {
        match typ {
            TypeRef::Scalar(ty) => Ok(*ty),
            TypeRef::Domain(name) => self
                .domains
                .get(name)
                .copied()
                .ok_or_else(|| meta_error(format!("unknown domain alias '{name}'"))),
            TypeRef::List(_) | TypeRef::Term | TypeRef::Compound | TypeRef::PredRef => {
                Ok(TERM_ID_TYPE)
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

fn require_arity(atom: &Atom, expected: usize) -> Result<()> {
    if atom.terms.len() == expected {
        Ok(())
    } else {
        Err(meta_error(format!(
            "{} expects {} arguments, got {}",
            atom.predicate,
            expected,
            atom.terms.len()
        )))
    }
}

fn compound_goal_atom(term: &Term, context: &str) -> Result<Atom> {
    match term {
        Term::Compound { functor, args } => Ok(Atom {
            predicate: functor.clone(),
            terms: args.clone(),
        }),
        _ => Err(meta_error(format!(
            "{context} goal must be a finite atom expression"
        ))),
    }
}

fn finite_list_items<'a>(term: &'a Term, context: &str) -> Result<&'a [Term]> {
    match term {
        Term::List(items) => Ok(items),
        _ => Err(meta_error(format!(
            "{context} requires a finite list literal in G085_META"
        ))),
    }
}

fn static_pred_name(term: &Term) -> Result<String> {
    match term {
        Term::Symbol(id) => Ok(symbol::resolve(*id)),
        Term::String(name) | Term::PredRef(name) => Ok(name.clone()),
        Term::Variable(_) | Term::Anonymous => Err(meta_error(
            "maplist requires a static predicate reference; runtime-variable predicate names are rejected",
        )),
        _ => Err(meta_error(
            "maplist predicate argument must be a static predicate reference",
        )),
    }
}

fn infer_type_ref_from_terms(items: &[Term]) -> TypeRef {
    if let Some(first) = items.first() {
        match first {
            Term::Integer(value) if *value >= 0 && *value <= u32::MAX as i64 => {
                TypeRef::Scalar(ScalarType::U32)
            }
            Term::Integer(_) => TypeRef::Scalar(ScalarType::I64),
            Term::Float(_) => TypeRef::Scalar(ScalarType::F64),
            Term::String(_) | Term::Symbol(_) => TypeRef::Scalar(ScalarType::Symbol),
            Term::Compound { .. } | Term::PredRef(_) | Term::List(_) | Term::Cons { .. } => {
                TypeRef::Term
            }
            Term::Variable(_) | Term::Anonymous | Term::Aggregate(_) => {
                TypeRef::Scalar(ScalarType::U64)
            }
        }
    } else {
        TypeRef::Scalar(ScalarType::U64)
    }
}

fn cartesian_product(choices: &[Vec<Term>]) -> Vec<Vec<Term>> {
    let mut rows: Vec<Vec<Term>> = vec![Vec::new()];
    for choice in choices {
        let mut next = Vec::new();
        for row in &rows {
            for value in choice {
                let mut row = row.clone();
                row.push(value.clone());
                next.push(row);
            }
        }
        rows = next;
    }
    rows
}

fn term_match_key(term: &Term) -> String {
    match term {
        Term::Variable(name) => format!("var:{name}"),
        Term::Anonymous => "_".to_string(),
        Term::Integer(value) => format!("i:{value}"),
        Term::Float(value) => format!("f:{}", value.to_bits()),
        Term::String(value) => format!("s:{value}"),
        Term::Symbol(id) => format!("sym:{}", symbol::resolve(*id)),
        Term::List(items) => format!(
            "list:[{}]",
            items
                .iter()
                .map(term_match_key)
                .collect::<Vec<_>>()
                .join(",")
        ),
        Term::Cons { head, tail } => {
            format!("cons:{}|{}", term_match_key(head), term_match_key(tail))
        }
        Term::Compound { functor, args } => format!(
            "compound:{functor}({})",
            args.iter()
                .map(term_match_key)
                .collect::<Vec<_>>()
                .join(",")
        ),
        Term::PredRef(name) => format!("predref:{name}"),
        Term::Aggregate(agg) => format!("agg:{:?}:{}", agg.op, agg.variable),
    }
}

fn scalar_col(name: &str, typ: ScalarType) -> PredColumn {
    PredColumn {
        name: Some(name.to_string()),
        typ: TypeRef::Scalar(typ),
    }
}

fn lower_meta_type_refs_in_predicates(program: &mut Program) {
    for pred in &mut program.predicates {
        for typ in &mut pred.types {
            *typ = lower_meta_type_ref(typ);
        }
        for col in &mut pred.columns {
            col.typ = lower_meta_type_ref(&col.typ);
        }
    }
}

fn lower_meta_type_ref(typ: &TypeRef) -> TypeRef {
    match typ {
        TypeRef::List(inner) => TypeRef::List(Box::new(lower_meta_type_ref(inner))),
        TypeRef::Term | TypeRef::Compound | TypeRef::PredRef => TypeRef::Scalar(TERM_ID_TYPE),
        TypeRef::Scalar(_) | TypeRef::Domain(_) => typ.clone(),
    }
}

fn join_ids(ids: &[u64]) -> String {
    ids.iter().map(u64::to_string).collect::<Vec<_>>().join(",")
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

fn dynamic_predicate_error(name: &str) -> XlogError {
    match name {
        "call" => meta_error("dynamic call/N is outside the v0.8.5 safe meta subset"),
        "assert" | "asserta" | "assertz" | "retract" => {
            meta_error("dynamic database mutation is outside the v0.8.5 safe meta subset")
        }
        _ => meta_error("unsupported dynamic meta predicate"),
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

fn meta_error(message: impl Into<String>) -> XlogError {
    XlogError::Compilation(format!("v0.8.5 meta error: {}", message.into()))
}
