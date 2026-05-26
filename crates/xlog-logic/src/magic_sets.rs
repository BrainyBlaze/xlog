//! Magic-set rewriting for the v0.8.5 deterministic language subset.

use std::collections::{BTreeSet, HashMap, HashSet};

use xlog_core::{Result, XlogError};

use crate::ast::{Atom, BodyLiteral, MagicSetsMode, Program, Rule, Term};

/// Status of a magic-set rewrite attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MagicSetStatus {
    /// Rewriting was disabled by source configuration or there was no directive.
    Disabled,
    /// A supported bound recursive query was rewritten.
    Applied,
    /// `auto` mode found an unsafe or inapplicable program and left it unchanged.
    Declined,
}

/// Human- and test-readable metadata for a magic-set rewrite attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MagicSetReport {
    /// Final status.
    pub status: MagicSetStatus,
    /// Generated magic predicate names.
    pub generated_predicates: Vec<String>,
    /// Adorned recursive predicates, formatted as `predicate/adornment`.
    pub adorned_predicates: Vec<String>,
    /// Reasons the rewrite declined.
    pub declined_reasons: Vec<String>,
}

/// Rewritten program plus its report.
#[derive(Debug, Clone)]
pub struct MagicSetRewrite {
    /// Program after rewriting, or the original program when disabled/declined.
    pub program: Program,
    /// Rewrite metadata.
    pub report: MagicSetReport,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Adornment {
    pred: String,
    pattern: Vec<bool>,
}

#[derive(Debug, Clone)]
struct Seed {
    pred: String,
    pattern: Vec<bool>,
    terms: Vec<Term>,
}

/// Rewrite supported v0.8.5 bound recursive queries with magic predicates.
pub fn rewrite_v085_magic_sets(program: &Program) -> Result<MagicSetRewrite> {
    let mode = program.directives.magic_sets;
    if mode.is_none() || mode == Some(MagicSetsMode::Off) {
        return Ok(with_status(program, MagicSetStatus::Disabled));
    }
    let mode = mode.expect("checked above");

    let recursive = recursive_predicates(program);
    let seeds = collect_query_seeds(program, &recursive);
    if seeds.is_empty() {
        return decline_or_error(
            program,
            mode,
            vec!["no bound recursive query eligible for magic_sets".to_string()],
        );
    }

    if program.is_probabilistic_profile() {
        return decline_or_error(
            program,
            mode,
            vec!["probabilistic profiles are handled outside deterministic magic_sets".to_string()],
        );
    }

    let target_preds: BTreeSet<String> = seeds.iter().map(|seed| seed.pred.clone()).collect();
    let unsafe_reasons = unsupported_target_reasons(program, &target_preds, &recursive);
    if !unsafe_reasons.is_empty() {
        return decline_or_error(program, mode, unsafe_reasons);
    }

    let mut adornments = initial_adornments(&seeds);
    expand_adornments(program, &target_preds, &mut adornments)?;

    let mut generated_predicates: BTreeSet<String> = BTreeSet::new();
    let mut adorned_predicates: BTreeSet<String> = BTreeSet::new();
    for adornment in &adornments {
        generated_predicates.insert(magic_predicate(&adornment.pred, &adornment.pattern));
        adorned_predicates.insert(format!(
            "{}/{}",
            adornment.pred,
            adornment_key(&adornment.pattern)
        ));
    }

    let mut rewritten = program.clone();
    rewritten.rules = rewrite_rules(program, &target_preds, &adornments, &seeds)?;

    Ok(MagicSetRewrite {
        program: rewritten,
        report: MagicSetReport {
            status: MagicSetStatus::Applied,
            generated_predicates: generated_predicates.into_iter().collect(),
            adorned_predicates: adorned_predicates.into_iter().collect(),
            declined_reasons: Vec::new(),
        },
    })
}

fn with_status(program: &Program, status: MagicSetStatus) -> MagicSetRewrite {
    MagicSetRewrite {
        program: program.clone(),
        report: MagicSetReport {
            status,
            generated_predicates: Vec::new(),
            adorned_predicates: Vec::new(),
            declined_reasons: Vec::new(),
        },
    }
}

fn decline_or_error(
    program: &Program,
    mode: MagicSetsMode,
    reasons: Vec<String>,
) -> Result<MagicSetRewrite> {
    if mode == MagicSetsMode::On {
        return Err(magic_error(reasons.join("; ")));
    }
    Ok(MagicSetRewrite {
        program: program.clone(),
        report: MagicSetReport {
            status: MagicSetStatus::Declined,
            generated_predicates: Vec::new(),
            adorned_predicates: Vec::new(),
            declined_reasons: reasons,
        },
    })
}

fn magic_error(message: impl Into<String>) -> XlogError {
    XlogError::Compilation(format!("v0.8.5 magic_sets error: {}", message.into()))
}

fn collect_query_seeds(program: &Program, recursive: &HashSet<String>) -> Vec<Seed> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for query in &program.queries {
        if !recursive.contains(&query.atom.predicate) {
            continue;
        }
        let pattern: Vec<bool> = query.atom.terms.iter().map(is_seed_term).collect();
        if !pattern.iter().any(|bound| *bound) {
            continue;
        }
        if !query
            .atom
            .terms
            .iter()
            .zip(&pattern)
            .all(|(term, bound)| !*bound || is_supported_magic_term(term))
        {
            continue;
        }
        let terms = bound_terms(&query.atom, &pattern);
        let key = format!(
            "{}:{}:{:?}",
            query.atom.predicate,
            adornment_key(&pattern),
            terms
        );
        if seen.insert(key) {
            out.push(Seed {
                pred: query.atom.predicate.clone(),
                pattern,
                terms,
            });
        }
    }
    out
}

fn initial_adornments(seeds: &[Seed]) -> BTreeSet<Adornment> {
    seeds
        .iter()
        .map(|seed| Adornment {
            pred: seed.pred.clone(),
            pattern: seed.pattern.clone(),
        })
        .collect()
}

fn expand_adornments(
    program: &Program,
    target_preds: &BTreeSet<String>,
    adornments: &mut BTreeSet<Adornment>,
) -> Result<()> {
    let mut changed = true;
    while changed {
        changed = false;
        let snapshot: Vec<Adornment> = adornments.iter().cloned().collect();
        for adornment in snapshot {
            for rule in program
                .rules
                .iter()
                .filter(|rule| rule.head.predicate == adornment.pred)
            {
                for discovered in discover_body_adornments(rule, &adornment.pattern, target_preds)?
                {
                    changed |= adornments.insert(discovered);
                }
            }
        }
    }
    Ok(())
}

fn discover_body_adornments(
    rule: &Rule,
    head_pattern: &[bool],
    target_preds: &BTreeSet<String>,
) -> Result<Vec<Adornment>> {
    let mut bound = head_bound_variables(&rule.head, head_pattern);
    let mut out = Vec::new();
    for lit in &rule.body {
        match lit {
            BodyLiteral::Positive(atom) => {
                if target_preds.contains(&atom.predicate) {
                    let pattern = atom_adornment(atom, &bound);
                    if !pattern.iter().any(|is_bound| *is_bound) {
                        return Err(magic_error(format!(
                            "recursive call {}/{} has no bound argument under supported SIPS",
                            atom.predicate,
                            atom.arity()
                        )));
                    }
                    out.push(Adornment {
                        pred: atom.predicate.clone(),
                        pattern,
                    });
                }
                bind_atom_variables(atom, &mut bound);
            }
            BodyLiteral::Comparison(_)
            | BodyLiteral::Epistemic(_)
            | BodyLiteral::IsExpr(_)
            | BodyLiteral::Negated(_)
            | BodyLiteral::Univ(_) => {}
        }
    }
    Ok(out)
}

fn rewrite_rules(
    program: &Program,
    target_preds: &BTreeSet<String>,
    adornments: &BTreeSet<Adornment>,
    seeds: &[Seed],
) -> Result<Vec<Rule>> {
    let mut out: Vec<Rule> = program
        .rules
        .iter()
        .filter(|rule| !target_preds.contains(&rule.head.predicate))
        .cloned()
        .collect();

    let mut emitted = HashSet::new();
    for seed in seeds {
        let rule = Rule {
            head: Atom {
                predicate: magic_predicate(&seed.pred, &seed.pattern),
                terms: seed.terms.clone(),
            },
            body: Vec::new(),
        };
        push_unique_rule(&mut out, &mut emitted, rule);
    }

    for adornment in adornments {
        for rule in program
            .rules
            .iter()
            .filter(|rule| rule.head.predicate == adornment.pred)
        {
            for magic_rule in propagation_rules(rule, &adornment.pattern, target_preds)? {
                push_unique_rule(&mut out, &mut emitted, magic_rule);
            }
        }
    }

    for adornment in adornments {
        for rule in program
            .rules
            .iter()
            .filter(|rule| rule.head.predicate == adornment.pred)
        {
            let mut body = vec![BodyLiteral::Positive(magic_atom_for(
                &rule.head,
                &adornment.pattern,
            ))];
            body.extend(rule.body.clone());
            out.push(Rule {
                head: rule.head.clone(),
                body,
            });
        }
    }

    Ok(out)
}

fn propagation_rules(
    rule: &Rule,
    head_pattern: &[bool],
    target_preds: &BTreeSet<String>,
) -> Result<Vec<Rule>> {
    let caller_magic = magic_atom_for(&rule.head, head_pattern);
    let mut prefix = vec![BodyLiteral::Positive(caller_magic.clone())];
    let mut bound = head_bound_variables(&rule.head, head_pattern);
    let mut out = Vec::new();

    for lit in &rule.body {
        let BodyLiteral::Positive(atom) = lit else {
            continue;
        };
        if target_preds.contains(&atom.predicate) {
            let pattern = atom_adornment(atom, &bound);
            if !pattern.iter().any(|is_bound| *is_bound) {
                return Err(magic_error(format!(
                    "recursive call {}/{} has no bound argument under supported SIPS",
                    atom.predicate,
                    atom.arity()
                )));
            }
            let head = magic_atom_for(atom, &pattern);
            let is_trivial = prefix.len() == 1
                && matches!(&prefix[0], BodyLiteral::Positive(prefix_atom) if *prefix_atom == head);
            if !is_trivial {
                out.push(Rule {
                    head,
                    body: prefix.clone(),
                });
            }
        }
        bind_atom_variables(atom, &mut bound);
        prefix.push(lit.clone());
    }

    Ok(out)
}

fn unsupported_target_reasons(
    program: &Program,
    target_preds: &BTreeSet<String>,
    recursive: &HashSet<String>,
) -> Vec<String> {
    let mut reasons = BTreeSet::new();
    for rule in &program.rules {
        if !target_preds.contains(&rule.head.predicate) {
            continue;
        }
        if rule.has_negation() {
            reasons.insert(format!(
                "negation in recursive rule for {} is outside the supported magic_sets subset",
                rule.head.predicate
            ));
        }
        if rule.has_aggregation() || rule.body.iter().any(body_literal_has_aggregate) {
            reasons.insert(format!(
                "aggregation in recursive rule for {} is outside the supported magic_sets subset",
                rule.head.predicate
            ));
        }
        for lit in &rule.body {
            match lit {
                BodyLiteral::Positive(atom) => {
                    if recursive.contains(&atom.predicate) && atom.predicate != rule.head.predicate
                    {
                        reasons.insert(format!(
                            "mutual recursion through {} is outside the supported magic_sets subset",
                            atom.predicate
                        ));
                    }
                    if atom.predicate.starts_with("__xlog_meta_")
                        || atom.predicate.starts_with("__xlog_list_")
                    {
                        reasons.insert(format!(
                            "meta/list helper {} in recursive rule is outside the supported magic_sets subset",
                            atom.predicate
                        ));
                    }
                }
                BodyLiteral::Negated(_) => {}
                BodyLiteral::Comparison(_)
                | BodyLiteral::Epistemic(_)
                | BodyLiteral::IsExpr(_)
                | BodyLiteral::Univ(_) => {
                    reasons.insert(format!(
                        "non-positive literal in recursive rule for {} is outside the supported magic_sets subset",
                        rule.head.predicate
                    ));
                }
            }
        }
    }
    reasons.into_iter().collect()
}

fn recursive_predicates(program: &Program) -> HashSet<String> {
    let mut deps: HashMap<String, HashSet<String>> = HashMap::new();
    for rule in &program.rules {
        let entry = deps.entry(rule.head.predicate.clone()).or_default();
        for pred in rule.body_predicates() {
            entry.insert(pred.to_string());
        }
    }
    deps.keys()
        .filter(|pred| reaches(pred, pred, &deps, &mut HashSet::new()))
        .cloned()
        .collect()
}

fn reaches(
    start: &str,
    target: &str,
    deps: &HashMap<String, HashSet<String>>,
    seen: &mut HashSet<String>,
) -> bool {
    let Some(next) = deps.get(start) else {
        return false;
    };
    for pred in next {
        if pred == target {
            return true;
        }
        if seen.insert(pred.clone()) && reaches(pred, target, deps, seen) {
            return true;
        }
    }
    false
}

fn head_bound_variables(atom: &Atom, pattern: &[bool]) -> HashSet<String> {
    atom.terms
        .iter()
        .zip(pattern)
        .filter(|(_, bound)| **bound)
        .flat_map(|(term, _)| term.variables().into_iter().map(str::to_string))
        .collect()
}

fn atom_adornment(atom: &Atom, bound: &HashSet<String>) -> Vec<bool> {
    atom.terms
        .iter()
        .map(|term| term_is_bound(term, bound))
        .collect()
}

fn term_is_bound(term: &Term, bound: &HashSet<String>) -> bool {
    match term {
        Term::Variable(name) => bound.contains(name),
        Term::Anonymous => false,
        Term::List(items) => items.iter().all(|item| term_is_bound(item, bound)),
        Term::Cons { head, tail } => term_is_bound(head, bound) && term_is_bound(tail, bound),
        Term::Compound { args, .. } => args.iter().all(|arg| term_is_bound(arg, bound)),
        Term::Integer(_)
        | Term::Float(_)
        | Term::String(_)
        | Term::Symbol(_)
        | Term::PredRef(_) => true,
        Term::Aggregate(_) => false,
    }
}

fn bind_atom_variables(atom: &Atom, bound: &mut HashSet<String>) {
    for name in atom.variables() {
        bound.insert(name.to_string());
    }
}

fn body_literal_has_aggregate(lit: &BodyLiteral) -> bool {
    match lit {
        BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => atom_has_aggregate(atom),
        BodyLiteral::Epistemic(lit) => atom_has_aggregate(&lit.atom),
        BodyLiteral::Comparison(comparison) => {
            term_has_aggregate(&comparison.left) || term_has_aggregate(&comparison.right)
        }
        BodyLiteral::IsExpr(_) => false,
        BodyLiteral::Univ(univ) => {
            term_has_aggregate(&univ.term) || term_has_aggregate(&univ.parts)
        }
    }
}

fn atom_has_aggregate(atom: &Atom) -> bool {
    atom.terms.iter().any(term_has_aggregate)
}

fn term_has_aggregate(term: &Term) -> bool {
    match term {
        Term::Aggregate(_) => true,
        Term::List(items) => items.iter().any(term_has_aggregate),
        Term::Cons { head, tail } => term_has_aggregate(head) || term_has_aggregate(tail),
        Term::Compound { args, .. } => args.iter().any(term_has_aggregate),
        Term::Variable(_)
        | Term::Anonymous
        | Term::Integer(_)
        | Term::Float(_)
        | Term::String(_)
        | Term::Symbol(_)
        | Term::PredRef(_) => false,
    }
}

fn is_seed_term(term: &Term) -> bool {
    is_supported_magic_term(term) && !term.is_any_variable()
}

fn is_supported_magic_term(term: &Term) -> bool {
    matches!(
        term,
        Term::Integer(_) | Term::Float(_) | Term::String(_) | Term::Symbol(_)
    )
}

fn bound_terms(atom: &Atom, pattern: &[bool]) -> Vec<Term> {
    atom.terms
        .iter()
        .zip(pattern)
        .filter(|(_, bound)| **bound)
        .map(|(term, _)| term.clone())
        .collect()
}

fn magic_atom_for(atom: &Atom, pattern: &[bool]) -> Atom {
    Atom {
        predicate: magic_predicate(&atom.predicate, pattern),
        terms: bound_terms(atom, pattern),
    }
}

fn magic_predicate(pred: &str, pattern: &[bool]) -> String {
    format!("__xlog_magic_{}_{}", pred, adornment_key(pattern))
}

fn adornment_key(pattern: &[bool]) -> String {
    pattern
        .iter()
        .map(|bound| if *bound { 'b' } else { 'f' })
        .collect()
}

fn push_unique_rule(out: &mut Vec<Rule>, emitted: &mut HashSet<String>, rule: Rule) {
    let key = format!("{:?}", rule);
    if emitted.insert(key) {
        out.push(rule);
    }
}
