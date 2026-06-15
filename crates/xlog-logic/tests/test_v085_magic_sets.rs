use std::collections::{BTreeMap, BTreeSet, HashMap};

use xlog_logic::{
    parse_program, rewrite_magic_sets, Atom, BodyLiteral, Compiler, MagicSetStatus, MagicSetsMode,
    Program, Term,
};

#[test]
fn parses_magic_sets_pragmas() {
    let auto = parse_program("#pragma magic_sets = auto").expect("parse auto");
    assert_eq!(auto.directives.magic_sets, Some(MagicSetsMode::Auto));

    let on = parse_program("#pragma magic_sets = on").expect("parse on");
    assert_eq!(on.directives.magic_sets, Some(MagicSetsMode::On));

    let off = parse_program("#pragma magic_sets = off").expect("parse off");
    assert_eq!(off.directives.magic_sets, Some(MagicSetsMode::Off));
}

#[test]
fn rewrites_bound_recursive_query_to_magic_predicates() {
    let program = parse_program(
        r#"
        #pragma magic_sets = on
        edge(1, 2).
        edge(2, 3).
        edge(10, 11).
        edge(11, 12).

        reach(X, Y) :- edge(X, Y).
        reach(X, Z) :- reach(X, Y), edge(Y, Z).

        ?- reach(1, Y).
    "#,
    )
    .expect("parse");

    let rewritten = rewrite_magic_sets(&program).expect("rewrite");
    assert_eq!(rewritten.report.status, MagicSetStatus::Applied);
    assert_eq!(
        rewritten.report.generated_predicates,
        vec!["__xlog_magic_reach_bf".to_string()]
    );
    assert!(rewritten
        .program
        .rules
        .iter()
        .any(|rule| rule.head.predicate == "__xlog_magic_reach_bf"
            && rule.body.is_empty()
            && rule.head.terms == vec![Term::Integer(1)]));
    assert!(rewritten
        .program
        .rules
        .iter()
        .filter(|rule| rule.head.predicate == "reach")
        .all(|rule| matches!(
            rule.body.first(),
            Some(BodyLiteral::Positive(atom))
                if atom.predicate == "__xlog_magic_reach_bf"
        )));
}

#[test]
fn magic_rewrite_preserves_query_output_and_reduces_recursive_rows() {
    let program = parse_program(
        r#"
        #pragma magic_sets = on
        edge(1, 2).
        edge(2, 3).
        edge(10, 11).
        edge(11, 12).

        reach(X, Y) :- edge(X, Y).
        reach(X, Z) :- reach(X, Y), edge(Y, Z).

        ?- reach(1, Y).
    "#,
    )
    .expect("parse");

    let rewritten = rewrite_magic_sets(&program).expect("rewrite");
    let original_relations = evaluate_positive_i64(&program);
    let rewritten_relations = evaluate_positive_i64(&rewritten.program);

    let original_query = query_rows(&program.queries[0].atom, &original_relations);
    let rewritten_query = query_rows(&rewritten.program.queries[0].atom, &rewritten_relations);
    assert_eq!(rewritten_query, original_query);
    assert_eq!(original_query, rows([vec![1, 2], vec![1, 3]]));

    let original_reach_rows = original_relations
        .get("reach")
        .expect("original reach")
        .len();
    let rewritten_reach_rows = rewritten_relations
        .get("reach")
        .expect("rewritten reach")
        .len();
    assert_eq!(original_reach_rows, 6);
    assert_eq!(rewritten_reach_rows, 2);
    assert!(
        rewritten_reach_rows * 2 <= original_reach_rows,
        "expected at least 50% recursive-row reduction"
    );
}

#[test]
fn declines_unsafe_magic_interactions_with_typed_diagnostics() {
    let auto = parse_program(
        r#"
        #pragma magic_sets = auto
        edge(1, 2).
        stop(2).
        p(X) :- edge(X, Y), not stop(Y).
        p(X) :- p(Y), edge(Y, X).
        ?- p(1).
    "#,
    )
    .expect("parse auto");

    let rewritten = rewrite_magic_sets(&auto).expect("auto decline should not fail");
    assert_eq!(rewritten.report.status, MagicSetStatus::Declined);
    assert!(rewritten
        .report
        .declined_reasons
        .iter()
        .any(|reason| reason.contains("negation")));

    let on = r#"
        #pragma magic_sets = on
        edge(1, 2).
        stop(2).
        p(X) :- edge(X, Y), not stop(Y).
        p(X) :- p(Y), edge(Y, X).
        ?- p(1).
    "#;
    let err = Compiler::new()
        .compile(on)
        .expect_err("magic_sets=on should fail");
    assert!(
        err.to_string().contains("magic_sets error") && err.to_string().contains("negation"),
        "unexpected diagnostic: {err}"
    );
}

#[test]
fn compiler_uses_magic_rewrite_before_lowering() {
    let source = r#"
        #pragma magic_sets = on
        edge(1, 2).
        edge(2, 3).
        reach(X, Y) :- edge(X, Y).
        reach(X, Z) :- reach(X, Y), edge(Y, Z).
        ?- reach(1, Y).
    "#;
    let mut compiler = Compiler::new();
    compiler.compile(source).expect("magic program compiles");
    assert!(compiler
        .rel_ids()
        .keys()
        .any(|name| name == "__xlog_magic_reach_bf"));
}

#[test]
fn committed_magic_sets_example_compiles() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root");
    let path = "examples/language-completeness/magic_sets/reach_bound.xlog";
    let full_path = repo_root.join(path);
    let src = std::fs::read_to_string(&full_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", full_path.display()));
    Compiler::new()
        .compile(&src)
        .unwrap_or_else(|err| panic!("{path} failed to compile: {err}"));
}

type Row = Vec<i64>;
type Relations = BTreeMap<String, BTreeSet<Row>>;

fn rows<const N: usize>(items: [Row; N]) -> BTreeSet<Row> {
    items.into_iter().collect()
}

fn evaluate_positive_i64(program: &Program) -> Relations {
    let mut relations: Relations = BTreeMap::new();
    for rule in &program.rules {
        if rule.body.is_empty() {
            if let Some(row) = atom_ground_i64_row(&rule.head) {
                relations
                    .entry(rule.head.predicate.clone())
                    .or_default()
                    .insert(row);
            }
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for rule in &program.rules {
            if rule.body.is_empty()
                || rule
                    .body
                    .iter()
                    .any(|lit| !matches!(lit, BodyLiteral::Positive(_)))
            {
                continue;
            }

            let mut bindings = vec![HashMap::<String, i64>::new()];
            for lit in &rule.body {
                let BodyLiteral::Positive(atom) = lit else {
                    unreachable!();
                };
                let relation = relations.get(&atom.predicate).cloned().unwrap_or_default();
                let mut next = Vec::new();
                for binding in &bindings {
                    for row in &relation {
                        if let Some(merged) = match_atom_i64(atom, row, binding) {
                            next.push(merged);
                        }
                    }
                }
                bindings = next;
            }

            for binding in bindings {
                if let Some(row) = instantiate_atom_i64(&rule.head, &binding) {
                    changed |= relations
                        .entry(rule.head.predicate.clone())
                        .or_default()
                        .insert(row);
                }
            }
        }
    }

    relations
}

fn query_rows(atom: &Atom, relations: &Relations) -> BTreeSet<Row> {
    relations
        .get(&atom.predicate)
        .into_iter()
        .flat_map(|rows| rows.iter())
        .filter(|row| match_atom_i64(atom, row, &HashMap::new()).is_some())
        .cloned()
        .collect()
}

fn atom_ground_i64_row(atom: &Atom) -> Option<Row> {
    atom.terms
        .iter()
        .map(|term| match term {
            Term::Integer(value) => Some(*value),
            _ => None,
        })
        .collect()
}

fn instantiate_atom_i64(atom: &Atom, binding: &HashMap<String, i64>) -> Option<Row> {
    atom.terms
        .iter()
        .map(|term| match term {
            Term::Integer(value) => Some(*value),
            Term::Variable(name) => binding.get(name).copied(),
            _ => None,
        })
        .collect()
}

fn match_atom_i64(
    atom: &Atom,
    row: &[i64],
    binding: &HashMap<String, i64>,
) -> Option<HashMap<String, i64>> {
    if atom.terms.len() != row.len() {
        return None;
    }
    let mut out = binding.clone();
    for (term, value) in atom.terms.iter().zip(row) {
        match term {
            Term::Integer(expected) if expected == value => {}
            Term::Integer(_) => return None,
            Term::Variable(name) => match out.get(name) {
                Some(existing) if existing != value => return None,
                Some(_) => {}
                None => {
                    out.insert(name.clone(), *value);
                }
            },
            Term::Anonymous => {}
            _ => return None,
        }
    }
    Some(out)
}
