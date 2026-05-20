use std::path::Path;

use xlog_logic::{ParserSession, Term};

fn facts(count: usize) -> String {
    let mut source = String::new();
    for idx in 0..count {
        source.push_str(&format!("edge({}, {}).\n", idx, idx + 1));
    }
    source
}

#[test]
fn splits_statement_units_with_stable_spans() {
    let source =
        "\n// comment\n#pragma prob_engine = mc\n0.5::edge(1, 2).\nreach(X, Y) :- edge(X, Y).\n";
    let units = ParserSession::split_statements(source);
    assert_eq!(units.len(), 3);
    assert_eq!(units[0].text, "#pragma prob_engine = mc");
    assert_eq!(units[0].span.line, 3);
    assert_eq!(units[0].span.column, 1);
    assert_eq!(units[1].text, "0.5::edge(1, 2).");
    assert!(units[1].span.start < units[1].span.end);
}

#[test]
fn reuses_unchanged_statement_parses_after_single_edit() {
    let mut session = ParserSession::new();
    let path = Path::new("synthetic/main.xlog");
    let original = facts(500);
    let first = session
        .parse_path(path, &original)
        .expect("initial incremental parse");
    assert_eq!(first.stats.hits, 0);
    assert_eq!(first.stats.misses, 500);

    let edited = original.replace("edge(250, 251).", "edge(250, 999).");
    let second = session
        .parse_path(path, &edited)
        .expect("second incremental parse");
    assert_eq!(second.stats.statement_count, 500);
    assert_eq!(second.stats.hits, 499);
    assert_eq!(second.stats.misses, 1);
    assert_eq!(second.stats.invalidated, 1);
    assert!(second.stats.estimated_speedup() >= 500.0);

    let edited_fact = second
        .program
        .facts()
        .find(|rule| {
            matches!(
                rule.head.terms.as_slice(),
                [Term::Integer(250), Term::Integer(999)]
            )
        })
        .expect("edited fact present");
    assert_eq!(edited_fact.head.predicate, "edge");
}

#[test]
fn invalidates_module_and_sources_that_import_it() {
    let mut session = ParserSession::new();
    let module = Path::new("modules/graph.xlog");
    let main = Path::new("main.xlog");

    session
        .parse_path(module, "edge(1, 2).\n")
        .expect("parse module");
    session
        .parse_path(main, "use graph.\nreach(X, Y) :- edge(X, Y).\n")
        .expect("parse importer");
    assert_eq!(session.cached_source_count(), 2);

    let removed = session.invalidate_module(module);
    assert_eq!(removed, 2);
    assert_eq!(session.cached_source_count(), 0);

    let parsed = session
        .parse_path(main, "use graph.\nreach(X, Y) :- edge(X, Y).\n")
        .expect("reparse importer");
    assert_eq!(parsed.stats.module_invalidations, 2);
    assert_eq!(parsed.stats.hits, 0);
}

#[test]
fn parse_errors_report_original_statement_span() {
    let mut session = ParserSession::new();
    let err = session
        .parse_path("bad.xlog", "\nedge(1, 2).\nbad(.\n")
        .expect_err("bad statement should fail")
        .to_string();
    assert!(err.contains("incremental parse error"), "err={err}");
    assert!(err.contains("3:1"), "err={err}");
    assert!(err.contains("bytes"), "err={err}");
}
