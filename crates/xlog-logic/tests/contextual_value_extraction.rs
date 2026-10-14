use std::fs;

use serde_json::{json, Value};
use tempfile::TempDir;
use xlog_logic::compile::load_modules;

fn extract(source: &str) -> Value {
    let fixture = TempDir::new().unwrap();
    fs::write(fixture.path().join("main.xlog"), source).unwrap();
    extract_at(fixture.path(), "main.xlog")
}

fn extract_at(root: &std::path::Path, entry: &str) -> Value {
    let extraction = load_modules(&root.join(entry), vec![root.to_path_buf()])
        .unwrap()
        .resolved_program_extraction(root)
        .unwrap();
    serde_json::to_value(extraction).unwrap()
}

fn occurrences<'a>(report: &'a Value, relation: &str, ownership: &str) -> Vec<&'a Value> {
    report["native_values"]["occurrences"]
        .as_array()
        .expect("native contextual occurrences must be present")
        .iter()
        .filter(|term| term["context"]["relation_id"] == relation && term["ownership"] == ownership)
        .collect()
}

#[test]
fn scalar_symbol_aliases_preserve_authored_identity_and_conflicting_reference_values() {
    let report = extract(
        "pred source_ref(symbol, symbol).\nsource_ref(ref_a, reviewed).\nsource_ref(\"ref_a\", foreign).\n",
    );
    let terms = occurrences(&report, "relation:source_ref/2", "retained");
    assert_eq!(terms.len(), 4);
    assert_ne!(terms[0]["authored_term"], terms[2]["authored_term"]);
    assert_eq!(terms[0]["outcome"], terms[2]["outcome"]);
    assert_ne!(terms[1]["outcome"], terms[3]["outcome"]);
    assert_eq!(terms[0]["scalar_type"], "symbol");
    assert_eq!(terms[0]["scope"], "declared_scalar_carrier");
    assert_eq!(
        report["native_values"]["normalization"]["status"],
        "accepted"
    );
    assert_eq!(
        report["native_values"]["schema_validation"]["status"],
        "accepted"
    );
    assert_eq!(
        report["native_values"]["ordinary_rir"]["status"],
        "accepted"
    );
}

#[test]
fn native_numeric_values_preserve_conversion_width_rounding_and_signed_zero() {
    let report = extract(concat!(
        "pred u(u32). pred big(u64). pred small(i32). pred wide(i64). ",
        "pred f(f32). pred d(f64). pred b(bool).\n",
        "u(4294967295). big(9223372036854775807). small(-2147483648). ",
        "wide(-9223372036854775808). f(16777217). f(16777216.0). ",
        "f(-0.0). f(0.0). d(1). d(1.0). b(1). b(true). b(0). b(false).\n",
    ));
    let value = |relation: &str, index: usize| {
        occurrences(&report, relation, "retained")[index]["outcome"]["value"].clone()
    };
    assert_eq!(
        value("relation:u/1", 0),
        json!({"kind":"u32", "value":"4294967295"})
    );
    assert_eq!(
        value("relation:big/1", 0),
        json!({"kind":"u64", "value":"9223372036854775807"})
    );
    assert_eq!(
        value("relation:small/1", 0),
        json!({"kind":"i32", "value":"-2147483648"})
    );
    assert_eq!(
        value("relation:wide/1", 0),
        json!({"kind":"i64", "value":"-9223372036854775808"})
    );
    assert_eq!(value("relation:f/1", 0), value("relation:f/1", 1));
    assert_eq!(
        value("relation:f/1", 0),
        json!({"kind":"f32", "bits":"4b800000"})
    );
    assert_eq!(
        value("relation:f/1", 2),
        json!({"kind":"f32", "bits":"80000000"})
    );
    assert_eq!(
        value("relation:f/1", 3),
        json!({"kind":"f32", "bits":"00000000"})
    );
    assert_eq!(value("relation:d/1", 0), value("relation:d/1", 1));
    assert_eq!(value("relation:b/1", 0), value("relation:b/1", 1));
    assert_eq!(value("relation:b/1", 2), value("relation:b/1", 3));
    assert_ne!(value("relation:b/1", 0), value("relation:b/1", 2));
}

#[test]
fn nominal_joins_and_anonymous_occurrences_keep_scalar_context() {
    let report = extract(concat!(
        "domain person: symbol. domain event: symbol.\n",
        "pred p(person). pred e(event). pred joined(person).\n",
        "p(x). e(x). joined(X) :- p(X), e(X), e(_).\n",
    ));
    let terms = occurrences(&report, "relation:e/1", "retained");
    assert_eq!(
        terms.last().unwrap()["outcome"],
        json!({"kind":"anonymous"})
    );
    assert_eq!(terms.last().unwrap()["scalar_type"], "symbol");
    assert_eq!(
        terms.last().unwrap()["context"]["type_reference"],
        json!({"kind":"domain","name":"event"})
    );
    assert_eq!(
        report["native_values"]["schema_validation"]["status"],
        "accepted"
    );
    assert_eq!(
        report["native_values"]["ordinary_rir"]["status"],
        "accepted"
    );
    let bad = extract("pred p(symbol). pred n(i64). pred q(symbol). q(X) :- p(X), n(X).\n");
    assert_eq!(
        bad["native_values"]["schema_validation"]["status"],
        "rejected"
    );
    assert!(bad["native_values"]["schema_validation"]["reason"]
        .as_str()
        .unwrap()
        .contains("Type mismatch"));
}

#[test]
fn coalesced_authored_declarations_keep_their_own_nominal_context() {
    let fixture = TempDir::new().unwrap();
    fs::write(
        fixture.path().join("nominal.xlog"),
        "domain label: symbol. pred shared(author_role: label). shared(same).\n",
    )
    .unwrap();
    fs::write(
        fixture.path().join("scalar.xlog"),
        "pred shared(author_role: symbol). shared(same).\n",
    )
    .unwrap();
    fs::write(
        fixture.path().join("main.xlog"),
        "use nominal. use scalar.\n",
    )
    .unwrap();
    let report = extract_at(fixture.path(), "main.xlog");
    let authored = occurrences(&report, "relation:shared/1", "authored");
    assert_eq!(authored.len(), 2);
    let contexts: Vec<_> = authored
        .iter()
        .map(|term| term["context"]["authored_type_reference"].clone())
        .collect();
    assert!(contexts.contains(&json!({"kind":"domain","name":"label"})));
    assert!(contexts.contains(&json!({"kind":"scalar","scalar_type":"symbol"})));
    let roles: Vec<_> = authored
        .iter()
        .map(|term| term["context"]["authored_column_name"].clone())
        .collect();
    assert_eq!(roles, vec![json!("author_role"), json!("author_role")]);
    assert!(authored
        .iter()
        .all(|term| term["context"]["column_name"] == "author_role"));
    assert_eq!(authored[0]["outcome"], authored[1]["outcome"]);
}

#[test]
fn coalesced_signed_zero_keeps_each_authored_value_and_the_selected_value() {
    for (first, second) in [("0.0", "-0.0"), ("-0.0", "0.0")] {
        let fixture = TempDir::new().unwrap();
        fs::write(
            fixture.path().join("left.xlog"),
            format!("pred p(f32). p({first}).\n"),
        )
        .unwrap();
        fs::write(
            fixture.path().join("right.xlog"),
            format!("pred p(f32). p({second}).\n"),
        )
        .unwrap();
        fs::write(fixture.path().join("main.xlog"), "use left. use right.\n").unwrap();
        let report = extract_at(fixture.path(), "main.xlog");
        let authored = occurrences(&report, "relation:p/1", "authored");
        let retained = occurrences(&report, "relation:p/1", "retained");
        assert_eq!(authored.len(), 2);
        assert_eq!(retained.len(), 1);
        assert_ne!(authored[0]["outcome"], authored[1]["outcome"]);
        assert_eq!(
            authored[0]["executable_source_object_id"],
            authored[1]["executable_source_object_id"]
        );
        let expected = if first.starts_with('-') {
            "80000000"
        } else {
            "00000000"
        };
        assert_eq!(retained[0]["outcome"]["value"]["bits"], expected);
    }
}

#[test]
fn extraction_value_context_is_relocatable_and_entrypoint_specific() {
    let source = "pred p(symbol). p(\"x\").\n";
    assert_eq!(extract(source), extract(source));
    let fixture = TempDir::new().unwrap();
    fs::write(fixture.path().join("symbol.xlog"), source).unwrap();
    fs::write(fixture.path().join("numeric.xlog"), "pred p(i64). p(1).\n").unwrap();
    let symbols = extract_at(fixture.path(), "symbol.xlog");
    let numbers = extract_at(fixture.path(), "numeric.xlog");
    assert_ne!(
        symbols["native_values"]["entry_module_id"],
        numbers["native_values"]["entry_module_id"]
    );
    assert_eq!(
        occurrences(&symbols, "relation:p/1", "retained")[0]["scalar_type"],
        "symbol"
    );
    assert_eq!(
        occurrences(&numbers, "relation:p/1", "retained")[0]["scalar_type"],
        "i64"
    );
}

#[test]
fn comparison_constants_use_the_bound_carrier_and_constant_only_expression_types() {
    let report = extract(concat!(
        "pred p(f32). pred q(f32). p(16777216).\n",
        "q(X) :- p(X), X = 16777217, 16777217 = X, 1 = 1.0.\n",
    ));
    let terms: Vec<_> = report["native_values"]["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|term| term["ownership"] == "retained" && term["path"]["kind"] == "comparison")
        .collect();
    assert_eq!(terms.len(), 6);
    assert_eq!(
        terms[1]["outcome"]["value"],
        json!({"kind":"f32","bits":"4b800000"})
    );
    assert_eq!(terms[2]["outcome"], terms[1]["outcome"]);
    assert_eq!(
        terms[4]["outcome"]["value"],
        json!({"kind":"i64","value":"1"})
    );
    assert_eq!(
        terms[5]["outcome"]["value"],
        json!({"kind":"f64","bits":"3ff0000000000000"})
    );
    assert_eq!(
        report["native_values"]["ordinary_rir"]["status"],
        "accepted"
    );
}

#[test]
fn rejected_scalar_conversions_have_native_diagnostics_and_no_values() {
    for (scalar, literal) in [
        ("u32", "4294967296"),
        ("u64", "-1"),
        ("i32", "2147483648"),
        ("bool", "2"),
        ("bool", "\"true\""),
        ("symbol", "1"),
        ("i64", "1.0"),
        ("f32", "350000000000000000000000000000000000000.0"),
    ] {
        let report = extract(&format!("pred p({scalar}). p({literal}).\n"));
        let terms = occurrences(&report, "relation:p/1", "retained");
        assert_eq!(
            terms[0]["outcome"]["reason_code"], "constant_lowering_rejected",
            "{scalar}: {literal}"
        );
        assert!(terms[0]["outcome"].get("value").is_none());
        assert_eq!(
            report["native_values"]["schema_validation"]["status"],
            "rejected"
        );
    }
}

#[test]
fn transformed_meta_terms_are_not_given_a_fabricated_scalar_identity() {
    let report = extract("pred p(term). p(x). p(\"x\").\n");
    assert_eq!(
        report["native_values"]["normalization"]["status"],
        "accepted"
    );
    assert_eq!(
        report["native_values"]["ordinary_rir"]["status"],
        "accepted"
    );
    for term in occurrences(&report, "relation:p/1", "retained") {
        assert_eq!(term["outcome"]["reason_code"], "transformed_context");
        assert!(term["outcome"].get("value").is_none());
    }
    let invalid = extract("pred p(predref). p(1).\n");
    assert_eq!(
        invalid["native_values"]["normalization"]["status"],
        "rejected"
    );
    assert_eq!(
        invalid["native_values"]["schema_validation"]["status"],
        "unavailable"
    );
}

#[test]
fn modal_missing_schema_and_scalar_carriers_have_independent_statuses() {
    let report = extract(
        "#pragma epistemic_mode = g91\npred p(symbol). p(x). q(X) :- p(X), know missing(X).\n",
    );
    assert_eq!(
        report["native_values"]["ordinary_rir"]["status"],
        "rejected"
    );
    let p = occurrences(&report, "relation:p/1", "retained");
    assert_eq!(
        p[0]["outcome"]["value"],
        json!({"kind":"symbol","value":"x"})
    );
    let missing = occurrences(&report, "relation:missing/1", "retained");
    assert_eq!(missing[0]["outcome"]["reason_code"], "missing_schema");
    assert!(missing[0]["scalar_type"].is_null());
}

#[test]
fn imported_alias_providers_are_complete_and_selected_per_entrypoint() {
    let fixture = TempDir::new().unwrap();
    fs::write(
        fixture.path().join("library.xlog"),
        concat!(
            "pred source_ref(symbol, symbol). source_ref(ref_a, reviewed). ",
            "source_ref(\"ref_a\", \"reviewed\"). pred omitted(symbol). omitted(x).\n",
        ),
    )
    .unwrap();
    fs::write(
        fixture.path().join("twin.xlog"),
        "pred source_ref(symbol, symbol). source_ref(ref_a, reviewed).\n",
    )
    .unwrap();
    fs::write(
        fixture.path().join("main.xlog"),
        "use library::{source_ref}. use twin.\n",
    )
    .unwrap();
    fs::write(
        fixture.path().join("other.xlog"),
        "use library::{omitted}.\n",
    )
    .unwrap();
    let report = extract_at(fixture.path(), "main.xlog");
    let authored = occurrences(&report, "relation:source_ref/2", "authored");
    let retained = occurrences(&report, "relation:source_ref/2", "retained");
    assert_eq!(authored.len(), 6);
    assert_eq!(retained.len(), 4);
    for pair in authored.chunks_exact(2) {
        assert_eq!(pair[0]["outcome"], authored[0]["outcome"]);
        assert_eq!(pair[1]["outcome"], authored[1]["outcome"]);
    }
    for row in report["source_object_dispositions"].as_array().unwrap() {
        let projected = authored
            .iter()
            .any(|term| term["source_object_id"] == row["source_object_id"]);
        if projected {
            assert!(matches!(
                row["disposition"].as_str(),
                Some("admitted" | "coalesced")
            ));
            assert!(authored
                .iter()
                .filter(|term| term["source_object_id"] == row["source_object_id"])
                .all(|term| term["executable_source_object_id"]
                    == row["executable_source_object_id"]));
        }
    }
    assert!(occurrences(&report, "relation:omitted/1", "authored").is_empty());
    let other = extract_at(fixture.path(), "other.xlog");
    assert!(occurrences(&other, "relation:source_ref/2", "authored").is_empty());
    assert_eq!(
        occurrences(&other, "relation:omitted/1", "retained").len(),
        1
    );
}

#[test]
fn float_bits_follow_native_equality_and_conversion_boundaries() {
    use xlog_logic::arithmetic_eval::{compare_arithmetic_values, ArithmeticValue};
    use xlog_logic::ast::CompOp;

    let report = extract(&format!(
        "pred f(f32). f({:.1}). f(0.00000000000000000000000000000000000000000000000001). f(-0.00000000000000000000000000000000000000000000000001). pred d(f64). d(0.0). d(-0.0).\n", f32::MAX,
    ));
    let floats = occurrences(&report, "relation:f/1", "retained");
    assert_eq!(floats[0]["outcome"]["value"]["bits"], "7f7fffff");
    assert_eq!(floats[1]["outcome"]["value"]["bits"], "00000000");
    assert_eq!(floats[2]["outcome"]["value"]["bits"], "80000000");
    for (relation, values) in [
        (
            "relation:f/1",
            [ArithmeticValue::F32(0.0), ArithmeticValue::F32(-0.0)],
        ),
        (
            "relation:d/1",
            [ArithmeticValue::F64(0.0), ArithmeticValue::F64(-0.0)],
        ),
    ] {
        assert!(!compare_arithmetic_values(&values[0], CompOp::Eq, &values[1]).unwrap());
        let rows = occurrences(&report, relation, "retained");
        let last = rows.len() - 1;
        assert_ne!(rows[last - 1]["outcome"], rows[last]["outcome"]);
    }
    assert_eq!(
        report["native_values"]["ordinary_rir"]["status"],
        "accepted"
    );
}

#[test]
fn identical_authored_integers_have_positional_types_and_parser_limits() {
    let report = extract("pred n(u32, i32, i64, u64, f32, f64, bool). n(1, 1, 1, 1, 1, 1, 1).\n");
    let terms = occurrences(&report, "relation:n/7", "retained");
    for (term, kind) in terms
        .iter()
        .zip(["u32", "i32", "i64", "u64", "f32", "f64", "bool"])
    {
        assert_eq!(term["authored_term"], terms[0]["authored_term"]);
        assert_eq!(term["outcome"]["value"]["kind"], kind);
    }
    // Authored AST integer syntax is i64. This does not describe external U64 ingress.
    assert!(xlog_logic::parser::parse_program("pred p(u64). p(18446744073709551615).").is_err());
    let inferred = extract("p(1). q(X) :- p(X).\n");
    let inferred = occurrences(&inferred, "relation:p/1", "retained");
    assert_eq!(inferred[0]["scope"], "inferred_scalar_carrier");
    assert_eq!(
        inferred[0]["outcome"]["value"],
        json!({"kind":"u32", "value":"1"})
    );
}

#[test]
fn canonical_function_preparation_and_rejected_stages_remain_visible() {
    let source = "func number(X) = cast(X, u32). pred p(u32). pred q(u32). p(1). q(Y) :- p(X), Y is number(X).\n";
    let report = extract(source);
    assert_eq!(
        report["native_values"]["normalization"]["status"],
        "accepted"
    );
    assert_eq!(
        report["native_values"]["ordinary_rir"]["status"],
        "accepted"
    );
    assert!(report["native_values"]["source_diagnostics"]
        .as_array()
        .unwrap()
        .iter()
        .any(|diagnostic| diagnostic["reason_code"] == "transformed_context"));
    let invalid = "pred p(missing_domain). p(x).\n";
    let report = extract(invalid);
    assert_eq!(
        report["native_values"]["schema_validation"]["status"],
        "rejected"
    );
    assert_eq!(
        occurrences(&report, "relation:p/1", "retained")[0]["outcome"]["reason_code"],
        "schema_rejected"
    );
    // Native diagnostics and their ordering must also survive physical relocation.
    assert_eq!(extract(invalid), extract(invalid));
    let cyclic = extract("pred p(symbol). pred q(symbol). p(x) :- not q(x). q(x) :- not p(x).\n");
    assert_eq!(
        cyclic["native_values"]["schema_validation"]["status"],
        "accepted"
    );
    assert_eq!(
        cyclic["native_values"]["ordinary_rir"]["status"],
        "rejected"
    );
}

#[test]
fn comparison_carrier_matches_calculated_arithmetic_guard() {
    let source = "pred q(u32). q(Y) :- Y is cast(1, u32), Y = 1.\n";
    xlog_logic::Compiler::new().compile(source).unwrap();
    let report = extract(source);
    let constant = report["native_values"]["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .find(|term| {
            term["ownership"] == "retained"
                && term["path"]["kind"] == "comparison"
                && term["path"]["operand"] == "right"
        })
        .unwrap();
    assert_eq!(
        constant["outcome"]["value"],
        json!({"kind":"u32", "value":"1"})
    );
    assert_eq!(
        report["native_values"]["schema_validation"]["status"],
        "accepted"
    );
    assert_eq!(
        report["native_values"]["ordinary_rir"]["status"],
        "accepted"
    );
}

#[test]
fn shared_normalization_preserves_constraint_identity_and_transformed_boundaries() {
    use xlog_logic::ast::BodyLiteral;
    let source = "#pragma epistemic_mode = g91\npred p(symbol). p(x). :- p(X), know p(X).\n";
    let parsed = xlog_logic::parser::parse_program(source).unwrap();
    let normalized = xlog_logic::normalize_program_for_execution(parsed).unwrap();
    assert_eq!(normalized.authored_constraint_source_bound, Some(1));
    assert_eq!(normalized.constraints[0].authored_index, Some(0));
    assert_eq!(normalized.constraints[0].body.len(), 1);
    let helper = normalized
        .rules
        .iter()
        .find(|rule| rule.head.predicate == "__epi_join_0")
        .unwrap();
    assert_eq!(helper.body.len(), 2);
    assert!(helper
        .body
        .iter()
        .all(|literal| matches!(literal, BodyLiteral::Positive(_))));
    let report = extract(source);
    let terms = occurrences(&report, "relation:p/1", "retained");
    assert_eq!(
        terms.len(),
        3,
        "the report keeps the original constraint occurrences"
    );
    assert_eq!(
        report["native_values"]["normalization"]["status"],
        "accepted"
    );
    assert_eq!(
        report["native_values"]["ordinary_rir"]["status"],
        "rejected"
    );

    let report = extract("pred p(list<i64>). p([1, 2]).\n");
    assert_eq!(
        report["native_values"]["ordinary_rir"]["status"],
        "accepted"
    );
    assert_eq!(
        occurrences(&report, "relation:p/1", "retained")[0]["outcome"]["reason_code"],
        "transformed_context"
    );
}
