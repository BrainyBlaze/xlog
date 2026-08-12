#![cfg(not(feature = "extension-module"))]

#[path = "../src/diagnostic_serialization.rs"]
mod diagnostic_serialization;

use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyList, PyListMethods};
use pyo3::{PyResult, Python};
use xlog_logic::{RuleProvenance, RuleSourceKind};

use diagnostic_serialization::{pack_query_proof_traces, pack_rule_provenance};

#[test]
fn rule_provenance_keeps_absent_optional_fields_as_python_none() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let packed = pack_rule_provenance(
            py,
            &[RuleProvenance {
                rule_id: "rule:generated:0:example".to_string(),
                head: "generated(X)".to_string(),
                source_kind: RuleSourceKind::Generated,
                source_span: None,
                generation_trace_hash: None,
                support_relation_ids: vec!["source".to_string()],
                counterexample_relation_ids: Vec::new(),
            }],
        )?;

        let records = packed.bind(py).cast::<PyList>()?;
        let record = records.get_item(0)?.cast_into::<PyDict>()?;
        for key in ["source_span", "generation_trace_hash"] {
            let value = record
                .get_item(key)?
                .unwrap_or_else(|| panic!("serializer omitted optional key {key}"));
            assert!(value.is_none(), "{key} must be represented by Python None");
        }

        let proof_traces = pack_query_proof_traces(py, &[])?;
        assert!(proof_traces.bind(py).cast::<PyList>()?.is_empty());
        Ok(())
    })
}
