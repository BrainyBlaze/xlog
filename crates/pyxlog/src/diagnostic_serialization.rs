use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

pub(crate) fn pack_rule_provenance(
    py: Python<'_>,
    entries: &[xlog_logic::RuleProvenance],
) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);
    for entry in entries {
        let dict = PyDict::new(py);
        dict.set_item("rule_id", &entry.rule_id)?;
        dict.set_item("head", &entry.head)?;
        dict.set_item("source_kind", entry.source_kind.as_str())?;
        match &entry.source_span {
            Some(source_span) => dict.set_item("source_span", source_span)?,
            None => dict.set_item("source_span", py.None())?,
        }
        match &entry.generation_trace_hash {
            Some(hash) => dict.set_item("generation_trace_hash", hash)?,
            None => dict.set_item("generation_trace_hash", py.None())?,
        }
        dict.set_item("support_relation_ids", &entry.support_relation_ids)?;
        dict.set_item(
            "counterexample_relation_ids",
            &entry.counterexample_relation_ids,
        )?;
        list.append(dict)?;
    }
    Ok(list.into())
}

pub(crate) fn pack_query_proof_traces(
    py: Python<'_>,
    entries: &[xlog_logic::QueryProofTrace],
) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);
    for entry in entries {
        let dict = PyDict::new(py);
        dict.set_item("query_id", &entry.query_id)?;
        dict.set_item("query", &entry.query)?;
        dict.set_item("answer_relation", &entry.answer_relation)?;
        dict.set_item("rule_ids", &entry.rule_ids)?;
        dict.set_item("source_facts", &entry.source_facts)?;
        dict.set_item("rejected_alternatives", &entry.rejected_alternatives)?;
        list.append(dict)?;
    }
    Ok(list.into())
}
