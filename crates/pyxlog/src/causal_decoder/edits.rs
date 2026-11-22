use super::*;
use pyo3::types::PyList;
use xlog_runtime::causal_decoder::edits::{EditOperand, EditScore, NumericalEdit};

fn field<'py>(row: &Bound<'py, PyDict>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    row.get_item(name)?
        .ok_or_else(|| PyValueError::new_err(format!("missing edit field {name}")))
}

pub(super) fn extract(row: &Bound<'_, PyDict>) -> PyResult<NumericalEdit> {
    if row.len() != 6 {
        return Err(PyValueError::new_err("expected a complete typed edit"));
    }
    let arguments = field(row, "arguments")?
        .extract::<Vec<Bound<'_, PyDict>>>()?
        .into_iter()
        .map(|arg| {
            if arg.len() != 2 {
                return Err(PyValueError::new_err("expected a source and its operand"));
            }
            let source = field(&arg, "source")?.extract::<String>()?;
            match source.as_str() {
                "history" => Ok(EditOperand::History {
                    coordinates: field(&arg, "coordinates")?.extract()?,
                }),
                "descriptor" => Ok(EditOperand::Descriptor {
                    coordinates: field(&arg, "coordinates")?.extract()?,
                }),
                "definition" => Ok(EditOperand::Definition {
                    name: field(&arg, "name")?.extract()?,
                }),
                _ => Err(PyValueError::new_err("unknown typed edit operand")),
            }
        })
        .collect::<PyResult<Vec<_>>>()?;
    let operation = serde_json::from_value(serde_json::Value::String(
        field(row, "operation")?.extract()?,
    ))
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NumericalEdit {
        parent_id: field(row, "parent_id")?.extract()?,
        edit_id: field(row, "edit_id")?.extract()?,
        target: field(row, "target")?.extract()?,
        site: field(row, "site")?.extract()?,
        operation,
        arguments,
    })
}

pub(super) fn record(py: Python<'_>, e: NumericalEdit) -> PyResult<Py<PyDict>> {
    let row = PyDict::new(py);
    row.set_item("parent_id", e.parent_id)?;
    row.set_item("edit_id", e.edit_id)?;
    row.set_item("target", e.target)?;
    row.set_item("site", e.site)?;
    let op = serde_json::to_value(e.operation).map_err(xlog_err)?;
    row.set_item(
        "operation",
        op.as_str().expect("edit operation serializes as its name"),
    )?;
    let args = PyList::empty(py);
    for arg in e.arguments {
        let operand = PyDict::new(py);
        match arg {
            EditOperand::History { coordinates } => {
                operand.set_item("source", "history")?;
                operand.set_item("coordinates", coordinates.to_vec())?;
            }
            EditOperand::Descriptor { coordinates } => {
                operand.set_item("source", "descriptor")?;
                operand.set_item("coordinates", coordinates.to_vec())?;
            }
            EditOperand::Definition { name } => {
                operand.set_item("source", "definition")?;
                operand.set_item("name", name)?;
            }
        }
        args.append(operand)?;
    }
    row.set_item("arguments", args)?;
    Ok(row.unbind())
}

pub(super) fn score(py: Python<'_>, v: EditScore) -> PyResult<Py<PyDict>> {
    let row = PyDict::new(py);
    row.set_item("parent_id", v.parent_id)?;
    row.set_item("edit_id", v.edit_id)?;
    row.set_item("base_score", v.base_score)?;
    row.set_item("rule_delta", v.rule_delta)?;
    row.set_item("rule_id", v.rule_id)?;
    Ok(row.unbind())
}
