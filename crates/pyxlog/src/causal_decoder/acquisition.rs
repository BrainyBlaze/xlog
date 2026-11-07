//! Contextual wire restrictions and shared resource admission before token choice.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
use serde::de::{DeserializeSeed, Error, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use xlog_ir::tensor::{TensorOperation, TensorParameter, TensorProgram, TensorSignature};
use xlog_logic::ast::FuncDef;

mod rows;
mod typed;
use xlog_runtime::causal_decoder::acquired::{
    definition_usage, ACQUIRED_NODES, ACQUIRED_SCALARS, ACQUIRED_SOURCE_BYTES, ACQUIRED_WIDTH,
};

const LANGUAGE: &str = "xlog-contextual-acquisition-v1";
const ORDERED_LANGUAGE: &str = "xlog-contextual-acquisition-v2";
const OPERAND_LANGUAGE: &str = "xlog-contextual-acquisition-v3";
const TYPED_LANGUAGE: &str = "xlog-contextual-acquisition-v4";
const PLANNED_LANGUAGE: &str = "xlog-contextual-acquisition-v5";
const SINGLE_OPERATION_LANGUAGE: &str = "xlog-contextual-acquisition-v6";
const NUMERICAL_ROWS_LANGUAGE: &str = "xlog-contextual-acquisition-v7";
const CONTEXT_ENTRY_ROWS_LANGUAGE: &str = "xlog-contextual-acquisition-v8";
const CANONICAL_ENTRY_LANGUAGE: &str = "xlog-contextual-acquisition-v9";
const INTENT_LANGUAGE: &str = "xlog-contextual-acquisition-v10";
const ENTRY_BUDGET_LANGUAGE: &str = "xlog-contextual-acquisition-v11";
pub(super) const NUMERICAL_ROWS_ENCODING: &str = rows::ENCODING;
pub(super) const CONTEXT_ENTRY_ROWS_ENCODING: &str = rows::CONTEXT_ENTRY_ENCODING;
pub(super) const INTENT_ROWS_ENCODING: &str = rows::INTENT_ENCODING;
const WIRE: &str = include_str!("../../python/pyxlog/acquired_funcdef_wire_v1.schema.json");

fn integer(minimum: usize, maximum: usize) -> Value {
    json!({"type": "integer", "minimum": minimum, "maximum": maximum})
}

/// Specialize the sole wire schema with the native model's actual ABI/registry.
pub(super) fn language(
    mut context: Value,
    version: usize,
    raw_token_ids: Option<Vec<u32>>,
) -> PyResult<String> {
    if !matches!(version, 1..=11) {
        return Err(value_error(
            "unsupported native acquisition language version",
        ));
    }
    let mut schema: Value = serde_json::from_str(WIRE).map_err(value_error)?;
    schema["$id"] = json!("https://xlog.md/schemas/acquired-funcdef-context/v1");
    schema["description"] = json!(
        "Contextual acquisition wire language. The native prefix mask additionally enforces cumulative new scalar/node/source budgets. Native compilation owns remaining shape, reference, reachability and causal checks."
    );
    let width = context["signature"]["input_width"].as_u64().unwrap() as usize;
    let vocabulary = context["vocabulary"].as_u64().unwrap() as usize;
    if version >= 2 {
        let tokens = raw_token_ids.ok_or_else(|| {
            value_error("ordered acquisition requires an explicit raw-token alphabet")
        })?;
        raw_alphabet(&tokens, vocabulary)?;
        context["raw_token_ids"] = json!(tokens);
    } else if raw_token_ids.is_some() {
        return Err(value_error(
            "a raw-token alphabet requires acquisition language version 2, 3 or 4",
        ));
    }
    let max_context = context["max_context"].as_u64().unwrap() as usize;
    let maximum_width = width.max(vocabulary).max(ACQUIRED_SCALARS);
    let function_name = json!({"type": "string", "pattern": format!(
        "^{}[a-zA-Z_][a-zA-Z0-9_]{{0,63}}$", context["function_prefix"].as_str().unwrap())});
    let parameter_name = json!({"type": "string", "pattern": format!(
        "^{}[a-zA-Z_][a-zA-Z0-9_]{{0,63}}$", context["parameter_prefix"].as_str().unwrap())});
    schema["properties"]["entrypoint"] = json!({"const": context["entrypoint"]});
    schema["properties"]["definitions"] = json!({
        "type": "array", "minItems": 1, "maxItems": ACQUIRED_NODES,
        "prefixItems": [{"$ref": "#/$defs/EntryFuncDef"}], "items": {"$ref": "#/$defs/FuncDef"},
    });
    if version >= 4 {
        // Helper names extend the entry's function namespace, so they cannot
        // match the entry's exact name. This union is disjoint without oneOf.
        schema["properties"]["definitions"] = json!({
            "type": "array", "minItems": 1, "maxItems": ACQUIRED_NODES,
            "items": {"anyOf": [{"$ref": "#/$defs/FuncDef"}, {"$ref": "#/$defs/EntryFuncDef"}]},
        });
    }
    let defs = &mut schema["$defs"];
    defs["Index"] = integer(0, ACQUIRED_NODES - 1);
    defs["Width"] = integer(1, maximum_width);
    defs["TokenId"] = integer(0, vocabulary - 1);
    if version >= 2 {
        defs["TokenId"] = json!({"enum": context["raw_token_ids"]});
    }
    defs["FunctionElement"]["enum"] = json!(["u32", "f32"]);
    defs["FunctionDimension"]["oneOf"][0]["properties"]["size"] = json!({"$ref": "#/$defs/Width"});
    defs["FunctionDimension"]["oneOf"][1]["properties"]["maximum"] = json!({"const": max_context});
    // All acquired signatures are tensor-valued; helper ABI equality is still
    // checked by the same FuncDef registration that admits the final body.
    defs["FunctionType"] = defs["FunctionType"]["oneOf"][1].clone();
    defs["FunctionType"]["properties"]["dimensions"]["minItems"] = json!(1);
    defs["FunctionType"]["properties"]["dimensions"]["maxItems"] = json!(2);
    defs["OptionalFunctionType"] = json!({"$ref": "#/$defs/FunctionType"});
    defs["FuncParam"]["required"] = json!(["name", "typ"]);
    defs["FuncParam"]["properties"]["name"]["minLength"] = json!(1);
    defs["FuncParam"]["properties"]["name"]["maxLength"] = json!(64);
    defs["FuncDef"]["properties"]["name"] = function_name.clone();
    defs["FuncDef"]["properties"]["params"]["maxItems"] = json!(ACQUIRED_NODES + 1);
    defs["FuncDef"]["properties"]["is_private"] = json!({"const": false});
    defs["FuncDef"]["required"] = if version >= 4 {
        json!(["name", "body", "params", "return_type", "is_private"])
    } else {
        json!(["name", "params", "return_type", "body", "is_private"])
    };
    for signature in defs["TensorSignature"]["oneOf"].as_array_mut().unwrap() {
        let properties = &mut signature["properties"];
        properties["output_width"] = json!({"$ref": "#/$defs/Width"});
        if properties.get("input_width").is_some() {
            properties["input_width"] = json!({"$ref": "#/$defs/Width"});
        } else {
            properties["input_widths"]["items"] = json!({"$ref": "#/$defs/Width"});
            properties["input_widths"]["maxItems"] = json!(ACQUIRED_NODES);
        }
    }
    let mut parameter = defs["TensorParameter"].clone();
    parameter["properties"]["name"] = parameter_name;
    parameter["properties"]["shape"] = json!({"type": "array", "minItems": 1, "maxItems": 2,
        "items": integer(1, ACQUIRED_SCALARS)});
    parameter["properties"]["float32"] = json!({"const": true});
    let mut choices = vec![parameter];
    choices.extend(
        context["parameters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| json!({"const": value})),
    );
    defs["TensorParameter"] = json!({"anyOf": choices});
    let mut callees = vec![function_name];
    let retained = context["callees"]
        .as_object()
        .unwrap()
        .keys()
        .collect::<Vec<_>>();
    if !retained.is_empty() {
        callees.push(json!({"enum": retained}));
    }
    defs["Callee"] = json!({"anyOf": callees});
    for operation in defs["TensorOperation"]["oneOf"].as_array_mut().unwrap() {
        let properties = &mut operation["properties"];
        match properties["operation"]["const"].as_str().unwrap() {
            "zeros" => properties["columns"] = json!({"$ref": "#/$defs/Width"}),
            "token_indicators" => {
                properties["token_ids"]["minItems"] = json!(1);
                if version >= 2 {
                    properties["token_ids"]["maxItems"] =
                        json!(context["raw_token_ids"].as_array().unwrap().len());
                    // Distinctness belongs to TokenIds below. The strict
                    // pinned grammar engine does not implement uniqueItems.
                }
            }
            "gather_columns" => {
                properties["indices"]["items"] = integer(0, maximum_width - 1);
                properties["indices"]["minItems"] = json!(1);
            }
            "call" | "scan" => {
                properties["function"] = json!({"$ref": "#/$defs/Callee"});
                if properties.get("max_steps").is_some() {
                    properties["max_steps"] = integer(1, max_context);
                } else {
                    properties["arguments"]["maxItems"] = json!(ACQUIRED_NODES);
                }
            }
            _ => {}
        }
    }
    let program = &mut defs["TensorProgram"]["properties"];
    program["vocabulary"] = json!({"const": vocabulary});
    program["max_context"] = json!({"const": max_context});
    program["parameters"]["maxItems"] = json!(ACQUIRED_NODES);
    program["operations"]["minItems"] = json!(1);
    program["operations"]["maxItems"] = json!(ACQUIRED_NODES);
    if version >= 5 {
        defs["TensorProgram"]["required"] = json!([
            "signature",
            "parameters",
            "output",
            "operations",
            "vocabulary",
            "max_context"
        ]);
    }
    let mut entry_program = defs["TensorProgram"].clone();
    entry_program["properties"]["signature"] = json!({"const": context["signature"]});
    if version >= 2 {
        // The entry's numeric ABI comes from TensorSignature, not the global
        // operation count or the number of formal parameters (tokens included).
        let signature: TensorSignature =
            serde_json::from_value(context["signature"].clone()).map_err(value_error)?;
        let arguments = signature.arguments(1).map_err(value_error)?.len();
        let mut operations = defs["TensorOperation"].clone();
        for operation in operations["oneOf"].as_array_mut().unwrap() {
            let properties = &mut operation["properties"];
            if properties["operation"]["const"] == "argument" {
                properties["argument"] = integer(0, arguments - 1);
            }
        }
        defs["EntryTensorOperation"] = operations;
        entry_program["properties"]["operations"]["items"] =
            json!({"$ref": "#/$defs/EntryTensorOperation"});
    }
    let mut entry = defs["FuncDef"].clone();
    entry["properties"]["name"] = json!({"const": context["entrypoint"]});
    entry["properties"]["params"] = json!({"const": context["params"]});
    entry["properties"]["return_type"] = json!({"const": context["return_type"]});
    entry["properties"]["body"]["properties"]["Numerical"] = entry_program;
    defs["EntryFuncDef"] = entry;
    if version >= 2 {
        schema["$id"] = json!(format!(
            "https://xlog.md/schemas/acquired-funcdef-context/v{version}"
        ));
        schema["description"] = json!(
            "Ordered contextual acquisition. Properties follow the canonical required-field order: signature and parameters before operations, operation tag before operands. The native prefix mask additionally bounds arguments by the actual signature, parameter references by declarations, and value/output references by preceding operations. Native compilation owns remaining type, shape, call and causal checks."
        );
        if version >= 3 {
            schema["description"] = json!(format!(
                "{} The native prefix mask also restricts operation names, including unfinished byte-token spellings, to operations whose mandatory numeric references have a nonempty available extent.",
                schema["description"].as_str().unwrap()
            ));
        }
        if version >= 4 {
            schema["description"] = json!(format!(
                "{} Definitions precede their callers and the entry is last. Typed operand choices and outputs use canonical operation inference at one and maximum context length with known callee signatures. Numerical bodies precede formal parameters and return types, whose canonical serialization comes from FuncDef::numerical_types. Entry closure requires actual callers for every new definition and constant. This is not a guarantee of termination or usefulness.",
                schema["description"].as_str().unwrap()
            ));
        }
        if version >= 5 {
            schema["description"] = json!(format!(
                "{} Each numerical body chooses output before operations, within the remaining shared node allowance. Output p requires exactly p+1 operations; slot p must match the canonical signature output at both context lengths. Earlier slots retain typed choices. This output-last search restriction does not guarantee completion within a token budget.",
                schema["description"].as_str().unwrap()
            ));
        }
        if version >= 6 {
            schema["description"] = json!(format!(
                "{} Output zero is available only when canonical prospective inference on the empty body admits a final operation using the declared parameters, known callees and raw-token alphabet. Positive output indices keep the same domain; this does not establish whole-body reachability.",
                schema["description"].as_str().unwrap()
            ));
        }
        if version >= 7 {
            rows::schema(&mut schema, version >= 8, version >= 10);
        }
        if version >= 9 {
            schema["description"] = json!(format!(
                "{} Entry formal-type prefixes use FuncDef::numerical_types and its struct serialization, matching the compact codec and helper ABI owner.",
                schema["description"].as_str().unwrap()
            ));
        }
        if version >= 11 {
            schema["description"] = json!(format!(
                "{} Entry plans and prospective operations must leave enough rows for every required root helper call and new physical parameter read, reserving a separate result row when none of the remaining obligations can have the required output shape. This necessary lower bound neither supplies operands nor guarantees completion or usefulness.",
                schema["description"].as_str().unwrap()
            ));
        }
        // Keep ordering local to this language. Enabling preserve_order for
        // serde_json workspace-wide would change unrelated durable contracts.
        #[derive(Serialize)]
        struct Packet<'a> {
            id: &'a str,
            context: &'a Value,
            schema: OrderedSchema<'a>,
        }
        return serde_json::to_string(&Packet {
            id: match version {
                11 => ENTRY_BUDGET_LANGUAGE,
                10 => INTENT_LANGUAGE,
                9 => CANONICAL_ENTRY_LANGUAGE,
                8 => CONTEXT_ENTRY_ROWS_LANGUAGE,
                7 => NUMERICAL_ROWS_LANGUAGE,
                6 => SINGLE_OPERATION_LANGUAGE,
                5 => PLANNED_LANGUAGE,
                4 => TYPED_LANGUAGE,
                3 => OPERAND_LANGUAGE,
                _ => ORDERED_LANGUAGE,
            },
            context: &context,
            schema: OrderedSchema(&schema),
        })
        .map_err(value_error);
    }
    serde_json::to_string(&json!({"id": LANGUAGE, "context": context, "schema": schema}))
        .map_err(value_error)
}

/// The canonical schema's required lists already specify its dependency order.
/// Preserve that order in properties while recursively specializing the schema.
struct OrderedSchema<'a>(&'a Value);

impl Serialize for OrderedSchema<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            Value::Object(object) => {
                let mut map = serializer.serialize_map(Some(object.len()))?;
                for (key, value) in object {
                    if key == "properties" {
                        map.serialize_entry(
                            key,
                            &Properties {
                                values: value.as_object().unwrap(),
                                required: object.get("required").and_then(Value::as_array),
                            },
                        )?;
                    } else {
                        map.serialize_entry(key, &OrderedSchema(value))?;
                    }
                }
                map.end()
            }
            Value::Array(values) => {
                let mut sequence = serializer.serialize_seq(Some(values.len()))?;
                for value in values {
                    sequence.serialize_element(&OrderedSchema(value))?;
                }
                sequence.end()
            }
            value => value.serialize(serializer),
        }
    }
}

struct Properties<'a> {
    values: &'a serde_json::Map<String, Value>,
    required: Option<&'a Vec<Value>>,
}

impl Serialize for Properties<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.values.len()))?;
        let required = self.required.map_or(&[][..], Vec::as_slice);
        for name in required.iter().filter_map(Value::as_str) {
            if let Some(value) = self.values.get(name) {
                map.serialize_entry(name, &OrderedSchema(value))?;
            }
        }
        for (name, value) in self.values {
            if !required.iter().any(|value| value.as_str() == Some(name)) {
                map.serialize_entry(name, &OrderedSchema(value))?;
            }
        }
        map.end()
    }
}

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn raw_alphabet(tokens: &[u32], vocabulary: usize) -> PyResult<BTreeSet<u32>> {
    let unique: BTreeSet<_> = tokens.iter().copied().collect();
    if tokens.is_empty()
        || tokens.len() != unique.len()
        || tokens.iter().any(|&token| token as usize >= vocabulary)
    {
        return Err(value_error(
            "raw-token alphabet must be nonempty, unique and inside the vocabulary",
        ));
    }
    Ok(unique)
}

/// Byte spellings are supplied by the pinned tokenizer, once per language.
/// This is a prefix resource owner, not a second numerical compiler.
#[pyclass(name = "NativeAcquisitionMask", module = "pyxlog._native")]
pub(crate) struct PyNativeAcquisitionMask {
    token_bytes: Vec<Vec<u8>>,
    retained: BTreeSet<String>,
    ordered: bool,
    planned_output: bool,
    check_single_operation: bool,
    raw_tokens: BTreeSet<u32>,
    operation_fields: Option<BTreeMap<String, Vec<String>>>,
    typed_context: Option<Value>,
    numerical_rows: bool,
    context_entry_rows: bool,
    canonical_entry_formals: bool,
    has_intent: bool,
    entry_budget: bool,
}

#[pymethods]
impl PyNativeAcquisitionMask {
    #[new]
    fn new(language_json: &str, token_bytes: Vec<Vec<u8>>) -> PyResult<Self> {
        let language: Value = serde_json::from_str(language_json).map_err(value_error)?;
        let context = &language["context"];
        if (language["id"] != LANGUAGE
            && language["id"] != ORDERED_LANGUAGE
            && language["id"] != OPERAND_LANGUAGE
            && language["id"] != TYPED_LANGUAGE
            && language["id"] != PLANNED_LANGUAGE
            && language["id"] != SINGLE_OPERATION_LANGUAGE
            && language["id"] != NUMERICAL_ROWS_LANGUAGE
            && language["id"] != CONTEXT_ENTRY_ROWS_LANGUAGE
            && language["id"] != CANONICAL_ENTRY_LANGUAGE
            && language["id"] != INTENT_LANGUAGE
            && language["id"] != ENTRY_BUDGET_LANGUAGE)
            || context["version"] != 1
            || context["limits"]
                != json!({"source_bytes": ACQUIRED_SOURCE_BYTES,
                "nodes": ACQUIRED_NODES, "new_scalars": ACQUIRED_SCALARS, "output_width": ACQUIRED_WIDTH})
            || context["vocabulary"].as_u64() != Some(token_bytes.len() as u64)
        {
            return Err(value_error("native acquisition language contract differs"));
        }
        let parameters: Vec<TensorParameter> =
            serde_json::from_value(context["parameters"].clone()).map_err(value_error)?;
        if parameters.iter().any(|parameter| !parameter.float32) {
            return Err(value_error("retained acquisition parameters must be FP32"));
        }
        let ordered = language["id"] != LANGUAGE;
        let entry_budget = language["id"] == ENTRY_BUDGET_LANGUAGE;
        let has_intent = language["id"] == INTENT_LANGUAGE || entry_budget;
        let canonical_entry_formals = language["id"] == CANONICAL_ENTRY_LANGUAGE || has_intent;
        let context_entry_rows =
            language["id"] == CONTEXT_ENTRY_ROWS_LANGUAGE || canonical_entry_formals;
        let numerical_rows = language["id"] == NUMERICAL_ROWS_LANGUAGE || context_entry_rows;
        let check_single_operation = language["id"] == SINGLE_OPERATION_LANGUAGE || numerical_rows;
        let planned_output = language["id"] == PLANNED_LANGUAGE || check_single_operation;
        let typed = language["id"] == TYPED_LANGUAGE || planned_output;
        let raw_tokens = if ordered {
            let tokens = serde_json::from_value::<Vec<u32>>(context["raw_token_ids"].clone())
                .map_err(value_error)?;
            raw_alphabet(&tokens, token_bytes.len())?
        } else {
            BTreeSet::new()
        };
        // Required operand fields come from the same wire schema as the
        // compiler-facing packet, never from an authored operation sequence.
        let operation_fields = if language["id"] == OPERAND_LANGUAGE || typed {
            Some(operation_fields()?)
        } else {
            None
        };
        if typed {
            let max_context = context["max_context"]
                .as_u64()
                .filter(|&value| value > 0)
                .ok_or_else(|| value_error("typed acquisition requires a positive context bound"))?
                as usize;
            if token_bytes.is_empty()
                || context["entrypoint"].as_str().is_none_or(str::is_empty)
                || !context["params"].is_array()
                || !context["return_type"].is_object()
            {
                return Err(value_error("typed acquisition ABI is incomplete"));
            }
            let callees = context["callees"]
                .as_object()
                .ok_or_else(|| value_error("typed acquisition requires known callee signatures"))?;
            for callee in callees.values() {
                let signature: TensorSignature =
                    serde_json::from_value(callee["signature"].clone()).map_err(value_error)?;
                signature.arguments(max_context).map_err(value_error)?;
                signature
                    .output(max_context, token_bytes.len())
                    .map_err(value_error)?;
            }
        }
        Ok(Self {
            numerical_rows,
            context_entry_rows,
            canonical_entry_formals,
            has_intent,
            entry_budget,
            ordered,
            planned_output,
            check_single_operation,
            raw_tokens,
            token_bytes,
            operation_fields,
            typed_context: typed.then(|| context.clone()),
            retained: parameters
                .into_iter()
                .map(|parameter| parameter.name)
                .collect(),
        })
    }

    /// Intersect the grammar's mask with cumulative native resource admission.
    /// Empty spellings represent EOS; the grammar decides when EOS is legal.
    fn filter(&self, py: Python<'_>, prefix: &[u8], allowed_mask: &[u8]) -> PyResult<Py<PyBytes>> {
        if allowed_mask.len() != self.token_bytes.len().div_ceil(32) * 4 {
            return Err(value_error("acquisition mask vocabulary differs"));
        }
        let mask = py.detach(|| {
            let mut mask = allowed_mask.to_vec();
            let mut bytes = prefix.to_vec();
            for (id, spelling) in self.token_bytes.iter().enumerate() {
                let bit = 1 << (id % 8);
                if mask[id / 8] & bit == 0 {
                    continue;
                }
                bytes.truncate(prefix.len());
                bytes.extend_from_slice(spelling);
                let canonical = match self.canonical_prefix(&bytes) {
                    Ok(canonical) => canonical,
                    Err(_) => {
                        mask[id / 8] &= !bit;
                        continue;
                    }
                };
                if resource_prefix(
                    &canonical,
                    &self.retained,
                    self.ordered,
                    &self.raw_tokens,
                    self.operation_fields.as_ref(),
                    self.typed_context.as_ref(),
                    self.planned_output,
                    self.check_single_operation,
                    self.canonical_entry_formals,
                    self.entry_budget,
                )
                .is_err()
                {
                    mask[id / 8] &= !bit;
                }
            }
            mask
        });
        Ok(PyBytes::new(py, &mask).unbind())
    }

    /// Describe completed prefix state without selecting tokens or changing it.
    fn completion_state(&self, prefix: &[u8]) -> PyResult<String> {
        if self.typed_context.is_none() {
            return Err(value_error(
                "completion state requires acquisition language version 4",
            ));
        }
        let canonical = self.canonical_prefix(prefix).map_err(value_error)?;
        let resources = resource_prefix(
            &canonical,
            &self.retained,
            self.ordered,
            &self.raw_tokens,
            self.operation_fields.as_ref(),
            self.typed_context.as_ref(),
            self.planned_output,
            self.check_single_operation,
            self.canonical_entry_formals,
            self.entry_budget,
        )
        .map_err(value_error)?;
        serde_json::to_string(&resources.completion_state().map_err(value_error)?)
            .map_err(value_error)
    }

    /// Elaborate complete compact source with the packet's captured native ABI.
    fn decode_definition(&self, source_bytes: &[u8]) -> PyResult<String> {
        if !self.numerical_rows {
            return Err(value_error(
                "definition decoding requires numerical rows language",
            ));
        }
        let (entrypoint, definitions) = rows::definitions(
            source_bytes,
            self.typed_context.as_ref().unwrap(),
            self.operation_fields.as_ref().unwrap(),
            self.context_entry_rows,
            self.has_intent,
        )
        .map_err(value_error)?;
        serde_json::to_string(&json!({"entrypoint": entrypoint, "definitions": definitions}))
            .map_err(value_error)
    }
}

impl PyNativeAcquisitionMask {
    fn canonical_prefix<'a>(&self, source: &'a [u8]) -> Result<Cow<'a, [u8]>, String> {
        if self.numerical_rows {
            rows::prefix(
                source,
                self.typed_context.as_ref().unwrap(),
                self.operation_fields.as_ref().unwrap(),
                self.context_entry_rows,
                self.has_intent,
            )
            .map(|prefix| Cow::Owned(prefix.bytes))
        } else if source.len() > ACQUIRED_SOURCE_BYTES {
            Err("acquisition source byte budget exceeded".into())
        } else {
            Ok(Cow::Borrowed(source))
        }
    }
}

fn operation_fields() -> PyResult<BTreeMap<String, Vec<String>>> {
    let wire: Value = serde_json::from_str(WIRE).map_err(value_error)?;
    Ok(wire["$defs"]["TensorOperation"]["oneOf"]
        .as_array()
        .unwrap()
        .iter()
        .map(|operation| {
            (
                operation["properties"]["operation"]["const"]
                    .as_str()
                    .unwrap()
                    .to_owned(),
                operation["required"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|field| field.as_str().unwrap().to_owned())
                    .collect(),
            )
        })
        .collect())
}

pub(super) fn decode_numerical_rows(
    source: &[u8],
    context: &Value,
    context_entry: bool,
    has_intent: bool,
) -> PyResult<(String, Vec<FuncDef>)> {
    rows::definitions(
        source,
        context,
        &operation_fields()?,
        context_entry,
        has_intent,
    )
    .map_err(value_error)
}

struct Resources<'a> {
    bytes: &'a [u8],
    retained: &'a BTreeSet<String>,
    parameters: BTreeMap<String, usize>,
    scalars: usize,
    nodes: usize,
    ordered: bool,
    planned_output: bool,
    check_single_operation: bool,
    canonical_entry_formals: bool,
    entry_budget: bool,
    raw_tokens: &'a BTreeSet<u32>,
    operation_fields: Option<&'a BTreeMap<String, Vec<String>>>,
    program: Program,
    typed_context: Option<&'a Value>,
    signatures: typed::Signatures,
    definition_name: Option<String>,
    definition_index: Option<usize>,
    program_definition_index: Option<usize>,
    definitions: BTreeMap<String, TensorProgram>,
    declarations: BTreeMap<String, TensorParameter>,
}

#[derive(Default)]
struct Program {
    arguments: Option<usize>,
    parameters: usize,
    parameters_complete: bool,
    operations: usize,
    typed: Option<typed::Program>,
}

fn resource_prefix<'a>(
    bytes: &'a [u8],
    retained: &'a BTreeSet<String>,
    ordered: bool,
    raw_tokens: &'a BTreeSet<u32>,
    operation_fields: Option<&'a BTreeMap<String, Vec<String>>>,
    typed_context: Option<&'a Value>,
    planned_output: bool,
    check_single_operation: bool,
    canonical_entry_formals: bool,
    entry_budget: bool,
) -> Result<Resources<'a>, serde_json::Error> {
    let mut resources = Resources {
        bytes,
        retained,
        parameters: BTreeMap::new(),
        scalars: 0,
        nodes: 0,
        ordered,
        planned_output,
        check_single_operation,
        canonical_entry_formals,
        entry_budget,
        raw_tokens,
        operation_fields,
        program: Program::default(),
        typed_context,
        signatures: typed_context
            .map(|context| {
                context["callees"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(name, value)| {
                        (
                            name.clone(),
                            serde_json::from_value(value["signature"].clone()).unwrap(),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default(),
        definition_name: None,
        definition_index: None,
        program_definition_index: None,
        definitions: BTreeMap::new(),
        declarations: typed_context
            .map(|context| {
                serde_json::from_value::<Vec<TensorParameter>>(context["parameters"].clone())
                    .unwrap()
                    .into_iter()
                    .map(|parameter| (parameter.name.clone(), parameter))
                    .collect()
            })
            .unwrap_or_default(),
    };
    let mut decoder = serde_json::Deserializer::from_slice(bytes);
    match (Object {
        resources: &mut resources,
        level: Level::Root,
    })
    .deserialize(&mut decoder)
    {
        Ok(()) => decoder.end()?,
        Err(error) if error.is_eof() => {}
        Err(error) => return Err(error),
    }
    Ok(resources)
}

impl Resources<'_> {
    fn entry_requirements(&self) -> Result<Option<typed::EntryRequirements>, String> {
        if !self.entry_budget
            || self.definition_name.as_deref()
                != self
                    .typed_context
                    .and_then(|context| context["entrypoint"].as_str())
        {
            return Ok(None);
        }
        let mut calls = self.definitions.keys().cloned().collect::<BTreeSet<_>>();
        let mut parameters = self
            .declarations
            .keys()
            .filter(|name| !self.retained.contains(*name))
            .cloned()
            .collect::<BTreeSet<_>>();
        // Callee-first order makes the new-definition graph acyclic. Roots need
        // direct entry calls; all other new helpers become reachable through them.
        // Reads in any helper are therefore covered by those required calls.
        for body in self.definitions.values() {
            for operation in &body.operations {
                if let Some(name) = operation.callee() {
                    calls.remove(name);
                }
                if let Some(weight) = operation.parameter() {
                    let parameter = body
                        .parameters
                        .get(weight)
                        .ok_or_else(|| "unknown acquired parameter reference".to_owned())?;
                    parameters.remove(&parameter.name);
                }
            }
        }
        Ok(Some(typed::EntryRequirements { calls, parameters }))
    }

    fn completion_state(&self) -> Result<Value, String> {
        // A newly opened definition has no shape facts until its own signature
        // completes, even while the parser still retains the preceding body.
        let program = self
            .program
            .typed
            .as_ref()
            .filter(|_| self.program_definition_index == self.definition_index);
        let mut read = BTreeSet::new();
        for body in self
            .definitions
            .values()
            .chain(program.map(|program| &program.body))
        {
            for operation in &body.operations {
                if let Some(id) = operation.parameter() {
                    let parameter = body
                        .parameters
                        .get(id)
                        .ok_or_else(|| "unknown acquired parameter reference".to_owned())?;
                    read.insert(&parameter.name);
                }
            }
        }
        // These are reads anywhere in completed operations, not reachability
        // from the entry. The constructor's final usage gate remains separate.
        let unread = self
            .declarations
            .keys()
            .filter(|name| !self.retained.contains(*name) && !read.contains(name))
            .count();
        Ok(json!({
            "version": 1,
            "definition_index": self.definition_index,
            "required_output": program.map(typed::Program::required_output).transpose()?,
            "last_value": program.and_then(typed::Program::last_value),
            "compatible_outputs": program.map_or(0, |program| program.outputs().len()),
            "unread_constants": unread,
            "entry_completed": self.definitions.contains_key(
                self.typed_context.unwrap()["entrypoint"].as_str().unwrap()
            ),
        }))
    }

    fn complete_entry(&self) -> Result<(), String> {
        let Some(context) = self.typed_context else {
            return Ok(());
        };
        let entry = context["entrypoint"].as_str().unwrap();
        if self.definition_name.as_deref() != Some(entry) {
            return Ok(());
        }
        let program = &self.program.typed.as_ref().unwrap().body;
        let usage = definition_usage(entry, |name| {
            if name == entry {
                Ok(Some(program))
            } else if let Some(body) = self.definitions.get(name) {
                Ok(Some(body))
            } else if self.signatures.contains_key(name) {
                Ok(None)
            } else {
                Err(xlog_core::XlogError::Compilation(
                    "callee must precede its caller".into(),
                ))
            }
        })
        .map_err(|error| error.to_string())?;
        if self
            .definitions
            .keys()
            .any(|name| !usage.definitions.contains(name))
            || self
                .declarations
                .keys()
                .any(|name| !self.retained.contains(name) && !usage.parameters.contains(name))
        {
            return Err("entry has unreachable definitions or unread new constants".into());
        }
        Ok(())
    }
}

fn next_key<'de, A: MapAccess<'de>>(
    map: &mut A,
    bytes: &[u8],
    raw: bool,
) -> Result<Option<(String, Option<usize>)>, A::Error> {
    if raw {
        let Some(key) = map.next_key::<&'de serde_json::value::RawValue>()? else {
            return Ok(None);
        };
        let offset = key.get().as_bytes().as_ptr_range().end as usize - bytes.as_ptr() as usize;
        let key = serde_json::from_str::<String>(key.get()).map_err(A::Error::custom)?;
        Ok(Some((key, Some(offset))))
    } else {
        Ok(map.next_key::<String>()?.map(|key| (key, None)))
    }
}

fn value_prefix(bytes: &[u8], expected: &[u8]) -> bool {
    let bytes = bytes.trim_ascii_start();
    if bytes.is_empty() {
        return true;
    }
    let Some(bytes) = bytes.strip_prefix(b":") else {
        return false;
    };
    let bytes = bytes.trim_ascii_start();
    let count = bytes.len().min(expected.len());
    bytes[..count] == expected[..count]
}

#[derive(Clone, Copy)]
enum Level {
    Root,
    Definition,
    Body,
    Program,
}

struct Object<'a, 'b> {
    resources: &'a mut Resources<'b>,
    level: Level,
}

impl<'de> DeserializeSeed<'de> for Object<'_, '_> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, decoder: D) -> Result<(), D::Error> {
        if matches!(self.level, Level::Program) {
            self.resources.program = Program::default();
        }
        decoder.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for Object<'_, '_> {
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("numerical definition object")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        if matches!(self.level, Level::Definition) {
            self.resources.definition_index =
                Some(self.resources.definition_index.map_or(0, |index| index + 1));
        }
        while let Some((key, offset)) = next_key(
            &mut map,
            self.resources.bytes,
            self.resources.typed_context.is_some(),
        )? {
            match (self.level, key.as_str()) {
                (Level::Root, "definitions") => map.next_value_seed(Array {
                    resources: self.resources,
                    kind: Items::Definitions,
                })?,
                (Level::Definition, "name") if self.resources.typed_context.is_some() => {
                    let name = map.next_value::<String>()?;
                    if self.resources.signatures.contains_key(&name) {
                        return Err(A::Error::custom("definition identity is already present"));
                    }
                    self.resources.definition_name = Some(name);
                }
                (Level::Definition, "params" | "return_type")
                    if self.resources.typed_context.is_some() =>
                {
                    let program = self.resources.program.typed.as_ref().ok_or_else(|| {
                        A::Error::custom("numerical body must precede formal types")
                    })?;
                    let context = self.resources.typed_context.unwrap();
                    let expected = if !self.resources.canonical_entry_formals
                        && self.resources.definition_name.as_deref()
                            == context["entrypoint"].as_str()
                    {
                        serde_json::to_vec(
                            &context[if key == "params" {
                                "params"
                            } else {
                                "return_type"
                            }],
                        )
                        .map_err(A::Error::custom)?
                    } else {
                        // Compact codecs serialize these structs directly.
                        // A context Value sorts object keys and has different
                        // bytes despite representing the same entry ABI.
                        let (params, result) = FuncDef::numerical_types(
                            &program.body.signature,
                            program.body.vocabulary,
                            program.body.max_context,
                        )
                        .map_err(A::Error::custom)?;
                        if key == "params" {
                            serde_json::to_vec(&params)
                        } else {
                            serde_json::to_vec(&result)
                        }
                        .map_err(A::Error::custom)?
                    };
                    if !value_prefix(&self.resources.bytes[offset.unwrap()..], &expected) {
                        return Err(A::Error::custom(
                            "formal type prefix differs from the numerical ABI",
                        ));
                    }
                    map.next_value::<IgnoredAny>()?;
                }
                (Level::Definition, "is_private") if self.resources.typed_context.is_some() => {
                    let raw = map.next_value::<&'de serde_json::value::RawValue>()?;
                    let end = raw.get().as_bytes().as_ptr_range().end as usize
                        - self.resources.bytes.as_ptr() as usize;
                    if self.resources.definition_name.as_deref()
                        == self.resources.typed_context.unwrap()["entrypoint"].as_str()
                    {
                        let rest = self.resources.bytes[end..].trim_ascii_start();
                        if rest
                            .strip_prefix(b"}")
                            .is_some_and(|rest| rest.trim_ascii_start().starts_with(b","))
                        {
                            return Err(A::Error::custom("entry must be the last definition"));
                        }
                    }
                }
                (Level::Definition, "body") => map.next_value_seed(Object {
                    resources: self.resources,
                    level: Level::Body,
                })?,
                (Level::Body, "Numerical") => map.next_value_seed(Object {
                    resources: self.resources,
                    level: Level::Program,
                })?,
                (Level::Program, "signature") if self.resources.ordered => {
                    let signature = map.next_value::<TensorSignature>()?;
                    let arguments = signature.arguments(1).map_err(A::Error::custom)?.len();
                    self.resources.program.arguments = Some(arguments);
                    if let Some(context) = self.resources.typed_context {
                        self.resources.program.typed = Some(typed::Program::new(
                            signature,
                            context["vocabulary"].as_u64().unwrap() as usize,
                            context["max_context"].as_u64().unwrap() as usize,
                            self.resources.raw_tokens.iter().copied().collect(),
                        ));
                        self.resources.program_definition_index = self.resources.definition_index;
                    }
                }
                (Level::Program, "parameters") => {
                    map.next_value_seed(Array {
                        resources: self.resources,
                        kind: Items::Parameters,
                    })?;
                    self.resources.program.parameters_complete = true;
                    let required = self
                        .resources
                        .entry_requirements()
                        .map_err(A::Error::custom)?;
                    if let Some(program) = &mut self.resources.program.typed {
                        program.entry_requirements = required;
                    }
                }
                (Level::Program, "operations") => {
                    if self.resources.ordered
                        && (self.resources.program.arguments.is_none()
                            || !self.resources.program.parameters_complete)
                    {
                        return Err(A::Error::custom(
                            "signature and parameters must precede operations",
                        ));
                    }
                    if self.resources.planned_output
                        && self
                            .resources
                            .program
                            .typed
                            .as_ref()
                            .is_none_or(|program| program.planned_output.is_none())
                    {
                        return Err(A::Error::custom("output must precede operations"));
                    }
                    map.next_value_seed(Array {
                        resources: self.resources,
                        kind: Items::Operations,
                    })?;
                }
                (Level::Program, "output") if self.resources.ordered => {
                    if self.resources.planned_output {
                        if self.resources.program.arguments.is_none()
                            || !self.resources.program.parameters_complete
                            || self.resources.program.operations != 0
                        {
                            return Err(A::Error::custom(
                                "output must follow signature and parameters, before operations",
                            ));
                        }
                        let zero_available = if self.resources.check_single_operation {
                            // Ask the same final-slot predicate before choosing
                            // p=0. This body has no operations yet; the clone's
                            // prospective target never changes the parsed body.
                            let mut candidate =
                                self.resources.program.typed.as_ref().unwrap().clone();
                            candidate.planned_output = Some(0);
                            self.resources.operation_fields.unwrap().keys().any(|tag| {
                                let mut fields = serde_json::Map::new();
                                fields.insert("operation".into(), json!(tag));
                                candidate.possible(&fields, None, &self.resources.signatures)
                            })
                        } else {
                            true
                        };
                        let choice = map.next_value_seed(NumberChoice {
                            values: 0..ACQUIRED_NODES - self.resources.nodes,
                            accepts: |output| {
                                (output != 0 || zero_available)
                                    && self
                                        .resources
                                        .program
                                        .typed
                                        .as_ref()
                                        .unwrap()
                                        .permits_plan(output, &self.resources.signatures)
                            },
                            bytes: self.resources.bytes,
                        })?;
                        // An EOF decimal is still a prefix, not a chosen slot.
                        if choice.end < self.resources.bytes.len() {
                            let program = self.resources.program.typed.as_mut().unwrap();
                            if program.planned_output.is_some() {
                                return Err(A::Error::custom("output slot is already chosen"));
                            }
                            program.body.output = choice.value;
                            program.planned_output = Some(choice.value);
                        }
                    } else if let Some(program) = &self.resources.program.typed {
                        let outputs = program.outputs();
                        let choice = map.next_value_seed(NumberChoice {
                            values: 0..program.body.operations.len(),
                            accepts: |output| outputs.contains(&output),
                            bytes: self.resources.bytes,
                        })?;
                        self.resources.program.typed.as_mut().unwrap().body.output = choice.value;
                    } else {
                        map.next_value_seed(Index {
                            extent: self.resources.program.operations,
                        })?;
                    }
                }
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }
        if matches!(self.level, Level::Definition) {
            if let (Some(name), Some(program)) = (
                &self.resources.definition_name,
                &self.resources.program.typed,
            ) {
                self.resources
                    .signatures
                    .insert(name.clone(), program.body.signature.clone());
                self.resources
                    .definitions
                    .insert(name.clone(), program.body.clone());
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Items {
    Definitions,
    Parameters,
    Operations,
}
struct Array<'a, 'b> {
    resources: &'a mut Resources<'b>,
    kind: Items,
}

impl<'de> DeserializeSeed<'de> for Array<'_, '_> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, decoder: D) -> Result<(), D::Error> {
        decoder.deserialize_seq(self)
    }
}
impl<'de> Visitor<'de> for Array<'_, '_> {
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("numerical definition array")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<(), A::Error> {
        while sequence
            .next_element_seed(Item {
                resources: self.resources,
                kind: self.kind,
            })?
            .is_some()
        {}
        if matches!(self.kind, Items::Operations)
            && self
                .resources
                .program
                .typed
                .as_ref()
                .is_some_and(|program| program.outputs().is_empty())
        {
            return Err(A::Error::custom(
                "operations have no output compatible with the signature",
            ));
        }
        if matches!(self.kind, Items::Operations) {
            if self.resources.planned_output {
                let program = self.resources.program.typed.as_ref().unwrap();
                if program.planned_output != self.resources.program.operations.checked_sub(1)
                    || !program.outputs().contains(&program.body.output)
                {
                    return Err(A::Error::custom(
                        "operations must end at the chosen compatible output slot",
                    ));
                }
            }
            self.resources.complete_entry().map_err(A::Error::custom)?;
        }
        if matches!(self.kind, Items::Definitions) {
            if let Some(context) = self.resources.typed_context {
                if !self
                    .resources
                    .definitions
                    .contains_key(context["entrypoint"].as_str().unwrap())
                {
                    return Err(A::Error::custom("definitions must end with the entry"));
                }
            }
        }
        Ok(())
    }
}
struct Item<'a, 'b> {
    resources: &'a mut Resources<'b>,
    kind: Items,
}
impl<'de> DeserializeSeed<'de> for Item<'_, '_> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, decoder: D) -> Result<(), D::Error> {
        match self.kind {
            Items::Definitions => {
                if self.resources.typed_context.is_some_and(|context| {
                    self.resources
                        .definitions
                        .contains_key(context["entrypoint"].as_str().unwrap())
                }) {
                    return Err(D::Error::custom("entry must be the last definition"));
                }
                if self.resources.nodes >= ACQUIRED_NODES {
                    return Err(D::Error::custom("no nodes remain for another definition"));
                }
                Object {
                    resources: self.resources,
                    level: Level::Definition,
                }
                .deserialize(decoder)
            }
            Items::Parameters => decoder.deserialize_map(Parameter {
                resources: self.resources,
            }),
            Items::Operations => {
                if self.resources.planned_output
                    && self.resources.program.typed.as_ref().is_none_or(|program| {
                        program
                            .planned_output
                            .is_none_or(|output| self.resources.program.operations > output)
                    })
                {
                    return Err(D::Error::custom("operations exceed the chosen output slot"));
                }
                self.resources.nodes += 1;
                if self.resources.nodes > ACQUIRED_NODES {
                    return Err(D::Error::custom("acquired node budget exceeded"));
                }
                if self.resources.ordered {
                    let operation = decoder.deserialize_map(Operation {
                        program: &self.resources.program,
                        raw_tokens: self.resources.raw_tokens,
                        bytes: self.resources.bytes,
                        operation_fields: self.resources.operation_fields,
                        signatures: &self.resources.signatures,
                    })?;
                    if let (Some(program), Some(operation)) =
                        (&mut self.resources.program.typed, operation)
                    {
                        program
                            .push(operation, &self.resources.signatures)
                            .map_err(D::Error::custom)?;
                    }
                } else {
                    IgnoredAny::deserialize(decoder)?;
                }
                self.resources.program.operations += 1;
                Ok(())
            }
        }
    }
}

struct Parameter<'a, 'b> {
    resources: &'a mut Resources<'b>,
}
impl<'de> Visitor<'de> for Parameter<'_, '_> {
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("FP32 parameter declaration")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        let mut name = None;
        let mut extent = None;
        let mut dimensions = Vec::new();
        while let Some((key, offset)) = next_key(
            &mut map,
            self.resources.bytes,
            self.resources.typed_context.is_some(),
        )? {
            match key.as_str() {
                "name" => {
                    let value = map.next_value::<String>()?;
                    if self
                        .resources
                        .program
                        .typed
                        .as_ref()
                        .is_some_and(|program| {
                            program
                                .body
                                .parameters
                                .iter()
                                .any(|parameter| parameter.name == value)
                        })
                    {
                        return Err(A::Error::custom(
                            "parameter table identities must be unique",
                        ));
                    }
                    name = Some(value);
                }
                "shape" => {
                    // The wire grammar orders name before shape. A partial
                    // dimension is already a lower bound on its final extent;
                    // reject it before a closing delimiter becomes impossible.
                    let name = name
                        .as_ref()
                        .ok_or_else(|| A::Error::custom("parameter name must precede shape"))?;
                    if let Some(parameter) = self.resources.declarations.get(name) {
                        let expected =
                            serde_json::to_vec(&parameter.shape).map_err(A::Error::custom)?;
                        if !value_prefix(&self.resources.bytes[offset.unwrap()..], &expected) {
                            return Err(A::Error::custom(
                                "parameter alias shape differs from its declaration",
                            ));
                        }
                    }
                    let new = !self.resources.retained.contains(name)
                        && !self.resources.parameters.contains_key(name);
                    let shape = map.next_value_seed(Shape {
                        remaining: if new {
                            ACQUIRED_SCALARS - self.resources.scalars
                        } else {
                            usize::MAX
                        },
                    })?;
                    extent = Some(shape.0);
                    dimensions = shape.1;
                }
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }
        let name = name.ok_or_else(|| A::Error::custom("parameter name absent"))?;
        let extent = extent.ok_or_else(|| A::Error::custom("parameter shape absent"))?;
        if !self.resources.retained.contains(&name)
            && !self.resources.parameters.contains_key(&name)
        {
            self.resources.scalars += extent;
            self.resources.parameters.insert(name.clone(), extent);
        }
        if let Some(program) = &mut self.resources.program.typed {
            let parameter = TensorParameter {
                name,
                shape: dimensions,
                float32: true,
            };
            if program
                .body
                .parameters
                .iter()
                .any(|old| old.name == parameter.name)
            {
                return Err(A::Error::custom(
                    "parameter table identities must be unique",
                ));
            }
            self.resources
                .declarations
                .insert(parameter.name.clone(), parameter.clone());
            program.body.parameters.push(parameter);
        }
        self.resources.program.parameters += 1;
        Ok(())
    }
}

/// Index checks are monotone even for an unfinished decimal token. They apply
/// to the current program, not the cumulative node allowance across functions.
struct Index {
    extent: usize,
}
impl<'de> DeserializeSeed<'de> for Index {
    type Value = usize;
    fn deserialize<D: serde::Deserializer<'de>>(self, decoder: D) -> Result<usize, D::Error> {
        let value = usize::deserialize(decoder)?;
        if value >= self.extent {
            return Err(D::Error::custom(
                "numerical reference is outside its available extent",
            ));
        }
        Ok(value)
    }
}

struct Operation<'a> {
    program: &'a Program,
    raw_tokens: &'a BTreeSet<u32>,
    bytes: &'a [u8],
    operation_fields: Option<&'a BTreeMap<String, Vec<String>>>,
    signatures: &'a typed::Signatures,
}

fn reference_extent(program: &Program, field: &str) -> Option<usize> {
    match field {
        "argument" => program.arguments,
        "weight" => Some(program.parameters),
        "input" | "initial" | "left" | "right" | "condition" | "positive" | "negative" => {
            Some(program.operations)
        }
        _ => None,
    }
}

/// Whether the value following a complete JSON key can still name this ASCII
/// operation. Include unfinished BPE spellings and JSON Unicode escapes: waiting
/// for the closing quote would merely move the empty-mask failure to that quote.
fn operation_prefix(bytes: &[u8], name: &str) -> bool {
    let bytes = bytes.trim_ascii_start();
    if bytes.is_empty() {
        return true;
    }
    let Some(bytes) = bytes.strip_prefix(b":") else {
        return false;
    };
    let bytes = bytes.trim_ascii_start();
    if bytes.is_empty() {
        return true;
    }
    let Some(mut bytes) = bytes.strip_prefix(b"\"") else {
        return false;
    };
    for character in name.bytes() {
        if bytes.is_empty() {
            return true;
        }
        if bytes[0] == b'\\' {
            let escaped = format!("\\u{character:04x}");
            let count = bytes.len().min(escaped.len());
            if !bytes[..count].eq_ignore_ascii_case(&escaped.as_bytes()[..count]) {
                return false;
            }
            if count < escaped.len() {
                return true;
            }
            bytes = &bytes[count..];
        } else {
            if bytes[0] != character {
                return false;
            }
            bytes = &bytes[1..];
        }
    }
    bytes.is_empty() || bytes[0] == b'"'
}

impl<'de> Visitor<'de> for Operation<'_> {
    type Value = Option<TensorOperation>;
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("operation tag followed by available operands")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut tagged = false;
        let mut material = serde_json::Map::new();
        let mut last_value_offset = None;
        loop {
            let Some((key, offset)) =
                next_key(&mut map, self.bytes, self.operation_fields.is_some())?
            else {
                break;
            };
            last_value_offset = offset;
            let after_key = offset.map(|offset| &self.bytes[offset..]);
            if key == "operation" {
                if let Some(fields) = self.operation_fields {
                    let viable = fields.iter().any(|(name, required)| {
                        required
                            .iter()
                            .all(|field| reference_extent(self.program, field) != Some(0))
                            && operation_prefix(after_key.unwrap(), name)
                            && self.program.typed.as_ref().is_none_or(|program| {
                                let mut candidate = material.clone();
                                candidate.insert(key.clone(), Value::String(name.clone()));
                                program.possible(&candidate, None, self.signatures)
                            })
                    });
                    if !viable {
                        return Err(A::Error::custom(
                            "operation has no compatible mandatory operands",
                        ));
                    }
                }
                let tag = map.next_value::<String>()?;
                if self.program.typed.is_some() {
                    material.insert(key, Value::String(tag));
                }
                tagged = true;
                continue;
            }
            if !tagged {
                return Err(A::Error::custom("operation tag must precede operands"));
            }
            let extent = reference_extent(self.program, &key);
            if let Some(program) = &self.program.typed {
                let choices = if key == "columns" {
                    program.output_columns().map(|columns| columns..columns + 1)
                } else {
                    extent.map(|extent| 0..extent)
                };
                let value = if let Some(values) = choices {
                    let choice = map.next_value_seed(NumberChoice {
                        values,
                        accepts: |value| {
                            key == "columns"
                                || program.number_possible(&material, &key, value, self.signatures)
                        },
                        bytes: self.bytes,
                    })?;
                    if choice.end == self.bytes.len() {
                        // The domain already admitted this unfinished decimal.
                        // Its temporary integer is not yet a selected operand.
                        continue;
                    }
                    Value::from(choice.value)
                } else if key == "arguments" || key == "indices" {
                    serde_json::to_value(map.next_value_seed(TypedIndices {
                        program,
                        material: &material,
                        field: &key,
                        signatures: self.signatures,
                        bytes: self.bytes,
                    })?)
                    .map_err(A::Error::custom)?
                } else if key == "token_ids" {
                    serde_json::to_value(map.next_value_seed(TokenIds {
                        alphabet: self.raw_tokens,
                        bytes: self.bytes,
                        required_length: program.output_columns(),
                    })?)
                    .map_err(A::Error::custom)?
                } else if key == "function" {
                    let tail = after_key.unwrap();
                    let known = self.signatures.keys().any(|name| {
                        let mut candidate = material.clone();
                        candidate.insert(key.clone(), Value::String(name.clone()));
                        operation_prefix(tail, name)
                            && program.possible(&candidate, None, self.signatures)
                    });
                    if !known {
                        return Err(A::Error::custom("callee has no compatible operands"));
                    }
                    Value::String(map.next_value::<String>()?)
                } else {
                    map.next_value::<Value>()?
                };
                material.insert(key, value);
                if !program.possible(&material, None, self.signatures) {
                    return Err(A::Error::custom(
                        "operands have no compatible operation completion",
                    ));
                }
            } else if let Some(extent) = extent {
                map.next_value_seed(Index { extent })?;
            } else if key == "arguments" {
                map.next_value_seed(Indices {
                    extent: self.program.operations,
                })?;
            } else if key == "token_ids" {
                map.next_value_seed(TokenIds {
                    alphabet: self.raw_tokens,
                    bytes: self.bytes,
                    required_length: None,
                })?;
            } else {
                map.next_value::<IgnoredAny>()?;
            }
        }
        if self
            .program
            .typed
            .as_ref()
            .is_some_and(typed::Program::is_output_slot)
        {
            // The map is complete. Locate its final value in the original
            // bytes so even a trailing comma without a next operation fails.
            // RawValue only finds the delimiter; operand inference above owns
            // the semantics, including unfinished numeric and array prefixes.
            if let Some(offset) = last_value_offset {
                let tail = self.bytes[offset..].trim_ascii_start();
                let tail = tail.strip_prefix(b":").unwrap().trim_ascii_start();
                let mut decoder = serde_json::Deserializer::from_slice(tail);
                let raw = <&serde_json::value::RawValue>::deserialize(&mut decoder)
                    .map_err(A::Error::custom)?;
                let end =
                    raw.get().as_bytes().as_ptr_range().end as usize - self.bytes.as_ptr() as usize;
                if self.bytes[end..]
                    .trim_ascii_start()
                    .strip_prefix(b"}")
                    .is_some_and(|rest| rest.trim_ascii_start().starts_with(b","))
                {
                    return Err(A::Error::custom("no operation may follow the output slot"));
                }
            }
        }
        if self.program.typed.is_some() {
            serde_json::from_value(Value::Object(material))
                .map(Some)
                .map_err(A::Error::custom)
        } else {
            Ok(None)
        }
    }
}

struct ChosenNumber {
    value: usize,
    end: usize,
}
struct NumberChoice<'a, F> {
    values: Range<usize>,
    accepts: F,
    bytes: &'a [u8],
}
impl<'de, F: Fn(usize) -> bool> DeserializeSeed<'de> for NumberChoice<'_, F> {
    type Value = ChosenNumber;
    fn deserialize<D: serde::Deserializer<'de>>(self, decoder: D) -> Result<Self::Value, D::Error> {
        let raw = <&'de serde_json::value::RawValue>::deserialize(decoder)?;
        let spelling = raw.get();
        let end = spelling.as_bytes().as_ptr_range().end as usize - self.bytes.as_ptr() as usize;
        let value = spelling.parse::<usize>().map_err(D::Error::custom)?;
        let viable = if end == self.bytes.len() {
            // Only unfinished decimals need a search for compatible extensions.
            self.values
                .clone()
                .any(|id| id.to_string().starts_with(spelling) && (self.accepts)(id))
        } else {
            self.values.contains(&value) && (self.accepts)(value)
        };
        if !viable {
            return Err(D::Error::custom(
                "reference has no compatible decimal continuation",
            ));
        }
        Ok(ChosenNumber { value, end })
    }
}

struct TypedIndices<'a> {
    program: &'a typed::Program,
    material: &'a serde_json::Map<String, Value>,
    field: &'a str,
    signatures: &'a typed::Signatures,
    bytes: &'a [u8],
}
impl<'de> DeserializeSeed<'de> for TypedIndices<'_> {
    type Value = Vec<usize>;
    fn deserialize<D: serde::Deserializer<'de>>(self, decoder: D) -> Result<Self::Value, D::Error> {
        decoder.deserialize_seq(self)
    }
}
impl<'de> Visitor<'de> for TypedIndices<'_> {
    type Value = Vec<usize>;
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("compatible numeric operands")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
        let mut values = Vec::new();
        loop {
            let Some(value) = sequence.next_element_seed(NumberChoice {
                values: self.program.array_range(self.material, self.field, &values),
                accepts: |value| {
                    self.program.array_number_possible(
                        self.material,
                        self.field,
                        &values,
                        value,
                        self.signatures,
                    )
                },
                bytes: self.bytes,
            })?
            else {
                break;
            };
            values.push(value.value);
            // Once all slots are filled, a comma cannot be accepted merely
            // because its next element has not arrived in this byte token yet.
            if self.bytes[value.end..].trim_ascii_start().starts_with(b",")
                && !self
                    .program
                    .array_range(self.material, self.field, &values)
                    .any(|value| {
                        self.program.array_number_possible(
                            self.material,
                            self.field,
                            &values,
                            value,
                            self.signatures,
                        )
                    })
            {
                return Err(A::Error::custom(
                    "operand array has no compatible next slot",
                ));
            }
        }
        if !self
            .program
            .array_can_end(self.material, self.field, &values, self.signatures)
        {
            return Err(A::Error::custom(
                "operand array cannot complete this operation",
            ));
        }
        Ok(values)
    }
}

struct Indices {
    extent: usize,
}

struct TokenIds<'a> {
    alphabet: &'a BTreeSet<u32>,
    bytes: &'a [u8],
    required_length: Option<usize>,
}
impl<'de> DeserializeSeed<'de> for TokenIds<'_> {
    type Value = Vec<u32>;
    fn deserialize<D: serde::Deserializer<'de>>(self, decoder: D) -> Result<Self::Value, D::Error> {
        decoder.deserialize_seq(self)
    }
}
impl<'de> Visitor<'de> for TokenIds<'_> {
    type Value = Vec<u32>;
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("distinct IDs from the declared raw-token alphabet")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
        let mut seen = BTreeSet::new();
        let mut values = Vec::new();
        while let Some(raw) = sequence.next_element::<&'de serde_json::value::RawValue>()? {
            if self
                .required_length
                .is_some_and(|length| values.len() >= length)
            {
                return Err(A::Error::custom("token array exceeds the output width"));
            }
            let spelling = raw.get();
            // The borrowed RawValue identifies the exact source slice. At EOF
            // a decimal may still extend across BPE tokens (e.g. "1" -> "15").
            // Once a delimiter follows it, membership/uniqueness are final.
            if spelling.as_bytes().as_ptr_range().end == self.bytes.as_ptr_range().end {
                if !self
                    .alphabet
                    .iter()
                    .any(|id| !seen.contains(id) && id.to_string().starts_with(spelling))
                {
                    return Err(A::Error::custom(
                        "no unused raw-token ID extends this decimal prefix",
                    ));
                }
                continue;
            }
            let token = spelling.parse::<u32>().map_err(A::Error::custom)?;
            values.push(token);
            if !self.alphabet.contains(&token) || !seen.insert(token) {
                return Err(A::Error::custom(
                    "token indicator must name a distinct raw-alphabet ID",
                ));
            }
            let end =
                spelling.as_bytes().as_ptr_range().end as usize - self.bytes.as_ptr() as usize;
            if self.required_length == Some(values.len())
                && self.bytes[end..].trim_ascii_start().starts_with(b",")
            {
                return Err(A::Error::custom("token array has no remaining output slot"));
            }
        }
        if self
            .required_length
            .is_some_and(|length| values.len() != length)
        {
            return Err(A::Error::custom("token array must match the output width"));
        }
        Ok(values)
    }
}
impl<'de> DeserializeSeed<'de> for Indices {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, decoder: D) -> Result<(), D::Error> {
        decoder.deserialize_seq(self)
    }
}
impl<'de> Visitor<'de> for Indices {
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("preceding numerical value references")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<(), A::Error> {
        while sequence
            .next_element_seed(Index {
                extent: self.extent,
            })?
            .is_some()
        {}
        Ok(())
    }
}

struct Shape {
    remaining: usize,
}
impl<'de> DeserializeSeed<'de> for Shape {
    type Value = (usize, Vec<usize>);
    fn deserialize<D: serde::Deserializer<'de>>(self, decoder: D) -> Result<Self::Value, D::Error> {
        decoder.deserialize_seq(self)
    }
}
impl<'de> Visitor<'de> for Shape {
    type Value = (usize, Vec<usize>);
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("bounded parameter dimensions")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
        let mut extent = 1usize;
        let mut dimensions = Vec::new();
        while let Some(dimension) = sequence.next_element::<usize>()? {
            dimensions.push(dimension);
            extent = extent
                .checked_mul(dimension)
                .filter(|&value| value > 0 && value <= self.remaining)
                .ok_or_else(|| A::Error::custom("acquired scalar budget exceeded"))?;
        }
        Ok((extent, dimensions))
    }
}
