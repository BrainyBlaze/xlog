//! Contextual wire restrictions and shared resource admission before token choice.

use std::collections::{BTreeMap, BTreeSet};

use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
use serde::de::{DeserializeSeed, Error, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use xlog_ir::tensor::{TensorParameter, TensorSignature};
use xlog_runtime::causal_decoder::acquired::{
    ACQUIRED_NODES, ACQUIRED_SCALARS, ACQUIRED_SOURCE_BYTES, ACQUIRED_WIDTH,
};

const LANGUAGE: &str = "xlog-contextual-acquisition-v1";
const ORDERED_LANGUAGE: &str = "xlog-contextual-acquisition-v2";
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
    if !matches!(version, 1 | 2) {
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
    if version == 2 {
        let tokens = raw_token_ids.ok_or_else(|| {
            value_error("ordered acquisition requires an explicit raw-token alphabet")
        })?;
        raw_alphabet(&tokens, vocabulary)?;
        context["raw_token_ids"] = json!(tokens);
    } else if raw_token_ids.is_some() {
        return Err(value_error(
            "a raw-token alphabet requires acquisition language version 2",
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
    let defs = &mut schema["$defs"];
    defs["Index"] = integer(0, ACQUIRED_NODES - 1);
    defs["Width"] = integer(1, maximum_width);
    defs["TokenId"] = integer(0, vocabulary - 1);
    if version == 2 {
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
    defs["FuncDef"]["required"] = json!(["name", "params", "return_type", "body", "is_private"]);
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
                if version == 2 {
                    properties["token_ids"]["maxItems"] =
                        json!(context["raw_token_ids"].as_array().unwrap().len());
                    properties["token_ids"]["uniqueItems"] = json!(true);
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
    let mut entry_program = defs["TensorProgram"].clone();
    entry_program["properties"]["signature"] = json!({"const": context["signature"]});
    if version == 2 {
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
    if version == 2 {
        schema["$id"] = json!("https://xlog.md/schemas/acquired-funcdef-context/v2");
        schema["description"] = json!(
            "Ordered contextual acquisition. Properties follow the canonical required-field order: signature and parameters before operations, operation tag before operands. The native prefix mask additionally bounds arguments by the actual signature, parameter references by declarations, and value/output references by preceding operations. Native compilation owns remaining type, shape, call and causal checks."
        );
        // Keep ordering local to this language. Enabling preserve_order for
        // serde_json workspace-wide would change unrelated durable contracts.
        #[derive(Serialize)]
        struct Packet<'a> {
            id: &'a str,
            context: &'a Value,
            schema: OrderedSchema<'a>,
        }
        return serde_json::to_string(&Packet {
            id: ORDERED_LANGUAGE,
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
    raw_tokens: BTreeSet<u32>,
}

#[pymethods]
impl PyNativeAcquisitionMask {
    #[new]
    fn new(language_json: &str, token_bytes: Vec<Vec<u8>>) -> PyResult<Self> {
        let language: Value = serde_json::from_str(language_json).map_err(value_error)?;
        let context = &language["context"];
        if (language["id"] != LANGUAGE && language["id"] != ORDERED_LANGUAGE)
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
        let ordered = language["id"] == ORDERED_LANGUAGE;
        let raw_tokens = if ordered {
            let tokens = serde_json::from_value::<Vec<u32>>(context["raw_token_ids"].clone())
                .map_err(value_error)?;
            raw_alphabet(&tokens, token_bytes.len())?
        } else {
            BTreeSet::new()
        };
        Ok(Self {
            ordered,
            raw_tokens,
            token_bytes,
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
                if !resource_prefix(&bytes, &self.retained, self.ordered, &self.raw_tokens) {
                    mask[id / 8] &= !bit;
                }
            }
            mask
        });
        Ok(PyBytes::new(py, &mask).unbind())
    }
}

struct Resources<'a> {
    bytes: &'a [u8],
    retained: &'a BTreeSet<String>,
    parameters: BTreeMap<String, usize>,
    scalars: usize,
    nodes: usize,
    ordered: bool,
    raw_tokens: &'a BTreeSet<u32>,
    program: Program,
}

#[derive(Default)]
struct Program {
    arguments: Option<usize>,
    parameters: usize,
    parameters_complete: bool,
    operations: usize,
}

fn resource_prefix(
    bytes: &[u8],
    retained: &BTreeSet<String>,
    ordered: bool,
    raw_tokens: &BTreeSet<u32>,
) -> bool {
    if bytes.len() > ACQUIRED_SOURCE_BYTES {
        return false;
    }
    let mut resources = Resources {
        bytes,
        retained,
        parameters: BTreeMap::new(),
        scalars: 0,
        nodes: 0,
        ordered,
        raw_tokens,
        program: Program::default(),
    };
    let mut decoder = serde_json::Deserializer::from_slice(bytes);
    match (Object {
        resources: &mut resources,
        level: Level::Root,
    })
    .deserialize(&mut decoder)
    {
        Ok(()) => decoder.end().is_ok(),
        Err(error) => error.is_eof(),
    }
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
        while let Some(key) = map.next_key::<String>()? {
            match (self.level, key.as_str()) {
                (Level::Root, "definitions") => map.next_value_seed(Array {
                    resources: self.resources,
                    kind: Items::Definitions,
                })?,
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
                }
                (Level::Program, "parameters") => {
                    map.next_value_seed(Array {
                        resources: self.resources,
                        kind: Items::Parameters,
                    })?;
                    self.resources.program.parameters_complete = true;
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
                    map.next_value_seed(Array {
                        resources: self.resources,
                        kind: Items::Operations,
                    })?;
                }
                (Level::Program, "output") if self.resources.ordered => {
                    map.next_value_seed(Index {
                        extent: self.resources.program.operations,
                    })?;
                }
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
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
                self.resources.nodes += 1;
                if self.resources.nodes > ACQUIRED_NODES {
                    return Err(D::Error::custom("acquired node budget exceeded"));
                }
                if self.resources.ordered {
                    decoder.deserialize_map(Operation {
                        program: &self.resources.program,
                        raw_tokens: self.resources.raw_tokens,
                        bytes: self.resources.bytes,
                    })?;
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
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "name" => name = Some(map.next_value::<String>()?),
                "shape" => {
                    // The wire grammar orders name before shape. A partial
                    // dimension is already a lower bound on its final extent;
                    // reject it before a closing delimiter becomes impossible.
                    let name = name
                        .as_ref()
                        .ok_or_else(|| A::Error::custom("parameter name must precede shape"))?;
                    let new = !self.resources.retained.contains(name)
                        && !self.resources.parameters.contains_key(name);
                    extent = Some(map.next_value_seed(Shape {
                        remaining: if new {
                            ACQUIRED_SCALARS - self.resources.scalars
                        } else {
                            usize::MAX
                        },
                    })?);
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
            self.resources.parameters.insert(name, extent);
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
}
impl<'de> Visitor<'de> for Operation<'_> {
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("operation tag followed by available operands")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        let mut tagged = false;
        while let Some(key) = map.next_key::<String>()? {
            if key == "operation" {
                map.next_value::<String>()?;
                tagged = true;
                continue;
            }
            if !tagged {
                return Err(A::Error::custom("operation tag must precede operands"));
            }
            let extent = match key.as_str() {
                "argument" => self.program.arguments,
                "weight" => Some(self.program.parameters),
                "input" | "initial" | "left" | "right" | "condition" | "positive" | "negative" => {
                    Some(self.program.operations)
                }
                _ => None,
            };
            if let Some(extent) = extent {
                map.next_value_seed(Index { extent })?;
            } else if key == "arguments" {
                map.next_value_seed(Indices {
                    extent: self.program.operations,
                })?;
            } else if key == "token_ids" {
                map.next_value_seed(TokenIds {
                    alphabet: self.raw_tokens,
                    bytes: self.bytes,
                })?;
            } else {
                map.next_value::<IgnoredAny>()?;
            }
        }
        Ok(())
    }
}

struct Indices {
    extent: usize,
}

struct TokenIds<'a> {
    alphabet: &'a BTreeSet<u32>,
    bytes: &'a [u8],
}
impl<'de> DeserializeSeed<'de> for TokenIds<'_> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, decoder: D) -> Result<(), D::Error> {
        decoder.deserialize_seq(self)
    }
}
impl<'de> Visitor<'de> for TokenIds<'_> {
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("distinct IDs from the declared raw-token alphabet")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<(), A::Error> {
        let mut seen = BTreeSet::new();
        while let Some(raw) = sequence.next_element::<&'de serde_json::value::RawValue>()? {
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
            if !self.alphabet.contains(&token) || !seen.insert(token) {
                return Err(A::Error::custom(
                    "token indicator must name a distinct raw-alphabet ID",
                ));
            }
        }
        Ok(())
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
    type Value = usize;
    fn deserialize<D: serde::Deserializer<'de>>(self, decoder: D) -> Result<usize, D::Error> {
        decoder.deserialize_seq(self)
    }
}
impl<'de> Visitor<'de> for Shape {
    type Value = usize;
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("bounded parameter dimensions")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<usize, A::Error> {
        let mut extent = 1usize;
        while let Some(dimension) = sequence.next_element::<usize>()? {
            extent = extent
                .checked_mul(dimension)
                .filter(|&value| value > 0 && value <= self.remaining)
                .ok_or_else(|| A::Error::custom("acquired scalar budget exceeded"))?;
        }
        Ok(extent)
    }
}
