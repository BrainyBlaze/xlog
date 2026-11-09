//! Positional numerical source syntax, elaborated into the canonical definition wire.

use super::*;
use serde::de::DeserializeOwned;
use serde_json::value::RawValue;
use xlog_logic::ast::FuncBody;

pub(super) mod indices;
pub(super) mod metadata;

pub(super) const ENCODING: &str = "numerical-rows-v1";
pub(super) const CONTEXT_ENTRY_ENCODING: &str = "numerical-rows-v2";
pub(super) const INTENT_ENCODING: &str = "numerical-rows-v3";
pub(super) const COMPACT_INDICES_ENCODING: &str = "numerical-rows-v4";
pub(super) const COMPACT_METADATA_ENCODING: &str = "numerical-rows-v5";
const INTENT_CHARACTERS: usize = 256;

fn tuple(items: Vec<Value>) -> Value {
    json!({"type": "array", "minItems": items.len(),
        "maxItems": items.len(), "prefixItems": items, "items": false})
}

/// Derive positional rows from the existing specialized canonical operation
/// schema. Its required-field lists own both the operand order and exact arity.
pub(super) fn schema(
    schema: &mut Value,
    context_entry: bool,
    has_intent: bool,
    compact_indices: bool,
) {
    let defs = &mut schema["$defs"];
    for name in ["TensorOperation", "EntryTensorOperation"] {
        let mut rows = Vec::new();
        for operation in defs[name]["oneOf"].as_array_mut().unwrap() {
            let fields = operation["required"].as_array().unwrap();
            let items = fields
                .iter()
                .map(|field| operation["properties"][field.as_str().unwrap()].clone())
                .collect::<Vec<_>>();
            rows.push(tuple(items));
        }
        // Distinct constant tags make this an exact disjoint union. The pinned
        // strict grammar need not prove oneOf disjointness for tuple schemas.
        defs[name] = json!({"anyOf": rows});
    }
    let definition = |name: Value, program: &Value| {
        json!({"type": "object", "additionalProperties": false,
            "required": ["name", "signature", "parameters", "output", "operations"],
            "properties": {"name": name,
                "signature": program["properties"]["signature"],
                "parameters": program["properties"]["parameters"],
                "output": program["properties"]["output"],
                "operations": program["properties"]["operations"]}})
    };
    let helper = definition(
        defs["FuncDef"]["properties"]["name"].clone(),
        &defs["TensorProgram"],
    );
    let entry = definition(
        defs["EntryFuncDef"]["properties"]["name"].clone(),
        &defs["EntryFuncDef"]["properties"]["body"]["properties"]["Numerical"],
    );
    defs["FuncDef"] = helper;
    defs["EntryFuncDef"] = entry;
    schema["description"] = json!(
        "Compact numerical-rows-v1 acquisition. Definitions contain name, signature, parameters, output and operations in that order. Operation arrays contain the canonical tag followed by its required operands in canonical wire order. Native context supplies vocabulary and max_context; FuncDef::numerical supplies formal types. The same typed prefix owner enforces callee-first/entry-last order, planned output p with p+1 operations, single-operation feasibility, shape at one and maximum context length, causal/usage checks and shared original-source/node/scalar budgets. Termination and usefulness are not guaranteed."
    );
    if context_entry {
        let defs = &mut schema["$defs"];
        for name in ["FuncDef", "EntryFuncDef"] {
            let definition = &mut defs[name];
            definition["properties"]
                .as_object_mut()
                .unwrap()
                .remove("output");
            definition["properties"]["operation_count"] = integer(1, ACQUIRED_NODES);
            definition["required"] = json!([
                "name",
                "signature",
                "parameters",
                "operation_count",
                "operations"
            ]);
        }
        let entry = &mut defs["EntryFuncDef"];
        for field in ["name", "signature"] {
            entry["properties"].as_object_mut().unwrap().remove(field);
        }
        entry["required"] = json!(["parameters", "operation_count", "operations"]);
        schema["required"] = json!(["helpers", "entry"]);
        schema["properties"] = json!({
            "helpers": {"type": "array", "minItems": 0, "maxItems": ACQUIRED_NODES - 1,
                "items": {"$ref": "#/$defs/FuncDef"}},
            "entry": {"$ref": "#/$defs/EntryFuncDef"}
        });
        schema["description"] = json!(
            "Compact numerical-rows-v2 acquisition. Source contains helpers followed by entry. Helpers contain name, signature, parameters, operation_count and operations; entry contains parameters, operation_count and operations. Native context supplies the entry name and signature. Positive operation_count n requires exactly n operations and maps to canonical output n-1 after its number delimiter is visible. Operation rows, FuncDef::numerical ABI derivation, typed final-slot admission, single-operation feasibility, both context lengths, causal/usage checks and shared original-source/node/scalar budgets retain the numerical-rows-v1 owners. Every helper, constant, operation and operand is selected in source. Termination and usefulness are not guaranteed."
        );
    }
    if has_intent {
        schema["required"] = json!(["intent", "helpers", "entry"]);
        schema["properties"]["intent"] =
            json!({"type": "string", "minLength": 1, "maxLength": INTENT_CHARACTERS});
        schema["description"] = json!(format!(
            "{} Numerical-rows-v3 additionally starts with a model-authored intent string of one through {INTENT_CHARACTERS} Unicode scalar values. Intent remains in original source and its byte budget, but describes rather than executes the subsequent model-selected graph. The codec removes only this descriptive field during canonical elaboration; it does not check usefulness or agreement between intent and operations.",
            schema["description"].as_str().unwrap()
                .replace("numerical-rows-v2", "numerical-rows-v3")
        ));
    }
    if compact_indices {
        let maximum = schema["$defs"]["Width"]["maximum"].as_u64().unwrap() as usize;
        for name in ["TensorOperation", "EntryTensorOperation"] {
            for operation in schema["$defs"][name]["anyOf"].as_array_mut().unwrap() {
                if operation["prefixItems"][0]["const"] == "gather_columns" {
                    let literal = operation["prefixItems"][2].clone();
                    operation["prefixItems"][2] = json!({"anyOf": [literal, {
                        "type": "object", "additionalProperties": false,
                        "required": indices::FIELDS,
                        "properties": {
                            "count": {"$ref": "#/$defs/Width"},
                            "modulus": {"$ref": "#/$defs/Width"},
                            "start": integer(0, maximum - 1),
                            "step": integer(0, maximum - 1)
                        }
                    }]});
                }
            }
        }
        schema["description"] = json!(format!(
            "{} Gather indices additionally accept an ordered object with count, modulus, start and step. It expands to indices[j]=(start+j*step)%modulus; all four numbers are selected in source. Positive count uses the existing Width and typed Gather domain, modulus is within the input width, and start/step are below modulus. Original and cumulatively elaborated canonical source each fit the native source-byte allowance. Canonical Gather execution and gradients are unchanged.",
            schema["description"].as_str().unwrap().replace("numerical-rows-v3", "numerical-rows-v4")
        ));
    }
}

pub(super) struct Prefix<'a> {
    pub bytes: Vec<u8>,
    pub complete: bool,
    pub operation_count: Option<&'a [u8]>,
    pub indices: Vec<indices::Record>,
    pub pending_indices: Option<indices::Pending<'a>>,
}

enum Stop {
    Incomplete,
    Invalid(String),
}

impl From<serde_json::Error> for Stop {
    fn from(error: serde_json::Error) -> Self {
        Self::Invalid(error.to_string())
    }
}

impl From<std::io::Error> for Stop {
    fn from(error: std::io::Error) -> Self {
        Self::Invalid(error.to_string())
    }
}

struct Output {
    bytes: Vec<u8>,
    bounded: bool,
}

impl Output {
    fn ensure(&self, additional: usize) -> Result<(), std::io::Error> {
        if self.bounded
            && self
                .bytes
                .len()
                .checked_add(additional)
                .is_none_or(|size| size > ACQUIRED_SOURCE_BYTES)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "projected acquisition source byte budget exceeded",
            ));
        }
        Ok(())
    }

    fn push(&mut self, byte: u8) -> Result<(), Stop> {
        self.extend_from_slice(&[byte])
    }

    fn extend_from_slice(&mut self, bytes: &[u8]) -> Result<(), Stop> {
        std::io::Write::write_all(self, bytes).map_err(Into::into)
    }

    fn json(&mut self, value: &impl Serialize) -> Result<(), Stop> {
        serde_json::to_writer(self, value).map_err(Into::into)
    }
}

impl std::io::Write for Output {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.ensure(bytes.len())?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

struct Reader<'a> {
    source: &'a [u8],
    offset: usize,
    output: Output,
    context: &'a Value,
    fields: &'a BTreeMap<String, Vec<String>>,
    context_entry: bool,
    has_intent: bool,
    pending_operation_count: Option<usize>,
    compact_indices: bool,
    compact_metadata: bool,
    definition_name: String,
    definition_index: usize,
    indices: Vec<indices::Record>,
    pending_indices: Option<(usize, indices::Target)>,
}

/// Only syntax is expanded. An unfinished operand remains the same byte prefix
/// for the existing resource/type owner, including strings, decimals and arrays.
pub(super) fn prefix<'a>(
    source: &'a [u8],
    context: &Value,
    fields: &BTreeMap<String, Vec<String>>,
    context_entry: bool,
    has_intent: bool,
    compact_indices: bool,
    compact_metadata: bool,
) -> Result<Prefix<'a>, String> {
    if source.len() > ACQUIRED_SOURCE_BYTES {
        return Err("acquisition source byte budget exceeded".into());
    }
    let mut reader = Reader {
        source,
        offset: 0,
        output: Output {
            bytes: Vec::with_capacity(source.len()),
            bounded: compact_indices,
        },
        context,
        fields,
        context_entry,
        has_intent,
        pending_operation_count: None,
        compact_indices,
        compact_metadata,
        definition_name: String::new(),
        definition_index: 0,
        indices: Vec::new(),
        pending_indices: None,
    };
    let complete = match reader.document() {
        Ok(()) => true,
        Err(Stop::Incomplete) => false,
        Err(Stop::Invalid(error)) => return Err(error),
    };
    Ok(Prefix {
        bytes: reader.output.bytes,
        complete,
        operation_count: reader.pending_operation_count.map(|start| &source[start..]),
        indices: reader.indices,
        pending_indices: reader
            .pending_indices
            .map(|(start, target)| indices::Pending {
                bytes: &source[start..],
                target,
            }),
    })
}

/// Decode the same syntax used by prefix admission, then construct definitions
/// through their canonical ABI owner. No operation or parameter is selected here.
pub(super) fn definitions(
    source: &[u8],
    context: &Value,
    fields: &BTreeMap<String, Vec<String>>,
    context_entry: bool,
    has_intent: bool,
    compact_indices: bool,
    compact_metadata: bool,
) -> Result<(String, Vec<FuncDef>), String> {
    let decoded = prefix(
        source,
        context,
        fields,
        context_entry,
        has_intent,
        compact_indices,
        compact_metadata,
    )?;
    if !decoded.complete {
        return Err("incomplete numerical rows source".into());
    }
    let document: Value = serde_json::from_slice(&decoded.bytes).map_err(|e| e.to_string())?;
    let definitions: Vec<FuncDef> =
        serde_json::from_value(document["definitions"].clone()).map_err(|e| e.to_string())?;
    let definitions = definitions
        .into_iter()
        .map(|definition| match definition.body {
            FuncBody::Numerical(body) => {
                FuncDef::numerical(definition.name, body).map_err(|e| e.to_string())
            }
            _ => Err("numerical rows require numerical definitions".into()),
        })
        .collect::<Result<Vec<_>, String>>()?;
    if compact_indices {
        let mut signatures = indices::signatures(context)?;
        for (index, definition) in definitions.iter().enumerate() {
            let FuncBody::Numerical(body) = &definition.body else {
                unreachable!()
            };
            for record in decoded
                .indices
                .iter()
                .filter(|record| record.target.definition_index == index)
            {
                if record.target.definition != definition.name {
                    return Err("structured indices definition differs".into());
                }
                record.validate_body(body, &signatures, maximum_width(context))?;
            }
            signatures.insert(definition.name.clone(), body.signature.clone());
        }
    }
    Ok((
        context["entrypoint"].as_str().unwrap().to_owned(),
        definitions,
    ))
}

impl Reader<'_> {
    fn whitespace(&mut self) -> Result<(), Stop> {
        while self
            .source
            .get(self.offset)
            .is_some_and(u8::is_ascii_whitespace)
        {
            self.output.push(self.source[self.offset])?;
            self.offset += 1;
        }
        Ok(())
    }

    fn peek(&mut self) -> Result<u8, Stop> {
        self.whitespace()?;
        self.source
            .get(self.offset)
            .copied()
            .ok_or(Stop::Incomplete)
    }

    fn take(&mut self, expected: u8) -> Result<(), Stop> {
        if self.peek()? != expected {
            return Err(Stop::Invalid(
                "numerical rows field order or arity differs".into(),
            ));
        }
        self.offset += 1;
        Ok(())
    }

    fn punctuation(&mut self, expected: u8, canonical: u8) -> Result<(), Stop> {
        self.take(expected)?;
        self.output.push(canonical)?;
        Ok(())
    }

    fn raw(&mut self, copy: bool) -> Result<&RawValue, Stop> {
        self.whitespace()?;
        let tail = &self.source[self.offset..];
        let mut values = serde_json::Deserializer::from_slice(tail).into_iter::<&RawValue>();
        let value = match values.next() {
            Some(Ok(value)) => value,
            None => return Err(Stop::Incomplete),
            Some(Err(error)) if error.is_eof() => {
                if copy {
                    self.output.extend_from_slice(tail)?;
                }
                return Err(Stop::Incomplete);
            }
            Some(Err(error)) => return Err(error.into()),
        };
        self.offset += values.byte_offset();
        if copy {
            self.output.extend_from_slice(value.get().as_bytes())?;
        }
        Ok(value)
    }

    fn value<T: DeserializeOwned>(&mut self) -> Result<T, Stop> {
        Ok(serde_json::from_str(self.raw(true)?.get())?)
    }

    fn key(&mut self, expected: &str) -> Result<(), Stop> {
        let key: String = serde_json::from_str(self.raw(false)?.get())?;
        if key != expected {
            return Err(Stop::Invalid("numerical rows fields differ".into()));
        }
        self.take(b':')
    }

    fn field(&mut self, name: &str) -> Result<(), Stop> {
        self.key(name)?;
        self.output.json(&name)?;
        self.output.push(b':')?;
        Ok(())
    }

    fn document(&mut self) -> Result<(), Stop> {
        if self.context_entry {
            return self.context_entry_document();
        }
        self.punctuation(b'{', b'{')?;
        self.field("entrypoint")?;
        let entry: String = self.value()?;
        if self.context["entrypoint"].as_str() != Some(entry.as_str()) {
            return Err(Stop::Invalid(
                "numerical rows entrypoint differs from context".into(),
            ));
        }
        self.punctuation(b',', b',')?;
        self.field("definitions")?;
        self.punctuation(b'[', b'[')?;
        if self.peek()? != b']' {
            loop {
                self.definition(false)?;
                if self.peek()? != b',' {
                    break;
                }
                self.punctuation(b',', b',')?;
            }
        }
        self.punctuation(b']', b']')?;
        self.punctuation(b'}', b'}')?;
        self.whitespace()?;
        if self.offset != self.source.len() {
            return Err(Stop::Invalid("trailing numerical rows source".into()));
        }
        Ok(())
    }

    fn context_entry_document(&mut self) -> Result<(), Stop> {
        self.punctuation(b'{', b'{')?;
        if self.has_intent {
            self.key("intent")?;
            let intent: String = serde_json::from_str(self.raw(false)?.get())?;
            if !(1..=INTENT_CHARACTERS).contains(&intent.chars().count()) {
                return Err(Stop::Invalid(
                    "numerical intent must contain one through 256 Unicode scalar values".into(),
                ));
            }
            self.take(b',')?;
        }
        self.key("helpers")?;
        self.output.extend_from_slice(b"\"entrypoint\":")?;
        self.output.json(&self.context["entrypoint"])?;
        self.output.extend_from_slice(b",\"definitions\":[")?;
        self.take(b'[')?;
        let mut has_helpers = false;
        if self.peek()? != b']' {
            loop {
                self.definition(false)?;
                has_helpers = true;
                if self.peek()? != b',' {
                    break;
                }
                self.punctuation(b',', b',')?;
            }
        }
        self.take(b']')?;
        self.take(b',')?;
        self.key("entry")?;
        if has_helpers {
            self.output.push(b',')?;
        }
        self.definition(true)?;
        self.output.push(b']')?;
        self.punctuation(b'}', b'}')?;
        self.whitespace()?;
        if self.offset != self.source.len() {
            return Err(Stop::Invalid("trailing numerical rows source".into()));
        }
        Ok(())
    }

    fn definition(&mut self, entry: bool) -> Result<(), Stop> {
        self.punctuation(if self.compact_metadata { b'[' } else { b'{' }, b'{')?;
        let signature: TensorSignature = if entry {
            self.definition_name = self.context["entrypoint"].as_str().unwrap().to_owned();
            self.output.extend_from_slice(b"\"name\":")?;
            self.output.json(&self.context["entrypoint"])?;
            self.output
                .extend_from_slice(b",\"body\":{\"Numerical\":{\"signature\":")?;
            self.output.json(&self.context["signature"])?;
            self.output.push(b',')?;
            serde_json::from_value(self.context["signature"].clone())?
        } else {
            if self.compact_metadata {
                self.output.extend_from_slice(b"\"name\":")?;
                self.definition_name = self.local_symbol(true)?;
            } else {
                self.field("name")?;
                self.definition_name = self.value::<String>()?;
            }
            self.take(b',')?;
            self.output
                .extend_from_slice(b",\"body\":{\"Numerical\":{")?;
            let signature = if self.compact_metadata {
                self.output.extend_from_slice(b"\"signature\":")?;
                self.compact_signature()?
            } else {
                self.field("signature")?;
                self.value()?
            };
            self.punctuation(b',', b',')?;
            signature
        };
        if self.compact_metadata {
            self.output.extend_from_slice(b"\"parameters\":")?;
            self.compact_parameters()?;
        } else {
            self.field("parameters")?;
            self.value::<Vec<TensorParameter>>()?;
        }
        self.punctuation(b',', b',')?;
        let operation_count = if self.context_entry {
            if !self.compact_metadata {
                self.key("operation_count")?;
            }
            self.output.extend_from_slice(b"\"output\":")?;
            // Retain the original decimal until its canonical output index can
            // be projected. Subtraction does not preserve decimal prefixes.
            self.pending_operation_count = Some(self.offset);
            let count: usize = serde_json::from_str(self.raw(false)?.get())?;
            // A prefix such as `1` may still become `10`. Do not project its
            // canonical output index until the number is visibly delimited.
            if self.peek()? != b',' || !(1..=ACQUIRED_NODES).contains(&count) {
                return Err(Stop::Invalid("invalid numerical operation count".into()));
            }
            self.pending_operation_count = None;
            self.output
                .extend_from_slice((count - 1).to_string().as_bytes())?;
            Some(count)
        } else {
            self.field("output")?;
            self.value::<usize>()?;
            None
        };
        self.punctuation(b',', b',')?;
        if self.compact_metadata {
            self.output.extend_from_slice(b"\"operations\":")?;
        } else {
            self.field("operations")?;
        }
        self.punctuation(b'[', b'[')?;
        let mut operations = 0;
        if self.peek()? != b']' {
            loop {
                self.operation(operations)?;
                operations += 1;
                if self.peek()? != b',' {
                    break;
                }
                self.punctuation(b',', b',')?;
            }
        }
        self.punctuation(b']', b']')?;
        if operation_count.is_some_and(|count| count != operations) {
            return Err(Stop::Invalid(
                "numerical operation count differs from rows".into(),
            ));
        }
        self.take(if self.compact_metadata { b']' } else { b'}' })?;
        let vocabulary = self.context["vocabulary"].as_u64().unwrap() as usize;
        let max_context = self.context["max_context"].as_u64().unwrap() as usize;
        let (params, result) = FuncDef::numerical_types(&signature, vocabulary, max_context)
            .map_err(|error| Stop::Invalid(error.to_string()))?;
        self.output.extend_from_slice(
            format!(",\"vocabulary\":{vocabulary},\"max_context\":{max_context}}}}},\"params\":")
                .as_bytes(),
        )?;
        self.output.json(&params)?;
        self.output.extend_from_slice(b",\"return_type\":")?;
        self.output.json(&result)?;
        self.output.extend_from_slice(b",\"is_private\":false}")?;
        self.definition_index += 1;
        Ok(())
    }

    fn operation(&mut self, index: usize) -> Result<(), Stop> {
        self.take(b'[')?;
        self.output.extend_from_slice(b"{\"operation\":")?;
        let tag: String = self.value()?;
        let fields = self
            .fields
            .get(&tag)
            .ok_or_else(|| Stop::Invalid("unknown numerical operation tag".into()))?;
        let mut input = None;
        for field in fields.iter().skip(1) {
            self.take(b',')?;
            self.output.push(b',')?;
            self.output.json(field)?;
            self.output.push(b':')?;
            if self.compact_metadata && field == "function" {
                self.symbol(true)?;
            } else if self.compact_indices && tag == "gather_columns" && field == "input" {
                input = Some(self.value::<usize>()?);
            } else if self.compact_indices
                && tag == "gather_columns"
                && field == "indices"
                && self.peek()? == b'{'
            {
                self.structured_indices(index, input.unwrap())?;
            } else {
                self.raw(true)?;
            }
        }
        self.punctuation(b']', b'}')
    }

    fn structured_indices(&mut self, operation: usize, input: usize) -> Result<(), Stop> {
        let target = indices::Target {
            definition: self.definition_name.clone(),
            definition_index: self.definition_index,
            operation,
            input,
        };
        self.pending_indices = Some((self.offset, target.clone()));
        self.take(b'{')?;
        let mut numbers = [0usize; 4];
        for (field, name) in indices::FIELDS.into_iter().enumerate() {
            if field != 0 {
                self.take(b',')?;
            }
            self.key(name)?;
            numbers[field] = serde_json::from_str(self.raw(false)?.get())?;
        }
        self.take(b'}')?;
        let pattern = indices::Pattern::from(numbers);
        pattern
            .validate(maximum_width(self.context))
            .map_err(Stop::Invalid)?;
        pattern.write(&mut self.output)?;
        self.indices.push(indices::Record { target, pattern });
        self.pending_indices = None;
        Ok(())
    }
}
