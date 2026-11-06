//! Positional numerical source syntax, elaborated into the canonical definition wire.

use super::*;
use serde::de::DeserializeOwned;
use serde_json::value::RawValue;
use xlog_logic::ast::FuncBody;

pub(super) const ENCODING: &str = "numerical-rows-v1";
pub(super) const CONTEXT_ENTRY_ENCODING: &str = "numerical-rows-v2";

/// Derive positional rows from the existing specialized canonical operation
/// schema. Its required-field lists own both the operand order and exact arity.
pub(super) fn schema(schema: &mut Value, context_entry: bool) {
    let defs = &mut schema["$defs"];
    for name in ["TensorOperation", "EntryTensorOperation"] {
        let mut rows = Vec::new();
        for operation in defs[name]["oneOf"].as_array_mut().unwrap() {
            let fields = operation["required"].as_array().unwrap();
            let items = fields
                .iter()
                .map(|field| operation["properties"][field.as_str().unwrap()].clone())
                .collect::<Vec<_>>();
            rows.push(json!({"type": "array", "minItems": items.len(),
                "maxItems": items.len(), "prefixItems": items, "items": false}));
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
}

pub(super) struct Prefix {
    pub bytes: Vec<u8>,
    pub complete: bool,
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

struct Reader<'a> {
    source: &'a [u8],
    offset: usize,
    output: Vec<u8>,
    context: &'a Value,
    fields: &'a BTreeMap<String, Vec<String>>,
    context_entry: bool,
}

/// Only syntax is expanded. An unfinished operand remains the same byte prefix
/// for the existing resource/type owner, including strings, decimals and arrays.
pub(super) fn prefix(
    source: &[u8],
    context: &Value,
    fields: &BTreeMap<String, Vec<String>>,
    context_entry: bool,
) -> Result<Prefix, String> {
    if source.len() > ACQUIRED_SOURCE_BYTES {
        return Err("acquisition source byte budget exceeded".into());
    }
    let mut reader = Reader {
        source,
        offset: 0,
        output: Vec::with_capacity(source.len()),
        context,
        fields,
        context_entry,
    };
    let complete = match reader.document() {
        Ok(()) => true,
        Err(Stop::Incomplete) => false,
        Err(Stop::Invalid(error)) => return Err(error),
    };
    Ok(Prefix {
        bytes: reader.output,
        complete,
    })
}

/// Decode the same syntax used by prefix admission, then construct definitions
/// through their canonical ABI owner. No operation or parameter is selected here.
pub(super) fn definitions(
    source: &[u8],
    context: &Value,
    fields: &BTreeMap<String, Vec<String>>,
    context_entry: bool,
) -> Result<(String, Vec<FuncDef>), String> {
    let decoded = prefix(source, context, fields, context_entry)?;
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
    Ok((
        context["entrypoint"].as_str().unwrap().to_owned(),
        definitions,
    ))
}

impl Reader<'_> {
    fn whitespace(&mut self) {
        while self
            .source
            .get(self.offset)
            .is_some_and(u8::is_ascii_whitespace)
        {
            self.output.push(self.source[self.offset]);
            self.offset += 1;
        }
    }

    fn peek(&mut self) -> Result<u8, Stop> {
        self.whitespace();
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
        self.output.push(canonical);
        Ok(())
    }

    fn raw(&mut self, copy: bool) -> Result<&RawValue, Stop> {
        self.whitespace();
        let tail = &self.source[self.offset..];
        let mut values = serde_json::Deserializer::from_slice(tail).into_iter::<&RawValue>();
        let value = match values.next() {
            Some(Ok(value)) => value,
            None => return Err(Stop::Incomplete),
            Some(Err(error)) if error.is_eof() => {
                if copy {
                    self.output.extend_from_slice(tail);
                }
                return Err(Stop::Incomplete);
            }
            Some(Err(error)) => return Err(error.into()),
        };
        self.offset += values.byte_offset();
        if copy {
            self.output.extend_from_slice(value.get().as_bytes());
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
        self.output.extend_from_slice(&serde_json::to_vec(name)?);
        self.output.push(b':');
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
        self.whitespace();
        if self.offset != self.source.len() {
            return Err(Stop::Invalid("trailing numerical rows source".into()));
        }
        Ok(())
    }

    fn context_entry_document(&mut self) -> Result<(), Stop> {
        self.punctuation(b'{', b'{')?;
        self.key("helpers")?;
        self.output.extend_from_slice(b"\"entrypoint\":");
        self.output
            .extend_from_slice(&serde_json::to_vec(&self.context["entrypoint"])?);
        self.output.extend_from_slice(b",\"definitions\":[");
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
            self.output.push(b',');
        }
        self.definition(true)?;
        self.output.push(b']');
        self.punctuation(b'}', b'}')?;
        self.whitespace();
        if self.offset != self.source.len() {
            return Err(Stop::Invalid("trailing numerical rows source".into()));
        }
        Ok(())
    }

    fn definition(&mut self, entry: bool) -> Result<(), Stop> {
        self.punctuation(b'{', b'{')?;
        let signature: TensorSignature = if entry {
            self.output.extend_from_slice(b"\"name\":");
            self.output
                .extend_from_slice(&serde_json::to_vec(&self.context["entrypoint"])?);
            self.output
                .extend_from_slice(b",\"body\":{\"Numerical\":{\"signature\":");
            self.output
                .extend_from_slice(&serde_json::to_vec(&self.context["signature"])?);
            self.output.push(b',');
            serde_json::from_value(self.context["signature"].clone())?
        } else {
            self.field("name")?;
            self.value::<String>()?;
            self.take(b',')?;
            self.output.extend_from_slice(b",\"body\":{\"Numerical\":{");
            self.field("signature")?;
            let signature = self.value()?;
            self.punctuation(b',', b',')?;
            signature
        };
        self.field("parameters")?;
        self.value::<Vec<TensorParameter>>()?;
        self.punctuation(b',', b',')?;
        let operation_count = if self.context_entry {
            self.key("operation_count")?;
            self.output.extend_from_slice(b"\"output\":");
            let count: usize = serde_json::from_str(self.raw(false)?.get())?;
            // A prefix such as `1` may still become `10`. Do not project its
            // canonical output index until the number is visibly delimited.
            if self.peek()? != b',' || !(1..=ACQUIRED_NODES).contains(&count) {
                return Err(Stop::Invalid("invalid numerical operation count".into()));
            }
            self.output
                .extend_from_slice((count - 1).to_string().as_bytes());
            Some(count)
        } else {
            self.field("output")?;
            self.value::<usize>()?;
            None
        };
        self.punctuation(b',', b',')?;
        self.field("operations")?;
        self.punctuation(b'[', b'[')?;
        let mut operations = 0;
        if self.peek()? != b']' {
            loop {
                self.operation()?;
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
        self.take(b'}')?;
        let vocabulary = self.context["vocabulary"].as_u64().unwrap() as usize;
        let max_context = self.context["max_context"].as_u64().unwrap() as usize;
        let (params, result) = FuncDef::numerical_types(&signature, vocabulary, max_context)
            .map_err(|error| Stop::Invalid(error.to_string()))?;
        self.output.extend_from_slice(
            format!(",\"vocabulary\":{vocabulary},\"max_context\":{max_context}}}}},\"params\":")
                .as_bytes(),
        );
        self.output.extend_from_slice(&serde_json::to_vec(&params)?);
        self.output.extend_from_slice(b",\"return_type\":");
        self.output.extend_from_slice(&serde_json::to_vec(&result)?);
        self.output.extend_from_slice(b",\"is_private\":false}");
        Ok(())
    }

    fn operation(&mut self) -> Result<(), Stop> {
        self.take(b'[')?;
        self.output.extend_from_slice(b"{\"operation\":");
        let tag: String = self.value()?;
        let fields = self
            .fields
            .get(&tag)
            .ok_or_else(|| Stop::Invalid("unknown numerical operation tag".into()))?;
        for field in fields.iter().skip(1) {
            self.take(b',')?;
            self.output.push(b',');
            self.output.extend_from_slice(&serde_json::to_vec(field)?);
            self.output.push(b':');
            self.raw(true)?;
        }
        self.punctuation(b']', b'}')
    }
}
