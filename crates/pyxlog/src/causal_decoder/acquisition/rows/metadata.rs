//! Positional declarations and source-selected symbols in the shared rows codec.

use super::*;

const RETAINED: &str = "retained";

pub(crate) struct PendingReference {
    pub function: bool,
    /// Position immediately after the canonical name/function field's colon.
    pub value_offset: usize,
}

pub(crate) fn retained_names(context: &Value, function: bool) -> Vec<&str> {
    if function {
        context["callees"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect()
    } else {
        context["parameters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|parameter| parameter["name"].as_str().unwrap())
            .collect()
    }
}

fn local() -> Value {
    json!({"type": "string", "pattern": "^[a-zA-Z_][a-zA-Z0-9_]{0,63}$"})
}

fn reference(names: Vec<Value>) -> Value {
    if names.is_empty() {
        local()
    } else {
        json!({"anyOf": [local(), tuple(vec![json!({"const": RETAINED}),
            json!({"type": "string", "enum": names})])]})
    }
}

pub(crate) fn schema(schema: &mut Value, context: &Value) {
    let defs = &mut schema["$defs"];
    defs["Callee"] = reference(
        retained_names(context, true)
            .into_iter()
            .map(|name| json!(name))
            .collect(),
    );
    let parameter = reference(
        retained_names(context, false)
            .into_iter()
            .map(|name| json!(name))
            .collect(),
    );
    let mut signatures = Vec::new();
    for signature in defs["TensorSignature"]["oneOf"].as_array().unwrap() {
        let fields = &signature["properties"];
        let input = if fields["kind"]["const"] == "pure" {
            "input_widths"
        } else {
            "input_width"
        };
        signatures.push(tuple(vec![
            fields["kind"].clone(),
            fields[input].clone(),
            fields["output_width"].clone(),
        ]));
    }
    defs["TensorSignature"] = json!({"anyOf": signatures});
    defs["TensorParameter"] = tuple(vec![
        parameter,
        defs["TensorParameter"]["anyOf"][0]["properties"]["shape"].clone(),
    ]);
    let helper = &defs["FuncDef"]["properties"];
    let parameters = helper["parameters"].clone();
    let count = helper["operation_count"].clone();
    let operations = helper["operations"].clone();
    let entry_operations = defs["EntryFuncDef"]["properties"]["operations"].clone();
    defs["FuncDef"] = tuple(vec![
        local(),
        json!({"$ref": "#/$defs/TensorSignature"}),
        parameters.clone(),
        count.clone(),
        operations,
    ]);
    defs["EntryFuncDef"] = tuple(vec![parameters, count, entry_operations]);
    schema["description"] = json!(format!(
        "{} Helpers use [local_name,signature,parameters,operation_count,operations]; entry uses [parameters,operation_count,operations]. Signatures are [\"pure\",input_widths,output_width] or [\"causal\",input_width,output_width]. Parameters use [reference,shape] with canonical float32=true. Local ASCII identifier strings expand with the captured function or parameter namespace. Call/Scan callees and parameter declarations may instead use [\"retained\",full_name] naming an actual captured callee or physical parameter. The same reader projects partial symbols before canonical prefix checks and bounds original and projected source. All operations, references and shapes remain selected in source; no token-budget completion or usefulness is guaranteed.",
        schema["description"].as_str().unwrap().replace("numerical-rows-v4", "numerical-rows-v5")
    ));
}

impl Reader<'_> {
    pub(super) fn local_symbol(&mut self, function: bool) -> Result<String, Stop> {
        if self.peek()? != b'"' {
            return Err(Stop::Invalid("local symbol must be a string".into()));
        }
        let context = self.context;
        let namespace = context[if function {
            "function_prefix"
        } else {
            "parameter_prefix"
        }]
        .as_str()
        .unwrap();
        let prefix = serde_json::to_vec(namespace)?;
        self.output.extend_from_slice(&prefix[..prefix.len() - 1])?;
        let start = self.offset;
        let decoded = self
            .raw(false)
            .and_then(|value| serde_json::from_str::<String>(value.get()).map_err(Stop::from));
        let local = match decoded {
            Ok(value) => value,
            Err(Stop::Incomplete) => {
                // Namespace concatenation preserves every partial string
                // spelling, including an unfinished JSON escape. Do not wait
                // for the tuple or symbol to close before prefix admission.
                self.output.extend_from_slice(&self.source[start + 1..])?;
                return Err(Stop::Incomplete);
            }
            Err(error) => return Err(error),
        };
        let bytes = local.as_bytes();
        if bytes.is_empty()
            || bytes.len() > 64
            || !(bytes[0].is_ascii_alphabetic() || bytes[0] == b'_')
            || !bytes
                .iter()
                .all(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
        {
            return Err(Stop::Invalid("invalid local acquisition symbol".into()));
        }
        self.output
            .extend_from_slice(&self.source[start + 1..self.offset])?;
        Ok(format!("{namespace}{local}"))
    }

    pub(super) fn symbol(&mut self, function: bool) -> Result<String, Stop> {
        let value_offset = self.output.bytes.len();
        if self.peek()? == b'"' {
            return self.local_symbol(function);
        }
        self.pending_retained = Some(PendingReference {
            function,
            value_offset,
        });
        self.take(b'[')?;
        let tag: String = serde_json::from_str(self.raw(false)?.get())?;
        if tag != RETAINED {
            return Err(Stop::Invalid("unknown acquisition symbol reference".into()));
        }
        self.take(b',')?;
        let name = self.value::<String>()?;
        let retained = retained_names(self.context, function).contains(&name.as_str());
        if !retained {
            return Err(Stop::Invalid(
                "symbol is absent from the retained native context".into(),
            ));
        }
        self.take(b']')?;
        self.pending_retained = None;
        Ok(name)
    }

    pub(super) fn compact_signature(&mut self) -> Result<TensorSignature, Stop> {
        self.punctuation(b'[', b'{')?;
        self.output.extend_from_slice(b"\"kind\":")?;
        let kind = self.value::<String>()?;
        let input = match kind.as_str() {
            "pure" => "input_widths",
            "causal" => "input_width",
            _ => return Err(Stop::Invalid("unknown numerical signature kind".into())),
        };
        self.punctuation(b',', b',')?;
        self.output.json(&input)?;
        self.output.push(b':')?;
        let signature = if input == "input_widths" {
            let input_widths = self.value::<Vec<usize>>()?;
            self.punctuation(b',', b',')?;
            self.output.extend_from_slice(b"\"output_width\":")?;
            TensorSignature::Pure {
                input_widths,
                output_width: self.value()?,
            }
        } else {
            let input_width = self.value::<usize>()?;
            self.punctuation(b',', b',')?;
            self.output.extend_from_slice(b"\"output_width\":")?;
            TensorSignature::Causal {
                input_width,
                output_width: self.value()?,
            }
        };
        self.punctuation(b']', b'}')?;
        Ok(signature)
    }

    pub(super) fn compact_parameters(&mut self) -> Result<(), Stop> {
        self.punctuation(b'[', b'[')?;
        if self.peek()? != b']' {
            loop {
                self.take(b'[')?;
                self.output.extend_from_slice(b"{\"name\":")?;
                self.symbol(false)?;
                self.punctuation(b',', b',')?;
                self.output.extend_from_slice(b"\"shape\":")?;
                self.value::<Vec<usize>>()?;
                self.take(b']')?;
                self.output.extend_from_slice(b",\"float32\":true}")?;
                if self.peek()? != b',' {
                    break;
                }
                self.punctuation(b',', b',')?;
            }
        }
        self.punctuation(b']', b']')
    }
}
