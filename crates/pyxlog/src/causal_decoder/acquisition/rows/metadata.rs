//! Positional declarations and source-selected symbols in the shared rows codec.

use super::*;

const RETAINED: &str = "retained";

/// Physical declarations shared by source elaboration and prefix admission.
pub(crate) struct Declarations(BTreeMap<String, TensorParameter>);

impl Declarations {
    pub(crate) fn new(context: Option<&Value>) -> Self {
        Self(context.map_or_else(BTreeMap::new, |context| {
            serde_json::from_value::<Vec<TensorParameter>>(context["parameters"].clone())
                .unwrap()
                .into_iter()
                .map(|parameter| (parameter.name.clone(), parameter))
                .collect()
        }))
    }

    pub(crate) fn get(&self, name: &str) -> Option<&TensorParameter> {
        self.0.get(name)
    }

    pub(crate) fn keys(&self) -> impl Iterator<Item = &String> {
        self.0.keys()
    }

    pub(crate) fn record(&mut self, parameter: TensorParameter) -> Result<(), String> {
        if self
            .get(&parameter.name)
            .is_some_and(|old| old != &parameter)
        {
            return Err("parameter alias shape differs from its declaration".into());
        }
        self.0.insert(parameter.name.clone(), parameter);
        Ok(())
    }
}

pub(crate) fn is_local(value: &str) -> bool {
    let bytes = value.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= 64
        && (bytes[0].is_ascii_alphabetic() || bytes[0] == b'_')
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
}

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

pub(crate) fn schema(schema: &mut Value, context: &Value, self_delimited: bool) {
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
    let declaration = tuple(vec![
        parameter.clone(),
        defs["TensorParameter"]["anyOf"][0]["properties"]["shape"].clone(),
    ]);
    defs["TensorParameter"] = if self_delimited {
        json!({"anyOf": [declaration, tuple(vec![parameter])]})
    } else {
        declaration
    };
    let helper = &defs["FuncDef"]["properties"];
    let parameters = helper["parameters"].clone();
    let count = helper["operation_count"].clone();
    let operations = helper["operations"].clone();
    let entry_operations = defs["EntryFuncDef"]["properties"]["operations"].clone();
    let mut helper = vec![
        local(),
        json!({"$ref": "#/$defs/TensorSignature"}),
        parameters.clone(),
    ];
    let mut entry = vec![parameters];
    if !self_delimited {
        helper.push(count.clone());
        entry.push(count);
    }
    helper.push(operations);
    entry.push(entry_operations);
    defs["FuncDef"] = tuple(helper);
    defs["EntryFuncDef"] = tuple(entry);
    if self_delimited {
        schema["description"] = json!(
            "Compact numerical-rows-v6 acquisition. Root fields are intent,helpers,entry. Helpers use [local_name,signature,parameters,operations]; entry uses [parameters,operations]. Operations are nonempty and the last actual row is the output. Signatures use [kind,input_width_or_widths,output_width]. Parameters use [symbol,shape] to declare or check named FP32 storage, or [symbol] to reference an already existing physical declaration without changing its shape. Local names expand with the captured namespace; retained symbols use [\"retained\",full_name]. The native typed owner checks prospective rows within the cumulative row allowance without selecting an output. Closure checks the actual last row and output-reaching dependencies. Original-ID suffix coverage remains a necessary relaxation, not a completion or usefulness guarantee."
        );
        return;
    }
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
        if !is_local(&local) {
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
        let frame = self.budget.enter();
        self.budget.set(frame, || vec![budget::text("\"\",\"\"]")]);
        self.pending_retained = Some(PendingReference {
            function,
            value_offset,
        });
        self.take(b'[')?;
        self.budget.set(frame, || vec![budget::text(",\"\"]")]);
        let tag: String = serde_json::from_str(self.raw(false)?.get())?;
        if tag != RETAINED {
            return Err(Stop::Invalid("unknown acquisition symbol reference".into()));
        }
        self.budget.set(frame, || vec![budget::text("\"\"]")]);
        self.take(b',')?;
        self.budget.set(frame, || vec![budget::text("]")]);
        let name = self.value::<String>()?;
        let retained = retained_names(self.context, function).contains(&name.as_str());
        if !retained {
            return Err(Stop::Invalid(
                "symbol is absent from the retained native context".into(),
            ));
        }
        self.budget.set(frame, || vec![]);
        self.take(b']')?;
        self.pending_retained = None;
        self.budget.leave(frame);
        Ok(name)
    }

    pub(super) fn compact_signature(&mut self) -> Result<TensorSignature, Stop> {
        let frame = self.budget.enter();
        self.budget.set(frame, || vec![budget::text("\"\",,1]")]);
        self.punctuation(b'[', b'{')?;
        self.output.extend_from_slice(b"\"kind\":")?;
        self.budget.set(frame, || vec![budget::text(",,1]")]);
        let kind = self.value::<String>()?;
        let input = match kind.as_str() {
            "pure" => "input_widths",
            "causal" => "input_width",
            _ => return Err(Stop::Invalid("unknown numerical signature kind".into())),
        };
        self.budget.set(frame, || {
            vec![budget::text(if input == "input_widths" {
                "[],1]"
            } else {
                "1,1]"
            })]
        });
        self.punctuation(b',', b',')?;
        self.output.json(&input)?;
        self.output.push(b':')?;
        self.budget.set(frame, || vec![budget::text(",1]")]);
        let signature = if input == "input_widths" {
            let input_widths = self.value::<Vec<usize>>()?;
            self.budget.set(frame, || vec![budget::text("1]")]);
            self.punctuation(b',', b',')?;
            self.output.extend_from_slice(b"\"output_width\":")?;
            self.budget.set(frame, || vec![budget::text("]")]);
            TensorSignature::Pure {
                input_widths,
                output_width: self.value()?,
            }
        } else {
            let input_width = self.value::<usize>()?;
            self.budget.set(frame, || vec![budget::text("1]")]);
            self.punctuation(b',', b',')?;
            self.output.extend_from_slice(b"\"output_width\":")?;
            self.budget.set(frame, || vec![budget::text("]")]);
            TensorSignature::Causal {
                input_width,
                output_width: self.value()?,
            }
        };
        self.budget.set(frame, || vec![]);
        self.punctuation(b']', b'}')?;
        self.budget.leave(frame);
        Ok(signature)
    }

    pub(super) fn compact_parameters(&mut self) -> Result<(), Stop> {
        let frame = self.budget.enter();
        self.budget.set(frame, || {
            vec![
                budget::Unit::Parameters {
                    active: false,
                    leading_comma: false,
                },
                budget::text("]"),
            ]
        });
        self.punctuation(b'[', b'[')?;
        if self.peek()? != b']' {
            loop {
                self.budget.set(frame, || {
                    vec![
                        budget::Unit::Parameters {
                            active: true,
                            leading_comma: true,
                        },
                        budget::text("]"),
                    ]
                });
                let parameter = self.budget.enter();
                self.budget.set(parameter, || {
                    vec![budget::text(if self.self_delimited {
                        "\"\"]"
                    } else {
                        "\"\",[]]"
                    })]
                });
                self.take(b'[')?;
                self.output.extend_from_slice(b"{\"name\":")?;
                self.budget.set(parameter, || {
                    vec![budget::text(if self.self_delimited { "]" } else { ",[]]" })]
                });
                let name = self.symbol(false)?;
                let reference = self.self_delimited && self.peek()? == b']';
                let shape = if reference {
                    // The closing tuple delimiter selects a reference. Before
                    // it, an unknown name can still introduce an explicit shape.
                    let shape = self
                        .declarations
                        .as_ref()
                        .unwrap()
                        .get(&name)
                        .ok_or_else(|| Stop::Invalid("parameter reference is not declared".into()))?
                        .shape
                        .clone();
                    self.output.extend_from_slice(b",\"shape\":")?;
                    self.output.json(&shape)?;
                    shape
                } else {
                    self.budget.set(parameter, || vec![budget::text("[]]")]);
                    self.punctuation(b',', b',')?;
                    self.output.extend_from_slice(b"\"shape\":")?;
                    self.budget.set(parameter, || vec![budget::text("]")]);
                    self.value::<Vec<usize>>()?
                };
                self.budget.set(parameter, || vec![]);
                self.take(b']')?;
                if let Some(declarations) = &mut self.declarations {
                    declarations
                        .record(TensorParameter {
                            name,
                            shape,
                            float32: true,
                        })
                        .map_err(Stop::Invalid)?;
                }
                self.output.extend_from_slice(b",\"float32\":true}")?;
                self.budget.leave(parameter);
                self.budget.set(frame, || {
                    vec![
                        budget::Unit::Parameters {
                            active: false,
                            leading_comma: true,
                        },
                        budget::text("]"),
                    ]
                });
                if self.peek()? != b',' {
                    break;
                }
                self.punctuation(b',', b',')?;
            }
        }
        self.budget.set(frame, || vec![]);
        self.punctuation(b']', b']')?;
        self.budget.leave(frame);
        Ok(())
    }
}
