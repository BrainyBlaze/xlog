//! Source-selected expression trees projected by the shared bounded reader.

use super::*;

pub(crate) const ENCODING: &str = "numerical-expressions-v1";

#[derive(Clone, Copy, Default)]
pub(crate) struct Nodes {
    pub spent: usize,
    pub pending_roots: usize,
    pub current_root_pending: bool,
}

/// Source-selected parent facts, separate from not-yet-emitted child values.
#[derive(Clone)]
pub(crate) struct Parent {
    pub definition_index: usize,
    pub tag_prefix: Vec<u8>,
    pub function_prefix: Option<Vec<u8>>,
    pub selected: serde_json::Map<String, Value>,
    pub operand: Option<typed::Operand>,
    pub projected: bool,
}

pub(crate) fn schema(schema: &mut Value, fields: &BTreeMap<String, Vec<String>>) {
    let defs = &mut schema["$defs"];
    for name in ["TensorOperation", "EntryTensorOperation"] {
        for operation in defs[name]["anyOf"].as_array_mut().unwrap() {
            let tag = operation["prefixItems"][0]["const"].as_str().unwrap();
            let fields = &fields[tag];
            for (position, field) in fields.iter().enumerate().skip(1) {
                let child = json!({"$ref": format!("#/$defs/{name}")});
                if is_value_reference(field) {
                    operation["prefixItems"][position] = child;
                } else if field == "arguments" {
                    operation["prefixItems"][position]["items"] = child;
                }
            }
        }
    }
    for (name, helper) in [("FuncDef", true), ("EntryFuncDef", false)] {
        let body = if helper {
            &mut defs[name]["prefixItems"][2]
        } else {
            &mut defs[name]
        };
        let operations = body["properties"]
            .as_object_mut()
            .unwrap()
            .remove("ops")
            .unwrap();
        body["properties"]["return"] = operations["items"].clone();
        body["required"] = json!(["weights", "return"]);
    }
    schema["description"] = json!(
        "Numerical-expressions-v1 acquisition. Root fields are intent,helpers,entry. Entry has exactly the ordered keys weights,return; helpers use [local_name,signature,body] with the same body. Weights retain the physical FP32 declaration/reference table. Return is one model-authored operation tuple from the canonical catalogue: value operands contain child expressions, and Call arguments contain an array of expressions. Other scalar indices, widths, symbols, token arrays and Gather indices keep their existing positions and domains. The shared native reader lowers each source node once in deterministic left-to-right postorder, without CSE, chosen operands or a supplied graph. The root is the last canonical operation. All source nodes, including duplicates and unfinished ancestors, share the existing 64-node limit; source and projected bytes remain bounded. Canonical shape, causality, physical dependencies, initialization and execution are unchanged. Ordered-structural-subsequence-v2 reserves mandatory selected syntax and ancestor closings plus future entry and paid EOS. Unknown tree structure is relaxed, never projected as a flat row sequence or assigned a token-derived row capacity. This necessary bound does not prove whole-grammar completion, original-ID fit, semantic nonconstancy or usefulness."
    );
}

impl Reader<'_> {
    pub(super) fn return_expression(&mut self, frame: usize, ending: &str) -> Result<(), Stop> {
        self.budget
            .set(frame, || vec![budget::expression(), budget::text(ending)]);
        self.key("return")?;
        self.output.extend_from_slice(b"\"operations\":[")?;
        self.expression_operations = 0;
        self.budget.set(frame, || vec![budget::text(ending)]);
        let output = self.expression()?;
        self.output.extend_from_slice(b"],\"output\":")?;
        self.output.json(&output)?;
        Ok(())
    }

    fn expression_bytes(&mut self, operation: &mut Output, bytes: &[u8]) -> Result<(), Stop> {
        operation.extend_from_slice(bytes)?;
        self.expression_pending_bytes += bytes.len();
        // Pending parents count toward the same projected byte cap even while
        // their child holes prevent serialization into the canonical prefix.
        self.output.ensure(self.expression_pending_bytes)?;
        Ok(())
    }

    fn expression_json(
        &mut self,
        operation: &mut Output,
        value: &impl Serialize,
    ) -> Result<(), Stop> {
        self.expression_bytes(operation, &serde_json::to_vec(value)?)
    }

    fn emit_expression(&mut self, operation: &Output, complete: bool) -> Result<usize, Stop> {
        let comma = usize::from(self.expression_operations != 0);
        self.output.ensure(self.expression_pending_bytes + comma)?;
        self.expression_pending_bytes -= operation.bytes.len();
        if comma != 0 {
            self.output.push(b',')?;
        }
        let start = self.output.bytes.len();
        self.output.extend_from_slice(&operation.bytes)?;
        if complete {
            self.expression_operations += 1;
        } else if self.ancestor_typing {
            self.expression_parents.last_mut().unwrap().projected = true;
        }
        Ok(start)
    }

    /// Scalar projection uses the existing readers for symbols, descriptors and
    /// JSON prefixes. A parent is exposed only when no child hole remains.
    fn expression_scalar<T>(
        &mut self,
        operation: &mut Output,
        project_partial: bool,
        minimum: budget::Unit,
        read: impl FnOnce(&mut Self) -> Result<T, Stop>,
    ) -> Result<T, Stop> {
        let value_offset = operation.bytes.len();
        let previous = std::mem::replace(
            &mut self.output,
            Output {
                bytes: Vec::new(),
                bounded: true,
            },
        );
        let result = if self.source[self.offset..].trim_ascii_start().is_empty() {
            // No byte of this selected operand has been consumed. A partial
            // RawValue instead reserves only its own unfinished delimiters.
            let frame = self.budget.enter();
            self.budget.set(frame, || vec![minimum]);
            Err(Stop::Incomplete)
        } else {
            read(self)
        };
        let scalar = std::mem::replace(&mut self.output, previous);
        let pending = self.pending_retained.take();
        self.expression_bytes(operation, &scalar.bytes)?;
        match result {
            Err(Stop::Incomplete) => {
                if project_partial {
                    let start = self.emit_expression(operation, false)?;
                    if let Some(mut reference) = pending {
                        reference.value_offset += start + value_offset;
                        self.pending_retained = Some(reference);
                    }
                }
                Err(Stop::Incomplete)
            }
            result => result,
        }
    }

    fn expression(&mut self) -> Result<usize, Stop> {
        let frame = self.budget.enter();
        self.budget.set(frame, || vec![budget::text("\"\",]")]);
        self.take(b'[')?;
        self.expression_nodes += 1;
        if self.expression_root_pending {
            self.expression_roots -= 1;
            self.expression_root_pending = false;
        }
        if self.expression_nodes > ACQUIRED_NODES {
            return Err(Stop::Invalid(
                "expression nodes exceed the acquisition node allowance".into(),
            ));
        }
        let mut operation = Output {
            bytes: Vec::new(),
            bounded: true,
        };
        self.expression_bytes(&mut operation, b"{\"operation\":")?;
        let tag_start = operation.bytes.len() - 1;
        let parent = self.expression_parents.len();
        if self.parent_prerequisites {
            self.expression_parents.push(Parent {
                definition_index: self.definition_index,
                tag_prefix: Vec::new(),
                function_prefix: None,
                selected: serde_json::Map::new(),
                operand: None,
                projected: false,
            });
        }
        self.budget.set(frame, || vec![budget::text(",]")]);
        let tag = self.expression_scalar(&mut operation, false, budget::text("\"\""), |reader| {
            reader.value::<String>()
        });
        if self.parent_prerequisites {
            // Keep the canonical colon/string prefix for the existing tag
            // matcher, including incomplete and JSON-escaped spellings.
            self.expression_parents[parent].tag_prefix = operation.bytes[tag_start..].to_vec();
        }
        let tag = tag?;
        if self.ancestor_typing {
            self.expression_parents[parent]
                .selected
                .insert("operation".into(), json!(tag));
        }
        let fields = self
            .fields
            .get(&tag)
            .ok_or_else(|| Stop::Invalid("unknown numerical operation tag".into()))?;
        let mut input = None;
        for (position, field) in fields.iter().enumerate().skip(1) {
            let following = || {
                let mut units = Vec::new();
                for field in fields.iter().skip(position + 1) {
                    units.push(budget::text(","));
                    units.push(budget::expression_field(field));
                }
                units.push(budget::text("]"));
                units
            };
            self.budget.set(frame, || {
                let mut units = vec![budget::expression_field(field)];
                units.extend(following());
                units
            });
            self.take(b',')?;
            self.expression_bytes(&mut operation, b",")?;
            self.expression_json(&mut operation, field)?;
            self.expression_bytes(&mut operation, b":")?;
            self.budget.set(frame, following);
            if is_value_reference(field) {
                if self.ancestor_typing {
                    self.expression_parents[parent].operand =
                        Some(typed::Operand::Value(field.clone()));
                }
                let id = self.expression()?;
                if self.ancestor_typing {
                    self.expression_parents[parent].operand = None;
                    self.expression_parents[parent]
                        .selected
                        .insert(field.clone(), json!(id));
                }
                if field == "input" {
                    input = Some(id);
                }
                self.expression_json(&mut operation, &id)?;
            } else if field == "arguments" {
                let arguments = self.expression_arguments(parent)?;
                self.expression_json(&mut operation, &arguments)?;
            } else {
                let project_partial = !fields
                    .iter()
                    .skip(position + 1)
                    .any(|field| is_value_reference(field) || field == "arguments");
                if field == "function" {
                    let start = operation.bytes.len() - 1;
                    let function = self.expression_scalar(
                        &mut operation,
                        project_partial,
                        budget::expression_field(field),
                        |reader| reader.symbol(true),
                    );
                    if self.parent_prerequisites {
                        self.expression_parents[parent].function_prefix =
                            Some(operation.bytes[start..].to_vec());
                    }
                    let function = function?;
                    if self.ancestor_typing {
                        self.expression_parents[parent]
                            .selected
                            .insert(field.clone(), json!(function));
                    }
                } else {
                    let index = self.expression_operations;
                    let value = self.expression_scalar(
                        &mut operation,
                        project_partial,
                        budget::expression_field(field),
                        |reader| {
                            if field == "indices" && reader.peek()? == b'{' {
                                reader
                                    .structured_indices(index, input.unwrap())
                                    .map(|_| None)
                            } else {
                                reader.value::<Value>().map(Some)
                            }
                        },
                    )?;
                    // A number at EOF may still extend. NumberChoice, rather
                    // than this snapshot, owns its prospective decimal domain.
                    if let Some(value) = value.filter(|value| {
                        self.ancestor_typing
                            && (!value.is_number() || self.offset < self.source.len())
                    }) {
                        self.expression_parents[parent]
                            .selected
                            .insert(field.clone(), value);
                    }
                }
            }
        }
        self.budget.set(frame, || vec![]);
        match self.take(b']') {
            Ok(()) => {}
            Err(Stop::Incomplete) => {
                self.emit_expression(&operation, false)?;
                return Err(Stop::Incomplete);
            }
            Err(error) => return Err(error),
        }
        self.expression_bytes(&mut operation, b"}")?;
        let id = self.expression_operations;
        self.emit_expression(&operation, true)?;
        if self.parent_prerequisites {
            self.expression_parents.pop();
        }
        self.budget.leave(frame);
        Ok(id)
    }

    fn expression_arguments(&mut self, parent: usize) -> Result<Vec<usize>, Stop> {
        let frame = self.budget.enter();
        self.budget.set(frame, || vec![budget::text("]")]);
        self.take(b'[')?;
        let mut arguments = Vec::new();
        if self.ancestor_typing {
            self.expression_parents[parent]
                .selected
                .insert("arguments".into(), json!([]));
        }
        if self.peek()? != b']' {
            loop {
                if self.ancestor_typing {
                    self.expression_parents[parent].operand =
                        Some(typed::Operand::Argument(arguments.len()));
                }
                arguments.push(self.expression()?);
                if self.ancestor_typing {
                    self.expression_parents[parent].operand = None;
                    self.expression_parents[parent]
                        .selected
                        .insert("arguments".into(), json!(arguments));
                }
                if self.peek()? != b',' {
                    break;
                }
                self.take(b',')?;
            }
        }
        self.budget.set(frame, || vec![]);
        self.take(b']')?;
        self.budget.leave(frame);
        Ok(arguments)
    }
}
