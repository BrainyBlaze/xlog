//! Prospective operand choices use the executable compiler's shape inference.

use std::collections::BTreeMap;

use serde_json::{Map, Value};
use xlog_core::XlogError;
use xlog_ir::tensor::{TensorOperation as Op, TensorProgram, TensorSignature, TensorValueShape};

pub(super) type Signatures = BTreeMap<String, TensorSignature>;

pub(super) struct Program {
    pub(super) body: TensorProgram,
    shapes: [Vec<TensorValueShape>; 2],
    first_raw_token: u32,
}

impl Program {
    pub(super) fn new(
        signature: TensorSignature,
        vocabulary: usize,
        max_context: usize,
        first_raw_token: u32,
    ) -> Self {
        Self {
            body: TensorProgram {
                signature,
                parameters: Vec::new(),
                operations: Vec::new(),
                output: 0,
                vocabulary,
                max_context,
            },
            shapes: [Vec::new(), Vec::new()],
            first_raw_token,
        }
    }

    fn infer(
        &self,
        operation: &Op,
        signatures: &Signatures,
        length: usize,
    ) -> Result<TensorValueShape, XlogError> {
        let tokens = [1, self.body.max_context][length];
        self.body.operation_shape_with(
            operation,
            tokens,
            &|id| {
                self.shapes[length].get(id).copied().ok_or_else(|| {
                    XlogError::Compilation("numerical value is not preceding".into())
                })
            },
            &|name| {
                signatures
                    .get(name)
                    .cloned()
                    .ok_or_else(|| XlogError::Compilation("callee must precede its caller".into()))
            },
        )
    }

    pub(super) fn push(&mut self, operation: Op, signatures: &Signatures) -> Result<(), String> {
        let first = self
            .infer(&operation, signatures, 0)
            .map_err(|error| error.to_string())?;
        let last = self
            .infer(&operation, signatures, 1)
            .map_err(|error| error.to_string())?;
        self.body.operations.push(operation);
        self.body
            .validate_acquired_body()
            .map_err(|error| error.to_string())?;
        self.shapes[0].push(first);
        self.shapes[1].push(last);
        Ok(())
    }

    pub(super) fn outputs(&self) -> Vec<usize> {
        (0..self.body.operations.len())
            .filter(|&id| {
                [1, self.body.max_context]
                    .into_iter()
                    .enumerate()
                    .all(|(length, tokens)| {
                        self.body
                            .signature
                            .output(tokens, self.body.vocabulary)
                            .ok()
                            == Some(self.shapes[length][id])
                    })
            })
            .collect()
    }

    fn compatible(&self, operation: Op, signatures: &Signatures) -> bool {
        // The same causal-dependency check used by the complete compiler applies
        // to this prospective operation. No prospective graph is ever emitted.
        let mut body = self.body.clone();
        body.operations.push(operation.clone());
        if body.validate_acquired_body().is_err() {
            return false;
        }
        (0..2).all(|length| self.infer(&operation, signatures, length).is_ok())
    }

    fn references(&self, fields: &Map<String, Value>, name: &str, extent: usize) -> Vec<usize> {
        match fields.get(name).and_then(Value::as_u64) {
            Some(id) => vec![id as usize],
            None => (0..extent).collect(),
        }
    }

    pub(super) fn numbers(
        &self,
        fields: &Map<String, Value>,
        name: &str,
        extent: usize,
        signatures: &Signatures,
    ) -> Vec<usize> {
        let mut candidate = fields.clone();
        (0..extent)
            .filter(|&id| {
                candidate.insert(name.into(), Value::from(id));
                self.possible(&candidate, None, signatures)
            })
            .collect()
    }

    pub(super) fn array_numbers(
        &self,
        fields: &Map<String, Value>,
        name: &str,
        prefix: &[usize],
        signatures: &Signatures,
    ) -> Vec<usize> {
        if name == "indices" {
            let id = fields["input"].as_u64().unwrap() as usize;
            return (0..self
                .shapes
                .iter()
                .map(|shapes| shapes[id].columns)
                .min()
                .unwrap())
                .collect();
        }
        let extent = self.body.operations.len();
        let mut candidate = fields.clone();
        let mut values = prefix.to_vec();
        values.push(0);
        (0..extent)
            .filter(|&id| {
                *values.last_mut().unwrap() = id;
                candidate.insert(name.into(), serde_json::json!(values));
                self.possible(&candidate, Some(name), signatures)
            })
            .collect()
    }

    pub(super) fn array_can_end(
        &self,
        fields: &Map<String, Value>,
        name: &str,
        prefix: &[usize],
        signatures: &Signatures,
    ) -> bool {
        let mut candidate = fields.clone();
        candidate.insert(name.into(), serde_json::json!(prefix));
        self.possible(&candidate, None, signatures)
    }

    pub(super) fn possible(
        &self,
        fields: &Map<String, Value>,
        open_array: Option<&str>,
        signatures: &Signatures,
    ) -> bool {
        let Some(tag) = fields.get("operation").and_then(Value::as_str) else {
            return false;
        };
        let count = self.body.operations.len();
        let refs = |name| self.references(fields, name, count);
        let scalar = |name: &str, first: usize| {
            fields
                .get(name)
                .and_then(Value::as_u64)
                .map_or(first, |value| value as usize)
        };
        let array = |name: &str| {
            fields.get(name).and_then(Value::as_array).map(|values| {
                values
                    .iter()
                    .map(|value| value.as_u64().unwrap() as usize)
                    .collect::<Vec<_>>()
            })
        };
        let check = |operation| self.compatible(operation, signatures);
        match tag {
            "zeros" => check(Op::Zeros {
                columns: scalar("columns", 1),
            }),
            "argument" => self
                .references(
                    fields,
                    "argument",
                    self.body.signature.arguments(1).unwrap().len(),
                )
                .into_iter()
                .any(|argument| check(Op::Argument { argument })),
            "parameter" => self
                .references(fields, "weight", self.body.parameters.len())
                .into_iter()
                .any(|weight| check(Op::Parameter { weight })),
            "token_indicators" => check(Op::TokenIndicators {
                // A nonempty list is a possible continuation until its actual
                // IDs arrive. The separate raw-alphabet owner checks those IDs.
                token_ids: array("token_ids")
                    .unwrap_or_else(|| vec![self.first_raw_token as usize])
                    .into_iter()
                    .map(|id| id as u32)
                    .collect(),
            }),
            "broadcast" => refs("input")
                .into_iter()
                .any(|input| check(Op::Broadcast { input })),
            "tanh" => refs("input")
                .into_iter()
                .any(|input| check(Op::Tanh { input })),
            "linear" => refs("input").into_iter().any(|input| {
                self.references(fields, "weight", self.body.parameters.len())
                    .into_iter()
                    .any(|weight| check(Op::Linear { input, weight }))
            }),
            "add" | "subtract" | "multiply" | "divide" => refs("left").into_iter().any(|left| {
                refs("right").into_iter().any(|right| {
                    check(match tag {
                        "add" => Op::Add { left, right },
                        "subtract" => Op::Subtract { left, right },
                        "multiply" => Op::Multiply { left, right },
                        _ => Op::Divide { left, right },
                    })
                })
            }),
            "select" => refs("condition").into_iter().any(|condition| {
                refs("positive").into_iter().any(|positive| {
                    refs("negative").into_iter().any(|negative| {
                        check(Op::Select {
                            condition,
                            positive,
                            negative,
                        })
                    })
                })
            }),
            "gather_columns" => refs("input").into_iter().any(|input| {
                let mut indices = array("indices").unwrap_or_else(|| vec![0]);
                if indices.is_empty() && open_array == Some("indices") {
                    indices.push(0);
                }
                check(Op::GatherColumns { input, indices })
            }),
            "call" => {
                let selected = fields.get("function").and_then(Value::as_str);
                signatures
                    .iter()
                    .filter(|(name, _)| selected.is_none_or(|selected| selected == name.as_str()))
                    .any(|(function, signature)| {
                        let mut arguments = array("arguments").unwrap_or_default();
                        let expected = [
                            signature.arguments(1),
                            signature.arguments(self.body.max_context),
                        ];
                        let [Ok(first), Ok(last)] = expected else {
                            return false;
                        };
                        if arguments.len() > first.len()
                            || (fields.contains_key("arguments")
                                && open_array != Some("arguments")
                                && arguments.len() != first.len())
                        {
                            return false;
                        }
                        for slot in arguments.len()..first.len() {
                            let Some(id) = (0..count).find(|&id| {
                                self.shapes[0][id] == first[slot]
                                    && self.shapes[1][id] == last[slot]
                            }) else {
                                return false;
                            };
                            arguments.push(id);
                        }
                        check(Op::Call {
                            function: function.clone(),
                            arguments,
                        })
                    })
            }
            "scan" => {
                let selected = fields.get("function").and_then(Value::as_str);
                signatures
                    .keys()
                    .filter(|name| selected.is_none_or(|selected| selected == name.as_str()))
                    .any(|function| {
                        refs("input").into_iter().any(|input| {
                            refs("initial").into_iter().any(|initial| {
                                check(Op::Scan {
                                    input,
                                    initial,
                                    function: function.clone(),
                                    max_steps: scalar("max_steps", self.body.max_context),
                                })
                            })
                        })
                    })
            }
            _ => false,
        }
    }
}
