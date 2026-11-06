//! Prospective operand choices use the executable compiler's shape inference.

use std::collections::BTreeMap;

use serde_json::{Map, Value};
use xlog_core::XlogError;
use xlog_ir::tensor::{TensorOperation as Op, TensorProgram, TensorSignature, TensorValueShape};

pub(super) type Signatures = BTreeMap<String, TensorSignature>;

pub(super) struct Program {
    pub(super) body: TensorProgram,
    pub(super) planned_output: Option<usize>,
    shapes: [Vec<TensorValueShape>; 2],
    raw_tokens: Vec<u32>,
}

impl Program {
    pub(super) fn new(
        signature: TensorSignature,
        vocabulary: usize,
        max_context: usize,
        raw_tokens: Vec<u32>,
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
            planned_output: None,
            shapes: [Vec::new(), Vec::new()],
            raw_tokens,
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
        if self
            .planned_output
            .is_some_and(|output| self.body.operations.len() > output)
        {
            return Err("operations exceed the planned output slot".into());
        }
        let first = self
            .infer(&operation, signatures, 0)
            .map_err(|error| error.to_string())?;
        let last = self
            .infer(&operation, signatures, 1)
            .map_err(|error| error.to_string())?;
        if self.is_output_slot()
            && (!self.matches_output(first, 0) || !self.matches_output(last, 1))
        {
            return Err("planned output does not match the signature".into());
        }
        self.body.operations.push(operation);
        self.body
            .validate_acquired_body()
            .map_err(|error| error.to_string())?;
        self.shapes[0].push(first);
        self.shapes[1].push(last);
        Ok(())
    }

    pub(super) fn is_output_slot(&self) -> bool {
        self.planned_output == Some(self.body.operations.len())
    }

    pub(super) fn output_columns(&self) -> Option<usize> {
        self.is_output_slot().then(|| {
            self.body
                .signature
                .output(1, self.body.vocabulary)
                .unwrap()
                .columns
        })
    }

    fn matches_output(&self, shape: TensorValueShape, length: usize) -> bool {
        self.body
            .signature
            .output([1, self.body.max_context][length], self.body.vocabulary)
            .ok()
            == Some(shape)
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

    pub(super) fn required_output(&self) -> Result<[[usize; 2]; 2], String> {
        let shape = |tokens| {
            self.body
                .signature
                .output(tokens, self.body.vocabulary)
                .map(|shape| [shape.rows, shape.columns])
                .map_err(|error| error.to_string())
        };
        Ok([shape(1)?, shape(self.body.max_context)?])
    }

    pub(super) fn last_value(&self) -> Option<[[usize; 2]; 2]> {
        let first = self.shapes[0].last()?;
        let last = self.shapes[1].last()?;
        Some([[first.rows, first.columns], [last.rows, last.columns]])
    }

    fn compatible(&self, operation: Op, signatures: &Signatures) -> bool {
        // The same causal-dependency check used by the complete compiler applies
        // to this prospective operation. No prospective graph is ever emitted.
        let mut body = self.body.clone();
        body.operations.push(operation.clone());
        if body.validate_acquired_body().is_err() {
            return false;
        }
        (0..2).all(|length| {
            self.infer(&operation, signatures, length)
                .is_ok_and(|shape| !self.is_output_slot() || self.matches_output(shape, length))
        })
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
            if self
                .output_columns()
                .is_some_and(|columns| prefix.len() >= columns)
            {
                return Vec::new();
            }
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
                columns: scalar("columns", self.output_columns().unwrap_or(1)),
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
            "token_indicators" => {
                let mut token_ids = array("token_ids").unwrap_or_default();
                if !fields.contains_key("token_ids") || open_array == Some("token_ids") {
                    let columns = self.output_columns().unwrap_or(1);
                    // Prospective distinct IDs prove that a continuation exists;
                    // only the model's actual array is ever added to the body.
                    for &id in &self.raw_tokens {
                        if token_ids.len() >= columns {
                            break;
                        }
                        if !token_ids.contains(&(id as usize)) {
                            token_ids.push(id as usize);
                        }
                    }
                }
                check(Op::TokenIndicators {
                    token_ids: token_ids.into_iter().map(|id| id as u32).collect(),
                })
            }
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
                if let Some(columns) = self.output_columns() {
                    if !fields.contains_key("indices") || open_array == Some("indices") {
                        if indices.len() > columns {
                            return false;
                        }
                        indices.resize(columns, 0);
                    }
                } else if indices.is_empty() && open_array == Some("indices") {
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
