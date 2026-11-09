//! Source-only structured indices, elaborated to the canonical Gather vector.

use super::super::{maximum_width, typed, NumberChoice, ACQUIRED_SOURCE_BYTES};
use serde::de::{DeserializeSeed, Error, MapAccess, Visitor};
use serde_json::{json, Value};
use xlog_ir::tensor::TensorProgram;

pub(super) const FIELDS: [&str; 4] = ["count", "modulus", "start", "step"];

#[derive(Clone)]
pub(crate) struct Target {
    pub definition: String,
    pub definition_index: usize,
    pub operation: usize,
    pub input: usize,
}

pub(crate) struct Pending<'a> {
    pub bytes: &'a [u8],
    pub target: Target,
}

pub(crate) struct Record {
    pub target: Target,
    pub pattern: Pattern,
}

pub(crate) struct Pattern {
    pub count: usize,
    pub modulus: usize,
    pub start: usize,
    pub step: usize,
}

impl From<[usize; 4]> for Pattern {
    fn from([count, modulus, start, step]: [usize; 4]) -> Self {
        Self {
            count,
            modulus,
            start,
            step,
        }
    }
}

impl Pattern {
    pub(super) fn validate(&self, maximum: usize) -> Result<(), String> {
        if self.count == 0
            || self.count > maximum
            || self.modulus == 0
            || self.modulus > maximum
            || self.start >= self.modulus
            || self.step >= self.modulus
        {
            return Err(
                "structured indices count, modulus, start or step is outside its domain".into(),
            );
        }
        Ok(())
    }

    pub(crate) fn validate_input(&self, columns: usize, maximum: usize) -> Result<(), String> {
        self.validate(maximum)?;
        if self.modulus > columns {
            return Err("structured indices modulus exceeds the input width".into());
        }
        Ok(())
    }

    fn next(&self, value: usize) -> Result<usize, String> {
        value
            .checked_add(self.step)
            .and_then(|value| value.checked_rem(self.modulus))
            .ok_or_else(|| "structured indices arithmetic overflow".into())
    }

    pub(super) fn write(&self, output: &mut super::Output) -> Result<(), super::Stop> {
        // Count exact canonical bytes before expansion. No intermediate index
        // vector or unbounded serialization buffer is allocated by the codec.
        let mut bytes = self.count.checked_add(1).ok_or_else(|| {
            super::Stop::Invalid("structured indices byte extent overflow".into())
        })?;
        output.ensure(bytes.checked_add(self.count).ok_or_else(|| {
            super::Stop::Invalid("structured indices byte extent overflow".into())
        })?)?;
        let mut value = self.start;
        for index in 0..self.count {
            let digits = if value == 0 {
                1
            } else {
                value.ilog10() as usize + 1
            };
            bytes = bytes.checked_add(digits).ok_or_else(|| {
                super::Stop::Invalid("structured indices byte extent overflow".into())
            })?;
            if index + 1 < self.count {
                value = self.next(value).map_err(super::Stop::Invalid)?;
            }
        }
        output.ensure(bytes)?;
        output.push(b'[')?;
        value = self.start;
        for index in 0..self.count {
            if index != 0 {
                output.push(b',')?;
            }
            output.json(&value)?;
            if index + 1 < self.count {
                value = self.next(value).map_err(super::Stop::Invalid)?;
            }
        }
        output.push(b']')
    }
}

impl Record {
    pub(crate) fn validate_body(
        &self,
        body: &TensorProgram,
        signatures: &typed::Signatures,
        maximum: usize,
    ) -> Result<(), String> {
        let mut columns = usize::MAX;
        for tokens in [1, body.max_context] {
            let shapes = body
                .shapes_resolved(tokens, tokens - 1, &|name| {
                    signatures.get(name).cloned().ok_or_else(|| {
                        xlog_core::XlogError::Compilation(
                            "structured indices callee is absent".into(),
                        )
                    })
                })
                .map_err(|error| error.to_string())?;
            let shape = shapes
                .get(self.target.input)
                .ok_or_else(|| "structured indices input is absent".to_owned())?;
            columns = columns.min(shape.columns);
        }
        if self.target.input >= self.target.operation {
            return Err("structured indices input must precede its operation".into());
        }
        self.pattern.validate_input(columns, maximum)
    }
}

pub(super) fn signatures(context: &Value) -> Result<typed::Signatures, String> {
    context["callees"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, callee)| {
            serde_json::from_value(callee["signature"].clone())
                .map(|signature| (name.clone(), signature))
                .map_err(|error| error.to_string())
        })
        .collect()
}

pub(crate) fn admit_prefix(
    pending: &Pending<'_>,
    program: &typed::Program,
    signatures: &typed::Signatures,
    projected_bytes: usize,
    context: &Value,
) -> Result<(), serde_json::Error> {
    let columns = program
        .input_columns(pending.target.input)
        .ok_or_else(|| serde_json::Error::custom("structured indices input is absent"))?;
    let mut decoder = serde_json::Deserializer::from_slice(pending.bytes);
    let result = (Fields {
        pending,
        program,
        signatures,
        columns,
        projected_bytes,
        maximum: maximum_width(context),
    })
    .deserialize(&mut decoder);
    match result {
        Ok(()) => decoder.end(),
        Err(error) if error.is_eof() => Ok(()),
        Err(error) => Err(error),
    }
}

struct Fields<'a> {
    pending: &'a Pending<'a>,
    program: &'a typed::Program,
    signatures: &'a typed::Signatures,
    columns: usize,
    projected_bytes: usize,
    maximum: usize,
}

impl<'de> DeserializeSeed<'de> for Fields<'_> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, decoder: D) -> Result<(), D::Error> {
        decoder.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for Fields<'_> {
    type Value = ();
    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ordered count, modulus, start and step")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        let mut numbers = [0usize; 4];
        for (field, expected) in FIELDS.into_iter().enumerate() {
            if map.next_key::<String>()?.as_deref() != Some(expected) {
                return Err(A::Error::custom(
                    "structured indices fields or order differ",
                ));
            }
            let values = match field {
                0 => {
                    let mut range = self.program.column_range();
                    // An array of n zeroes already requires 2*n+1 bytes. This
                    // bounds prospective allocations before consulting typed
                    // completion; complete descriptors check their exact bytes.
                    let count_bound = ACQUIRED_SOURCE_BYTES
                        .saturating_sub(self.projected_bytes)
                        .saturating_sub(1)
                        / 2;
                    range.end = range.end.min(count_bound + 1).min(self.maximum + 1);
                    range
                }
                1 => 1..self.columns + 1,
                _ => 0..numbers[1],
            };
            let choice = NumberChoice {
                values,
                extensions: (field == 0).then(|| {
                    self.program
                        .completion_widths(self.signatures)
                        .into_iter()
                        .collect()
                }),
                accepts: |count| {
                    if field != 0 {
                        return true;
                    }
                    let material = serde_json::Map::from_iter([
                        ("operation".into(), json!("gather_columns")),
                        ("input".into(), json!(self.pending.target.input)),
                        ("indices".into(), Value::Array(vec![json!(0); count])),
                    ]);
                    self.program.possible(&material, None, self.signatures)
                },
                bytes: self.pending.bytes,
            };
            if !choice.permits_prefix("") {
                return Err(A::Error::custom(
                    "structured indices field has no compatible continuation",
                ));
            }
            numbers[field] = map.next_value_seed(choice)?.value;
        }
        if map.next_key::<String>()?.is_some() {
            return Err(A::Error::custom("unknown structured indices field"));
        }
        Pattern::from(numbers)
            .validate_input(self.columns, self.maximum)
            .map_err(A::Error::custom)
    }
}
