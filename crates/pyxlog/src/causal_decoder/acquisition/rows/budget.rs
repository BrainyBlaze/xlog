//! Original-ID lower bounds from the shared source reader's unfinished units.

use std::collections::BTreeMap;
use std::sync::Mutex;

use super::super::{is_value_reference, ACQUIRED_SOURCE_BYTES};

pub(crate) const CONTRACT: &str = "ordered-structural-subsequence-v1";
pub(crate) const EXPRESSION_CONTRACT: &str = "ordered-structural-subsequence-v2";
pub(crate) const REFERENCE_CONTRACT: &str = "ordered-structural-subsequence-v3";

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum Atom {
    Byte(u8),
    Letter,
    Digit,
    Symbol,
    ValueStart,
}

impl Atom {
    fn of(byte: u8) -> Option<Self> {
        match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => Some(Self::Letter),
            b'0'..=b'9' => Some(Self::Digit),
            b'[' | b']' | b'{' | b'}' | b'"' | b',' | b':' => Some(Self::Byte(byte)),
            _ => None,
        }
    }

    fn matches(self, actual: Self) -> bool {
        self == actual
            || (self == Self::Symbol && matches!(actual, Self::Letter | Self::Digit))
            || (self == Self::ValueStart && matches!(actual, Self::Digit | Self::Byte(b'[')))
    }
}

pub(crate) enum Unit {
    Bytes(Vec<Atom>),
    Rows {
        definition: usize,
        count: Option<usize>,
        completed: usize,
        leading_comma: bool,
    },
    Parameters {
        active: bool,
        leading_comma: bool,
    },
}

pub(crate) fn text(bytes: &str) -> Unit {
    Unit::Bytes(bytes.bytes().filter_map(Atom::of).collect())
}

pub(crate) fn field(name: &str) -> Unit {
    text(match name {
        "function" => "\"\"",
        "arguments" | "token_ids" => "[]",
        // Gather also accepts an object descriptor. These alternatives share
        // no opening/closing byte; the reader reserves the chosen form later.
        "indices" => "",
        _ => "1",
    })
}

/// Unknown trees retain only mandatory root syntax, never a flattened row list.
pub(crate) fn expression() -> Unit {
    text("[\"\",]")
}

pub(crate) fn expression_value(references: bool) -> Unit {
    if references {
        // A future value may begin with an integer or an operation tuple.
        // Its remaining unknown syntax has no common required delimiters.
        Unit::Bytes(vec![Atom::ValueStart])
    } else {
        expression()
    }
}

pub(crate) fn expression_field(name: &str, references: bool) -> Unit {
    if is_value_reference(name) {
        expression_value(references)
    } else {
        field(name)
    }
}

pub(crate) fn result(
    expressions: bool,
    definition: usize,
    count: Option<usize>,
    completed: usize,
    leading_comma: bool,
) -> Unit {
    if expressions {
        expression()
    } else {
        rows(definition, count, completed, leading_comma)
    }
}

pub(crate) fn entry(self_delimited: bool, named_body: bool, expressions: bool) -> Vec<Unit> {
    let mut units = vec![text(",\"entry\":")];
    units.extend(entry_body(self_delimited, named_body, expressions));
    units.push(text("}"));
    units
}

pub(crate) fn entry_body(self_delimited: bool, named_body: bool, expressions: bool) -> Vec<Unit> {
    vec![
        text(if named_body { "{\"weights\":[" } else { "[[" }),
        Unit::Parameters {
            active: false,
            leading_comma: false,
        },
        text(if expressions {
            "],\"return\":"
        } else if named_body {
            "],\"ops\":["
        } else if self_delimited {
            "],["
        } else {
            "],1,["
        }),
        result(expressions, usize::MAX, None, 0, false),
        text(if expressions {
            "}"
        } else if named_body {
            "]}"
        } else {
            "]]"
        }),
    ]
}

pub(crate) fn rows(
    definition: usize,
    count: Option<usize>,
    completed: usize,
    leading_comma: bool,
) -> Unit {
    Unit::Rows {
        definition,
        count,
        completed,
        leading_comma,
    }
}

#[derive(Default)]
pub(crate) struct State {
    enabled: bool,
    frames: Vec<Vec<Unit>>,
    pending: Vec<Atom>,
    pub definition: usize,
    pub entry: bool,
    pub entry_pending: bool,
    pub complete: bool,
    pub count_prefix: Option<Vec<u8>>,
    pub operation_open: bool,
    row_commas: usize,
    parameter_references: bool,
    expressions: bool,
}

impl State {
    fn append_row(&self, pattern: &mut Vec<Atom>) {
        // Canonical operation tags are nonempty ASCII words. Operands remain
        // relaxed; neither a tag nor a graph is selected by this projection.
        pattern.extend([
            Atom::Byte(b'['),
            Atom::Byte(b'"'),
            Atom::Letter,
            Atom::Byte(b'"'),
        ]);
        pattern.extend(std::iter::repeat_n(Atom::Byte(b','), self.row_commas));
        pattern.push(Atom::Byte(b']'));
    }

    fn operations_pattern(&self, rows: usize) -> Vec<Atom> {
        let mut pattern = Vec::new();
        for _ in 0..rows {
            self.append_row(&mut pattern);
        }
        pattern
    }

    pub(crate) fn new(
        enabled: bool,
        fields: &BTreeMap<String, Vec<String>>,
        parameter_references: bool,
        expressions: bool,
    ) -> Self {
        Self {
            enabled,
            parameter_references,
            expressions,
            row_commas: if enabled {
                fields
                    .values()
                    .map(|fields| fields.len().saturating_sub(1))
                    .min()
                    .unwrap_or(0)
            } else {
                0
            },
            ..Self::default()
        }
    }

    pub(crate) fn enter(&mut self) -> usize {
        if !self.enabled {
            return 0;
        }
        let frame = self.frames.len();
        self.frames.push(Vec::new());
        frame
    }

    pub(crate) fn set(&mut self, frame: usize, units: impl FnOnce() -> Vec<Unit>) {
        if !self.enabled {
            return;
        }
        self.frames.truncate(frame + 1);
        self.frames[frame] = units();
        self.pending.clear();
    }

    pub(crate) fn leave(&mut self, frame: usize) {
        if !self.enabled {
            return;
        }
        self.frames.truncate(frame);
    }

    pub(crate) fn expect(&mut self, byte: u8) {
        if !self.enabled {
            return;
        }
        self.pending = Atom::of(byte).into_iter().collect();
    }

    /// Only unfinished delimiters of this RawValue are retained. Its variable
    /// content, escaping and operand constraints remain optimistic, not repaired.
    pub(crate) fn raw(&mut self, bytes: &[u8]) {
        if !self.enabled {
            return;
        }
        let mut closing = Vec::new();
        let mut quoted = false;
        let mut escaped = false;
        for &byte in bytes {
            if quoted {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    quoted = false;
                }
            } else {
                match byte {
                    b'"' => quoted = true,
                    b'[' => closing.push(b']'),
                    b'{' => closing.push(b'}'),
                    b']' | b'}' => {
                        closing.pop();
                    }
                    _ => {}
                }
            }
        }
        self.pending = quoted
            .then_some(Atom::Byte(b'"'))
            .into_iter()
            .chain(closing.into_iter().rev().map(Atom::Byte))
            .collect();
    }

    fn pattern(&self, count: Option<usize>, parameters: Obligations) -> Vec<Atom> {
        let mut result = self.pending.clone();
        if let (Some(count), Some(prefix)) = (count, &self.count_prefix) {
            if let Some(tail) = count.to_string().as_bytes().strip_prefix(prefix.as_slice()) {
                result.extend(std::iter::repeat_n(Atom::Digit, tail.len()));
            }
        }
        for frame in self.frames.iter().rev() {
            for unit in frame {
                match unit {
                    Unit::Bytes(atoms) => result.extend(atoms),
                    Unit::Rows {
                        definition,
                        count: chosen,
                        completed,
                        leading_comma,
                    } => {
                        let count = if *definition == self.definition {
                            count.or(*chosen)
                        } else {
                            *chosen
                        }
                        .unwrap_or(1)
                        .max(if *definition == usize::MAX || self.entry {
                            parameters.minimum_rows
                        } else {
                            1
                        });
                        for index in *completed..count {
                            if *leading_comma || index > *completed {
                                result.push(Atom::Byte(b','));
                            }
                            self.append_row(&mut result);
                        }
                    }
                    Unit::Parameters {
                        active,
                        leading_comma,
                    } => {
                        let missing = parameters.missing.saturating_sub(usize::from(*active));
                        for index in 0..missing {
                            if *active || *leading_comma || index != 0 {
                                result.push(Atom::Byte(b','));
                            }
                            result.extend([Atom::Byte(b'['), Atom::Byte(b'"')]);
                            result.extend(std::iter::repeat_n(Atom::Symbol, parameters.symbols));
                            if self.parameter_references {
                                result.extend([Atom::Byte(b'"'), Atom::Byte(b']')]);
                                continue;
                            }
                            result.extend([Atom::Byte(b'"'), Atom::Byte(b','), Atom::Byte(b'[')]);
                            for dimension in 0..parameters.rank {
                                if dimension != 0 {
                                    result.push(Atom::Byte(b','));
                                }
                                result.push(Atom::Digit);
                            }
                            result.extend([Atom::Byte(b']'), Atom::Byte(b']')]);
                        }
                    }
                }
            }
        }
        result
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct Obligations {
    pub missing: usize,
    pub symbols: usize,
    pub rank: usize,
    pub minimum_rows: usize,
}

#[derive(Default)]
struct Costs {
    values: BTreeMap<Vec<Atom>, Coverage>,
    bytes: usize,
}

#[derive(Clone, Copy, Default)]
struct Coverage {
    position: usize,
    tokens: usize,
}

pub(crate) struct Alphabet {
    spellings: Vec<Vec<Atom>>,
    costs: Mutex<Costs>,
}

fn extends(pattern: &[Atom], spelling: &[Atom], mut position: usize) -> usize {
    for &atom in spelling {
        if pattern.get(position).is_some_and(|next| next.matches(atom)) {
            position += 1;
        }
    }
    position
}

impl Alphabet {
    pub(crate) fn new(token_bytes: &[Vec<u8>]) -> Self {
        let mut spellings = token_bytes
            .iter()
            .map(|bytes| bytes.iter().filter_map(|&byte| Atom::of(byte)).collect())
            .filter(|spelling: &Vec<Atom>| !spelling.is_empty())
            .collect::<Vec<_>>();
        spellings.sort_unstable();
        spellings.dedup();
        // A longer supplied spelling with this exact projected prefix covers
        // every obligation covered by the shorter one at the same one-ID cost.
        // Lexical neighbours find these in linear time after sorting; do not
        // run a vocabulary-squared subsequence search at decoder construction.
        let maximal = spellings
            .iter()
            .enumerate()
            .filter(|(index, spelling)| {
                !spellings
                    .get(index + 1)
                    .is_some_and(|next| next.starts_with(spelling))
            })
            .map(|(_, spelling)| spelling.clone())
            .collect();
        Self {
            spellings: maximal,
            costs: Mutex::new(Costs::default()),
        }
    }

    fn covers(&self, pattern: Vec<Atom>, allowance: usize) -> bool {
        let mut costs = self.costs.lock().unwrap();
        let mut coverage = costs.values.get(&pattern).copied().unwrap_or_default();
        while coverage.position < pattern.len() && coverage.tokens < allowance {
            let next = self
                .spellings
                .iter()
                .map(|spelling| extends(&pattern, spelling, coverage.position))
                .max()
                .unwrap_or(coverage.position);
            if next == coverage.position {
                coverage.tokens = usize::MAX;
                break;
            }
            // Any longer prefix coverage leaves a suffix of every shorter
            // coverage's obligation. Thus the farthest transition dominates.
            // If it cannot finish within the allowance, no shorter transition
            // can. A later larger allowance resumes this same greedy frontier.
            coverage.position = next;
            coverage.tokens += 1;
        }
        let covered = coverage.position == pattern.len() && coverage.tokens <= allowance;
        let bytes = pattern.len() * std::mem::size_of::<Atom>()
            + std::mem::size_of::<(Vec<Atom>, Coverage)>();
        if bytes <= ACQUIRED_SOURCE_BYTES {
            if !costs.values.contains_key(&pattern) {
                if costs.bytes + bytes > ACQUIRED_SOURCE_BYTES {
                    *costs = Costs::default();
                }
                costs.bytes += bytes;
            }
            costs.values.insert(pattern, coverage);
        }
        covered
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Bound<'a> {
    pub alphabet: &'a Alphabet,
    pub state: &'a State,
    /// The hypothetical next original ID is already subtracted, never inferred.
    pub remaining: usize,
    pub eos: bool,
}

impl Bound<'_> {
    /// Decimal references use the same alphabet and suffix obligations. A
    /// viable longer ID must cover its still-unconsumed digits and paid EOS.
    pub(crate) fn permits_decimal(
        &self,
        prefix: &[u8],
        value: usize,
        parameters: Obligations,
    ) -> bool {
        let spelling = value.to_string();
        let Some(tail) = spelling.as_bytes().strip_prefix(prefix) else {
            return false;
        };
        if self.eos {
            return false;
        }
        let mut pattern = vec![Atom::Digit; tail.len()];
        pattern.extend(self.state.pattern(None, parameters));
        self.remaining
            .checked_sub(1)
            .is_some_and(|allowance| self.alphabet.covers(pattern, allowance))
    }

    /// Invert the same necessary suffix bound for the active body's total rows.
    /// The reader's completed count includes a partly emitted row whose remaining
    /// bytes are already in the unfinished units. No row is paid twice.
    pub(crate) fn row_capacity(&self, maximum: usize, parameters: Obligations) -> Option<usize> {
        if self.state.expressions {
            return None;
        }
        let completed = self.state.frames.iter().flatten().find_map(|unit| {
            if let Unit::Rows {
                definition,
                completed,
                ..
            } = unit
            {
                (*definition == self.state.definition).then_some(*completed)
            } else {
                None
            }
        })?;
        let mut minimum = completed.max(1);
        if minimum > maximum || !self.permits(Some(minimum), parameters) {
            return Some(0);
        }
        let mut maximum = maximum;
        // Increasing count only appends mandatory row atoms in the same unit;
        // suffix coverage is monotone even when a spelling crosses units.
        while minimum < maximum {
            let middle = minimum + (maximum - minimum).div_ceil(2);
            if self.permits(Some(middle), parameters) {
                minimum = middle;
            } else {
                maximum = middle - 1;
            }
        }
        Some(minimum)
    }

    /// These operations may occur in any later definition. Their ordered
    /// projection is a separate necessary bound, not an assigned entry body.
    pub(crate) fn permits_unplaced_rows(&self, rows: usize) -> bool {
        if self.state.expressions {
            // The flat operation projection is not ordered for sibling/nested
            // trees. Canonical node/dependency admission remains independent.
            return true;
        }
        let rows = rows.saturating_sub(usize::from(self.state.operation_open));
        if self.eos {
            return self.state.complete && rows == 0;
        }
        self.remaining.checked_sub(1).is_some_and(|allowance| {
            self.alphabet
                .covers(self.state.operations_pattern(rows), allowance)
        })
    }

    pub(crate) fn permits(&self, count: Option<usize>, parameters: Obligations) -> bool {
        if self.eos {
            return self.state.complete;
        }
        self.remaining.checked_sub(1).is_some_and(|allowance| {
            self.alphabet
                .covers(self.state.pattern(count, parameters), allowance)
        })
    }
}
