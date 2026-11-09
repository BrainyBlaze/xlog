//! Original-ID lower bounds from the shared source reader's unfinished units.

use std::collections::BTreeMap;
use std::sync::Mutex;

use super::super::ACQUIRED_SOURCE_BYTES;

pub(crate) const CONTRACT: &str = "ordered-structural-subsequence-v1";

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum Atom {
    Byte(u8),
    Letter,
    Digit,
    Symbol,
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
        self == actual || (self == Self::Symbol && matches!(actual, Self::Letter | Self::Digit))
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

pub(crate) fn entry() -> Vec<Unit> {
    let mut units = vec![text(",\"entry\":")];
    units.extend(entry_body());
    units.push(text("}"));
    units
}

pub(crate) fn entry_body() -> Vec<Unit> {
    vec![
        text("[["),
        Unit::Parameters {
            active: false,
            leading_comma: false,
        },
        text("],1,["),
        rows(usize::MAX, None, 0, false),
        text("]]"),
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
    row_commas: usize,
}

impl State {
    pub(crate) fn new(enabled: bool, fields: &BTreeMap<String, Vec<String>>) -> Self {
        Self {
            enabled,
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
                            // All canonical operation tags are nonempty ASCII
                            // words. Operands are relaxed; their tag is not chosen.
                            result.extend([
                                Atom::Byte(b'['),
                                Atom::Byte(b'"'),
                                Atom::Letter,
                                Atom::Byte(b'"'),
                            ]);
                            result.extend(std::iter::repeat_n(Atom::Byte(b','), self.row_commas));
                            result.push(Atom::Byte(b']'));
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
