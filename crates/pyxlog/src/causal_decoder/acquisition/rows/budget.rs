//! A lower bound on original-token cost, owned by the source reader.

use super::super::ACQUIRED_SOURCE_BYTES;
use std::collections::BTreeMap;
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum Atom {
    Open,
    Close,
    Quote,
    Letter,
    Digit,
}

impl Atom {
    pub(crate) fn of(byte: u8) -> Option<Self> {
        match byte {
            b'[' | b'{' => Some(Self::Open),
            b']' | b'}' => Some(Self::Close),
            b'"' => Some(Self::Quote),
            b'a'..=b'z' | b'A'..=b'Z' => Some(Self::Letter),
            b'0'..=b'9' => Some(Self::Digit),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) enum Phase {
    #[default]
    Root,
    Helpers,
    EntryStart,
    Name,
    Signature,
    Parameters,
    Count,
    Operations,
    Row,
    OperationsEnd,
    DefinitionEnd,
    RootEnd,
    Complete,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct State {
    pub phase: Phase,
    pub entry: bool,
    pub count: Option<usize>,
    pub completed_rows: usize,
    pub definition_index: usize,
    pub pending: Option<Atom>,
}

impl State {
    pub(crate) fn definition(&mut self, entry: bool) {
        self.entry = entry;
        self.count = None;
        self.completed_rows = 0;
        self.phase = if entry { Phase::Parameters } else { Phase::Name };
    }

    pub(crate) fn complete(&self) -> bool {
        matches!(self.phase, Phase::Complete)
    }

    fn pattern(&self, count: Option<usize>) -> Vec<Atom> {
        use Atom::*;
        let mut pattern = Vec::new();
        pattern.extend(self.pending);
        let key = [Quote, Letter, Quote];
        let parameter_table = [Open, Close];
        // A row has an opening array, a nonempty alphabetic opcode string,
        // and a closing array. Escaped opcodes still contain the ASCII `u`.
        // Variable operands are deliberately omitted from this relaxation.
        let row = [Open, Quote, Letter, Quote, Close];
        let entry = |pattern: &mut Vec<Atom>| {
            pattern.push(Open);
            pattern.extend(key);
            pattern.extend(parameter_table);
            pattern.extend(key);
            pattern.push(Digit);
            pattern.extend(key);
            pattern.push(Open);
            pattern.extend(row);
            pattern.extend([Close, Close, Close]);
        };
        match self.phase {
            Phase::Root => {
                // The intent is the active unit. Helpers may be empty.
                pattern.extend(key);
                pattern.extend([Open, Close]);
                pattern.extend(key);
                entry(&mut pattern);
                return pattern;
            }
            Phase::Helpers => {
                pattern.push(Close);
                pattern.extend(key);
                entry(&mut pattern);
                return pattern;
            }
            Phase::EntryStart => {
                entry(&mut pattern);
                return pattern;
            }
            Phase::RootEnd | Phase::Complete => return pattern,
            _ => {}
        }
        // The current metadata unit or unfinished operation is omitted. All
        // later fields remain compulsory, even if another helper is added.
        if matches!(self.phase, Phase::Name) {
            pattern.extend(key);
            pattern.extend([Open, Close]);
        }
        if matches!(self.phase, Phase::Name | Phase::Signature) {
            pattern.extend(key);
            pattern.extend(parameter_table);
        }
        if matches!(self.phase, Phase::Name | Phase::Signature | Phase::Parameters) {
            pattern.extend(key);
            pattern.push(Digit);
        }
        if matches!(
            self.phase,
            Phase::Name | Phase::Signature | Phase::Parameters | Phase::Count
        ) {
            pattern.extend(key);
            pattern.push(Open);
        }
        if !matches!(self.phase, Phase::OperationsEnd | Phase::DefinitionEnd) {
            let count = if matches!(self.phase, Phase::Count) {
                count.or(self.count)
            } else {
                self.count
            }
            .unwrap_or(1);
            let active = usize::from(matches!(self.phase, Phase::Row));
            for _ in 0..count.saturating_sub(self.completed_rows + active) {
                pattern.extend(row);
            }
            pattern.push(Close);
        }
        if !matches!(self.phase, Phase::DefinitionEnd) {
            pattern.push(Close);
        }
        if self.entry {
            pattern.push(Close);
        } else {
            pattern.push(Close);
            pattern.extend(key);
            entry(&mut pattern);
        }
        pattern
    }
}

#[derive(Default)]
struct Costs {
    values: BTreeMap<Vec<Atom>, usize>,
    bytes: usize,
}

pub(crate) struct Alphabet {
    spellings: Vec<Vec<Atom>>,
    costs: Mutex<Costs>,
}

fn subsequence(needle: &[Atom], haystack: &[Atom]) -> bool {
    let mut position = 0;
    for atom in haystack {
        if needle.get(position) == Some(atom) {
            position += 1;
        }
    }
    position == needle.len()
}

impl Alphabet {
    pub(crate) fn new(token_bytes: &[Vec<u8>]) -> Self {
        let mut spellings = token_bytes
            .iter()
            .map(|bytes| bytes.iter().filter_map(|&byte| Atom::of(byte)).collect())
            .filter(|spelling: &Vec<Atom>| !spelling.is_empty())
            .collect::<Vec<_>>();
        spellings.sort_unstable_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        spellings.dedup();
        let mut maximal: Vec<Vec<Atom>> = Vec::new();
        for spelling in spellings {
            // A dominating spelling can cover every subsequence covered by
            // this one, at the same one-ID cost. No model IDs are filtered.
            if !maximal.iter().any(|other| subsequence(&spelling, other)) {
                maximal.push(spelling);
            }
        }
        Self {
            spellings: maximal,
            costs: Mutex::new(Costs::default()),
        }
    }

    fn minimum(&self, pattern: Vec<Atom>) -> usize {
        let mut costs = self.costs.lock().unwrap();
        if let Some(&cost) = costs.values.get(&pattern) {
            return cost;
        }
        let mut position = 0;
        let mut cost = 0usize;
        while position < pattern.len() {
            let mut farthest = position;
            for spelling in &self.spellings {
                let mut next = position;
                for atom in spelling {
                    if pattern.get(next) == Some(atom) {
                        next += 1;
                    }
                }
                farthest = farthest.max(next);
            }
            if farthest == position {
                cost = usize::MAX;
                break;
            }
            // Subsequence progress is monotone: starting farther along the
            // same pattern never needs more tokens for its remaining suffix.
            // Thus the farthest one-token transition dominates shorter ones.
            position = farthest;
            cost += 1;
        }
        let bytes = pattern.len() * std::mem::size_of::<Atom>()
            + std::mem::size_of::<(Vec<Atom>, usize)>();
        if bytes <= ACQUIRED_SOURCE_BYTES {
            if costs.bytes + bytes > ACQUIRED_SOURCE_BYTES {
                costs.values.clear();
                costs.bytes = 0;
            }
            costs.bytes += bytes;
            costs.values.insert(pattern, cost);
        }
        cost
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Bound<'a> {
    pub alphabet: &'a Alphabet,
    pub state: State,
    /// The next original ID has already been charged by the caller of this check.
    pub remaining: usize,
    pub eos: bool,
}

impl Bound<'_> {
    pub(crate) fn permits(&self, count: Option<usize>) -> bool {
        if self.eos {
            return self.state.complete();
        }
        self.alphabet
            .minimum(self.state.pattern(count))
            .checked_add(1) // A separate paid EOS follows every complete source.
            .is_some_and(|cost| cost <= self.remaining)
    }
}
