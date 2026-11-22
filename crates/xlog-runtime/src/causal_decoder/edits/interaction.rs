//! Track whether a rule's new score term can change relative edit preferences across histories.

use super::*;

#[derive(Clone, Copy, Default, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct Sources {
    history: bool,
    edit: bool,
    interaction: bool,
}

impl Sources {
    const HISTORY: Self = Self {
        history: true,
        edit: false,
        interaction: false,
    };
    const EDIT: Self = Self {
        history: false,
        edit: true,
        interaction: false,
    };

    fn sum(self, other: Self) -> Self {
        Self {
            history: self.history || other.history,
            edit: self.edit || other.edit,
            interaction: self.interaction || other.interaction,
        }
    }

    fn product(self, other: Self) -> Self {
        let mut output = self.sum(other);
        output.interaction |= (self.history && other.edit) || (self.edit && other.history);
        output
    }

    fn nonlinear(mut self) -> Self {
        self.interaction |= self.history && self.edit;
        self
    }

    fn select(self, positive: Self, negative: Self) -> Self {
        self.nonlinear().product(positive.sum(negative))
    }
}

type CallCache = BTreeMap<(String, Vec<Sources>), Sources>;

fn call_sources(
    functions: &FunctionRegistry,
    name: &str,
    arguments: Vec<Sources>,
    cache: &mut CallCache,
) -> Result<Sources> {
    let key = (name.to_owned(), arguments);
    if let Some(&sources) = cache.get(&key) {
        return Ok(sources);
    }
    let body = functions.numerical(name)?;
    let mut values: Vec<Sources> = Vec::with_capacity(body.operations.len());
    for op in &body.operations {
        let output = match op {
            Op::Argument { argument } => key.1[*argument],
            Op::Call {
                function,
                arguments,
            } => {
                let supplied = arguments.iter().map(|&id| values[id]).collect();
                call_sources(functions, function, supplied, cache)?
            }
            Op::Tanh { input } => values[*input].nonlinear(),
            Op::Multiply { left, right } | Op::MatMul { left, right } => {
                values[*left].product(values[*right])
            }
            Op::Divide { left, right } => values[*left].product(values[*right].nonlinear()),
            Op::Subtract { left, right } if left == right => Sources::default(),
            Op::Select {
                condition,
                positive,
                negative,
            } => {
                if positive == negative {
                    values[*positive]
                } else {
                    values[*condition].select(values[*positive], values[*negative])
                }
            }
            // Linear projections, additions and coordinate reads preserve existing interactions.
            _ => op
                .inputs()
                .into_iter()
                .fold(Sources::default(), |terms, id| terms.sum(values[id])),
        };
        values.push(output);
    }
    let output = values[body.output];
    cache.insert(key, output);
    Ok(output)
}

pub(super) fn callees(
    functions: &FunctionRegistry,
    names: &[(String, usize)],
) -> Result<BTreeMap<String, Sources>> {
    let mut cache = CallCache::new();
    names
        .iter()
        .map(|(name, _)| {
            let sources = call_sources(
                functions,
                name,
                vec![Sources::HISTORY, Sources::EDIT],
                &mut cache,
            )?;
            Ok((name.clone(), sources))
        })
        .collect()
}

pub(super) fn permits(edit: &NumericalEdit, callees: &BTreeMap<String, Sources>) -> bool {
    let source = |operand: &EditOperand| match operand {
        EditOperand::History { .. } => Sources::HISTORY,
        EditOperand::Descriptor { coordinates } => {
            // These fields identify the common parent and site, so they are constant across its menu.
            if coordinates.iter().any(|id| !(6..24).contains(id)) {
                Sources::EDIT
            } else {
                Sources::default()
            }
        }
        EditOperand::Definition { name } => callees[name],
    };
    use EditOperation::*;
    match (edit.operation, edit.arguments.as_slice()) {
        (Read, [a]) => source(a).interaction,
        (Tanh, [a]) => source(a).nonlinear().interaction,
        (Subtract, [a, b]) if a == b => false,
        (Add | Subtract, [a, b]) => source(a).sum(source(b)).interaction,
        (Multiply, [a, b]) => source(a).product(source(b)).interaction,
        (Select, [_, b, c]) if b == c => source(b).interaction,
        (Select, [a, b, c]) => source(a).select(source(b), source(c)).interaction,
        _ => false,
    }
}
