//! Necessary unopened source occurrences, independent of canonical row reuse.

use super::*;
use xlog_ir::tensor::TensorShapeConstraint;

type ShapeKey = [(Option<usize>, Option<usize>, Option<bool>); 2];

/// Each query owns its environment. No source-conditioned answer enters a
/// generic completion cache or survives into another prefix/definition.
#[derive(Default)]
struct Leaves(BTreeMap<ShapeKey, bool>);

impl Leaves {
    fn minimum(
        &mut self,
        resources: &Resources<'_>,
        expected: [TensorShapeConstraint; 2],
    ) -> usize {
        let program = resources.program.typed.as_ref().unwrap();
        if !program.searches_completion() {
            // This mode normally has the complete scalar-domain enumerator.
            // An incomplete proof cannot strengthen a source hole's bound.
            return 1;
        }
        let key = expected.map(|shape| (shape.rows, shape.columns, shape.float32));
        let possible = *self.0.entry(key).or_insert_with(|| {
            program.leaf_possible(
                resources.operation_fields.unwrap(),
                &resources.signatures,
                expected,
            )
        });
        1 + usize::from(!possible)
    }
}

impl Resources<'_> {
    fn expression_expected(
        &self,
        ancestors: &[typed::Ancestor],
    ) -> Result<[TensorShapeConstraint; 2], serde_json::Error> {
        let partial = typed::PartialConstraints {
            ancestors,
            fields: self.operation_fields.unwrap(),
        };
        let program = self.program.typed.as_ref().unwrap();
        Ok([
            partial
                .expected(program, 0)
                .map_err(serde_json::Error::custom)?,
            partial
                .expected(program, 1)
                .map_err(serde_json::Error::custom)?,
        ])
    }

    pub(super) fn admit_expression_nodes(&self) -> Result<(), serde_json::Error> {
        let Some(nodes) = self.expression_nodes else {
            return Ok(());
        };
        let mut necessary = nodes.spent + nodes.pending_roots;
        let mut leaves = Leaves::default();
        if nodes.current_root_pending
            && self.program_definition_index == self.definition_index
            && self.program.parameters_complete
        {
            // The current root is already included once. Strengthening its
            // type does not reserve it again, nor assign future optional helpers.
            necessary += leaves.minimum(self, self.expression_expected(&[])?) - 1;
        }
        let (parents, _) = self.source_expression.unwrap();
        let mut ancestors = Vec::new();
        for (index, parent) in parents.iter().enumerate() {
            let active_open = index + 1 < parents.len();
            let mut minimum = None;
            for (name, required) in self.operation_fields.unwrap() {
                if !operation_prefix(&parent.tag_prefix, name)
                    || !self.expression_parent_domains(parent, required)
                {
                    continue;
                }
                let mut fields = parent.selected.clone();
                fields.insert("operation".into(), json!(name));
                let partial = typed::PartialConstraints {
                    ancestors: &ancestors,
                    fields: self.operation_fields.unwrap(),
                };
                let program = self.program.typed.as_ref().unwrap();
                if !partial.permits(program, &fields) {
                    continue;
                }
                if !active_open
                    && !parent.projected
                    && parent.operand.is_none()
                    && self.expression_children_known(parent, required)
                    && !program.possible(
                        &fields,
                        None,
                        &self.signatures,
                        Some(
                            self.expression_continuation()
                                .unwrap()
                                .borrow(self.operation_fields.unwrap()),
                        ),
                    )
                {
                    continue;
                }
                let mut future = 0;
                for field in required.iter().filter(|field| is_value_reference(field)) {
                    if fields.contains_key(field)
                        || (active_open
                            && matches!(&parent.operand,
                            Some(typed::Operand::Value(active)) if active == field))
                    {
                        continue;
                    }
                    ancestors.push(typed::Ancestor {
                        fields: fields.clone(),
                        operand: typed::Operand::Value(field.clone()),
                    });
                    let expected = self.expression_expected(&ancestors)?;
                    ancestors.pop();
                    future += leaves.minimum(self, expected);
                }
                if required.iter().any(|field| field == "arguments") {
                    let arguments = fields
                        .get("arguments")
                        .and_then(Value::as_array)
                        .map(Vec::as_slice)
                        .unwrap_or_default();
                    let active = matches!(parent.operand, Some(typed::Operand::Argument(_)));
                    let committed = arguments.len() + usize::from(active);
                    let opened = arguments.len() + usize::from(active && active_open);
                    let expected = self.expression_expected(&ancestors)?;
                    let remaining = self
                        .signatures
                        .iter()
                        .filter(|(name, _)| {
                            fields
                                .get("function")
                                .and_then(Value::as_str)
                                .is_none_or(|selected| selected == name.as_str())
                                && parent
                                    .function_prefix
                                    .as_ref()
                                    .is_none_or(|prefix| operation_prefix(prefix, name))
                        })
                        .filter_map(|(_, signature)| {
                            let count = program
                                .source_call_arity(signature, arguments, committed, expected)?;
                            let first = signature.arguments(1).ok()?;
                            let last = signature.arguments(program.body.max_context).ok()?;
                            Some(
                                (opened..count)
                                    .map(|slot| {
                                        leaves
                                            .minimum(self, [first[slot].into(), last[slot].into()])
                                    })
                                    .sum::<usize>(),
                            )
                        })
                        .min();
                    let Some(remaining) = remaining else {
                        continue;
                    };
                    future += remaining;
                }
                minimum = Some(minimum.map_or(future, |previous: usize| previous.min(future)));
            }
            let Some(minimum) = minimum else {
                return Err(serde_json::Error::custom(
                    "expression has no source-compatible node obligation domain",
                ));
            };
            necessary += minimum;
            if active_open {
                ancestors.push(typed::Ancestor {
                    fields: parent.selected.clone(),
                    operand: parent.operand.as_ref().unwrap().clone(),
                });
            }
        }
        if necessary > ACQUIRED_NODES {
            return Err(serde_json::Error::custom(
                "mandatory unopened expressions exceed the shared source node allowance",
            ));
        }
        Ok(())
    }
}
