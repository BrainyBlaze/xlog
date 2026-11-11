//! Necessary unopened source occurrences, independent of canonical row reuse.

use super::*;
use xlog_ir::tensor::TensorShapeConstraint;

type ShapeKey = [(Option<usize>, Option<usize>, Option<bool>); 2];

/// Each query owns its environment. No source-conditioned answer enters a
/// generic completion cache or survives into another prefix/definition.
#[derive(Default)]
struct Leaves(BTreeMap<ShapeKey, bool>);

impl Leaves {
    fn lowered_minimum(
        &mut self,
        resources: &Resources<'_>,
        expected: [TensorShapeConstraint; 2],
    ) -> usize {
        let program = resources.program.typed.as_ref().unwrap();
        if resources.value_references {
            if (0..program.body.operations.len()).any(|id| {
                program.reference_possible(
                    id,
                    &resources.signatures,
                    typed::Continuation {
                        ancestors: &[],
                        root: false,
                        partial: None,
                        expected: Some(expected),
                        operation_count: None,
                    },
                )
            }) {
                return 0;
            }
            if !program.body.operations.is_empty() {
                // A new operation can consume a closed reference even when
                // no zero-child operation matches the required result shape.
                return 1;
            }
        }
        self.minimum(resources, expected)
    }

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
        let operation_count = if self.planned_output
            && self.program_definition_index == self.definition_index
            && self.program.parameters_complete
        {
            self.program
                .typed
                .as_ref()
                .and_then(|program| program.planned_output.map(|output| output + 1))
        } else {
            None
        };
        let accounted = self.source_expression.unwrap().1 + parents.len();
        if operation_count.is_some_and(|count| accounted > count) {
            return Err(serde_json::Error::custom(
                "open and emitted operations exceed the selected count",
            ));
        }
        let mut can_grow = nodes.current_root_pending;
        let mut ancestors = Vec::new();
        for (index, parent) in parents.iter().enumerate() {
            let active_open = index + 1 < parents.len()
                || (index + 1 == parents.len() && nodes.reference_pending);
            let mut minimum = None;
            let mut lowered_minimum = None;
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
                let mut candidate_can_grow = false;
                let mut next_child_seen = false;
                let mut next_lowered = 0;
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
                    candidate_can_grow = true;
                    if operation_count.is_some()
                        && index + 1 == parents.len()
                        && !active_open
                        && !next_child_seen
                    {
                        // Only this next child cannot borrow a value from an
                        // earlier unfinished sibling. Later holes stay free.
                        next_lowered = leaves.lowered_minimum(self, expected);
                        next_child_seen = true;
                    }
                    // A preceding unfinished sibling may later provide a
                    // matching reference. Absence among today's closed values
                    // cannot strengthen this future source occurrence to two.
                    future += if self.value_references {
                        1
                    } else {
                        leaves.minimum(self, expected)
                    };
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
                    candidate_can_grow |= fields
                        .get("function")
                        .and_then(Value::as_str)
                        .and_then(|name| self.signatures.get(name))
                        .and_then(|signature| signature.arguments(1).ok())
                        .is_none_or(|arguments| arguments.len() > opened);
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
                                        if self.value_references {
                                            1
                                        } else {
                                            leaves.minimum(
                                                self,
                                                [first[slot].into(), last[slot].into()],
                                            )
                                        }
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
                lowered_minimum = Some(
                    lowered_minimum
                        .map_or(next_lowered, |previous: usize| previous.min(next_lowered)),
                );
                can_grow |= candidate_can_grow;
            }
            let Some(minimum) = minimum else {
                return Err(serde_json::Error::custom(
                    "expression has no source-compatible node obligation domain",
                ));
            };
            necessary += minimum;
            if operation_count.is_some_and(|count| accounted + lowered_minimum.unwrap_or(0) > count)
            {
                return Err(serde_json::Error::custom(
                    "next expression exceeds the selected lowered operation count",
                ));
            }
            if active_open {
                ancestors.push(typed::Ancestor {
                    fields: parent.selected.clone(),
                    operand: parent.operand.as_ref().unwrap().clone(),
                });
            }
        }
        if let Some(count) = operation_count {
            // Open tuples already own source occurrences. Only the remaining
            // unopened operations need more, alongside other pending roots.
            // Reference occurrences can consume this allowance without
            // advancing the selected lowered count.
            let other_roots = nodes
                .pending_roots
                .saturating_sub(usize::from(nodes.current_root_pending));
            necessary = necessary.max(nodes.spent + count.saturating_sub(accounted) + other_roots);
        }
        if necessary > ACQUIRED_NODES {
            return Err(serde_json::Error::custom(
                "mandatory unopened expressions exceed the shared source node allowance",
            ));
        }
        if operation_count.is_some_and(|count| accounted < count) && !can_grow {
            return Err(serde_json::Error::custom(
                "selected expression cannot emit the declared operation count",
            ));
        }
        Ok(())
    }
}
