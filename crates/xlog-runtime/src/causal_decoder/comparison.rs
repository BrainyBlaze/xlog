//! Matched grid predictions with a learned feature or a fixed Elman recurrence.

use super::*;
use xlog_ir::tensor::TensorOperation as Op;
use xlog_neural::causal_snapshot::GridFunctionComparison;

impl NativeCausalModel {
    pub(super) fn require_editable_functions(&self) -> Result<()> {
        if self.adaptation.comparison.is_some() {
            return Err(XlogError::Compilation(
                "a function comparison retains its frozen definitions".into(),
            ));
        }
        Ok(())
    }

    /// Actual feature callee at the fitted connection, including after cold restore.
    pub fn comparison_function(&self) -> Option<&str> {
        self.adaptation
            .comparison
            .as_ref()
            .map(|c| c.function.as_str())
    }

    /// Fork two immutable versions at one insertion. Only the two independently
    /// fitted copies of the same zero-initialized residual projection can change.
    pub fn fork_grid_function_comparison(
        &self,
        definition_id: &str,
        seed: u64,
    ) -> Result<(Self, Self)> {
        self.grid()?;
        self.require_editable_functions()?;
        let insertion = self
            .adaptation
            .insertions
            .iter()
            .find(|entry| entry.function == definition_id)
            .ok_or_else(|| {
                XlogError::Compilation("comparison requires an installed acquired feature".into())
            })?;
        let projection = insertion.projection;
        let mut root = self.program().clone();
        let [hidden, width] = root.parameters[projection].linear_dimensions()?;
        let mut acquired = self.fork_candidate();
        let mut registry = (*self.functions).clone();
        let prefix = format!("model::fixed_recurrence::{seed:016x}");
        let cell_name = format!("{prefix}::cell");
        let mut rng = NumericalSeed(seed);
        let wx = acquired.add_frozen_matrix(
            &mut root,
            format!("{prefix}::input"),
            width,
            hidden,
            &mut rng,
        )?;
        let wh = acquired.add_frozen_matrix(
            &mut root,
            format!("{prefix}::state"),
            width,
            width,
            &mut rng,
        )?;

        let mut cell = self.numerical_body(TensorSignature::Pure {
            input_widths: vec![width, width],
            output_width: width,
        });
        cell.parameters.push(root.parameters[wh].clone());
        cell.operations = vec![
            Op::Argument { argument: 0 },
            Op::Argument { argument: 1 },
            Op::Linear {
                input: 0,
                weight: 0,
            },
            Op::Add { left: 2, right: 1 },
            Op::Tanh { input: 3 },
        ];
        cell.output = 4;
        register(&mut registry, &cell_name, cell)?;
        let mut recurrence = self.numerical_body(TensorSignature::Recurrent {
            input_width: hidden,
            output_width: width,
        });
        recurrence.parameters.push(root.parameters[wx].clone());
        recurrence.operations = vec![
            Op::Argument { argument: 0 },
            Op::Linear {
                input: 0,
                weight: 0,
            },
            Op::Zeros { columns: width },
            Op::Scan {
                input: 1,
                initial: 2,
                function: cell_name,
                max_steps: root.max_context,
            },
        ];
        recurrence.output = 3;
        register(&mut registry, &prefix, recurrence)?;

        let learned_entry = format!("{prefix}::learned_forward");
        let fixed_entry = format!("{prefix}::fixed_forward");
        register(&mut registry, &learned_entry, root.clone())?;
        let call = insertion
            .output
            .checked_sub(3)
            .and_then(|id| root.operations.get_mut(id));
        match call {
            Some(Op::Call { function, .. }) if function == definition_id => {
                *function = prefix.clone()
            }
            _ => {
                return Err(XlogError::Compilation(
                    "selected residual has a different feature call".into(),
                ))
            }
        }
        register(&mut registry, &fixed_entry, root)?;
        for entry in [&learned_entry, &fixed_entry] {
            registry.compile_numerical(entry, 1, 0, Some((0, 1)))?;
        }
        let (zero, adam) =
            self.zero_projection(projection, &self.program().parameters[projection])?;
        acquired.parameters[projection] = zero;
        acquired.optimizers.insert(projection, adam);
        acquired.functions = Arc::new(registry);
        acquired.entrypoint = learned_entry;
        acquired.adaptation.down_projection = None;
        acquired.adaptation.trainable = vec![projection];
        acquired.adaptation.comparison = Some(GridFunctionComparison {
            projection,
            function: definition_id.into(),
        });
        acquired.version = Arc::new(());
        let mut fixed = acquired.fork_candidate();
        fixed.entrypoint = fixed_entry;
        fixed.adaptation.comparison = Some(GridFunctionComparison {
            projection,
            function: prefix,
        });
        fixed.version = Arc::new(());
        Ok((acquired, fixed))
    }
}
