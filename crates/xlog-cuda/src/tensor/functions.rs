//! FP32 function primitives on the existing recorded native executor.

use super::*;

impl TensorExecutor {
    fn indices(&self, indices: &[u32]) -> Result<TrackedCudaSlice<u32>> {
        let mut result = self.provider.memory().alloc::<u32>(indices.len())?;
        self.provider
            .htod_sync_copy_into_tracked(indices, &mut result)?;
        Ok(result)
    }

    pub(super) fn function_operation(
        &self,
        operation: &Op,
        parameters: &[NativeTensor],
        values: &[Option<NativeTensor>],
        tokens: &TrackedCudaSlice<u32>,
        history_span: Option<(usize, usize)>,
        output: &NativeTensor,
    ) -> Result<()> {
        let input = |id: usize| {
            values[id]
                .as_ref()
                .ok_or_else(|| failure("missing numerical function input"))
        };
        let count = output.elements() as u64;
        let config = grid(output.elements().div_ceil(256))?;
        match operation {
            Op::Zeros { .. } => self.launch(
                self.recorder(&[], output),
                "tensor_fill_f32",
                config,
                (output.pointer(), count),
            ),
            Op::Finite { input: source } => {
                let source = input(*source)?;
                self.require_finite(source, "function result")?;
                self.launch(
                    self.recorder(&[source], output),
                    "tensor_scale_f32",
                    config,
                    (source.pointer(), output.pointer(), count, 1.0f32),
                )
            }
            Op::Parameter { weight } => {
                let source = &parameters[*weight];
                self.launch(
                    self.recorder(&[source], output),
                    "tensor_scale_f32",
                    config,
                    (source.pointer(), output.pointer(), count, 1.0f32),
                )
            }
            Op::TokenIndicators { token_ids } => {
                let (start, end) = history_span
                    .ok_or_else(|| failure("token indicators require a raw history span"))?;
                let ids = self.indices(token_ids)?;
                let mut recorder = self.recorder(&[], output);
                recorder.read(tokens).read(&ids);
                self.launch(
                    recorder,
                    "tensor_token_indicators_f32",
                    config,
                    (
                        tokens.device_ptr_value(),
                        ids.device_ptr_value(),
                        output.pointer(),
                        count,
                        output.shape.columns as u32,
                        start as u32,
                        end as u32,
                    ),
                )
            }
            Op::Broadcast { input: source } => {
                let source = input(*source)?;
                self.launch(
                    self.recorder(&[source], output),
                    "tensor_broadcast_f32",
                    config,
                    (
                        source.pointer(),
                        output.pointer(),
                        count,
                        output.shape.columns as u32,
                    ),
                )
            }
            Op::Tanh { input: source } | Op::Bfloat16 { input: source } => {
                let source = input(*source)?;
                self.launch(
                    self.recorder(&[source], output),
                    if matches!(operation, Op::Tanh { .. }) {
                        "tensor_tanh_f32"
                    } else {
                        "tensor_f32_to_bf16"
                    },
                    config,
                    (source.pointer(), output.pointer(), count),
                )
            }
            Op::MatMul { left, right } => {
                let (left, right) = (input(*left)?, input(*right)?);
                // Row-major C=A*B is column-major C^T=B^T*A^T on the same bytes.
                self.gemm(
                    right,
                    left,
                    output,
                    false,
                    false,
                    right.shape.columns,
                    left.shape.rows,
                    left.shape.columns,
                    1,
                )
            }
            Op::Multiply { left, right }
            | Op::Subtract { left, right }
            | Op::Divide { left, right } => {
                let (left, right) = (input(*left)?, input(*right)?);
                self.launch(
                    self.recorder(&[left, right], output),
                    "tensor_binary_f32",
                    config,
                    (
                        left.pointer(),
                        right.pointer(),
                        output.pointer(),
                        count,
                        binary_kind(operation),
                    ),
                )
            }
            Op::Select {
                condition,
                positive,
                negative,
            } => {
                let (condition, positive, negative) =
                    (input(*condition)?, input(*positive)?, input(*negative)?);
                self.launch(
                    self.recorder(&[condition, positive, negative], output),
                    "tensor_mask_f32",
                    config,
                    (
                        condition.pointer(),
                        positive.pointer(),
                        negative.pointer(),
                        output.pointer(),
                        count,
                    ),
                )
            }
            Op::GatherColumns {
                input: source,
                indices,
            } => {
                let source = input(*source)?;
                let ids = indices
                    .iter()
                    .map(|&id| u32::try_from(id).map_err(failure))
                    .collect::<Result<Vec<_>>>()?;
                let ids = self.indices(&ids)?;
                let mut recorder = self.recorder(&[source], output);
                recorder.read(&ids);
                self.launch(
                    recorder,
                    "tensor_gather_columns_f32",
                    config,
                    (
                        source.pointer(),
                        ids.device_ptr_value(),
                        output.pointer(),
                        count,
                        source.shape.columns as u32,
                        output.shape.columns as u32,
                    ),
                )
            }
            Op::Row { input: source, row } => self.copy_row(input(*source)?, *row, output, 0),
            Op::Stack { inputs } => {
                for (row, &source) in inputs.iter().enumerate() {
                    self.copy_row(input(source)?, 0, output, row)?;
                }
                Ok(())
            }
            _ => Err(failure(
                "numerical calls and arguments must be resolved before execution",
            )),
        }
    }

    fn copy_row(
        &self,
        source: &NativeTensor,
        source_row: usize,
        output: &NativeTensor,
        output_row: usize,
    ) -> Result<()> {
        self.launch(
            self.recorder(&[source], output),
            "tensor_copy_row_f32",
            grid(output.shape.columns.div_ceil(256))?,
            (
                source.pointer(),
                output.pointer(),
                output.shape.columns as u32,
                source_row as u32,
                output_row as u32,
            ),
        )
    }

    pub(super) fn function_vjp(
        &self,
        operation: &Op,
        values: &[Option<NativeTensor>],
        dy: &NativeTensor,
    ) -> Result<Vec<(usize, NativeTensor)>> {
        let input = |id: usize| {
            values[id]
                .as_ref()
                .ok_or_else(|| failure("missing numerical function primal"))
        };
        let mut contributions = Vec::new();
        match operation {
            Op::Bfloat16 { input } | Op::Finite { input } => {
                contributions.push((*input, dy.clone()))
            }
            Op::Tanh { input: source } => {
                let x = input(*source)?;
                let dx = self.float_shape(x.shape)?;
                self.launch(
                    self.recorder(&[x, dy], &dx),
                    "tensor_tanh_vjp_f32",
                    grid(dx.elements().div_ceil(256))?,
                    (
                        x.pointer(),
                        dy.pointer(),
                        dx.pointer(),
                        dx.elements() as u64,
                    ),
                )?;
                contributions.push((*source, dx));
            }
            Op::MatMul { left, right } => {
                let (a, b) = (input(*left)?, input(*right)?);
                let (da, db) = (self.float_shape(a.shape)?, self.float_shape(b.shape)?);
                // dA=dC*B^T and dB=A^T*dC, using the forward row-major storage.
                self.gemm(
                    b,
                    dy,
                    &da,
                    true,
                    false,
                    a.shape.columns,
                    a.shape.rows,
                    b.shape.columns,
                    1,
                )?;
                self.gemm(
                    dy,
                    a,
                    &db,
                    false,
                    true,
                    b.shape.columns,
                    a.shape.columns,
                    a.shape.rows,
                    1,
                )?;
                contributions.extend([(*left, da), (*right, db)]);
            }
            Op::Multiply { left, right }
            | Op::Subtract { left, right }
            | Op::Divide { left, right } => {
                let (x, y) = (input(*left)?, input(*right)?);
                let (dx, dy_right) = (self.float_shape(dy.shape)?, self.float_shape(dy.shape)?);
                let mut recorder = self.recorder(&[x, y, dy], &dx);
                recorder.write(dy_right.storage.as_ref());
                self.launch(
                    recorder,
                    "tensor_binary_vjp_f32",
                    grid(dx.elements().div_ceil(256))?,
                    (
                        x.pointer(),
                        y.pointer(),
                        dy.pointer(),
                        dx.pointer(),
                        dy_right.pointer(),
                        dx.elements() as u64,
                        binary_kind(operation),
                    ),
                )?;
                contributions.extend([(*left, dx), (*right, dy_right)]);
            }
            Op::Select {
                condition,
                positive,
                negative,
            } => {
                let condition = input(*condition)?;
                for (source, branch) in [(*positive, 1u32), (*negative, 0u32)] {
                    let dx = self.float_shape(dy.shape)?;
                    self.launch(
                        self.recorder(&[condition, dy], &dx),
                        "tensor_mask_vjp_f32",
                        grid(dx.elements().div_ceil(256))?,
                        (
                            condition.pointer(),
                            dy.pointer(),
                            dx.pointer(),
                            dx.elements() as u64,
                            branch,
                        ),
                    )?;
                    contributions.push((source, dx));
                }
            }
            Op::Broadcast { input: source } => {
                let dx = self.float_shape(input(*source)?.shape)?;
                self.launch(
                    self.recorder(&[dy], &dx),
                    "tensor_broadcast_vjp_f32",
                    grid(dx.elements().div_ceil(256))?,
                    (
                        dy.pointer(),
                        dx.pointer(),
                        dy.shape.rows as u32,
                        dx.shape.columns as u32,
                    ),
                )?;
                contributions.push((*source, dx));
            }
            Op::GatherColumns {
                input: source,
                indices,
            } => {
                let dx = self.zeros(input(*source)?.shape)?;
                let ids = indices
                    .iter()
                    .map(|&id| u32::try_from(id).map_err(failure))
                    .collect::<Result<Vec<_>>>()?;
                let ids = self.indices(&ids)?;
                let mut recorder = LaunchRecorder::new_strict(self.stream);
                recorder
                    .read(dy.storage.as_ref())
                    .read(&ids)
                    .read_write(dx.storage.as_ref());
                self.launch(
                    recorder,
                    "tensor_gather_columns_vjp_f32",
                    grid(dy.elements().div_ceil(256))?,
                    (
                        dy.pointer(),
                        ids.device_ptr_value(),
                        dx.pointer(),
                        dy.elements() as u64,
                        dx.shape.columns as u32,
                        dy.shape.columns as u32,
                    ),
                )?;
                contributions.push((*source, dx));
            }
            Op::Row { input: source, row } => {
                let dx = self.zeros(input(*source)?.shape)?;
                self.copy_row(dy, 0, &dx, *row)?;
                contributions.push((*source, dx));
            }
            Op::Stack { inputs } => {
                for (row, &source) in inputs.iter().enumerate() {
                    let dx = self.float_shape(input(source)?.shape)?;
                    self.copy_row(dy, row, &dx, 0)?;
                    contributions.push((source, dx));
                }
            }
            Op::Zeros { .. } | Op::TokenIndicators { .. } => {}
            _ => return Err(failure("unresolved numerical function reverse operation")),
        }
        Ok(contributions)
    }
}

fn binary_kind(operation: &Op) -> u32 {
    match operation {
        Op::Multiply { .. } => 0,
        Op::Subtract { .. } => 1,
        Op::Divide { .. } => 2,
        _ => unreachable!("binary arithmetic dispatcher"),
    }
}
