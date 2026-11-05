//! Reverse execution of the same numerical graph, with FP32 cotangents.

use super::*;

/// A supervised loss on the complete native vocabulary row.
#[derive(Clone, Copy)]
pub enum TensorLoss {
    /// Cross entropy over every original vocabulary coordinate.
    Token {
        /// Original vocabulary ID of the supervised next token.
        target: u32,
    },
    /// Bernoulli NLL of sigmoid(logit[positive] - logit[negative]).
    Binary {
        /// Original ID whose logit is subtracted.
        negative: u32,
        /// Original ID whose logit represents outcome one.
        positive: u32,
        /// Observed binary target.
        outcome: bool,
    },
}

impl TensorLoss {
    fn arguments(self, vocabulary: usize) -> Result<(u32, u32, u32, u32)> {
        match self {
            Self::Token { target } if (target as usize) < vocabulary => Ok((0, target, 0, 0)),
            Self::Binary {
                negative,
                positive,
                outcome,
            } if negative != positive
                && (negative as usize) < vocabulary
                && (positive as usize) < vocabulary =>
            {
                Ok((1, u32::from(outcome), negative, positive))
            }
            _ => Err(failure(
                "training target is outside the vocabulary or binary IDs coincide",
            )),
        }
    }
}

/// Immutable per-parameter optimizer version. Forks share old state until update.
#[derive(Clone)]
pub struct TensorAdamState {
    parameter: usize,
    forward: NativeTensor,
    master: NativeTensor,
    first: NativeTensor,
    second: NativeTensor,
    step: u64,
}

impl TensorAdamState {
    /// Retained master, first and second moments in their durable storage order.
    pub fn snapshot_tensors(&self) -> [&NativeTensor; 3] {
        [&self.master, &self.first, &self.second]
    }

    /// Completed updates for the exact retained parameter version.
    pub fn step(&self) -> u64 {
        self.step
    }
}

/// Scalars from a completed minibatch update, before publication by its owner.
pub struct TensorTrainingResult {
    /// Mean pre-update loss across the supplied causal prefixes.
    pub mean_loss: f32,
    /// L2 norm of the mean FP32 gradient, before clipping to one.
    pub gradient_norm: f32,
    /// Completed update count supplied by the publishing model owner.
    pub step: u64,
}

impl TensorExecutor {
    /// Restore optimizer bytes only when they agree with their actual typed
    /// forward view, have finite moments and satisfy the zero-step contract.
    pub fn restore_optimizer(
        &self,
        parameter: usize,
        forward: &NativeTensor,
        master: NativeTensor,
        first: NativeTensor,
        second: NativeTensor,
        step: u64,
    ) -> Result<TensorAdamState> {
        if forward.device != self.provider.device().ordinal()
            || [&master, &first, &second].iter().any(|tensor| {
                !tensor.shape.float32
                    || tensor.device != forward.device
                    || tensor.shape.rows != forward.shape.rows
                    || tensor.shape.columns != forward.shape.columns
            })
        {
            return Err(failure(
                "restored optimizer storage differs from its physical parameter",
            ));
        }
        let mut invalid = self.provider.memory().alloc::<u32>(1)?;
        self.provider
            .htod_sync_copy_into_tracked(&[0u32], &mut invalid)?;
        let mut recorder = LaunchRecorder::new_strict(self.stream);
        for tensor in [forward, &master, &first, &second] {
            recorder.read(tensor.storage.as_ref());
        }
        recorder.read_write(&invalid);
        self.launch(
            recorder,
            "tensor_check_adam_state_f32",
            grid(master.elements().div_ceil(256))?,
            (
                forward.pointer(),
                master.pointer(),
                first.pointer(),
                second.pointer(),
                invalid.device_ptr_value(),
                master.elements() as u64,
                step,
                u32::from(forward.shape.float32),
            ),
        )?;
        let mut flag = [0u32];
        self.provider
            .dtoh_sync_copy_into_tracked(&invalid, &mut flag)?;
        if flag[0] != 0 {
            return Err(failure(
                "snapshot optimizer/master/forward bytes violate the numerical state contract",
            ));
        }
        Ok(TensorAdamState {
            parameter,
            forward: forward.clone(),
            master,
            first,
            second,
            step,
        })
    }

    pub(super) fn float_shape(&self, shape: TensorValueShape) -> Result<NativeTensor> {
        self.allocate_tensor(TensorValueShape {
            float32: true,
            ..shape
        })
    }

    pub(super) fn as_float(&self, input: &NativeTensor) -> Result<NativeTensor> {
        if input.shape.float32 {
            return Ok(input.clone());
        }
        let output = self.float_shape(input.shape)?;
        self.launch(
            self.recorder(&[input], &output),
            "tensor_bf16_to_f32",
            grid(output.elements().div_ceil(256))?,
            (input.pointer(), output.pointer(), output.elements() as u64),
        )?;
        Ok(output)
    }

    pub(super) fn zeros(&self, shape: TensorValueShape) -> Result<NativeTensor> {
        let output = self.float_shape(shape)?;
        self.launch(
            self.recorder(&[], &output),
            "tensor_fill_f32",
            grid(output.elements().div_ceil(256))?,
            (output.pointer(), output.elements() as u64),
        )?;
        Ok(output)
    }

    pub(super) fn accumulate(
        &self,
        slot: &mut Option<NativeTensor>,
        contribution: NativeTensor,
    ) -> Result<()> {
        if let Some(previous) = slot.as_ref() {
            if previous.shape != contribution.shape {
                return Err(failure("cotangent shape mismatch"));
            }
            let output = self.float_shape(previous.shape)?;
            self.launch(
                self.recorder(&[previous, &contribution], &output),
                "tensor_accumulate_f32",
                grid(output.elements().div_ceil(256))?,
                (
                    previous.pointer(),
                    contribution.pointer(),
                    output.pointer(),
                    output.elements() as u64,
                ),
            )?;
            *slot = Some(output);
        } else {
            *slot = Some(contribution);
        }
        Ok(())
    }

    pub(super) fn attention_pack(
        &self,
        input: &NativeTensor,
        heads: usize,
        source_heads: usize,
        head_dim: usize,
    ) -> Result<NativeTensor> {
        let output = self.allocate_tensor(TensorValueShape {
            rows: heads * input.shape.rows,
            columns: head_dim,
            float32: input.shape.float32,
        })?;
        self.launch(
            self.recorder(&[input], &output),
            if input.shape.float32 {
                "tensor_attention_pack_f32"
            } else {
                "tensor_attention_pack_bf16"
            },
            grid(output.elements().div_ceil(256))?,
            (
                input.pointer(),
                output.pointer(),
                output.elements() as u64,
                input.shape.rows as u32,
                heads as u32,
                source_heads as u32,
                head_dim as u32,
            ),
        )?;
        Ok(output)
    }

    fn attention_reduce(
        &self,
        input: &NativeTensor,
        sequence: usize,
        heads: usize,
        source_heads: usize,
        head_dim: usize,
    ) -> Result<NativeTensor> {
        let output = self.float_shape(TensorValueShape {
            rows: sequence,
            columns: source_heads * head_dim,
            float32: true,
        })?;
        self.launch(
            self.recorder(&[input], &output),
            "tensor_attention_reduce_f32",
            grid(output.elements().div_ceil(256))?,
            (
                input.pointer(),
                output.pointer(),
                output.elements() as u64,
                sequence as u32,
                heads as u32,
                source_heads as u32,
                head_dim as u32,
            ),
        )?;
        Ok(output)
    }

    pub(super) fn attention_probabilities(
        &self,
        scores: &NativeTensor,
        sequence: usize,
        head_dim: usize,
    ) -> Result<NativeTensor> {
        let output = self.allocate_tensor(scores.shape)?;
        self.launch(
            self.recorder(&[scores], &output),
            "tensor_causal_softmax_bf16",
            grid(scores.shape.rows)?,
            (
                scores.pointer(),
                output.pointer(),
                sequence as u32,
                1.0f32 / (head_dim as f32).sqrt(),
            ),
        )?;
        Ok(output)
    }

    fn attention_vjp(
        &self,
        query: &NativeTensor,
        key: &NativeTensor,
        value: &NativeTensor,
        cotangent: &NativeTensor,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<(NativeTensor, NativeTensor, NativeTensor)> {
        let sequence = query.shape.rows;
        let q = self.attention_pack(query, heads, heads, head_dim)?;
        let k = self.attention_pack(key, heads, kv_heads, head_dim)?;
        let v = self.attention_pack(value, heads, kv_heads, head_dim)?;
        let scores = self.allocate_tensor(TensorValueShape {
            rows: heads * sequence,
            columns: sequence,
            float32: false,
        })?;
        self.gemm(
            &k, &q, &scores, true, false, sequence, sequence, head_dim, heads,
        )?;
        let probabilities =
            self.as_float(&self.attention_probabilities(&scores, sequence, head_dim)?)?;
        let (q, k, v) = (self.as_float(&q)?, self.as_float(&k)?, self.as_float(&v)?);
        let dy = self.attention_pack(cotangent, heads, heads, head_dim)?;
        let dv = self.float_shape(v.shape)?;
        self.gemm(
            &dy,
            &probabilities,
            &dv,
            false,
            true,
            head_dim,
            sequence,
            sequence,
            heads,
        )?;
        let dp = self.float_shape(scores.shape)?;
        self.gemm(
            &v, &dy, &dp, true, false, sequence, sequence, head_dim, heads,
        )?;
        drop(v);
        drop(dy);
        drop(probabilities);
        let ds = self.float_shape(scores.shape)?;
        self.launch(
            self.recorder(&[&scores, &dp], &ds),
            "tensor_causal_softmax_vjp_f32",
            grid(heads * sequence)?,
            (
                scores.pointer(),
                dp.pointer(),
                ds.pointer(),
                sequence as u32,
                1.0f32 / (head_dim as f32).sqrt(),
            ),
        )?;
        drop(dp);
        drop(scores);
        let dq = self.float_shape(q.shape)?;
        self.gemm(
            &k, &ds, &dq, false, false, head_dim, sequence, sequence, heads,
        )?;
        let dk = self.float_shape(k.shape)?;
        self.gemm(
            &q, &ds, &dk, false, true, head_dim, sequence, sequence, heads,
        )?;
        Ok((
            self.attention_reduce(&dq, sequence, heads, heads, head_dim)?,
            self.attention_reduce(&dk, sequence, heads, kv_heads, head_dim)?,
            self.attention_reduce(&dv, sequence, heads, kv_heads, head_dim)?,
        ))
    }

    fn gradients(
        &self,
        program: &TensorProgram,
        parameters: &[NativeTensor],
        trainable: &[usize],
        token_ids: &[u32],
        prediction_position: usize,
        history_span: Option<(usize, usize)>,
        loss: TensorLoss,
    ) -> Result<(Vec<NativeTensor>, f32)> {
        let mut active = vec![false; program.operations.len()];
        let mut retained = vec![false; program.operations.len()];
        for (id, operation) in program.operations.iter().enumerate() {
            active[id] = operation
                .parameter()
                .is_some_and(|parameter| trainable.contains(&parameter))
                || operation.inputs().iter().any(|&input| active[input]);
            if active[id] {
                for input in operation.inputs() {
                    retained[input] = true;
                }
            }
        }
        if !active[program.output] {
            return Err(failure("trainable parameter does not reach the output"));
        }
        let mut values = self.execute_retaining(
            program,
            parameters,
            token_ids,
            prediction_position,
            history_span,
            &retained,
        )?;
        let logits = values[program.output]
            .as_ref()
            .ok_or_else(|| failure("missing training logits"))?;
        let seed = self.float_shape(logits.shape)?;
        let scalar = self.float_shape(TensorValueShape {
            rows: 1,
            columns: 1,
            float32: true,
        })?;
        let (kind, target, negative, positive) = loss.arguments(program.vocabulary)?;
        let mut recorder = self.recorder(&[logits], &seed);
        recorder.write(scalar.storage.as_ref());
        self.launch(
            recorder,
            "tensor_loss_f32",
            grid(1)?,
            (
                logits.pointer(),
                seed.pointer(),
                scalar.pointer(),
                program.vocabulary as u32,
                kind,
                target,
                negative,
                positive,
            ),
        )?;
        let loss = self.read_logits(&scalar, &[0])?[0];
        if !loss.is_finite() {
            return Err(failure("nonfinite supervised loss"));
        }
        let mut cotangents: Vec<Option<NativeTensor>> = vec![None; program.operations.len()];
        cotangents[program.output] = Some(seed);
        let mut parameter_gradients: Vec<Option<NativeTensor>> = vec![None; trainable.len()];
        for id in (0..program.operations.len()).rev() {
            let Some(dy) = cotangents[id].take() else {
                continue;
            };
            let input = |index: usize| {
                values[index]
                    .as_ref()
                    .ok_or_else(|| failure("missing retained primal"))
            };
            let mut contributions = Vec::new();
            match program.operations[id] {
                Op::Linear {
                    input: source,
                    weight,
                } => {
                    let shape = input(source)?.shape;
                    if let Some(slot) = trainable.iter().position(|&id| id == weight) {
                        let x = self.as_float(input(source)?)?;
                        let dw = self.float_shape(parameters[weight].shape)?;
                        self.gemm(
                            &x,
                            &dy,
                            &dw,
                            false,
                            true,
                            shape.columns,
                            dy.shape.columns,
                            shape.rows,
                            1,
                        )?;
                        self.accumulate(&mut parameter_gradients[slot], dw)?;
                    }
                    if active[source] {
                        let w = self.as_float(&parameters[weight])?;
                        let dx = self.float_shape(shape)?;
                        self.gemm(
                            &w,
                            &dy,
                            &dx,
                            false,
                            false,
                            shape.columns,
                            shape.rows,
                            dy.shape.columns,
                            1,
                        )?;
                        contributions.push((source, dx));
                    }
                }
                Op::RmsNorm {
                    input: source,
                    weight,
                    epsilon,
                } => {
                    let x = input(source)?;
                    let w = &parameters[weight];
                    let dx = self.float_shape(x.shape)?;
                    self.launch(
                        self.recorder(&[x, w, &dy], &dx),
                        "tensor_rms_norm_vjp_f32",
                        grid(dx.elements() / w.elements())?,
                        (
                            x.pointer(),
                            w.pointer(),
                            dy.pointer(),
                            dx.pointer(),
                            w.elements() as u32,
                            epsilon,
                        ),
                    )?;
                    contributions.push((source, dx));
                }
                Op::Rotary {
                    input: source,
                    heads,
                    head_dim,
                    theta,
                } => {
                    let dx = self.float_shape(dy.shape)?;
                    self.launch(
                        self.recorder(&[&dy], &dx),
                        "tensor_rotary_vjp_f32",
                        grid(dx.elements().div_ceil(256))?,
                        (
                            dy.pointer(),
                            dx.pointer(),
                            dx.elements() as u64,
                            heads as u32,
                            head_dim as u32,
                            theta,
                        ),
                    )?;
                    contributions.push((source, dx));
                }
                Op::Attention {
                    query,
                    key,
                    value,
                    heads,
                    kv_heads,
                    head_dim,
                } => {
                    let (dq, dk, dv) = self.attention_vjp(
                        input(query)?,
                        input(key)?,
                        input(value)?,
                        &dy,
                        heads,
                        kv_heads,
                        head_dim,
                    )?;
                    contributions.extend([(query, dq), (key, dk), (value, dv)]);
                }
                Op::Add { left, right } => contributions.extend([(left, dy.clone()), (right, dy)]),
                Op::SiluGate { gate, value } => {
                    let (x, y) = (input(gate)?, input(value)?);
                    let (dx, dv) = (self.float_shape(x.shape)?, self.float_shape(y.shape)?);
                    let mut recorder = self.recorder(&[x, y, &dy], &dx);
                    recorder.write(dv.storage.as_ref());
                    self.launch(
                        recorder,
                        "tensor_silu_gate_vjp_f32",
                        grid(dx.elements().div_ceil(256))?,
                        (
                            x.pointer(),
                            y.pointer(),
                            dy.pointer(),
                            dx.pointer(),
                            dv.pointer(),
                            dx.elements() as u64,
                        ),
                    )?;
                    contributions.extend([(gate, dx), (value, dv)]);
                }
                Op::PredictionRow { input: source } => {
                    let dx = self.float_shape(input(source)?.shape)?;
                    self.launch(
                        self.recorder(&[&dy], &dx),
                        "tensor_prediction_row_vjp_f32",
                        grid(dx.elements().div_ceil(256))?,
                        (
                            dy.pointer(),
                            dx.pointer(),
                            dx.elements() as u64,
                            dx.shape.columns as u32,
                            prediction_position as u32,
                        ),
                    )?;
                    contributions.push((source, dx));
                }
                Op::Float32 { input: source } => contributions.push((source, dy)),
                Op::Parameter { weight } => {
                    if let Some(slot) = trainable.iter().position(|&id| id == weight) {
                        self.accumulate(&mut parameter_gradients[slot], dy)?;
                    }
                }
                Op::Embedding { .. } => {
                    return Err(failure(
                        "linear training reached an unsupported embedding gradient",
                    ))
                }
                _ => contributions.extend(self.function_vjp(
                    &program.operations[id],
                    &values,
                    &dy,
                )?),
            }
            for (source, contribution) in contributions {
                if active[source] {
                    self.accumulate(&mut cotangents[source], contribution)?;
                }
            }
            // Every reverse consumer has a larger operation index.
            values[id] = None;
        }
        let gradients = parameter_gradients
            .into_iter()
            .zip(trainable)
            .map(|(gradient, &id)| {
                let gradient = match gradient {
                    Some(gradient) => gradient,
                    None => self.zeros(parameters[id].shape)?,
                };
                self.require_finite(&gradient)?;
                Ok(gradient)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((gradients, loss))
    }

    /// Initialize an FP32 master and zero moments from the actual physical parameter.
    pub fn optimizer(&self, parameter: usize, value: &NativeTensor) -> Result<TensorAdamState> {
        if value.device != self.provider.device().ordinal() {
            return Err(failure("optimizer requires this provider's parameter"));
        }
        let master = self.as_float(value)?;
        self.require_finite(&master)?;
        Ok(TensorAdamState {
            parameter,
            forward: value.clone(),
            master,
            first: self.zeros(value.shape)?,
            second: self.zeros(value.shape)?,
            step: 0,
        })
    }

    /// Differentiate all selected parameters in one reverse pass per prefix,
    /// clip their joint mean gradient, and prepare an atomic set of Adam updates.
    pub fn train_parameters(
        &self,
        parameters: &[NativeTensor],
        states: &[(&TensorAdamState, f32)],
        batch: &[(
            &TensorProgram,
            &[u32],
            usize,
            Option<(usize, usize)>,
            TensorLoss,
        )],
    ) -> Result<(
        Vec<(usize, NativeTensor, TensorAdamState)>,
        TensorTrainingResult,
    )> {
        if batch.is_empty() || states.is_empty() {
            return Err(failure("training requires data and selected parameters"));
        }
        let ids = states
            .iter()
            .map(|(state, _)| state.parameter)
            .collect::<Vec<_>>();
        let mut unique = std::collections::HashSet::new();
        for (state, rate) in states {
            let parameter = parameters
                .get(state.parameter)
                .ok_or_else(|| failure("optimizer parameter is absent"))?;
            if !unique.insert(state.parameter)
                || !rate.is_finite()
                || *rate <= 0.0
                || !Arc::ptr_eq(&parameter.storage, &state.forward.storage)
            {
                return Err(failure(
                    "optimizer identity, uniqueness or learning rate differs",
                ));
            }
            state
                .step
                .checked_add(1)
                .ok_or_else(|| failure("optimizer step overflow"))?;
        }
        let mut sums: Vec<Option<NativeTensor>> = vec![None; ids.len()];
        let mut mean_loss = 0.0f32;
        for &(program, tokens, position, history_span, loss) in batch {
            let (gradients, loss) = self.gradients(
                program,
                parameters,
                &ids,
                tokens,
                position,
                history_span,
                loss,
            )?;
            for (sum, gradient) in sums.iter_mut().zip(gradients) {
                self.accumulate(sum, gradient)?;
            }
            mean_loss += loss / batch.len() as f32;
        }
        if !mean_loss.is_finite() {
            return Err(failure("nonfinite minibatch loss"));
        }
        let mut gradients = Vec::with_capacity(ids.len());
        let mut squared_norm = 0.0f32;
        for sum in sums {
            let sum = sum.ok_or_else(|| failure("empty gradient accumulation"))?;
            let gradient = self.float_shape(sum.shape)?;
            self.launch(
                self.recorder(&[&sum], &gradient),
                "tensor_scale_f32",
                grid(sum.elements().div_ceil(256))?,
                (
                    sum.pointer(),
                    gradient.pointer(),
                    sum.elements() as u64,
                    1.0f32 / batch.len() as f32,
                ),
            )?;
            let partials = self.float_shape(TensorValueShape {
                rows: 1,
                columns: gradient.elements().div_ceil(256),
                float32: true,
            })?;
            self.launch(
                self.recorder(&[&gradient], &partials),
                "tensor_sum_squares_f32",
                grid(partials.elements())?,
                (
                    gradient.pointer(),
                    partials.pointer(),
                    gradient.elements() as u64,
                ),
            )?;
            let sum = self.float_shape(TensorValueShape {
                rows: 1,
                columns: 1,
                float32: true,
            })?;
            self.launch(
                self.recorder(&[&partials], &sum),
                "tensor_sum_f32",
                grid(1)?,
                (
                    partials.pointer(),
                    sum.pointer(),
                    partials.elements() as u64,
                ),
            )?;
            squared_norm += self.read_logits(&sum, &[0])?[0];
            gradients.push(gradient);
        }
        let gradient_norm = squared_norm.sqrt();
        if !gradient_norm.is_finite() {
            return Err(failure("nonfinite global gradient norm"));
        }
        let mut updates = Vec::with_capacity(states.len());
        for ((state, rate), gradient) in states.iter().zip(gradients) {
            let step = state.step + 1;
            let master = self.float_shape(state.master.shape)?;
            let forward = if parameters[state.parameter].shape.float32 {
                master.clone()
            } else {
                self.allocate_tensor(parameters[state.parameter].shape)?
            };
            let next = TensorAdamState {
                parameter: state.parameter,
                forward: forward.clone(),
                master,
                first: self.float_shape(state.first.shape)?,
                second: self.float_shape(state.second.shape)?,
                step,
            };
            let mut recorder = self.recorder(
                &[&state.master, &state.first, &state.second, &gradient],
                &next.master,
            );
            recorder
                .write(next.first.storage.as_ref())
                .write(next.second.storage.as_ref());
            self.launch(
                recorder,
                "tensor_adam_f32",
                grid(gradient.elements().div_ceil(256))?,
                (
                    state.master.pointer(),
                    state.first.pointer(),
                    state.second.pointer(),
                    gradient.pointer(),
                    next.master.pointer(),
                    next.first.pointer(),
                    next.second.pointer(),
                    gradient.elements() as u64,
                    *rate,
                    1.0f32 / gradient_norm.max(1.0),
                    1.0f32 - 0.9f32.powf(step as f32),
                    1.0f32 - 0.999f32.powf(step as f32),
                ),
            )?;
            if !forward.shape.float32 {
                self.launch(
                    self.recorder(&[&next.master], &forward),
                    "tensor_f32_to_bf16",
                    grid(forward.elements().div_ceil(256))?,
                    (
                        next.master.pointer(),
                        forward.pointer(),
                        forward.elements() as u64,
                    ),
                )?;
            }
            self.require_finite(&next.master)?;
            self.require_finite(&next.first)?;
            self.require_finite(&next.second)?;
            self.require_finite(&self.as_float(&forward)?)?;
            updates.push((state.parameter, forward, next));
        }
        Ok((
            updates,
            TensorTrainingResult {
                mean_loss,
                gradient_norm,
                step: states
                    .iter()
                    .map(|(state, _)| state.step + 1)
                    .max()
                    .unwrap_or(0),
            },
        ))
    }
}
