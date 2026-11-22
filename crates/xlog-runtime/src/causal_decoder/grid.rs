//! Recurrent public colour-grid observations in the native causal backbone.

use super::*;
use xlog_ir::tensor::TensorOperation as Op;
use xlog_logic::ast::FuncDef;
use xlog_neural::causal_snapshot::GridAdaptation;

const COLORS: usize = 16;
const PATCH: usize = 4;
// The target occupies row 1, column 1 of the four-by-four query window.
const TARGET_IN_WINDOW: usize = (PATCH - 1) / 2;
const TARGET_COLOR: usize = (TARGET_IN_WINDOW * PATCH + TARGET_IN_WINDOW) * (COLORS + 1);
const BOARD: usize = 64;
const ACTIONS: usize = 8;
// One channel per colour or padding at each position in the patch.
const PATCH_FEATURES: usize = PATCH * PATCH * (COLORS + 1);
const POSITION: usize = PATCH_FEATURES; // patch y/x, board height/width
const KIND: usize = POSITION + 4; // prior state, patch, frame boundary, query
const TURN_START: usize = KIND + 4;
const ACTION: usize = TURN_START + 1;
const AVAILABLE: usize = ACTION + ACTIONS;
const CLICK: usize = AVAILABLE + ACTIONS;
const TARGET: usize = CLICK + 2;
const TARGET_OFFSET: usize = TARGET + 2;
const FEATURES: usize = TARGET_OFFSET + PATCH * PATCH;
const READOUT: usize = COLORS + 1;

fn invalid(message: &str) -> XlogError {
    XlogError::Compilation(message.into())
}

/// One public move. Only action 6 has a click, in (x,y) order.
#[derive(Clone, Copy)]
pub struct GridAction {
    /// Public action number.
    pub id: usize,
    /// Public click, if the action takes coordinates.
    pub click: Option<(usize, usize)>,
}

impl GridAction {
    fn write(&self, row: &mut [f32], height: usize, width: usize) -> Result<()> {
        if self.id == 0 || self.id >= ACTIONS || (self.id == 6) != self.click.is_some() {
            return Err(invalid("invalid grid move or click coordinates"));
        }
        row[ACTION + self.id] = 1.0;
        if let Some((x, y)) = self.click {
            if x >= width || y >= height {
                return Err(invalid("click outside public board"));
            }
            row[CLICK] = x as f32 / (BOARD - 1) as f32;
            row[CLICK + 1] = y as f32 / (BOARD - 1) as f32;
        }
        Ok(())
    }
}

/// An action-conditioned cell query fixed before seeing its outcome.
#[derive(Clone)]
pub struct GridQuery {
    /// Public legal action types at this decision.
    pub available_actions: Vec<usize>,
    /// Proposed move, independent of the prediction target.
    pub action: GridAction,
    /// Observed cell to predict, in (y,x) order.
    pub target: (usize, usize),
    /// Patch selected for the attention readout, in (patch y, patch x) order.
    pub attention_patch: (usize, usize),
}

struct GridFrame {
    height: usize,
    width: usize,
    cells: Box<[u8]>,
}

impl GridFrame {
    fn new(cells: &[Vec<u8>]) -> Result<Self> {
        let height = cells.len();
        let width = cells.first().map_or(0, Vec::len);
        if !(1..=BOARD).contains(&height)
            || !(1..=BOARD).contains(&width)
            || cells
                .iter()
                .any(|row| row.len() != width || row.iter().any(|&c| c as usize >= COLORS))
        {
            return Err(invalid(
                "expected a rectangular public board of up to 64 by 64 colours in 0..16",
            ));
        }
        Ok(Self {
            height,
            width,
            cells: cells.iter().flatten().copied().collect(),
        })
    }

    fn rows(&self) -> usize {
        2 + self.height.div_ceil(PATCH) * self.width.div_ceil(PATCH)
    }

    fn write_patch(&self, row: &mut [f32; FEATURES], top: isize, left: isize) {
        for dy in 0..PATCH {
            for dx in 0..PATCH {
                let y = (top + dy as isize) as usize;
                let x = (left + dx as isize) as usize;
                let colour = if y < self.height && x < self.width {
                    self.cells[y * self.width + x] as usize
                } else {
                    COLORS
                };
                row[(dy * PATCH + dx) * (COLORS + 1) + colour] = 1.0;
            }
        }
    }

    fn features(&self, query_row: &[f32; FEATURES]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.rows() * FEATURES * 4);
        let mut prior = [0.0f32; FEATURES];
        prior[KIND] = 1.0;
        bytes.extend(prior.iter().flat_map(|v| v.to_le_bytes()));
        for y in (0..self.height).step_by(PATCH) {
            for x in (0..self.width).step_by(PATCH) {
                let mut row = [0.0f32; FEATURES];
                row[KIND + 1] = 1.0;
                row[POSITION] = y as f32 / (BOARD - 1) as f32;
                row[POSITION + 1] = x as f32 / (BOARD - 1) as f32;
                row[POSITION + 2] = self.height as f32 / BOARD as f32;
                row[POSITION + 3] = self.width as f32 / BOARD as f32;
                self.write_patch(&mut row, y as isize, x as isize);
                bytes.extend(row.iter().flat_map(|v| v.to_le_bytes()));
            }
        }
        bytes.extend(query_row.iter().flat_map(|v| v.to_le_bytes()));
        bytes
    }

    fn query(&self, query: &GridQuery, grid: &GridAdaptation) -> Result<([f32; FEATURES], usize)> {
        let (y, x) = query.target;
        let (py, px) = query.attention_patch;
        if y >= self.height
            || x >= self.width
            || py >= self.height.div_ceil(PATCH)
            || px >= self.width.div_ceil(PATCH)
        {
            return Err(invalid(
                "prediction cell or attention patch outside observed board",
            ));
        }
        let mut row = [0.0; FEATURES];
        if grid.color_prior.is_some() {
            self.write_patch(
                &mut row,
                y as isize - TARGET_IN_WINDOW as isize,
                x as isize - TARGET_IN_WINDOW as isize,
            );
        }
        row[KIND + 3] = 1.0;
        row[TARGET] = y as f32 / (BOARD - 1) as f32;
        row[TARGET + 1] = x as f32 / (BOARD - 1) as f32;
        row[TARGET_OFFSET + (y % PATCH) * PATCH + x % PATCH] = 1.0;
        for &id in &query.available_actions {
            if id >= ACTIONS || row[AVAILABLE + id] != 0.0 {
                return Err(invalid("invalid public action set"));
            }
            row[AVAILABLE + id] = 1.0;
        }
        query.action.write(&mut row, self.height, self.width)?;
        if row[AVAILABLE + query.action.id] == 0.0 {
            return Err(invalid("proposed move is unavailable"));
        }
        Ok((row, 1 + py * self.width.div_ceil(PATCH) + px))
    }
}

/// Immutable learned episode state and most recently observed public board.
#[derive(Clone)]
pub struct GridMemory {
    pub(super) state: NativeTensor,
    frame: Arc<GridFrame>,
    pub(super) version: Arc<()>,
    /// Definitions replaced during all successful frame updates leading to this state.
    pub disabled_calls: usize,
}

impl GridMemory {
    /// State position, actual patch positions and final readout position.
    pub fn sequence_rows(&self) -> usize {
        self.frame.rows()
    }
}

/// Joint prediction and attention measurement from the same query forward.
pub struct GridPrediction {
    /// Natural logarithms of the probabilities of the sixteen possible colours.
    pub color_log_probs: Vec<f32>,
    /// Predicted increase in completed levels over the labelled continuation.
    pub expected_progress: f32,
    /// Mean attention probability at the preselected patch in the first layer after insertion.
    pub attention_mass: f32,
    /// Calls replaced during this query alone.
    pub disabled_calls: usize,
}

pub(super) fn append(program: &mut TensorProgram, operation: Op) -> usize {
    let index = program.operations.len();
    program.operations.push(operation);
    index
}

impl NativeCausalModel {
    fn grid_prior_tensor(
        &self,
        counts: &[[u64; COLORS]; COLORS],
        progress_prior: f32,
    ) -> Result<NativeTensor> {
        if !progress_prior.is_finite() || progress_prior <= 0.0 {
            return Err(invalid(
                "grid progress prior must be a finite positive mean",
            ));
        }
        let mut weights = [[0.0f32; COLORS]; READOUT];
        let mean = f64::from(progress_prior);
        // The last coordinate uses inverse softplus so its initial output is the requested mean.
        weights[COLORS].fill((mean + (-(-mean).exp_m1()).ln()) as f32);
        for (before, row) in counts.iter().enumerate() {
            let total = row.iter().map(|&n| n as f64).sum::<f64>() + COLORS as f64;
            for (after, &n) in row.iter().enumerate() {
                weights[after][before] = (((n as f64 + 1.0) / total).ln()) as f32;
            }
        }
        let bytes = weights
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        let mut tensor = self.executor.allocate_tensor(TensorValueShape {
            rows: READOUT,
            columns: COLORS,
            float32: true,
        })?;
        self.executor.write_tensor(&mut tensor, 0, &bytes)?;
        Ok(tensor)
    }

    /// Start a new fit against a frozen transition table with zero output corrections.
    /// Retain the sensor, decoder and editor. Reset the output weights and their Adam moments together.
    pub fn fork_grid_priors(
        &self,
        counts: &[[u64; COLORS]; COLORS],
        progress_prior: f32,
    ) -> Result<Self> {
        self.require_editable_functions()?;
        let grid = self.grid()?;
        let prior = grid
            .color_prior
            .ok_or_else(|| invalid("this grid snapshot uses an uncalibrated readout"))?;
        let mut fork = self.fork_candidate();
        fork.parameters[prior] = self.grid_prior_tensor(counts, progress_prior)?;
        let (zero, optimizer) =
            self.zero_projection(grid.readout, &self.program().parameters[grid.readout])?;
        fork.parameters[grid.readout] = zero;
        fork.optimizers.insert(grid.readout, optimizer);
        fork.version = Arc::new(());
        Ok(fork)
    }

    /// Install trainable observation and prediction matrices around the imported causal layers.
    /// Both matrices, their optimizer and the full numerical definition survive cold restoration.
    pub fn fork_grid(&self, seed: u64, insertion_layer: usize) -> Result<Self> {
        if self.program().signature != TensorSignature::Decoder
            || !self.adaptation.insertions.is_empty()
        {
            return Err(invalid(
                "install the grid boundary before acquiring functions on this model",
            ));
        }
        let mut candidate = self.fork_candidate();
        let mut program = self.program().clone();
        let original = std::mem::take(&mut program.operations);
        if insertion_layer.saturating_add(1)
            >= original
                .iter()
                .filter(|op| matches!(op, Op::Attention { .. }))
                .count()
        {
            return Err(invalid(
                "grid readout requires an attention layer after feature insertion",
            ));
        }
        let (Some(Op::Embedding { weight: embedding }), Some(Op::Float32 { input: logits })) =
            (original.first(), original.get(program.output))
        else {
            return Err(invalid("expected the imported causal decoder boundary"));
        };
        let Some(Op::Linear { input: row, weight }) = original.get(*logits) else {
            return Err(invalid("missing tied language projection"));
        };
        if weight != embedding || !matches!(original.get(*row), Some(Op::PredictionRow { .. })) {
            return Err(invalid(
                "tied prediction row differs from the imported decoder",
            ));
        }
        let hidden = program.parameters[*embedding].linear_dimensions()?[1];
        let mut rng = NumericalSeed(seed);
        let sensor = candidate.add_matrix(
            &mut program,
            "model::grid::sensor".into(),
            hidden,
            FEATURES,
            &mut rng,
        )?;
        let readout = program.parameters.len();
        let head = xlog_ir::tensor::TensorParameter {
            name: "model::grid::readout".into(),
            shape: vec![READOUT, hidden],
            float32: true,
        };
        let (zero, optimizer) = candidate.zero_projection(readout, &head)?;
        candidate.parameters.push(zero);
        candidate.optimizers.insert(readout, optimizer);
        program.parameters.push(head);
        candidate.adaptation.trainable.extend([sensor, readout]);
        let color_prior = program.parameters.len();
        candidate
            .parameters
            .push(candidate.grid_prior_tensor(&[[0; COLORS]; COLORS], 1.0 / 32.0)?);
        program.parameters.push(xlog_ir::tensor::TensorParameter {
            name: "model::grid::prediction_prior".into(),
            shape: vec![READOUT, COLORS],
            float32: true,
        });
        program.signature = TensorSignature::Observations {
            input_widths: vec![FEATURES, hidden],
            output_rows: 1,
            output_width: READOUT,
        };
        let features = append(&mut program, Op::Argument { argument: 0 });
        let prior = append(&mut program, Op::Argument { argument: 1 });
        let embedded = append(
            &mut program,
            Op::Linear {
                input: features,
                weight: sensor,
            },
        );
        let residual = append(
            &mut program,
            Op::Add {
                left: embedded,
                right: prior,
            },
        );
        let mut mapping = vec![append(&mut program, Op::Bfloat16 { input: residual })];
        let parameter_ids = (0..program.parameters.len()).collect::<Vec<_>>();
        for op in &original[1..=*row] {
            let remapped = op.remap(&mapping, &parameter_ids);
            mapping.push(append(&mut program, remapped));
        }
        let state = append(
            &mut program,
            Op::Float32 {
                input: mapping[*row],
            },
        );
        let correction = append(
            &mut program,
            Op::Linear {
                input: state,
                weight: readout,
            },
        );
        let query = append(&mut program, Op::PredictionRow { input: features });
        let current_color = append(
            &mut program,
            Op::GatherColumns {
                input: query,
                indices: (TARGET_COLOR..TARGET_COLOR + COLORS).collect(),
            },
        );
        let prior = append(
            &mut program,
            Op::Linear {
                input: current_color,
                weight: color_prior,
            },
        );
        let logits = append(
            &mut program,
            Op::Add {
                left: correction,
                right: prior,
            },
        );
        program.output = append(
            &mut program,
            Op::SoftplusColumn {
                input: logits,
                column: COLORS,
            },
        );
        let entrypoint = String::from("model::grid");
        let mut functions = (*self.functions).clone();
        functions
            .register(FuncDef::numerical(entrypoint.clone(), program)?)
            .map_err(|e| XlogError::Compilation(e.to_string()))?;
        functions.compile_numerical(&entrypoint, 1, 0, Some((0, 1)))?;
        candidate.functions = Arc::new(functions);
        candidate.entrypoint = entrypoint;
        candidate.version = Arc::new(());
        candidate.adaptation.grid = Some(GridAdaptation {
            sensor,
            readout,
            color_prior: Some(color_prior),
            insertion_layer,
        });
        candidate.install_editor(&mut rng)?;
        Ok(candidate)
    }

    pub(super) fn grid(&self) -> Result<&GridAdaptation> {
        self.adaptation
            .grid
            .as_ref()
            .ok_or_else(|| invalid("install or restore the numerical grid boundary first"))
    }

    fn grid_inputs(
        &self,
        frame: &GridFrame,
        memory: Option<&GridMemory>,
        row: &[f32; FEATURES],
    ) -> Result<[NativeTensor; 2]> {
        let grid = self.grid()?;
        let rows = frame.rows();
        if rows > self.max_context() {
            return Err(invalid(
                "observation exceeds this checkpoint's admitted context",
            ));
        }
        if memory.is_some_and(|memory| !Arc::ptr_eq(&memory.version, &self.version)) {
            return Err(invalid(
                "replay observations on the selected model version before using memory",
            ));
        }
        let bytes = frame.features(row);
        let mut features = self.executor.allocate_tensor(TensorValueShape {
            rows,
            columns: FEATURES,
            float32: true,
        })?;
        self.executor.write_tensor(&mut features, 0, &bytes)?;
        let hidden = self.program().parameters[grid.sensor].shape[0];
        let prior = self
            .executor
            .state_prefix(rows, hidden, memory.map(|m| &m.state))?;
        Ok([features, prior])
    }

    fn grid_program(&self, rows: usize, disabled: Option<&str>) -> Result<(TensorProgram, usize)> {
        self.functions.compile_numerical_with_intervention(
            &self.entrypoint,
            rows,
            rows - 1,
            Some((0, rows)),
            disabled,
        )
    }

    /// Advance the retained state once for this public frame and actual preceding move.
    pub fn observe_grid(
        &self,
        cells: &[Vec<u8>],
        memory: Option<&GridMemory>,
        turn_start: bool,
        executed_action: Option<GridAction>,
        disabled: Option<&str>,
    ) -> Result<GridMemory> {
        let frame = GridFrame::new(cells)?;
        if (memory.is_none() && !turn_start)
            || (memory.is_some() && turn_start) != executed_action.is_some()
        {
            return Err(invalid(
                "only the first frame of an executed move carries its action",
            ));
        }
        let mut row = [0.0; FEATURES];
        row[KIND + 2] = 1.0;
        row[TURN_START] = u8::from(turn_start) as f32;
        if let (Some(action), Some(prior)) = (executed_action, memory) {
            action.write(&mut row, prior.frame.height, prior.frame.width)?;
        }
        let inputs = self.grid_inputs(&frame, memory, &row)?;
        let rows = frame.rows();
        let (mut program, substitutions) = self.grid_program(rows, disabled)?;
        let disabled_calls = memory
            .map_or(0, |m| m.disabled_calls)
            .checked_add(substitutions)
            .ok_or_else(|| invalid("grid intervention counter overflow"))?;
        let grid = self.grid()?;
        program.output = grid.prediction_state(&program)?;
        program.operations.truncate(program.output + 1);
        if let TensorSignature::Observations { output_width, .. } = &mut program.signature {
            *output_width = program.parameters[grid.sensor].shape[0];
        }
        let (state, _) = self.executor.execute_inputs(
            &program,
            &self.parameters,
            TensorInputs::Observations(&inputs),
            rows - 1,
            Some((0, rows)),
            None,
        )?;
        Ok(GridMemory {
            state,
            frame: Arc::new(frame),
            disabled_calls,
            version: Arc::clone(&self.version),
        })
    }

    /// Read every proposed action from the same immutable predecision state.
    pub fn predict_grid_actions(
        &self,
        memory: &GridMemory,
        queries: &[GridQuery],
        disabled: Option<&str>,
    ) -> Result<Vec<GridPrediction>> {
        let rows = memory.frame.rows();
        let attention_layer = self.grid()?.insertion_layer + 1;
        let (program, disabled_calls) = self.grid_program(rows, disabled)?;
        let indices = (0..READOUT as u32).collect::<Vec<_>>();
        queries
            .iter()
            .map(|query| {
                let (row, patch) = memory.frame.query(query, self.grid()?)?;
                let inputs = self.grid_inputs(&memory.frame, Some(memory), &row)?;
                let (output, mass) = self.executor.execute_inputs(
                    &program,
                    &self.parameters,
                    TensorInputs::Observations(&inputs),
                    rows - 1,
                    Some((0, rows)),
                    Some((attention_layer, patch)),
                )?;
                let logits = self.executor.read_logits(&output, &indices)?;
                let mass =
                    mass.ok_or_else(|| invalid("selected attention layer produced no readout"))?;
                if !logits.iter().all(|v| v.is_finite()) || !(0.0..=1.0).contains(&mass) {
                    return Err(invalid(
                        "nonfinite grid prediction or invalid attention probability",
                    ));
                }
                let maximum = logits[..COLORS]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let denominator = logits[..COLORS]
                    .iter()
                    .map(|v| (v - maximum).exp())
                    .sum::<f32>()
                    .ln();
                let color_log_probs = logits[..COLORS]
                    .iter()
                    .map(|v| (v - maximum) - denominator)
                    .collect();
                Ok(GridPrediction {
                    color_log_probs,
                    expected_progress: logits[COLORS],
                    attention_mass: mass,
                    disabled_calls,
                })
            })
            .collect()
    }

    /// One atomic minibatch update on the action actually played and any observed level gain.
    pub fn fit_grid_batch(
        &mut self,
        examples: &[(GridMemory, GridQuery, u32, Option<f32>)],
    ) -> Result<TensorTrainingResult> {
        self.fit_grid_parameters(examples, self.adaptation.trainable.clone())
    }

    /// Fit the parent control on the same common parameters that a new insertion can update.
    pub fn fit_grid_shared_batch(
        &mut self,
        examples: &[(GridMemory, GridQuery, u32, Option<f32>)],
    ) -> Result<TensorTrainingResult> {
        self.require_editable_functions()?;
        let grid = self.grid()?;
        let mut ids = vec![grid.sensor, grid.readout];
        ids.extend(self.adaptation.down_projection);
        self.fit_grid_parameters(examples, ids)
    }

    fn fit_grid_parameters(
        &mut self,
        examples: &[(GridMemory, GridQuery, u32, Option<f32>)],
        ids: Vec<usize>,
    ) -> Result<TensorTrainingResult> {
        let grid = self.grid()?;
        let samples = examples
            .iter()
            .map(|(memory, query, color, progress)| {
                let rows = memory.frame.rows();
                if grid.color_prior.is_some() && progress.is_some_and(|v| v < 0.0) {
                    return Err(invalid("observed grid progress must be nonnegative"));
                }
                let (row, _) = memory.frame.query(query, grid)?;
                let inputs = self.grid_inputs(&memory.frame, Some(memory), &row)?;
                let (program, _) = self.grid_program(rows, None)?;
                Ok((
                    program,
                    inputs,
                    rows,
                    TensorLoss::CategoricalRegression {
                        target: *color,
                        regression: *progress,
                    },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let batch = samples
            .iter()
            .map(|(program, inputs, rows, loss)| {
                (
                    program,
                    TensorInputs::Observations(inputs),
                    rows - 1,
                    Some((0, *rows)),
                    *loss,
                )
            })
            .collect::<Vec<_>>();
        self.fit_inputs(&ids, &batch)
    }
}
