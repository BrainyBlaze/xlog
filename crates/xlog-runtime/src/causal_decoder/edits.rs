//! Whole typed residual edits and versioned acquisition rules over actual grid memory.

use super::acquired::ACQUIRED_WIDTH;
use super::grid::{append, GridMemory};
use super::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use xlog_ir::tensor::{TensorOperation as Op, TensorParameter};
use xlog_logic::ast::FuncDef;
use xlog_neural::causal_snapshot::EditAdaptation;

mod interaction;

const WIDTH: usize = ACQUIRED_WIDTH;
const DESCRIPTOR: usize = 96;
const FEATURES: &str = "model::edit_features";
const BASE: &str = "model::edit_score";
const INITIAL: &str = "model::initial_rule";

fn invalid(message: impl Into<String>) -> XlogError {
    XlogError::Compilation(message.into())
}

/// A typed operand of a whole edit. Coordinates keep their actual within-row order.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case", deny_unknown_fields)]
pub enum EditOperand {
    /// Read columns of the observed hidden state.
    History {
        /// Selected positions in the memory or residual.
        coordinates: [usize; WIDTH],
    },
    /// Read columns describing the program and proposed edit.
    Descriptor {
        /// Selected positions in the numerical edit description.
        coordinates: [usize; WIDTH],
    },
    /// Invoke a retained function with its declared arguments.
    Definition {
        /// Immutable registry identity.
        name: String,
    },
}

/// Elementwise operations on type-compatible numerical expressions.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EditOperation {
    /// Return the selected argument.
    Read,
    /// Hyperbolic tangent.
    Tanh,
    /// Pointwise sum.
    Add,
    /// Ordered difference.
    Subtract,
    /// Pointwise product.
    Multiply,
    /// Choose each coordinate by the sign of the first argument.
    Select,
}

impl EditOperation {
    const ALL: [Self; 6] = [
        Self::Read,
        Self::Tanh,
        Self::Add,
        Self::Subtract,
        Self::Multiply,
        Self::Select,
    ];
    fn arity(self) -> usize {
        match self {
            Self::Read | Self::Tanh => 1,
            Self::Add | Self::Subtract | Self::Multiply => 2,
            Self::Select => 3,
        }
    }
    fn emit(self, p: &mut TensorProgram, args: &[usize]) -> usize {
        let op = match self {
            Self::Read => return args[0],
            Self::Tanh => Op::Tanh { input: args[0] },
            Self::Add => Op::Add {
                left: args[0],
                right: args[1],
            },
            Self::Subtract => Op::Subtract {
                left: args[0],
                right: args[1],
            },
            Self::Multiply => Op::Multiply {
                left: args[0],
                right: args[1],
            },
            Self::Select => Op::Select {
                condition: args[0],
                positive: args[1],
                negative: args[2],
            },
        };
        append(p, op)
    }
}

/// One complete edit bound to a concrete program, numerical site, operation and operands.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalEdit {
    /// Structural identity of the parent program.
    pub parent_id: String,
    /// Structural identity of this exact edit in that parent.
    pub edit_id: String,
    /// Optional function whose output is extended in a new version.
    pub target: Option<String>,
    /// Output operation in the target or the residual anchor for a new feature.
    pub site: usize,
    /// Typed operation applied at the selected site.
    pub operation: EditOperation,
    /// Ordered concrete operands.
    pub arguments: Vec<EditOperand>,
}

impl NumericalEdit {
    fn identity(&self) -> Result<String> {
        digest(&(
            &self.parent_id,
            &self.target,
            self.site,
            self.operation,
            &self.arguments,
        ))
    }
}

/// Contributions of the ordinary editor and the rule that actually ran.
pub struct EditScore {
    /// Parent's structural identity.
    pub parent_id: String,
    /// Scored edit's structural identity.
    pub edit_id: String,
    /// Output of the ordinary editor.
    pub base_score: f32,
    /// Output of the selected acquisition rule.
    pub rule_delta: f32,
    /// Immutable name used in the native call.
    pub rule_id: String,
}

fn digest(value: &impl Serialize) -> Result<String> {
    let bytes = serde_json::to_vec(value).map_err(|e| invalid(e.to_string()))?;
    Ok(Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

struct EditSite {
    rule: bool,
    index: usize,
    hidden: usize,
    callees: Vec<(String, usize)>,
    rule_inputs: Option<BTreeMap<String, interaction::Sources>>,
}

impl NativeCausalModel {
    fn score_abi(&self, width: usize) -> Result<TensorSignature> {
        let hidden = self.program().parameters[self.grid()?.sensor].shape[0];
        Ok(TensorSignature::Pure {
            input_widths: vec![hidden, DESCRIPTOR],
            output_width: width,
        })
    }

    fn editor(&self) -> Result<&EditAdaptation> {
        self.adaptation
            .editor
            .as_ref()
            .ok_or_else(|| invalid("install the numerical grid editor before selecting edits"))
    }

    /// Install the editor on a previously saved numerical grid, preserving its state.
    pub fn fork_grid_editor(&self, seed: u64) -> Result<Self> {
        let mut model = self.fork_candidate();
        model.install_editor(&mut NumericalSeed(seed))?;
        Ok(model)
    }

    pub(super) fn install_editor(&mut self, seed: &mut NumericalSeed) -> Result<()> {
        self.require_editable_functions()?;
        if self.adaptation.editor.is_some() {
            return Err(invalid("grid editor already installed"));
        }
        let grid = self.grid()?;
        let hidden = self.program().parameters[grid.sensor].shape[0];
        let mut root = self.program().clone();
        let h = self.add_matrix(
            &mut root,
            "model::editor::history".into(),
            WIDTH,
            hidden,
            seed,
        )?;
        let e = self.add_matrix(
            &mut root,
            "model::editor::edit".into(),
            WIDTH,
            DESCRIPTOR,
            seed,
        )?;
        let output = self.add_matrix(&mut root, "model::editor::readout".into(), 1, WIDTH, seed)?;
        let mut registry = (*self.functions).clone();
        let mut features = self.numerical_body(self.score_abi(WIDTH)?);
        features.parameters = vec![root.parameters[h].clone(), root.parameters[e].clone()];
        features.operations = vec![
            Op::Argument { argument: 0 },
            Op::Argument { argument: 1 },
            Op::Linear {
                input: 0,
                weight: 0,
            },
            Op::Linear {
                input: 1,
                weight: 1,
            },
            Op::Add { left: 2, right: 3 },
            Op::Tanh { input: 4 },
        ];
        features.output = 5;
        register(&mut registry, FEATURES, features)?;
        let mut base = self.numerical_body(self.score_abi(1)?);
        base.parameters.push(root.parameters[output].clone());
        base.operations = vec![
            Op::Argument { argument: 0 },
            Op::Argument { argument: 1 },
            Op::Call {
                function: FEATURES.into(),
                arguments: vec![0, 1],
            },
            Op::Linear {
                input: 2,
                weight: 0,
            },
        ];
        base.output = 3;
        register(&mut registry, BASE, base)?;
        let mut initial = self.numerical_body(self.score_abi(1)?);
        initial.operations.push(Op::Zeros { columns: 1 });
        register(&mut registry, INITIAL, initial)?;
        self.adaptation.editor = Some(EditAdaptation {
            base: BASE.into(),
            base_parameters: vec![h, e, output],
            rules: BTreeMap::from([(INITIAL.into(), vec![])]),
        });
        self.publish_edit_root(registry, "model::grid::editor".into(), root)
    }

    fn publish_edit_root(
        &mut self,
        mut registry: FunctionRegistry,
        name: String,
        root: TensorProgram,
    ) -> Result<()> {
        register(&mut registry, &name, root)?;
        registry.compile_numerical(&name, 1, 0, Some((0, 1)))?;
        self.entrypoint = name;
        self.functions = Arc::new(registry);
        self.version = Arc::new(());
        Ok(())
    }

    /// SHA-256 of the ordered function registry and active entrypoint, independent of fitted weights.
    pub fn program_id(&self) -> Result<String> {
        digest(&(&self.entrypoint, self.functions.as_ref()))
    }

    /// Name of the registered zero-valued search rule.
    pub fn initial_rule_id(&self) -> Result<&'static str> {
        self.editor()?;
        Ok(INITIAL)
    }

    fn edit_site(&self, target: Option<&str>) -> Result<EditSite> {
        let editor = self.editor()?;
        let grid = self.grid()?;
        let hidden = self.program().parameters[grid.sensor].shape[0];
        let features = TensorSignature::Causal {
            input_width: hidden,
            output_width: WIDTH,
        };
        let rule = target.is_some_and(|t| editor.rules.contains_key(t));
        let index = if let Some(target) = target {
            let body = self.functions.numerical(target)?;
            if !rule
                && (body.signature != features
                    || !self
                        .adaptation
                        .insertions
                        .iter()
                        .any(|i| i.function == target))
            {
                return Err(invalid(
                    "edit target is not a retained feature or acquisition rule",
                ));
            }
            body.output
        } else if let Some(last) = self
            .adaptation
            .insertions
            .iter()
            .rev()
            .find(|i| i.layer == grid.insertion_layer)
        {
            last.output
        } else {
            let weight_name = format!("model.layers.{}.mlp.down_proj.weight", grid.insertion_layer);
            let p = self.program();
            p.operations.iter().position(|op|matches!(op, Op::Add {right,..} if
                matches!(p.operations[*right],Op::Linear {weight,..} if p.parameters[weight].name==weight_name)))
                .ok_or_else(|| invalid("could not resolve edit residual anchor"))?
        };
        let mut callees = Vec::new();
        if rule {
            callees.push((FEATURES.into(), WIDTH));
            callees.extend(
                editor
                    .rules
                    .keys()
                    .filter(|name| name.as_str() != INITIAL)
                    .map(|name| (name.clone(), 1)),
            );
        } else {
            callees.extend(
                self.adaptation
                    .insertions
                    .iter()
                    .filter(|i| {
                        self.functions
                            .numerical(&i.function)
                            .is_ok_and(|p| p.signature == features)
                    })
                    .map(|i| (i.function.clone(), WIDTH)),
            );
        }
        callees.sort();
        callees.dedup();
        let rule_inputs = if rule && grid.color_prior.is_some() {
            Some(interaction::callees(&self.functions, &callees)?)
        } else {
            None
        };
        Ok(EditSite {
            rule,
            index,
            hidden,
            callees,
            rule_inputs,
        })
    }

    /// Sample complete, shape-admissible edits from the parent's numerical sites and callable library.
    pub fn enumerate_edits(
        &self,
        target: Option<&str>,
        limit: usize,
        seed: u64,
    ) -> Result<Vec<NumericalEdit>> {
        let site = self.edit_site(target)?;
        let parent = self.program_id()?;
        let mut rng = NumericalSeed(
            seed ^ u64::from_str_radix(&parent[..16], 16).map_err(|e| invalid(e.to_string()))?,
        );
        let mut seen = BTreeSet::new();
        let mut edits = Vec::new();
        for _ in 0..limit.saturating_mul(8) {
            if edits.len() == limit {
                break;
            }
            let operation = EditOperation::ALL[rng.index(EditOperation::ALL.len())];
            let arguments = (0..operation.arity())
                .map(|_| {
                    let coordinates = 1 + usize::from(site.rule);
                    let choice = rng.index(coordinates + site.callees.len());
                    if choice >= coordinates {
                        EditOperand::Definition {
                            name: site.callees[choice - coordinates].0.clone(),
                        }
                    } else if choice == 0 {
                        EditOperand::History {
                            coordinates: std::array::from_fn(|_| rng.index(site.hidden)),
                        }
                    } else {
                        EditOperand::Descriptor {
                            coordinates: std::array::from_fn(|_| rng.index(DESCRIPTOR)),
                        }
                    }
                })
                .collect();
            let mut edit = NumericalEdit {
                parent_id: parent.clone(),
                edit_id: String::new(),
                target: target.map(str::to_string),
                site: site.index,
                operation,
                arguments,
            };
            if site
                .rule_inputs
                .as_ref()
                .is_some_and(|inputs| !interaction::permits(&edit, inputs))
            {
                continue;
            }
            edit.edit_id = edit.identity()?;
            if seen.insert(edit.edit_id.clone()) {
                edits.push(edit);
            }
        }
        Ok(edits)
    }

    fn check_edit(&self, edit: &NumericalEdit, parent: &str) -> Result<EditSite> {
        let site = self.edit_site(edit.target.as_deref())?;
        if edit.parent_id != parent
            || edit.edit_id != edit.identity()?
            || edit.site != site.index
            || edit.arguments.len() != edit.operation.arity()
        {
            return Err(invalid(
                "edit identity, target site or operation arity differs from this parent",
            ));
        }
        for argument in &edit.arguments {
            let valid = match argument {
                EditOperand::History { coordinates } => {
                    coordinates.iter().all(|&v| v < site.hidden)
                }
                EditOperand::Descriptor { coordinates } => {
                    site.rule && coordinates.iter().all(|&v| v < DESCRIPTOR)
                }
                EditOperand::Definition { name } => site.callees.iter().any(|(n, _)| n == name),
            };
            if !valid {
                return Err(invalid("edit operand is not available at its typed site"));
            }
        }
        if site
            .rule_inputs
            .as_ref()
            .is_some_and(|inputs| !interaction::permits(edit, inputs))
        {
            return Err(invalid(
                "the new rule term must couple history with differences between edits",
            ));
        }
        Ok(site)
    }

    fn edit_expression(
        &self,
        body: &mut TensorProgram,
        edit: &NumericalEdit,
        site: &EditSite,
    ) -> usize {
        let arguments = if site.rule { vec![0, 1] } else { vec![0] };
        let mut values = Vec::with_capacity(edit.arguments.len());
        for operand in &edit.arguments {
            let op = match operand {
                EditOperand::History { coordinates } => Op::GatherColumns {
                    input: 0,
                    indices: coordinates.to_vec(),
                },
                EditOperand::Descriptor { coordinates } => Op::GatherColumns {
                    input: 1,
                    indices: coordinates.to_vec(),
                },
                EditOperand::Definition { name } => Op::Call {
                    function: name.clone(),
                    arguments: arguments.clone(),
                },
            };
            let mut v = append(body, op);
            if matches!(operand,EditOperand::Definition{name} if site.callees.iter().any(|(n,w)| n==name && *w==1))
            {
                v = append(
                    body,
                    Op::GatherColumns {
                        input: v,
                        indices: vec![0; WIDTH],
                    },
                );
            }
            values.push(v);
        }
        edit.operation.emit(body, &values)
    }

    /// Apply one complete typed edit on a fork, preserving older callable versions and their weights.
    pub fn apply_edit(&self, edit: &NumericalEdit, seed: u64) -> Result<(Self, String)> {
        self.require_editable_functions()?;
        let site = self.check_edit(edit, &self.program_id()?)?;
        let mut rng = NumericalSeed(seed);
        let name = format!(
            "{}::{}",
            if site.rule { "rule" } else { "feature" },
            edit.edit_id
        );
        let signature = if site.rule {
            self.score_abi(1)?
        } else {
            TensorSignature::Causal {
                input_width: site.hidden,
                output_width: WIDTH,
            }
        };
        let mut body = self.numerical_body(signature);
        append(&mut body, Op::Argument { argument: 0 });
        if site.rule {
            append(&mut body, Op::Argument { argument: 1 });
        }
        let expression = self.edit_expression(&mut body, edit, &site);
        let rows = if site.rule { 1 } else { WIDTH };
        let weight_name = format!("{name}::readout");
        let result = append(
            &mut body,
            Op::Linear {
                input: expression,
                weight: 0,
            },
        );
        body.output = if let Some(target) = &edit.target {
            let old = append(
                &mut body,
                Op::Call {
                    function: target.clone(),
                    arguments: if site.rule { vec![0, 1] } else { vec![0] },
                },
            );
            append(
                &mut body,
                Op::Add {
                    left: old,
                    right: result,
                },
            )
        } else {
            result
        };
        body.parameters.push(TensorParameter {
            name: weight_name.clone(),
            shape: vec![rows, WIDTH],
            float32: true,
        });
        if site.rule {
            let mut fork = self.fork_candidate();
            let mut root = self.program().clone();
            let id = fork.add_matrix(&mut root, weight_name, rows, WIDTH, &mut rng)?;
            let mut functions = (*self.functions).clone();
            register(&mut functions, &name, body)?;
            fork.adaptation
                .editor
                .as_mut()
                .expect("selected rule belongs to the editor")
                .rules
                .insert(name.clone(), vec![id]);
            fork.publish_edit_root(functions, format!("model::grid::{}", edit.edit_id), root)?;
            Ok((fork, name))
        } else {
            let weights = (0..rows * WIDTH).map(|_| rng.weight(WIDTH)).collect();
            let definitions = vec![FuncDef::numerical(name.clone(), body)?];
            let insertion_layer = self.grid()?.insertion_layer;
            let model = self.fork_acquired_definitions(
                name.clone(),
                definitions,
                BTreeMap::from([(weight_name, weights)]),
                insertion_layer,
                self.adaptation.down_projection.is_some(),
            )?;
            Ok((model, name))
        }
    }
}

impl NativeCausalModel {
    // Structured operator, site and ordered argument features remain comparable across program versions.
    // Entire registry identities are kept separately for admission and cold dispatch.
    fn edit_descriptor(&self, edit: &NumericalEdit, site: &EditSite) -> Result<[f32; DESCRIPTOR]> {
        let mut values = [0.0; DESCRIPTOR];
        let op = EditOperation::ALL
            .iter()
            .position(|&op| op == edit.operation)
            .expect("complete edit operators");
        values[op] = 1.0;
        values[6] = f32::from(site.rule);
        values[7] = f32::from(edit.target.is_some());
        values[8] = edit.site as f32 / (1.0 + edit.site as f32);
        values[9] = (self.adaptation.insertions.len() as f32).ln_1p();
        values[10] = (self.editor()?.rules.len() as f32).ln_1p();
        values[11] = (self.program().parameters.len() as f32).ln_1p();
        let target = if let Some(name) = &edit.target {
            self.functions.numerical(name)?
        } else {
            self.program()
        };
        let profile = |body: &TensorProgram, dst: &mut [f32]| {
            for operation in &body.operations {
                let slot = match operation {
                    Op::Argument { .. } => 0,
                    Op::Zeros { .. } | Op::Parameter { .. } => 1,
                    Op::GatherColumns { .. } => 2,
                    Op::Linear { .. } | Op::MatMul { .. } => 3,
                    Op::Tanh { .. } | Op::SiluGate { .. } => 4,
                    Op::Add { .. } => 5,
                    Op::Subtract { .. } => 6,
                    Op::Multiply { .. } | Op::Divide { .. } => 7,
                    Op::Call { .. } => 8,
                    Op::Select { .. } => 9,
                    Op::Attention { .. } => 10,
                    _ => 11,
                };
                dst[slot] += 1.0 / (body.operations.len().max(1) as f32);
            }
        };
        profile(target, &mut values[12..24]);
        for (i, operand) in edit.arguments.iter().enumerate() {
            let name_slot = 24 + 8 * i;
            let begin = 48 + WIDTH * i;
            match operand {
                EditOperand::History { coordinates } | EditOperand::Descriptor { coordinates } => {
                    let descriptor = matches!(operand, EditOperand::Descriptor { .. });
                    values[name_slot + usize::from(descriptor)] = 1.0;
                    let width = if descriptor { DESCRIPTOR } else { site.hidden };
                    for (j, &v) in coordinates.iter().enumerate() {
                        values[begin + j] =
                            (2.0 * v as f32 / (width.saturating_sub(1).max(1) as f32)) - 1.0;
                    }
                }
                EditOperand::Definition { name } => {
                    values[name_slot + 2] = 1.0;
                    let body = self.functions.numerical(name)?;
                    profile(body, &mut values[begin..begin + 12]);
                    let id = Sha256::digest(name.as_bytes());
                    for (j, &byte) in id[..4].iter().enumerate() {
                        values[begin + 12 + j] = (byte as f32) / 127.5 - 1.0;
                    }
                    values[name_slot + 3] = (body.operations.len() as f32).ln_1p();
                    values[name_slot + 4] = (body.parameters.len() as f32).ln_1p();
                }
            }
        }
        Ok(values)
    }

    fn score_program(
        &self,
        memory: &GridMemory,
        edits: &[NumericalEdit],
        rule: &str,
        combined: bool,
    ) -> Result<(TensorProgram, Vec<NativeTensor>)> {
        if !Arc::ptr_eq(&self.version, &memory.version) {
            return Err(invalid(
                "edit scoring requires observations from the selected model version",
            ));
        }
        let editor = self.editor()?;
        if !editor.rules.contains_key(rule) {
            return Err(invalid(
                "requested acquisition rule is not registered in this model",
            ));
        }
        let parent = self.program_id()?;
        let hidden = self.program().parameters[self.grid()?.sensor].shape[0];
        let mut widths = vec![hidden];
        widths.resize(1 + edits.len(), DESCRIPTOR);
        let output_rows = edits
            .len()
            .checked_mul(if combined { 1 } else { 2 })
            .ok_or_else(|| invalid("too many edit outputs"))?;
        let mut body = self.numerical_body(TensorSignature::Observations {
            input_widths: widths,
            output_rows,
            output_width: 1,
        });
        body.parameters = self.program().parameters.clone();
        let history = append(&mut body, Op::Argument { argument: 0 });
        let history = append(
            &mut body,
            Op::Row {
                input: history,
                row: 0,
            },
        );
        let mut tensors = vec![memory.state.clone()];
        let mut outputs = Vec::with_capacity(output_rows);
        for (i, edit) in edits.iter().enumerate() {
            let site = self.check_edit(edit, &parent)?;
            let features = self.edit_descriptor(edit, &site)?;
            let bytes = features
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            let mut tensor = self.executor.allocate_tensor(TensorValueShape {
                rows: 1,
                columns: DESCRIPTOR,
                float32: true,
            })?;
            self.executor.write_tensor(&mut tensor, 0, &bytes)?;
            tensors.push(tensor);
            let arg = append(&mut body, Op::Argument { argument: i + 1 });
            let descriptor = append(&mut body, Op::Row { input: arg, row: 0 });
            let args = vec![history, descriptor];
            let base = append(
                &mut body,
                Op::Call {
                    function: editor.base.clone(),
                    arguments: args.clone(),
                },
            );
            let delta = append(
                &mut body,
                Op::Call {
                    function: rule.into(),
                    arguments: args,
                },
            );
            if combined {
                outputs.push(append(
                    &mut body,
                    Op::Add {
                        left: base,
                        right: delta,
                    },
                ));
            } else {
                outputs.extend([base, delta]);
            }
        }
        body.output = append(&mut body, Op::Stack { inputs: outputs });
        let name = "model::query_edit_scores";
        let mut functions = (*self.functions).clone();
        register(&mut functions, name, body)?;
        let compiled = functions.compile_numerical(name, 1, 0, None)?;
        Ok((compiled, tensors))
    }

    /// Execute the ordinary editor and the selected immutable rule together for all proposed edits.
    pub fn score_edits(
        &self,
        memory: &GridMemory,
        edits: &[NumericalEdit],
        rule: &str,
    ) -> Result<Vec<EditScore>> {
        if edits.is_empty() {
            return Ok(vec![]);
        }
        let (p, inputs) = self.score_program(memory, edits, rule, false)?;
        let (tensor, _) = self.executor.execute_inputs(
            &p,
            &self.parameters,
            TensorInputs::Observations(&inputs),
            0,
            None,
            None,
        )?;
        let mut bytes = vec![0; edits.len() * 2 * 4];
        self.executor.read_tensor(&tensor, 0, &mut bytes)?;
        edits
            .iter()
            .zip(bytes.chunks_exact(8))
            .map(|(edit, row)| {
                let base_score = f32::from_le_bytes(row[..4].try_into().expect("scalar extent"));
                let rule_delta = f32::from_le_bytes(row[4..].try_into().expect("scalar extent"));
                if !base_score.is_finite() || !rule_delta.is_finite() {
                    return Err(invalid("nonfinite edit score"));
                }
                Ok(EditScore {
                    parent_id: edit.parent_id.clone(),
                    edit_id: edit.edit_id.clone(),
                    base_score,
                    rule_delta,
                    rule_id: rule.into(),
                })
            })
            .collect()
    }

    /// Fit the lower-loss choice on a measured comparison. A tie makes no optimizer update.
    pub fn fit_edit_preference(
        &mut self,
        memory: &GridMemory,
        left: &NumericalEdit,
        right: &NumericalEdit,
        left_loss: f64,
        right_loss: f64,
        rule: &str,
    ) -> Result<Option<TensorTrainingResult>> {
        if !left_loss.is_finite() || !right_loss.is_finite() || left_loss < 0.0 || right_loss < 0.0
        {
            return Err(invalid(
                "edit preference requires two observed, finite nonnegative losses",
            ));
        }
        if left.edit_id == right.edit_id || left.target != right.target {
            return Err(invalid("preference requires distinct edits to one target"));
        }
        if left_loss == right_loss {
            let editor = self.editor()?;
            if !editor.rules.contains_key(rule) || !Arc::ptr_eq(&memory.version, &self.version) {
                return Err(invalid(
                    "tie refers to another model or an unavailable rule",
                ));
            }
            let parent = self.program_id()?;
            self.check_edit(left, &parent)?;
            self.check_edit(right, &parent)?;
            return Ok(None);
        }
        let (p, inputs) = self.score_program(memory, &[left.clone(), right.clone()], rule, true)?;
        let editor = self.editor()?;
        let ids = if rule == INITIAL {
            &editor.base_parameters
        } else {
            &editor.rules[rule]
        }
        .clone();
        let loss = TensorLoss::WeightedBinary {
            negative: 1,
            positive: 0,
            outcome: left_loss < right_loss,
            weight: (left_loss - right_loss).abs() as f32,
        };
        self.fit_inputs(
            &ids,
            &[(&p, TensorInputs::Observations(&inputs), 0, None, loss)],
        )
        .map(Some)
    }
}
