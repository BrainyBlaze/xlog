//! Native causal decoder lowering and streamed checkpoint input.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path};

use safetensors::tensor::Dtype;
use serde_json::Value;
use sha2::{Digest, Sha256};
use xlog_core::{Result, XlogError};
use xlog_ir::tensor::{TensorOperation as Op, TensorParameter, TensorProgram};

use crate::tensor_archive::{TensorArchive, MAX_HEADER_BYTES as MAX_JSON_BYTES};

/// The actual files and typed numerical program admitted before GPU allocation.
pub struct CausalDecoderCheckpoint {
    /// Executable program whose indices preserve parameter sharing.
    pub program: TensorProgram,
    shards: BTreeMap<String, TensorArchive>,
    locations: BTreeMap<String, String>,
    duplicated_head: bool,
    configuration: Vec<u8>,
}

fn invalid(message: impl Into<String>) -> XlogError {
    XlogError::Compilation(message.into())
}

fn io_error(error: std::io::Error) -> XlogError {
    invalid(format!("decoder checkpoint I/O: {error}"))
}

fn read_json_file(path: &Path) -> Result<Vec<u8>> {
    let file = File::open(path).map_err(io_error)?;
    if file.metadata().map_err(io_error)?.len() > MAX_JSON_BYTES {
        return Err(invalid(
            "decoder checkpoint JSON exceeds the admitted header bound",
        ));
    }
    let mut bytes = Vec::new();
    file.take(MAX_JSON_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(io_error)?;
    if bytes.len() as u64 > MAX_JSON_BYTES {
        return Err(invalid(
            "decoder checkpoint JSON grew beyond its admitted bound",
        ));
    }
    Ok(bytes)
}

fn parse_json(bytes: &[u8]) -> Result<Value> {
    serde_json::from_slice(bytes)
        .map_err(|error| invalid(format!("decoder checkpoint JSON: {error}")))
}

fn size(config: &Value, key: &str) -> Result<usize> {
    config
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0 && n <= i32::MAX as usize)
        .ok_or_else(|| {
            invalid(format!(
                "decoder configuration requires positive integer {key}"
            ))
        })
}

fn positive_float(config: &Value, key: &str) -> Result<f32> {
    let value = config
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid(format!("decoder configuration requires numeric {key}")))?
        as f32;
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(invalid(format!(
            "decoder configuration requires finite positive {key}"
        )))
    }
}

/// Lower the admitted Qwen3 numerical definition; no weights are synthesized.
fn lower(config: &Value, max_context: usize) -> Result<TensorProgram> {
    if config["model_type"].as_str() != Some("qwen3")
        || config["hidden_act"].as_str() != Some("silu")
        || config["tie_word_embeddings"].as_bool() != Some(true)
        || config["attention_bias"].as_bool() != Some(false)
        || config["use_sliding_window"].as_bool() != Some(false)
        || !config["rope_scaling"].is_null()
        || !config["rope_parameters"].is_null()
        || config["partial_rotary_factor"]
            .as_f64()
            .is_some_and(|value| value != 1.0)
        || !config["quantization_config"].is_null()
        || config["attention_dropout"].as_f64().unwrap_or(0.0) != 0.0
    {
        return Err(invalid("native decoder requires dense Qwen3, tied BF16 weights, SiLU, full causal attention, and unscaled RoPE"));
    }
    let dtype = config
        .get("dtype")
        .or_else(|| config.get("torch_dtype"))
        .and_then(Value::as_str);
    if dtype != Some("bfloat16") {
        return Err(invalid(
            "native decoder requires an explicitly BF16 checkpoint",
        ));
    }
    let hidden = size(config, "hidden_size")?;
    let intermediate = size(config, "intermediate_size")?;
    let layers = size(config, "num_hidden_layers")?;
    if let Some(layer_types) = config.get("layer_types") {
        let full_attention = layer_types.as_array().is_some_and(|types| {
            types.len() == layers
                && types
                    .iter()
                    .all(|kind| kind.as_str() == Some("full_attention"))
        });
        if !full_attention {
            return Err(invalid(
                "native decoder requires full attention in every layer",
            ));
        }
    }
    let heads = size(config, "num_attention_heads")?;
    let kv_heads = size(config, "num_key_value_heads")?;
    let head_dim = size(config, "head_dim")?;
    let vocabulary = size(config, "vocab_size")?;
    let epsilon = positive_float(config, "rms_norm_eps")?;
    let theta = positive_float(config, "rope_theta")?;
    if max_context == 0
        || max_context > size(config, "max_position_embeddings")?
        || heads % kv_heads != 0
        || head_dim % 2 != 0
    {
        return Err(invalid(
            "decoder context or grouped-query geometry is invalid",
        ));
    }
    let q_width = heads
        .checked_mul(head_dim)
        .ok_or_else(|| invalid("query width overflow"))?;
    let kv_width = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| invalid("key/value width overflow"))?;
    let mut program = TensorProgram {
        parameters: Vec::new(),
        operations: Vec::new(),
        output: 0,
        vocabulary,
        max_context,
    };
    fn parameter(program: &mut TensorProgram, name: String, shape: Vec<usize>) -> usize {
        let id = program.parameters.len();
        program.parameters.push(TensorParameter { name, shape });
        id
    }
    fn emit(program: &mut TensorProgram, operation: Op) -> usize {
        let id = program.operations.len();
        program.operations.push(operation);
        id
    }
    let embedding = parameter(
        &mut program,
        "model.embed_tokens.weight".into(),
        vec![vocabulary, hidden],
    );
    let mut state = emit(&mut program, Op::Embedding { weight: embedding });
    for layer in 0..layers {
        let prefix = format!("model.layers.{layer}");
        let norm = parameter(
            &mut program,
            format!("{prefix}.input_layernorm.weight"),
            vec![hidden],
        );
        let q = parameter(
            &mut program,
            format!("{prefix}.self_attn.q_proj.weight"),
            vec![q_width, hidden],
        );
        let k = parameter(
            &mut program,
            format!("{prefix}.self_attn.k_proj.weight"),
            vec![kv_width, hidden],
        );
        let v = parameter(
            &mut program,
            format!("{prefix}.self_attn.v_proj.weight"),
            vec![kv_width, hidden],
        );
        let q_norm = parameter(
            &mut program,
            format!("{prefix}.self_attn.q_norm.weight"),
            vec![head_dim],
        );
        let k_norm = parameter(
            &mut program,
            format!("{prefix}.self_attn.k_norm.weight"),
            vec![head_dim],
        );
        let projection = parameter(
            &mut program,
            format!("{prefix}.self_attn.o_proj.weight"),
            vec![hidden, q_width],
        );
        let post_norm = parameter(
            &mut program,
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![hidden],
        );
        let gate = parameter(
            &mut program,
            format!("{prefix}.mlp.gate_proj.weight"),
            vec![intermediate, hidden],
        );
        let up = parameter(
            &mut program,
            format!("{prefix}.mlp.up_proj.weight"),
            vec![intermediate, hidden],
        );
        let down = parameter(
            &mut program,
            format!("{prefix}.mlp.down_proj.weight"),
            vec![hidden, intermediate],
        );
        let normalized = emit(
            &mut program,
            Op::RmsNorm {
                input: state,
                weight: norm,
                epsilon,
            },
        );
        let query = emit(
            &mut program,
            Op::Linear {
                input: normalized,
                weight: q,
            },
        );
        let key = emit(
            &mut program,
            Op::Linear {
                input: normalized,
                weight: k,
            },
        );
        let value = emit(
            &mut program,
            Op::Linear {
                input: normalized,
                weight: v,
            },
        );
        let query = emit(
            &mut program,
            Op::RmsNorm {
                input: query,
                weight: q_norm,
                epsilon,
            },
        );
        let key = emit(
            &mut program,
            Op::RmsNorm {
                input: key,
                weight: k_norm,
                epsilon,
            },
        );
        let query = emit(
            &mut program,
            Op::Rotary {
                input: query,
                heads,
                head_dim,
                theta,
            },
        );
        let key = emit(
            &mut program,
            Op::Rotary {
                input: key,
                heads: kv_heads,
                head_dim,
                theta,
            },
        );
        let attended = emit(
            &mut program,
            Op::Attention {
                query,
                key,
                value,
                heads,
                kv_heads,
                head_dim,
            },
        );
        let attended = emit(
            &mut program,
            Op::Linear {
                input: attended,
                weight: projection,
            },
        );
        state = emit(
            &mut program,
            Op::Add {
                left: state,
                right: attended,
            },
        );
        let normalized = emit(
            &mut program,
            Op::RmsNorm {
                input: state,
                weight: post_norm,
                epsilon,
            },
        );
        let gate = emit(
            &mut program,
            Op::Linear {
                input: normalized,
                weight: gate,
            },
        );
        let up = emit(
            &mut program,
            Op::Linear {
                input: normalized,
                weight: up,
            },
        );
        let activated = emit(&mut program, Op::SiluGate { gate, value: up });
        let projected = emit(
            &mut program,
            Op::Linear {
                input: activated,
                weight: down,
            },
        );
        state = emit(
            &mut program,
            Op::Add {
                left: state,
                right: projected,
            },
        );
    }
    let norm = parameter(&mut program, "model.norm.weight".into(), vec![hidden]);
    state = emit(
        &mut program,
        Op::RmsNorm {
            input: state,
            weight: norm,
            epsilon,
        },
    );
    state = emit(&mut program, Op::PredictionRow { input: state });
    // The LM head references the embedding parameter ID and therefore its actual storage.
    state = emit(
        &mut program,
        Op::Linear {
            input: state,
            weight: embedding,
        },
    );
    program.output = emit(&mut program, Op::Float32 { input: state });
    program.shapes(1, 0)?;
    Ok(program)
}

impl CausalDecoderCheckpoint {
    /// Read configuration and all headers, checking the entire parameter contract.
    /// File payloads remain on disk until the supplied native writer consumes them.
    pub fn open(directory: &Path, max_context: usize) -> Result<Self> {
        let configuration = read_json_file(&directory.join("config.json"))?;
        let program = lower(&parse_json(&configuration)?, max_context)?;
        let index_path = directory.join("model.safetensors.index.json");
        let mut locations = BTreeMap::new();
        let mut files = BTreeSet::new();
        if index_path.try_exists().map_err(io_error)? {
            let index = parse_json(&read_json_file(&index_path)?)?;
            let weights = index["weight_map"]
                .as_object()
                .ok_or_else(|| invalid("checkpoint index has no weight_map"))?;
            if weights.is_empty() {
                return Err(invalid("checkpoint index weight_map is empty"));
            }
            for (name, file) in weights {
                let file = file
                    .as_str()
                    .ok_or_else(|| invalid("checkpoint shard name must be a string"))?;
                let mut components = Path::new(file).components();
                if !matches!(components.next(), Some(Component::Normal(_)))
                    || components.next().is_some()
                    || !file.ends_with(".safetensors")
                {
                    return Err(invalid(
                        "checkpoint shard must name a local safetensors file",
                    ));
                }
                locations.insert(name.clone(), file.to_owned());
                files.insert(file.to_owned());
            }
        } else {
            files.insert("model.safetensors".into());
        }
        let indexed = !locations.is_empty();
        let mut shards = BTreeMap::new();
        for name in files {
            let archive = TensorArchive::open(&directory.join(&name))?;
            for tensor in archive.metadata.offset_keys() {
                if indexed {
                    if locations.get(&tensor) != Some(&name) {
                        return Err(invalid(format!(
                            "checkpoint index disagrees with shard for {tensor}"
                        )));
                    }
                } else {
                    locations.insert(tensor, name.clone());
                }
            }
            shards.insert(name, archive);
        }
        let duplicated_head = locations.contains_key("lm_head.weight");
        let allowed: BTreeSet<_> = program
            .parameters
            .iter()
            .map(|parameter| parameter.name.as_str())
            .collect();
        for name in locations.keys() {
            if !allowed.contains(name.as_str()) && name != "lm_head.weight" {
                return Err(invalid(format!(
                    "unsupported tensor in native decoder checkpoint: {name}"
                )));
            }
        }
        for parameter in &program.parameters {
            let file = locations
                .get(&parameter.name)
                .ok_or_else(|| invalid(format!("checkpoint is missing {}", parameter.name)))?;
            let info = shards[file]
                .metadata
                .info(&parameter.name)
                .ok_or_else(|| invalid("checkpoint index names an absent tensor"))?;
            if info.dtype != Dtype::BF16 || info.shape != parameter.shape {
                return Err(invalid(format!(
                    "checkpoint dtype/shape differs for {}",
                    parameter.name
                )));
            }
        }
        if duplicated_head {
            let info = shards[&locations["lm_head.weight"]]
                .metadata
                .info("lm_head.weight")
                .ok_or_else(|| invalid("checkpoint index names an absent LM head"))?;
            if info.dtype != Dtype::BF16 || info.shape != program.parameters[0].shape {
                return Err(invalid("tied LM-head dtype/shape differs from embedding"));
            }
        }
        Ok(Self {
            program,
            shards,
            locations,
            duplicated_head,
            configuration,
        })
    }

    /// Stream authentic parameter bytes to an owned native allocation writer.
    /// Returns the digest of the actual configuration and parameters consumed.
    pub fn load<F>(&mut self, mut write: F) -> Result<[u8; 32]>
    where
        F: FnMut(usize, usize, &[u8]) -> Result<()>,
    {
        let mut identity = Sha256::new();
        identity.update((self.configuration.len() as u64).to_le_bytes());
        identity.update(&self.configuration);
        let mut embedding_digest = None;
        for id in 0..self.program.parameters.len() {
            let name = self.program.parameters[id].name.clone();
            identity.update((name.len() as u64).to_le_bytes());
            identity.update(name.as_bytes());
            let digest = self.read_parameter(&name, |offset, bytes| {
                identity.update(bytes);
                write(id, offset, bytes)
            })?;
            if id == 0 {
                embedding_digest = Some(digest);
            }
        }
        if self.duplicated_head {
            let digest = self.read_parameter("lm_head.weight", |_, _| Ok(()))?;
            if Some(digest) != embedding_digest {
                return Err(invalid(
                    "checkpoint declares tied weights but embedding and LM head bytes differ",
                ));
            }
        }
        Ok(identity.finalize().into())
    }

    fn read_parameter<F>(&mut self, name: &str, write: F) -> Result<[u8; 32]>
    where
        F: FnMut(usize, &[u8]) -> Result<()>,
    {
        self.shards
            .get_mut(&self.locations[name])
            .expect("validated shard")
            .read_tensor(name, write)
    }
}
