//! Native ownership and lowering of an authored immutable forest declaration.

use super::*;
use std::fmt::Write;
use xlog_cuda::{SemanticProgramImageAdmission, SemanticProgramLeaf, SemanticProgramNode};

const IMAGE_SOURCE_PREFIX: &str = "xlog.program-image.v1:";

#[derive(Clone, Debug, PartialEq, Eq)]
enum DeclaredBlock {
    InitialRules,
    Rules(Vec<u32>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum DeclaredNode {
    Branch { guard: u32, children: [u32; 4] },
    Leaf(SemanticProgramLeaf),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ProgramImageDeclaration {
    value: ColdValue,
    roots: Vec<u32>,
    nodes: Vec<DeclaredNode>,
    blocks: Vec<DeclaredBlock>,
    mutations: u32,
    source: String,
}

fn sequence(value: &ColdValue) -> PyResult<&[ColdValue]> {
    match value {
        ColdValue::Sequence(values) => Ok(values),
        _ => Err(invalid("program image requires an immutable sequence")),
    }
}

fn integer(value: &ColdValue) -> PyResult<u32> {
    match value {
        ColdValue::Integer(text) => text
            .parse::<u32>()
            .map_err(|_| invalid("program image index or capacity exceeds u32")),
        _ => Err(invalid(
            "program image indices require exact unsigned integers",
        )),
    }
}

fn text(value: &ColdValue) -> PyResult<&str> {
    match value {
        ColdValue::Text(value) => Ok(value),
        _ => Err(invalid("program image tags require exact text")),
    }
}

fn immutable(value: &Bound<'_, PyAny>, depth: usize) -> PyResult<()> {
    if depth > 32 {
        return Err(invalid("program image exceeds the cold transport depth"));
    }
    if value.is_exact_instance_of::<PyTuple>() {
        for item in value.cast::<PyTuple>()?.iter() {
            immutable(&item, depth + 1)?;
        }
        Ok(())
    } else if value.is_exact_instance_of::<PyInt>() || value.is_exact_instance_of::<PyString>() {
        Ok(())
    } else {
        Err(invalid(
            "program image admits only exact tuples, integers and text",
        ))
    }
}

impl ProgramImageDeclaration {
    fn from_value(value: ColdValue) -> PyResult<Self> {
        let fields = sequence(&value)?;
        if fields.len() != 4 {
            return Err(invalid(
                "program image requires roots, nodes, rule blocks and mutation capacity",
            ));
        }
        let roots = sequence(&fields[0])?
            .iter()
            .map(integer)
            .collect::<PyResult<Vec<_>>>()?;
        let mut nodes = Vec::new();
        for node in sequence(&fields[1])? {
            let row = sequence(node)?;
            let kind = row
                .first()
                .map(text)
                .transpose()?
                .ok_or_else(|| invalid("program node has no kind"))?;
            nodes.push(match (kind, row.len()) {
                ("branch", 3) => {
                    let children = sequence(&row[2])?;
                    if children.len() != 4 {
                        return Err(invalid(
                            "program guard requires TRUE, FALSE, BOTH and NEITHER children",
                        ));
                    }
                    DeclaredNode::Branch {
                        guard: integer(&row[1])?,
                        children: [
                            integer(&children[0])?,
                            integer(&children[1])?,
                            integer(&children[2])?,
                            integer(&children[3])?,
                        ],
                    }
                }
                ("leaf", 2) => {
                    let effect = sequence(&row[1])?;
                    let tag = effect
                        .first()
                        .map(text)
                        .transpose()?
                        .ok_or_else(|| invalid("program leaf has no effect"))?;
                    let leaf = match (tag, effect.len()) {
                        ("abstain", 1) => SemanticProgramLeaf::Abstain,
                        ("rule_block", 2) => SemanticProgramLeaf::RuleBlock(integer(&effect[1])?),
                        ("evidence", 2) => {
                            let mut events = Vec::new();
                            for event in sequence(&effect[1])? {
                                let event = sequence(event)?;
                                if event.len() != 2 {
                                    return Err(invalid(
                                        "program evidence requires query and polarity",
                                    ));
                                }
                                let polarity = match text(&event[1])? {
                                    "pro" => 1,
                                    "contra" => 2,
                                    _ => {
                                        return Err(invalid(
                                            "program evidence polarity must be pro or contra",
                                        ))
                                    }
                                };
                                events.push((integer(&event[0])?, polarity));
                            }
                            if events.is_empty() {
                                return Err(invalid(
                                    "an evidence leaf requires a genuine support event",
                                ));
                            }
                            SemanticProgramLeaf::Evidence(events)
                        }
                        _ => return Err(invalid("unknown program leaf effect or field count")),
                    };
                    DeclaredNode::Leaf(leaf)
                }
                _ => return Err(invalid("unknown program node kind or field count")),
            });
        }
        let mut blocks = Vec::new();
        for block in sequence(&fields[2])? {
            let row = sequence(block)?;
            let tag = row
                .first()
                .map(text)
                .transpose()?
                .ok_or_else(|| invalid("program rule block has no kind"))?;
            blocks.push(match (tag, row.len()) {
                ("initial_rules", 1) => DeclaredBlock::InitialRules,
                ("rules", 2) => DeclaredBlock::Rules(
                    sequence(&row[1])?
                        .iter()
                        .map(integer)
                        .collect::<PyResult<_>>()?,
                ),
                _ => return Err(invalid("unknown program rule block or field count")),
            });
        }
        let mutations = integer(&fields[3])?;
        if roots.is_empty()
            || nodes.is_empty()
            || blocks.is_empty()
            || roots.iter().any(|&root| root as usize >= nodes.len())
        {
            return Err(invalid(
                "program image requires nonempty roots, nodes and rule blocks",
            ));
        }
        let bytes = checkpoint_cold_value_bytes(&value);
        let length = bytes
            .len()
            .checked_mul(2)
            .and_then(|n| n.checked_add(IMAGE_SOURCE_PREFIX.len()))
            .ok_or_else(|| invalid("program image material extent overflow"))?;
        let mut source = String::with_capacity(length);
        source.push_str(IMAGE_SOURCE_PREFIX);
        for byte in bytes {
            write!(&mut source, "{byte:02x}")
                .map_err(|_| invalid("program image material encoding failed"))?;
        }
        Ok(Self {
            value,
            roots,
            nodes,
            blocks,
            mutations,
            source,
        })
    }

    pub(super) fn from_source(
        source: &str,
        budget: Option<&xlog_cuda::SemanticReplayMaterialBudget>,
    ) -> PyResult<(
        Arc<Self>,
        Option<xlog_cuda::SemanticReplayMaterialReservation>,
    )> {
        let encoded = source
            .strip_prefix(IMAGE_SOURCE_PREFIX)
            .filter(|value| value.len() % 2 == 0)
            .ok_or_else(|| invalid("program image has no canonical source material"))?;
        let digit = |byte: u8| -> PyResult<u8> {
            match byte {
                b'0'..=b'9' => Ok(byte - b'0'),
                b'a'..=b'f' => Ok(byte - b'a' + 10),
                _ => Err(invalid(
                    "program image source material is not canonical lowercase hex",
                )),
            }
        };
        let _hex_staging = budget
            .map(|budget| {
                budget
                    .reserve_auxiliary((encoded.len() / 2) as u64)
                    .map_err(xlog_err)
            })
            .transpose()?;
        let mut bytes = Vec::with_capacity(encoded.len() / 2);
        for pair in encoded.as_bytes().chunks_exact(2) {
            bytes.push((digit(pair[0])? << 4) | digit(pair[1])?);
        }
        let plan = checked_checkpoint_cold_value(&bytes, bytes.len(), false, None)?;
        let extent = plan
            .owned_bytes
            .checked_mul(3)
            .and_then(|bytes| bytes.checked_add((source.len() as u64).checked_mul(2)?))
            .and_then(|bytes| {
                bytes.checked_add(
                    std::mem::size_of::<Self>() as u64 + 2 * std::mem::size_of::<usize>() as u64,
                )
            })
            .ok_or_else(|| invalid("program image declaration allocation extent overflow"))?;
        let owner = budget
            .map(|budget| budget.reserve_auxiliary(extent).map_err(xlog_err))
            .transpose()?;
        let value = checkpoint_cold_value(&bytes, bytes.len())?;
        let declaration = Self::from_value(value)?;
        if declaration.source != source {
            return Err(invalid(
                "program image source material changes its canonical encoding",
            ));
        }
        Ok((Arc::new(declaration), owner))
    }

    pub(super) fn source(&self) -> &str {
        &self.source
    }

    pub(super) fn allocation_bound(&self, program: &SemanticProgramAdmission) -> PyResult<u64> {
        let overflow = || invalid("program image allocation exceeds its original addressable pool");
        let mut bytes = std::mem::size_of::<SemanticProgramImageAdmission>();
        let roots = self
            .roots
            .len()
            .checked_add(self.mutations as usize)
            .ok_or_else(overflow)?;
        let nodes = self
            .nodes
            .len()
            .checked_add(self.mutations as usize)
            .ok_or_else(overflow)?;
        let leaves = self
            .blocks
            .len()
            .checked_add(1)
            .and_then(|n| n.checked_add(program.queries.len().checked_mul(2)?))
            .and_then(|n| n.checked_add(self.nodes.len()))
            .ok_or_else(overflow)?;
        for (count, width) in [
            (roots, std::mem::size_of::<u32>()),
            (nodes, std::mem::size_of::<SemanticProgramNode>()),
            (
                program.queries.len(),
                std::mem::size_of::<xlog_cuda::SemanticProgramFact>() + std::mem::size_of::<u32>(),
            ),
            (leaves, std::mem::size_of::<SemanticProgramLeaf>()),
            (self.blocks.len(), std::mem::size_of::<Vec<u32>>()),
            (
                nodes,
                std::mem::size_of::<u8>() + 5 * std::mem::size_of::<(usize, bool)>(),
            ),
            (program.queries.len(), 2 * std::mem::size_of::<(u32, u32)>()),
        ] {
            bytes = count
                .checked_mul(width)
                .and_then(|n| bytes.checked_add(n))
                .ok_or_else(overflow)?;
        }
        for node in &self.nodes {
            if let DeclaredNode::Leaf(SemanticProgramLeaf::Evidence(events)) = node {
                bytes = events
                    .len()
                    .checked_mul(std::mem::size_of::<(u32, u32)>())
                    .and_then(|n| bytes.checked_add(n))
                    .ok_or_else(overflow)?;
            }
        }
        for block in &self.blocks {
            let count = match block {
                DeclaredBlock::InitialRules => program.initial_rules.len(),
                DeclaredBlock::Rules(rules) => rules.len(),
            };
            bytes = count
                .checked_mul(std::mem::size_of::<u32>())
                .and_then(|n| bytes.checked_add(n))
                .ok_or_else(overflow)?;
        }
        u64::try_from(bytes).map_err(|_| overflow())
    }

    pub(super) fn lower(
        &self,
        program: &SemanticProgramAdmission,
    ) -> PyResult<SemanticProgramImageAdmission> {
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            blocks.push(match block {
                DeclaredBlock::InitialRules => (0..u32::try_from(program.initial_rules.len())
                    .map_err(|_| invalid("initial rule arena exceeds u32"))?)
                    .collect(),
                DeclaredBlock::Rules(rules) => rules.clone(),
            });
        }
        // Every schema-legal evidence or block effect has a real cold descriptor,
        // including effects not selected by the initial genome.
        let mut leaves = vec![SemanticProgramLeaf::Abstain];
        for block in 0..blocks.len() {
            leaves.push(SemanticProgramLeaf::RuleBlock(block as u32));
        }
        for guard in 0..program.queries.len() {
            leaves.push(SemanticProgramLeaf::Evidence(vec![(guard as u32, 1)]));
            leaves.push(SemanticProgramLeaf::Evidence(vec![(guard as u32, 2)]));
        }
        for node in &self.nodes {
            if let DeclaredNode::Leaf(leaf) = node {
                if !leaves.contains(leaf) {
                    leaves.push(leaf.clone());
                }
            }
        }
        let nodes = self
            .nodes
            .iter()
            .map(|node| match node {
                DeclaredNode::Branch { guard, children } => SemanticProgramNode::Branch {
                    guard: *guard,
                    children: *children,
                },
                DeclaredNode::Leaf(leaf) => SemanticProgramNode::Leaf {
                    effect: leaves
                        .iter()
                        .position(|value| value == leaf)
                        .expect("original leaf descriptor retained")
                        as u32,
                },
            })
            .collect();
        let mut image = SemanticProgramImageAdmission {
            roots: self.roots.clone(),
            nodes,
            guards: program.queries.clone(),
            guard_records: program.query_records.clone(),
            leaves,
            rule_blocks: blocks,
            mutation_capacity: 0,
            local_mutation_capacity: 0,
            local_root_begin: 0,
            local_node_begin: 0,
            local_guard_begin: 0,
            local_leaf_begin: 0,
            local_block_begin: 0,
        };
        image
            .validate(&program.predicate_arities, program.initial_rules.len())
            .map_err(invalid)?;
        image.reserve_mutations(self.mutations).map_err(invalid)?;
        Ok(image)
    }
}

/// Original authored declaration. Selected genomes remain in the native bank;
/// this immutable cold owner is passed unchanged to dependent task factories.
#[pyclass(name = "SemanticProgramImage", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticProgramImage {
    pub(super) inner: Arc<ProgramImageDeclaration>,
}

#[pymethods]
impl PySemanticProgramImage {
    #[new]
    #[pyo3(signature = (*, roots, nodes, rule_blocks, mutation_capacity))]
    fn new(
        roots: &Bound<'_, PyAny>,
        nodes: &Bound<'_, PyAny>,
        rule_blocks: &Bound<'_, PyAny>,
        mutation_capacity: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let mut budget = 16 * 1024 * 1024;
        let mut fields = Vec::with_capacity(4);
        for field in [roots, nodes, rule_blocks, mutation_capacity] {
            immutable(field, 0)?;
            fields.push(ColdValue::read(field, &mut budget, 0)?);
        }
        Ok(Self {
            inner: Arc::new(ProgramImageDeclaration::from_value(ColdValue::Sequence(
                fields,
            ))?),
        })
    }

    /// Exact native-authored cold material, before context or model binding.
    #[getter]
    fn material(&self, py: Python<'_>) -> Py<PyBytes> {
        PyBytes::new(py, &checkpoint_cold_value_bytes(&self.inner.value)).unbind()
    }

    #[getter]
    fn mutation_capacity(&self) -> u32 {
        self.inner.mutations
    }
}

pub(super) fn declaration(value: &Bound<'_, PyAny>) -> PyResult<Arc<ProgramImageDeclaration>> {
    if !value.is_exact_instance_of::<PySemanticProgramImage>() {
        return Err(invalid(
            "program_image requires its original native SemanticProgramImage",
        ));
    }
    Ok(Arc::clone(
        &value.extract::<PyRef<'_, PySemanticProgramImage>>()?.inner,
    ))
}
