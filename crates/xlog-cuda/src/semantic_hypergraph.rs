use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use sha2::{Digest, Sha256};
use xlog_core::{symbol, RelId, ScalarType, Schema, XlogError};

use crate::launch::LaunchEnqueueError;
use crate::memory::TrackedCudaSlice;
use crate::provider::resident_schedule::{validate_execution_domain, ResidentExecutionDomain};
use crate::semantic_transition::Identity256;
use crate::{
    CudaFunction, CudaKernelProvider, CudaStream, DeviceRepr, DriverError, LaunchAsync,
    LaunchConfig,
};

const MODULE: &str = "xlog_semantic_hypergraph";
const KERNEL: &str = "semantic_hypergraph_execute";
const COMMAND_WORDS: usize = 32;
const RECEIPT_WORDS: usize = 42;
// Version word 13 retains root-local insertion order across physical slot reuse.
const HYPERGRAPH_ABI_GENERATION: u64 = 7;

const CONTROL_WORDS: u64 = 16;
const ROOT_WORDS: u64 = 16;
const CANDIDATE_WORDS: u64 = 16;
const STATEMENT_WORDS: u64 = 8;
const SUPPORT_WORDS: u64 = 17;
const VERSION_WORDS: u64 = 16;

const OP_INITIALIZE: u64 = 1;
const OP_FORK: u64 = 2;
const OP_INSERT_SUPPORT: u64 = 3;
const OP_DISCARD: u64 = 4;
const OP_SEAL: u64 = 5;
const OP_SNAPSHOT: u64 = 6;
const OP_TRUTH: u64 = 7;
const OP_INSPECT_STATEMENT: u64 = 8;
const OP_INSPECT_SUPPORT: u64 = 9;
const OP_INSPECT_VERSION: u64 = 10;
#[cfg(test)]
const OP_PREFLIGHT_TRANSITION: u64 = 11;

const HOST_COMMAND_ADMISSION: u64 = 0;
const RESIDENT_EMPTY_ROOT_HANDLE_ADMISSION: u64 = 1;
const RESIDENT_FORK_ADMISSION: u64 = 2;
const RESIDENT_INSERT_SUPPORT_ADMISSION: u64 = 3;
const RESIDENT_FORK_TRUTH_ADMISSION: u64 = 4;
const RESIDENT_SEAL_ADMISSION: u64 = 5;
const RESIDENT_CONSUME_TRUTH_ADMISSION: u64 = 6;
const RESIDENT_MATERIALIZE_DECODED_ADMISSION: u64 = 7;
const RESIDENT_PREFLIGHT_TRANSITION_ADMISSION: u64 = 8;
const RESIDENT_ROOT_TRUTH_ADMISSION: u64 = 9;

const STATUS_OK: u64 = 0;
const STATUS_FOREIGN_OWNER: u64 = 1;
const STATUS_SLOT_OUT_OF_RANGE: u64 = 2;
const STATUS_STALE_GENERATION: u64 = 3;
const STATUS_INACTIVE: u64 = 4;
const STATUS_NOT_REACHABLE: u64 = 5;
const STATUS_STATEMENT_CAPACITY: u64 = 6;
const STATUS_SUPPORT_CAPACITY: u64 = 7;
const STATUS_VERSION_CAPACITY: u64 = 8;
const STATUS_ROOT_CAPACITY: u64 = 9;
const STATUS_GENERATION_EXHAUSTED: u64 = 10;
const STATUS_CORRUPT_LINEAGE: u64 = 11;
const STATUS_INVALID_COMMAND: u64 = 12;
const STATUS_ARENA_MISMATCH: u64 = 13;

const OUTCOME_INSERTED: u64 = 1;
const OUTCOME_UNCHANGED: u64 = 2;
const INSERT_NEW_STATEMENT: u64 = 1;
const INSERT_PREVIOUS_TRUTH_SHIFT: u32 = 1;
const INSERT_PREVIOUS_TRUTH_MASK: u64 = 3 << INSERT_PREVIOUS_TRUTH_SHIFT;

static NEXT_OWNER_ID: AtomicU64 = AtomicU64::new(1);

/// The four-valued semantic state derived from reachable support events.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SemanticTruth {
    /// Neither positive nor negative support is reachable.
    Neither,
    /// Positive support, and no negative support, is reachable.
    True,
    /// Negative support, and no positive support, is reachable.
    False,
    /// Both positive and negative support are reachable.
    Both,
}

impl SemanticTruth {
    /// Maps exact positive/negative presence bits to the four-valued state.
    pub const fn from_presence(pro: bool, contra: bool) -> Self {
        match (pro, contra) {
            (false, false) => Self::Neither,
            (true, false) => Self::True,
            (false, true) => Self::False,
            (true, true) => Self::Both,
        }
    }

    /// Returns exact positive/negative presence bits.
    pub const fn presence(self) -> (bool, bool) {
        match self {
            Self::Neither => (false, false),
            Self::True => (true, false),
            Self::False => (false, true),
            Self::Both => (true, true),
        }
    }

    fn from_bits(bits: u64) -> Result<Self, SemanticHypergraphError> {
        match bits {
            0 => Ok(Self::Neither),
            1 => Ok(Self::True),
            2 => Ok(Self::False),
            3 => Ok(Self::Both),
            _ => Err(SemanticHypergraphError::CorruptLineage {
                detail: format!("device returned invalid semantic bits {bits}"),
            }),
        }
    }
}

/// Polarity of one exact support event.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SemanticPolarity {
    /// Positive support.
    Pro,
    /// Negative support.
    Contra,
}

impl SemanticPolarity {
    const fn code(self) -> u64 {
        match self {
            Self::Pro => 1,
            Self::Contra => 2,
        }
    }
}

/// One typed argument in a canonical statement identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SemanticArgument {
    U32(u32),
    U64(u64),
    I32(i32),
    I64(i64),
    F32Bits(u32),
    F64Bits(u64),
    Bool(bool),
    Symbol(u32),
}

/// Schema-declared role of a retained typed record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SemanticRecordRole {
    Statement,
    Qualifier,
    Provenance,
    Source,
    Context,
    Scope,
}

impl SemanticRecordRole {
    fn code(self) -> u8 {
        match self {
            Self::Statement => 1,
            Self::Qualifier => 2,
            Self::Provenance => 3,
            Self::Source => 4,
            Self::Context => 5,
            Self::Scope => 6,
        }
    }
}

/// A predicate and its complete existing XLOG schema, not a caller-supplied digest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticPredicateRecord {
    pub predicate: RelId,
    pub role: SemanticRecordRole,
    pub schema: Schema,
}

/// Original typed content. Qualifier indices address this admission's records.
///
/// Only statements carry ordered identity-bearing qualifiers. The referenced
/// records must have a qualifier schema; provenance is carried by support events.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticTypedRecord {
    pub predicate: RelId,
    pub arguments: Vec<SemanticArgument>,
    pub qualifiers: Vec<u32>,
}

/// One unit of support and its complete, role-checked record references.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticSupportRecord {
    pub statement: u32,
    pub polarity: SemanticPolarity,
    pub provenance: u32,
    pub source: u32,
    pub context: u32,
    pub scope: u32,
}

/// Cold input transferred into the semantic owner's immutable admission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticAdmissionRecords {
    pub predicates: Vec<SemanticPredicateRecord>,
    pub records: Vec<SemanticTypedRecord>,
    pub supports: Vec<SemanticSupportRecord>,
}

/// Aggregate cold-input limits, checked before copying symbols or allocating views.
#[derive(Clone, Copy, Debug)]
pub struct SemanticAdmissionLimits {
    /// Total predicate, typed-record and support-record count.
    pub max_records: u32,
    /// Total schema columns, schema key columns and typed arguments.
    pub max_terms: u32,
    /// Total qualifier and support-to-record references.
    pub max_references: u32,
    /// Total column names, sort labels and copied symbol text, including duplicates.
    pub max_utf8_bytes: usize,
}

/// Complete immutable admission and the selected root's chronological insertions.
/// Symbol arguments index `symbols` in occurrence order, never a process registry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SemanticRootMaterial {
    pub(crate) records: SemanticAdmissionRecords,
    pub(crate) symbols: Vec<String>,
    pub(crate) insertions: Vec<SemanticRootInsertion>,
    pub(crate) digest: [u8; 32],
    pub(crate) extents: [u32; 3],
    pub(crate) admission_base_digest: [u8; 32],
    pub(crate) admission_base_extents: [u32; 3],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SemanticRootInsertion {
    /// A genuine original occurrence, independently of equal reconstructed bytes.
    pub(crate) statement: Option<u32>,
    /// Derived predicate, four (record, argument) pairs, and qualifier owner.
    /// The complete typed values remain in this material's immutable admission.
    pub(crate) reconstruction: [u32; 10],
    pub(crate) support: u32,
    pub(crate) version: [u8; 32],
}

/// Bounded little-endian reader shared by native execution material payloads.
pub(crate) struct SemanticMaterialReader<'a> {
    remaining: &'a [u8],
}

impl<'a> SemanticMaterialReader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { remaining: bytes }
    }

    pub(crate) fn take(&mut self, len: usize) -> Result<&'a [u8], SemanticHypergraphError> {
        if len > self.remaining.len() {
            return Err(admission_error("truncated execution material"));
        }
        let (value, rest) = self.remaining.split_at(len);
        self.remaining = rest;
        Ok(value)
    }

    pub(crate) fn u8(&mut self) -> Result<u8, SemanticHypergraphError> {
        Ok(self.take(1)?[0])
    }
    pub(crate) fn u32(&mut self) -> Result<u32, SemanticHypergraphError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub(crate) fn u64(&mut self) -> Result<u64, SemanticHypergraphError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub(crate) fn count(
        &mut self,
        min_bytes_per_item: usize,
    ) -> Result<usize, SemanticHypergraphError> {
        let count = self.u32()? as usize;
        if min_bytes_per_item == 0 || count > self.remaining.len() / min_bytes_per_item {
            return Err(admission_error(
                "execution material count exceeds remaining bytes",
            ));
        }
        Ok(count)
    }
    pub(crate) fn bytes(&mut self) -> Result<&'a [u8], SemanticHypergraphError> {
        let len = self.count(1)?;
        self.take(len)
    }
    pub(crate) fn finish(self) -> Result<(), SemanticHypergraphError> {
        if self.remaining.is_empty() {
            Ok(())
        } else {
            Err(admission_error("trailing execution material bytes"))
        }
    }
}

pub(crate) fn material_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}
pub(crate) fn material_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}
pub(crate) fn material_bytes(
    output: &mut Vec<u8>,
    bytes: &[u8],
) -> Result<(), SemanticHypergraphError> {
    material_count(output, bytes.len())?;
    output.extend_from_slice(bytes);
    Ok(())
}
fn material_count(output: &mut Vec<u8>, count: usize) -> Result<(), SemanticHypergraphError> {
    material_u32(
        output,
        u32::try_from(count).map_err(|_| admission_error("material count exceeds u32"))?,
    );
    Ok(())
}
fn material_budget(remaining: &mut usize, count: usize) -> Result<(), SemanticHypergraphError> {
    *remaining = remaining
        .checked_sub(count)
        .ok_or_else(|| admission_error("execution material exceeds admission bound"))?;
    Ok(())
}
fn material_string(
    reader: &mut SemanticMaterialReader<'_>,
    budget: &mut usize,
) -> Result<String, SemanticHypergraphError> {
    let bytes = reader.bytes()?;
    material_budget(budget, bytes.len())?;
    Ok(std::str::from_utf8(bytes)
        .map_err(|_| admission_error("execution material text is not UTF-8"))?
        .to_owned())
}

impl SemanticRootMaterial {
    fn validate_symbol_indices(&self) -> Result<(), SemanticHypergraphError> {
        let mut occurrence = 0usize;
        for record in &self.records.records {
            for argument in &record.arguments {
                if let SemanticArgument::Symbol(index) = argument {
                    if *index as usize != occurrence || occurrence >= self.symbols.len() {
                        return Err(admission_error(
                            "material symbols are not in canonical occurrence order",
                        ));
                    }
                    occurrence += 1;
                }
            }
        }
        if occurrence != self.symbols.len() {
            return Err(admission_error(
                "material contains unreferenced symbol text",
            ));
        }
        Ok(())
    }

    /// Resolves retained text once for the existing cold typed-admission path.
    #[cfg(test)]
    pub(crate) fn admission_records(
        &self,
    ) -> Result<SemanticAdmissionRecords, SemanticHypergraphError> {
        self.validate_symbol_indices()?;
        let mut records = self.records.clone();
        for record in &mut records.records {
            for argument in &mut record.arguments {
                if let SemanticArgument::Symbol(index) = argument {
                    *index = symbol::intern(&self.symbols[*index as usize]);
                }
            }
        }
        Ok(records)
    }

    pub(crate) fn encode(&self) -> Result<Vec<u8>, SemanticHypergraphError> {
        self.validate_symbol_indices()?;
        let mut out = b"XLOGROOT".to_vec();
        material_u32(&mut out, 2);
        material_count(&mut out, self.records.predicates.len())?;
        for predicate in &self.records.predicates {
            material_u32(&mut out, predicate.predicate.0);
            out.push(predicate.role.code());
            material_count(&mut out, predicate.schema.columns.len())?;
            for ((name, scalar), label) in predicate
                .schema
                .columns
                .iter()
                .zip(predicate.schema.sort_labels())
            {
                material_bytes(&mut out, name.as_bytes())?;
                out.push(scalar.to_code());
                material_bytes(&mut out, label.as_bytes())?;
            }
            material_count(&mut out, predicate.schema.key_columns.len())?;
            for &column in &predicate.schema.key_columns {
                material_count(&mut out, column)?;
            }
        }
        material_count(&mut out, self.records.records.len())?;
        for record in &self.records.records {
            material_u32(&mut out, record.predicate.0);
            material_count(&mut out, record.arguments.len())?;
            for argument in &record.arguments {
                out.push(argument_type(*argument).to_code());
                match argument {
                    SemanticArgument::U32(value)
                    | SemanticArgument::F32Bits(value)
                    | SemanticArgument::Symbol(value) => material_u32(&mut out, *value),
                    SemanticArgument::U64(value) | SemanticArgument::F64Bits(value) => {
                        material_u64(&mut out, *value)
                    }
                    SemanticArgument::I32(value) => material_u32(&mut out, *value as u32),
                    SemanticArgument::I64(value) => material_u64(&mut out, *value as u64),
                    SemanticArgument::Bool(value) => out.push(u8::from(*value)),
                }
            }
            material_count(&mut out, record.qualifiers.len())?;
            for &qualifier in &record.qualifiers {
                material_u32(&mut out, qualifier);
            }
        }
        material_count(&mut out, self.records.supports.len())?;
        for support in &self.records.supports {
            material_u32(&mut out, support.statement);
            out.push(support.polarity.code() as u8);
            for value in [
                support.provenance,
                support.source,
                support.context,
                support.scope,
            ] {
                material_u32(&mut out, value);
            }
        }
        material_count(&mut out, self.symbols.len())?;
        for symbol in &self.symbols {
            material_bytes(&mut out, symbol.as_bytes())?;
        }
        material_count(&mut out, self.insertions.len())?;
        for insertion in &self.insertions {
            if insertion.statement.is_some() && insertion.reconstruction != [0; 10] {
                return Err(admission_error(
                    "original material target has derived reconstruction references",
                ));
            }
            out.push(u8::from(insertion.statement.is_some()));
            if let Some(statement) = insertion.statement {
                material_u32(&mut out, statement);
            }
            for reference in insertion.reconstruction {
                material_u32(&mut out, reference);
            }
            material_u32(&mut out, insertion.support);
            out.extend_from_slice(&insertion.version);
        }
        out.extend_from_slice(&self.digest);
        for extent in self.extents {
            material_u32(&mut out, extent);
        }
        out.extend_from_slice(&self.admission_base_digest);
        for extent in self.admission_base_extents {
            material_u32(&mut out, extent);
        }
        Ok(out)
    }

    pub(crate) fn decode(
        bytes: &[u8],
        limits: SemanticAdmissionLimits,
    ) -> Result<Self, SemanticHypergraphError> {
        let mut reader = SemanticMaterialReader::new(bytes);
        if reader.take(8)? != b"XLOGROOT" || reader.u32()? != 2 {
            return Err(admission_error(
                "unsupported semantic root material encoding",
            ));
        }
        let mut record_budget = limits.max_records as usize;
        let mut term_budget = limits.max_terms as usize;
        let mut reference_budget = limits.max_references as usize;
        let mut text_budget = limits.max_utf8_bytes;
        let count = reader.count(13)?;
        material_budget(&mut record_budget, count)?;
        let mut predicates = Vec::with_capacity(count);
        for _ in 0..count {
            let predicate = RelId(reader.u32()?);
            let role = match reader.u8()? {
                1 => SemanticRecordRole::Statement,
                2 => SemanticRecordRole::Qualifier,
                3 => SemanticRecordRole::Provenance,
                4 => SemanticRecordRole::Source,
                5 => SemanticRecordRole::Context,
                6 => SemanticRecordRole::Scope,
                _ => return Err(admission_error("invalid material record role")),
            };
            let count = reader.count(9)?;
            material_budget(&mut term_budget, count)?;
            let mut columns = Vec::with_capacity(count);
            let mut labels = Vec::with_capacity(count);
            for _ in 0..count {
                let name = material_string(&mut reader, &mut text_budget)?;
                let scalar = ScalarType::from_code(reader.u8()?)
                    .ok_or_else(|| admission_error("invalid material scalar code"))?;
                let label = material_string(&mut reader, &mut text_budget)?;
                if label.trim().is_empty() {
                    return Err(admission_error("empty material sort label"));
                }
                columns.push((name, scalar));
                labels.push(label);
            }
            let mut schema = Schema::new(columns)
                .with_sort_labels(labels)
                .map_err(admission_error)?;
            let count = reader.count(4)?;
            material_budget(&mut term_budget, count)?;
            schema.key_columns.clear();
            for _ in 0..count {
                schema.key_columns.push(reader.u32()? as usize);
            }
            predicates.push(SemanticPredicateRecord {
                predicate,
                role,
                schema,
            });
        }
        let count = reader.count(12)?;
        material_budget(&mut record_budget, count)?;
        let mut records = Vec::with_capacity(count);
        for _ in 0..count {
            let predicate = RelId(reader.u32()?);
            let count = reader.count(2)?;
            material_budget(&mut term_budget, count)?;
            let mut arguments = Vec::with_capacity(count);
            for _ in 0..count {
                let scalar = ScalarType::from_code(reader.u8()?)
                    .ok_or_else(|| admission_error("invalid material scalar code"))?;
                arguments.push(match scalar {
                    ScalarType::U32 => SemanticArgument::U32(reader.u32()?),
                    ScalarType::U64 => SemanticArgument::U64(reader.u64()?),
                    ScalarType::I32 => SemanticArgument::I32(reader.u32()? as i32),
                    ScalarType::I64 => SemanticArgument::I64(reader.u64()? as i64),
                    ScalarType::F32 => SemanticArgument::F32Bits(reader.u32()?),
                    ScalarType::F64 => SemanticArgument::F64Bits(reader.u64()?),
                    ScalarType::Symbol => SemanticArgument::Symbol(reader.u32()?),
                    ScalarType::Bool => SemanticArgument::Bool(match reader.u8()? {
                        0 => false,
                        1 => true,
                        _ => return Err(admission_error("noncanonical material boolean")),
                    }),
                });
            }
            let count = reader.count(4)?;
            material_budget(&mut reference_budget, count)?;
            let mut qualifiers = Vec::with_capacity(count);
            for _ in 0..count {
                qualifiers.push(reader.u32()?);
            }
            records.push(SemanticTypedRecord {
                predicate,
                arguments,
                qualifiers,
            });
        }
        let count = reader.count(21)?;
        material_budget(&mut record_budget, count)?;
        material_budget(
            &mut reference_budget,
            count.checked_mul(5).ok_or_else(size_overflow)?,
        )?;
        let mut supports = Vec::with_capacity(count);
        for _ in 0..count {
            let statement = reader.u32()?;
            let polarity = match reader.u8()? {
                1 => SemanticPolarity::Pro,
                2 => SemanticPolarity::Contra,
                _ => return Err(admission_error("invalid material polarity")),
            };
            supports.push(SemanticSupportRecord {
                statement,
                polarity,
                provenance: reader.u32()?,
                source: reader.u32()?,
                context: reader.u32()?,
                scope: reader.u32()?,
            });
        }
        let count = reader.count(4)?;
        if count > limits.max_terms as usize {
            return Err(admission_error(
                "material symbol count exceeds admission bound",
            ));
        }
        let mut symbols = Vec::with_capacity(count);
        for _ in 0..count {
            symbols.push(material_string(&mut reader, &mut text_budget)?);
        }
        let count = reader.count(77)?;
        let mut insertions = Vec::with_capacity(count);
        for _ in 0..count {
            let statement = match reader.u8()? {
                0 => None,
                1 => Some(reader.u32()?),
                _ => return Err(admission_error("invalid material target origin")),
            };
            let mut reconstruction = [0; 10];
            for reference in &mut reconstruction {
                *reference = reader.u32()?;
            }
            if statement.is_some() && reconstruction != [0; 10] {
                return Err(admission_error(
                    "original material target has derived reconstruction references",
                ));
            }
            insertions.push(SemanticRootInsertion {
                statement,
                reconstruction,
                support: reader.u32()?,
                version: reader.take(32)?.try_into().unwrap(),
            });
        }
        let digest = reader.take(32)?.try_into().unwrap();
        let extents = [reader.u32()?, reader.u32()?, reader.u32()?];
        let admission_base_digest = reader.take(32)?.try_into().unwrap();
        let admission_base_extents = [reader.u32()?, reader.u32()?, reader.u32()?];
        reader.finish()?;
        let material = Self {
            records: SemanticAdmissionRecords {
                predicates,
                records,
                supports,
            },
            symbols,
            insertions,
            digest,
            extents,
            admission_base_digest,
            admission_base_extents,
        };
        material.validate_symbol_indices()?;
        Ok(material)
    }
}

fn normalized_material_admission(
    admission: &SemanticAdmission,
) -> Result<(SemanticAdmissionRecords, Vec<String>), SemanticHypergraphError> {
    let mut records = admission.records.clone();
    let mut symbols = Vec::with_capacity(admission.symbols.entries().len());
    for record in &mut records.records {
        for argument in &mut record.arguments {
            if let SemanticArgument::Symbol(index) = argument {
                let (accepted, text) = admission
                    .symbols
                    .entries()
                    .get(symbols.len())
                    .ok_or_else(|| admission_error("retained admission symbol is missing"))?;
                if *index != *accepted {
                    return Err(admission_error("retained admission symbol order differs"));
                }
                *index = u32::try_from(symbols.len()).map_err(|_| size_overflow())?;
                symbols.push(text.to_string());
            }
        }
    }
    if symbols.len() != admission.symbols.entries().len() {
        return Err(admission_error("retained admission has excess symbols"));
    }
    Ok((records, symbols))
}

fn material_root_digest(previous: [u8; 32], version: [u8; 32], extents: [u32; 3]) -> [u8; 32] {
    let mut bytes = [0u8; 108];
    bytes[..22].copy_from_slice(b"xlog.semantic.root.v1\0");
    bytes[32..64].copy_from_slice(&previous);
    bytes[64..96].copy_from_slice(&version);
    for (chunk, extent) in bytes[96..].as_chunks_mut::<4>().0.iter_mut().zip(extents) {
        chunk.copy_from_slice(&extent.to_le_bytes());
    }
    Sha256::digest(bytes).into()
}

fn material_version_digest(previous: [u8; 32], support: [u8; 32], truth: u64) -> [u8; 32] {
    let mut bytes = [0u8; 104];
    bytes[..25].copy_from_slice(b"xlog.semantic.version.v1\0");
    bytes[32..64].copy_from_slice(&previous);
    bytes[64..96].copy_from_slice(&support);
    bytes[96..].copy_from_slice(&truth.to_le_bytes());
    Sha256::digest(bytes).into()
}

fn record_encoding_prefix(schema: Identity256, predicate: RelId, arity: usize) -> Vec<u8> {
    let mut bytes = b"xlog.semantic.record.v2\0".to_vec();
    bytes.extend_from_slice(schema.as_bytes());
    bytes.extend_from_slice(&predicate.0.to_le_bytes());
    bytes.extend_from_slice(&(arity as u32).to_le_bytes());
    bytes
}

fn qualified_statement_identity(
    atom: [u8; 32],
    qualifiers: impl ExactSizeIterator<Item = [u8; 32]>,
) -> SemanticStatementIdentity {
    let mut hash = Sha256::new();
    hash.update(b"xlog.semantic.statement.v2\0");
    hash.update(atom);
    hash.update((qualifiers.len() as u32).to_le_bytes());
    for qualifier in qualifiers {
        hash.update(qualifier);
    }
    SemanticStatementIdentity(hash.finalize().into())
}

/// Reconstructs a decoder-selected target from the same retained typed input.
/// Source references select values only; they never create an original target.
fn material_statement_key(
    admission: &SemanticAdmission,
    original: Option<u32>,
    reconstruction: &[u32; 10],
) -> Result<SemanticStatementKey, SemanticHypergraphError> {
    if let Some(record) = original {
        if *reconstruction != [0; 10] {
            return Err(admission_error(
                "original material target has derived reconstruction references",
            ));
        }
        return admission.statement_key(record);
    }
    let predicates = &admission.records.predicates;
    let target = predicates
        .iter()
        .find(|entry| entry.predicate.0 == reconstruction[0])
        .ok_or_else(|| {
            admission_error("derived target predicate is outside the retained admission")
        })?;
    let arity = target.schema.arity();
    if target.role != SemanticRecordRole::Statement || arity > 4 {
        return Err(admission_error(
            "derived target requires an admitted statement schema of at most four arguments",
        ));
    }
    let mut bytes = record_encoding_prefix(admission.schema_generation, target.predicate, arity);
    for argument in 0..4 {
        let record_index = reconstruction[1 + 2 * argument] as usize;
        let argument_index = reconstruction[2 + 2 * argument] as usize;
        if argument >= arity {
            if record_index != 0 || argument_index != 0 {
                return Err(admission_error(
                    "derived target has nonzero inactive argument references",
                ));
            }
            continue;
        }
        let record = admission.records.records.get(record_index).ok_or_else(|| {
            admission_error("derived argument record is outside the retained admission")
        })?;
        let source = predicates
            .iter()
            .find(|entry| entry.predicate == record.predicate)
            .ok_or_else(|| {
                admission_error("derived argument predicate is outside the retained admission")
            })?;
        let column = source
            .schema
            .columns
            .get(argument_index)
            .ok_or_else(|| admission_error("derived argument is outside its source schema"))?;
        if column.1 != target.schema.columns[argument].1
            || source.schema.sort_labels().get(argument_index)
                != target.schema.sort_labels().get(argument)
        {
            return Err(admission_error(
                "derived argument type or sort differs from its target schema",
            ));
        }
        let encoding = &admission.encoded_records[record_index];
        let span = encoding.arguments.get(argument_index).ok_or_else(|| {
            admission_error("derived argument lacks its retained canonical bytes")
        })?;
        bytes.extend_from_slice(&encoding.bytes[span.clone()]);
    }
    let qualifiers = if reconstruction[9] == u32::MAX {
        &[][..]
    } else {
        admission.statement_key(reconstruction[9])?;
        &admission.records.records[reconstruction[9] as usize].qualifiers
    };
    let identity = qualified_statement_identity(
        Sha256::digest(bytes).into(),
        qualifiers
            .iter()
            .map(|&index| Sha256::digest(&admission.encoded_records[index as usize].bytes).into()),
    );
    Ok(SemanticStatementKey {
        identity,
        owner: admission.base.owner,
        record: u32::MAX,
    })
}

/// Packs an already-validated insertion identically for original and restored
/// targets. Runtime ownership and the typed reconstruction are checked by callers.
fn encode_support_insertion(
    command: &mut DeviceCommand,
    fork: SemanticForkHandle,
    statement: &SemanticStatementKey,
    event: &SemanticSupportEvent,
    reconstruction: &[u32; 10],
) {
    command.words[0] = OP_INSERT_SUPPORT;
    command.words[1] = fork.owner;
    command.words[8] = u64::from(fork.slot);
    command.words[9] = fork.generation;
    command.words[12] = event.polarity.code();
    command.words[13] = u64::from(statement.record);
    command.words[14] = u64::from(event.record);
    command.words[16..20].copy_from_slice(&identity_words(statement.identity.0));
    command.words[20..24].copy_from_slice(&identity_words(event.identity(statement.identity).0));
    for word in 0..5 {
        command.words[24 + word] =
            u64::from(reconstruction[2 * word]) | (u64::from(reconstruction[2 * word + 1]) << 32);
    }
}

impl SemanticRootMaterial {
    /// Project only contributors to the selected heads from this complete,
    /// owner-validated insertion history. Value equality selects a head, never
    /// replaces an insertion's original target or support occurrence.
    pub(crate) fn task_observation_roots(
        &self,
        admission: &SemanticAdmission,
        query_records: [u32; 3],
    ) -> Result<crate::semantic_transition::SemanticTaskObservationRoots, SemanticHypergraphError>
    {
        self.validate_lineage(admission)?;
        let queries = query_records.map(|record| admission.statement_key(record));
        let mut contributors = Vec::new();
        for (ordinal, query) in queries.into_iter().enumerate() {
            let query = query?;
            for insertion in &self.insertions {
                let key = material_statement_key(
                    admission,
                    insertion.statement,
                    &insertion.reconstruction,
                )?;
                if key.identity == query.identity {
                    contributors.push((ordinal as u32, insertion.statement, insertion.support));
                }
            }
        }
        Ok(crate::semantic_transition::SemanticTaskObservationRoots {
            root_digest: crate::semantic_transition::Identity256::from_bytes(self.digest),
            root_extents: self.extents,
            query_records,
            contributors,
        })
    }

    fn validate_lineage(
        &self,
        admission: &SemanticAdmission,
    ) -> Result<(), SemanticHypergraphError> {
        let (records, symbols) = normalized_material_admission(admission)?;
        if self.records != records || self.symbols != symbols {
            return Err(admission_error(
                "root material differs from the owner's complete typed admission",
            ));
        }
        let mut heads = std::collections::BTreeMap::<[u8; 32], ([u8; 32], u64)>::new();
        let mut supports = std::collections::BTreeSet::new();
        let mut digest = material_root_digest([0; 32], [0; 32], [0; 3]);
        let mut extents = [0u32; 3];
        let mut found_base =
            self.admission_base_extents == extents && self.admission_base_digest == digest;
        for insertion in &self.insertions {
            let key =
                material_statement_key(admission, insertion.statement, &insertion.reconstruction)?;
            let event = admission.support_event(insertion.support)?;
            let support = event.identity(key.identity).0;
            if !supports.insert((key.identity.0, support)) {
                return Err(admission_error(
                    "material repeats an unchanged support insertion",
                ));
            }
            let previous = heads.get(&key.identity.0).copied().unwrap_or(([0; 32], 0));
            let truth = previous.1 | event.polarity.code();
            let version = material_version_digest(previous.0, support, truth);
            if version != insertion.version {
                return Err(admission_error(
                    "material version identity does not match typed insertion history",
                ));
            }
            heads.insert(key.identity.0, (version, truth));
            extents = [
                u32::try_from(heads.len()).map_err(|_| size_overflow())?,
                extents[1].checked_add(1).ok_or_else(size_overflow)?,
                extents[2].checked_add(1).ok_or_else(size_overflow)?,
            ];
            digest = material_root_digest(digest, version, extents);
            if extents == self.admission_base_extents && digest == self.admission_base_digest {
                found_base = true;
            }
        }
        if digest != self.digest || extents != self.extents || !found_base {
            return Err(admission_error(
                "material root or original admission base differs from insertion history",
            ));
        }
        Ok(())
    }
}

/// Extracts only a sealed root's generation-valid reachable records. The transient
/// arena copy is not itself material: candidate state and other roots never escape.
fn material_from_arena(
    arena: &[u64],
    capacities: SemanticHypergraphCapacities,
    root: SemanticRootHandle,
    snapshot: SemanticRootSnapshot,
    admission: &SemanticAdmission,
) -> Result<SemanticRootMaterial, SemanticHypergraphError> {
    let corrupt = || SemanticHypergraphError::CorruptLineage {
        detail: "root material contains invalid native reachability or generation".into(),
    };
    if arena.len() as u64 != checked_arena_words(capacities)? || root.slot >= capacities.roots {
        return Err(corrupt());
    }
    let statements = CONTROL_WORDS as usize
        + capacities.roots as usize * ROOT_WORDS as usize
        + CANDIDATE_WORDS as usize;
    let supports = statements + capacities.statements as usize * STATEMENT_WORDS as usize;
    let versions = supports + capacities.supports as usize * SUPPORT_WORDS as usize;
    let heads = versions
        + capacities.versions as usize * VERSION_WORDS as usize
        + root.slot as usize * capacities.statements as usize;
    let root_offset = CONTROL_WORDS as usize + root.slot as usize * ROOT_WORDS as usize;
    let native_root = &arena[root_offset..root_offset + ROOT_WORDS as usize];
    if arena[1] != root.owner
        || native_root[0] != 3
        || native_root[1] != root.generation
        || root.generation == 0
    {
        return Err(corrupt());
    }
    let identity = |words: &[u64]| -> [u8; 32] {
        let mut bytes = [0; 32];
        for (chunk, word) in bytes.as_chunks_mut::<8>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    };
    if identity(&native_root[6..10]) != snapshot.digest.0
        || native_root[3..6]
            != [
                u64::from(snapshot.extents.statements),
                u64::from(snapshot.extents.supports),
                u64::from(snapshot.extents.versions),
            ]
    {
        return Err(corrupt());
    }
    let mut seen_versions = std::collections::BTreeSet::new();
    let mut seen_supports = std::collections::BTreeSet::new();
    let mut ordered = Vec::new();
    let mut statement_count = 0u32;
    for statement_slot in 0..capacities.statements as usize {
        let mut encoded = arena[heads + statement_slot];
        if encoded == 0 {
            continue;
        }
        statement_count += 1;
        let offset = statements + statement_slot * STATEMENT_WORDS as usize;
        let statement = &arena[offset..offset + STATEMENT_WORDS as usize];
        if statement[0] != 3 || statement[1] == 0 {
            return Err(corrupt());
        }
        let statement_identity = identity(&statement[3..7]);
        let mut newer_ordinal = u64::MAX;
        while encoded != 0 {
            let version_slot = usize::try_from(encoded - 1).map_err(|_| corrupt())?;
            if version_slot >= capacities.versions as usize || !seen_versions.insert(version_slot) {
                return Err(corrupt());
            }
            let offset = versions + version_slot * VERSION_WORDS as usize;
            let version = &arena[offset..offset + VERSION_WORDS as usize];
            if version[0] != 3
                || version[1] == 0
                || version[3] != statement_slot as u64
                || version[4] != statement[1]
                || version[13] == 0
                || version[13] >= newer_ordinal
            {
                return Err(corrupt());
            }
            newer_ordinal = version[13];
            let support_slot = usize::try_from(version[5]).map_err(|_| corrupt())?;
            if support_slot >= capacities.supports as usize || !seen_supports.insert(support_slot) {
                return Err(corrupt());
            }
            let offset = supports + support_slot * SUPPORT_WORDS as usize;
            let support = &arena[offset..offset + SUPPORT_WORDS as usize];
            if support[0] != 3
                || support[1] == 0
                || support[1] != version[6]
                || support[3] != statement_slot as u64
                || support[4] != statement[1]
                || !matches!(support[5], 1 | 2)
            {
                return Err(corrupt());
            }
            let support_identity = identity(&support[6..10]);
            let statement_index = u32::try_from(support[10]).map_err(|_| corrupt())?;
            let statement_index = (statement_index != u32::MAX).then_some(statement_index);
            let support_index = u32::try_from(support[11]).map_err(|_| corrupt())?;
            let reconstruction =
                std::array::from_fn(|word| (support[12 + word / 2] >> (32 * (word % 2))) as u32);
            let key = material_statement_key(admission, statement_index, &reconstruction)?;
            let event = admission.support_event(support_index)?;
            if key.identity.0 != statement_identity
                || event.polarity.code() != support[5]
                || event.identity(key.identity).0 != support_identity
            {
                return Err(admission_error(
                    "reachable insertion differs from its original typed occurrences",
                ));
            }
            let (previous, previous_truth) = if version[7] == 0 {
                ([0; 32], 0)
            } else {
                let slot = usize::try_from(version[7] - 1).map_err(|_| corrupt())?;
                if slot >= capacities.versions as usize {
                    return Err(corrupt());
                }
                let offset = versions + slot * VERSION_WORDS as usize;
                (identity(&arena[offset + 9..offset + 13]), arena[offset + 8])
            };
            let digest = identity(&version[9..13]);
            if previous_truth > 3
                || version[8] != previous_truth | support[5]
                || digest != material_version_digest(previous, support_identity, version[8])
            {
                return Err(corrupt());
            }
            ordered.push((
                version[13],
                SemanticRootInsertion {
                    statement: statement_index,
                    reconstruction,
                    support: support_index,
                    version: digest,
                },
            ));
            encoded = version[7];
        }
    }
    ordered.sort_unstable_by_key(|(ordinal, _)| *ordinal);
    if ordered
        .iter()
        .enumerate()
        .any(|(index, (ordinal, _))| *ordinal != index as u64 + 1)
        || statement_count != snapshot.extents.statements
        || seen_supports.len() != snapshot.extents.supports as usize
        || seen_versions.len() != snapshot.extents.versions as usize
    {
        return Err(corrupt());
    }
    let (records, symbols) = normalized_material_admission(admission)?;
    let material = SemanticRootMaterial {
        records,
        symbols,
        insertions: ordered
            .into_iter()
            .map(|(_, insertion)| insertion)
            .collect(),
        digest: snapshot.digest.0,
        extents: [
            snapshot.extents.statements,
            snapshot.extents.supports,
            snapshot.extents.versions,
        ],
        admission_base_digest: admission.base_snapshot.digest.0,
        admission_base_extents: [
            admission.base_snapshot.extents.statements,
            admission.base_snapshot.extents.supports,
            admission.base_snapshot.extents.versions,
        ],
    };
    material.validate_lineage(admission)?;
    Ok(material)
}

/// Immutable typed content and identities held by one graph for one acquired base.
///
/// Registry IDs are diagnostic references only. Consumers use `symbols()` for
/// accepted meanings and must not resolve them again through the live registry.
/// Admission validates general semantic records; it does not qualify targets for
/// an action catalogue's operand limits or its completion-aware draw mask.
pub struct SemanticAdmission {
    records: SemanticAdmissionRecords,
    symbols: symbol::SymbolSnapshot,
    schema_generation: Identity256,
    identity: Identity256,
    base: SemanticRootHandle,
    base_snapshot: SemanticRootSnapshot,
    statement_keys: Vec<Option<SemanticStatementKey>>,
    support_events: Vec<SemanticSupportEvent>,
    pub(crate) encoded_records: Vec<SemanticRecordEncoding>,
}

/// The exact admitted atom preimage and its typed argument ranges. Derived device
/// views borrow these bytes; symbols are never looked up a second time.
pub(crate) struct SemanticRecordEncoding {
    pub bytes: Vec<u8>,
    pub arguments: Vec<std::ops::Range<usize>>,
}

impl SemanticAdmission {
    /// Typed record and symbol-content encoding, distinct from the kernel launch ABI.
    pub const fn encoding_generation(&self) -> u32 {
        2
    }
    pub fn records(&self) -> &SemanticAdmissionRecords {
        &self.records
    }
    pub fn symbols(&self) -> &symbol::SymbolSnapshot {
        &self.symbols
    }
    pub const fn schema_generation(&self) -> Identity256 {
        self.schema_generation
    }
    pub const fn identity(&self) -> Identity256 {
        self.identity
    }
    pub const fn base(&self) -> SemanticRootHandle {
        self.base
    }
    pub const fn base_snapshot(&self) -> &SemanticRootSnapshot {
        &self.base_snapshot
    }

    /// Selects a validated statement record; metadata records cannot become statements.
    pub fn statement_key(
        &self,
        record: u32,
    ) -> Result<SemanticStatementKey, SemanticHypergraphError> {
        self.statement_keys
            .get(record as usize)
            .copied()
            .flatten()
            .ok_or_else(|| admission_error("record is not an admitted statement"))
    }

    /// Selects owner-bound support metadata whose source references were checked together.
    /// The original statement link remains in `records()`; insertion derives a
    /// support identity for the selected admitted target statement.
    pub fn support_event(
        &self,
        index: u32,
    ) -> Result<SemanticSupportEvent, SemanticHypergraphError> {
        self.support_events
            .get(index as usize)
            .copied()
            .ok_or_else(|| admission_error("support event is outside the admitted records"))
    }
}

fn admission_error(detail: impl Into<String>) -> SemanticHypergraphError {
    SemanticHypergraphError::InvalidInput {
        detail: detail.into(),
    }
}

fn consume_admission_bound(
    remaining: &mut usize,
    count: usize,
    name: &str,
) -> Result<(), SemanticHypergraphError> {
    *remaining = remaining
        .checked_sub(count)
        .ok_or_else(|| admission_error(format!("semantic admission exceeds {name} bound")))?;
    Ok(())
}

fn argument_type(argument: SemanticArgument) -> ScalarType {
    match argument {
        SemanticArgument::U32(_) => ScalarType::U32,
        SemanticArgument::U64(_) => ScalarType::U64,
        SemanticArgument::I32(_) => ScalarType::I32,
        SemanticArgument::I64(_) => ScalarType::I64,
        SemanticArgument::F32Bits(_) => ScalarType::F32,
        SemanticArgument::F64Bits(_) => ScalarType::F64,
        SemanticArgument::Bool(_) => ScalarType::Bool,
        SemanticArgument::Symbol(_) => ScalarType::Symbol,
    }
}

fn hash_sized_bytes(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

// One private admission path. It owns the input, validates the complete closure
// before symbol copying, and derives all identities from that retained content.
fn admit_semantic_records(
    records: SemanticAdmissionRecords,
    limits: SemanticAdmissionLimits,
    base: SemanticRootHandle,
    observe_base: impl FnOnce() -> Result<SemanticRootSnapshot, SemanticHypergraphError>,
) -> Result<SemanticAdmission, SemanticHypergraphError> {
    let mut record_budget = limits.max_records as usize;
    for count in [
        records.predicates.len(),
        records.records.len(),
        records.supports.len(),
    ] {
        consume_admission_bound(&mut record_budget, count, "record count")?;
    }
    let mut term_budget = limits.max_terms as usize;
    let mut reference_budget = limits.max_references as usize;
    let mut byte_budget = limits.max_utf8_bytes;
    for predicate in &records.predicates {
        let schema = &predicate.schema;
        consume_admission_bound(&mut term_budget, schema.columns.len(), "term count")?;
        consume_admission_bound(&mut term_budget, schema.key_columns.len(), "term count")?;
        if !schema.has_authoritative_sort_labels() {
            return Err(admission_error("schema has missing or invalid sort labels"));
        }
        for (name, _) in &schema.columns {
            consume_admission_bound(&mut byte_budget, name.len(), "UTF-8 bytes")?;
        }
        for label in schema.sort_labels() {
            consume_admission_bound(&mut byte_budget, label.len(), "UTF-8 bytes")?;
        }
        let mut keys = std::collections::BTreeSet::new();
        for &key in &schema.key_columns {
            if key >= schema.arity() || !keys.insert(key) {
                return Err(admission_error(
                    "schema key column is out of range or repeated",
                ));
            }
        }
    }
    for record in &records.records {
        consume_admission_bound(&mut term_budget, record.arguments.len(), "term count")?;
        consume_admission_bound(
            &mut reference_budget,
            record.qualifiers.len(),
            "reference count",
        )?;
    }
    for _ in &records.supports {
        consume_admission_bound(&mut reference_budget, 5, "reference count")?;
    }
    let mut predicates = std::collections::BTreeMap::new();
    for predicate in &records.predicates {
        if predicates
            .insert(predicate.predicate.0, predicate)
            .is_some()
        {
            return Err(admission_error("predicate schema is repeated"));
        }
    }
    let mut roles = Vec::with_capacity(records.records.len());
    let mut symbol_ids = Vec::new();
    for record in &records.records {
        let predicate = predicates
            .get(&record.predicate.0)
            .ok_or_else(|| admission_error("typed record references an unknown predicate"))?;
        if record.arguments.len() != predicate.schema.arity() {
            return Err(admission_error(
                "typed record arity differs from its predicate schema",
            ));
        }
        for (argument, (_, expected)) in record.arguments.iter().zip(&predicate.schema.columns) {
            if argument_type(*argument) != *expected {
                return Err(admission_error(
                    "typed argument differs from its schema column type",
                ));
            }
            if let SemanticArgument::Symbol(id) = argument {
                symbol_ids.push(*id);
            }
        }
        if predicate.role != SemanticRecordRole::Statement && !record.qualifiers.is_empty() {
            return Err(admission_error(
                "only statements may carry identity-bearing qualifiers",
            ));
        }
        roles.push(predicate.role);
    }
    let require_role = |index: u32, role| {
        if roles.get(index as usize) == Some(&role) {
            Ok(())
        } else {
            Err(admission_error(format!(
                "record reference {index} is not an admitted {role:?}"
            )))
        }
    };
    for record in &records.records {
        for &index in &record.qualifiers {
            require_role(index, SemanticRecordRole::Qualifier)?;
        }
    }
    for support in &records.supports {
        for (index, role) in [
            (support.statement, SemanticRecordRole::Statement),
            (support.provenance, SemanticRecordRole::Provenance),
            (support.source, SemanticRecordRole::Source),
            (support.context, SemanticRecordRole::Context),
            (support.scope, SemanticRecordRole::Scope),
        ] {
            require_role(index, role)?;
        }
    }
    let symbols = symbol::snapshot_checked(&symbol_ids, limits.max_terms as usize, byte_budget)
        .map_err(|error| admission_error(error.to_string()))?;
    let mut schema_hash = Sha256::new();
    schema_hash.update(b"xlog.semantic.schema.v2\0");
    schema_hash.update((predicates.len() as u32).to_le_bytes());
    for predicate in predicates.values() {
        schema_hash.update(predicate.predicate.0.to_le_bytes());
        schema_hash.update([predicate.role.code()]);
        schema_hash.update((predicate.schema.arity() as u32).to_le_bytes());
        for (name, ty) in &predicate.schema.columns {
            hash_sized_bytes(&mut schema_hash, name.as_bytes());
            schema_hash.update([ty.to_code()]);
        }
        schema_hash.update((predicate.schema.key_columns.len() as u32).to_le_bytes());
        for &key in &predicate.schema.key_columns {
            schema_hash.update((key as u32).to_le_bytes());
        }
        for label in predicate.schema.sort_labels() {
            hash_sized_bytes(&mut schema_hash, label.as_bytes());
        }
    }
    let schema_generation = Identity256::from_bytes(schema_hash.finalize().into());
    let mut symbol_text = symbols.entries().iter();
    let mut atoms = Vec::<[u8; 32]>::with_capacity(records.records.len());
    let mut encoded_records = Vec::with_capacity(records.records.len());
    for record in &records.records {
        let mut bytes =
            record_encoding_prefix(schema_generation, record.predicate, record.arguments.len());
        let mut arguments = Vec::with_capacity(record.arguments.len());
        for &argument in &record.arguments {
            let start = bytes.len();
            bytes.push(argument_type(argument).to_code() + 1);
            match argument {
                SemanticArgument::U32(value) | SemanticArgument::F32Bits(value) => {
                    bytes.extend_from_slice(&value.to_le_bytes())
                }
                SemanticArgument::U64(value) | SemanticArgument::F64Bits(value) => {
                    bytes.extend_from_slice(&value.to_le_bytes())
                }
                SemanticArgument::I32(value) => bytes.extend_from_slice(&value.to_le_bytes()),
                SemanticArgument::I64(value) => bytes.extend_from_slice(&value.to_le_bytes()),
                SemanticArgument::Bool(value) => bytes.push(u8::from(value)),
                SemanticArgument::Symbol(id) => {
                    let (accepted_id, text) = symbol_text
                        .next()
                        .expect("validated complete symbol snapshot");
                    debug_assert_eq!(*accepted_id, id);
                    // Encoding two binds UTF-8 content, not a process-local ID.
                    bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
                    bytes.extend_from_slice(text.as_bytes());
                }
            }
            arguments.push(start..bytes.len());
        }
        atoms.push(Sha256::digest(&bytes).into());
        encoded_records.push(SemanticRecordEncoding { bytes, arguments });
    }
    let statement_keys: Vec<_> = records
        .records
        .iter()
        .enumerate()
        .map(|(index, record)| {
            if roles[index] != SemanticRecordRole::Statement {
                return None;
            }
            Some(SemanticStatementKey {
                identity: qualified_statement_identity(
                    atoms[index],
                    record
                        .qualifiers
                        .iter()
                        .map(|&qualifier| atoms[qualifier as usize]),
                ),
                owner: base.owner,
                record: index as u32,
            })
        })
        .collect();
    let support_events: Vec<_> = records
        .supports
        .iter()
        .enumerate()
        .map(|(index, record)| SemanticSupportEvent {
            owner: base.owner,
            record: index as u32,
            polarity: record.polarity,
            provenance: Identity256::from_bytes(atoms[record.provenance as usize]),
            source: Identity256::from_bytes(atoms[record.source as usize]),
            context: Identity256::from_bytes(atoms[record.context as usize]),
            scope: Identity256::from_bytes(atoms[record.scope as usize]),
        })
        .collect();
    let base_snapshot = observe_base()?;
    let identity = derive_admission_identity(
        &records,
        schema_generation,
        &encoded_records,
        &statement_keys,
        &support_events,
        &base_snapshot,
    );
    Ok(SemanticAdmission {
        records,
        symbols,
        schema_generation,
        identity,
        base,
        base_snapshot,
        statement_keys,
        support_events,
        encoded_records,
    })
}

fn derive_admission_identity(
    records: &SemanticAdmissionRecords,
    schema_generation: Identity256,
    encoded_records: &[SemanticRecordEncoding],
    statement_keys: &[Option<SemanticStatementKey>],
    support_events: &[SemanticSupportEvent],
    base_snapshot: &SemanticRootSnapshot,
) -> Identity256 {
    let mut hash = Sha256::new();
    hash.update(b"xlog.semantic.admission.v2\0");
    hash.update(base_snapshot.canonical_bytes());
    hash.update(schema_generation.as_bytes());
    for predicate in &records.predicates {
        hash.update(predicate.predicate.0.to_le_bytes());
    }
    hash.update((encoded_records.len() as u32).to_le_bytes());
    for ((encoded, key), record) in encoded_records
        .iter()
        .zip(statement_keys)
        .zip(&records.records)
    {
        hash.update(Sha256::digest(&encoded.bytes));
        hash.update((record.qualifiers.len() as u32).to_le_bytes());
        for &reference in &record.qualifiers {
            hash.update(reference.to_le_bytes());
        }
        hash.update([u8::from(key.is_some())]);
        if let Some(key) = key {
            hash.update(key.identity.as_bytes());
        }
    }
    let symbol_ids: Vec<_> = records
        .records
        .iter()
        .flat_map(|record| &record.arguments)
        .filter_map(|argument| match argument {
            SemanticArgument::Symbol(id) => Some(*id),
            _ => None,
        })
        .collect();
    hash.update((symbol_ids.len() as u32).to_le_bytes());
    for id in symbol_ids {
        hash.update(id.to_le_bytes());
    }
    hash.update((support_events.len() as u32).to_le_bytes());
    for (event, record) in support_events.iter().zip(&records.supports) {
        for index in [
            record.statement,
            record.provenance,
            record.source,
            record.context,
            record.scope,
        ] {
            hash.update(index.to_le_bytes());
        }
        let source_statement = statement_keys[record.statement as usize]
            .expect("validated source statement reference")
            .identity;
        hash.update(event.identity(source_statement).as_bytes());
    }
    Identity256::from_bytes(hash.finalize().into())
}

/// Content identity of a semantic statement.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SemanticStatementIdentity([u8; 32]);

impl SemanticStatementIdentity {
    /// Returns the canonical digest bytes.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Content identity of one exact support event.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SemanticSupportIdentity([u8; 32]);

impl SemanticSupportIdentity {
    /// Returns the canonical digest bytes.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Content identity of an immutable statement version.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SemanticVersionIdentity([u8; 32]);

impl SemanticVersionIdentity {
    /// Returns the canonical digest bytes.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Content digest of an immutable semantic root.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SemanticRootDigest([u8; 32]);

impl SemanticRootDigest {
    /// Returns the canonical digest bytes.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Canonical, identity-bearing statement key.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SemanticStatementKey {
    identity: SemanticStatementIdentity,
    owner: u64,
    record: u32,
}

impl SemanticStatementKey {
    /// Returns the derived statement identity.
    pub const fn identity(&self) -> SemanticStatementIdentity {
        self.identity
    }
}

/// One provenance-bearing support event. Presence is exactly one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SemanticSupportEvent {
    owner: u64,
    record: u32,
    polarity: SemanticPolarity,
    provenance: Identity256,
    source: Identity256,
    context: Identity256,
    scope: Identity256,
}

impl SemanticSupportEvent {
    pub(crate) fn identity(self, statement: SemanticStatementIdentity) -> SemanticSupportIdentity {
        let mut hash = Sha256::new();
        hash.update(b"xlog.semantic.support.v1\0");
        hash.update(statement.as_bytes());
        hash.update((self.polarity.code() as u32).to_le_bytes());
        hash.update(1u32.to_le_bytes());
        hash.update(self.provenance.as_bytes());
        hash.update(self.source.as_bytes());
        hash.update(self.context.as_bytes());
        hash.update(self.scope.as_bytes());
        SemanticSupportIdentity(hash.finalize().into())
    }
}

/// Kind of opaque semantic handle used in typed diagnostics.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SemanticHandleKind {
    Root,
    Fork,
    Statement,
    Support,
    Version,
}

macro_rules! semantic_handle {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
        pub struct $name {
            owner: u64,
            slot: u32,
            generation: u64,
        }

        impl $name {
            const fn new(owner: u64, slot: u32, generation: u64) -> Self {
                Self {
                    owner,
                    slot,
                    generation,
                }
            }

            /// Returns the opaque physical slot for diagnostics.
            pub const fn slot(self) -> u32 {
                self.slot
            }

            /// Returns the exact physical generation.
            pub const fn generation(self) -> u64 {
                self.generation
            }
        }
    };
}

semantic_handle!(SemanticRootHandle);
semantic_handle!(SemanticForkHandle);
semantic_handle!(SemanticStatementHandle);
semantic_handle!(SemanticSupportHandle);
semantic_handle!(SemanticVersionHandle);

/// A sealed root or one live fork used for device observation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SemanticView {
    Root(SemanticRootHandle),
    Fork(SemanticForkHandle),
}

/// Exact reachable logical extents.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct SemanticExtents {
    statements: u32,
    supports: u32,
    versions: u32,
}

impl SemanticExtents {
    pub const fn new(statements: u32, supports: u32, versions: u32) -> Self {
        Self {
            statements,
            supports,
            versions,
        }
    }

    pub const fn statements(self) -> u32 {
        self.statements
    }

    pub const fn supports(self) -> u32 {
        self.supports
    }

    pub const fn versions(self) -> u32 {
        self.versions
    }
}

/// Immutable device-derived root receipt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SemanticRootSnapshot {
    digest: SemanticRootDigest,
    extents: SemanticExtents,
    canonical_bytes: [u8; 64],
}

impl SemanticRootSnapshot {
    fn new(digest: SemanticRootDigest, extents: SemanticExtents) -> Self {
        let mut canonical_bytes = [0u8; 64];
        canonical_bytes[..8].copy_from_slice(b"XLOGSHG1");
        canonical_bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        canonical_bytes[12..16].copy_from_slice(&extents.statements.to_le_bytes());
        canonical_bytes[16..20].copy_from_slice(&extents.supports.to_le_bytes());
        canonical_bytes[20..24].copy_from_slice(&extents.versions.to_le_bytes());
        canonical_bytes[24..56].copy_from_slice(digest.as_bytes());
        Self {
            digest,
            extents,
            canonical_bytes,
        }
    }

    pub const fn digest(&self) -> &SemanticRootDigest {
        &self.digest
    }

    pub const fn extents(&self) -> SemanticExtents {
        self.extents
    }

    pub const fn canonical_bytes(&self) -> &[u8; 64] {
        &self.canonical_bytes
    }
}

/// Checked fixed capacities for one resident semantic graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SemanticHypergraphCapacities {
    roots: u32,
    statements: u32,
    supports: u32,
    versions: u32,
}

impl SemanticHypergraphCapacities {
    pub fn try_new(
        roots: u32,
        statements: u32,
        supports: u32,
        versions: u32,
    ) -> Result<Self, SemanticHypergraphError> {
        for (name, value) in [
            ("root", roots),
            ("statement", statements),
            ("support", supports),
            ("version", versions),
        ] {
            if value == 0 {
                return Err(SemanticHypergraphError::InvalidCapacity { kind: name, value });
            }
        }
        Ok(Self {
            roots,
            statements,
            supports,
            versions,
        })
    }
}

/// Statement identity paired with its opaque physical handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SemanticStatementRef {
    handle: SemanticStatementHandle,
    identity: SemanticStatementIdentity,
}

impl SemanticStatementRef {
    pub const fn handle(self) -> SemanticStatementHandle {
        self.handle
    }

    pub const fn identity(self) -> SemanticStatementIdentity {
        self.identity
    }
}

/// Support identity paired with its opaque physical handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SemanticSupportRef {
    handle: SemanticSupportHandle,
    identity: SemanticSupportIdentity,
}

impl SemanticSupportRef {
    pub const fn handle(self) -> SemanticSupportHandle {
        self.handle
    }

    pub const fn identity(self) -> SemanticSupportIdentity {
        self.identity
    }
}

/// Immutable statement-version identity and device-derived truth.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SemanticVersionRef {
    handle: SemanticVersionHandle,
    identity: SemanticVersionIdentity,
    truth: SemanticTruth,
}

impl SemanticVersionRef {
    pub const fn handle(self) -> SemanticVersionHandle {
        self.handle
    }

    pub const fn identity(self) -> SemanticVersionIdentity {
        self.identity
    }

    pub const fn truth(self) -> SemanticTruth {
        self.truth
    }
}

/// Complete result of inserting a new support event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SemanticInsertedSupport {
    statement: SemanticStatementRef,
    support: SemanticSupportRef,
    version: SemanticVersionRef,
    previous_truth: SemanticTruth,
}

impl SemanticInsertedSupport {
    pub const fn statement(self) -> SemanticStatementRef {
        self.statement
    }

    pub const fn support(self) -> SemanticSupportRef {
        self.support
    }

    pub const fn version(self) -> SemanticVersionRef {
        self.version
    }

    /// Truth immediately before this support was attached, retained by the insertion.
    pub const fn previous_truth(self) -> SemanticTruth {
        self.previous_truth
    }

    /// Whether the attachment changed a truth that already had reachable support.
    /// A first support and a new event with unchanged truth both return false.
    pub fn changes_defined_truth(self) -> bool {
        self.previous_truth != SemanticTruth::Neither && self.previous_truth != self.version.truth
    }
}

/// Result of exact support insertion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SemanticInsertOutcome {
    Inserted(SemanticInsertedSupport),
    Unchanged(SemanticVersionRef),
}

/// Production-path execution counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SemanticHypergraphExecutionStats {
    cuda_kernel_launches: u64,
}

impl SemanticHypergraphExecutionStats {
    pub const fn cuda_kernel_launches(self) -> u64 {
        self.cuda_kernel_launches
    }
}

/// Typed failures from semantic owner and device lifecycle validation.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticHypergraphError {
    InvalidCapacity {
        kind: &'static str,
        value: u32,
    },
    InvalidInput {
        detail: String,
    },
    ForeignHandle {
        kind: SemanticHandleKind,
    },
    SlotOutOfRange {
        kind: SemanticHandleKind,
        slot: u32,
        capacity: u32,
    },
    StaleGeneration {
        kind: SemanticHandleKind,
        slot: u32,
        presented: u64,
        current: u64,
    },
    InactiveHandle {
        kind: SemanticHandleKind,
        slot: u32,
    },
    NotReachable {
        kind: SemanticHandleKind,
        slot: u32,
    },
    CapacityExceeded {
        kind: SemanticHandleKind,
        capacity: u32,
    },
    GenerationExhausted {
        kind: SemanticHandleKind,
        slot: u32,
    },
    CorruptLineage {
        detail: String,
    },
    KernelUnavailable,
    Runtime {
        operation: &'static str,
        detail: String,
    },
    DeviceControlled,
    Poisoned,
}

impl fmt::Display for SemanticHypergraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCapacity { kind, value } => {
                write!(
                    formatter,
                    "semantic {kind} capacity must be positive, got {value}"
                )
            }
            Self::InvalidInput { detail } => write!(formatter, "invalid semantic input: {detail}"),
            Self::ForeignHandle { kind } => write!(formatter, "foreign {kind:?} handle"),
            Self::SlotOutOfRange {
                kind,
                slot,
                capacity,
            } => write!(
                formatter,
                "semantic {kind:?} slot {slot} exceeds capacity {capacity}"
            ),
            Self::StaleGeneration {
                kind,
                slot,
                presented,
                current,
            } => write!(
                formatter,
                "stale semantic {kind:?} generation at slot {slot}: {presented}, current {current}"
            ),
            Self::InactiveHandle { kind, slot } => {
                write!(formatter, "inactive semantic {kind:?} slot {slot}")
            }
            Self::NotReachable { kind, slot } => {
                write!(formatter, "semantic {kind:?} slot {slot} is not reachable")
            }
            Self::CapacityExceeded { kind, capacity } => {
                write!(
                    formatter,
                    "semantic {kind:?} capacity {capacity} is exhausted"
                )
            }
            Self::GenerationExhausted { kind, slot } => {
                write!(
                    formatter,
                    "semantic {kind:?} generation exhausted at slot {slot}"
                )
            }
            Self::CorruptLineage { detail } => {
                write!(formatter, "corrupt semantic device lineage: {detail}")
            }
            Self::KernelUnavailable => {
                write!(formatter, "semantic hypergraph CUDA kernel unavailable")
            }
            Self::Runtime { operation, detail } => {
                write!(formatter, "semantic {operation} failed: {detail}")
            }
            Self::DeviceControlled => write!(
                formatter,
                "semantic hypergraph is device-controlled after resident admission"
            ),
            Self::Poisoned => write!(
                formatter,
                "semantic hypergraph is poisoned after runtime or integrity failure"
            ),
        }
    }
}

impl std::error::Error for SemanticHypergraphError {}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct DeviceCommand {
    words: [u64; COMMAND_WORDS],
}

impl Default for DeviceCommand {
    fn default() -> Self {
        Self {
            words: [0; COMMAND_WORDS],
        }
    }
}

// SAFETY: this fixed-size C-layout value contains only `u64` words and no references.
unsafe impl DeviceRepr for DeviceCommand {}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct SemanticResidentReceiptRecord {
    // Successful insertion word 32 holds new-statement bit 0 and the actual
    // previous truth in bits 1..2. Outcome word 1 distinguishes new attachment
    // from duplicate; the returned duplicate version can predate the current head.
    // The final two words preserve the successfully acquired candidate slot and
    // generation through refusal diagnostics. A failed fork never grants them.
    words: [u64; RECEIPT_WORDS],
}

impl Default for SemanticResidentReceiptRecord {
    fn default() -> Self {
        Self {
            words: [0; RECEIPT_WORDS],
        }
    }
}

// SAFETY: this fixed-size C-layout value contains only `u64` words and no references.
unsafe impl DeviceRepr for SemanticResidentReceiptRecord {}

type DeviceReceipt = SemanticResidentReceiptRecord;

/// Canonical statement identity decoded into a resident semantic input bank.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct SemanticResidentDecodedStatement {
    identity_words: [u32; 8],
    record: u32,
    reconstruction: [u32; 10],
}

// SAFETY: this fixed-size C-layout value contains only `u32` words and no references.
unsafe impl DeviceRepr for SemanticResidentDecodedStatement {}

/// Canonical support event components decoded into a resident semantic input bank.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct SemanticResidentDecodedSupport {
    polarity: u32,
    provenance_words: [u32; 8],
    source_words: [u32; 8],
    context_words: [u32; 8],
    scope_words: [u32; 8],
    record: u32,
}

// SAFETY: this fixed-size C-layout value contains only `u32` words and no references.
unsafe impl DeviceRepr for SemanticResidentDecodedSupport {}

/// One decoder-owned semantic record before it is split into owner input banks.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct SemanticResidentDecodedInput {
    statement_identity_words: [u32; 8],
    polarity: u32,
    provenance_words: [u32; 8],
    source_words: [u32; 8],
    context_words: [u32; 8],
    scope_words: [u32; 8],
    statement_record: u32,
    support_record: u32,
    reconstruction: [u32; 10],
}

// SAFETY: this fixed-size C-layout value contains only `u32` words and no references.
unsafe impl DeviceRepr for SemanticResidentDecodedInput {}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct SemanticResidentTruthValue {
    status: u64,
    truth: u64,
    owner: u64,
    reserved: u64,
}

// SAFETY: this type is a fixed-width C-layout POD shared with the CUDA kernel.
unsafe impl DeviceRepr for SemanticResidentTruthValue {}

/// Generation-bound semantic handle stored in a resident device bank.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct SemanticResidentHandleRecord {
    owner: u64,
    kind: u64,
    slot: u64,
    generation: u64,
}

// SAFETY: this fixed-size C-layout value contains only `u64` words and no references.
unsafe impl DeviceRepr for SemanticResidentHandleRecord {}

const _: () = assert!(std::mem::size_of::<DeviceCommand>() == COMMAND_WORDS * 8);
const _: () = assert!(std::mem::align_of::<DeviceCommand>() == 8);
const _: () = assert!(std::mem::size_of::<DeviceReceipt>() == RECEIPT_WORDS * 8);
const _: () = assert!(std::mem::align_of::<DeviceReceipt>() == 8);
const _: () = assert!(std::mem::size_of::<SemanticResidentDecodedStatement>() == 76);
const _: () = assert!(std::mem::align_of::<SemanticResidentDecodedStatement>() == 4);
const _: () = assert!(std::mem::size_of::<SemanticResidentDecodedSupport>() == 136);
const _: () = assert!(std::mem::align_of::<SemanticResidentDecodedSupport>() == 4);
const _: () = assert!(std::mem::size_of::<SemanticResidentDecodedInput>() == 212);
const _: () = assert!(std::mem::align_of::<SemanticResidentDecodedInput>() == 4);
const _: () = assert!(std::mem::size_of::<SemanticResidentTruthValue>() == 32);
const _: () = assert!(std::mem::align_of::<SemanticResidentTruthValue>() == 8);
const _: () = assert!(std::mem::size_of::<SemanticResidentHandleRecord>() == 32);
const _: () = assert!(std::mem::align_of::<SemanticResidentHandleRecord>() == 8);

/// One writable slot in a resident generation-bound handle bank.
pub(crate) struct SemanticResidentHandleSlot<'a> {
    allocation: &'a TrackedCudaSlice<SemanticResidentHandleRecord>,
    index: u32,
}

impl<'a> SemanticResidentHandleSlot<'a> {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "constructed by crate-internal device-resident pipelines"
        )
    )]
    pub(crate) fn new(
        allocation: &'a TrackedCudaSlice<SemanticResidentHandleRecord>,
        index: u32,
    ) -> Result<Self, SemanticHypergraphError> {
        validate_resident_bank_index(index, allocation.len(), "handle output")?;
        Ok(Self { allocation, index })
    }
}

/// One immutable-root handle selected from an already-resident handle bank.
pub(crate) struct SemanticResidentRootHandleBank<'a> {
    allocation: &'a TrackedCudaSlice<SemanticResidentHandleRecord>,
    index: u32,
}

impl SemanticResidentRootHandleBank<'_> {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "consumed by crate-internal device-resident pipelines"
        )
    )]
    pub(crate) fn handle(&self) -> SemanticResidentRootHandle<'_> {
        SemanticResidentRootHandle::Bank {
            allocation: self.allocation,
            index: self.index,
        }
    }
}

/// One canonical statement identity selected from an already-resident decoded bank.
pub(crate) struct SemanticResidentStatementBank<'a> {
    allocation: &'a TrackedCudaSlice<SemanticResidentDecodedStatement>,
    index: u32,
}

/// One record selected from a decoder-owned resident semantic input bank.
pub(crate) struct SemanticResidentDecodedInputBank<'a> {
    allocation: &'a TrackedCudaSlice<SemanticResidentDecodedInput>,
    index: u32,
}

impl<'a> SemanticResidentDecodedInputBank<'a> {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "constructed by crate-internal device-resident pipelines"
        )
    )]
    pub(crate) fn new(
        allocation: &'a TrackedCudaSlice<SemanticResidentDecodedInput>,
        index: u32,
    ) -> Result<Self, SemanticHypergraphError> {
        validate_resident_bank_index(index, allocation.len(), "decoded input")?;
        Ok(Self { allocation, index })
    }
}

impl<'a> SemanticResidentStatementBank<'a> {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "constructed by crate-internal device-resident pipelines"
        )
    )]
    pub(crate) fn new(
        allocation: &'a TrackedCudaSlice<SemanticResidentDecodedStatement>,
        index: u32,
    ) -> Result<Self, SemanticHypergraphError> {
        validate_resident_bank_index(index, allocation.len(), "decoded statement")?;
        Ok(Self { allocation, index })
    }
}

/// One canonical support event selected from an already-resident decoded bank.
pub(crate) struct SemanticResidentSupportBank<'a> {
    allocation: &'a TrackedCudaSlice<SemanticResidentDecodedSupport>,
    index: u32,
}

impl<'a> SemanticResidentSupportBank<'a> {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "constructed by crate-internal device-resident pipelines"
        )
    )]
    pub(crate) fn new(
        allocation: &'a TrackedCudaSlice<SemanticResidentDecodedSupport>,
        index: u32,
    ) -> Result<Self, SemanticHypergraphError> {
        validate_resident_bank_index(index, allocation.len(), "decoded support")?;
        Ok(Self { allocation, index })
    }
}

/// One writable receipt slot selected from a private device bank.
pub(crate) struct SemanticResidentReceiptSlot<'a> {
    allocation: &'a TrackedCudaSlice<SemanticResidentReceiptRecord>,
    index: u32,
}

impl<'a> SemanticResidentReceiptSlot<'a> {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "constructed by crate-internal device-resident pipelines"
        )
    )]
    pub(crate) fn new(
        allocation: &'a TrackedCudaSlice<SemanticResidentReceiptRecord>,
        index: u32,
    ) -> Result<Self, SemanticHypergraphError> {
        validate_resident_bank_index(index, allocation.len(), "receipt")?;
        Ok(Self { allocation, index })
    }
}

/// Device pointer and dependency view for one typed resident semantic receipt.
pub(crate) struct SemanticResidentReceiptView<'a> {
    allocation: &'a TrackedCudaSlice<SemanticResidentReceiptRecord>,
    index: u32,
}

/// A typed, device-resident four-valued truth receipt.
pub(crate) struct SemanticResidentTruthView<'a> {
    receipt: SemanticResidentReceiptView<'a>,
}

/// One writable output selected from a resident truth-value bank.
pub(crate) struct SemanticResidentTruthSlot<'a> {
    allocation: &'a TrackedCudaSlice<SemanticResidentTruthValue>,
    index: u32,
}

impl<'a> SemanticResidentTruthSlot<'a> {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "constructed by crate-internal device-resident pipelines"
        )
    )]
    pub(crate) fn new(
        allocation: &'a TrackedCudaSlice<SemanticResidentTruthValue>,
        index: u32,
    ) -> Result<Self, SemanticHypergraphError> {
        validate_resident_bank_index(index, allocation.len(), "truth output")?;
        Ok(Self { allocation, index })
    }
}

impl SemanticResidentReceiptView<'_> {
    pub(crate) fn record_read(&self, recorder: &mut crate::launch::LaunchRecorder) {
        recorder.read(self.allocation);
    }
}

/// A generation-bound immutable-root handle carried by a typed bank or prior receipt.
pub(crate) enum SemanticResidentRootHandle<'a> {
    Bank {
        allocation: &'a TrackedCudaSlice<SemanticResidentHandleRecord>,
        index: u32,
    },
    Receipt(SemanticResidentReceiptView<'a>),
}

impl SemanticResidentRootHandle<'_> {
    fn record_input(
        self,
        descriptor: &mut DeviceLaunchDescriptor,
        recorder: &mut crate::launch::LaunchRecorder,
    ) {
        match self {
            Self::Bank { allocation, index } => {
                recorder.read(allocation);
                descriptor.handle_ptr = allocation.device_ptr_value();
                descriptor.handle_index = u64::from(index);
            }
            Self::Receipt(receipt) => {
                receipt.record_read(recorder);
                descriptor.source_receipt_ptr = receipt.allocation.device_ptr_value();
                descriptor.source_receipt_index = u64::from(receipt.index);
            }
        }
    }
}

/// A generation-bound candidate handle carried by a pending device receipt.
pub(crate) struct SemanticResidentCandidateHandle<'a> {
    receipt: SemanticResidentReceiptView<'a>,
}

/// Explicit query source; roots never require acquiring a mutable candidate.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "root and fork construction is exercised by the CUDA resident-query qualification"
    )
)]
pub(crate) enum SemanticResidentView<'a> {
    Root(SemanticResidentRootHandle<'a>),
    Fork(SemanticResidentCandidateHandle<'a>),
}

/// A pending fork or insertion result that keeps its candidate handle resident.
pub(crate) struct SemanticResidentMutationReceipt<'a> {
    receipt: SemanticResidentReceiptView<'a>,
}

impl SemanticResidentMutationReceipt<'_> {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "consumed by crate-internal device-resident pipelines"
        )
    )]
    pub(crate) fn candidate_handle(&self) -> SemanticResidentCandidateHandle<'_> {
        SemanticResidentCandidateHandle {
            receipt: SemanticResidentReceiptView {
                allocation: self.receipt.allocation,
                index: self.receipt.index,
            },
        }
    }
}

/// A sealed immutable-root result that remains device resident.
pub(crate) struct SemanticResidentSealReceipt<'a> {
    receipt: SemanticResidentReceiptView<'a>,
}

impl SemanticResidentSealReceipt<'_> {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "consumed by crate-internal device-resident pipelines"
        )
    )]
    pub(crate) fn root_handle(&self) -> SemanticResidentRootHandle<'_> {
        SemanticResidentRootHandle::Receipt(SemanticResidentReceiptView {
            allocation: self.receipt.allocation,
            index: self.receipt.index,
        })
    }
}

/// Four-valued truth result produced and retained on the resident device path.
pub(crate) struct SemanticResidentTruthReceipt<'a> {
    receipt: SemanticResidentReceiptView<'a>,
}

impl SemanticResidentTruthReceipt<'_> {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "consumed by crate-internal device-resident pipelines"
        )
    )]
    pub(crate) fn device_view(&self) -> SemanticResidentTruthView<'_> {
        SemanticResidentTruthView {
            receipt: SemanticResidentReceiptView {
                allocation: self.receipt.allocation,
                index: self.receipt.index,
            },
        }
    }
}

#[derive(Clone, Copy)]
struct SlotLedger {
    generation: u64,
    live: bool,
}

impl SlotLedger {
    const FREE: Self = Self {
        generation: 1,
        live: false,
    };
}

struct CurrentFork {
    handle: SemanticForkHandle,
    staged_statements: Vec<u32>,
    staged_supports: Vec<u32>,
    staged_versions: Vec<u32>,
}

#[derive(Clone, Copy)]
enum ArenaAccess {
    Read,
    ReadWrite,
}

struct SemanticKernelLaunchSpec<'a> {
    domain: &'a ResidentExecutionDomain,
    execute: &'a CudaFunction,
    owner: u64,
    capacities: SemanticHypergraphCapacities,
    arena_words: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DeviceLaunchDescriptor {
    expected_owner: u64,
    root_capacity: u64,
    statement_capacity: u64,
    support_capacity: u64,
    version_capacity: u64,
    arena_words: u64,
    command_ptr: u64,
    command_index: u64,
    handle_ptr: u64,
    handle_index: u64,
    source_receipt_ptr: u64,
    source_receipt_index: u64,
    decoded_input_ptr: u64,
    decoded_input_index: u64,
    decoded_statement_ptr: u64,
    decoded_statement_index: u64,
    decoded_support_ptr: u64,
    decoded_support_index: u64,
    output_ptr: u64,
    output_index: u64,
    receipt_ptr: u64,
    receipt_index: u64,
    admission: u64,
    abi_generation: u64,
}

const _: () = assert!(std::mem::size_of::<DeviceLaunchDescriptor>() == 192);
const _: () = assert!(std::mem::align_of::<DeviceLaunchDescriptor>() == 8);

// SAFETY: this fixed-size C-layout value contains only `u64` words and no references.
unsafe impl DeviceRepr for DeviceLaunchDescriptor {}

struct DeviceLaunchDescriptorParam(DeviceLaunchDescriptor);

impl crate::cuda_compat::KernelParamStorage for DeviceLaunchDescriptorParam {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (&self.0 as *const DeviceLaunchDescriptor).cast_mut().cast()
    }
}

impl crate::cuda_compat::IntoKernelParamStorage for DeviceLaunchDescriptor {
    type Storage = DeviceLaunchDescriptorParam;

    fn into_kernel_param_storage(self) -> Self::Storage {
        DeviceLaunchDescriptorParam(self)
    }
}

enum SemanticKernelInput<'a> {
    HostCommand {
        commands: &'a TrackedCudaSlice<DeviceCommand>,
        index: u32,
    },
    ResidentEmptyRootHandle {
        handle: &'a TrackedCudaSlice<SemanticResidentHandleRecord>,
        index: u32,
    },
    ResidentFork {
        root: SemanticResidentRootHandle<'a>,
    },
    ResidentPreflightTransition {
        root: SemanticResidentRootHandle<'a>,
    },
    ResidentInsertSupport {
        candidate: SemanticResidentCandidateHandle<'a>,
        statement: SemanticResidentStatementBank<'a>,
        support: SemanticResidentSupportBank<'a>,
    },
    ResidentSeal {
        candidate: SemanticResidentCandidateHandle<'a>,
    },
    ResidentTruth {
        view: SemanticResidentView<'a>,
        statement: SemanticResidentStatementBank<'a>,
    },
    ResidentConsumeTruth {
        truth: SemanticResidentTruthView<'a>,
        output: SemanticResidentTruthSlot<'a>,
    },
    ResidentMaterializeDecoded {
        input: SemanticResidentDecodedInputBank<'a>,
        statement: SemanticResidentStatementBank<'a>,
        support: SemanticResidentSupportBank<'a>,
    },
}

struct SemanticKernelLaunchIo<'a> {
    arena: &'a mut TrackedCudaSlice<u64>,
    arena_access: ArenaAccess,
    input: SemanticKernelInput<'a>,
    receipts: &'a TrackedCudaSlice<DeviceReceipt>,
    receipt_index: u32,
}

/// Device-resident immutable-root semantic owner retaining its provider.
pub struct SemanticHypergraph {
    domain: ResidentExecutionDomain,
    stream: Arc<CudaStream>,
    execute: CudaFunction,
    arena: TrackedCudaSlice<u64>,
    command: TrackedCudaSlice<DeviceCommand>,
    receipt: TrackedCudaSlice<DeviceReceipt>,
    arena_words: u64,
    owner: u64,
    capacities: SemanticHypergraphCapacities,
    roots: Vec<SlotLedger>,
    fork: SlotLedger,
    statements: Vec<SlotLedger>,
    supports: Vec<SlotLedger>,
    versions: Vec<SlotLedger>,
    current_fork: Option<CurrentFork>,
    empty_root: SemanticRootHandle,
    admission: Option<SemanticAdmission>,
    stats: SemanticHypergraphExecutionStats,
    device_controlled: bool,
    poisoned: bool,
    // Retire device state before releasing its allocation and driver owner.
    provider: Arc<CudaKernelProvider>,
}

impl SemanticHypergraph {
    pub(crate) fn transition_owner(
        &self,
    ) -> Result<(Arc<CudaKernelProvider>, ResidentExecutionDomain), SemanticHypergraphError> {
        self.ensure_host_facade_available()?;
        if self.current_fork.is_some() || self.admission.is_none() {
            return Err(admission_error(
                "transition requires admitted records and no live candidate",
            ));
        }
        Ok((Arc::clone(&self.provider), self.domain.clone()))
    }

    /// Registers the canonical arena in the enclosing captured transaction. The
    /// consuming session owns this graph exclusively until all lane terminals.
    pub(crate) fn record_transition(&self, recorder: &mut crate::launch::LaunchRecorder) {
        recorder.read_write(&self.arena);
    }

    pub(crate) fn transition_arena(&self) -> [u64; 7] {
        [
            self.arena.device_ptr_value(),
            self.owner,
            self.capacities.roots as u64,
            self.capacities.statements as u64,
            self.capacities.supports as u64,
            self.capacities.versions as u64,
            self.arena_words,
        ]
    }

    pub(crate) fn enter_transition(&mut self) {
        self.device_controlled = true;
    }

    pub(crate) fn observe_transition_root(
        &mut self,
        words: [u64; 42],
    ) -> Result<(SemanticRootHandle, SemanticRootSnapshot), SemanticHypergraphError> {
        let receipt = DeviceReceipt { words };
        self.expect_success(&receipt)?;
        let result = (|| {
            let slot =
                checked_receipt_slot(words[3], SemanticHandleKind::Root, self.capacities.roots)?;
            if words[39] != self.owner || words[4] == 0 {
                return self.corrupt("transition root receipt has invalid owner or generation");
            }
            Ok((
                SemanticRootHandle::new(self.owner, slot, words[4]),
                SemanticRootSnapshot::new(
                    SemanticRootDigest(receipt_identity(&receipt, 16)),
                    receipt_extents(&receipt)?,
                ),
            ))
        })();
        poison_after_reconciliation_error(&mut self.poisoned, result)
    }

    pub(crate) fn observe_transition_edit(
        &mut self,
        words: [u64; 42],
    ) -> Result<Option<SemanticInsertOutcome>, SemanticHypergraphError> {
        let receipt = DeviceReceipt { words };
        self.expect_success(&receipt)?;
        let result = (|| {
            if words[39] != self.owner
                || (words[1] == OUTCOME_INSERTED
                    && (words[8] == 0 || words[10] == 0 || words[12] == 0))
                || (words[1] == OUTCOME_UNCHANGED && words[12] == 0)
            {
                return self.corrupt("transition edit receipt has invalid owner or generation");
            }
            match words[1] {
                0 => Ok(None),
                OUTCOME_UNCHANGED => Ok(Some(SemanticInsertOutcome::Unchanged(
                    self.version_ref(&receipt)?,
                ))),
                OUTCOME_INSERTED => Ok(Some(SemanticInsertOutcome::Inserted(
                    SemanticInsertedSupport {
                        statement: self.statement_ref(&receipt)?,
                        support: self.support_ref(&receipt)?,
                        version: self.version_ref(&receipt)?,
                        previous_truth: inserted_support_previous_truth(&receipt)?,
                    },
                ))),
                _ => self.corrupt("invalid transition edit outcome"),
            }
        })();
        poison_after_reconciliation_error(&mut self.poisoned, result)
    }
}

impl CudaKernelProvider {
    /// Allocates one fixed-capacity semantic graph and initializes its empty root on CUDA.
    pub fn allocate_semantic_hypergraph(
        self: &Arc<Self>,
        domain: &ResidentExecutionDomain,
        capacities: SemanticHypergraphCapacities,
    ) -> Result<SemanticHypergraph, SemanticHypergraphError> {
        validate_execution_domain(self, domain)
            .map_err(|error| runtime_error("domain validation", error))?;
        let execute = self
            .device()
            .inner()
            .get_func(MODULE, KERNEL)
            .ok_or(SemanticHypergraphError::KernelUnavailable)?;
        let arena_words = checked_arena_words(capacities)?;
        let arena_bytes =
            arena_words
                .checked_mul(8)
                .ok_or_else(|| SemanticHypergraphError::InvalidInput {
                    detail: "semantic arena byte size overflow".into(),
                })?;
        let total_bytes = arena_bytes
            .checked_add(std::mem::size_of::<DeviceCommand>() as u64)
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<DeviceReceipt>() as u64))
            .ok_or_else(|| SemanticHypergraphError::InvalidInput {
                detail: "semantic allocation byte size overflow".into(),
            })?;
        let arena_len =
            usize::try_from(arena_words).map_err(|_| SemanticHypergraphError::InvalidInput {
                detail: "semantic arena exceeds platform usize".into(),
            })?;
        let mut reservation = self
            .memory()
            .reserve_bytes(total_bytes)
            .map_err(|error| runtime_error("reservation", error))?;
        let arena = reservation
            .alloc::<u64>(arena_len)
            .map_err(|error| runtime_error("arena allocation", error))?;
        let command = reservation
            .alloc::<DeviceCommand>(1)
            .map_err(|error| runtime_error("command allocation", error))?;
        let receipt = reservation
            .alloc::<DeviceReceipt>(1)
            .map_err(|error| runtime_error("receipt allocation", error))?;
        if reservation.remaining_bytes() != 0 {
            return Err(SemanticHypergraphError::CorruptLineage {
                detail: format!(
                    "{} reserved semantic bytes were not materialized",
                    reservation.remaining_bytes()
                ),
            });
        }
        let runtime = self
            .memory()
            .runtime()
            .ok_or_else(|| SemanticHypergraphError::Runtime {
                operation: "stream resolution",
                detail: "provider has no device runtime".into(),
            })?;
        let stream = runtime
            .stream_pool()
            .resolve(domain.stream_id())
            .ok_or_else(|| SemanticHypergraphError::Runtime {
                operation: "stream resolution",
                detail: "resident stream id is not live in provider runtime".into(),
            })?;
        let owner = NEXT_OWNER_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1).filter(|next| *next != 0)
            })
            .map_err(|_| SemanticHypergraphError::GenerationExhausted {
                kind: SemanticHandleKind::Root,
                slot: 0,
            })?;
        let mut graph = SemanticHypergraph {
            provider: Arc::clone(self),
            domain: domain.clone(),
            stream,
            execute,
            arena,
            command,
            receipt,
            arena_words,
            owner,
            capacities,
            roots: vec![SlotLedger::FREE; capacities.roots as usize],
            fork: SlotLedger::FREE,
            statements: vec![SlotLedger::FREE; capacities.statements as usize],
            supports: vec![SlotLedger::FREE; capacities.supports as usize],
            versions: vec![SlotLedger::FREE; capacities.versions as usize],
            current_fork: None,
            empty_root: SemanticRootHandle::new(owner, 0, 1),
            admission: None,
            stats: SemanticHypergraphExecutionStats::default(),
            device_controlled: false,
            poisoned: false,
        };
        let command = graph.command_for(OP_INITIALIZE);
        let receipt = graph.run(command, ArenaAccess::ReadWrite)?;
        graph.expect_success(&receipt)?;
        if receipt.words[3] != 0 || receipt.words[4] != 1 {
            return Err(SemanticHypergraphError::CorruptLineage {
                detail: "CUDA initialization did not publish empty root slot 0 generation 1".into(),
            });
        }
        graph.roots[0].live = true;
        Ok(graph)
    }
}

impl SemanticHypergraph {
    /// Consumes a cold graph and imports the explicitly selected support assertions.
    ///
    /// Each index selects an admitted support record with its declared statement
    /// link. Order and duplicate-insertion semantics are preserved; unselected
    /// events remain declarations available to future proposals, not initial truth.
    /// The assertion count is bounded by `limits.max_records`.
    ///
    /// Initialization uses the existing device mutation path, then binds admission
    /// to the sealed populated root without re-resolving accepted symbol content.
    /// Errors return no partially initialized owner. This cold import does not
    /// certify external truth or provenance, or publish joint neural/text state.
    pub fn admit_initial_records(
        mut self,
        records: SemanticAdmissionRecords,
        initial_supports: &[u32],
        limits: SemanticAdmissionLimits,
    ) -> Result<Self, SemanticHypergraphError> {
        self.ensure_host_facade_available()?;
        if initial_supports.len() > limits.max_records as usize {
            return Err(admission_error(
                "initial support count exceeds admission bound",
            ));
        }
        if initial_supports
            .iter()
            .any(|&index| records.supports.get(index as usize).is_none())
        {
            return Err(admission_error(
                "initial support is outside the declared records",
            ));
        }
        let empty = self.empty_root();
        self.admit_records(empty, records, limits)?;
        if initial_supports.is_empty() {
            return Ok(self);
        }
        let fork = self.fork(empty)?;
        for &index in initial_supports {
            let admission = self.admission.as_ref().expect("validated cold admission");
            let statement = admission.records.supports[index as usize].statement;
            let key = admission.statement_key(statement)?;
            let event = admission.support_event(index)?;
            self.insert_support(fork, &key, &event)?;
        }
        let base = self.seal(fork)?;
        let base_snapshot = self.snapshot(SemanticView::Root(base))?;
        let admission = self.admission.as_mut().expect("validated cold admission");
        let identity = derive_admission_identity(
            &admission.records,
            admission.schema_generation,
            &admission.encoded_records,
            &admission.statement_keys,
            &admission.support_events,
            &base_snapshot,
        );
        admission.base = base;
        admission.base_snapshot = base_snapshot;
        admission.identity = identity;
        Ok(self)
    }

    /// Admits and owns typed content for one exact sealed base before resident execution.
    ///
    /// This is not an import of a separate relation store, and does not assert the
    /// external truth of provenance. Invalid input is rejected before CUDA work.
    /// The accepted snapshot cannot be replaced or mutated by the caller.
    pub fn admit_records(
        &mut self,
        base: SemanticRootHandle,
        records: SemanticAdmissionRecords,
        limits: SemanticAdmissionLimits,
    ) -> Result<&SemanticAdmission, SemanticHypergraphError> {
        self.ensure_host_facade_available()?;
        self.validate_root(base)?;
        if self.admission.is_some() || self.current_fork.is_some() {
            return Err(admission_error(
                "typed admission requires an unbound owner with no live fork",
            ));
        }
        let admission = admit_semantic_records(records, limits, base, || {
            self.snapshot(SemanticView::Root(base))
        })?;
        self.admission = Some(admission);
        Ok(self
            .admission
            .as_ref()
            .expect("just published complete admission"))
    }

    /// Returns the retained cold snapshot, not a re-resolution of live inputs.
    pub fn admission(&self) -> Option<&SemanticAdmission> {
        self.admission.as_ref()
    }

    /// Cold export under the transition owner's exclusive, quiescent acquisition.
    /// Native validation is intentional: resident execution invalidates host ledgers.
    pub(crate) fn export_transition_root_parts(
        &mut self,
        owner: u64,
        slot: u64,
        generation: u64,
    ) -> Result<SemanticRootMaterial, SemanticHypergraphError> {
        self.ensure_not_poisoned()?;
        if owner != self.owner {
            return Err(SemanticHypergraphError::ForeignHandle {
                kind: SemanticHandleKind::Root,
            });
        }
        let slot = checked_receipt_slot(slot, SemanticHandleKind::Root, self.capacities.roots)?;
        self.export_transition_root(SemanticRootHandle::new(owner, slot, generation))
    }

    /// Exports an existing opaque root through native generation validation.
    pub(crate) fn export_transition_root(
        &mut self,
        root: SemanticRootHandle,
    ) -> Result<SemanticRootMaterial, SemanticHypergraphError> {
        self.ensure_not_poisoned()?;
        if root.owner != self.owner {
            return Err(SemanticHypergraphError::ForeignHandle {
                kind: SemanticHandleKind::Root,
            });
        }
        if self.admission.is_none() {
            return Err(admission_error(
                "root material export requires typed admission",
            ));
        }
        let mut command = self.command_for(OP_SNAPSHOT);
        write_view(&mut command, SemanticView::Root(root));
        let receipt = self.run(command, ArenaAccess::Read)?;
        self.expect_success(&receipt)?;
        let result = (|| {
            let snapshot = SemanticRootSnapshot::new(
                SemanticRootDigest(receipt_identity(&receipt, 16)),
                receipt_extents(&receipt)?,
            );
            let arena_words = usize::try_from(self.arena_words).map_err(|_| size_overflow())?;
            let mut arena = vec![0; arena_words];
            self.provider
                .dtoh_sync_copy_into_tracked(&self.arena, &mut arena)
                .map_err(|error| runtime_error("root material arena read", error))?;
            material_from_arena(
                &arena,
                self.capacities,
                root,
                snapshot,
                self.admission.as_ref().expect("checked typed admission"),
            )
        })();
        poison_after_reconciliation_error(&mut self.poisoned, result)
    }

    /// Rematerializes a complete root on a fresh, already-admitted native owner.
    /// No original physical slot, generation, or encoded embedding is imported.
    pub(crate) fn restore_root(
        &mut self,
        material: &SemanticRootMaterial,
    ) -> Result<SemanticRootHandle, SemanticHypergraphError> {
        self.ensure_host_facade_available()?;
        let admission = self
            .admission
            .as_ref()
            .ok_or_else(|| admission_error("root restoration requires typed admission"))?;
        if admission.base != self.empty_root
            || self.current_fork.is_some()
            || self.fork.generation != 1
            || self.roots.iter().filter(|slot| slot.live).count() != 1
            || self.statements.iter().any(|slot| slot.live)
            || self.supports.iter().any(|slot| slot.live)
            || self.versions.iter().any(|slot| slot.live)
        {
            return Err(admission_error(
                "root restoration requires a fresh empty native owner",
            ));
        }
        material.validate_lineage(admission)?;
        let base_versions = material.admission_base_extents[2] as usize;
        let needed_roots = 1
            + u32::from(!material.insertions.is_empty())
            + u32::from(base_versions > 0 && base_versions < material.insertions.len());
        if material.extents[0] > self.capacities.statements
            || material.extents[1] > self.capacities.supports
            || material.extents[2] > self.capacities.versions
            || self.capacities.roots < needed_roots
        {
            return Err(admission_error(
                "root material exceeds fresh native capacities",
            ));
        }
        let result = (|| {
            let mut admission_root = self.empty_root;
            let root = if material.insertions.is_empty() {
                self.empty_root
            } else {
                let mut fork = self.fork(self.empty_root)?;
                for (index, insertion) in material.insertions.iter().enumerate() {
                    let admission = self.admission.as_ref().expect("checked typed admission");
                    let key = material_statement_key(
                        admission,
                        insertion.statement,
                        &insertion.reconstruction,
                    )?;
                    let event = admission.support_event(insertion.support)?;
                    match self.insert_support_reconstructed(
                        fork,
                        &key,
                        &event,
                        &insertion.reconstruction,
                    )? {
                        SemanticInsertOutcome::Inserted(inserted)
                            if inserted.version.identity.0 == insertion.version => {}
                        _ => {
                            return self
                                .corrupt("restored insertion differs from canonical root material")
                        }
                    }
                    if index + 1 == base_versions && base_versions < material.insertions.len() {
                        admission_root = self.seal(fork)?;
                        fork = self.fork(admission_root)?;
                    }
                }
                self.seal(fork)?
            };
            if base_versions == material.insertions.len() {
                admission_root = root;
            }
            let snapshot = self.snapshot(SemanticView::Root(root))?;
            if snapshot.digest.0 != material.digest
                || snapshot.extents
                    != SemanticExtents::new(
                        material.extents[0],
                        material.extents[1],
                        material.extents[2],
                    )
            {
                return self.corrupt(
                    "fresh native root differs from restored digest or reachable extents",
                );
            }
            let base_snapshot = if admission_root == root {
                snapshot
            } else {
                self.snapshot(SemanticView::Root(admission_root))?
            };
            if base_snapshot.digest.0 != material.admission_base_digest
                || base_snapshot.extents
                    != SemanticExtents::new(
                        material.admission_base_extents[0],
                        material.admission_base_extents[1],
                        material.admission_base_extents[2],
                    )
            {
                return self.corrupt(
                    "fresh native admission base differs from its retained original binding",
                );
            }
            let admission = self.admission.as_mut().expect("checked typed admission");
            admission.base = admission_root;
            admission.base_snapshot = base_snapshot;
            admission.identity = derive_admission_identity(
                &admission.records,
                admission.schema_generation,
                &admission.encoded_records,
                &admission.statement_keys,
                &admission.support_events,
                &base_snapshot,
            );
            Ok(root)
        })();
        poison_after_reconciliation_error(&mut self.poisoned, result)
    }

    fn validate_statement_key(
        &self,
        key: &SemanticStatementKey,
    ) -> Result<(), SemanticHypergraphError> {
        if key.owner != self.owner
            || self
                .admission
                .as_ref()
                .and_then(|admission| admission.statement_keys.get(key.record as usize))
                != Some(&Some(*key))
        {
            return Err(admission_error(
                "statement key is not from this owner's typed admission",
            ));
        }
        Ok(())
    }

    pub const fn empty_root(&self) -> SemanticRootHandle {
        self.empty_root
    }

    pub const fn execution_stats(&self) -> SemanticHypergraphExecutionStats {
        self.stats
    }

    pub fn fork(
        &mut self,
        base: SemanticRootHandle,
    ) -> Result<SemanticForkHandle, SemanticHypergraphError> {
        self.ensure_host_facade_available()?;
        self.validate_root(base)?;
        if self.current_fork.is_some() {
            return Err(SemanticHypergraphError::InactiveHandle {
                kind: SemanticHandleKind::Fork,
                slot: 0,
            });
        }
        let mut command = self.command_for(OP_FORK);
        command.words[1] = base.owner;
        command.words[8] = u64::from(base.slot);
        command.words[9] = base.generation;
        let receipt = self.run(command, ArenaAccess::ReadWrite)?;
        self.expect_success(&receipt)?;
        let result = (|| {
            let slot = checked_receipt_slot(receipt.words[5], SemanticHandleKind::Fork, 1)?;
            let generation = receipt.words[6];
            if slot != 0 || generation != self.fork.generation {
                return self.corrupt("fork receipt disagrees with host physical generation");
            }
            self.fork.live = true;
            let handle = SemanticForkHandle::new(self.owner, slot, generation);
            self.current_fork = Some(CurrentFork {
                handle,
                staged_statements: Vec::new(),
                staged_supports: Vec::new(),
                staged_versions: Vec::new(),
            });
            Ok(handle)
        })();
        poison_after_reconciliation_error(&mut self.poisoned, result)
    }

    pub fn insert_support(
        &mut self,
        fork: SemanticForkHandle,
        statement: &SemanticStatementKey,
        event: &SemanticSupportEvent,
    ) -> Result<SemanticInsertOutcome, SemanticHypergraphError> {
        self.insert_support_reconstructed(fork, statement, event, &[0; 10])
    }

    fn insert_support_reconstructed(
        &mut self,
        fork: SemanticForkHandle,
        statement: &SemanticStatementKey,
        event: &SemanticSupportEvent,
        reconstruction: &[u32; 10],
    ) -> Result<SemanticInsertOutcome, SemanticHypergraphError> {
        self.ensure_host_facade_available()?;
        self.validate_fork(fork)?;
        if statement.record == u32::MAX {
            let admission = self.admission.as_ref().ok_or_else(|| {
                admission_error("derived insertion requires retained typed admission")
            })?;
            if material_statement_key(admission, None, reconstruction)? != *statement {
                return Err(admission_error(
                    "derived insertion key differs from its typed reconstruction",
                ));
            }
        } else {
            if *reconstruction != [0; 10] {
                return Err(admission_error(
                    "original insertion has derived reconstruction references",
                ));
            }
            self.validate_statement_key(statement)?;
        }
        if event.owner != self.owner {
            return Err(admission_error(
                "support event does not belong to this owner's typed admission",
            ));
        }
        let mut command = self.command_for(OP_INSERT_SUPPORT);
        encode_support_insertion(&mut command, fork, statement, event, reconstruction);
        let receipt = self.run(command, ArenaAccess::ReadWrite)?;
        self.expect_success(&receipt)?;
        let result = (|| {
            let statement_ref = self.statement_ref(&receipt)?;
            let version_ref = self.version_ref(&receipt)?;
            match receipt.words[1] {
                OUTCOME_UNCHANGED => Ok(SemanticInsertOutcome::Unchanged(version_ref)),
                OUTCOME_INSERTED => {
                    let previous_truth = inserted_support_previous_truth(&receipt)?;
                    let support_ref = self.support_ref(&receipt)?;
                    let flags = receipt.words[32];
                    if flags & INSERT_NEW_STATEMENT != 0 {
                        self.publish_statement(statement_ref.handle)?;
                        self.current_fork
                            .as_mut()
                            .expect("validated live fork")
                            .staged_statements
                            .push(statement_ref.handle.slot);
                    } else {
                        self.validate_statement(statement_ref.handle)?;
                    }
                    self.publish_support(support_ref.handle)?;
                    self.publish_version(version_ref.handle)?;
                    let current = self.current_fork.as_mut().expect("validated live fork");
                    current.staged_supports.push(support_ref.handle.slot);
                    current.staged_versions.push(version_ref.handle.slot);
                    Ok(SemanticInsertOutcome::Inserted(SemanticInsertedSupport {
                        statement: statement_ref,
                        support: support_ref,
                        version: version_ref,
                        previous_truth,
                    }))
                }
                other => self.corrupt(format!("invalid insertion outcome {other}")),
            }
        })();
        poison_after_reconciliation_error(&mut self.poisoned, result)
    }

    pub fn discard(&mut self, fork: SemanticForkHandle) -> Result<(), SemanticHypergraphError> {
        self.ensure_host_facade_available()?;
        self.validate_fork(fork)?;
        let mut command = self.command_for(OP_DISCARD);
        command.words[1] = fork.owner;
        command.words[8] = u64::from(fork.slot);
        command.words[9] = fork.generation;
        let receipt = self.run(command, ArenaAccess::ReadWrite)?;
        self.expect_success(&receipt)?;
        let result = (|| {
            let current = self.current_fork.take().expect("validated live fork");
            for slot in current.staged_statements {
                retire_slot(
                    &mut self.statements[slot as usize],
                    SemanticHandleKind::Statement,
                    slot,
                )?;
            }
            for slot in current.staged_supports {
                retire_slot(
                    &mut self.supports[slot as usize],
                    SemanticHandleKind::Support,
                    slot,
                )?;
            }
            for slot in current.staged_versions {
                retire_slot(
                    &mut self.versions[slot as usize],
                    SemanticHandleKind::Version,
                    slot,
                )?;
            }
            retire_slot(&mut self.fork, SemanticHandleKind::Fork, 0)?;
            if receipt.words[6] != self.fork.generation {
                return self.corrupt("discard receipt disagrees with advanced fork generation");
            }
            Ok(())
        })();
        poison_after_reconciliation_error(&mut self.poisoned, result)
    }

    pub fn seal(
        &mut self,
        fork: SemanticForkHandle,
    ) -> Result<SemanticRootHandle, SemanticHypergraphError> {
        self.ensure_host_facade_available()?;
        self.validate_fork(fork)?;
        let mut command = self.command_for(OP_SEAL);
        command.words[1] = fork.owner;
        command.words[8] = u64::from(fork.slot);
        command.words[9] = fork.generation;
        let receipt = self.run(command, ArenaAccess::ReadWrite)?;
        self.expect_success(&receipt)?;
        let result = (|| {
            let slot = checked_receipt_slot(
                receipt.words[3],
                SemanticHandleKind::Root,
                self.capacities.roots,
            )?;
            let generation = receipt.words[4];
            let unchanged = match receipt.words[1] {
                OUTCOME_INSERTED => false,
                OUTCOME_UNCHANGED => true,
                _ => return self.corrupt("seal receipt has an invalid outcome"),
            };
            if unchanged {
                let current = self.current_fork.as_ref().expect("validated live fork");
                if !current.staged_statements.is_empty()
                    || !current.staged_supports.is_empty()
                    || !current.staged_versions.is_empty()
                {
                    return self.corrupt("unchanged seal discarded staged insertions");
                }
            }
            let root_ledger = &mut self.roots[slot as usize];
            if root_ledger.live != unchanged || root_ledger.generation != generation {
                return self.corrupt("seal receipt disagrees with root physical generation");
            }
            root_ledger.live = true;
            self.current_fork.take().expect("validated live fork");
            retire_slot(&mut self.fork, SemanticHandleKind::Fork, 0)?;
            if receipt.words[6] != self.fork.generation {
                return self.corrupt("seal receipt disagrees with advanced fork generation");
            }
            Ok(SemanticRootHandle::new(self.owner, slot, generation))
        })();
        poison_after_reconciliation_error(&mut self.poisoned, result)
    }

    pub fn snapshot(
        &mut self,
        view: SemanticView,
    ) -> Result<SemanticRootSnapshot, SemanticHypergraphError> {
        self.ensure_host_facade_available()?;
        self.validate_view(view)?;
        let mut command = self.command_for(OP_SNAPSHOT);
        write_view(&mut command, view);
        let receipt = self.run(command, ArenaAccess::Read)?;
        self.expect_success(&receipt)?;
        let result = (|| {
            let extents = receipt_extents(&receipt)?;
            let digest = SemanticRootDigest(receipt_identity(&receipt, 16));
            Ok(SemanticRootSnapshot::new(digest, extents))
        })();
        poison_after_reconciliation_error(&mut self.poisoned, result)
    }

    pub fn truth(
        &mut self,
        view: SemanticView,
        statement: &SemanticStatementKey,
    ) -> Result<SemanticTruth, SemanticHypergraphError> {
        self.ensure_host_facade_available()?;
        self.validate_view(view)?;
        self.validate_statement_key(statement)?;
        let mut command = self.command_for(OP_TRUTH);
        write_view(&mut command, view);
        command.words[16..20].copy_from_slice(&identity_words(statement.identity.0));
        let receipt = self.run(command, ArenaAccess::Read)?;
        self.expect_success(&receipt)?;
        let result = SemanticTruth::from_bits(receipt.words[2]);
        poison_after_reconciliation_error(&mut self.poisoned, result)
    }

    pub fn inspect_statement(
        &mut self,
        view: SemanticView,
        handle: SemanticStatementHandle,
    ) -> Result<SemanticStatementRef, SemanticHypergraphError> {
        self.ensure_host_facade_available()?;
        self.validate_view(view)?;
        self.validate_statement(handle)?;
        let mut command = self.command_for(OP_INSPECT_STATEMENT);
        write_view(&mut command, view);
        command.words[10] = u64::from(handle.slot);
        command.words[11] = handle.generation;
        let receipt = self.run(command, ArenaAccess::Read)?;
        self.expect_success(&receipt)?;
        let result = self.statement_ref(&receipt);
        poison_after_reconciliation_error(&mut self.poisoned, result)
    }

    pub fn inspect_support(
        &mut self,
        view: SemanticView,
        handle: SemanticSupportHandle,
    ) -> Result<SemanticSupportRef, SemanticHypergraphError> {
        self.ensure_host_facade_available()?;
        self.validate_view(view)?;
        self.validate_support(handle)?;
        let mut command = self.command_for(OP_INSPECT_SUPPORT);
        write_view(&mut command, view);
        command.words[10] = u64::from(handle.slot);
        command.words[11] = handle.generation;
        let receipt = self.run(command, ArenaAccess::Read)?;
        self.expect_success(&receipt)?;
        let result = self.support_ref(&receipt);
        poison_after_reconciliation_error(&mut self.poisoned, result)
    }

    pub fn inspect_version(
        &mut self,
        view: SemanticView,
        handle: SemanticVersionHandle,
    ) -> Result<SemanticVersionRef, SemanticHypergraphError> {
        self.ensure_host_facade_available()?;
        self.validate_view(view)?;
        self.validate_version(handle)?;
        let mut command = self.command_for(OP_INSPECT_VERSION);
        write_view(&mut command, view);
        command.words[10] = u64::from(handle.slot);
        command.words[11] = handle.generation;
        let receipt = self.run(command, ArenaAccess::Read)?;
        self.expect_success(&receipt)?;
        let result = self.version_ref(&receipt);
        poison_after_reconciliation_error(&mut self.poisoned, result)
    }

    fn command_for(&self, operation: u64) -> DeviceCommand {
        let mut command = DeviceCommand::default();
        command.words[0] = operation;
        command.words[1] = self.owner;
        command.words[2] = u64::from(self.capacities.roots);
        command.words[3] = u64::from(self.capacities.statements);
        command.words[4] = u64::from(self.capacities.supports);
        command.words[5] = u64::from(self.capacities.versions);
        command.words[6] = self.arena_words;
        command
    }

    /// Materializes the immutable empty-root handle into a resident typed bank.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "reserved for crate-internal device-resident pipelines"
        )
    )]
    pub(crate) fn enqueue_resident_empty_root_handle<'handle, 'receipt>(
        &mut self,
        handle: SemanticResidentHandleSlot<'handle>,
        receipt: SemanticResidentReceiptSlot<'receipt>,
    ) -> Result<SemanticResidentRootHandleBank<'handle>, SemanticHypergraphError> {
        let SemanticResidentHandleSlot {
            allocation: handles,
            index: handle_index,
        } = handle;
        self.enqueue_resident_input(
            SemanticKernelInput::ResidentEmptyRootHandle {
                handle: handles,
                index: handle_index,
            },
            ArenaAccess::Read,
            false,
            receipt,
        )?;
        Ok(SemanticResidentRootHandleBank {
            allocation: handles,
            index: handle_index,
        })
    }

    /// Checks both lanes' full worst-case reserve without acquiring scratch.
    /// The owner must remain exclusive through both lane terminals; the returned
    /// root receipt propagates refusal, but is not a reusable allocation permit.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "consumed by the resident transition owner")
    )]
    pub(crate) fn enqueue_resident_preflight_transition<'input, 'receipt>(
        &mut self,
        root: SemanticResidentRootHandle<'input>,
        receipt: SemanticResidentReceiptSlot<'receipt>,
    ) -> Result<SemanticResidentRootHandle<'receipt>, SemanticHypergraphError> {
        self.enqueue_resident_input(
            SemanticKernelInput::ResidentPreflightTransition { root },
            ArenaAccess::Read,
            false,
            receipt,
        )
        .map(SemanticResidentRootHandle::Receipt)
    }

    /// Forks one immutable root selected from a resident generation-bound handle bank.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "reserved for crate-internal device-resident pipelines"
        )
    )]
    pub(crate) fn enqueue_resident_fork<'input, 'receipt>(
        &mut self,
        root: SemanticResidentRootHandle<'input>,
        receipt: SemanticResidentReceiptSlot<'receipt>,
    ) -> Result<SemanticResidentMutationReceipt<'receipt>, SemanticHypergraphError> {
        let receipt = self.enqueue_resident_input(
            SemanticKernelInput::ResidentFork { root },
            ArenaAccess::ReadWrite,
            true,
            receipt,
        )?;
        Ok(SemanticResidentMutationReceipt { receipt })
    }

    /// Inserts one support from typed resident decoded banks and a resident candidate handle.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "reserved for crate-internal device-resident pipelines"
        )
    )]
    pub(crate) fn enqueue_resident_insert_support<'input, 'receipt>(
        &mut self,
        candidate: SemanticResidentCandidateHandle<'input>,
        statement: SemanticResidentStatementBank<'input>,
        support: SemanticResidentSupportBank<'input>,
        receipt: SemanticResidentReceiptSlot<'receipt>,
    ) -> Result<SemanticResidentMutationReceipt<'receipt>, SemanticHypergraphError> {
        let receipt = self.enqueue_resident_input(
            SemanticKernelInput::ResidentInsertSupport {
                candidate,
                statement,
                support,
            },
            ArenaAccess::ReadWrite,
            true,
            receipt,
        )?;
        Ok(SemanticResidentMutationReceipt { receipt })
    }

    /// Seals a resident candidate whose generation is carried by a prior device receipt.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "reserved for crate-internal device-resident pipelines"
        )
    )]
    pub(crate) fn enqueue_resident_seal<'input, 'receipt>(
        &mut self,
        candidate: SemanticResidentCandidateHandle<'input>,
        receipt: SemanticResidentReceiptSlot<'receipt>,
    ) -> Result<SemanticResidentSealReceipt<'receipt>, SemanticHypergraphError> {
        let receipt = self.enqueue_resident_input(
            SemanticKernelInput::ResidentSeal { candidate },
            ArenaAccess::ReadWrite,
            true,
            receipt,
        )?;
        Ok(SemanticResidentSealReceipt { receipt })
    }

    /// Derives truth from an immutable root or live fork and a decoded statement identity.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "reserved for crate-internal device-resident pipelines"
        )
    )]
    pub(crate) fn enqueue_resident_truth<'input, 'receipt>(
        &mut self,
        view: SemanticResidentView<'input>,
        statement: SemanticResidentStatementBank<'input>,
        receipt: SemanticResidentReceiptSlot<'receipt>,
    ) -> Result<SemanticResidentTruthReceipt<'receipt>, SemanticHypergraphError> {
        let receipt = self.enqueue_resident_input(
            SemanticKernelInput::ResidentTruth { view, statement },
            ArenaAccess::Read,
            false,
            receipt,
        )?;
        Ok(SemanticResidentTruthReceipt { receipt })
    }

    /// Validates a pending truth receipt on device and projects its semantic value.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "reserved for crate-internal device-resident pipelines"
        )
    )]
    pub(crate) fn enqueue_resident_truth_consumer<'input, 'output, 'receipt>(
        &mut self,
        truth: SemanticResidentTruthView<'input>,
        output: SemanticResidentTruthSlot<'output>,
        receipt: SemanticResidentReceiptSlot<'receipt>,
    ) -> Result<SemanticResidentReceiptView<'receipt>, SemanticHypergraphError> {
        self.enqueue_resident_input(
            SemanticKernelInput::ResidentConsumeTruth { truth, output },
            ArenaAccess::Read,
            false,
            receipt,
        )
    }

    /// Splits a decoder-owned resident record into canonical semantic input banks.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "reserved for crate-internal device-resident pipelines"
        )
    )]
    pub(crate) fn enqueue_resident_materialize_decoded<'input, 'receipt>(
        &mut self,
        input: SemanticResidentDecodedInputBank<'input>,
        statement: SemanticResidentStatementBank<'input>,
        support: SemanticResidentSupportBank<'input>,
        receipt: SemanticResidentReceiptSlot<'receipt>,
    ) -> Result<SemanticResidentReceiptView<'receipt>, SemanticHypergraphError> {
        self.enqueue_resident_input(
            SemanticKernelInput::ResidentMaterializeDecoded {
                input,
                statement,
                support,
            },
            ArenaAccess::Read,
            false,
            receipt,
        )
    }

    fn enqueue_resident_input<'receipt>(
        &mut self,
        input: SemanticKernelInput<'_>,
        arena_access: ArenaAccess,
        device_mutation: bool,
        receipt: SemanticResidentReceiptSlot<'receipt>,
    ) -> Result<SemanticResidentReceiptView<'receipt>, SemanticHypergraphError> {
        self.ensure_not_poisoned()?;
        let next_launches = self.next_launch_count()?;
        let SemanticResidentReceiptSlot {
            allocation: receipts,
            index: receipt_index,
        } = receipt;
        enqueue_device_command(
            SemanticKernelLaunchSpec {
                domain: &self.domain,
                execute: &self.execute,
                owner: self.owner,
                capacities: self.capacities,
                arena_words: self.arena_words,
            },
            SemanticKernelLaunchIo {
                arena: &mut self.arena,
                arena_access,
                input,
                receipts,
                receipt_index,
            },
            &mut self.poisoned,
        )?;
        self.stats.cuda_kernel_launches = next_launches;
        self.device_controlled |= device_mutation;
        Ok(SemanticResidentReceiptView {
            allocation: receipts,
            index: receipt_index,
        })
    }

    fn run(
        &mut self,
        command: DeviceCommand,
        arena_access: ArenaAccess,
    ) -> Result<DeviceReceipt, SemanticHypergraphError> {
        let next_launches = self.next_launch_count()?;
        self.provider
            .htod_launch_metadata_sync_copy_into(&[command], &mut self.command)
            .map_err(|error| runtime_error("command upload", error))?;
        enqueue_device_command(
            SemanticKernelLaunchSpec {
                domain: &self.domain,
                execute: &self.execute,
                owner: self.owner,
                capacities: self.capacities,
                arena_words: self.arena_words,
            },
            SemanticKernelLaunchIo {
                arena: &mut self.arena,
                arena_access,
                input: SemanticKernelInput::HostCommand {
                    commands: &self.command,
                    index: 0,
                },
                receipts: &mut self.receipt,
                receipt_index: 0,
            },
            &mut self.poisoned,
        )?;
        self.stats.cuda_kernel_launches = next_launches;
        if let Err(error) = self.stream.synchronize() {
            self.poisoned = true;
            return Err(SemanticHypergraphError::Runtime {
                operation: "stream synchronization",
                detail: error.to_string(),
            });
        }
        let mut receipts = match self
            .provider
            .dtoh_small_metadata_untracked(&self.receipt, 1)
        {
            Ok(receipts) => receipts,
            Err(error) => {
                self.poisoned = true;
                return Err(runtime_error("receipt read", error));
            }
        };
        receipts.pop().ok_or_else(|| {
            self.poisoned = true;
            SemanticHypergraphError::CorruptLineage {
                detail: "CUDA receipt read returned no record".into(),
            }
        })
    }

    fn next_launch_count(&self) -> Result<u64, SemanticHypergraphError> {
        self.stats.cuda_kernel_launches.checked_add(1).ok_or(
            SemanticHypergraphError::GenerationExhausted {
                kind: SemanticHandleKind::Root,
                slot: 0,
            },
        )
    }

    fn expect_success(&mut self, receipt: &DeviceReceipt) -> Result<(), SemanticHypergraphError> {
        let status = receipt.words[0];
        if status == STATUS_OK {
            return Ok(());
        }
        let kind = decode_kind(receipt.words[33]);
        let slot = u32::try_from(receipt.words[34]).unwrap_or(u32::MAX);
        let result = match status {
            STATUS_FOREIGN_OWNER => Err(SemanticHypergraphError::ForeignHandle { kind }),
            STATUS_SLOT_OUT_OF_RANGE => Err(SemanticHypergraphError::SlotOutOfRange {
                kind,
                slot,
                capacity: u32::try_from(receipt.words[37]).unwrap_or(u32::MAX),
            }),
            STATUS_STALE_GENERATION => Err(SemanticHypergraphError::StaleGeneration {
                kind,
                slot,
                presented: receipt.words[35],
                current: receipt.words[36],
            }),
            STATUS_INACTIVE => Err(SemanticHypergraphError::InactiveHandle { kind, slot }),
            STATUS_NOT_REACHABLE => Err(SemanticHypergraphError::NotReachable { kind, slot }),
            STATUS_STATEMENT_CAPACITY => Err(SemanticHypergraphError::CapacityExceeded {
                kind: SemanticHandleKind::Statement,
                capacity: self.capacities.statements,
            }),
            STATUS_SUPPORT_CAPACITY => Err(SemanticHypergraphError::CapacityExceeded {
                kind: SemanticHandleKind::Support,
                capacity: self.capacities.supports,
            }),
            STATUS_VERSION_CAPACITY => Err(SemanticHypergraphError::CapacityExceeded {
                kind: SemanticHandleKind::Version,
                capacity: self.capacities.versions,
            }),
            STATUS_ROOT_CAPACITY => Err(SemanticHypergraphError::CapacityExceeded {
                kind: SemanticHandleKind::Root,
                capacity: self.capacities.roots,
            }),
            STATUS_GENERATION_EXHAUSTED => {
                Err(SemanticHypergraphError::GenerationExhausted { kind, slot })
            }
            STATUS_CORRUPT_LINEAGE => Err(SemanticHypergraphError::CorruptLineage {
                detail: format!("device validation code {}", receipt.words[38]),
            }),
            STATUS_INVALID_COMMAND => Err(SemanticHypergraphError::CorruptLineage {
                detail: "device rejected an internal operation code".into(),
            }),
            STATUS_ARENA_MISMATCH => Err(SemanticHypergraphError::CorruptLineage {
                detail: "device and host semantic arena layouts disagree".into(),
            }),
            _ => Err(SemanticHypergraphError::CorruptLineage {
                detail: format!("unknown device status {status}"),
            }),
        };
        if device_status_is_integrity_failure(status) {
            self.poisoned = true;
        }
        result
    }

    fn statement_ref(
        &self,
        receipt: &DeviceReceipt,
    ) -> Result<SemanticStatementRef, SemanticHypergraphError> {
        let slot = checked_receipt_slot(
            receipt.words[7],
            SemanticHandleKind::Statement,
            self.capacities.statements,
        )?;
        Ok(SemanticStatementRef {
            handle: SemanticStatementHandle::new(self.owner, slot, receipt.words[8]),
            identity: SemanticStatementIdentity(receipt_identity(receipt, 20)),
        })
    }

    fn support_ref(
        &self,
        receipt: &DeviceReceipt,
    ) -> Result<SemanticSupportRef, SemanticHypergraphError> {
        let slot = checked_receipt_slot(
            receipt.words[9],
            SemanticHandleKind::Support,
            self.capacities.supports,
        )?;
        Ok(SemanticSupportRef {
            handle: SemanticSupportHandle::new(self.owner, slot, receipt.words[10]),
            identity: SemanticSupportIdentity(receipt_identity(receipt, 24)),
        })
    }

    fn version_ref(
        &self,
        receipt: &DeviceReceipt,
    ) -> Result<SemanticVersionRef, SemanticHypergraphError> {
        let slot = checked_receipt_slot(
            receipt.words[11],
            SemanticHandleKind::Version,
            self.capacities.versions,
        )?;
        Ok(SemanticVersionRef {
            handle: SemanticVersionHandle::new(self.owner, slot, receipt.words[12]),
            identity: SemanticVersionIdentity(receipt_identity(receipt, 28)),
            truth: SemanticTruth::from_bits(receipt.words[2])?,
        })
    }

    fn publish_statement(
        &mut self,
        handle: SemanticStatementHandle,
    ) -> Result<(), SemanticHypergraphError> {
        publish_slot(
            &mut self.statements,
            handle.slot,
            handle.generation,
            SemanticHandleKind::Statement,
        )
    }

    fn publish_support(
        &mut self,
        handle: SemanticSupportHandle,
    ) -> Result<(), SemanticHypergraphError> {
        publish_slot(
            &mut self.supports,
            handle.slot,
            handle.generation,
            SemanticHandleKind::Support,
        )
    }

    fn publish_version(
        &mut self,
        handle: SemanticVersionHandle,
    ) -> Result<(), SemanticHypergraphError> {
        publish_slot(
            &mut self.versions,
            handle.slot,
            handle.generation,
            SemanticHandleKind::Version,
        )
    }

    fn validate_view(&self, view: SemanticView) -> Result<(), SemanticHypergraphError> {
        match view {
            SemanticView::Root(handle) => self.validate_root(handle),
            SemanticView::Fork(handle) => self.validate_fork(handle),
        }
    }

    fn validate_root(&self, handle: SemanticRootHandle) -> Result<(), SemanticHypergraphError> {
        validate_slot(
            self.owner,
            handle.owner,
            handle.slot,
            handle.generation,
            &self.roots,
            SemanticHandleKind::Root,
        )
    }

    fn validate_fork(&self, handle: SemanticForkHandle) -> Result<(), SemanticHypergraphError> {
        if handle.owner != self.owner {
            return Err(SemanticHypergraphError::ForeignHandle {
                kind: SemanticHandleKind::Fork,
            });
        }
        if handle.slot != 0 {
            return Err(SemanticHypergraphError::SlotOutOfRange {
                kind: SemanticHandleKind::Fork,
                slot: handle.slot,
                capacity: 1,
            });
        }
        if handle.generation != self.fork.generation {
            return Err(SemanticHypergraphError::StaleGeneration {
                kind: SemanticHandleKind::Fork,
                slot: 0,
                presented: handle.generation,
                current: self.fork.generation,
            });
        }
        if !self.fork.live
            || self
                .current_fork
                .as_ref()
                .is_none_or(|current| current.handle != handle)
        {
            return Err(SemanticHypergraphError::InactiveHandle {
                kind: SemanticHandleKind::Fork,
                slot: 0,
            });
        }
        Ok(())
    }

    fn validate_statement(
        &self,
        handle: SemanticStatementHandle,
    ) -> Result<(), SemanticHypergraphError> {
        validate_slot(
            self.owner,
            handle.owner,
            handle.slot,
            handle.generation,
            &self.statements,
            SemanticHandleKind::Statement,
        )
    }

    fn validate_support(
        &self,
        handle: SemanticSupportHandle,
    ) -> Result<(), SemanticHypergraphError> {
        validate_slot(
            self.owner,
            handle.owner,
            handle.slot,
            handle.generation,
            &self.supports,
            SemanticHandleKind::Support,
        )
    }

    fn validate_version(
        &self,
        handle: SemanticVersionHandle,
    ) -> Result<(), SemanticHypergraphError> {
        validate_slot(
            self.owner,
            handle.owner,
            handle.slot,
            handle.generation,
            &self.versions,
            SemanticHandleKind::Version,
        )
    }

    pub(crate) fn ensure_not_poisoned(&self) -> Result<(), SemanticHypergraphError> {
        if self.poisoned {
            Err(SemanticHypergraphError::Poisoned)
        } else {
            Ok(())
        }
    }

    fn ensure_host_facade_available(&self) -> Result<(), SemanticHypergraphError> {
        self.ensure_not_poisoned()?;
        if self.device_controlled {
            return Err(SemanticHypergraphError::DeviceControlled);
        }
        Ok(())
    }

    fn corrupt<T>(&self, detail: impl Into<String>) -> Result<T, SemanticHypergraphError> {
        Err(SemanticHypergraphError::CorruptLineage {
            detail: detail.into(),
        })
    }
}

fn device_status_is_integrity_failure(status: u64) -> bool {
    !matches!(
        status,
        STATUS_OK
            | STATUS_NOT_REACHABLE
            | STATUS_STATEMENT_CAPACITY
            | STATUS_SUPPORT_CAPACITY
            | STATUS_VERSION_CAPACITY
            | STATUS_ROOT_CAPACITY
            | STATUS_GENERATION_EXHAUSTED
    )
}

fn poison_after_reconciliation_error<T>(
    poisoned: &mut bool,
    result: Result<T, SemanticHypergraphError>,
) -> Result<T, SemanticHypergraphError> {
    if result.is_err() {
        *poisoned = true;
    }
    result
}

fn checked_arena_words(
    capacities: SemanticHypergraphCapacities,
) -> Result<u64, SemanticHypergraphError> {
    let roots = u64::from(capacities.roots);
    let statements = u64::from(capacities.statements);
    let supports = u64::from(capacities.supports);
    let versions = u64::from(capacities.versions);
    CONTROL_WORDS
        .checked_add(roots.checked_mul(ROOT_WORDS).ok_or_else(size_overflow)?)
        .and_then(|words| words.checked_add(CANDIDATE_WORDS))
        .and_then(|words| words.checked_add(statements.checked_mul(STATEMENT_WORDS)?))
        .and_then(|words| words.checked_add(supports.checked_mul(SUPPORT_WORDS)?))
        .and_then(|words| words.checked_add(versions.checked_mul(VERSION_WORDS)?))
        .and_then(|words| words.checked_add(roots.checked_mul(statements)?))
        .and_then(|words| words.checked_add(statements))
        .ok_or_else(size_overflow)
}

fn size_overflow() -> SemanticHypergraphError {
    SemanticHypergraphError::InvalidInput {
        detail: "semantic arena size overflow".into(),
    }
}

fn validate_slot(
    expected_owner: u64,
    presented_owner: u64,
    slot: u32,
    generation: u64,
    ledger: &[SlotLedger],
    kind: SemanticHandleKind,
) -> Result<(), SemanticHypergraphError> {
    if presented_owner != expected_owner {
        return Err(SemanticHypergraphError::ForeignHandle { kind });
    }
    let entry = ledger
        .get(slot as usize)
        .ok_or(SemanticHypergraphError::SlotOutOfRange {
            kind,
            slot,
            capacity: u32::try_from(ledger.len()).unwrap_or(u32::MAX),
        })?;
    if generation != entry.generation {
        return Err(SemanticHypergraphError::StaleGeneration {
            kind,
            slot,
            presented: generation,
            current: entry.generation,
        });
    }
    if !entry.live {
        return Err(SemanticHypergraphError::InactiveHandle { kind, slot });
    }
    Ok(())
}

fn publish_slot(
    ledger: &mut [SlotLedger],
    slot: u32,
    generation: u64,
    kind: SemanticHandleKind,
) -> Result<(), SemanticHypergraphError> {
    let capacity = u32::try_from(ledger.len()).unwrap_or(u32::MAX);
    let entry = ledger
        .get_mut(slot as usize)
        .ok_or(SemanticHypergraphError::SlotOutOfRange {
            kind,
            slot,
            capacity,
        })?;
    if entry.live || entry.generation != generation {
        return Err(SemanticHypergraphError::CorruptLineage {
            detail: format!("device published inconsistent {kind:?} slot {slot}"),
        });
    }
    entry.live = true;
    Ok(())
}

fn retire_slot(
    slot: &mut SlotLedger,
    kind: SemanticHandleKind,
    index: u32,
) -> Result<(), SemanticHypergraphError> {
    slot.generation = slot
        .generation
        .checked_add(1)
        .ok_or(SemanticHypergraphError::GenerationExhausted { kind, slot: index })?;
    slot.live = false;
    Ok(())
}

fn enqueue_device_command(
    spec: SemanticKernelLaunchSpec<'_>,
    io: SemanticKernelLaunchIo<'_>,
    poisoned: &mut bool,
) -> Result<(), SemanticHypergraphError> {
    let SemanticKernelLaunchIo {
        arena,
        arena_access,
        input,
        receipts,
        receipt_index,
    } = io;
    let mut recorder = spec.domain.new_strict_recorder();
    match arena_access {
        ArenaAccess::Read => {
            recorder.read(arena);
        }
        ArenaAccess::ReadWrite => {
            recorder.read_write(arena);
        }
    }
    recorder.write(receipts);
    let mut descriptor = DeviceLaunchDescriptor {
        expected_owner: spec.owner,
        root_capacity: u64::from(spec.capacities.roots),
        statement_capacity: u64::from(spec.capacities.statements),
        support_capacity: u64::from(spec.capacities.supports),
        version_capacity: u64::from(spec.capacities.versions),
        arena_words: spec.arena_words,
        command_ptr: 0,
        command_index: 0,
        handle_ptr: 0,
        handle_index: 0,
        source_receipt_ptr: 0,
        source_receipt_index: 0,
        decoded_input_ptr: 0,
        decoded_input_index: 0,
        decoded_statement_ptr: 0,
        decoded_statement_index: 0,
        decoded_support_ptr: 0,
        decoded_support_index: 0,
        output_ptr: 0,
        output_index: 0,
        receipt_ptr: receipts.device_ptr_value(),
        receipt_index: u64::from(receipt_index),
        admission: HOST_COMMAND_ADMISSION,
        abi_generation: HYPERGRAPH_ABI_GENERATION,
    };
    let preflight = matches!(
        &input,
        SemanticKernelInput::ResidentPreflightTransition { .. }
    );
    match input {
        SemanticKernelInput::HostCommand { commands, index } => {
            recorder.read(commands);
            descriptor.command_ptr = commands.device_ptr_value();
            descriptor.command_index = u64::from(index);
        }
        SemanticKernelInput::ResidentEmptyRootHandle { handle, index } => {
            recorder.write(handle);
            descriptor.handle_ptr = handle.device_ptr_value();
            descriptor.handle_index = u64::from(index);
            descriptor.admission = RESIDENT_EMPTY_ROOT_HANDLE_ADMISSION;
        }
        SemanticKernelInput::ResidentFork { root }
        | SemanticKernelInput::ResidentPreflightTransition { root } => {
            root.record_input(&mut descriptor, &mut recorder);
            descriptor.admission = if preflight {
                RESIDENT_PREFLIGHT_TRANSITION_ADMISSION
            } else {
                RESIDENT_FORK_ADMISSION
            };
        }
        SemanticKernelInput::ResidentInsertSupport {
            candidate,
            statement,
            support,
        } => {
            candidate.receipt.record_read(&mut recorder);
            descriptor.source_receipt_ptr = candidate.receipt.allocation.device_ptr_value();
            descriptor.source_receipt_index = u64::from(candidate.receipt.index);
            recorder.read(statement.allocation);
            recorder.read(support.allocation);
            descriptor.decoded_statement_ptr = statement.allocation.device_ptr_value();
            descriptor.decoded_statement_index = u64::from(statement.index);
            descriptor.decoded_support_ptr = support.allocation.device_ptr_value();
            descriptor.decoded_support_index = u64::from(support.index);
            descriptor.admission = RESIDENT_INSERT_SUPPORT_ADMISSION;
        }
        SemanticKernelInput::ResidentSeal { candidate } => {
            candidate.receipt.record_read(&mut recorder);
            descriptor.source_receipt_ptr = candidate.receipt.allocation.device_ptr_value();
            descriptor.source_receipt_index = u64::from(candidate.receipt.index);
            descriptor.admission = RESIDENT_SEAL_ADMISSION;
        }
        SemanticKernelInput::ResidentTruth { view, statement } => {
            match view {
                SemanticResidentView::Root(root) => {
                    root.record_input(&mut descriptor, &mut recorder);
                    descriptor.admission = RESIDENT_ROOT_TRUTH_ADMISSION;
                }
                SemanticResidentView::Fork(candidate) => {
                    candidate.receipt.record_read(&mut recorder);
                    descriptor.source_receipt_ptr = candidate.receipt.allocation.device_ptr_value();
                    descriptor.source_receipt_index = u64::from(candidate.receipt.index);
                    descriptor.admission = RESIDENT_FORK_TRUTH_ADMISSION;
                }
            }
            recorder.read(statement.allocation);
            descriptor.decoded_statement_ptr = statement.allocation.device_ptr_value();
            descriptor.decoded_statement_index = u64::from(statement.index);
        }
        SemanticKernelInput::ResidentConsumeTruth { truth, output } => {
            truth.receipt.record_read(&mut recorder);
            recorder.write(output.allocation);
            descriptor.source_receipt_ptr = truth.receipt.allocation.device_ptr_value();
            descriptor.source_receipt_index = u64::from(truth.receipt.index);
            descriptor.output_ptr = output.allocation.device_ptr_value();
            descriptor.output_index = u64::from(output.index);
            descriptor.admission = RESIDENT_CONSUME_TRUTH_ADMISSION;
        }
        SemanticKernelInput::ResidentMaterializeDecoded {
            input,
            statement,
            support,
        } => {
            recorder.read(input.allocation);
            recorder.write(statement.allocation);
            recorder.write(support.allocation);
            descriptor.decoded_input_ptr = input.allocation.device_ptr_value();
            descriptor.decoded_input_index = u64::from(input.index);
            descriptor.decoded_statement_ptr = statement.allocation.device_ptr_value();
            descriptor.decoded_statement_index = u64::from(statement.index);
            descriptor.decoded_support_ptr = support.allocation.device_ptr_value();
            descriptor.decoded_support_index = u64::from(support.index);
            descriptor.admission = RESIDENT_MATERIALIZE_DECODED_ADMISSION;
        }
    }
    let execute = spec.execute.clone();
    let enqueued = match unsafe {
        spec.domain.enqueue(recorder, |stream| {
            execute.clone().launch_in(
                stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (arena, descriptor),
            )
        })
    } {
        Ok(enqueued) => enqueued,
        Err(error) => {
            if matches!(
                &error,
                LaunchEnqueueError::Operation(_) | LaunchEnqueueError::OperationAndCleanup { .. }
            ) {
                *poisoned = true;
            }
            return Err(map_enqueue_error(error));
        }
    };
    if let Err(error) = enqueued.commit() {
        *poisoned = true;
        return Err(runtime_error("launch commit", error));
    }
    Ok(())
}

fn runtime_error(operation: &'static str, error: XlogError) -> SemanticHypergraphError {
    SemanticHypergraphError::Runtime {
        operation,
        detail: error.to_string(),
    }
}

fn validate_resident_bank_index(
    index: u32,
    len: usize,
    role: &'static str,
) -> Result<(), SemanticHypergraphError> {
    if (index as usize) < len {
        return Ok(());
    }
    Err(SemanticHypergraphError::InvalidInput {
        detail: format!("resident semantic {role} index {index} exceeds bank length {len}"),
    })
}

fn map_enqueue_error(error: LaunchEnqueueError<DriverError>) -> SemanticHypergraphError {
    let operation = match &error {
        LaunchEnqueueError::Preparation(_) => "launch preparation",
        LaunchEnqueueError::PreparationAndCleanup { .. } => "launch preparation and cleanup",
        LaunchEnqueueError::Operation(_) => "kernel enqueue",
        LaunchEnqueueError::OperationAndCleanup { .. } => "kernel enqueue and cleanup",
    };
    SemanticHypergraphError::Runtime {
        operation,
        detail: error.to_string(),
    }
}

fn identity_words(bytes: [u8; 32]) -> [u64; 4] {
    core::array::from_fn(|index| {
        let start = index * 8;
        u64::from_le_bytes(
            bytes[start..start + 8]
                .try_into()
                .expect("fixed identity chunk"),
        )
    })
}

fn receipt_identity(receipt: &DeviceReceipt, start: usize) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for (index, word) in receipt.words[start..start + 4].iter().enumerate() {
        bytes[index * 8..index * 8 + 8].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn inserted_support_previous_truth(
    receipt: &DeviceReceipt,
) -> Result<SemanticTruth, SemanticHypergraphError> {
    let metadata = receipt.words[32];
    let previous = (metadata & INSERT_PREVIOUS_TRUTH_MASK) >> INSERT_PREVIOUS_TRUTH_SHIFT;
    let resulting = receipt.words[2];
    let new_statement = metadata & INSERT_NEW_STATEMENT != 0;
    if receipt.words[0] != STATUS_OK
        || receipt.words[1] != OUTCOME_INSERTED
        || metadata & !(INSERT_NEW_STATEMENT | INSERT_PREVIOUS_TRUTH_MASK) != 0
        || new_statement != (previous == 0)
        || !(1..=3).contains(&resulting)
        || previous & resulting != previous
        || (previous ^ resulting).count_ones() > 1
    {
        return Err(SemanticHypergraphError::CorruptLineage {
            detail: "insertion receipt has invalid previous or resulting truth".into(),
        });
    }
    SemanticTruth::from_bits(previous)
}

fn receipt_extents(receipt: &DeviceReceipt) -> Result<SemanticExtents, SemanticHypergraphError> {
    Ok(SemanticExtents::new(
        u32::try_from(receipt.words[13]).map_err(|_| SemanticHypergraphError::CorruptLineage {
            detail: "statement extent exceeds u32".into(),
        })?,
        u32::try_from(receipt.words[14]).map_err(|_| SemanticHypergraphError::CorruptLineage {
            detail: "support extent exceeds u32".into(),
        })?,
        u32::try_from(receipt.words[15]).map_err(|_| SemanticHypergraphError::CorruptLineage {
            detail: "version extent exceeds u32".into(),
        })?,
    ))
}

fn checked_receipt_slot(
    value: u64,
    kind: SemanticHandleKind,
    capacity: u32,
) -> Result<u32, SemanticHypergraphError> {
    let slot = u32::try_from(value).map_err(|_| SemanticHypergraphError::CorruptLineage {
        detail: format!("device returned non-u32 {kind:?} slot {value}"),
    })?;
    if slot >= capacity {
        return Err(SemanticHypergraphError::CorruptLineage {
            detail: format!("device returned out-of-range {kind:?} slot {slot}"),
        });
    }
    Ok(slot)
}

fn write_view(command: &mut DeviceCommand, view: SemanticView) {
    match view {
        SemanticView::Root(handle) => {
            command.words[1] = handle.owner;
            command.words[7] = 1;
            command.words[8] = u64::from(handle.slot);
            command.words[9] = handle.generation;
        }
        SemanticView::Fork(handle) => {
            command.words[1] = handle.owner;
            command.words[7] = 2;
            command.words[8] = u64::from(handle.slot);
            command.words[9] = handle.generation;
        }
    }
}

fn decode_kind(code: u64) -> SemanticHandleKind {
    match code {
        1 => SemanticHandleKind::Root,
        2 => SemanticHandleKind::Fork,
        3 => SemanticHandleKind::Statement,
        4 => SemanticHandleKind::Support,
        5 => SemanticHandleKind::Version,
        _ => SemanticHandleKind::Root,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::cuda_graph::CapturedCudaGraph;
    use crate::CudaProviderBuilder;
    use xlog_core::MemoryBudget;

    const OP_RETIRE_ROOT: u64 = 12;

    fn kernel_error(error: impl fmt::Display) -> XlogError {
        XlogError::Kernel(error.to_string())
    }

    fn download_test_arena(
        provider: &CudaKernelProvider,
        graph: &SemanticHypergraph,
    ) -> Vec<u64> {
        let mut arena = vec![0; graph.arena_words as usize];
        provider
            .dtoh_sync_copy_into_tracked(&graph.arena, &mut arena)
            .unwrap();
        arena
    }

    fn retirement_test_graph() -> Option<SemanticHypergraph> {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
            eprintln!("Skipping: set XLOG_REQUIRE_CUDA=1 to run this real-CUDA contract");
            return None;
        }
        let provider = Arc::new(
            CudaProviderBuilder::new(0, MemoryBudget::with_limit(64 * 1024 * 1024))
                .with_stream_capacity(1)
                .build()
                .unwrap(),
        );
        let runtime = Arc::clone(provider.memory().runtime().unwrap());
        let stream_id = runtime.stream_pool().acquire().unwrap();
        let stream = runtime.stream_pool().resolve(stream_id).unwrap();
        let domain = provider
            .bind_resident_execution_domain(runtime, stream_id, stream)
            .unwrap();
        let mut graph = provider
            .allocate_semantic_hypergraph(
                &domain,
                SemanticHypergraphCapacities::try_new(8, 8, 16, 16).unwrap(),
            )
            .unwrap();
        graph.enter_transition();
        Some(graph)
    }

    fn retirement_command(
        graph: &mut SemanticHypergraph,
        operation: u64,
        fields: &[(usize, u64)],
    ) -> DeviceReceipt {
        let mut command = graph.command_for(operation);
        for &(index, value) in fields {
            command.words[index] = value;
        }
        graph.run(command, ArenaAccess::ReadWrite).unwrap()
    }

    fn retirement_root(
        graph: &mut SemanticHypergraph,
        root: SemanticRootHandle,
        protected_base: SemanticRootHandle,
    ) -> DeviceReceipt {
        retirement_command(
            graph,
            OP_RETIRE_ROOT,
            &[
                (8, root.slot as u64),
                (9, root.generation),
                (10, protected_base.slot as u64),
                (11, protected_base.generation),
            ],
        )
    }

    fn retirement_insert_root(
        graph: &mut SemanticHypergraph,
        base: SemanticRootHandle,
        statement: u64,
        support: u64,
    ) -> (SemanticRootHandle, DeviceReceipt) {
        let fork = retirement_command(
            graph,
            OP_FORK,
            &[(8, base.slot as u64), (9, base.generation)],
        );
        assert_eq!(fork.words[0], STATUS_OK);
        let edit = retirement_command(
            graph,
            OP_INSERT_SUPPORT,
            &[(9, fork.words[6]), (12, 1), (16, statement), (20, support)],
        );
        assert_eq!(edit.words[0], STATUS_OK);
        let sealed = retirement_command(graph, OP_SEAL, &[(9, fork.words[6])]);
        assert_eq!(sealed.words[0], STATUS_OK);
        (
            SemanticRootHandle::new(graph.owner, sealed.words[3] as u32, sealed.words[4]),
            edit,
        )
    }

    #[test]
    fn sealed_root_retirement_preserves_descendants_and_reuses_generations() {
        let Some(mut graph) = retirement_test_graph() else {
            return;
        };
        let provider = Arc::clone(&graph.provider);
        let empty = graph.empty_root();
        let (parent, inherited) = retirement_insert_root(&mut graph, empty, 100, 200);
        let (loser, exclusive) = retirement_insert_root(&mut graph, parent, 300, 400);
        let (descendant, _) = retirement_insert_root(&mut graph, parent, 100, 201);
        let parent_before = retirement_command(
            &mut graph,
            OP_SNAPSHOT,
            &[(7, 1), (8, parent.slot as u64), (9, parent.generation)],
        );
        let retired = retirement_root(&mut graph, loser, parent);
        assert_eq!(retired.words[0], STATUS_OK);
        assert_eq!(retired.words[3], loser.slot as u64);
        assert_eq!(retired.words[4], loser.generation + 1);
        assert_eq!(retired.words[39], graph.owner);
        let parent_after = retirement_command(
            &mut graph,
            OP_SNAPSHOT,
            &[(7, 1), (8, parent.slot as u64), (9, parent.generation)],
        );
        assert_eq!(&parent_after.words[13..20], &parent_before.words[13..20]);
        let retained = download_test_arena(&provider, &graph);
        let repeated = retirement_command(
            &mut graph,
            OP_RETIRE_ROOT,
            &[
                (8, loser.slot as u64),
                (9, loser.generation),
                (10, parent.slot as u64),
                (11, parent.generation),
            ],
        );
        assert_eq!(repeated.words[0], STATUS_STALE_GENERATION);
        assert_eq!(
            download_test_arena(&provider, &graph),
            retained
        );
        let (reused, replacement) = retirement_insert_root(&mut graph, parent, 301, 401);
        assert_eq!(reused.slot, loser.slot);
        assert_eq!(reused.generation, loser.generation + 1);
        for (slot, generation) in [(7, 8), (9, 10), (11, 12)] {
            assert_eq!(replacement.words[slot], exclusive.words[slot]);
            assert_eq!(
                replacement.words[generation],
                exclusive.words[generation] + 1
            );
        }
        // An unpublished ancestor's root slot can retire while descendant roots
        // retain its exact statement, support and version lineage.
        let candidate = retirement_command(
            &mut graph,
            OP_FORK,
            &[(8, descendant.slot as u64), (9, descendant.generation)],
        );
        assert_eq!(candidate.words[0], STATUS_OK);
        let staged = retirement_command(
            &mut graph,
            OP_INSERT_SUPPORT,
            &[(9, candidate.words[6]), (12, 2), (16, 100), (20, 202)],
        );
        assert_eq!(staged.words[0], STATUS_OK);
        assert_eq!(
            retirement_root(&mut graph, parent, empty).words[0],
            STATUS_OK
        );
        let candidate_truth = retirement_command(
            &mut graph,
            OP_TRUTH,
            &[(7, 2), (9, candidate.words[6]), (16, 100)],
        );
        assert_eq!(candidate_truth.words[0], STATUS_OK);
        assert_eq!(candidate_truth.words[2], 3);
        assert_eq!(
            retirement_command(&mut graph, OP_SEAL, &[(9, candidate.words[6])]).words[0],
            STATUS_OK
        );
        for (operation, slot, generation) in [
            (OP_INSPECT_STATEMENT, 7, 8),
            (OP_INSPECT_SUPPORT, 9, 10),
            (OP_INSPECT_VERSION, 11, 12),
        ] {
            let inspected = retirement_command(
                &mut graph,
                operation,
                &[
                    (7, 1),
                    (8, descendant.slot as u64),
                    (9, descendant.generation),
                    (10, inherited.words[slot]),
                    (11, inherited.words[generation]),
                ],
            );
            assert_eq!(inspected.words[0], STATUS_OK);
            assert_eq!(inspected.words[slot], inherited.words[slot]);
            assert_eq!(inspected.words[generation], inherited.words[generation]);
        }
        let truth = retirement_command(
            &mut graph,
            OP_TRUTH,
            &[
                (7, 1),
                (8, descendant.slot as u64),
                (9, descendant.generation),
                (16, 100),
            ],
        );
        assert_eq!(truth.words[0], STATUS_OK);
        assert_eq!(truth.words[2], 1);
        let stale = retirement_root(&mut graph, loser, empty);
        assert_eq!(stale.words[0], STATUS_STALE_GENERATION);
    }

    #[test]
    fn sealed_root_retirement_refuses_without_partial_reclamation() {
        let Some(mut graph) = retirement_test_graph() else {
            return;
        };
        let provider = Arc::clone(&graph.provider);
        let empty = graph.empty_root();
        let (target, target_edit) = retirement_insert_root(&mut graph, empty, 100, 200);
        let (other, other_edit) = retirement_insert_root(&mut graph, empty, 300, 400);
        let read_arena = |graph: &SemanticHypergraph| download_test_arena(&provider, graph);
        for (root, base) in [(empty, empty), (target, target)] {
            let before = read_arena(&graph);
            let refused = retirement_command(
                &mut graph,
                OP_RETIRE_ROOT,
                &[
                    (8, root.slot as u64),
                    (9, root.generation),
                    (10, base.slot as u64),
                    (11, base.generation),
                ],
            );
            assert_eq!(refused.words[0], STATUS_INACTIVE);
            assert_eq!(read_arena(&graph), before);
        }
        let fork = retirement_command(
            &mut graph,
            OP_FORK,
            &[(8, target.slot as u64), (9, target.generation)],
        );
        assert_eq!(fork.words[0], STATUS_OK);
        let before = read_arena(&graph);
        let refused = retirement_command(
            &mut graph,
            OP_RETIRE_ROOT,
            &[
                (8, target.slot as u64),
                (9, target.generation),
                (11, empty.generation),
            ],
        );
        assert_eq!(refused.words[0], STATUS_INACTIVE);
        assert_eq!(read_arena(&graph), before);
        assert_eq!(
            retirement_command(&mut graph, OP_DISCARD, &[(9, fork.words[6])]).words[0],
            STATUS_OK
        );
        let version_start = CONTROL_WORDS
            + 8 * ROOT_WORDS
            + CANDIDATE_WORDS
            + 8 * STATEMENT_WORDS
            + 16 * SUPPORT_WORDS;
        let clean = read_arena(&graph);
        for (offset, value, expected_status) in [
            (
                version_start + target_edit.words[11] * VERSION_WORDS + 1,
                u64::MAX,
                STATUS_GENERATION_EXHAUSTED,
            ),
            (
                version_start + other_edit.words[11] * VERSION_WORDS + 7,
                17,
                STATUS_CORRUPT_LINEAGE,
            ),
        ] {
            let mut arena = clean.clone();
            arena[offset as usize] = value;
            provider
                .htod_sync_copy_into_tracked(&arena, &mut graph.arena)
                .unwrap();
            let refused = retirement_command(
                &mut graph,
                OP_RETIRE_ROOT,
                &[
                    (8, target.slot as u64),
                    (9, target.generation),
                    (11, empty.generation),
                ],
            );
            assert_eq!(refused.words[0], expected_status);
            assert_eq!(read_arena(&graph), arena);
        }
        provider
            .htod_sync_copy_into_tracked(&clean, &mut graph.arena)
            .unwrap();
        assert_eq!(
            retirement_root(&mut graph, target, empty).words[0],
            STATUS_OK
        );
        let truth = retirement_command(
            &mut graph,
            OP_TRUTH,
            &[
                (7, 1),
                (8, other.slot as u64),
                (9, other.generation),
                (16, 300),
            ],
        );
        assert_eq!(truth.words[0], STATUS_OK);
        assert_eq!(truth.words[2], 1);
    }

    #[test]
    fn insertion_receipt_retains_previous_truth_for_actual_support_effects() {
        for (previous, resulting, new_statement) in [
            (0, 1, true),
            (0, 2, true),
            (1, 1, false),
            (2, 2, false),
            (1, 3, false),
            (2, 3, false),
            (3, 3, false),
        ] {
            let mut receipt = DeviceReceipt::default();
            receipt.words[1] = OUTCOME_INSERTED;
            receipt.words[2] = resulting;
            receipt.words[32] = u64::from(new_statement) | (previous << 1);
            assert_eq!(
                inserted_support_previous_truth(&receipt).unwrap(),
                SemanticTruth::from_bits(previous).unwrap(),
            );
        }
    }

    #[test]
    fn resident_discard_retires_refused_acquisition_and_rejects_stale_replay() {
        let Some(mut graph) = retirement_test_graph() else {
            return;
        };
        let provider = Arc::clone(&graph.provider);
        let bytes = std::mem::size_of::<SemanticResidentDecodedStatement>()
            + 2 * std::mem::size_of::<SemanticResidentDecodedSupport>()
            + std::mem::size_of::<SemanticResidentHandleRecord>()
            + 5 * std::mem::size_of::<SemanticResidentReceiptRecord>();
        let mut reservation = provider.memory().reserve_bytes(bytes as u64).unwrap();
        let mut statements = reservation
            .alloc::<SemanticResidentDecodedStatement>(1)
            .unwrap();
        let mut supports = reservation
            .alloc::<SemanticResidentDecodedSupport>(2)
            .unwrap();
        let handles = reservation
            .alloc::<SemanticResidentHandleRecord>(1)
            .unwrap();
        let receipts = reservation
            .alloc::<SemanticResidentReceiptRecord>(5)
            .unwrap();
        provider
            .htod_sync_copy_into_tracked(
                &[SemanticResidentDecodedStatement {
                    identity_words: [17; 8],
                    record: 0,
                    reconstruction: [0; 10],
                }],
                &mut statements,
            )
            .unwrap();
        provider
            .htod_sync_copy_into_tracked(
                &[1, 0].map(|polarity| SemanticResidentDecodedSupport {
                    polarity,
                    provenance_words: [polarity; 8],
                    source_words: [3; 8],
                    context_words: [4; 8],
                    scope_words: [5; 8],
                    record: polarity,
                }),
                &mut supports,
            )
            .unwrap();
        let mut setup = graph.domain.new_strict_recorder();
        setup.read_write(&statements);
        setup.read_write(&supports);
        setup.write(&handles);
        setup.write(&receipts);
        unsafe { graph.domain.enqueue(setup, |_| Ok::<(), XlogError>(())) }
            .unwrap()
            .commit()
            .unwrap();
        let slot = |index| SemanticResidentReceiptSlot::new(&receipts, index).unwrap();
        let statement = || SemanticResidentStatementBank::new(&statements, 0).unwrap();
        let support = |index| SemanticResidentSupportBank::new(&supports, index).unwrap();
        graph
            .enqueue_resident_empty_root_handle(
                SemanticResidentHandleSlot::new(&handles, 0).unwrap(),
                slot(0),
            )
            .unwrap();
        let base = || SemanticResidentRootHandle::Bank {
            allocation: &handles,
            index: 0,
        };
        let fork = graph.enqueue_resident_fork(base(), slot(1)).unwrap();
        let inserted = graph
            .enqueue_resident_insert_support(
                fork.candidate_handle(),
                statement(),
                support(0),
                slot(2),
            )
            .unwrap();
        graph
            .enqueue_resident_insert_support(
                inserted.candidate_handle(),
                statement(),
                support(1),
                slot(3),
            )
            .unwrap();
        graph.stream.synchronize().unwrap();
        let before = provider
            .dtoh_small_metadata_untracked(&receipts, receipts.len())
            .unwrap();
        assert_eq!(before[2].words[1], OUTCOME_INSERTED);
        assert_eq!(before[3].words[0], STATUS_INVALID_COMMAND);
        assert_ne!(before[3].words[41], 0);

        let discard = |graph: &SemanticHypergraph| {
            let descriptor = DeviceLaunchDescriptor {
                expected_owner: graph.owner,
                root_capacity: graph.capacities.roots as u64,
                statement_capacity: graph.capacities.statements as u64,
                support_capacity: graph.capacities.supports as u64,
                version_capacity: graph.capacities.versions as u64,
                arena_words: graph.arena_words,
                command_ptr: 0,
                command_index: 0,
                handle_ptr: 0,
                handle_index: 0,
                source_receipt_ptr: receipts.device_ptr_value(),
                source_receipt_index: 3,
                decoded_input_ptr: 0,
                decoded_input_index: 0,
                decoded_statement_ptr: 0,
                decoded_statement_index: 0,
                decoded_support_ptr: 0,
                decoded_support_index: 0,
                output_ptr: 0,
                output_index: 0,
                receipt_ptr: receipts.device_ptr_value(),
                receipt_index: 4,
                admission: 10,
                abi_generation: HYPERGRAPH_ABI_GENERATION,
            };
            let mut recorder = graph.domain.new_strict_recorder();
            recorder.read_write(&graph.arena);
            recorder.read_write(&receipts);
            unsafe {
                graph.domain.enqueue(recorder, |stream| {
                    graph.execute.clone().launch_in(
                        stream,
                        LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (1, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (graph.arena.device_ptr_value(), descriptor),
                    )
                })
            }
            .unwrap()
            .commit()
            .unwrap();
            graph.stream.synchronize().unwrap();
            provider
                .dtoh_small_metadata_untracked(&receipts, receipts.len())
                .unwrap()
        };
        let cleaned = discard(&graph);
        assert_eq!(cleaned[4].words[0], STATUS_OK);
        assert_eq!(&cleaned[4].words[40..42], &[0, 0]);
        assert_eq!(cleaned[3].words, before[3].words);
        let next = graph.enqueue_resident_fork(base(), slot(1)).unwrap();
        graph.stream.synchronize().unwrap();
        let arena = download_test_arena(&provider, &graph);
        let stale = discard(&graph);
        assert_eq!(stale[4].words[0], STATUS_STALE_GENERATION);
        assert_eq!(stale[3].words, before[3].words);
        assert_eq!(
            download_test_arena(&provider, &graph),
            arena
        );
        let reused = graph
            .enqueue_resident_insert_support(
                next.candidate_handle(),
                statement(),
                support(0),
                slot(2),
            )
            .unwrap();
        graph
            .enqueue_resident_seal(reused.candidate_handle(), slot(3))
            .unwrap();
        graph.stream.synchronize().unwrap();
        let sealed_arena = download_test_arena(&provider, &graph);
        let sealed = discard(&graph);
        assert_eq!(sealed[3].words[0], STATUS_OK);
        assert_eq!(sealed[2].words[1], OUTCOME_INSERTED);
        assert_eq!(sealed[2].words[9], before[2].words[9]);
        assert_eq!(sealed[2].words[10], before[2].words[10] + 1);
        assert_eq!(sealed[4].words[0], STATUS_INVALID_COMMAND);
        assert_eq!(
            download_test_arena(&provider, &graph),
            sealed_arena
        );
    }

    #[test]
    fn insertion_receipt_rejects_refused_duplicate_and_impossible_truth_effects() {
        for (status, outcome, resulting, metadata) in [
            (STATUS_SUPPORT_CAPACITY, OUTCOME_INSERTED, 1, 1),
            (STATUS_OK, OUTCOME_UNCHANGED, 1, 2),
            (STATUS_OK, 0, 1, 1),
            (STATUS_OK, OUTCOME_INSERTED, 0, 1),
            (STATUS_OK, OUTCOME_INSERTED, 4, 1),
            (STATUS_OK, OUTCOME_INSERTED, 3, 1),
            (STATUS_OK, OUTCOME_INSERTED, 1, 0),
            (STATUS_OK, OUTCOME_INSERTED, 1, 3),
            (STATUS_OK, OUTCOME_INSERTED, 1, 4),
            (STATUS_OK, OUTCOME_INSERTED, 1, 6),
            (STATUS_OK, OUTCOME_INSERTED, 1, 9),
        ] {
            let mut receipt = DeviceReceipt::default();
            receipt.words[0] = status;
            receipt.words[1] = outcome;
            receipt.words[2] = resulting;
            receipt.words[32] = metadata;
            assert!(
                matches!(
                    inserted_support_previous_truth(&receipt),
                    Err(SemanticHypergraphError::CorruptLineage { .. }),
                ),
                "accepted invalid insertion receipt: {status}/{outcome}/{resulting}/{metadata}",
            );
        }
    }

    fn identity_bytes_from_dwords(words: [u32; 8]) -> [u8; 32] {
        let mut bytes = [0; 32];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn root_material_codec_preserves_typed_bits_and_rejects_truncation() {
        let material = SemanticRootMaterial {
            records: numeric_records(),
            symbols: vec![],
            insertions: vec![],
            digest: [0; 32],
            extents: [0; 3],
            admission_base_digest: material_root_digest([0; 32], [0; 32], [0; 3]),
            admission_base_extents: [0; 3],
        };
        let limits = SemanticAdmissionLimits {
            max_records: 5,
            max_terms: 25,
            max_references: 2,
            max_utf8_bytes: 1024,
        };
        let encoded = material.encode().unwrap();
        assert_eq!(
            SemanticRootMaterial::decode(&encoded, limits).unwrap(),
            material
        );
        for len in 0..encoded.len() {
            assert!(SemanticRootMaterial::decode(&encoded[..len], limits).is_err());
        }
        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(SemanticRootMaterial::decode(&trailing, limits).is_err());
        let tight = SemanticAdmissionLimits {
            max_terms: 24,
            ..limits
        };
        assert!(SemanticRootMaterial::decode(&encoded, tight).is_err());
        let mut reader = SemanticMaterialReader::new(&[255; 4]);
        assert!(reader.count(1).is_err());
    }

    pub(crate) fn root_material_records() -> SemanticAdmissionRecords {
        let roles = [
            SemanticRecordRole::Statement,
            SemanticRecordRole::Provenance,
            SemanticRecordRole::Source,
            SemanticRecordRole::Context,
            SemanticRecordRole::Scope,
        ];
        let predicates = roles
            .into_iter()
            .enumerate()
            .map(|(index, role)| SemanticPredicateRecord {
                predicate: RelId(index as u32),
                role,
                schema: Schema::new(vec![("value".into(), ScalarType::U32)]),
            })
            .collect();
        let records = [0, 0, 1, 2, 3, 4, 2]
            .into_iter()
            .enumerate()
            .map(|(index, predicate)| SemanticTypedRecord {
                predicate: RelId(predicate),
                arguments: vec![SemanticArgument::U32(index as u32)],
                qualifiers: vec![],
            })
            .collect();
        let mut supports: Vec<_> = [SemanticPolarity::Pro, SemanticPolarity::Contra]
            .into_iter()
            .map(|polarity| SemanticSupportRecord {
                statement: 0,
                polarity,
                provenance: 2,
                source: 3,
                context: 4,
                scope: 5,
            })
            .collect();
        supports.push(SemanticSupportRecord {
            statement: 0,
            polarity: SemanticPolarity::Pro,
            provenance: 2,
            source: 6,
            context: 4,
            scope: 5,
        });
        SemanticAdmissionRecords {
            predicates,
            records,
            supports,
        }
    }

    pub(crate) fn root_material_limits() -> SemanticAdmissionLimits {
        SemanticAdmissionLimits {
            max_records: 32,
            max_terms: 64,
            max_references: 32,
            max_utf8_bytes: 1024,
        }
    }

    pub(crate) fn admit_material_records(records: SemanticAdmissionRecords) -> SemanticAdmission {
        admit_semantic_records(
            records,
            root_material_limits(),
            SemanticRootHandle::new(1, 0, 1),
            || {
                Ok(SemanticRootSnapshot::new(
                    SemanticRootDigest(material_root_digest([0; 32], [0; 32], [0; 3])),
                    SemanticExtents::default(),
                ))
            },
        )
        .unwrap()
    }

    pub(crate) fn reconstructed_material_statement(
        admission: &SemanticAdmission,
        original: Option<u32>,
        reconstruction: &[u32; 10],
    ) -> Result<SemanticStatementKey, SemanticHypergraphError> {
        material_statement_key(admission, original, reconstruction)
    }

    fn verify_native_material_reconstruction(
        executable: &std::path::Path,
        directory: &std::path::Path,
        run: &impl Fn(&mut std::process::Command),
    ) {
        let mut records = root_material_records();
        records.predicates[0].schema = Schema::new(vec![
            ("left".into(), ScalarType::U32),
            ("right".into(), ScalarType::U32),
        ])
        .with_sort_labels(vec!["bit".into(), "bit".into()])
        .unwrap();
        records.records[0].arguments = vec![SemanticArgument::U32(0), SemanticArgument::U32(0)];
        records.records[1].arguments = vec![SemanticArgument::U32(1), SemanticArgument::U32(1)];
        records.supports.push(records.supports[0].clone());
        let empty_digest = material_root_digest([0; 32], [0; 32], [0; 3]);
        let admission = admit_semantic_records(
            records,
            root_material_limits(),
            SemanticRootHandle::new(91, 0, 1),
            || {
                Ok(SemanticRootSnapshot::new(
                    SemanticRootDigest(empty_digest),
                    SemanticExtents::default(),
                ))
            },
        )
        .unwrap();
        let (records, symbols) = normalized_material_admission(&admission).unwrap();
        let derived_equal = [0, 0, 0, 0, 1, 0, 0, 0, 0, u32::MAX];
        let derived_new = [0, 0, 0, 1, 1, 0, 0, 0, 0, u32::MAX];
        let original = material_statement_key(&admission, Some(0), &[0; 10]).unwrap();
        assert_eq!(
            original.identity,
            material_statement_key(&admission, None, &derived_equal)
                .unwrap()
                .identity
        );
        let new_key = material_statement_key(&admission, None, &derived_new).unwrap();
        assert!(admission
            .statement_keys
            .iter()
            .flatten()
            .all(|key| key.identity != new_key.identity));
        let mut encodings = Vec::new();
        for (case, (statement, reconstruction)) in [
            (Some(0), [0; 10]),
            (None, derived_equal),
            (None, derived_new),
        ]
        .into_iter()
        .enumerate()
        {
            let key = material_statement_key(&admission, statement, &reconstruction).unwrap();
            let event = admission.support_event(3).unwrap();
            let version = material_version_digest(
                [0; 32],
                event.identity(key.identity).0,
                event.polarity.code(),
            );
            let material = SemanticRootMaterial {
                records: records.clone(),
                symbols: symbols.clone(),
                insertions: vec![SemanticRootInsertion {
                    statement,
                    reconstruction,
                    support: 3,
                    version,
                }],
                digest: material_root_digest(empty_digest, version, [1; 3]),
                extents: [1; 3],
                admission_base_digest: empty_digest,
                admission_base_extents: [0; 3],
            };
            material.validate_lineage(&admission).unwrap();
            let encoded = material.encode().unwrap();
            let decoded = SemanticRootMaterial::decode(&encoded, root_material_limits()).unwrap();
            assert_eq!(decoded, material);
            let mut old_encoding = encoded.clone();
            old_encoding[8..12].copy_from_slice(&1u32.to_le_bytes());
            assert!(SemanticRootMaterial::decode(&old_encoding, root_material_limits()).is_err());
            encodings.push(encoded);
            for reuse in [false, true] {
                let mut commands = vec![DeviceCommand::default()];
                commands[0].words[0] = OP_INITIALIZE;
                let fork = || {
                    let mut command = DeviceCommand::default();
                    command.words[0] = OP_FORK;
                    command.words[9] = 1;
                    command
                };
                if reuse {
                    commands.push(fork());
                    let mut insert = DeviceCommand::default();
                    encode_support_insertion(
                        &mut insert,
                        SemanticForkHandle::new(91, 0, 1),
                        &key,
                        &event,
                        &reconstruction,
                    );
                    commands.push(insert);
                    let mut discard = DeviceCommand::default();
                    discard.words[0] = OP_DISCARD;
                    discard.words[9] = 1;
                    commands.push(discard);
                }
                commands.push(fork());
                let insertion = &decoded.insertions[0];
                let restored_key = material_statement_key(
                    &admission,
                    insertion.statement,
                    &insertion.reconstruction,
                )
                .unwrap();
                let mut insert = DeviceCommand::default();
                encode_support_insertion(
                    &mut insert,
                    SemanticForkHandle::new(91, 0, 1 + u64::from(reuse)),
                    &restored_key,
                    &admission.support_event(insertion.support).unwrap(),
                    &insertion.reconstruction,
                );
                commands.push(insert);
                let mut seal = DeviceCommand::default();
                seal.words[0] = OP_SEAL;
                seal.words[9] = 1 + u64::from(reuse);
                commands.push(seal);
                let command_path = directory.join(format!("material-{case}-{reuse}.commands"));
                let output_path = directory.join(format!("material-{case}-{reuse}.arena"));
                let bytes: Vec<_> = commands
                    .iter()
                    .flat_map(|command| command.words.iter().flat_map(|word| word.to_ne_bytes()))
                    .collect();
                std::fs::write(&command_path, bytes).unwrap();
                run(std::process::Command::new(executable)
                    .arg("--commands")
                    .arg(&command_path)
                    .arg(&output_path));
                let bytes = std::fs::read(&output_path).unwrap();
                assert_eq!(bytes.len() % 8, 0);
                let words: Vec<_> = bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|word| u64::from_ne_bytes(*word))
                    .collect();
                let root = SemanticRootHandle::new(91, words[3] as u32, words[4]);
                let digest =
                    std::array::from_fn(|byte| (words[16 + byte / 8] >> (8 * (byte % 8))) as u8);
                let snapshot = SemanticRootSnapshot::new(
                    SemanticRootDigest(digest),
                    SemanticExtents::new(words[13] as u32, words[14] as u32, words[15] as u32),
                );
                let capacities = SemanticHypergraphCapacities::try_new(4, 4, 8, 8).unwrap();
                let arena = &words[RECEIPT_WORDS..];
                let observed =
                    material_from_arena(arena, capacities, root, snapshot, &admission).unwrap();
                assert_eq!(
                    observed, decoded,
                    "native restoration changed original/derived target or support occurrence"
                );
                let support =
                    (CONTROL_WORDS + 4 * ROOT_WORDS + CANDIDATE_WORDS + 4 * STATEMENT_WORDS)
                        as usize;
                assert_eq!(arena[support + 1], 1 + u64::from(reuse));
                std::fs::remove_file(command_path).unwrap();
                std::fs::remove_file(output_path).unwrap();
            }
            let mut changed = material.clone();
            changed.insertions[0].support = 2;
            assert!(changed.validate_lineage(&admission).is_err());
            if statement.is_none() {
                for (field, value) in [(0, 1), (1, u32::MAX), (1, 2), (5, 1), (9, 2)] {
                    let mut changed = material.clone();
                    changed.insertions[0].reconstruction[field] = value;
                    assert!(changed.validate_lineage(&admission).is_err());
                }
                changed = material.clone();
                changed.insertions[0].statement = Some(0);
                assert!(changed.validate_lineage(&admission).is_err());
                assert!(changed.encode().is_err());
            }
        }
        assert_ne!(
            encodings[0], encodings[1],
            "derived bytes acquired original record zero"
        );
        assert_ne!(Sha256::digest(&encodings[0]), Sha256::digest(&encodings[1]));
    }

    #[test]
    fn native_truth_receipt_producer_preserves_view_and_head_fields() {
        use std::process::Command;
        use std::time::{Duration, Instant};
        let mut nonce = [0; 8];
        getrandom::fill(&mut nonce).unwrap();
        let directory = std::env::temp_dir().join(format!(
            "xlog-native-truth-{}-{}",
            std::process::id(),
            u64::from_le_bytes(nonce)
        ));
        std::fs::create_dir(&directory).unwrap();
        let deadline = Instant::now() + Duration::from_secs(120);
        let run = |command: &mut Command| {
            let mut child = command
                .current_dir(&directory)
                .spawn()
                .expect("native truth regression command must start; no silent compiler skip");
            loop {
                if let Some(status) = child
                    .try_wait()
                    .expect("native truth regression process status")
                {
                    assert!(
                        status.success(),
                        "native truth regression failed; output retained in {}",
                        directory.display()
                    );
                    break;
                }
                if Instant::now() >= deadline {
                    child
                        .kill()
                        .expect("stop timed-out native truth regression");
                    child
                        .wait()
                        .expect("reap timed-out native truth regression");
                    panic!(
                        "native truth regression exceeded two minutes; output retained in {}",
                        directory.display()
                    );
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        };
        for name in ["semantic_truth_receipt", "semantic_feedback_lineage"] {
            let executable = directory.join(name);
            let source =
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/{name}.cpp"));
            let mut compiler =
                Command::new(std::env::var_os("CXX").unwrap_or_else(|| "c++".into()));
            compiler.args([
                "-std=c++20",
                "-O0",
                "-Wall",
                "-Wextra",
                "-I",
                env!("OUT_DIR"),
            ]);
            if cfg!(feature = "semantic-policy") {
                compiler.arg("-DXLOG_SEMANTIC_POLICY");
            }
            run(compiler.arg(source).arg("-o").arg(&executable));
            run(&mut Command::new(&executable));
            if name == "semantic_truth_receipt" {
                verify_native_material_reconstruction(&executable, &directory, &run);
            } else {
                crate::semantic_transition::task_binding_tests::verify_native_decoded_reconstruction(
                    &executable, &directory, &run,
                );
            }
            std::fs::remove_file(&executable).unwrap();
        }
        std::fs::remove_dir(&directory).unwrap();
    }

    #[test]
    fn task_selection_identity_retains_exact_admitted_occurrences() {
        use crate::semantic_transition::{
            task_binding_tests::{arithmetic_observation, task_spec},
            TaskEvaluationBinding,
        };
        let mut records = root_material_records();
        records.predicates[0].schema = Schema::new(vec![
            ("left".into(), ScalarType::U32),
            ("right".into(), ScalarType::U32),
        ])
        .with_sort_labels(vec!["bit".into(), "bit".into()])
        .unwrap();
        records.records[0].arguments = vec![SemanticArgument::U32(1), SemanticArgument::U32(1)];
        records.records[1].arguments = vec![SemanticArgument::U32(0), SemanticArgument::U32(1)];
        records.records.push(records.records[0].clone());
        records.supports.push(records.supports[0].clone());
        let admission = admit_material_records(records);
        let bind = |statement_records, allowed_support_records: Vec<u32>| {
            let binding = TaskEvaluationBinding::bind(
                &admission,
                task_spec(statement_records, allowed_support_records.clone()),
                arithmetic_observation(),
            )
            .unwrap();
            assert_eq!(binding.spec().statement_records, statement_records);
            assert_eq!(
                binding.spec().allowed_support_records,
                allowed_support_records
            );
            binding
        };
        let first = bind([0, 1, 1], vec![0]).words(1);
        let equal_occurrence = bind([0, 1, 1], vec![3]).words(1);
        assert_ne!(
            first[6..],
            equal_occurrence[6..],
            "resident task bank collapsed equal support occurrences"
        );
        assert_eq!(first[22..25], [0, 1, 1]);
        assert_eq!(first[34], 0);
        assert_eq!(equal_occurrence[34], 3);
        let both = bind([0, 1, 1], vec![0, 3]).words(1);
        assert_eq!(both[21], 2);
        assert_eq!([both[34], both[39]], [0, 3]);
        for (left, right) in [
            (bind([0, 1, 1], vec![]), bind([7, 1, 1], vec![])),
            (bind([0, 1, 1], vec![0]), bind([0, 1, 1], vec![3])),
            (bind([0, 1, 1], vec![0, 3]), bind([0, 1, 1], vec![3, 0])),
            (bind([0, 1, 1], vec![0, 0]), bind([0, 1, 1], vec![0])),
        ] {
            // Same semantic query/support operands are not the same original
            // admitted source selections for restoration and read provenance.
            assert_ne!(
                left.identity(),
                right.identity(),
                "task identity lost original record selections"
            );
        }
    }

    #[test]
    fn root_material_lineage_binds_cross_statement_order_and_original_support() {
        let admission = admit_semantic_records(
            root_material_records(),
            root_material_limits(),
            SemanticRootHandle::new(1, 0, 1),
            || {
                Ok(SemanticRootSnapshot::new(
                    SemanticRootDigest(material_root_digest([0; 32], [0; 32], [0; 3])),
                    SemanticExtents::default(),
                ))
            },
        )
        .unwrap();
        let (records, symbols) = normalized_material_admission(&admission).unwrap();
        let mut material = SemanticRootMaterial {
            records,
            symbols,
            insertions: vec![],
            digest: material_root_digest([0; 32], [0; 32], [0; 3]),
            extents: [0; 3],
            admission_base_digest: material_root_digest([0; 32], [0; 32], [0; 3]),
            admission_base_extents: [0; 3],
        };
        for (ordinal, (statement, support)) in [(1, 0), (0, 1)].into_iter().enumerate() {
            let key = admission.statement_key(statement).unwrap();
            let event = admission.support_event(support).unwrap();
            let version = material_version_digest(
                [0; 32],
                event.identity(key.identity).0,
                event.polarity.code(),
            );
            material.insertions.push(SemanticRootInsertion {
                statement: Some(statement),
                reconstruction: [0; 10],
                support,
                version,
            });
            material.extents = [(ordinal + 1) as u32; 3];
            material.digest = material_root_digest(material.digest, version, material.extents);
        }
        material.validate_lineage(&admission).unwrap();
        // The immutable admission binding may name a proper prefix of the
        // acquired root, even after that original physical root is retired.
        let first = &material.insertions[0];
        material.admission_base_digest = material_root_digest(
            material_root_digest([0; 32], [0; 32], [0; 3]),
            first.version,
            [1; 3],
        );
        material.admission_base_extents = [1; 3];
        material.validate_lineage(&admission).unwrap();
        let mut wrong_base = material.clone();
        wrong_base.admission_base_digest[0] ^= 1;
        assert!(wrong_base.validate_lineage(&admission).is_err());
        let decoded =
            SemanticRootMaterial::decode(&material.encode().unwrap(), root_material_limits())
                .unwrap();
        assert_eq!(decoded, material);
        assert_eq!(decoded.records.supports[0].statement, 0);
        assert_eq!(decoded.insertions[0].statement, Some(1));
        let mut reversed = material.clone();
        reversed.insertions.reverse();
        assert!(reversed.validate_lineage(&admission).is_err());
        let mut duplicate = material.clone();
        duplicate.insertions.push(duplicate.insertions[0].clone());
        assert!(duplicate.validate_lineage(&admission).is_err());
        let mut altered = material;
        altered.insertions[0].version[0] ^= 1;
        assert!(altered.validate_lineage(&admission).is_err());
    }

    #[test]
    fn root_material_observation_roots_preserve_actual_aliases_and_both_polarities() {
        let mut declared = root_material_records();
        declared.records.push(declared.records[0].clone());
        let mut other_query = declared.records[1].clone();
        other_query.arguments = vec![SemanticArgument::U32(8)];
        declared.records.push(other_query);
        let admission = admit_material_records(declared);
        let (records, symbols) = normalized_material_admission(&admission).unwrap();
        let empty = material_root_digest([0; 32], [0; 32], [0; 3]);
        let mut material = SemanticRootMaterial {
            records,
            symbols,
            insertions: vec![],
            digest: empty,
            extents: [0; 3],
            admission_base_digest: empty,
            admission_base_extents: [0; 3],
        };
        let mut previous = [0; 32];
        for (statement, support, extents, truth) in [
            (7, 0, [1, 1, 1], 1),
            (7, 1, [1, 2, 2], 3),
            (1, 2, [2, 3, 3], 1),
        ] {
            let key = admission.statement_key(statement).unwrap();
            let event = admission.support_event(support).unwrap();
            let version = material_version_digest(
                if statement == 7 { previous } else { [0; 32] },
                event.identity(key.identity).0,
                truth,
            );
            material.insertions.push(SemanticRootInsertion {
                statement: Some(statement),
                reconstruction: [0; 10],
                support,
                version,
            });
            material.extents = extents;
            material.digest = material_root_digest(material.digest, version, extents);
            previous = version;
            if support == 0 {
                material.admission_base_digest = material.digest;
                material.admission_base_extents = extents;
            }
        }
        let material =
            SemanticRootMaterial::decode(&material.encode().unwrap(), root_material_limits())
                .unwrap();
        let roots = material
            .task_observation_roots(&admission, [0, 8, 8])
            .unwrap();
        assert_eq!(roots.query_records, [0, 8, 8]);
        assert_eq!(roots.root_digest.as_bytes(), &material.digest);
        assert_eq!(roots.root_extents, [2, 3, 3]);
        assert_eq!(roots.contributors, [(0, Some(7), 0), (0, Some(7), 1)]);
        let reverse = material
            .task_observation_roots(&admission, [8, 0, 8])
            .unwrap();
        assert_eq!(reverse.contributors, [(1, Some(7), 0), (1, Some(7), 1)]);
        let mut corrupt = material.clone();
        corrupt.insertions[0].version[0] ^= 1;
        assert!(corrupt
            .task_observation_roots(&admission, [0, 8, 8])
            .is_err());
        assert!(material
            .task_observation_roots(&admission, [0, 2, 2])
            .is_err());
        let mut derived = material;
        derived.insertions[0].statement = None;
        derived.insertions[0].reconstruction = [0; 10];
        derived.insertions[0].reconstruction[9] = u32::MAX;
        let roots = derived
            .task_observation_roots(&admission, [0, 8, 8])
            .unwrap();
        assert_eq!(roots.contributors, [(0, None, 0), (0, Some(7), 1)]);
    }

    #[test]
    fn root_material_symbols_retain_occurrence_order_and_enforce_byte_budget() {
        let mut records = root_material_records();
        records.predicates[0].schema = Schema::new(vec![("value".into(), ScalarType::Symbol)]);
        records.records[0].arguments =
            vec![SemanticArgument::Symbol(symbol::intern("root-material-a"))];
        records.records[1].arguments =
            vec![SemanticArgument::Symbol(symbol::intern("root-material-b"))];
        let admission = admit_semantic_records(
            records,
            root_material_limits(),
            SemanticRootHandle::new(1, 0, 1),
            || {
                Ok(SemanticRootSnapshot::new(
                    SemanticRootDigest([0; 32]),
                    SemanticExtents::default(),
                ))
            },
        )
        .unwrap();
        let (records, symbols) = normalized_material_admission(&admission).unwrap();
        let mut material = SemanticRootMaterial {
            records,
            symbols,
            insertions: vec![],
            digest: [0; 32],
            extents: [0; 3],
            admission_base_digest: material_root_digest([0; 32], [0; 32], [0; 3]),
            admission_base_extents: [0; 3],
        };
        assert_eq!(
            material.records.records[0].arguments,
            [SemanticArgument::Symbol(0)]
        );
        assert_eq!(
            material.records.records[1].arguments,
            [SemanticArgument::Symbol(1)]
        );
        let bytes = material.encode().unwrap();
        assert_eq!(
            SemanticRootMaterial::decode(&bytes, root_material_limits()).unwrap(),
            material
        );
        let records = material.admission_records().unwrap();
        assert_eq!(records, admission.records);
        assert!(SemanticRootMaterial::decode(
            &bytes,
            SemanticAdmissionLimits {
                max_utf8_bytes: 1,
                ..root_material_limits()
            }
        )
        .is_err());
        material.records.records[1].arguments = vec![SemanticArgument::Symbol(0)];
        assert!(material.encode().is_err());
    }

    #[test]
    fn root_material_capture_uses_ordinals_and_rejects_invalid_reachable_generations() {
        let root = SemanticRootHandle::new(17, 1, 4);
        let mut records = root_material_records();
        records.records.push(records.records[0].clone());
        records.supports.push(records.supports[1].clone());
        let admission = admit_semantic_records(
            records,
            root_material_limits(),
            SemanticRootHandle::new(17, 0, 1),
            || {
                Ok(SemanticRootSnapshot::new(
                    SemanticRootDigest(material_root_digest([0; 32], [0; 32], [0; 3])),
                    SemanticExtents::default(),
                ))
            },
        )
        .unwrap();
        let capacities = SemanticHypergraphCapacities::try_new(2, 3, 3, 3).unwrap();
        // Native ABI input to the same extraction function used after the cold
        // device copy. This checks the CPU decoder, not CUDA execution.
        let mut arena = vec![0u64; checked_arena_words(capacities).unwrap() as usize];
        arena[1] = root.owner;
        let statements = (CONTROL_WORDS + 2 * ROOT_WORDS + CANDIDATE_WORDS) as usize;
        let supports = statements + 3 * STATEMENT_WORDS as usize;
        let versions = supports + 3 * SUPPORT_WORDS as usize;
        let heads = versions + 3 * VERSION_WORDS as usize + 3;
        let root_offset = (CONTROL_WORDS + ROOT_WORDS) as usize;
        arena[root_offset] = 3;
        arena[root_offset + 1] = root.generation;
        arena[root_offset + 3..root_offset + 6].copy_from_slice(&[2, 2, 2]);
        // Unreachable records and a pending candidate are deliberately not valid
        // admissions. They must never enter the selected root's material.
        arena[CONTROL_WORDS as usize] = 2;
        arena[(CONTROL_WORDS + 2 * ROOT_WORDS) as usize] = 1;
        arena[versions + VERSION_WORDS as usize] = 99;
        let mut digest = material_root_digest([0; 32], [0; 32], [0; 3]);
        for (index, (statement_index, support_index, statement_slot, slot)) in
            [(1u32, 0u32, 1usize, 2usize), (7, 3, 0, 0)]
                .into_iter()
                .enumerate()
        {
            let key = admission.statement_key(statement_index).unwrap();
            let event = admission.support_event(support_index).unwrap();
            let support_digest = event.identity(key.identity).0;
            let version_digest =
                material_version_digest([0; 32], support_digest, event.polarity.code());
            let statement = statements + statement_slot * STATEMENT_WORDS as usize;
            arena[statement] = 3;
            arena[statement + 1] = 8;
            arena[statement + 3..statement + 7].copy_from_slice(&identity_words(key.identity.0));
            let support = supports + slot * SUPPORT_WORDS as usize;
            arena[support..support + 6].copy_from_slice(&[
                3,
                9,
                0,
                statement_slot as u64,
                8,
                event.polarity.code(),
            ]);
            arena[support + 6..support + 10].copy_from_slice(&identity_words(support_digest));
            arena[support + 10] = u64::from(statement_index);
            arena[support + 11] = u64::from(support_index);
            let version = versions + slot * VERSION_WORDS as usize;
            arena[version..version + 9].copy_from_slice(&[
                3,
                10,
                0,
                statement_slot as u64,
                8,
                slot as u64,
                9,
                0,
                event.polarity.code(),
            ]);
            arena[version + 9..version + 13].copy_from_slice(&identity_words(version_digest));
            arena[version + 13] = index as u64 + 1;
            arena[heads + statement_slot] = slot as u64 + 1;
            digest = material_root_digest(digest, version_digest, [(index + 1) as u32; 3]);
        }
        arena[root_offset + 6..root_offset + 10].copy_from_slice(&identity_words(digest));
        let snapshot =
            SemanticRootSnapshot::new(SemanticRootDigest(digest), SemanticExtents::new(2, 2, 2));
        let material = material_from_arena(&arena, capacities, root, snapshot, &admission).unwrap();
        assert_eq!(
            material
                .insertions
                .iter()
                .map(|insertion| insertion.statement)
                .collect::<Vec<_>>(),
            [Some(1), Some(7)]
        );
        assert_eq!(
            material
                .insertions
                .iter()
                .map(|insertion| insertion.support)
                .collect::<Vec<_>>(),
            [0, 3]
        );
        assert_eq!(material.extents, [2; 3]);
        for (offset, value) in [
            (versions + 13, 0),
            (versions + 13, 1),
            (versions + 6, 8),
            (versions + 7, 1),
            (heads, 4),
            (root_offset + 1, 3),
            (supports + 10, 1),
            (supports + 10, u64::MAX),
            (supports + 11, 0),
            (supports + 11, u64::MAX),
        ] {
            let mut damaged = arena.clone();
            damaged[offset] = value;
            assert!(material_from_arena(&damaged, capacities, root, snapshot, &admission).is_err());
        }
    }

    #[test]
    fn root_material_codec_rejects_noncanonical_boolean_and_unknown_scalar() {
        let mut records = root_material_records();
        records.predicates[0].schema = Schema::new(vec![("value".into(), ScalarType::Bool)]);
        records.records[0].arguments = vec![SemanticArgument::Bool(false)];
        records.records[1].arguments = vec![SemanticArgument::Bool(true)];
        let material = SemanticRootMaterial {
            records,
            symbols: vec![],
            insertions: vec![],
            digest: [0; 32],
            extents: [0; 3],
            admission_base_digest: material_root_digest([0; 32], [0; 32], [0; 3]),
            admission_base_extents: [0; 3],
        };
        let encoded = material.encode().unwrap();
        let mut reader = SemanticMaterialReader::new(&encoded);
        reader.take(12).unwrap();
        for _ in 0..reader.u32().unwrap() {
            reader.take(5).unwrap();
            for _ in 0..reader.u32().unwrap() {
                reader.bytes().unwrap();
                reader.u8().unwrap();
                reader.bytes().unwrap();
            }
            let keys = reader.u32().unwrap() as usize;
            reader.take(keys * 4).unwrap();
        }
        assert_eq!(reader.u32().unwrap(), material.records.records.len() as u32);
        reader.u32().unwrap();
        assert_eq!(reader.u32().unwrap(), 1);
        let scalar_offset = encoded.len() - reader.remaining.len();
        assert_eq!(reader.u8().unwrap(), ScalarType::Bool.to_code());
        let mut invalid = encoded.clone();
        invalid[scalar_offset + 1] = 2;
        assert!(SemanticRootMaterial::decode(&invalid, root_material_limits()).is_err());
        invalid = encoded;
        invalid[scalar_offset] = u8::MAX;
        assert!(SemanticRootMaterial::decode(&invalid, root_material_limits()).is_err());
    }

    #[test]
    #[ignore = "requires explicitly authorized CUDA execution"]
    fn root_material_native_restore_preserves_history_after_retirement_and_slot_reuse() {
        let mut graph = retirement_test_graph().expect("XLOG_REQUIRE_CUDA=1 is required");
        graph.device_controlled = false;
        graph = graph
            .admit_initial_records(root_material_records(), &[0], root_material_limits())
            .unwrap();
        graph.enter_transition();
        let insert = |graph: &mut SemanticHypergraph,
                      base: SemanticRootHandle,
                      statement: u32,
                      support: u32| {
            let admission = graph.admission.as_ref().unwrap();
            let key = admission.statement_key(statement).unwrap();
            let event = admission.support_event(support).unwrap();
            let fork = retirement_command(
                graph,
                OP_FORK,
                &[(8, base.slot as u64), (9, base.generation)],
            );
            assert_eq!(fork.words[0], STATUS_OK);
            let mut command = graph.command_for(OP_INSERT_SUPPORT);
            command.words[9] = fork.words[6];
            command.words[12] = event.polarity.code();
            command.words[13] = u64::from(key.record);
            command.words[14] = u64::from(event.record);
            command.words[16..20].copy_from_slice(&identity_words(key.identity.0));
            command.words[20..24].copy_from_slice(&identity_words(event.identity(key.identity).0));
            assert_eq!(
                graph.run(command, ArenaAccess::ReadWrite).unwrap().words[0],
                STATUS_OK
            );
            let sealed = retirement_command(graph, OP_SEAL, &[(9, fork.words[6])]);
            assert_eq!(sealed.words[0], STATUS_OK);
            SemanticRootHandle::new(graph.owner, sealed.words[3] as u32, sealed.words[4])
        };
        let original_base = graph.admission().unwrap().base();
        let original_base_snapshot = *graph.admission().unwrap().base_snapshot();
        let unrelated = insert(&mut graph, original_base, 1, 1);
        let first = insert(&mut graph, original_base, 1, 0);
        assert_eq!(
            retirement_root(&mut graph, unrelated, first).words[0],
            STATUS_OK
        );
        let second = insert(&mut graph, first, 0, 1);
        let root = insert(&mut graph, second, 1, 1);
        assert_eq!(
            retirement_root(&mut graph, original_base, root).words[0],
            STATUS_OK
        );
        let material = graph.export_transition_root(root).unwrap();
        assert_eq!(
            material
                .insertions
                .iter()
                .map(|insertion| (insertion.statement, insertion.support))
                .collect::<Vec<_>>(),
            [(Some(0), 0), (Some(1), 0), (Some(0), 1), (Some(1), 1)]
        );
        assert_eq!(
            material.admission_base_digest,
            original_base_snapshot.digest.0
        );
        assert_eq!(material.admission_base_extents, [1; 3]);
        let pending = retirement_command(
            &mut graph,
            OP_FORK,
            &[(8, root.slot as u64), (9, root.generation)],
        );
        assert_eq!(pending.words[0], STATUS_OK);
        let key = graph.admission().unwrap().statement_key(0).unwrap();
        let event = graph.admission().unwrap().support_event(2).unwrap();
        let mut pending_insert = graph.command_for(OP_INSERT_SUPPORT);
        pending_insert.words[9] = pending.words[6];
        pending_insert.words[12] = event.polarity.code();
        pending_insert.words[13] = u64::from(key.record);
        pending_insert.words[14] = u64::from(event.record);
        pending_insert.words[16..20].copy_from_slice(&identity_words(key.identity.0));
        pending_insert.words[20..24]
            .copy_from_slice(&identity_words(event.identity(key.identity).0));
        let pending_inserted = graph.run(pending_insert, ArenaAccess::ReadWrite).unwrap();
        assert_eq!(pending_inserted.words[0], STATUS_OK);
        assert_eq!(pending_inserted.words[1], OUTCOME_INSERTED);
        assert_eq!(graph.export_transition_root(root).unwrap(), material);
        let decoded =
            SemanticRootMaterial::decode(&material.encode().unwrap(), root_material_limits())
                .unwrap();
        let mut too_small = graph
            .provider
            .allocate_semantic_hypergraph(
                &graph.domain,
                SemanticHypergraphCapacities::try_new(2, 8, 16, 16).unwrap(),
            )
            .unwrap();
        too_small
            .admit_records(
                too_small.empty_root,
                decoded.admission_records().unwrap(),
                root_material_limits(),
            )
            .unwrap();
        let before = too_small.execution_stats();
        assert!(too_small.restore_root(&decoded).is_err());
        assert_eq!(too_small.execution_stats(), before);
        let mut restored = graph
            .provider
            .allocate_semantic_hypergraph(&graph.domain, graph.capacities)
            .unwrap();
        restored
            .admit_records(
                restored.empty_root,
                decoded.admission_records().unwrap(),
                root_material_limits(),
            )
            .unwrap();
        let before = restored.execution_stats();
        let mut damaged = decoded.clone();
        damaged.digest[0] ^= 1;
        assert!(restored.restore_root(&damaged).is_err());
        assert_eq!(restored.execution_stats(), before);
        let fresh = restored.restore_root(&decoded).unwrap();
        assert_ne!(fresh.owner, root.owner);
        assert_ne!(restored.admission().unwrap().base(), fresh);
        assert_eq!(
            *restored.admission().unwrap().base_snapshot(),
            original_base_snapshot
        );
        assert_eq!(restored.export_transition_root(fresh).unwrap(), decoded);
        assert!(restored.restore_root(&decoded).is_err());
        assert!(matches!(
            restored.export_transition_root(root),
            Err(SemanticHypergraphError::ForeignHandle { .. })
        ));
        assert!(graph.export_transition_root(unrelated).is_err());
    }

    fn numeric_records() -> SemanticAdmissionRecords {
        let types = [
            ScalarType::U32,
            ScalarType::U64,
            ScalarType::I32,
            ScalarType::I64,
            ScalarType::F32,
            ScalarType::F64,
            ScalarType::Bool,
        ];
        SemanticAdmissionRecords {
            predicates: vec![
                SemanticPredicateRecord {
                    predicate: RelId(7),
                    role: SemanticRecordRole::Statement,
                    schema: Schema::new(
                        types
                            .into_iter()
                            .enumerate()
                            .map(|(i, ty)| (format!("column_{i}"), ty))
                            .collect(),
                    ),
                },
                SemanticPredicateRecord {
                    predicate: RelId(8),
                    role: SemanticRecordRole::Qualifier,
                    schema: Schema::new(vec![("qualifier".into(), ScalarType::U64)]),
                },
            ],
            records: vec![
                SemanticTypedRecord {
                    predicate: RelId(7),
                    arguments: vec![
                        SemanticArgument::U32(u32::MAX),
                        SemanticArgument::U64(u64::MAX),
                        SemanticArgument::I32(i32::MIN),
                        SemanticArgument::I64(i64::MIN),
                        SemanticArgument::F32Bits(0x7fc0_0001),
                        SemanticArgument::F64Bits(0x8000_0000_0000_0000),
                        SemanticArgument::Bool(false),
                    ],
                    qualifiers: vec![1, 2],
                },
                SemanticTypedRecord {
                    predicate: RelId(8),
                    arguments: vec![SemanticArgument::U64(1)],
                    qualifiers: vec![],
                },
                SemanticTypedRecord {
                    predicate: RelId(8),
                    arguments: vec![SemanticArgument::U64(2)],
                    qualifiers: vec![],
                },
            ],
            supports: vec![],
        }
    }

    pub(crate) fn admit_numeric(records: SemanticAdmissionRecords) -> SemanticAdmission {
        admit_semantic_records(
            records,
            SemanticAdmissionLimits {
                max_records: 5,
                max_terms: 25,
                max_references: 2,
                max_utf8_bytes: 1024,
            },
            SemanticRootHandle::new(1, 0, 1),
            || {
                Ok(SemanticRootSnapshot::new(
                    SemanticRootDigest([0; 32]),
                    SemanticExtents::default(),
                ))
            },
        )
        .unwrap()
    }

    #[test]
    fn typed_admission_identity_preserves_numeric_bits_schema_and_qualifier_order() {
        let records = numeric_records();
        let baseline = admit_numeric(records.clone());
        let original = baseline.statement_key(0).unwrap().identity();
        for (column, value) in [
            (0, SemanticArgument::U32(0)),
            (1, SemanticArgument::U64(0)),
            (2, SemanticArgument::I32(-1)),
            (3, SemanticArgument::I64(-1)),
            (4, SemanticArgument::F32Bits(0x7fc0_0002)), // retain NaN payload
            (5, SemanticArgument::F64Bits(0)),           // distinguish negative zero
            (6, SemanticArgument::Bool(true)),
        ] {
            let mut changed = records.clone();
            changed.records[0].arguments[column] = value;
            assert_ne!(
                admit_numeric(changed).statement_key(0).unwrap().identity(),
                original
            );
        }
        let mut changed = records.clone();
        changed.records[0].qualifiers.reverse();
        assert_ne!(
            admit_numeric(changed).statement_key(0).unwrap().identity(),
            original
        );
        let mut changed = records.clone();
        changed.records[0].qualifiers.clear();
        assert_ne!(
            admit_numeric(changed).statement_key(0).unwrap().identity(),
            original
        );
        let mut changed = records.clone();
        changed.predicates[0].schema.key_columns.reverse();
        let changed = admit_numeric(changed);
        assert_ne!(changed.schema_generation(), baseline.schema_generation());
        assert_ne!(changed.statement_key(0).unwrap().identity(), original);
        let mut changed = records.clone();
        changed.predicates[0].schema = changed.predicates[0]
            .schema
            .clone()
            .with_sort_labels((0..7).map(|i| format!("sort_{i}")).collect())
            .unwrap();
        assert_ne!(
            admit_numeric(changed).statement_key(0).unwrap().identity(),
            original
        );
        let mut changed = records.clone();
        changed.predicates[0].schema.columns[0].0 = "renamed column".into();
        assert_ne!(
            admit_numeric(changed).statement_key(0).unwrap().identity(),
            original
        );
        assert_eq!(
            baseline.records(),
            &records,
            "the accepted schema and typed bytes are retained"
        );
    }

    #[test]
    fn typed_admission_checks_complete_input_before_observing_base() {
        let mut records = numeric_records();
        records.records[0].arguments[0] = SemanticArgument::I32(0);
        let result = admit_semantic_records(
            records,
            SemanticAdmissionLimits {
                max_records: 5,
                max_terms: 25,
                max_references: 2,
                max_utf8_bytes: 1024,
            },
            SemanticRootHandle::new(1, 0, 1),
            || panic!("invalid input must not observe CUDA"),
        );
        assert!(matches!(
            result,
            Err(SemanticHypergraphError::InvalidInput { .. })
        ));
    }

    #[test]
    fn typed_admission_identity_rebinds_the_base_without_reencoding_records() {
        let admission = admit_numeric(numeric_records());
        let identity_for = |snapshot| {
            derive_admission_identity(
                &admission.records,
                admission.schema_generation,
                &admission.encoded_records,
                &admission.statement_keys,
                &admission.support_events,
                snapshot,
            )
        };
        assert_eq!(identity_for(&admission.base_snapshot), admission.identity);
        let populated =
            SemanticRootSnapshot::new(SemanticRootDigest([17; 32]), SemanticExtents::new(1, 2, 2));
        assert_ne!(identity_for(&populated), admission.identity);
        let changed_extents =
            SemanticRootSnapshot::new(*populated.digest(), SemanticExtents::new(1, 2, 3));
        assert_ne!(identity_for(&populated), identity_for(&changed_extents));
        let changed_digest =
            SemanticRootSnapshot::new(SemanticRootDigest([18; 32]), populated.extents());
        assert_ne!(identity_for(&populated), identity_for(&changed_digest));
        assert_eq!(identity_for(&admission.base_snapshot), admission.identity);
    }

    #[test]
    fn host_device_divergence_statuses_are_integrity_failures() {
        for status in [
            STATUS_FOREIGN_OWNER,
            STATUS_SLOT_OUT_OF_RANGE,
            STATUS_STALE_GENERATION,
            STATUS_INACTIVE,
            STATUS_CORRUPT_LINEAGE,
            STATUS_INVALID_COMMAND,
            STATUS_ARENA_MISMATCH,
            u64::MAX,
        ] {
            assert!(device_status_is_integrity_failure(status));
        }
        for status in [
            STATUS_OK,
            STATUS_NOT_REACHABLE,
            STATUS_STATEMENT_CAPACITY,
            STATUS_SUPPORT_CAPACITY,
            STATUS_VERSION_CAPACITY,
            STATUS_ROOT_CAPACITY,
            STATUS_GENERATION_EXHAUSTED,
        ] {
            assert!(!device_status_is_integrity_failure(status));
        }
    }

    #[test]
    fn reconciliation_failure_poisoning_is_sticky() {
        let mut poisoned = false;
        let failure = Err::<(), _>(SemanticHypergraphError::CorruptLineage {
            detail: "test reconciliation failure".into(),
        });
        assert!(poison_after_reconciliation_error(&mut poisoned, failure).is_err());
        assert!(poisoned);
        assert!(poison_after_reconciliation_error(&mut poisoned, Ok(())).is_ok());
        assert!(poisoned);
    }

    #[test]
    fn resident_refusal_retires_only_its_acquired_candidate() {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
            eprintln!("Skipping: set XLOG_REQUIRE_CUDA=1 to run this real-CUDA contract");
            return;
        }
        let provider = CudaProviderBuilder::new(0, MemoryBudget::with_limit(64 * 1024 * 1024))
            .with_stream_capacity(1)
            .build()
            .unwrap();
        let provider = Arc::new(provider);
        let runtime = Arc::clone(provider.memory().runtime().unwrap());
        let stream_id = runtime.stream_pool().acquire().unwrap();
        let stream = runtime.stream_pool().resolve(stream_id).unwrap();
        let domain = provider
            .bind_resident_execution_domain(runtime, stream_id, Arc::clone(&stream))
            .unwrap();
        let mut graph = provider
            .allocate_semantic_hypergraph(
                &domain,
                SemanticHypergraphCapacities::try_new(2, 1, 1, 1).unwrap(),
            )
            .unwrap();
        let bytes = std::mem::size_of::<SemanticResidentDecodedStatement>()
            + 2 * std::mem::size_of::<SemanticResidentDecodedSupport>()
            + std::mem::size_of::<SemanticResidentHandleRecord>()
            + 11 * std::mem::size_of::<SemanticResidentReceiptRecord>();
        let mut reservation = provider.memory().reserve_bytes(bytes as u64).unwrap();
        let mut statements = reservation
            .alloc::<SemanticResidentDecodedStatement>(1)
            .unwrap();
        let mut supports = reservation
            .alloc::<SemanticResidentDecodedSupport>(2)
            .unwrap();
        let handles = reservation
            .alloc::<SemanticResidentHandleRecord>(1)
            .unwrap();
        let receipts = reservation
            .alloc::<SemanticResidentReceiptRecord>(11)
            .unwrap();
        assert_eq!(reservation.remaining_bytes(), 0);
        provider
            .htod_sync_copy_into_tracked(
                &[SemanticResidentDecodedStatement {
                    identity_words: [17; 8],
                    record: 0,
                    reconstruction: [0; 10],
                }],
                &mut statements,
            )
            .unwrap();
        let events = [1, 2].map(|value| SemanticResidentDecodedSupport {
            polarity: 1,
            provenance_words: [value; 8],
            source_words: [3; 8],
            context_words: [4; 8],
            scope_words: [5; 8],
            record: value - 1,
        });
        provider
            .htod_sync_copy_into_tracked(&events, &mut supports)
            .unwrap();
        let mut setup = domain.new_strict_recorder();
        setup.read_write(&statements);
        setup.read_write(&supports);
        setup.write(&handles);
        setup.write(&receipts);
        unsafe { domain.enqueue(setup, |_| Ok::<(), XlogError>(())) }
            .unwrap()
            .commit()
            .unwrap();
        stream.synchronize().unwrap();
        let before = provider.host_transfer_stats();
        let metadata_before = provider.host_launch_metadata_transfer_stats();
        let untracked_before = provider.untracked_metadata_dtoh_count();
        let captured = CapturedCudaGraph::capture_on_stream(&stream, || {
            let slot =
                |index| SemanticResidentReceiptSlot::new(&receipts, index).map_err(kernel_error);
            let statement =
                || SemanticResidentStatementBank::new(&statements, 0).map_err(kernel_error);
            let support =
                |index| SemanticResidentSupportBank::new(&supports, index).map_err(kernel_error);
            let base = graph
                .enqueue_resident_empty_root_handle(
                    SemanticResidentHandleSlot::new(&handles, 0).map_err(kernel_error)?,
                    slot(0)?,
                )
                .map_err(kernel_error)?;
            let first = graph
                .enqueue_resident_fork(base.handle(), slot(1)?)
                .map_err(kernel_error)?;
            let inserted = graph
                .enqueue_resident_insert_support(
                    first.candidate_handle(),
                    statement()?,
                    support(0)?,
                    slot(2)?,
                )
                .map_err(kernel_error)?;
            let refused = graph
                .enqueue_resident_insert_support(
                    inserted.candidate_handle(),
                    statement()?,
                    support(1)?,
                    slot(3)?,
                )
                .map_err(kernel_error)?;
            graph
                .enqueue_resident_seal(refused.candidate_handle(), slot(4)?)
                .map_err(kernel_error)?;
            let second = graph
                .enqueue_resident_fork(base.handle(), slot(5)?)
                .map_err(kernel_error)?;
            let denied = graph
                .enqueue_resident_fork(base.handle(), slot(6)?)
                .map_err(kernel_error)?;
            graph
                .enqueue_resident_seal(denied.candidate_handle(), slot(7)?)
                .map_err(kernel_error)?;
            let inserted = graph
                .enqueue_resident_insert_support(
                    second.candidate_handle(),
                    statement()?,
                    support(1)?,
                    slot(8)?,
                )
                .map_err(kernel_error)?;
            graph
                .enqueue_resident_seal(inserted.candidate_handle(), slot(9)?)
                .map_err(kernel_error)?;
            graph
                .enqueue_resident_insert_support(
                    first.candidate_handle(),
                    statement()?,
                    support(0)?,
                    slot(10)?,
                )
                .map_err(kernel_error)?;
            Ok(())
        })
        .unwrap();
        captured.launch(&stream).unwrap();
        let after = provider.host_transfer_stats();
        let metadata_after = provider.host_launch_metadata_transfer_stats();
        assert_eq!(
            (
                after.htod_calls,
                after.htod_bytes,
                after.dtoh_calls,
                after.dtoh_bytes
            ),
            (
                before.htod_calls,
                before.htod_bytes,
                before.dtoh_calls,
                before.dtoh_bytes
            )
        );
        assert_eq!(
            (metadata_after.htod_calls, metadata_after.htod_bytes),
            (metadata_before.htod_calls, metadata_before.htod_bytes)
        );
        assert_eq!(provider.untracked_metadata_dtoh_count(), untracked_before);
        stream.synchronize().unwrap();
        let observed = provider
            .dtoh_small_metadata_untracked(&receipts, 11)
            .unwrap();
        assert_eq!(observed[2].words[0], STATUS_OK);
        assert_eq!(observed[3].words[0], STATUS_SUPPORT_CAPACITY);
        assert_eq!(observed[4].words[0], STATUS_SUPPORT_CAPACITY);
        assert_eq!(
            observed[5].words[0], STATUS_OK,
            "the refused candidate must be retired"
        );
        assert_eq!(observed[6].words[0], STATUS_INACTIVE);
        assert_eq!(observed[7].words[0], STATUS_INACTIVE);
        assert_eq!(
            observed[8].words[0], STATUS_OK,
            "a failed fork must not retire another candidate"
        );
        assert_eq!(observed[9].words[0], STATUS_OK);
        assert_eq!(&observed[9].words[13..16], &[1, 1, 1]);
        assert_eq!(observed[10].words[0], STATUS_STALE_GENERATION);
        assert_eq!(observed[8].words[8], observed[2].words[8] + 1);
        assert_eq!(observed[8].words[10], observed[2].words[10] + 1);
        assert_eq!(observed[8].words[12], observed[2].words[12] + 1);
    }

    #[test]
    fn transition_reserve_checks_both_lanes_without_mutating_the_arena() {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
            return;
        }
        let provider = CudaProviderBuilder::new(0, MemoryBudget::with_limit(64 * 1024 * 1024))
            .with_stream_capacity(1)
            .build()
            .unwrap();
        let provider = Arc::new(provider);
        let runtime = Arc::clone(provider.memory().runtime().unwrap());
        let stream_id = runtime.stream_pool().acquire().unwrap();
        let stream = runtime.stream_pool().resolve(stream_id).unwrap();
        let domain = provider
            .bind_resident_execution_domain(runtime, stream_id, stream)
            .unwrap();
        let check = |caps: [u32; 4], changes: &[(usize, u64)], expected: u64| {
            let mut graph = provider
                .allocate_semantic_hypergraph(
                    &domain,
                    SemanticHypergraphCapacities::try_new(caps[0], caps[1], caps[2], caps[3])
                        .unwrap(),
                )
                .unwrap();
            let mut before = download_test_arena(&provider, &graph);
            for &(index, value) in changes {
                before[index] = value;
            }
            provider
                .htod_sync_copy_into_tracked(&before, &mut graph.arena)
                .unwrap();
            // The private command is the same read-only pre-draw operation used
            // by the resident dispatcher; it does not mint a reusable host permit.
            let mut command = graph.command_for(OP_PREFLIGHT_TRANSITION);
            command.words[8] = 0;
            command.words[9] = 1;
            let receipt = graph.run(command, ArenaAccess::Read).unwrap();
            assert_eq!(
                receipt.words[0], expected,
                "capacities={caps:?}, changes={changes:?}"
            );
            assert_eq!(
                receipt.words[41], 0,
                "preflight must not acquire a candidate"
            );
            let after = download_test_arena(&provider, &graph);
            assert_eq!(
                after, before,
                "preflight, including refusal, must be read-only"
            );
        };
        check([3, 4, 4, 4], &[], STATUS_OK);
        check([3, 4, 4, 4], &[(1, 0)], STATUS_FOREIGN_OWNER);
        check(
            [3, 4, 4, 4],
            &[(CONTROL_WORDS as usize + 1, 2)],
            STATUS_STALE_GENERATION,
        );
        check(
            [3, 4, 4, 4],
            &[(CONTROL_WORDS as usize, 0)],
            STATUS_INACTIVE,
        );
        check(
            [3, 4, 4, 4],
            &[((CONTROL_WORDS + ROOT_WORDS) as usize, 3)],
            STATUS_ROOT_CAPACITY,
        );
        check([2, 4, 4, 4], &[], STATUS_ROOT_CAPACITY);
        check([3, 3, 4, 4], &[], STATUS_STATEMENT_CAPACITY);
        check([3, 4, 3, 4], &[], STATUS_SUPPORT_CAPACITY);
        check([3, 4, 4, 3], &[], STATUS_VERSION_CAPACITY);
        let candidate = (CONTROL_WORDS + 3 * ROOT_WORDS) as usize;
        check([3, 4, 4, 4], &[(candidate, 1)], STATUS_INACTIVE);
        check([3, 4, 4, 4], &[(candidate + 1, u64::MAX - 2)], STATUS_OK);
        check(
            [3, 4, 4, 4],
            &[(candidate + 1, u64::MAX - 1)],
            STATUS_GENERATION_EXHAUSTED,
        );
        check(
            [3, 4, 4, 4],
            &[(candidate + 1, u64::MAX)],
            STATUS_GENERATION_EXHAUSTED,
        );
        let statements = candidate + CANDIDATE_WORDS as usize;
        let supports = statements + 4 * STATEMENT_WORDS as usize;
        let versions = supports + 4 * SUPPORT_WORDS as usize;
        // Count allocator order, not physical slot ordinal: a retained slot can
        // precede the first available record and need not have a reusable gen.
        check(
            [3, 5, 4, 4],
            &[(statements, 3), (statements + 1, u64::MAX)],
            STATUS_OK,
        );
        check(
            [3, 5, 4, 4],
            &[
                (statements, 3),
                (statements + STATEMENT_WORDS as usize + 1, u64::MAX - 1),
            ],
            STATUS_GENERATION_EXHAUSTED,
        );
        check(
            [3, 5, 4, 4],
            &[
                (statements, 3),
                (statements + 3 * STATEMENT_WORDS as usize + 1, u64::MAX - 1),
            ],
            STATUS_OK,
        );
        for (start, width, exhausted) in [
            (
                statements,
                STATEMENT_WORDS as usize,
                STATUS_STATEMENT_CAPACITY,
            ),
            (supports, SUPPORT_WORDS as usize, STATUS_SUPPORT_CAPACITY),
            (versions, VERSION_WORDS as usize, STATUS_VERSION_CAPACITY),
        ] {
            // A record retained by a different root is not available merely
            // because the acquired empty base has zero extents.
            check([3, 4, 4, 4], &[(start, 3)], exhausted);
            for slot in 0..4 {
                let generation = start + slot * width + 1;
                let maximum = u64::MAX - if slot < 2 { 2 } else { 1 };
                check([3, 4, 4, 4], &[(generation, maximum)], STATUS_OK);
                check(
                    [3, 4, 4, 4],
                    &[(generation, maximum + 1)],
                    STATUS_GENERATION_EXHAUSTED,
                );
            }
        }
        for extent in 3..6 {
            check(
                [3, 4, 4, 4],
                &[(CONTROL_WORDS as usize + extent, u32::MAX as u64 - 1)],
                STATUS_CORRUPT_LINEAGE,
            );
        }
        // Reachable statements need insertion headroom even though they are not
        // free. Keep the version/support generation references coherent so the
        // failure is generation exhaustion, not an earlier broken-link check.
        let statement = (CONTROL_WORDS + 4 * ROOT_WORDS + CANDIDATE_WORDS) as usize;
        let support = statement + 5 * STATEMENT_WORDS as usize;
        let version = support + 5 * SUPPORT_WORDS as usize;
        let heads = version + 5 * VERSION_WORDS as usize;
        let mut reachable = vec![
            (CONTROL_WORDS as usize + 3, 1),
            (CONTROL_WORDS as usize + 4, 1),
            (CONTROL_WORDS as usize + 5, 1),
            (statement, 3),
            (support, 3),
            (version, 3),
            (heads, 1),
            (version + 6, 1),
        ];
        for generation in [u64::MAX - 1, u64::MAX] {
            reachable.extend([
                (statement + 1, generation),
                (support + 4, generation),
                (version + 4, generation),
            ]);
            check(
                [4, 5, 5, 5],
                &reachable,
                if generation == u64::MAX {
                    STATUS_GENERATION_EXHAUSTED
                } else {
                    STATUS_OK
                },
            );
            reachable.truncate(8);
        }
    }

    #[test]
    fn resident_transition_reserve_precedes_two_lanes_and_survives_refusal() {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
            return;
        }
        let provider = CudaProviderBuilder::new(0, MemoryBudget::with_limit(64 * 1024 * 1024))
            .with_stream_capacity(1)
            .build()
            .unwrap();
        let provider = Arc::new(provider);
        let runtime = Arc::clone(provider.memory().runtime().unwrap());
        let stream_id = runtime.stream_pool().acquire().unwrap();
        let stream = runtime.stream_pool().resolve(stream_id).unwrap();
        let domain = provider
            .bind_resident_execution_domain(runtime, stream_id, Arc::clone(&stream))
            .unwrap();
        for refuse_first in [false, true] {
            let mut graph = provider
                .allocate_semantic_hypergraph(
                    &domain,
                    SemanticHypergraphCapacities::try_new(3, 4, 4, 4).unwrap(),
                )
                .unwrap();
            assert_eq!(graph.arena_words * 8, 2080);
            let candidate = (CONTROL_WORDS + 3 * ROOT_WORDS) as usize;
            let statement_start = candidate + CANDIDATE_WORDS as usize;
            let support_start = statement_start + 4 * STATEMENT_WORDS as usize;
            let version_start = support_start + 4 * SUPPORT_WORDS as usize;
            let mut arena = download_test_arena(&provider, &graph);
            arena[candidate + 1] = u64::MAX - 2;
            for (start, width) in [
                (statement_start, STATEMENT_WORDS as usize),
                (support_start, SUPPORT_WORDS as usize),
                (version_start, VERSION_WORDS as usize),
            ] {
                for slot in 0..4 {
                    arena[start + slot * width + 1] = u64::MAX - if slot < 2 { 2 } else { 1 };
                }
            }
            provider
                .htod_sync_copy_into_tracked(&arena, &mut graph.arena)
                .unwrap();
            let bytes = 4 * std::mem::size_of::<SemanticResidentDecodedStatement>()
                + 4 * std::mem::size_of::<SemanticResidentDecodedSupport>()
                + std::mem::size_of::<SemanticResidentHandleRecord>()
                + 11 * std::mem::size_of::<SemanticResidentReceiptRecord>();
            let mut reservation = provider.memory().reserve_bytes(bytes as u64).unwrap();
            let mut statements = reservation
                .alloc::<SemanticResidentDecodedStatement>(4)
                .unwrap();
            let mut supports = reservation
                .alloc::<SemanticResidentDecodedSupport>(4)
                .unwrap();
            let handles = reservation
                .alloc::<SemanticResidentHandleRecord>(1)
                .unwrap();
            let receipts = reservation
                .alloc::<SemanticResidentReceiptRecord>(11)
                .unwrap();
            assert_eq!(reservation.remaining_bytes(), 0);
            provider
                .htod_sync_copy_into_tracked(
                    &[17, 18, 19, 20].map(|value| SemanticResidentDecodedStatement {
                        identity_words: [value; 8],
                        record: value - 17,
                        reconstruction: [0; 10],
                    }),
                    &mut statements,
                )
                .unwrap();
            provider
                .htod_sync_copy_into_tracked(
                    &[0, 1, 2, 3].map(|index| SemanticResidentDecodedSupport {
                        polarity: if refuse_first && index == 1 { 0 } else { 1 },
                        provenance_words: [index + 1; 8],
                        source_words: [3; 8],
                        context_words: [4; 8],
                        scope_words: [5; 8],
                        record: index,
                    }),
                    &mut supports,
                )
                .unwrap();
            let mut setup = domain.new_strict_recorder();
            setup.read_write(&graph.arena);
            setup.read_write(&statements);
            setup.read_write(&supports);
            setup.write(&handles);
            setup.write(&receipts);
            unsafe { domain.enqueue(setup, |_| Ok::<(), XlogError>(())) }
                .unwrap()
                .commit()
                .unwrap();
            stream.synchronize().unwrap();
            let captured = CapturedCudaGraph::capture_on_stream(&stream, || {
                let slot = |index| {
                    SemanticResidentReceiptSlot::new(&receipts, index).map_err(kernel_error)
                };
                let base = graph
                    .enqueue_resident_empty_root_handle(
                        SemanticResidentHandleSlot::new(&handles, 0).map_err(kernel_error)?,
                        slot(0)?,
                    )
                    .map_err(kernel_error)?;
                graph
                    .enqueue_resident_preflight_transition(base.handle(), slot(1)?)
                    .map_err(kernel_error)?;
                for lane in 0..2 {
                    let base = SemanticResidentRootHandle::Receipt(SemanticResidentReceiptView {
                        allocation: &receipts,
                        index: 1,
                    });
                    let first = 2 + lane * 4;
                    let fork = graph
                        .enqueue_resident_fork(base, slot(first)?)
                        .map_err(kernel_error)?;
                    let inserted = graph
                        .enqueue_resident_insert_support(
                            fork.candidate_handle(),
                            SemanticResidentStatementBank::new(&statements, lane * 2)
                                .map_err(kernel_error)?,
                            SemanticResidentSupportBank::new(&supports, lane * 2)
                                .map_err(kernel_error)?,
                            slot(first + 1)?,
                        )
                        .map_err(kernel_error)?;
                    let inserted = graph
                        .enqueue_resident_insert_support(
                            inserted.candidate_handle(),
                            SemanticResidentStatementBank::new(&statements, lane * 2 + 1)
                                .map_err(kernel_error)?,
                            SemanticResidentSupportBank::new(&supports, lane * 2 + 1)
                                .map_err(kernel_error)?,
                            slot(first + 2)?,
                        )
                        .map_err(kernel_error)?;
                    graph
                        .enqueue_resident_seal(inserted.candidate_handle(), slot(first + 3)?)
                        .map_err(kernel_error)?;
                }
                // Both terminals consumed the two reserved candidate generations.
                graph
                    .enqueue_resident_preflight_transition(base.handle(), slot(10)?)
                    .map_err(kernel_error)?;
                Ok(())
            })
            .unwrap();
            let before = provider.host_transfer_stats();
            let metadata_before = provider.host_launch_metadata_transfer_stats();
            let untracked_before = provider.untracked_metadata_dtoh_count();
            captured.launch(&stream).unwrap();
            let after = provider.host_transfer_stats();
            let metadata_after = provider.host_launch_metadata_transfer_stats();
            assert_eq!(
                (
                    before.htod_calls,
                    before.htod_bytes,
                    before.dtoh_calls,
                    before.dtoh_bytes
                ),
                (
                    after.htod_calls,
                    after.htod_bytes,
                    after.dtoh_calls,
                    after.dtoh_bytes
                )
            );
            assert_eq!(
                (metadata_before.htod_calls, metadata_before.htod_bytes),
                (metadata_after.htod_calls, metadata_after.htod_bytes)
            );
            assert_eq!(untracked_before, provider.untracked_metadata_dtoh_count());
            stream.synchronize().unwrap();
            let observed = provider
                .dtoh_small_metadata_untracked(&receipts, 11)
                .unwrap();
            assert_eq!(observed[1].words[0], STATUS_OK);
            assert_eq!(&observed[1].words[33..36], &[1, 0, 1]);
            assert_eq!(observed[1].words[41], 0);
            assert_eq!(observed[2].words[0], STATUS_OK);
            assert_eq!(
                observed[5].words[0],
                if refuse_first {
                    STATUS_INVALID_COMMAND
                } else {
                    STATUS_OK
                }
            );
            assert_eq!(
                observed[6].words[0], STATUS_OK,
                "second lane must start after first retirement"
            );
            assert_eq!(observed[9].words[0], STATUS_OK);
            assert_eq!(
                &observed[9].words[13..16],
                &[2, 2, 2],
                "both lanes start from the same empty base"
            );
            assert_eq!(observed[10].words[0], STATUS_GENERATION_EXHAUSTED);
            if refuse_first {
                assert_eq!(
                    observed[7].words[8],
                    u64::MAX - 1,
                    "discarded statement slot is reused"
                );
                assert_eq!(observed[7].words[10], u64::MAX - 1);
                assert_eq!(observed[7].words[12], u64::MAX - 1);
            } else {
                assert_eq!(&observed[5].words[13..16], &[2, 2, 2]);
                assert_ne!(observed[5].words[3], observed[9].words[3]);
            }
        }
    }

    #[test]
    fn exhausted_candidate_generation_is_rejected_before_fork() {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
            eprintln!("Skipping: set XLOG_REQUIRE_CUDA=1 to run this real-CUDA contract");
            return;
        }
        let provider = CudaProviderBuilder::new(0, MemoryBudget::with_limit(64 * 1024 * 1024))
            .with_stream_capacity(1)
            .build()
            .unwrap();
        let provider = Arc::new(provider);
        let runtime = Arc::clone(provider.memory().runtime().unwrap());
        let stream_id = runtime.stream_pool().acquire().unwrap();
        let stream = runtime.stream_pool().resolve(stream_id).unwrap();
        let domain = provider
            .bind_resident_execution_domain(runtime, stream_id, stream)
            .unwrap();
        let mut graph = provider
            .allocate_semantic_hypergraph(
                &domain,
                SemanticHypergraphCapacities::try_new(2, 1, 1, 1).unwrap(),
            )
            .unwrap();
        // Establish a reachable cold boundary without iterating 2^64 retirements.
        let mut arena = download_test_arena(&provider, &graph);
        let candidate_offset = (CONTROL_WORDS + ROOT_WORDS * 2) as usize;
        arena[candidate_offset + 1] = u64::MAX - 1;
        provider
            .htod_sync_copy_into_tracked(&arena, &mut graph.arena)
            .unwrap();
        graph.fork.generation = u64::MAX - 1;
        let last = graph.fork(graph.empty_root()).unwrap();
        graph.discard(last).unwrap();
        let arena = download_test_arena(&provider, &graph);
        assert_eq!(arena[candidate_offset + 1], u64::MAX);
        let error = graph.fork(graph.empty_root()).unwrap_err();
        assert!(matches!(
            error,
            SemanticHypergraphError::GenerationExhausted {
                kind: SemanticHandleKind::Fork,
                slot: 0,
            }
        ));
        let after = download_test_arena(&provider, &graph);
        assert_eq!(
            after, arena,
            "an unretirable fork must not acquire or modify scratch"
        );
    }

    #[test]
    fn mismatched_launch_abi_is_rejected_without_arena_mutation() {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
            eprintln!("Skipping: set XLOG_REQUIRE_CUDA=1 to run this real-CUDA contract");
            return;
        }
        let provider = CudaProviderBuilder::new(0, MemoryBudget::with_limit(64 * 1024 * 1024))
            .with_stream_capacity(1)
            .build()
            .unwrap();
        let provider = Arc::new(provider);
        let runtime = Arc::clone(provider.memory().runtime().unwrap());
        let stream_id = runtime.stream_pool().acquire().unwrap();
        let stream = runtime.stream_pool().resolve(stream_id).unwrap();
        let domain = provider
            .bind_resident_execution_domain(runtime, stream_id, Arc::clone(&stream))
            .unwrap();
        let graph = provider
            .allocate_semantic_hypergraph(
                &domain,
                SemanticHypergraphCapacities::try_new(2, 1, 1, 1).unwrap(),
            )
            .unwrap();
        let before = download_test_arena(&provider, &graph);
        let descriptor = DeviceLaunchDescriptor {
            expected_owner: graph.owner,
            root_capacity: 2,
            statement_capacity: 1,
            support_capacity: 1,
            version_capacity: 1,
            arena_words: graph.arena_words,
            command_ptr: graph.command.device_ptr_value(),
            command_index: 0,
            handle_ptr: 0,
            handle_index: 0,
            source_receipt_ptr: 0,
            source_receipt_index: 0,
            decoded_input_ptr: 0,
            decoded_input_index: 0,
            decoded_statement_ptr: 0,
            decoded_statement_index: 0,
            decoded_support_ptr: 0,
            decoded_support_index: 0,
            output_ptr: 0,
            output_index: 0,
            receipt_ptr: graph.receipt.device_ptr_value(),
            receipt_index: 0,
            admission: HOST_COMMAND_ADMISSION,
            abi_generation: HYPERGRAPH_ABI_GENERATION - 1,
        };
        let mut recorder = domain.new_strict_recorder();
        recorder.read_write(&graph.arena);
        recorder.read(&graph.command);
        recorder.write(&graph.receipt);
        let enqueued = unsafe {
            domain.enqueue(recorder, |stream| {
                graph.execute.clone().launch_in(
                    stream,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (graph.arena.device_ptr_value(), descriptor),
                )
            })
        }
        .unwrap();
        enqueued.commit().unwrap();
        stream.synchronize().unwrap();
        let receipt = provider
            .dtoh_small_metadata_untracked(&graph.receipt, 1)
            .unwrap();
        assert_eq!(receipt[0].words[0], STATUS_ARENA_MISMATCH);
        let after = download_test_arena(&provider, &graph);
        assert_eq!(after, before);
    }

    #[test]
    fn resident_failed_seal_rolls_back_without_transferring_acquisition() {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
            eprintln!("Skipping: set XLOG_REQUIRE_CUDA=1 to run this real-CUDA contract");
            return;
        }
        let provider = CudaProviderBuilder::new(0, MemoryBudget::with_limit(64 * 1024 * 1024))
            .with_stream_capacity(1)
            .build()
            .unwrap();
        let provider = Arc::new(provider);
        let runtime = Arc::clone(provider.memory().runtime().unwrap());
        let stream_id = runtime.stream_pool().acquire().unwrap();
        let stream = runtime.stream_pool().resolve(stream_id).unwrap();
        let domain = provider
            .bind_resident_execution_domain(runtime, stream_id, Arc::clone(&stream))
            .unwrap();
        let mut graph = provider
            .allocate_semantic_hypergraph(
                &domain,
                SemanticHypergraphCapacities::try_new(1, 1, 1, 1).unwrap(),
            )
            .unwrap();
        let bytes = std::mem::size_of::<SemanticResidentDecodedStatement>()
            + std::mem::size_of::<SemanticResidentDecodedSupport>()
            + std::mem::size_of::<SemanticResidentHandleRecord>()
            + 9 * std::mem::size_of::<SemanticResidentReceiptRecord>();
        let mut reservation = provider.memory().reserve_bytes(bytes as u64).unwrap();
        let mut statements = reservation
            .alloc::<SemanticResidentDecodedStatement>(1)
            .unwrap();
        let mut supports = reservation
            .alloc::<SemanticResidentDecodedSupport>(1)
            .unwrap();
        let handles = reservation
            .alloc::<SemanticResidentHandleRecord>(1)
            .unwrap();
        let receipts = reservation
            .alloc::<SemanticResidentReceiptRecord>(9)
            .unwrap();
        assert_eq!(reservation.remaining_bytes(), 0);
        provider
            .htod_sync_copy_into_tracked(
                &[SemanticResidentDecodedStatement {
                    identity_words: [17; 8],
                    record: 0,
                    reconstruction: [0; 10],
                }],
                &mut statements,
            )
            .unwrap();
        provider
            .htod_sync_copy_into_tracked(
                &[SemanticResidentDecodedSupport {
                    polarity: 1,
                    provenance_words: [2; 8],
                    source_words: [3; 8],
                    context_words: [4; 8],
                    scope_words: [5; 8],
                    record: 0,
                }],
                &mut supports,
            )
            .unwrap();
        let mut setup = domain.new_strict_recorder();
        setup.read_write(&statements);
        setup.read_write(&supports);
        setup.write(&handles);
        setup.write(&receipts);
        unsafe { domain.enqueue(setup, |_| Ok::<(), XlogError>(())) }
            .unwrap()
            .commit()
            .unwrap();
        stream.synchronize().unwrap();
        let captured = CapturedCudaGraph::capture_on_stream(&stream, || {
            let slot =
                |index| SemanticResidentReceiptSlot::new(&receipts, index).map_err(kernel_error);
            let statement =
                || SemanticResidentStatementBank::new(&statements, 0).map_err(kernel_error);
            let base = graph
                .enqueue_resident_empty_root_handle(
                    SemanticResidentHandleSlot::new(&handles, 0).map_err(kernel_error)?,
                    slot(0)?,
                )
                .map_err(kernel_error)?;
            let acquired = graph
                .enqueue_resident_fork(base.handle(), slot(1)?)
                .map_err(kernel_error)?;
            let inserted = graph
                .enqueue_resident_insert_support(
                    acquired.candidate_handle(),
                    statement()?,
                    SemanticResidentSupportBank::new(&supports, 0).map_err(kernel_error)?,
                    slot(2)?,
                )
                .map_err(kernel_error)?;
            let refused_seal = graph
                .enqueue_resident_seal(inserted.candidate_handle(), slot(3)?)
                .map_err(kernel_error)?;
            let failed_fork = graph
                .enqueue_resident_fork(refused_seal.root_handle(), slot(4)?)
                .map_err(kernel_error)?;
            graph
                .enqueue_resident_seal(failed_fork.candidate_handle(), slot(5)?)
                .map_err(kernel_error)?;
            let next = graph
                .enqueue_resident_fork(base.handle(), slot(6)?)
                .map_err(kernel_error)?;
            graph
                .enqueue_resident_truth(
                    SemanticResidentView::Fork(next.candidate_handle()),
                    statement()?,
                    slot(7)?,
                )
                .map_err(kernel_error)?;
            graph
                .enqueue_resident_truth(
                    SemanticResidentView::Fork(acquired.candidate_handle()),
                    statement()?,
                    slot(8)?,
                )
                .map_err(kernel_error)?;
            Ok(())
        })
        .unwrap();
        captured.launch(&stream).unwrap();
        stream.synchronize().unwrap();
        let observed = provider
            .dtoh_small_metadata_untracked(&receipts, 9)
            .unwrap();
        assert_eq!(observed[3].words[0], STATUS_ROOT_CAPACITY);
        assert_eq!(observed[4].words[0], STATUS_ROOT_CAPACITY);
        assert_eq!(observed[5].words[0], STATUS_ROOT_CAPACITY);
        assert_eq!(
            observed[6].words[0], STATUS_OK,
            "a refused seal must roll back its candidate"
        );
        assert_eq!(observed[7].words[0], STATUS_OK);
        assert_eq!(
            observed[7].words[2], 0,
            "a refused lane must leave no partial semantic result"
        );
        assert_eq!(observed[8].words[0], STATUS_STALE_GENERATION);
        assert_eq!(&observed[4].words[40..42], &[0, 0]);
    }

    #[test]
    fn resident_admission_is_capture_safe_and_zero_host_io() {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
            eprintln!("Skipping: set XLOG_REQUIRE_CUDA=1 to run this real-CUDA contract");
            return;
        }

        let provider = CudaProviderBuilder::new(0, MemoryBudget::with_limit(64 * 1024 * 1024))
            .with_stream_capacity(1)
            .build()
            .expect("CUDA provider must be available when XLOG_REQUIRE_CUDA=1");
        let provider = Arc::new(provider);
        let runtime = Arc::clone(
            provider
                .memory()
                .runtime()
                .expect("canonical provider must own a runtime"),
        );
        let stream_id = runtime
            .stream_pool()
            .acquire()
            .expect("semantic hypergraph must acquire its resident stream");
        let stream = runtime
            .stream_pool()
            .resolve(stream_id)
            .expect("resident stream id must resolve through its runtime");
        let domain = provider
            .bind_resident_execution_domain(Arc::clone(&runtime), stream_id, Arc::clone(&stream))
            .expect("provider must bind its exact resident execution domain");
        let mut graph = provider
            .allocate_semantic_hypergraph(
                &domain,
                SemanticHypergraphCapacities::try_new(2, 1, 1, 1).unwrap(),
            )
            .expect("semantic storage initialization must execute on CUDA");

        let resident_bytes = (std::mem::size_of::<SemanticResidentDecodedInput>()
            + std::mem::size_of::<SemanticResidentHandleRecord>()
            + std::mem::size_of::<SemanticResidentDecodedStatement>()
            + std::mem::size_of::<SemanticResidentDecodedSupport>()
            + 12 * std::mem::size_of::<SemanticResidentReceiptRecord>()
            + 3 * std::mem::size_of::<SemanticResidentTruthValue>())
            as u64;
        let mut reservation = provider
            .memory()
            .reserve_bytes(resident_bytes)
            .expect("resident decoded banks and receipts reservation must succeed");
        let mut decoded_inputs = reservation
            .alloc::<SemanticResidentDecodedInput>(1)
            .expect("decoder-owned input allocation must succeed");
        let empty_root_handles = reservation
            .alloc::<SemanticResidentHandleRecord>(1)
            .expect("resident empty-root handle allocation must succeed");
        let decoded_statements = reservation
            .alloc::<SemanticResidentDecodedStatement>(1)
            .expect("resident decoded statement allocation must succeed");
        let decoded_supports = reservation
            .alloc::<SemanticResidentDecodedSupport>(1)
            .expect("resident decoded support allocation must succeed");
        let receipts = reservation
            .alloc::<SemanticResidentReceiptRecord>(12)
            .expect("resident receipt bank allocation must succeed");
        let truth_values = reservation
            .alloc::<SemanticResidentTruthValue>(3)
            .expect("resident truth-value output allocation must succeed");
        assert_eq!(reservation.remaining_bytes(), 0);

        let decoded_input = SemanticResidentDecodedInput {
            statement_identity_words: [
                0x1020_3040,
                0x5060_7080,
                0x90a0_b0c0,
                0xd0e0_f001,
                0x1234_5678,
                0x9abc_def0,
                0x0fed_cba9,
                0x8765_4321,
            ],
            polarity: 1,
            provenance_words: [1, 2, 3, 4, 5, 6, 7, 8],
            source_words: [11, 12, 13, 14, 15, 16, 17, 18],
            context_words: [21, 22, 23, 24, 25, 26, 27, 28],
            scope_words: [31, 32, 33, 34, 35, 36, 37, 38],
            statement_record: 0,
            support_record: 0,
            reconstruction: [0; 10],
        };
        provider
            .htod_sync_copy_into_tracked(&[decoded_input], &mut decoded_inputs)
            .expect("decoder-owned typed input upload must succeed before measurement");

        // Move allocation ownership to the exact resident stream without
        // initializing the output payloads. The captured producer below is
        // the first writer of every statement/support output field.
        let mut setup_recorder = domain.new_strict_recorder();
        setup_recorder.read_write(&decoded_inputs);
        setup_recorder.write(&empty_root_handles);
        setup_recorder.write(&decoded_statements);
        setup_recorder.write(&decoded_supports);
        setup_recorder.write(&receipts);
        setup_recorder.write(&truth_values);
        let setup_enqueued = unsafe { domain.enqueue(setup_recorder, |_| Ok::<(), XlogError>(())) }
            .expect("test setup must adopt resident allocations on the selected stream");
        setup_enqueued
            .commit()
            .expect("test setup must publish resident bank dependencies");
        stream
            .synchronize()
            .expect("test setup allocation adoption must complete before capture");

        let data_plane_before = provider.host_transfer_stats();
        let launch_metadata_before = provider.host_launch_metadata_transfer_stats();
        let tracked_dtoh_before = provider.d2h_transfer_count();
        let untracked_dtoh_before = provider.untracked_metadata_dtoh_count();
        let final_observation_before = provider.final_observation_transfer_stats();
        let semantic_launches_before = graph.execution_stats().cuda_kernel_launches();

        let captured = CapturedCudaGraph::capture_on_stream(&stream, || {
            let input =
                SemanticResidentDecodedInputBank::new(&decoded_inputs, 0).map_err(kernel_error)?;
            let statement =
                SemanticResidentStatementBank::new(&decoded_statements, 0).map_err(kernel_error)?;
            let support =
                SemanticResidentSupportBank::new(&decoded_supports, 0).map_err(kernel_error)?;
            let materialize_slot =
                SemanticResidentReceiptSlot::new(&receipts, 0).map_err(kernel_error)?;
            graph
                .enqueue_resident_materialize_decoded(input, statement, support, materialize_slot)
                .map_err(kernel_error)?;

            let handle_slot =
                SemanticResidentHandleSlot::new(&empty_root_handles, 0).map_err(kernel_error)?;
            let handle_receipt_slot =
                SemanticResidentReceiptSlot::new(&receipts, 1).map_err(kernel_error)?;
            let empty_root = graph
                .enqueue_resident_empty_root_handle(handle_slot, handle_receipt_slot)
                .map_err(kernel_error)?;

            let statement =
                SemanticResidentStatementBank::new(&decoded_statements, 0).map_err(kernel_error)?;
            let truth_slot =
                SemanticResidentReceiptSlot::new(&receipts, 8).map_err(kernel_error)?;
            let base_truth = graph
                .enqueue_resident_truth(
                    SemanticResidentView::Root(empty_root.handle()),
                    statement,
                    truth_slot,
                )
                .map_err(kernel_error)?;
            let output = SemanticResidentTruthSlot::new(&truth_values, 1).map_err(kernel_error)?;
            let consumer = SemanticResidentReceiptSlot::new(&receipts, 9).map_err(kernel_error)?;
            graph
                .enqueue_resident_truth_consumer(base_truth.device_view(), output, consumer)
                .map_err(kernel_error)?;

            let fork_slot = SemanticResidentReceiptSlot::new(&receipts, 2).map_err(kernel_error)?;
            let fork_receipt = graph
                .enqueue_resident_fork(empty_root.handle(), fork_slot)
                .map_err(kernel_error)?;

            let statement =
                SemanticResidentStatementBank::new(&decoded_statements, 0).map_err(kernel_error)?;
            let support =
                SemanticResidentSupportBank::new(&decoded_supports, 0).map_err(kernel_error)?;
            let insert_slot =
                SemanticResidentReceiptSlot::new(&receipts, 3).map_err(kernel_error)?;
            let insert_receipt = graph
                .enqueue_resident_insert_support(
                    fork_receipt.candidate_handle(),
                    statement,
                    support,
                    insert_slot,
                )
                .map_err(kernel_error)?;

            let seal_slot = SemanticResidentReceiptSlot::new(&receipts, 4).map_err(kernel_error)?;
            let sealed = graph
                .enqueue_resident_seal(insert_receipt.candidate_handle(), seal_slot)
                .map_err(kernel_error)?;

            let statement =
                SemanticResidentStatementBank::new(&decoded_statements, 0).map_err(kernel_error)?;
            let truth_slot =
                SemanticResidentReceiptSlot::new(&receipts, 10).map_err(kernel_error)?;
            let sealed_truth = graph
                .enqueue_resident_truth(
                    SemanticResidentView::Root(sealed.root_handle()),
                    statement,
                    truth_slot,
                )
                .map_err(kernel_error)?;
            let output = SemanticResidentTruthSlot::new(&truth_values, 2).map_err(kernel_error)?;
            let consumer = SemanticResidentReceiptSlot::new(&receipts, 11).map_err(kernel_error)?;
            graph
                .enqueue_resident_truth_consumer(sealed_truth.device_view(), output, consumer)
                .map_err(kernel_error)?;

            let refork_slot =
                SemanticResidentReceiptSlot::new(&receipts, 5).map_err(kernel_error)?;
            let refork = graph
                .enqueue_resident_fork(sealed.root_handle(), refork_slot)
                .map_err(kernel_error)?;

            let truth_slot =
                SemanticResidentReceiptSlot::new(&receipts, 6).map_err(kernel_error)?;
            let statement =
                SemanticResidentStatementBank::new(&decoded_statements, 0).map_err(kernel_error)?;
            let truth_receipt = graph
                .enqueue_resident_truth(
                    SemanticResidentView::Fork(refork.candidate_handle()),
                    statement,
                    truth_slot,
                )
                .map_err(kernel_error)?;

            let output = SemanticResidentTruthSlot::new(&truth_values, 0).map_err(kernel_error)?;
            let consumer_slot =
                SemanticResidentReceiptSlot::new(&receipts, 7).map_err(kernel_error)?;
            graph
                .enqueue_resident_truth_consumer(truth_receipt.device_view(), output, consumer_slot)
                .map_err(kernel_error)?;
            Ok(())
        })
        .expect("typed resident semantic chain must be CUDA-capture safe");
        assert_eq!(
            graph.execution_stats().cuda_kernel_launches(),
            semantic_launches_before + 12
        );
        captured
            .launch(&stream)
            .expect("captured resident semantic chain must launch");

        let data_plane_after = provider.host_transfer_stats();
        assert_eq!(data_plane_after.dtoh_bytes, data_plane_before.dtoh_bytes);
        assert_eq!(data_plane_after.htod_bytes, data_plane_before.htod_bytes);
        assert_eq!(data_plane_after.dtoh_calls, data_plane_before.dtoh_calls);
        assert_eq!(data_plane_after.htod_calls, data_plane_before.htod_calls);
        let launch_metadata_after = provider.host_launch_metadata_transfer_stats();
        assert_eq!(
            launch_metadata_after.htod_bytes,
            launch_metadata_before.htod_bytes
        );
        assert_eq!(
            launch_metadata_after.htod_calls,
            launch_metadata_before.htod_calls
        );
        assert_eq!(provider.d2h_transfer_count(), tracked_dtoh_before);
        assert_eq!(
            provider.untracked_metadata_dtoh_count(),
            untracked_dtoh_before
        );
        let final_observation_after = provider.final_observation_transfer_stats();
        assert_eq!(
            final_observation_after.dtoh_bytes,
            final_observation_before.dtoh_bytes
        );
        assert_eq!(
            final_observation_after.dtoh_calls,
            final_observation_before.dtoh_calls
        );
        assert_eq!(
            final_observation_after.pinned_receipts,
            final_observation_before.pinned_receipts
        );

        stream
            .synchronize()
            .expect("the test may observe only after the measured resident interval");
        let observed_truths = provider
            .dtoh_small_metadata_untracked(&truth_values, 3)
            .expect("the test may observe only the typed truth output after synchronization");
        for (truth_value, expected) in observed_truths.iter().zip([1, 0, 1]) {
            assert_eq!(truth_value.status, STATUS_OK);
            assert_eq!(truth_value.truth, expected);
            assert_eq!(truth_value.owner, graph.owner);
            assert_eq!(truth_value.reserved, 0);
        }
        let statement_identity = SemanticStatementIdentity(identity_bytes_from_dwords(
            decoded_input.statement_identity_words,
        ));
        let mut expected = Sha256::new();
        expected.update(b"xlog.semantic.support.v1\0");
        expected.update(statement_identity.as_bytes());
        expected.update(1u32.to_le_bytes()); // positive polarity
        expected.update(1u32.to_le_bytes()); // unit presence
        for words in [
            decoded_input.provenance_words,
            decoded_input.source_words,
            decoded_input.context_words,
            decoded_input.scope_words,
        ] {
            expected.update(identity_bytes_from_dwords(words));
        }
        let expected_support_identity = SemanticSupportIdentity(expected.finalize().into());
        assert_eq!(
            provider.untracked_metadata_dtoh_count(),
            untracked_dtoh_before + 1
        );
        let observed_receipts = provider
            .dtoh_small_metadata_untracked(&receipts, 4)
            .expect("the test may observe bounded receipts only after synchronization");
        let insert_receipt = &observed_receipts[3];
        assert_eq!(insert_receipt.words[0], STATUS_OK);
        assert_eq!(insert_receipt.words[1], OUTCOME_INSERTED);
        assert_eq!(
            receipt_identity(insert_receipt, 24),
            *expected_support_identity.as_bytes()
        );
        assert_eq!(
            provider.untracked_metadata_dtoh_count(),
            untracked_dtoh_before + 2
        );

        let replay_data_plane_before = provider.host_transfer_stats();
        let replay_launch_metadata_before = provider.host_launch_metadata_transfer_stats();
        let replay_tracked_dtoh_before = provider.d2h_transfer_count();
        let replay_untracked_dtoh_before = provider.untracked_metadata_dtoh_count();
        let replay_final_observation_before = provider.final_observation_transfer_stats();
        captured
            .launch(&stream)
            .expect("captured resident semantic chain must be replayable");
        let replay_data_plane_after = provider.host_transfer_stats();
        assert_eq!(
            replay_data_plane_after.dtoh_bytes,
            replay_data_plane_before.dtoh_bytes
        );
        assert_eq!(
            replay_data_plane_after.htod_bytes,
            replay_data_plane_before.htod_bytes
        );
        assert_eq!(
            replay_data_plane_after.dtoh_calls,
            replay_data_plane_before.dtoh_calls
        );
        assert_eq!(
            replay_data_plane_after.htod_calls,
            replay_data_plane_before.htod_calls
        );
        let replay_launch_metadata_after = provider.host_launch_metadata_transfer_stats();
        assert_eq!(
            replay_launch_metadata_after.htod_bytes,
            replay_launch_metadata_before.htod_bytes
        );
        assert_eq!(
            replay_launch_metadata_after.htod_calls,
            replay_launch_metadata_before.htod_calls
        );
        assert_eq!(provider.d2h_transfer_count(), replay_tracked_dtoh_before);
        assert_eq!(
            provider.untracked_metadata_dtoh_count(),
            replay_untracked_dtoh_before
        );
        let replay_final_observation_after = provider.final_observation_transfer_stats();
        assert_eq!(
            replay_final_observation_after.dtoh_bytes,
            replay_final_observation_before.dtoh_bytes
        );
        assert_eq!(
            replay_final_observation_after.dtoh_calls,
            replay_final_observation_before.dtoh_calls
        );
        assert_eq!(
            replay_final_observation_after.pinned_receipts,
            replay_final_observation_before.pinned_receipts
        );
        stream
            .synchronize()
            .expect("replayed resident semantic chain must complete");
        let replayed_truths = provider
            .dtoh_small_metadata_untracked(&truth_values, 3)
            .expect("the test may observe the replayed typed output after synchronization");
        for (truth_value, status) in
            replayed_truths
                .iter()
                .zip([STATUS_INACTIVE, STATUS_OK, STATUS_INACTIVE])
        {
            assert_eq!(truth_value.status, status);
            assert_eq!(truth_value.truth, 0);
            assert_eq!(truth_value.owner, graph.owner);
            assert_eq!(truth_value.reserved, 0);
        }
        let replayed_receipts = provider
            .dtoh_small_metadata_untracked(&receipts, 12)
            .expect("the test may observe bounded replay receipts only after synchronization");
        assert_eq!(replayed_receipts[0].words[0], STATUS_OK);
        assert_eq!(replayed_receipts[1].words[0], STATUS_OK);
        assert_eq!(replayed_receipts[2].words[0], STATUS_INACTIVE);
        assert_eq!(
            decode_kind(replayed_receipts[2].words[33]),
            SemanticHandleKind::Fork
        );
        for propagated in replayed_receipts[3..8]
            .iter()
            .chain(&replayed_receipts[10..])
        {
            assert_eq!(propagated.words, replayed_receipts[2].words);
        }
        assert_eq!(replayed_receipts[8].words[0], STATUS_OK);
        assert_eq!(replayed_receipts[8].words[2], 0);
        assert_eq!(replayed_receipts[9].words[0], STATUS_OK);
        assert_eq!(
            provider.untracked_metadata_dtoh_count(),
            untracked_dtoh_before + 4
        );

        let launches_before_host_rejection = graph.execution_stats().cuda_kernel_launches();
        let launch_metadata_before_host_rejection = provider.host_launch_metadata_transfer_stats();
        let empty_root = graph.empty_root();
        assert!(matches!(
            graph.snapshot(SemanticView::Root(empty_root)),
            Err(SemanticHypergraphError::DeviceControlled)
        ));
        assert_eq!(
            graph.execution_stats().cuda_kernel_launches(),
            launches_before_host_rejection
        );
        let launch_metadata_after_host_rejection = provider.host_launch_metadata_transfer_stats();
        assert_eq!(
            launch_metadata_after_host_rejection.htod_bytes,
            launch_metadata_before_host_rejection.htod_bytes
        );
        assert_eq!(
            launch_metadata_after_host_rejection.htod_calls,
            launch_metadata_before_host_rejection.htod_calls
        );
    }
}
