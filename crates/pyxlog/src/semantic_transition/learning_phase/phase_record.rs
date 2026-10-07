//! Native issuance and single-attempt custody of cold phase records.

use super::*;
use ed25519_dalek::{Signer, SigningKey};
use rand::rngs::OsRng;

const INPUT_DOMAIN: &[u8] = b"xlog.learning-phase.inputs.v2\0";
const RECORD_DOMAIN: &[u8] = b"xlog.learning-phase.record.v1\0";
const RECORD_OVERHEAD: usize = RECORD_DOMAIN.len() + 1 + 32 + 32 + 8 + 32 + 8 + 24 + 8 + 64;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum RecordKind {
    Admission = 0,
    PreparationOutcome = 1,
    Delivery = 2,
    TerminalRefusal = 3,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct RecordLimits {
    pub(super) record_bytes: u64,
    pub(super) total_bytes: u64,
    pub(super) records: u64,
}

impl RecordLimits {
    pub(super) fn read(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let value = ColdValue::read(value, &mut 1024, 0)?;
        let fields = value.fields(3)?;
        let limits = Self {
            record_bytes: fields[0].unsigned()?,
            total_bytes: fields[1].unsigned()?,
            records: fields[2].unsigned()?,
        };
        if limits.record_bytes == 0
            || limits.record_bytes > limits.total_bytes
            || limits.total_bytes > isize::MAX as u64
            || limits.records == 0
            || limits.records > isize::MAX as u64
        {
            return Err(invalid(
                "phase record limits require explicit positive record/total/count bounds within the platform index range",
            ));
        }
        Ok(limits)
    }

    pub(super) fn encode(self) -> [u8; 24] {
        let mut bytes = [0; 24];
        for (target, value) in bytes.as_chunks_mut::<8>().0.iter_mut().zip([
            self.record_bytes,
            self.total_bytes,
            self.records,
        ]) {
            target.copy_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

/// A bounded slice of an original signed record or length-delimited payload.
/// Parsing never invokes a Python callback or allocates from a declared length.
pub(super) struct RecordReader<'a>(pub(super) &'a [u8]);

impl<'a> RecordReader<'a> {
    pub(super) fn take(&mut self, count: usize) -> PyResult<&'a [u8]> {
        if count > self.0.len() {
            return Err(invalid("signed phase material is truncated"));
        }
        let (value, remaining) = self.0.split_at(count);
        self.0 = remaining;
        Ok(value)
    }

    pub(super) fn word(&mut self) -> PyResult<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub(super) fn field(&mut self) -> PyResult<&'a [u8]> {
        let count = usize::try_from(self.word()?)
            .map_err(|_| invalid("signed phase field length exceeds this host"))?;
        self.take(count)
    }

    pub(super) fn finish(self) -> PyResult<()> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(invalid(
                "signed phase material contains an unconsumed suffix",
            ))
        }
    }
}

/// Authenticate the complete original lifecycle before any replacement CUDA
/// owner exists. The issuer and limits come from the independent consumer pin,
/// never from an untrusted record header or a newly generated signing key.
pub(super) fn delivery_payloads<'a>(
    records: [&'a [u8]; 3],
    issuer: [u8; 32],
    phase_id: [u8; 32],
    limits: RecordLimits,
) -> PyResult<[&'a [u8]; 3]> {
    if limits.records < 3 {
        return Err(invalid(
            "original phase bounds cannot contain the complete lifecycle",
        ));
    }
    if (records[0].len() as u64)
        .checked_add(limits.record_bytes)
        .and_then(|required| required.checked_add(limits.record_bytes))
        .is_none_or(|required| required > limits.total_bytes)
    {
        return Err(invalid(
            "original phase limits lost their reserved complete outcome custody",
        ));
    }
    let key = ed25519_dalek::VerifyingKey::from_bytes(&issuer)
        .map_err(|_| invalid("independent phase issuer has an invalid encoding"))?;
    let mut previous = [0u8; 32];
    let mut total = 0u64;
    let mut payloads = [&[][..]; 3];
    for (ordinal, bytes) in records.into_iter().enumerate() {
        total = total
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| invalid("original phase record total overflowed"))?;
        if bytes.len() < RECORD_OVERHEAD
            || bytes.len() as u64 > limits.record_bytes
            || total > limits.total_bytes
        {
            return Err(invalid(
                "signed phase chain exceeds its original record bounds",
            ));
        }
        let (message, signature) = bytes.split_at(bytes.len() - 64);
        let signature = ed25519_dalek::Signature::from_slice(signature)
            .map_err(|_| invalid("signed phase record has an invalid signature encoding"))?;
        key.verify_strict(message, &signature)
            .map_err(|_| invalid("signed phase record differs from its independent issuer"))?;
        let mut reader = RecordReader(message);
        if reader.take(RECORD_DOMAIN.len())? != RECORD_DOMAIN
            || reader.take(1)? != [ordinal as u8]
            || reader.take(32)? != phase_id
            || reader.take(32)? != issuer
            || reader.word()? != ordinal as u64
            || reader.take(32)? != previous
            || reader.word()? != total
            || reader.take(24)? != limits.encode()
        {
            return Err(invalid(
                "signed phase record changed its original lifecycle or limits",
            ));
        }
        payloads[ordinal] = reader.field()?;
        reader.finish()?;
        previous = Sha256::digest(bytes).into();
    }
    let admission = payloads[0];
    let digest: [u8; 32] = Sha256::digest(admission).into();
    if !admission.starts_with(INPUT_DOMAIN) || digest != phase_id {
        return Err(invalid(
            "phase admission changed its original input identity",
        ));
    }
    let mut inputs = RecordReader(&admission[INPUT_DOMAIN.len()..]);
    for _ in 0..8 {
        inputs.field()?;
    }
    if inputs.field()? != limits.encode() {
        return Err(invalid(
            "phase admission changed its independent original bounds",
        ));
    }
    inputs.finish()?;
    Ok(payloads)
}

pub(super) struct RecordAttempt {
    pub(super) kind: RecordKind,
    pub(super) ordinal: u64,
    pub(super) bytes: Arc<[u8]>,
    pub(super) digest: [u8; 32],
    pub(super) readback_observed: bool,
}

/// The private key never enters Python or a checkpoint. Each write is retained
/// before calling the original store and may subsequently only be read back.
pub(super) struct PhaseRecords {
    pub(super) phase_id: [u8; 32],
    pub(super) limits: RecordLimits,
    key: SigningKey,
    pub(super) inputs: Arc<[u8]>,
    pub(super) issuer_pinned: bool,
    pub(super) pin_attempted: bool,
    pub(super) preparation_outcome_known: bool,
    pub(super) delivery_known: bool,
    pub(super) refusal_known: bool,
    pub(super) refusal_record: Option<Arc<[u8]>>,
    ordinal: u64,
    total_bytes: u64,
    previous: [u8; 32],
    admission: Option<Arc<[u8]>>,
    pub(super) attempt: Option<RecordAttempt>,
}

fn append_fields(target: &mut Vec<u8>, fields: &[&[u8]]) -> PyResult<()> {
    for field in fields {
        let length = u64::try_from(field.len())
            .map_err(|_| invalid("phase record field length exceeds its native index"))?;
        target.extend_from_slice(&length.to_le_bytes());
        target.extend_from_slice(field);
    }
    Ok(())
}

impl PhaseRecords {
    pub(super) fn new(fields: &[&[u8]], limits: RecordLimits) -> PyResult<Self> {
        let input_bytes = fields
            .iter()
            .try_fold(INPUT_DOMAIN.len() as u64 + 8 + 24, |total, field| {
                total.checked_add(8)?.checked_add(field.len() as u64)
            });
        let admission_bytes = input_bytes
            .and_then(|input_bytes| input_bytes.checked_add(RECORD_OVERHEAD as u64))
            .ok_or_else(|| invalid("phase admission record size overflowed"))?;
        // Reserve both complete outcomes before admission. The caller's original
        // bounds are never enlarged after model work or source retirement.
        if limits.records < 3
            || admission_bytes > limits.record_bytes
            || admission_bytes
                .checked_add(limits.record_bytes)
                .and_then(|required| required.checked_add(limits.record_bytes))
                .is_none_or(|required| required > limits.total_bytes)
        {
            return Err(invalid(
                "phase record bounds cannot retain the full admission, preparation outcome and delivery",
            ));
        }
        let mut inputs = INPUT_DOMAIN.to_vec();
        append_fields(&mut inputs, fields)?;
        append_fields(&mut inputs, &[&limits.encode()])?;
        let phase_id = Sha256::digest(&inputs).into();
        Ok(Self {
            phase_id,
            limits,
            key: SigningKey::generate(&mut OsRng),
            inputs: inputs.into(),
            issuer_pinned: false,
            pin_attempted: false,
            preparation_outcome_known: false,
            delivery_known: false,
            refusal_known: false,
            refusal_record: None,
            ordinal: 0,
            total_bytes: 0,
            previous: [0; 32],
            admission: None,
            attempt: None,
        })
    }

    pub(super) fn issuer(&self) -> [u8; 32] {
        self.key.verifying_key().to_bytes()
    }

    pub(super) fn delivered_identity(&self) -> PyResult<([u8; 32], [u8; 32], Arc<[u8]>)> {
        if !self.delivery_known
            || self.ordinal != 3
            || self.attempt.is_some()
            || !self.issuer_pinned
        {
            return Err(invalid(
                "checkpoint issuer requires known full signed Delivery readback",
            ));
        }
        Ok((
            self.phase_id,
            self.previous,
            self.admission
                .as_ref()
                .map(Arc::clone)
                .ok_or_else(|| invalid("known Delivery lost its original native Admission"))?,
        ))
    }

    pub(super) fn require_preparation_admission(&self) -> PyResult<()> {
        if !self.issuer_pinned
            || self.ordinal != 1
            || self.attempt.is_some()
            || self.admission.is_none()
        {
            return Err(invalid(
                "private phase allocation requires exact durable readback of its original signed admission",
            ));
        }
        Ok(())
    }

    /// Only the exact, signed, durably read-back Admission authorizes private
    /// native writes. Scientific acceptance is not obtained before those writes.
    pub(super) fn confirmed_admission(&self) -> PyResult<Arc<[u8]>> {
        self.require_preparation_admission()?;
        self.admission
            .as_ref()
            .map(Arc::clone)
            .ok_or_else(|| invalid("private phase lost its original confirmed native admission"))
    }

    /// A confirmed original pin may still precede the first admission write.
    /// An unresolved write is not permission to issue a second one.
    pub(super) fn preparation_admission_needed(&self) -> PyResult<bool> {
        if !self.pin_attempted
            || !self.issuer_pinned
            || self.attempt.is_some()
            || self.preparation_outcome_known
            || self.ordinal > 1
        {
            return Err(invalid(
                "preparation continuation requires the original confirmed issuer and no unresolved record write",
            ));
        }
        Ok(self.ordinal == 0)
    }

    pub(super) fn check_payload_length(&self, payload_length: usize) -> PyResult<u64> {
        let length = (payload_length as u64)
            .checked_add(RECORD_OVERHEAD as u64)
            .ok_or_else(|| invalid("signed phase record size overflowed"))?;
        let total = self
            .total_bytes
            .checked_add(length)
            .ok_or_else(|| invalid("phase record total size overflowed"))?;
        if length > self.limits.record_bytes || total > self.limits.total_bytes {
            return Err(invalid(
                "complete phase record exceeds its original storage bounds",
            ));
        }
        Ok(total)
    }

    pub(super) fn begin(&mut self, kind: RecordKind, payload: &[u8]) -> PyResult<()> {
        if !self.issuer_pinned || self.attempt.is_some() {
            return Err(invalid(
                "phase record requires its independently pinned issuer and resolution of the original write",
            ));
        }
        if self.ordinal >= self.limits.records {
            return Err(invalid("original phase record count is exhausted"));
        }
        let ordered = !self.refusal_known
            && match kind {
                RecordKind::Admission => self.ordinal == 0 && self.admission.is_none(),
                RecordKind::PreparationOutcome => self.ordinal == 1 && self.admission.is_some(),
                RecordKind::Delivery => {
                    self.ordinal == 2 && self.preparation_outcome_known && !self.delivery_known
                }
                RecordKind::TerminalRefusal => {
                    self.admission.is_some()
                        && !self.delivery_known
                        && (self.ordinal == 1
                            || self.ordinal == 2 && self.preparation_outcome_known)
                }
            };
        if !ordered {
            return Err(invalid(
                "phase record changed its original signed lifecycle order",
            ));
        }
        let total = self.check_payload_length(payload.len())?;
        let mut bytes = RECORD_DOMAIN.to_vec();
        bytes.push(kind as u8);
        bytes.extend_from_slice(&self.phase_id);
        bytes.extend_from_slice(&self.issuer());
        bytes.extend_from_slice(&self.ordinal.to_le_bytes());
        bytes.extend_from_slice(&self.previous);
        let total_offset = bytes.len();
        bytes.extend_from_slice(&[0; 8]);
        bytes.extend_from_slice(&self.limits.encode());
        append_fields(&mut bytes, &[payload])?;
        bytes[total_offset..total_offset + 8].copy_from_slice(&total.to_le_bytes());
        bytes.extend_from_slice(&self.key.sign(&bytes).to_bytes());
        self.attempt = Some(RecordAttempt {
            kind,
            ordinal: self.ordinal,
            digest: Sha256::digest(&bytes).into(),
            bytes: bytes.into(),
            readback_observed: false,
        });
        Ok(())
    }

    pub(super) fn confirm(&mut self, readback: &Bound<'_, PyAny>) -> PyResult<()> {
        let attempt = self
            .attempt
            .as_mut()
            .ok_or_else(|| invalid("phase record readback has no original write attempt"))?;
        require_model_bytes(readback, &attempt.bytes)?;
        let (message, signature) = attempt.bytes.split_at(attempt.bytes.len() - 64);
        let signature = ed25519_dalek::Signature::from_slice(signature)
            .map_err(|_| invalid("phase record signature has an invalid encoding"))?;
        self.key
            .verifying_key()
            .verify_strict(message, &signature)
            .map_err(|_| invalid("phase record failed original native issuer verification"))?;
        attempt.readback_observed = true;
        Ok(())
    }

    pub(super) fn advance(&mut self) -> PyResult<()> {
        let attempt = self
            .attempt
            .as_ref()
            .ok_or_else(|| invalid("phase record progress has no retained write"))?;
        if !attempt.readback_observed {
            return Err(invalid(
                "phase record progress requires exact durable readback",
            ));
        }
        self.ordinal = self
            .ordinal
            .checked_add(1)
            .ok_or_else(|| invalid("phase record ordinal overflowed"))?;
        self.total_bytes = self
            .total_bytes
            .checked_add(attempt.bytes.len() as u64)
            .ok_or_else(|| invalid("phase record total size overflowed"))?;
        self.previous = attempt.digest;
        match attempt.kind {
            RecordKind::Admission => self.admission = Some(Arc::clone(&attempt.bytes)),
            RecordKind::PreparationOutcome => self.preparation_outcome_known = true,
            RecordKind::Delivery => self.delivery_known = true,
            RecordKind::TerminalRefusal => {
                self.refusal_known = true;
                self.refusal_record = Some(Arc::clone(&attempt.bytes));
            }
        }
        self.attempt = None;
        Ok(())
    }
}
