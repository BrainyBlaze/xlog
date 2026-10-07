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
}

#[derive(Clone, Copy)]
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

    fn encode(self) -> [u8; 24] {
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
        let ordered = match kind {
            RecordKind::Admission => self.ordinal == 0 && self.admission.is_none(),
            RecordKind::PreparationOutcome => self.ordinal == 1 && self.admission.is_some(),
            RecordKind::Delivery => {
                self.ordinal == 2 && self.preparation_outcome_known && !self.delivery_known
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
        }
        self.attempt = None;
        Ok(())
    }
}
