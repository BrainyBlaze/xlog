//! W3.8 AOT stream schedule construction for independent WCOJ rules.

use xlog_ir::Stratum;

/// Hardware inputs used by the stream schedule pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HardwareCapabilities {
    /// Number of streaming multiprocessors visible to the runtime.
    pub sm_count: usize,
    /// Count of independent rules in the stratum.
    pub independent_rule_count: usize,
}

impl HardwareCapabilities {
    /// Build schedule inputs from an SM count and independent-rule count.
    pub fn new(sm_count: usize, independent_rule_count: usize) -> Self {
        Self {
            sm_count,
            independent_rule_count,
        }
    }
}

/// Phase node kind in the W3.8 Count -> Scan -> Resize -> Materialize schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamPhase {
    /// WCOJ count kernel phase.
    Count,
    /// Deterministic prefix scan over per-block counts.
    Scan,
    /// Output allocation phase after scan determines cardinality.
    Resize,
    /// WCOJ materialize kernel phase.
    Materialize,
}

/// One phase scheduled for one independent rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamPhaseNode {
    /// Stratum id that owns this phase node.
    pub stratum_id: u32,
    /// Rule index within the stratum's independent-rule list.
    pub rule_index: usize,
    /// Phase executed for this rule.
    pub phase: StreamPhase,
    /// CUDA stream slot selected by greedy bin assignment.
    pub stream_index: usize,
}

/// Phase-aligned stream schedule for a stratum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamSchedule {
    /// Stratum id this schedule covers.
    pub stratum_id: u32,
    /// Number of CUDA stream slots used by the schedule.
    pub stream_count: usize,
    /// Ordered phase nodes. Phases are grouped by phase kind, then rule index.
    pub phases: Vec<StreamPhaseNode>,
}

/// Build the W3.8 phase-aligned schedule for a stratum.
///
/// Stream count follows the paper-plan rule:
/// `min(SM_count / 4, max_independent_rules_in_stratum)`, with one stream
/// retained for single-rule strata so the scheduler produces the same serial
/// execution shape as the non-mux path.
pub fn schedule_streams(stratum: &Stratum, hw: &HardwareCapabilities) -> StreamSchedule {
    let rule_count = hw.independent_rule_count;
    let stream_count = if rule_count == 0 {
        0
    } else {
        let sm_lanes = (hw.sm_count / 4).max(1);
        sm_lanes.min(rule_count)
    };
    let mut phases = Vec::with_capacity(rule_count.saturating_mul(4));
    for phase in [
        StreamPhase::Count,
        StreamPhase::Scan,
        StreamPhase::Resize,
        StreamPhase::Materialize,
    ] {
        for rule_index in 0..rule_count {
            phases.push(StreamPhaseNode {
                stratum_id: stratum.id,
                rule_index,
                phase,
                stream_index: if stream_count == 0 {
                    0
                } else {
                    rule_index % stream_count
                },
            });
        }
    }
    StreamSchedule {
        stratum_id: stratum.id,
        stream_count,
        phases,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stratum() -> Stratum {
        Stratum {
            id: 7,
            sccs: vec![0],
        }
    }

    #[test]
    fn schedules_four_rules_on_four_streams_by_phase() {
        let schedule = schedule_streams(&stratum(), &HardwareCapabilities::new(16, 4));
        assert_eq!(schedule.stream_count, 4);
        assert_eq!(schedule.phases.len(), 16);
        assert!(schedule.phases[0..4]
            .iter()
            .all(|node| node.phase == StreamPhase::Count));
        assert!(schedule.phases[4..8]
            .iter()
            .all(|node| node.phase == StreamPhase::Scan));
        assert!(schedule.phases[8..12]
            .iter()
            .all(|node| node.phase == StreamPhase::Resize));
        assert!(schedule.phases[12..16]
            .iter()
            .all(|node| node.phase == StreamPhase::Materialize));
        let stream_slots: Vec<_> = schedule.phases[0..4]
            .iter()
            .map(|node| node.stream_index)
            .collect();
        assert_eq!(stream_slots, vec![0, 1, 2, 3]);
    }

    #[test]
    fn single_rule_uses_one_stream() {
        let schedule = schedule_streams(&stratum(), &HardwareCapabilities::new(16, 1));
        assert_eq!(schedule.stream_count, 1);
        assert_eq!(schedule.phases.len(), 4);
        assert!(schedule.phases.iter().all(|node| node.stream_index == 0));
    }
}
