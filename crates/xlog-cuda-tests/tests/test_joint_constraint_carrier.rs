//! Joint constraint carrier — slice 1: buffer ownership + registration.
//!
//! RED-first against the frozen carrier law: every solver buffer is
//! allocated by the xlog device runtime and exported outward, never
//! imported from an external DLPack producer. Strict launch recorders
//! must therefore record every carrier column (runtime block present),
//! and an externally-owned column offered as a carrier buffer is a
//! typed refusal, not a silent downgrade.

use std::sync::Arc;

use xlog_cuda::{CarrierError, CudaDevice, JointConstraintCarrier};

/// Domain/score buffer allocation is runtime-backed: every column the
/// carrier exposes carries a device-runtime block, which is exactly
/// what strict-mode launch recorders require to record (not reject)
/// the launch.
#[test]
fn carrier_buffers_are_runtime_backed_and_recordable() {
    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let carrier = JointConstraintCarrier::allocate(
        Arc::clone(&device),
        /* entities (sort-domain variables) */ 16,
        /* sort universe width (domain bitset lanes) */ 64,
        /* relation candidates (label/score rows) */ 32,
        /* relation label universe (incl. abstain) */ 8,
    )
    .expect("runtime-backed allocation must succeed");

    // Ownership law: every exposed column is xlog-owned — a runtime
    // block exists, so strict recorders record instead of rejecting.
    for column in carrier.columns() {
        assert!(
            column.runtime_block().is_some(),
            "carrier column without a runtime block would be rejected \
             by strict launch recorders"
        );
        assert!(
            !column.is_external(),
            "carrier columns must never be externally-managed memory"
        );
    }
}

/// Registration is once-per-session: registering the same schema a
/// second time is a typed refusal (duplicate registration must never
/// silently rebind live buffers).
#[test]
fn duplicate_schema_registration_refuses_typed() {
    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // Deterministic test identity: sha256 of the empty string — a
    // valid, reproducible catalog anchor with no placeholder text.
    const CATALOG_SHA: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    let mut carrier =
        JointConstraintCarrier::allocate(Arc::clone(&device), 4, 8, 4, 4).expect("allocation");
    carrier
        .register_schema(CATALOG_SHA, xlog_cuda::SOLVER_ABI_IDENTITY)
        .expect("first registration succeeds");
    let err = carrier
        .register_schema(CATALOG_SHA, xlog_cuda::SOLVER_ABI_IDENTITY)
        .expect_err("second registration must refuse");
    // Typed refusal: the concrete variant, never a string match — a
    // string-contains assertion cannot distinguish a typed error
    // from an arbitrary message.
    assert!(
        matches!(err, CarrierError::SchemaAlreadyRegistered { .. }),
        "duplicate registration must refuse with the typed variant, got: {err:?}"
    );
}

/// Capacity-overflow mutant: a request beyond the carrier budget is
/// a typed refusal from the budget guard itself — never a driver
/// crash, never a partially-built carrier. The case is sized so the
/// first buffer succeeds and a later one overflows, exercising the
/// partial-allocation unwind, and the inner variant is matched
/// exactly: a raw driver failure surfaces as `XlogError::Kernel`,
/// so only the budget guard produces `ResourceExhausted`.
#[test]
fn over_budget_allocation_refuses_typed_and_leaves_device_clean() {
    use xlog_core::XlogError;

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // domains = 1 x 1 lanes (8 bytes, succeeds); scores =
    // 9_000_000 candidates x 2 labels x 4 bytes = 72 MB > the
    // 64 MiB carrier budget — the second allocation must refuse.
    let err = match JointConstraintCarrier::allocate(Arc::clone(&device), 1, 1, 9_000_000, 2) {
        Ok(_) => panic!("over-budget capacity must refuse"),
        Err(err) => err,
    };
    assert!(
        matches!(
            &err,
            CarrierError::Allocation(XlogError::ResourceExhausted { .. })
        ),
        "refusal must come from the typed budget guard, got: {err:?}"
    );

    // The refusal is clean: a lawful capacity on the same device
    // still allocates — no poisoned state, no stranded budget.
    let carrier = JointConstraintCarrier::allocate(Arc::clone(&device), 16, 64, 32, 8)
        .expect("lawful allocation after a refusal must succeed");
    assert!(carrier.columns().all(|c| c.runtime_block().is_some()));
}

/// A zero capacity dimension is a typed refusal, not a silent clamp:
/// a carrier with no entities, lanes, candidates, or labels cannot
/// participate in a solve, and clamping would hide the caller's bug.
#[test]
fn zero_capacity_dimension_refuses_typed() {
    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    for (entities, domain_lanes, candidates, labels, dimension) in [
        (0, 8, 4, 4, "entities"),
        (4, 0, 4, 4, "domain_lanes"),
        (4, 8, 0, 4, "candidates"),
        (4, 8, 4, 0, "labels"),
    ] {
        let err = match JointConstraintCarrier::allocate(
            Arc::clone(&device),
            entities,
            domain_lanes,
            candidates,
            labels,
        ) {
            Ok(_) => panic!("zero {dimension} must refuse"),
            Err(err) => err,
        };
        assert!(
            matches!(&err, CarrierError::ZeroCapacity { dimension: d } if *d == dimension),
            "zero {dimension} must refuse with the named dimension, got: {err:?}"
        );
    }
}

/// Lifetime/drop mutant: dropping the carrier returns every device
/// byte it owned. Four allocate/drop cycles of a near-budget
/// carrier (~56 MB each) would strand ~225 MB if drop leaked —
/// far outside the measurement tolerance — so free device memory
/// after the cycles must recover to the pre-cycle level.
#[test]
fn dropping_carrier_releases_all_device_memory() {
    use cudarc::driver::result::mem_get_info;

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let allocate_near_budget = |device: &Arc<CudaDevice>| {
        // domains 1024 x 64 lanes = 512 KB; scores 1.1M x 4 = 17.6 MB;
        // constraints 1.1M x 2 = 8.8 MB; outputs 1.1M = 4.4 MB;
        // feasible sets 1.1M x 1 word = 8.8 MB; map results
        // 1.1M x 4 = 17.6 MB; ~57.7 MB total, inside the 64 MiB
        // budget.
        JointConstraintCarrier::allocate(Arc::clone(device), 1024, 64, 1_100_000, 4)
            .expect("near-budget allocation must succeed")
    };

    // Warmup cycle absorbs one-time driver/pool initialization so
    // the measured delta reflects carrier buffers only.
    drop(allocate_near_budget(&device));

    let (free_before, _total) = mem_get_info().expect("cudaMemGetInfo before cycles");
    for _ in 0..4 {
        let carrier = allocate_near_budget(&device);
        assert!(carrier.columns().all(|c| !c.is_external()));
        drop(carrier);
    }
    let (free_after, _total) = mem_get_info().expect("cudaMemGetInfo after cycles");

    // Leak-direction bound only: concurrent tests may free memory
    // (raising `free_after`), but a drop leak of ~225 MB cannot
    // hide inside a 32 MiB tolerance.
    const TOLERANCE_BYTES: usize = 32 * 1024 * 1024;
    assert!(
        free_after + TOLERANCE_BYTES >= free_before,
        "carrier drop leaked device memory: free before {free_before}, after {free_after}"
    );
}

/// The device solve stage implements the existential legality law:
/// a label stays feasible while SOME sort in each entity's domain
/// intersects the label's signature side — a plural (non-singleton)
/// domain must keep the label feasible, and the abstention label is
/// feasible unconditionally. The launch goes through a strict
/// recorder on a pool stream with event preflight/commit; results
/// stay device-resident and are read back here only as test
/// scaffolding.
#[test]
fn solve_label_feasibility_is_existential_on_device() {
    use xlog_cuda::{FuelMeter, SOLVER_ABI_IDENTITY};

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // 2 entities, 1 domain lane, 1 candidate pair, 3 labels
    // (label 0 = abstain).
    let mut carrier =
        JointConstraintCarrier::allocate(Arc::clone(&device), 2, 1, 1, 3).expect("allocation");
    carrier
        .register_schema(
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            SOLVER_ABI_IDENTITY,
        )
        .expect("registration");

    // Head signatures per label (1 lane each): label 1 accepts sort 2,
    // label 2 accepts sort 0. Tail signatures: both accept sort 3.
    carrier
        .bind_signatures(&[0, 0b0100, 0b0001], &[0, 0b1000, 0b1000])
        .expect("signature binding");

    // Entity 0 domain is PLURAL {sort 1, sort 2}; entity 1 is {sort 3}.
    // Label 1 must stay feasible through the plural domain (sort 2
    // intersects); label 2 must be infeasible (sort 0 does not).
    let columns: Vec<&xlog_cuda::CudaColumn> = carrier.columns().collect();
    let (domains_ptr, constraints_ptr, outputs_ptr, sets_ptr) = (
        *columns[0].device_ptr(),
        *columns[2].device_ptr(),
        *columns[3].device_ptr(),
        *columns[4].device_ptr(),
    );
    unsafe {
        cudarc::driver::result::memcpy_htod_sync(domains_ptr, &[0b0110u64, 0b1000u64])
            .expect("domain upload");
        cudarc::driver::result::memcpy_htod_sync(constraints_ptr, &[0u32, 1u32])
            .expect("pair upload");
    }
    device.inner().synchronize().expect("scaffold sync");

    let mut fuel = FuelMeter::new(1 << 22);
    carrier
        .solve_label_feasibility(0, &mut fuel)
        .expect("solve through strict recorder");
    device.inner().synchronize().expect("post-solve sync");

    let mut counts = [0u32; 1];
    let mut sets = [0u64; 1];
    unsafe {
        cudarc::driver::result::memcpy_dtoh_sync(&mut counts, outputs_ptr)
            .expect("counts readback");
        cudarc::driver::result::memcpy_dtoh_sync(&mut sets, sets_ptr).expect("sets readback");
    }
    assert_eq!(
        sets[0], 0b011,
        "abstain (0) and the plural-domain label (1) must be feasible; \
         the empty-intersection label (2) must not"
    );
    assert_eq!(counts[0], 2, "feasible count must match the set bits");
    assert_eq!(fuel.spent(), 3, "one expansion per (candidate, label) cell");
}

/// Every solve prerequisite refuses typed in order: no registration,
/// no signatures, wrong mask shape, rebinding, and an abstain index
/// outside the label universe. No path silently proceeds.
#[test]
fn solve_prerequisites_refuse_typed_in_order() {
    use xlog_cuda::{FuelMeter, SOLVER_ABI_IDENTITY};

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let mut carrier =
        JointConstraintCarrier::allocate(Arc::clone(&device), 2, 1, 1, 3).expect("allocation");
    let mut fuel = FuelMeter::new(1 << 10);

    let err = carrier.solve_label_feasibility(0, &mut fuel).unwrap_err();
    assert!(matches!(err, CarrierError::SchemaNotRegistered), "{err:?}");
    let err = carrier.bind_signatures(&[0; 3], &[0; 3]).unwrap_err();
    assert!(matches!(err, CarrierError::SchemaNotRegistered), "{err:?}");

    carrier
        .register_schema("00", SOLVER_ABI_IDENTITY)
        .expect("registration");
    let err = carrier.solve_label_feasibility(0, &mut fuel).unwrap_err();
    assert!(matches!(err, CarrierError::SignaturesUnbound), "{err:?}");

    let err = carrier.bind_signatures(&[0; 2], &[0; 3]).unwrap_err();
    assert!(
        matches!(
            err,
            CarrierError::SignatureShapeMismatch {
                side: "head",
                expected_words: 3,
                got_words: 2,
            }
        ),
        "{err:?}"
    );

    carrier
        .bind_signatures(&[0; 3], &[0; 3])
        .expect("lawful binding");
    let err = carrier.bind_signatures(&[0; 3], &[0; 3]).unwrap_err();
    assert!(
        matches!(err, CarrierError::SignaturesAlreadyBound),
        "{err:?}"
    );

    let err = carrier.solve_label_feasibility(3, &mut fuel).unwrap_err();
    assert!(
        matches!(
            err,
            CarrierError::AbstainOutOfRange {
                abstain_label: 3,
                labels: 3,
            }
        ),
        "{err:?}"
    );
}

/// Fuel exhaustion refuses BEFORE the launch: the typed refusal
/// carries the exact literals and the output buffers stay untouched
/// — no partial emission of any kind.
#[test]
fn solve_beyond_fuel_refuses_without_partial_emission() {
    use xlog_cuda::{FuelMeter, SolverError, SOLVER_ABI_IDENTITY};

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let mut carrier =
        JointConstraintCarrier::allocate(Arc::clone(&device), 2, 1, 1, 3).expect("allocation");
    carrier
        .register_schema("00", SOLVER_ABI_IDENTITY)
        .expect("registration");
    carrier
        .bind_signatures(&[0, !0u64, !0u64], &[0, !0u64, !0u64])
        .expect("binding");

    // Pin the output cells to a known sentinel so "untouched" is
    // provable (fresh device memory has no guaranteed contents).
    let columns: Vec<&xlog_cuda::CudaColumn> = carrier.columns().collect();
    let (outputs_ptr, sets_ptr) = (*columns[3].device_ptr(), *columns[4].device_ptr());
    unsafe {
        cudarc::driver::result::memcpy_htod_sync(outputs_ptr, &[0xDEAD_BEEFu32])
            .expect("sentinel upload");
        cudarc::driver::result::memcpy_htod_sync(sets_ptr, &[0xFEED_FACE_CAFE_D00Du64])
            .expect("sentinel upload");
    }
    device.inner().synchronize().expect("scaffold sync");

    // Budget of 2 expansions cannot cover 1 candidate x 3 labels.
    let mut fuel = FuelMeter::new(2);
    let err = carrier.solve_label_feasibility(0, &mut fuel).unwrap_err();
    assert!(
        matches!(
            err,
            CarrierError::Solver(SolverError::ResourceExhausted {
                fuel_spent: 0,
                fuel_limit: 2,
            })
        ),
        "{err:?}"
    );

    let mut counts = [0u32; 1];
    let mut sets = [0u64; 1];
    unsafe {
        cudarc::driver::result::memcpy_dtoh_sync(&mut counts, outputs_ptr)
            .expect("counts readback");
        cudarc::driver::result::memcpy_dtoh_sync(&mut sets, sets_ptr).expect("sets readback");
    }
    assert_eq!(
        counts[0], 0xDEAD_BEEF,
        "refused solve must not touch outputs"
    );
    assert_eq!(
        sets[0], 0xFEED_FACE_CAFE_D00D,
        "refused solve must not touch feasible sets"
    );
}

/// The top-two stage consumes the feasibility stage's output across
/// launch streams with NO host synchronization between the two
/// solves — ordering rides entirely on the recorded launch events.
/// Exactness: the best label is taken over FEASIBLE labels only (an
/// infeasible label with the highest raw score must be ignored), the
/// margin is best minus runner-up, and fuel accounts both stages.
#[test]
fn top2_consumes_feasibility_across_streams_exactly() {
    use xlog_cuda::{FuelMeter, SOLVER_ABI_IDENTITY};

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let mut carrier =
        JointConstraintCarrier::allocate(Arc::clone(&device), 2, 1, 1, 3).expect("allocation");
    carrier
        .register_schema("00", SOLVER_ABI_IDENTITY)
        .expect("registration");
    // Same masks as the existential test: label 1 feasible for the
    // pair, label 2 infeasible, label 0 = abstain.
    carrier
        .bind_signatures(&[0, 0b0100, 0b0001], &[0, 0b1000, 0b1000])
        .expect("binding");

    let (domains_ptr, scores_ptr, constraints_ptr, map_ptr) = {
        let columns: Vec<&xlog_cuda::CudaColumn> = carrier.columns().collect();
        (
            *columns[0].device_ptr(),
            *columns[1].device_ptr(),
            *columns[2].device_ptr(),
            *columns[5].device_ptr(),
        )
    };
    unsafe {
        cudarc::driver::result::memcpy_htod_sync(domains_ptr, &[0b0110u64, 0b1000u64])
            .expect("domain upload");
        cudarc::driver::result::memcpy_htod_sync(constraints_ptr, &[0u32, 1u32])
            .expect("pair upload");
        // The INFEASIBLE label 2 carries the highest raw score: a
        // mutant ignoring the feasible sets would select it.
        cudarc::driver::result::memcpy_htod_sync(scores_ptr, &[0.5f32, 2.0f32, 9.0f32])
            .expect("scores upload");
    }
    device.inner().synchronize().expect("scaffold sync");

    let mut fuel = FuelMeter::new(1 << 22);
    carrier
        .solve_label_feasibility(0, &mut fuel)
        .expect("feasibility stage");
    // Deliberately NO host synchronization here: the top-two launch
    // must order after the feasibility launch via recorded events.
    carrier
        .solve_label_map_top2(&mut fuel)
        .expect("top-two stage");
    device.inner().synchronize().expect("post-solve sync");

    let mut map = [0u32; 4];
    unsafe {
        cudarc::driver::result::memcpy_dtoh_sync(&mut map, map_ptr).expect("map readback");
    }
    assert_eq!(
        map[0], 1,
        "best label is the FEASIBLE maximum, not the raw one"
    );
    assert_eq!(map[1], 0, "unique maximum carries no ambiguity flag");
    assert_eq!(f32::from_bits(map[2]), 2.0, "best score");
    assert_eq!(f32::from_bits(map[3]), 1.5, "margin = best minus runner-up");
    assert_eq!(
        fuel.spent(),
        6,
        "both stages charge candidate x label cells"
    );
}

/// A tied maximum is a typed MAP-ambiguity signal: the ambiguity
/// flag is set and the margin is zero. A unique-label emission from
/// a tied maximum is exactly the ID-tie-break the law prohibits.
#[test]
fn tied_maximum_flags_ambiguity_never_unique() {
    use xlog_cuda::{FuelMeter, SOLVER_ABI_IDENTITY};

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let mut carrier =
        JointConstraintCarrier::allocate(Arc::clone(&device), 2, 1, 1, 3).expect("allocation");
    carrier
        .register_schema("00", SOLVER_ABI_IDENTITY)
        .expect("registration");
    carrier
        .bind_signatures(&[0, 0b0100, 0b0001], &[0, 0b1000, 0b1000])
        .expect("binding");

    let (domains_ptr, scores_ptr, constraints_ptr, map_ptr) = {
        let columns: Vec<&xlog_cuda::CudaColumn> = carrier.columns().collect();
        (
            *columns[0].device_ptr(),
            *columns[1].device_ptr(),
            *columns[2].device_ptr(),
            *columns[5].device_ptr(),
        )
    };
    unsafe {
        cudarc::driver::result::memcpy_htod_sync(domains_ptr, &[0b0110u64, 0b1000u64])
            .expect("domain upload");
        cudarc::driver::result::memcpy_htod_sync(constraints_ptr, &[0u32, 1u32])
            .expect("pair upload");
        // Abstain and label 1 tie at the maximum over feasible labels.
        cudarc::driver::result::memcpy_htod_sync(scores_ptr, &[2.0f32, 2.0f32, 9.0f32])
            .expect("scores upload");
    }
    device.inner().synchronize().expect("scaffold sync");

    let mut fuel = FuelMeter::new(1 << 22);
    carrier
        .solve_label_feasibility(0, &mut fuel)
        .expect("feasibility stage");
    carrier
        .solve_label_map_top2(&mut fuel)
        .expect("top-two stage");
    device.inner().synchronize().expect("post-solve sync");

    let mut map = [0u32; 4];
    unsafe {
        cudarc::driver::result::memcpy_dtoh_sync(&mut map, map_ptr).expect("map readback");
    }
    assert_eq!(map[1], 1, "tied maximum must set the ambiguity flag");
    assert_eq!(f32::from_bits(map[3]), 0.0, "tied maximum has zero margin");
}

/// A fresh carrier session is deterministically EMPTY: every buffer
/// reads back zero before any solve. Without this, reused device
/// memory could leak a previous session's bytes into a new one — and
/// garbage in the solve-status column could accidentally read as a
/// claimed authority.
#[test]
fn fresh_session_buffers_read_back_zero() {
    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // Dirty the allocator with a session, then drop it.
    drop(JointConstraintCarrier::allocate(Arc::clone(&device), 2, 1, 2, 3).expect("first"));
    let carrier = JointConstraintCarrier::allocate(Arc::clone(&device), 2, 1, 2, 3)
        .expect("second allocation reusing device memory");

    let columns: Vec<&xlog_cuda::CudaColumn> = carrier.columns().collect();
    let (outputs_ptr, map_ptr, status_ptr) = (
        *columns[3].device_ptr(),
        *columns[5].device_ptr(),
        *columns[6].device_ptr(),
    );
    let mut outputs = [0xAAu32; 2];
    let mut maps = [0xAAu32; 8];
    let mut status = [0xAAu32; 2];
    unsafe {
        cudarc::driver::result::memcpy_dtoh_sync(&mut outputs, outputs_ptr)
            .expect("outputs readback");
        cudarc::driver::result::memcpy_dtoh_sync(&mut maps, map_ptr).expect("map readback");
        cudarc::driver::result::memcpy_dtoh_sync(&mut status, status_ptr).expect("status readback");
    }
    assert_eq!(
        outputs, [0; 2],
        "fresh outputs are zero, never reused bytes"
    );
    assert_eq!(maps, [0; 8], "fresh map results are zero");
    assert_eq!(
        status, [0; 2],
        "fresh solve status is zero — no accidental authority claim"
    );
}

/// Exact component solve: two candidates share an entity, and each
/// candidate's individually-best label constrains that entity to a
/// DIFFERENT sort — the greedy per-candidate maximum is jointly
/// infeasible. Complete enumeration must select the consistent
/// optimum (one candidate abstains), overwrite the top-two rows with
/// joint-exact values and margins, and mark both rows
/// component-exact. A per-candidate argmax substitute dies here.
#[test]
fn component_solve_rejects_jointly_infeasible_greedy_maximum() {
    use xlog_cuda::{candidate_components, FuelMeter, SOLVER_ABI_IDENTITY};

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // 3 entities, 2 candidates c0=(e0,e1), c1=(e1,e2), 3 labels
    // (0 = abstain). Label 1 pins its tail to sort 2; label 2 pins
    // its head to sort 3 — both constrain shared entity e1, with
    // empty intersection.
    let mut carrier =
        JointConstraintCarrier::allocate(Arc::clone(&device), 3, 1, 2, 3).expect("allocation");
    carrier
        .register_schema("00", SOLVER_ABI_IDENTITY)
        .expect("registration");
    carrier
        .bind_signatures(&[0, 0b0010, 0b1000], &[0, 0b0100, 0b0001])
        .expect("binding");

    let (domains_ptr, scores_ptr, constraints_ptr, map_ptr, status_ptr) = {
        let columns: Vec<&xlog_cuda::CudaColumn> = carrier.columns().collect();
        (
            *columns[0].device_ptr(),
            *columns[1].device_ptr(),
            *columns[2].device_ptr(),
            *columns[5].device_ptr(),
            *columns[6].device_ptr(),
        )
    };
    let pairs = [(0u32, 1u32), (1u32, 2u32)];
    unsafe {
        cudarc::driver::result::memcpy_htod_sync(domains_ptr, &[0b0010u64, 0b1100u64, 0b0001u64])
            .expect("domain upload");
        cudarc::driver::result::memcpy_htod_sync(constraints_ptr, &[0u32, 1u32, 1u32, 2u32])
            .expect("pair upload");
        // Greedy favorites: c0 -> label 1 (5.0), c1 -> label 2 (4.0);
        // infeasible labels carry decoy-high scores.
        cudarc::driver::result::memcpy_htod_sync(scores_ptr, &[0.1f32, 5.0, 9.9, 0.1, 7.7, 4.0])
            .expect("scores upload");
    }
    device.inner().synchronize().expect("scaffold sync");

    let mut fuel = FuelMeter::new(1 << 22);
    carrier
        .solve_label_feasibility(0, &mut fuel)
        .expect("feasibility stage");
    carrier
        .solve_label_map_top2(&mut fuel)
        .expect("top-two stage");
    let (offsets, indices) = candidate_components(3, &pairs);
    assert_eq!(offsets, vec![0, 2], "both candidates share one component");
    carrier
        .solve_components_exact(&offsets, &indices, &mut fuel)
        .expect("component stage");
    device.inner().synchronize().expect("post-solve sync");

    let mut maps = [0u32; 8];
    let mut status = [0u32; 2];
    unsafe {
        cudarc::driver::result::memcpy_dtoh_sync(&mut maps, map_ptr).expect("map readback");
        cudarc::driver::result::memcpy_dtoh_sync(&mut status, status_ptr).expect("status readback");
    }

    // Mirror the kernel's f32 accumulation order exactly.
    let best_total = 5.0f32 + 0.1f32;
    let alt_total = 0.1f32 + 4.0f32;
    let margin = best_total - alt_total;

    assert_eq!(maps[0], 1, "c0 keeps its label in the consistent optimum");
    assert_eq!(maps[1], 0, "unique joint optimum");
    assert_eq!(f32::from_bits(maps[2]), best_total, "joint MAP total");
    assert_eq!(f32::from_bits(maps[3]), margin, "c0 per-edge global margin");
    assert_eq!(
        maps[4], 0,
        "c1 must ABSTAIN in the joint optimum although its greedy \
         favorite scores higher — the greedy pair is jointly infeasible"
    );
    assert_eq!(f32::from_bits(maps[6]), best_total, "same joint total");
    assert_eq!(f32::from_bits(maps[7]), margin, "c1 per-edge global margin");
    assert_eq!(status, [2, 2], "both rows are component-exact");
    // Device-exact fuel: feasibility 6 + top-two 6 + exactly the 4
    // enumerated combinations — the conservative upfront
    // authorization was refunded down to the measured literal.
    assert_eq!(
        fuel.spent(),
        16,
        "meter reconciles to device-measured expansions"
    );
}

/// Producer-event seam: writes enqueued ASYNCHRONOUSLY on an
/// external stream order against the solve purely through
/// `note_producer_stream` — no host synchronization between the
/// producer write and the solve stages. This is the device-side
/// ordering that replaces the caller's `synchronize()` barrier and
/// keeps the measured region free of host interactions. A null
/// stream handle refuses typed.
#[test]
fn noted_producer_stream_orders_async_writes_before_solve() {
    use xlog_cuda::{FuelMeter, SOLVER_ABI_IDENTITY};

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let mut carrier =
        JointConstraintCarrier::allocate(Arc::clone(&device), 2, 1, 1, 3).expect("allocation");
    carrier
        .register_schema("00", SOLVER_ABI_IDENTITY)
        .expect("registration");
    carrier
        .bind_signatures(&[0, 0b0100, 0b0001], &[0, 0b1000, 0b1000])
        .expect("binding");

    let err = carrier.note_producer_stream(0).unwrap_err();
    assert!(
        matches!(err, CarrierError::Launch(_)),
        "null producer stream must refuse typed, got: {err:?}"
    );

    let (domains_ptr, scores_ptr, constraints_ptr, map_ptr) = {
        let columns: Vec<&xlog_cuda::CudaColumn> = carrier.columns().collect();
        (
            *columns[0].device_ptr(),
            *columns[1].device_ptr(),
            *columns[2].device_ptr(),
            *columns[5].device_ptr(),
        )
    };

    // Stand-in external producer stream (same device, not the solve
    // stream): all input writes go on it ASYNCHRONOUSLY.
    let producer_pool = Arc::new(xlog_cuda::device_runtime::StreamPool::with_defaults(
        Arc::clone(&device),
    ));
    let producer_id = producer_pool.acquire().expect("acquire producer stream");
    let producer_stream = producer_pool
        .resolve(producer_id)
        .expect("resolve producer stream");
    let domains_host = [0b0110u64, 0b1000u64];
    let pairs_host = [0u32, 1u32];
    let scores_host = [0.5f32, 2.0f32, 9.0f32];
    unsafe {
        cudarc::driver::result::memcpy_htod_async(
            domains_ptr,
            &domains_host,
            producer_stream.cu_stream(),
        )
        .expect("async domain write");
        cudarc::driver::result::memcpy_htod_async(
            constraints_ptr,
            &pairs_host,
            producer_stream.cu_stream(),
        )
        .expect("async pair write");
        cudarc::driver::result::memcpy_htod_async(
            scores_ptr,
            &scores_host,
            producer_stream.cu_stream(),
        )
        .expect("async scores write");
    }
    carrier
        .note_producer_stream(producer_stream.cu_stream() as u64)
        .expect("note producer stream");

    // NO host synchronization between the producer writes and the
    // solve stages: ordering must ride on the noted event.
    let mut fuel = FuelMeter::new(1 << 22);
    carrier
        .solve_label_feasibility(0, &mut fuel)
        .expect("feasibility after noted producer");
    carrier
        .solve_label_map_top2(&mut fuel)
        .expect("top-two after noted producer");
    device.inner().synchronize().expect("post-solve sync");

    let mut map = [0u32; 4];
    unsafe {
        cudarc::driver::result::memcpy_dtoh_sync(&mut map, map_ptr).expect("map readback");
    }
    assert_eq!(
        map[0], 1,
        "event-ordered producer data yields the exact MAP label"
    );
    assert_eq!(
        f32::from_bits(map[3]),
        1.5,
        "exact margin through the event seam"
    );
}

/// A corrupt producer record — a pair index outside the entity
/// capacity — poisons its candidate row (count 0xFFFFFFFF, empty
/// feasible set, poisoned map row) instead of reading out-of-bounds
/// device memory. Healthy rows in the same batch stay unaffected.
#[test]
fn corrupt_pair_index_poisons_its_row_only() {
    use xlog_cuda::{FuelMeter, SOLVER_ABI_IDENTITY};

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // 2 candidates: row 0 healthy, row 1 references entity 7 of 2.
    let mut carrier =
        JointConstraintCarrier::allocate(Arc::clone(&device), 2, 1, 2, 3).expect("allocation");
    carrier
        .register_schema("00", SOLVER_ABI_IDENTITY)
        .expect("registration");
    carrier
        .bind_signatures(&[0, 0b0100, 0b0001], &[0, 0b1000, 0b1000])
        .expect("binding");

    let (domains_ptr, scores_ptr, constraints_ptr, outputs_ptr, map_ptr) = {
        let columns: Vec<&xlog_cuda::CudaColumn> = carrier.columns().collect();
        (
            *columns[0].device_ptr(),
            *columns[1].device_ptr(),
            *columns[2].device_ptr(),
            *columns[3].device_ptr(),
            *columns[5].device_ptr(),
        )
    };
    unsafe {
        cudarc::driver::result::memcpy_htod_sync(domains_ptr, &[0b0110u64, 0b1000u64])
            .expect("domain upload");
        cudarc::driver::result::memcpy_htod_sync(constraints_ptr, &[0u32, 1u32, 7u32, 1u32])
            .expect("pair upload");
        cudarc::driver::result::memcpy_htod_sync(
            scores_ptr,
            &[0.5f32, 2.0f32, 9.0f32, 0.5f32, 2.0f32, 9.0f32],
        )
        .expect("scores upload");
    }
    device.inner().synchronize().expect("scaffold sync");

    let mut fuel = FuelMeter::new(1 << 22);
    carrier
        .solve_label_feasibility(0, &mut fuel)
        .expect("feasibility stage");
    carrier
        .solve_label_map_top2(&mut fuel)
        .expect("top-two stage");
    device.inner().synchronize().expect("post-solve sync");

    let mut counts = [0u32; 2];
    let mut maps = [0u32; 8];
    unsafe {
        cudarc::driver::result::memcpy_dtoh_sync(&mut counts, outputs_ptr)
            .expect("counts readback");
        cudarc::driver::result::memcpy_dtoh_sync(&mut maps, map_ptr).expect("map readback");
    }
    assert_eq!(
        counts[0], 2,
        "healthy row keeps its exact feasibility count"
    );
    assert_eq!(
        counts[1], 0xFFFF_FFFF,
        "corrupt row carries the poison count"
    );
    assert_eq!(maps[0], 1, "healthy row keeps its exact MAP label");
    assert_eq!(
        (maps[4], maps[5]),
        (0xFFFF_FFFF, 1),
        "corrupt row's map result is poisoned, never garbage"
    );
}

/// The top-two stage refuses typed when the feasibility stage has
/// not populated the feasible sets it consumes.
#[test]
fn top2_before_feasibility_refuses_typed() {
    use xlog_cuda::{FuelMeter, SOLVER_ABI_IDENTITY};

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let mut carrier =
        JointConstraintCarrier::allocate(Arc::clone(&device), 2, 1, 1, 3).expect("allocation");
    carrier
        .register_schema("00", SOLVER_ABI_IDENTITY)
        .expect("registration");
    let mut fuel = FuelMeter::new(1 << 10);
    let err = carrier.solve_label_map_top2(&mut fuel).unwrap_err();
    assert!(matches!(err, CarrierError::FeasibilityNotSolved), "{err:?}");
    assert_eq!(fuel.spent(), 0, "a refused stage never charges fuel");
}

/// Outward export keeps xlog ownership: the exported slice is the
/// EXACT allocation the carrier solves on (same device pointer), and
/// a DLPack wrap of it preserves the recorder-acceptable ownership
/// signature — non-external with a resolvable runtime block. This is
/// the xlog side of the binding path: the capsule layer only adds
/// metadata, never a new owner.
#[test]
fn export_shares_allocation_identity_and_stays_recorder_acceptable() {
    use xlog_cuda::{CarrierBufferId, CudaColumn, DlpackManagedTensor};

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let carrier =
        JointConstraintCarrier::allocate(Arc::clone(&device), 2, 1, 1, 3).expect("allocation");

    use cudarc::driver::DevicePtr;
    let columns: Vec<&CudaColumn> = carrier.columns().collect();
    for (id, column_index) in [
        (CarrierBufferId::Domains, 0),
        (CarrierBufferId::Scores, 1),
        (CarrierBufferId::Constraints, 2),
        (CarrierBufferId::Outputs, 3),
        (CarrierBufferId::FeasibleSets, 4),
        (CarrierBufferId::MapResults, 5),
    ] {
        let export = carrier.export_buffer(id);
        assert_eq!(
            *export.slice.device_ptr(),
            *columns[column_index].device_ptr(),
            "{id:?}: export must share the solve column's allocation"
        );
        assert_eq!(
            export.slice.len(),
            export.rows * export.cols * export.elem_bytes,
            "{id:?}: export shape must cover the allocation exactly"
        );

        let tensor = unsafe { DlpackManagedTensor::from_raw(std::ptr::null_mut()) };
        let wrapped =
            CudaColumn::dlpack_xlog_owned(Arc::clone(&export.slice), export.stream.clone(), tensor);
        assert!(
            !wrapped.is_external(),
            "{id:?}: export wrap stays xlog-owned"
        );
        assert!(
            wrapped.runtime_block().is_some(),
            "{id:?}: export wrap keeps the runtime block — strict recorders record it"
        );
    }

    // The device-resident logical-counts buffer is part of the export
    // surface even though it is not a solve column.
    let counts = carrier.export_buffer(CarrierBufferId::LogicalCounts);
    assert_eq!(counts.slice.len(), 4 * 4, "4 u32 logical-state words");
    assert!(counts.slice.runtime_block().is_some());
}

/// Anti-vacuity mutant: the ownership predicate the carrier tests
/// assert must actually DISCRIMINATE. An externally-owned
/// DLPack column shows the exact inverse signature — external, with
/// no runtime block — so a strict recorder rejects it while every
/// carrier column records. If this test fails, the carrier
/// assertions above are vacuous.
#[test]
fn external_column_shows_inverse_ownership_signature() {
    use xlog_cuda::DlpackManagedTensor;

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    // SAFETY: null-pointer tensor is drop-safe (DlpackManagedTensor's
    // Drop impl null-checks before invoking the deleter); nothing
    // dereferences the tensor here.
    let tensor = unsafe { DlpackManagedTensor::from_raw(std::ptr::null_mut()) };
    let column = xlog_cuda::CudaColumn::dlpack(0, 0, device.inner().stream().clone(), tensor);

    assert!(column.is_external(), "raw DLPack column must be external");
    assert!(
        column.runtime_block().is_none(),
        "external column must have no runtime block — strict recorders reject it"
    );
}
