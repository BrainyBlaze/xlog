use std::sync::Arc;
use xlog_core::MemoryBudget;
use xlog_cuda::{
    CudaProviderBuilder, SemanticActionCatalogue, SemanticComponentInput, SemanticRngBinding,
    SemanticTransitionError, SemanticTransitionSession,
};

#[test]
fn feedback_coordinates_use_the_cuda_producers_shared_encoder() {
    // Compile and execute the exact host/device coordinate helper used by the
    // production kernel. This is CPU evidence, not a CUDA launch or lease test.
    let mut nonce = [0u8; 8];
    getrandom::fill(&mut nonce).unwrap();
    let directory = std::env::temp_dir().join(format!(
        "xlog-feedback-{}-{:016x}",
        std::process::id(),
        u64::from_ne_bytes(nonce)
    ));
    std::fs::create_dir(&directory).unwrap();
    struct Output(std::path::PathBuf);
    impl Drop for Output {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(self.0.join("encoder-test"));
            let _ = std::fs::remove_dir(&self.0);
        }
    }
    let output = Output(directory);
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let compile =
        std::process::Command::new(std::env::var_os("CXX").unwrap_or_else(|| "c++".into()))
            .args(["-std=c++17", "-Wall", "-Wextra", "-Werror", "-I"])
            .arg(root.join("kernels"))
            .arg(root.join("tests/semantic_feedback_encoding.cpp"))
            .arg("-o")
            .arg(output.0.join("encoder-test"))
            .output()
            .expect("C++ compiler required for the native coordinate contract");
    assert!(
        compile.status.success(),
        "{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let execution = std::process::Command::new(output.0.join("encoder-test"))
        .output()
        .unwrap();
    assert!(
        execution.status.success(),
        "{}",
        String::from_utf8_lossy(&execution.stderr)
    );
}

#[cfg(feature = "semantic-policy")]
#[test]
#[ignore = "requires an explicitly authorized remote CUDA run"]
fn device_policy_snapshots_inputs_and_computes_late_text_vjp() {
    let provider = Arc::new(
        CudaProviderBuilder::new(0, MemoryBudget::with_limit(512 * 1024 * 1024))
            .with_stream_capacity(2)
            .build()
            .unwrap(),
    );
    let runtime = Arc::clone(provider.memory().runtime().unwrap());
    let id = runtime.stream_pool().acquire().unwrap();
    let stream = runtime.stream_pool().resolve(id).unwrap();
    let domain = provider
        .bind_resident_execution_domain(runtime, id, stream)
        .unwrap();
    let mut session = SemanticTransitionSession::new(&provider, &domain).unwrap();
    let layout = session.policy_layout().unwrap();
    let text_cells = 64 * SemanticActionCatalogue::current().text_cardinality();
    let support: Vec<u8> = session
        .components()
        .iter()
        .flat_map(|c| (0..c.cardinality).map(move |i| u8::from(i == 0 || (c.is_text() && i == 1))))
        .collect();
    let mut reservation = provider
        .memory()
        .reserve_bytes(
            (text_cells * 4 + layout.parameter_cells * 4 + support.len() + 136 * 12) as u64,
        )
        .unwrap();
    let mut text = reservation.alloc::<f32>(text_cells).unwrap();
    let mut parameters = reservation.alloc::<f32>(layout.parameter_cells).unwrap();
    let mut product_support = reservation.alloc::<u8>(support.len()).unwrap();
    let mut component_baselines = reservation.alloc::<f32>(136).unwrap();
    let mut cotangents = reservation.alloc::<f64>(136).unwrap();
    provider
        .htod_sync_copy_into_tracked(&vec![f32::NAN; text_cells], &mut text)
        .unwrap();
    provider
        .htod_sync_copy_into_tracked(&vec![f32::NAN; layout.parameter_cells], &mut parameters)
        .unwrap();
    provider
        .htod_sync_copy_into_tracked(&support, &mut product_support)
        .unwrap();
    provider
        .htod_sync_copy_into_tracked(&vec![0.0; 136], &mut component_baselines)
        .unwrap();
    provider
        .htod_sync_copy_into_tracked(&vec![1.0; 136], &mut cotangents)
        .unwrap();
    let invocation = SemanticRngBinding {
        model_generation: 7,
        stream_serial: 19,
        family_id: 1,
        proposal: 0,
    };
    let runtime = Arc::clone(provider.memory().runtime().unwrap());
    let producer_id = runtime.stream_pool().acquire().unwrap();
    let producer_stream = runtime.stream_pool().resolve(producer_id).unwrap();
    let producer = provider
        .bind_resident_execution_domain(runtime, producer_id, producer_stream)
        .unwrap();
    let mut recorder = producer.new_strict_recorder();
    recorder.write(&text);
    recorder.write(&parameters);
    // SAFETY: the producer records both complete owned ranges on its own
    // stream. There is no host wait before the consumer's D2D acquisition.
    unsafe {
        producer
            .enqueue(recorder, |enqueue| {
                for buffer in [&text, &parameters] {
                    cudarc::driver::sys::cuMemsetD32Async(
                        buffer.device_ptr_value(),
                        0,
                        buffer.len(),
                        enqueue.stream().cu_stream(),
                    )
                    .result()
                    .map_err(|e| xlog_core::XlogError::Kernel(e.to_string()))?;
                }
                cudarc::driver::sys::cuMemsetD32Async(
                    text.device_ptr_value(),
                    0.25f32.to_bits(),
                    1,
                    enqueue.stream().cu_stream(),
                )
                .result()
                .map_err(|e| xlog_core::XlogError::Kernel(e.to_string()))?;
                Ok::<(), xlog_core::XlogError>(())
            })
            .unwrap()
            .commit()
            .unwrap();
    }
    session
        .bind_policy(
            session.binding(),
            invocation,
            &text,
            &product_support,
            &parameters,
            &component_baselines,
        )
        .unwrap();
    // The completed D2D snapshot, not these mutable caller buffers, is sampled.
    provider
        .htod_sync_copy_into_tracked(&vec![f32::NAN; text_cells], &mut text)
        .unwrap();
    provider
        .htod_sync_copy_into_tracked(&vec![f32::NAN; layout.parameter_cells], &mut parameters)
        .unwrap();
    session.capture().unwrap();
    session.launch().unwrap();
    let forward = session.observe(0).unwrap().into_published().unwrap();
    assert_eq!(forward.next_proposal, 1);
    // Every saved factor in this binary-exact two-category/singleton roster is
    // exactly one. The device reduction must retain the complete tuple weight.
    assert_eq!(forward.importance_weight, 1.0);
    assert_eq!(
        session.launch(),
        Err(SemanticTransitionError::UnconsumedPolicyTape)
    );
    // The first two-category projection has h*pi = 1.25 or 0.75. Either
    // branch distinguishes the required late FP32 cast from an early one.
    let (divisor, coefficient) = if forward.components[0].choice == 0 {
        (1.25, 1.0 + 2.0f64.powi(-24))
    } else {
        (0.75, 1.0 + 2.0f64.powi(-24) + 2.0f64.powi(-52))
    };
    let expected_first = (coefficient / divisor) as f32;
    assert_ne!(
        expected_first,
        (f64::from(coefficient as f32) / divisor) as f32
    );
    let mut coefficients = vec![1.0f64; 136];
    coefficients[0] = coefficient;
    provider
        .htod_sync_copy_into_tracked(&coefficients, &mut cotangents)
        .unwrap();
    let gradients = session.backward_policy(invocation, &cotangents).unwrap();
    assert_eq!(gradients.invocation, invocation);
    let text_gradient = provider
        .device()
        .inner()
        .dtoh_sync_copy(&gradients.text_logits)
        .unwrap();
    for (row, receipt) in text_gradient
        .chunks_exact(SemanticActionCatalogue::current().text_cardinality())
        .zip(forward.components.iter().filter(|r| r.ordinal % 68 < 32))
    {
        assert_eq!(receipt.legal_count, 2);
        let expected = if receipt.ordinal == 0 {
            expected_first
        } else {
            1.0
        };
        assert_eq!(row[receipt.choice as usize], expected);
        assert_eq!(row[1 - receipt.choice as usize], -expected);
        assert!(row[2..].iter().all(|&value| value == 0.0));
    }
    assert!(provider
        .device()
        .inner()
        .dtoh_sync_copy(&gradients.parameters)
        .unwrap()
        .iter()
        .all(|&x| x == 0.0));
    assert_eq!(session.launch(), Err(SemanticTransitionError::NotCaptured));
    // The new snapshot contains the now-invalid source, so no draw is allowed.
    session
        .bind_policy(
            session.binding(),
            invocation,
            &text,
            &product_support,
            &parameters,
            &component_baselines,
        )
        .unwrap();
    session.capture().unwrap();
    session.launch().unwrap();
    assert_eq!(
        session.observe(0),
        Err(SemanticTransitionError::NonFinitePolicyInput { completed_draws: 0 })
    );
    assert!(session.is_poisoned());
}

#[cfg(feature = "semantic-policy")]
#[test]
#[ignore = "requires an explicitly authorized remote CUDA run"]
fn policy_recurrent_vjp_crosses_slots_and_forced_null_fields() {
    use xlog_core::{RelId, Schema};
    use xlog_cuda::{
        SemanticAdmissionLimits, SemanticAdmissionRecords, SemanticHypergraphCapacities,
        SemanticPolarity, SemanticPredicateRecord, SemanticRecordRole, SemanticSupportRecord,
        SemanticTypedRecord,
    };
    let provider = Arc::new(
        CudaProviderBuilder::new(0, MemoryBudget::with_limit(512 * 1024 * 1024))
            .with_stream_capacity(1)
            .build()
            .unwrap(),
    );
    let runtime = Arc::clone(provider.memory().runtime().unwrap());
    let id = runtime.stream_pool().acquire().unwrap();
    let stream = runtime.stream_pool().resolve(id).unwrap();
    let domain = provider
        .bind_resident_execution_domain(runtime, id, stream)
        .unwrap();
    let make_session = || {
        let mut graph = provider
            .allocate_semantic_hypergraph(
                &domain,
                SemanticHypergraphCapacities::try_new(3, 4, 4, 4).unwrap(),
            )
            .unwrap();
        graph
            .admit_records(
                graph.empty_root(),
                SemanticAdmissionRecords {
                    predicates: [
                        SemanticRecordRole::Statement,
                        SemanticRecordRole::Provenance,
                        SemanticRecordRole::Source,
                        SemanticRecordRole::Context,
                        SemanticRecordRole::Scope,
                    ]
                    .into_iter()
                    .enumerate()
                    .map(|(i, role)| SemanticPredicateRecord {
                        predicate: RelId(i as u32),
                        role,
                        schema: Schema::new(vec![]),
                    })
                    .collect(),
                    records: (0..5)
                        .map(|i| SemanticTypedRecord {
                            predicate: RelId(i),
                            arguments: vec![],
                            qualifiers: vec![],
                        })
                        .collect(),
                    supports: [SemanticPolarity::Pro, SemanticPolarity::Contra]
                        .into_iter()
                        .map(|polarity| SemanticSupportRecord {
                            statement: 0,
                            polarity,
                            provenance: 1,
                            source: 2,
                            context: 3,
                            scope: 4,
                        })
                        .collect(),
                },
                SemanticAdmissionLimits {
                    max_records: 12,
                    max_terms: 0,
                    max_references: 10,
                    max_utf8_bytes: 0,
                },
            )
            .unwrap();
        SemanticTransitionSession::from_hypergraph(graph).unwrap()
    };
    let mut session = make_session();
    let layout = session.policy_layout().unwrap();
    let support: Vec<u8> = session
        .components()
        .iter()
        .flat_map(|c| {
            (0..c.cardinality).map(move |category| {
                u8::from(if c.is_text() || c.slot == 0 {
                    category == 0
                } else {
                    match c.field {
                        0 | 1 | 2 | 11 => category == 1,
                        17 => category != 0,
                        _ => category == 0,
                    }
                })
            })
        })
        .collect();
    let text_cells = 64 * SemanticActionCatalogue::current().text_cardinality();
    let mut values = vec![0.0f32; layout.parameter_cells];
    for d in 0..128 {
        values[layout.recurrence.start + d * 128 + d] = 0.98;
        values[layout.fields[0].embeddings.start + d] = 0.002;
        values[layout.fields[0].embeddings.start + 128 + d] = 0.002;
        values[layout.fields[17].embeddings.start + d] = 0.01;
        values[layout.fields[17].embeddings.start + 128 + d] = -0.01;
    }
    for value in &mut values[layout.positions.clone()] {
        *value = 0.001;
    }
    let mut reservation = provider
        .memory()
        .reserve_bytes((text_cells * 4 + values.len() * 4 + support.len() + 136 * 12) as u64)
        .unwrap();
    let mut text = reservation.alloc::<f32>(text_cells).unwrap();
    let mut parameters = reservation.alloc::<f32>(values.len()).unwrap();
    let mut product_support = reservation.alloc::<u8>(support.len()).unwrap();
    let mut component_baselines = reservation.alloc::<f32>(136).unwrap();
    let mut cotangents = reservation.alloc::<f64>(136).unwrap();
    provider
        .htod_sync_copy_into_tracked(&vec![0.0; text_cells], &mut text)
        .unwrap();
    provider
        .htod_sync_copy_into_tracked(&values, &mut parameters)
        .unwrap();
    provider
        .htod_sync_copy_into_tracked(&support, &mut product_support)
        .unwrap();
    provider
        .htod_sync_copy_into_tracked(&vec![0.0; 136], &mut component_baselines)
        .unwrap();
    let mut coefficients = vec![0.0; 136];
    coefficients[67] = 1.0;
    provider
        .htod_sync_copy_into_tracked(&coefficients, &mut cotangents)
        .unwrap();
    let invocation = SemanticRngBinding {
        model_generation: 7,
        stream_serial: 19,
        family_id: 1,
        proposal: 0,
    };
    session
        .bind_policy(
            session.binding(),
            invocation,
            &text,
            &product_support,
            &parameters,
            &component_baselines,
        )
        .unwrap();
    session.capture().unwrap();
    session.launch().unwrap();
    let forward = session.observe(0).unwrap().into_published().unwrap();
    let gradients = session.backward_policy(invocation, &cotangents).unwrap();
    let adjoints = provider
        .device()
        .inner()
        .dtoh_sync_copy(&gradients.parameters)
        .unwrap();
    assert!(forward.components[32..50].iter().all(|r| r.choice == 0));
    assert_eq!(forward.components[50].choice, 1);
    assert_eq!(forward.components[67].legal_count, 2);
    assert_eq!(forward.components[67].active_count, 2);
    let second_leaf = forward.components[67].choice as usize - 1;
    let first_slot_action_embedding = layout.fields[0].embeddings.start;
    let indices = [
        layout.recurrence.start,
        layout.positions.start + 8 * 128,
        layout.z.start + 128,
        layout.fields[17].embeddings.start + second_leaf * 128,
        first_slot_action_embedding,
    ];
    // Field eight is forced NULL in both slots, yet later scores differentiate
    // through its position/advance. NO_EDIT's action embedding can influence this
    // objective only through slot zero's advance: slot one selects APPLY and its leaf
    // has a nonzero cotangent. A state reset between slots removes this path.
    assert!(adjoints[indices[1]].abs() > 1e-5);
    assert!(adjoints[first_slot_action_embedding].abs() > 1e-5);
    let probability = |receipt: &xlog_cuda::SemanticTransitionReceipt| {
        let integer = |limbs: [u64; 3]| {
            (limbs[2] as f64) * 2.0f64.powi(128)
                + (limbs[1] as f64) * 2.0f64.powi(64)
                + limbs[0] as f64
        };
        (integer(receipt.p) / integer(receipt.q)).ln()
    };
    for index in indices {
        let mut scores = [0.0; 2];
        for (side, delta) in [-0.001f32, 0.001].into_iter().enumerate() {
            // Changed candidate roots are retained by their graph. Every finite
            // difference therefore starts from the same independently admitted
            // semantic state, not an exhausted prior proposal arena.
            session = make_session();
            let mut perturbed = values.clone();
            perturbed[index] += delta;
            provider
                .htod_sync_copy_into_tracked(&perturbed, &mut parameters)
                .unwrap();
            session
                .bind_policy(
                    session.binding(),
                    invocation,
                    &text,
                    &product_support,
                    &parameters,
                    &component_baselines,
                )
                .unwrap();
            session.capture().unwrap();
            session.launch().unwrap();
            let result = session.observe(0).unwrap().into_published().unwrap();
            assert_eq!(result.components[67].legal_count, 2);
            assert_eq!(result.components[67].active_count, 2);
            assert_eq!(
                result
                    .components
                    .iter()
                    .map(|r| r.choice)
                    .collect::<Vec<_>>(),
                forward
                    .components
                    .iter()
                    .map(|r| r.choice)
                    .collect::<Vec<_>>()
            );
            scores[side] = probability(&result.components[67]);
            // Exercise the actual late path for this invocation as well.
            drop(session.backward_policy(invocation, &cotangents).unwrap());
        }
        let finite_difference = (scores[1] - scores[0]) / 0.002;
        if index == first_slot_action_embedding {
            assert!(finite_difference.abs() > 1e-5);
        }
        assert!(
            (f64::from(adjoints[index]) - finite_difference).abs() < 3e-4,
            "parameter {index}: native {} vs finite difference {finite_difference}",
            adjoints[index]
        );
    }
}

#[test]
fn public_typed_decoder_retains_provider_and_captured_executable() {
    use xlog_core::{RelId, Schema};
    use xlog_cuda::{
        SemanticAdmissionLimits, SemanticAdmissionRecords, SemanticHypergraphCapacities,
        SemanticPolarity, SemanticPredicateRecord, SemanticRecordRole, SemanticSupportRecord,
        SemanticTruth, SemanticTypedRecord,
    };
    if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
        return;
    }
    let provider = Arc::new(
        CudaProviderBuilder::new(0, MemoryBudget::with_limit(256 * 1024 * 1024))
            .with_stream_capacity(1)
            .build()
            .unwrap(),
    );
    let provider_lifetime = Arc::downgrade(&provider);
    let runtime = Arc::clone(provider.memory().runtime().unwrap());
    let id = runtime.stream_pool().acquire().unwrap();
    let stream = runtime.stream_pool().resolve(id).unwrap();
    let domain = provider
        .bind_resident_execution_domain(runtime, id, stream)
        .unwrap();
    let mut graph = provider
        .allocate_semantic_hypergraph(
            &domain,
            SemanticHypergraphCapacities::try_new(3, 4, 4, 4).unwrap(),
        )
        .unwrap();
    drop(provider);
    assert!(provider_lifetime.upgrade().is_some());
    let base = graph.empty_root();
    let roles = [
        SemanticRecordRole::Statement,
        SemanticRecordRole::Provenance,
        SemanticRecordRole::Source,
        SemanticRecordRole::Context,
        SemanticRecordRole::Scope,
    ];
    graph
        .admit_records(
            base,
            SemanticAdmissionRecords {
                predicates: roles
                    .into_iter()
                    .enumerate()
                    .map(|(i, role)| SemanticPredicateRecord {
                        predicate: RelId(i as u32),
                        role,
                        schema: Schema::new(vec![]),
                    })
                    .collect(),
                records: (0..5)
                    .map(|i| SemanticTypedRecord {
                        predicate: RelId(i),
                        arguments: vec![],
                        qualifiers: vec![],
                    })
                    .collect(),
                supports: [SemanticPolarity::Pro, SemanticPolarity::Contra]
                    .into_iter()
                    .map(|polarity| SemanticSupportRecord {
                        statement: 0,
                        polarity,
                        provenance: 1,
                        source: 2,
                        context: 3,
                        scope: 4,
                    })
                    .collect(),
            },
            SemanticAdmissionLimits {
                max_records: 12,
                max_terms: 0,
                max_references: 10,
                max_utf8_bytes: 0,
            },
        )
        .unwrap();
    let mut session = SemanticTransitionSession::from_hypergraph(graph).unwrap();
    let mut inputs: Vec<_> = session
        .components()
        .iter()
        .map(|c| {
            let mut product_support = vec![0; c.cardinality as usize];
            let choice = if c.is_text() {
                0
            } else {
                match c.field {
                    0 | 1 | 2 | 11 => 1,
                    17 => {
                        if c.slot == 0 {
                            1
                        } else {
                            2
                        }
                    }
                    _ => 0,
                }
            };
            product_support[choice] = 1;
            SemanticComponentInput {
                logits: vec![0.0; product_support.len()],
                product_support,
            }
        })
        .collect();
    // Both edits act on the same nullary statement; two distinct support polarities.
    session
        .bind_inputs(
            session.binding(),
            SemanticRngBinding {
                model_generation: 1,
                stream_serial: 1,
                family_id: 1,
                proposal: 0,
            },
            &inputs,
        )
        .unwrap();
    session.capture().unwrap();
    // Replacing the registry name must not replace or unload the executable
    // already captured by this session. The replacement deliberately does no
    // work; the original graph must still produce both real semantic outcomes.
    provider_lifetime
        .upgrade()
        .unwrap()
        .device()
        .inner()
        .load_ptx(
            cudarc::nvrtc::Ptx::from_src(
                ".version 7.0\n.target sm_75\n.address_size 64\n\
                 .visible .entry semantic_transition_execute() { ret; }\n",
            ),
            "xlog_semantic_transition",
            &["semantic_transition_execute"],
        )
        .unwrap();
    let before = session.host_io_stats();
    session.launch().unwrap();
    assert_eq!(session.host_io_stats(), before);
    let result = session.observe(0).unwrap().into_published().unwrap();
    assert_eq!(result.components.len(), 136);
    for lane in &result.lanes {
        let (_, root) = lane.root.as_ref().unwrap().as_ref().unwrap();
        assert_eq!(root.extents(), xlog_cuda::SemanticExtents::new(1, 2, 2));
        let inserted = lane.edits[1].as_ref().unwrap().as_ref().unwrap();
        match inserted {
            xlog_cuda::SemanticInsertOutcome::Inserted(value) => {
                assert_eq!(value.version().truth(), SemanticTruth::Both)
            }
            other => panic!("expected a new support: {other:?}"),
        }
    }
    assert_ne!(
        result.lanes[0].root.as_ref().unwrap().as_ref().unwrap().0,
        result.lanes[1].root.as_ref().unwrap().as_ref().unwrap().0
    );
    inputs.clear();
    drop(session);
    assert!(provider_lifetime.upgrade().is_none());
}

#[test]
fn public_session_round_trips_all_136_components_as_no_edit_null_without_mutation() {
    if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
        eprintln!("Set XLOG_REQUIRE_CUDA=1 for the real CUDA public contract");
        return;
    }
    let provider = CudaProviderBuilder::new(0, MemoryBudget::with_limit(256 * 1024 * 1024))
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
    let catalogue = SemanticActionCatalogue::current();
    let mut session = SemanticTransitionSession::new(&provider, &domain).unwrap();
    let before_root = session.root_snapshot().unwrap();
    let mut inputs: Vec<_> = session
        .components()
        .iter()
        .map(|component| {
            let mut logits = vec![0.0; component.cardinality as usize];
            let mut product_support = vec![0u8; logits.len()];
            if component.is_text() {
                // Unequal, nonadjacent categories catch ordinal/CDF confusion.
                for (category, logit) in [(3, 0.0), (71, 0.125), (991, 0.5), (248076, -16.0)] {
                    logits[category] = logit;
                    product_support[category] = 1;
                }
            } else {
                product_support[0] = 1;
            }
            SemanticComponentInput {
                logits,
                product_support,
            }
        })
        .collect();
    let rng = SemanticRngBinding {
        model_generation: 19,
        stream_serial: 0x123456789abc,
        family_id: 7,
        proposal: 31,
    };
    session
        .bind_inputs(session.binding(), rng, &inputs)
        .unwrap();
    session.capture().unwrap();
    let before = session.host_io_stats();
    session.launch().unwrap();
    assert_eq!(
        session.host_io_stats(),
        before,
        "launch must have zero host I/O and sync"
    );
    eprintln!(
        "first launch interval: before={before:?}; after={:?}",
        session.host_io_stats()
    );
    assert!(matches!(
        session.launch(),
        Err(SemanticTransitionError::OverlappingLaunch)
    ));
    let first = session.observe(31).unwrap().into_published().unwrap();
    assert_eq!(first.root, before_root);
    assert_eq!(
        first.root.extents(),
        xlog_cuda::SemanticExtents::new(0, 0, 0)
    );
    assert_eq!(first.root_handle.generation(), 1);
    assert_eq!(
        session.host_io_stats().observation_bytes - before.observation_bytes,
        41672
    );
    assert_eq!(
        session.host_io_stats().observation_calls - before.observation_calls,
        2
    );
    assert_eq!(
        session.host_io_stats().session_stream_waits - before.session_stream_waits,
        2
    );
    assert_eq!(first.components.len(), 136);
    assert_eq!(first.next_proposal, 32);
    assert_eq!(first.components.iter().filter(|r| r.kind == 0).count(), 64);
    assert_eq!(
        first
            .components
            .iter()
            .filter(|r| r.kind == 1 && r.field == 0)
            .count(),
        4
    );
    for (ordinal, receipt) in first.components.iter().enumerate() {
        assert_eq!(receipt.ordinal as usize, ordinal);
        assert_eq!(receipt.proposal, 31);
        assert_eq!(receipt.catalogue_generation, catalogue.binding().generation);
        assert_eq!(receipt.catalogue_digest, catalogue.binding().digest);
        assert_eq!(receipt.key, [catalogue.binding().generation as u32, 19]);
        let block = 136 * 31 + ordinal as u64;
        assert_eq!(
            receipt.counter,
            [
                block as u32,
                (block >> 32) as u32,
                ((rng.stream_serial << 8) | 7) as u32,
                (((rng.stream_serial << 8) | 7) >> 32) as u32
            ]
        );
        assert!(receipt.cdf_start <= receipt.draw && receipt.draw < receipt.cdf_end);
        check_receipt(receipt, &inputs[ordinal]);
        if receipt.kind != 0 {
            assert_eq!(receipt.choice, 0);
            assert_eq!(receipt.p, [1, 0, 0]);
            assert_eq!(receipt.q, [1, 0, 0]);
            assert_eq!(receipt.mass, 1u64 << 63);
        }
    }
    let before = session.host_io_stats();
    session.launch().unwrap();
    assert_eq!(session.host_io_stats(), before);
    eprintln!(
        "replay interval: before={before:?}; after={:?}",
        session.host_io_stats()
    );
    let second = session.observe(32).unwrap().into_published().unwrap();
    assert_eq!(second.next_proposal, 33);
    assert_eq!(second.root, before_root);
    assert_eq!(second.root_handle, first.root_handle);
    assert_eq!(second.root.canonical_bytes(), first.root.canonical_bytes());
    assert_ne!(first.components[0].counter, second.components[0].counter);
    assert!(first
        .components
        .iter()
        .zip(&second.components)
        .any(|(a, b)| a.choice != b.choice));
    // A changed legal input must change the result through the same captured path.
    for (component, input) in catalogue.components().iter().zip(&mut inputs) {
        if component.is_text() {
            input.product_support.fill(0);
            input.product_support[17] = 1;
        }
    }
    session
        .bind_inputs(
            session.binding(),
            SemanticRngBinding {
                proposal: 33,
                ..rng
            },
            &inputs,
        )
        .unwrap();
    session.launch().unwrap();
    let changed = session.observe(33).unwrap().into_published().unwrap();
    assert!(changed
        .components
        .iter()
        .filter(|r| r.kind == 0)
        .all(|r| r.choice == 17));
    assert_eq!(changed.root, before_root);
    drop(session);
    provider.memory().reap_pending_deallocations().unwrap();
    assert_eq!(provider.memory().allocated_bytes(), 0);
    assert_eq!(provider.memory().deallocation_failure_bytes(), 0);
}

#[test]
#[ignore = "requires an explicitly authorized remote CUDA run"]
fn public_session_reduces_nonunit_importance_factors_across_both_lanes() {
    const TEXT_SUPPORT: usize = 65_533;
    let catalogue = SemanticActionCatalogue::current();
    assert!(catalogue.text_cardinality() >= TEXT_SUPPORT);
    let provider = Arc::new(
        CudaProviderBuilder::new(0, MemoryBudget::with_limit(256 * 1024 * 1024))
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
    let mut session = SemanticTransitionSession::new(&provider, &domain).unwrap();
    let inputs: Vec<_> = session
        .components()
        .iter()
        .map(|component| {
            let logits = vec![0.0; component.cardinality as usize];
            let mut product_support = vec![0; logits.len()];
            let legal_count = if component.is_text() { TEXT_SUPPORT } else { 1 };
            product_support[..legal_count].fill(1);
            SemanticComponentInput {
                logits,
                product_support,
            }
        })
        .collect();
    let rng = SemanticRngBinding {
        model_generation: 20,
        stream_serial: 0x123456789abc,
        family_id: 7,
        proposal: 31,
    };
    session
        .bind_inputs(session.binding(), rng, &inputs)
        .unwrap();
    session.capture().unwrap();
    session.launch().unwrap();
    let forward = session
        .observe(rng.proposal)
        .unwrap()
        .into_published()
        .unwrap();
    assert_eq!(forward.components.len(), 136);

    // Convert the exact integer to decimal before Rust's correctly rounded
    // parser, independently of the device's binary guard/sticky-bit conversion.
    let integer_to_f64 = |mut limbs: [u64; 3]| {
        let mut digits = Vec::new();
        loop {
            let mut remainder = 0u128;
            for limb in limbs.iter_mut().rev() {
                let dividend = (remainder << 64) | u128::from(*limb);
                *limb = (dividend / 10) as u64;
                remainder = dividend % 10;
            }
            digits.push(b'0' + remainder as u8);
            if limbs == [0; 3] {
                break;
            }
        }
        digits.reverse();
        String::from_utf8(digits).unwrap().parse::<f64>().unwrap()
    };
    let factors: Vec<_> = forward
        .components
        .iter()
        .enumerate()
        .map(|(ordinal, receipt)| {
            assert_eq!(receipt.ordinal as usize, ordinal);
            let factor = integer_to_f64(receipt.p) / integer_to_f64(receipt.factor_denominator);
            assert!(factor.is_finite() && factor > 0.0);
            if !session.components()[ordinal].is_text() {
                assert_eq!(receipt.legal_count, 1);
                assert_eq!(factor, 1.0);
            }
            factor
        })
        .collect();
    let expected = factors
        .iter()
        .copied()
        .fold(1.0f64, |weight, factor| weight * factor);
    // The fixed Philox binding was checked to avoid cancellation to one:
    // 0.9999999999999574 in full, 0.9999999999999609 omitting its first factor.
    // 2^63 mod 65533 = 32807 makes both possible text interval factors nonunit.
    assert_ne!(expected.to_bits(), 1.0f64.to_bits());
    assert_eq!(forward.importance_weight.to_bits(), expected.to_bits());
    for lane in 0..2 {
        let nonunit: Vec<_> = forward
            .components
            .iter()
            .enumerate()
            .filter(|(ordinal, receipt)| receipt.lane == lane && factors[*ordinal] != 1.0)
            .map(|(ordinal, _)| ordinal)
            .collect();
        assert_eq!(
            nonunit.len(),
            32,
            "each text factor in lane {lane} must be nonunit"
        );
        let omitted = factors
            .iter()
            .enumerate()
            .filter(|(ordinal, _)| *ordinal != nonunit[0])
            .fold(1.0f64, |weight, (_, factor)| weight * factor);
        assert_ne!(omitted.to_bits(), expected.to_bits());
    }
}

#[test]
fn public_decoder_preflight_rejects_capacity_and_incomplete_support_before_drawing() {
    use xlog_cuda::{
        SemanticAdmissionLimits, SemanticAdmissionRecords, SemanticHandleKind,
        SemanticHypergraphCapacities, SemanticHypergraphError,
    };
    if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
        return;
    }
    let provider = CudaProviderBuilder::new(0, MemoryBudget::with_limit(256 * 1024 * 1024))
        .with_stream_capacity(1)
        .build()
        .unwrap();
    let provider = Arc::new(provider);
    let runtime = Arc::clone(provider.memory().runtime().unwrap());
    let id = runtime.stream_pool().acquire().unwrap();
    let stream = runtime.stream_pool().resolve(id).unwrap();
    let domain = provider
        .bind_resident_execution_domain(runtime, id, stream)
        .unwrap();
    for incomplete in [false, true] {
        let mut graph = provider
            .allocate_semantic_hypergraph(
                &domain,
                SemanticHypergraphCapacities::try_new(if incomplete { 3 } else { 2 }, 4, 4, 4)
                    .unwrap(),
            )
            .unwrap();
        let root = graph.empty_root();
        graph
            .admit_records(
                root,
                SemanticAdmissionRecords {
                    predicates: vec![],
                    records: vec![],
                    supports: vec![],
                },
                SemanticAdmissionLimits {
                    max_records: 0,
                    max_terms: 0,
                    max_references: 0,
                    max_utf8_bytes: 0,
                },
            )
            .unwrap();
        let mut session = SemanticTransitionSession::from_hypergraph(graph).unwrap();
        let inputs: Vec<_> = session
            .components()
            .iter()
            .map(|c| {
                let mut support = vec![0; c.cardinality as usize];
                support[usize::from(incomplete && !c.is_text() && c.field == 0)] = 1;
                SemanticComponentInput {
                    logits: vec![0.; support.len()],
                    product_support: support,
                }
            })
            .collect();
        session
            .bind_inputs(
                session.binding(),
                SemanticRngBinding {
                    model_generation: 1,
                    stream_serial: 1,
                    family_id: 1,
                    proposal: 9,
                },
                &inputs,
            )
            .unwrap();
        session.capture().unwrap();
        let before = session.host_io_stats();
        session.launch().unwrap();
        assert_eq!(session.host_io_stats(), before);
        let error = session.observe(9).unwrap_err();
        if incomplete {
            assert!(matches!(
                error,
                SemanticTransitionError::InvalidFinalSupport { completed_draws: 0 }
            ));
        } else {
            assert!(matches!(
                error,
                SemanticTransitionError::Semantic(SemanticHypergraphError::CapacityExceeded {
                    kind: SemanticHandleKind::Root,
                    capacity: 2
                })
            ));
        }
        assert!(session.is_poisoned());
        drop(session);
        provider.memory().reap_pending_deallocations().unwrap();
        assert_eq!(provider.memory().allocated_bytes(), 0);
    }
}

#[test]
fn public_decoder_matches_canonical_typed_identity_and_ordered_support_lifecycle() {
    use xlog_core::{symbol, RelId, ScalarType, Schema};
    use xlog_cuda::{
        SemanticActionDescriptor as D, SemanticAdmissionLimits, SemanticAdmissionRecords,
        SemanticArgument as A, SemanticHypergraphCapacities, SemanticInsertOutcome,
        SemanticPolarity as P, SemanticPredicateRecord, SemanticRecordRole as Role,
        SemanticSupportRecord, SemanticTypedRecord, SemanticView,
    };
    if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
        return;
    }
    let provider = CudaProviderBuilder::new(0, MemoryBudget::with_limit(256 * 1024 * 1024))
        .with_stream_capacity(1)
        .build()
        .unwrap();
    let provider = Arc::new(provider);
    let runtime = Arc::clone(provider.memory().runtime().unwrap());
    let id = runtime.stream_pool().acquire().unwrap();
    let stream = runtime.stream_pool().resolve(id).unwrap();
    let domain = provider
        .bind_resident_execution_domain(runtime, id, stream)
        .unwrap();
    // Longer than the former fixed SHA scratch, with multi-byte UTF-8.
    let symbol_id = symbol::intern(&"accepted λ semantic content ".repeat(40));
    let roles = [
        Role::Statement,
        Role::Qualifier,
        Role::Provenance,
        Role::Source,
        Role::Context,
        Role::Scope,
    ];
    let mut records = vec![SemanticTypedRecord {
        predicate: RelId(0),
        arguments: vec![A::Symbol(symbol_id), A::U64(19)],
        qualifiers: vec![1, 2],
    }];
    for (p, v) in [(1, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (2, 7)] {
        records.push(SemanticTypedRecord {
            predicate: RelId(p),
            arguments: vec![A::U64(v)],
            qualifiers: vec![],
        });
    }
    let records = SemanticAdmissionRecords {
        predicates: roles
            .into_iter()
            .enumerate()
            .map(|(i, role)| SemanticPredicateRecord {
                predicate: RelId(i as u32),
                role,
                schema: if i == 0 {
                    Schema::new(vec![
                        ("entity".into(), ScalarType::Symbol),
                        ("value".into(), ScalarType::U64),
                    ])
                } else {
                    Schema::new(vec![("value".into(), ScalarType::U64)])
                },
            })
            .collect(),
        records,
        supports: [(P::Pro, 3), (P::Contra, 3), (P::Pro, 7)]
            .into_iter()
            .map(|(polarity, provenance)| SemanticSupportRecord {
                statement: 0,
                polarity,
                provenance,
                source: 4,
                context: 5,
                scope: 6,
            })
            .collect(),
    };
    let limits = SemanticAdmissionLimits {
        max_records: 32,
        max_terms: 64,
        max_references: 32,
        max_utf8_bytes: 4096,
    };
    let cases = [
        [[Some(0), Some(1)], [Some(0), Some(1)]], // BOTH, two versions
        [[Some(0), Some(2)], [Some(0), Some(2)]], // TRUE, two reasons
        [[Some(0), Some(0)], [Some(0), Some(0)]], // new + exact duplicate
        [[Some(0), Some(0)], [Some(1), Some(1)]], // independent TRUE / FALSE
        [[Some(0), None], [Some(1), None]],       // new + NO_EDIT
        [[None, None], [None, None]],             // no semantic commands
    ];
    for (case, selected) in cases.into_iter().enumerate() {
        let mut expected = provider
            .allocate_semantic_hypergraph(
                &domain,
                SemanticHypergraphCapacities::try_new(3, 4, 4, 4).unwrap(),
            )
            .unwrap();
        let base = expected.empty_root();
        expected
            .admit_records(base, records.clone(), limits)
            .unwrap();
        let key = expected.admission().unwrap().statement_key(0).unwrap();
        let events: Vec<_> = (0..3)
            .map(|i| expected.admission().unwrap().support_event(i).unwrap())
            .collect();
        let mut expected_roots = Vec::new();
        for lane in selected {
            let candidate = expected.fork(base).unwrap();
            for event in lane.into_iter().flatten() {
                expected
                    .insert_support(candidate, &key, &events[event])
                    .unwrap();
            }
            let root = expected.seal(candidate).unwrap();
            expected_roots.push(expected.snapshot(SemanticView::Root(root)).unwrap());
        }
        let mut graph = provider
            .allocate_semantic_hypergraph(
                &domain,
                SemanticHypergraphCapacities::try_new(3, 4, 4, 4).unwrap(),
            )
            .unwrap();
        let base = graph.empty_root();
        graph.admit_records(base, records.clone(), limits).unwrap();
        let mut session = SemanticTransitionSession::from_hypergraph(graph).unwrap();
        let category = |field, wanted: &dyn Fn(&D) -> bool| {
            (1..session.components()[32 + field as usize].cardinality)
                .find(|&i| wanted(session.descriptor_meaning(field, i).unwrap()))
                .unwrap()
        };
        let target = category(
            2,
            &|d| matches!(d, D::Target { predicate } if *predicate == RelId(0)),
        );
        let symbol = category(3, &|d| {
            matches!(
                d,
                D::Operand {
                    scalar_type: ScalarType::Symbol,
                    ..
                }
            )
        });
        let value = category(4, &|d| {
            matches!(d, D::Operand { scalar_type: ScalarType::U64, encoded_value, .. }
            if encoded_value[1..] == 19u64.to_le_bytes())
        });
        let qualifier = category(
            11,
            &|d| matches!(d, D::QualifierBundle { records } if records == &[1, 2]),
        );
        let leaves: Vec<_> = (0..3)
            .map(|index| {
                category(
                    17,
                    &|d| matches!(d, D::SupportEvent { source_record } if *source_record == index),
                )
            })
            .collect();
        let inputs: Vec<_> = session
            .components()
            .iter()
            .map(|c| {
                let mut support = vec![0; c.cardinality as usize];
                let mut logits = vec![0.0; support.len()];
                let event = selected[(c.lane - 1) as usize][c.slot as usize];
                let choice = match (c.is_text(), event) {
                    (false, Some(event)) => match c.field {
                        0 | 1 => 1,
                        2 => target,
                        3 => symbol,
                        4 => value,
                        11 => qualifier,
                        17 => leaves[event],
                        _ => 0,
                    },
                    _ => 0,
                };
                support[choice as usize] = 1;
                if !c.is_text() && event.is_some() && (2..=6).contains(&c.field) {
                    // Supported incompatible categories have higher logits, but cannot
                    // enter the one canonical CDF or become selected operands.
                    support[0] = 1;
                    logits[0] = 12.0;
                    if c.field == 3 {
                        support[value as usize] = 1;
                        logits[value as usize] = 12.0;
                    }
                    if c.field == 4 {
                        support[symbol as usize] = 1;
                        logits[symbol as usize] = 12.0;
                    }
                }
                SemanticComponentInput {
                    logits,
                    product_support: support,
                }
            })
            .collect();
        let before_bind = session.host_io_stats();
        assert!(matches!(
            session.bind_inputs(
                SemanticActionCatalogue::current().binding(),
                SemanticRngBinding {
                    model_generation: 7,
                    stream_serial: 9,
                    family_id: 1,
                    proposal: 0
                },
                &inputs
            ),
            Err(SemanticTransitionError::CatalogueMismatch)
        ));
        assert_eq!(session.host_io_stats(), before_bind);
        session
            .bind_inputs(
                session.binding(),
                SemanticRngBinding {
                    model_generation: 7,
                    stream_serial: 9,
                    family_id: 1,
                    proposal: 0,
                },
                &inputs,
            )
            .unwrap();
        session.capture().unwrap();
        let before = session.host_io_stats();
        session.launch().unwrap();
        assert_eq!(session.host_io_stats(), before);
        let actual = session.observe(0).unwrap().into_published().unwrap();
        assert_eq!(actual.components.len(), 136);
        for (lane_index, lane) in actual.lanes.iter().enumerate() {
            let expected_work = match case {
                0 => (2, 2, 1),
                1 => (2, 2, 0),
                2 | 3 => (2, 1, 0),
                4 => (1, 1, 0),
                5 => (0, 0, 0),
                _ => unreachable!(),
            };
            assert_eq!(
                (
                    lane.work.edit_commands,
                    lane.work.added_supports,
                    lane.work.defined_truth_changes,
                ),
                expected_work,
                "case {case}, lane {lane_index}: actual semantic work"
            );
            assert_eq!(
                lane.root.as_ref().unwrap().as_ref().unwrap().1,
                expected_roots[lane_index],
                "case {case}, lane {lane_index}"
            );
            for edit in &lane.edits {
                match edit.as_ref().unwrap() {
                    Some(SemanticInsertOutcome::Inserted(value)) => {
                        assert_eq!(value.statement().identity(), key.identity())
                    }
                    Some(SemanticInsertOutcome::Unchanged(_)) | None => {}
                }
            }
            if case == 2 || case == 3 {
                assert!(matches!(
                    lane.edits[1],
                    Ok(Some(SemanticInsertOutcome::Unchanged(_)))
                ));
            }
        }
        assert_eq!(
            actual.root.extents(),
            xlog_cuda::SemanticExtents::new(0, 0, 0)
        );
        eprintln!("typed decoder case {case}: 136 draws, hot I/O unchanged: {before:?}");
        drop(session);
        drop(expected);
        provider.memory().reap_pending_deallocations().unwrap();
        assert_eq!(provider.memory().allocated_bytes(), 0);
        assert_eq!(provider.memory().deallocation_failure_bytes(), 0);
    }
}

#[test]
#[ignore = "requires an explicitly authorized remote CUDA run"]
fn public_session_evaluates_actual_carry_edits_and_task_local_refusals() {
    use xlog_core::{RelId, ScalarType, Schema};
    use xlog_cuda::{
        SemanticActionDescriptor as D, SemanticAdmissionLimits, SemanticAdmissionRecords,
        SemanticArgument as A, SemanticHypergraphCapacities, SemanticPolarity as P,
        SemanticPredicateRecord, SemanticRecordRole as Role, SemanticSupportRecord,
        SemanticTaskEvaluationSpec, SemanticTaskObservation, SemanticTaskProgram,
        SemanticTaskRefusal as Refusal, SemanticTaskScoring, SemanticTypedRecord,
    };

    #[derive(Debug)]
    struct ArithmeticProgram;
    impl SemanticTaskProgram for ArithmeticProgram {
        fn observe(
            &self,
            _: Arc<xlog_cuda::CudaKernelProvider>,
        ) -> Result<SemanticTaskObservation, SemanticTransitionError> {
            let inputs = [[1u32, 1u32], [0u32, 1u32], [1u32, 0u32]];
            let results = inputs.map(|[left, right]| (left + right) / 2);
            Ok(SemanticTaskObservation {
                program_source: b"fn observe(left: u32, right: u32) -> u32 { (left + right) / 2 }"
                    .to_vec(),
                input_bytes: inputs
                    .into_iter()
                    .flatten()
                    .flat_map(u32::to_le_bytes)
                    .collect(),
                result_bytes: results.into_iter().flat_map(u32::to_le_bytes).collect(),
                expected_truth: results.map(|value| {
                    if value == 1 {
                        xlog_cuda::SemanticTruth::True
                    } else {
                        xlog_cuda::SemanticTruth::False
                    }
                }),
            })
        }
    }
    let task_spec = || SemanticTaskEvaluationSpec {
        statement_records: [0, 1, 1],
        allowed_support_records: vec![0, 1, 2, 3],
        program: Arc::new(ArithmeticProgram),
        scoring: SemanticTaskScoring {
            correct_weight: 14,
            all_correct_weight: 7,
            work_weight: 1,
            improvement_weight: 45,
            refusal_weight: 15,
            spent_weight: 1,
        },
        admissible_truth_masks: [7, 7, 7],
    };
    #[derive(Debug)]
    struct FailingProgram;
    impl SemanticTaskProgram for FailingProgram {
        fn observe(
            &self,
            _: Arc<xlog_cuda::CudaKernelProvider>,
        ) -> Result<SemanticTaskObservation, SemanticTransitionError> {
            Err(SemanticTransitionError::InvalidInput {
                detail: "observer failed".into(),
            })
        }
    }

    let provider = Arc::new(
        CudaProviderBuilder::new(0, MemoryBudget::with_limit(256 * 1024 * 1024))
            .with_stream_capacity(1)
            .build()
            .unwrap(),
    );
    let runtime = Arc::clone(provider.memory().runtime().unwrap());
    let id = runtime.stream_pool().acquire().unwrap();
    let stream = runtime.stream_pool().resolve(id).unwrap();
    let domain = provider
        .bind_resident_execution_domain(runtime, id, stream)
        .unwrap();
    let mut records = SemanticAdmissionRecords {
        predicates: vec![SemanticPredicateRecord {
            predicate: RelId(0),
            role: Role::Statement,
            schema: Schema::new(vec![
                ("left".into(), ScalarType::U32),
                ("right".into(), ScalarType::U32),
            ])
            .with_sort_labels(vec!["bit".into(), "bit".into()])
            .unwrap(),
        }],
        records: [[1, 1], [0, 1]]
            .into_iter()
            .map(|arguments| SemanticTypedRecord {
                predicate: RelId(0),
                arguments: arguments.into_iter().map(A::U32).collect(),
                qualifiers: vec![],
            })
            .collect(),
        supports: [0, 1]
            .into_iter()
            .flat_map(|statement| {
                [P::Pro, P::Contra]
                    .into_iter()
                    .map(move |polarity| SemanticSupportRecord {
                        statement,
                        polarity,
                        provenance: 2,
                        source: 3,
                        context: 4,
                        scope: 5,
                    })
            })
            .collect(),
    };
    for (index, role) in [Role::Provenance, Role::Source, Role::Context, Role::Scope]
        .into_iter()
        .enumerate()
    {
        let predicate = RelId(index as u32 + 1);
        records.predicates.push(SemanticPredicateRecord {
            predicate,
            role,
            schema: Schema::new(vec![]),
        });
        records.records.push(SemanticTypedRecord {
            predicate,
            arguments: vec![],
            qualifiers: vec![],
        });
    }
    let cases = [
        (
            "no edit",
            [[None, None], [None, None]],
            -9,
            9,
            [None, None],
            [(0, 0, 0), (0, 0, 0)],
        ),
        (
            "two wrong losers",
            [[Some(([0, 1], 0)), None], [Some(([1, 1], 1)), None]],
            -11,
            9,
            [None, None],
            [(1, 1, 0), (1, 1, 0)],
        ),
        (
            "opposite supports violate the hard constraint",
            [[Some(([1, 1], 0)), Some(([1, 1], 1))], [None, None]],
            -27,
            9,
            [Some(Refusal::HardConstraint), None],
            [(2, 2, 1), (0, 0, 0)],
        ),
        (
            "one early scope refusal",
            [[Some(([0, 0], 0)), Some(([1, 1], 0))], [None, None]],
            -21,
            6,
            [Some(Refusal::Scope), None],
            [(0, 0, 0), (0, 0, 0)],
        ),
        (
            "two early scope refusals",
            [
                [Some(([0, 0], 0)), Some(([1, 1], 0))],
                [Some(([1, 0], 0)), Some(([0, 1], 1))],
            ],
            -33,
            3,
            [Some(Refusal::Scope), Some(Refusal::Scope)],
            [(0, 0, 0), (0, 0, 0)],
        ),
        (
            "all answers become correct",
            [[Some(([1, 1], 0)), Some(([0, 1], 1))], [None, None]],
            2016,
            9,
            [None, None],
            [(2, 2, 0), (0, 0, 0)],
        ),
    ];
    let new_session = || {
        let mut graph = provider
            .allocate_semantic_hypergraph(
                &domain,
                SemanticHypergraphCapacities::try_new(8, 8, 16, 16).unwrap(),
            )
            .unwrap();
        let base = graph.empty_root();
        graph
            .admit_records(
                base,
                records.clone(),
                SemanticAdmissionLimits {
                    max_records: 16,
                    max_terms: 16,
                    max_references: 32,
                    max_utf8_bytes: 256,
                },
            )
            .unwrap();
        SemanticTransitionSession::from_hypergraph(graph).unwrap()
    };
    for (name, selected, expected_return, expected_queries, refusals, work) in cases {
        let mut session = new_session();
        let binding = session.bind_task_evaluation(task_spec()).unwrap();
        assert_eq!(session.task_evaluation_identity(), Some(binding));
        let category = |field, wanted: &dyn Fn(&D) -> bool| {
            (1..session.components()[32 + field as usize].cardinality)
                .find(|&index| wanted(session.descriptor_meaning(field, index).unwrap()))
                .unwrap()
        };
        let target = category(
            2,
            &|value| matches!(value, D::Target { predicate } if *predicate == RelId(0)),
        );
        let operands = [0u32, 1].map(|value| {
            category(3, &|descriptor| {
                matches!(descriptor,
                    D::Operand { scalar_type: ScalarType::U32, sort_label, encoded_value }
                    if sort_label == "bit" && encoded_value[1..] == value.to_le_bytes())
            })
        });
        assert_ne!(operands[0], 0, "numeric zero must not be NULL");
        let qualifier = category(
            11,
            &|value| matches!(value, D::QualifierBundle { records } if records.is_empty()),
        );
        // Logical leaves are shared across statements, but the task grant above
        // retains both original statement-specific PRO/CONTRA associations.
        let leaves = [0, 1].map(|record| {
            category(17, &|value| {
                matches!(value, D::SupportEvent { source_record } if *source_record == record)
            })
        });
        let mut expected_choices = Vec::new();
        let inputs: Vec<_> = session
            .components()
            .iter()
            .map(|component| {
                let mut product_support = vec![0; component.cardinality as usize];
                let mut logits = vec![-100.0; product_support.len()];
                let edit = selected[(component.lane - 1) as usize][component.slot as usize];
                let choice = if component.is_text() {
                    0
                } else {
                    match edit {
                        Some((arguments, polarity)) => match component.field {
                            0 | 1 => 1,
                            2 => target,
                            3 | 4 => operands[arguments[(component.field - 3) as usize] as usize],
                            11 => qualifier,
                            17 => leaves[polarity],
                            _ => 0,
                        },
                        None => 0,
                    }
                };
                product_support[0] = 1;
                if !component.is_text() {
                    match component.field {
                        0 | 1 => product_support[1] = 1,
                        2 => product_support[target as usize] = 1,
                        3 | 4 => {
                            for operand in operands {
                                product_support[operand as usize] = 1;
                            }
                        }
                        11 => product_support[qualifier as usize] = 1,
                        17 => {
                            for leaf in leaves {
                                product_support[leaf as usize] = 1;
                            }
                        }
                        _ => {}
                    }
                }
                logits[choice as usize] = 100.0;
                expected_choices.push(choice);
                SemanticComponentInput {
                    logits,
                    product_support,
                }
            })
            .collect();
        session
            .bind_inputs(
                session.binding(),
                SemanticRngBinding {
                    model_generation: 7,
                    stream_serial: 9,
                    family_id: 1,
                    proposal: 0,
                },
                &inputs,
            )
            .unwrap();
        session.capture().unwrap();
        let before = session.host_io_stats();
        session.launch().unwrap();
        assert_eq!(session.host_io_stats(), before, "{name}: hot host I/O");
        let observation = session.observe(0).unwrap().into_published().unwrap();
        assert_eq!(observation.next_proposal, 1, "{name}");
        assert_eq!(observation.components.len(), 136, "{name}");
        let after_observation = session.host_io_stats();
        assert!(matches!(
            session.bind_inputs(
                session.binding(),
                SemanticRngBinding {
                    model_generation: 7,
                    stream_serial: 9,
                    family_id: 1,
                    proposal: 1,
                },
                &inputs,
            ),
            Err(SemanticTransitionError::UnpublishedTask)
        ));
        assert!(matches!(
            session.bind_task_evaluation(task_spec()),
            Err(SemanticTransitionError::UnpublishedTask)
        ));
        assert_eq!(session.host_io_stats(), after_observation, "{name}");
        assert_eq!(session.task_evaluation_identity(), Some(binding));
        for (ordinal, receipt) in observation.components.iter().enumerate() {
            assert_eq!(receipt.ordinal as usize, ordinal, "{name}");
            assert_eq!(
                receipt.choice, expected_choices[ordinal],
                "{name}, {ordinal}"
            );
        }
        let evaluation = observation.task_evaluation.as_ref().unwrap();
        assert_eq!(evaluation.binding, binding, "{name}");
        assert_eq!(evaluation.return_value, expected_return, "{name}");
        assert_eq!(evaluation.query_count, expected_queries, "{name}");
        assert_eq!(evaluation.facts[0].correct, [0, 0, 0], "{name}");
        assert_eq!(evaluation.facts[0].v, 0, "{name}");
        assert_eq!(evaluation.winner, u64::from(expected_return > 0), "{name}");
        for (index, lane) in observation.lanes.iter().enumerate() {
            assert_eq!(lane.task_refusal, refusals[index], "{name}, lane {index}");
            assert_eq!(lane.root.is_none(), refusals[index].is_some(), "{name}");
            if let Some(root) = &lane.root {
                assert!(root.is_ok(), "{name}, lane {index}: {root:?}");
            }
            assert_eq!(
                (
                    lane.work.edit_commands,
                    lane.work.added_supports,
                    lane.work.defined_truth_changes
                ),
                work[index],
                "{name}, lane {index}"
            );
        }
        if expected_return > 0 {
            let facts = evaluation.facts[1];
            assert_eq!(
                (facts.correct, facts.g, facts.p, facts.c, facts.v),
                ([1, 1, 1], 3, 1, 4, 45)
            );
        }
        drop(session);
        provider.memory().reap_pending_deallocations().unwrap();
        assert_eq!(provider.memory().allocated_bytes(), 0, "{name}");
        assert_eq!(provider.memory().deallocation_failure_bytes(), 0, "{name}");
    }

    let mut session = new_session();
    let original = session.bind_task_evaluation(task_spec()).unwrap();
    let original_epoch = session.task_evaluation_epoch();
    let mut failing = task_spec();
    failing.program = Arc::new(FailingProgram);
    assert!(matches!(session.bind_task_evaluation(failing),
        Err(SemanticTransitionError::InvalidInput { detail }) if detail == "observer failed"));
    assert!(
        session.is_poisoned(),
        "observer failure must prevent reuse of the native Session"
    );
    assert_eq!(session.task_evaluation_identity(), Some(original));
    assert_eq!(
        session.task_evaluation_epoch(),
        original_epoch,
        "failed observation must not issue a new task generation"
    );
    assert!(matches!(
        session.bind_task_evaluation(task_spec()),
        Err(SemanticTransitionError::Poisoned)
    ));
    assert!(matches!(
        session.capture(),
        Err(SemanticTransitionError::Poisoned)
    ));
    drop(session);
    provider.memory().reap_pending_deallocations().unwrap();
    assert_eq!(provider.memory().allocated_bytes(), 0);
    assert_eq!(provider.memory().deallocation_failure_bytes(), 0);
}

fn philox_reference(mut counter: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    for _ in 0..10 {
        let products = [
            u64::from(counter[0]) * 0xd2511f53,
            u64::from(counter[2]) * 0xcd9e8d57,
        ];
        counter = [
            (products[1] >> 32) as u32 ^ counter[1] ^ key[0],
            products[1] as u32,
            (products[0] >> 32) as u32 ^ counter[3] ^ key[1],
            products[0] as u32,
        ];
        key[0] = key[0].wrapping_add(0x9e3779b9);
        key[1] = key[1].wrapping_add(0xbb67ae85);
    }
    counter
}

fn limbs_shift(value: u128, shift: usize) -> [u64; 3] {
    let mut result = [0; 3];
    for bit in 0..128 {
        if value & (1u128 << bit) != 0 {
            assert!(bit + shift < 192);
            result[(bit + shift) / 64] |= 1 << ((bit + shift) % 64);
        }
    }
    result
}

fn round_even(numerator: u128, denominator: u128) -> u64 {
    let (quotient, remainder) = (numerator / denominator, numerator % denominator);
    (quotient
        + u128::from(
            2 * remainder > denominator || (2 * remainder == denominator && quotient % 2 == 1),
        )) as u64
}

// Independent reference for FP32 fixtures whose exact differences are multiples
// of 2^-31: factor 2^118 out before projection, so all arithmetic fits u128.
// CUDA retains the full FP32 denominator and uses three-limb arithmetic instead.
fn check_receipt(r: &xlog_cuda::SemanticTransitionReceipt, input: &SemanticComponentInput) {
    assert_eq!(r.random_words, philox_reference(r.counter, r.key));
    assert_eq!(
        r.draw,
        (u64::from(r.random_words[0]) | (u64::from(r.random_words[1]) << 32)) & ((1 << 63) - 1)
    );
    let legal: Vec<usize> = input
        .product_support
        .iter()
        .enumerate()
        .filter(|(i, m)| **m == 1 && (r.kind == 0 || *i == 0))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(r.legal_count as usize, legal.len());
    if r.kind != 0 {
        assert_eq!((r.active_fields, r.null_fields), (1, 0x3fffe));
    } else {
        assert_eq!((r.active_fields, r.null_fields), (0, 0));
    }
    if legal.len() == 1 {
        assert_eq!(r.choice as usize, legal[0]);
        assert_eq!(r.p, [1, 0, 0]);
        assert_eq!(r.q, [1, 0, 0]);
        assert_eq!(r.factor_denominator, [1, 0, 0]);
        assert_eq!((r.cdf_start, r.cdf_end, r.mass), (0, 1 << 63, 1 << 63));
        return;
    }
    let maximum = legal
        .iter()
        .map(|&i| f64::from(input.logits[i]))
        .fold(f64::NEG_INFINITY, f64::max);
    let d: Vec<u128> = legal
        .iter()
        .map(|&i| {
            let scaled =
                (maximum - f64::from(input.logits[i])).clamp(0., 16.) * (1u64 << 31) as f64;
            assert_eq!(scaled.fract(), 0.0);
            scaled as u128
        })
        .collect();
    let mut sorted = d.clone();
    sorted.sort();
    let g = (1u128 << 31) - legal.len() as u128;
    let (mut sum, mut active_sum, mut h) = (0u128, 0u128, 0u128);
    for (i, distance) in sorted.into_iter().enumerate() {
        sum += distance;
        if (i as u128 + 1) * distance < sum + g {
            active_sum = sum;
            h = i as u128 + 1;
        }
    }
    assert_eq!(r.active_count as u128, h);
    let probabilities: Vec<_> = d
        .iter()
        .map(|distance| h + (active_sum + g).saturating_sub(h * distance))
        .collect();
    assert_eq!(probabilities.iter().sum::<u128>(), h << 31);
    let (mut prefix, mut lower, mut seen) = (0u128, 0u64, false);
    for (&choice, p) in legal.iter().zip(probabilities) {
        prefix += p;
        let upper = round_even(prefix << 32, h);
        assert!(upper - lower >= 1 << 32);
        if lower <= r.draw && r.draw < upper {
            seen = true;
            assert_eq!(r.choice as usize, choice);
            assert_eq!(r.p, limbs_shift(p, 118));
            assert_eq!(r.q, limbs_shift(h, 149));
            assert_eq!(
                (r.cdf_start, r.cdf_end, r.mass),
                (lower, upper, upper - lower)
            );
            assert_eq!(
                r.factor_denominator,
                limbs_shift(h * u128::from(upper - lower), 86)
            );
        }
        lower = upper;
    }
    assert!(seen);
    assert_eq!(lower, 1 << 63);
}

#[test]
fn public_catalogue_has_one_canonical_identity_and_exact_roster() {
    use sha2::{Digest, Sha256};
    let catalogue = SemanticActionCatalogue::current();
    let source = include_str!("../src/semantic_action_catalogue_v1.def");
    let normalized = source
        .lines()
        .filter_map(|line| {
            let line = line
                .split('#')
                .next()
                .unwrap()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            (!line.is_empty()).then_some(line)
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    assert_eq!(catalogue.canonical_bytes(), normalized.as_bytes());
    let digest: [u8; 32] = Sha256::digest(normalized.as_bytes()).into();
    assert_eq!(catalogue.binding().digest.as_bytes(), &digest);
    assert_eq!(
        digest,
        [
            0x45, 0x7e, 0x09, 0xcc, 0x05, 0x11, 0x82, 0x61, 0x08, 0x7d, 0x20, 0xf0, 0x95, 0xa0,
            0x8c, 0xbf, 0x65, 0xc8, 0xf9, 0x40, 0x65, 0x63, 0xc2, 0xa6, 0x04, 0x00, 0x53, 0x26,
            0xc0, 0x0d, 0xf8, 0xf0
        ]
    );
    assert_eq!(catalogue.binding().generation, 3);
    assert!(normalized.starts_with("domain xlog.semantic-action-catalogue.v1\ngeneration 3\n"));
    assert_eq!(catalogue.text_cardinality(), 248077);
    assert_eq!(catalogue.scratch_bytes(), 24 * 1024 * 1024);
    assert_eq!(catalogue.field_masks(), [(1, 0x3fffe), (0x2087f, 0x1f780)]);
    assert_eq!(catalogue.fields().len(), 18);
    let names: Vec<_> = catalogue.fields().iter().map(|f| f.name).collect();
    assert_eq!(
        names,
        [
            "action_class",
            "opcode",
            "target_handle",
            "operand_0",
            "operand_1",
            "operand_2",
            "operand_3",
            "binding_0",
            "binding_1",
            "binding_2",
            "binding_3",
            "qualifier_bundle",
            "guard_descriptor",
            "edge_TRUE",
            "edge_FALSE",
            "edge_BOTH",
            "edge_NEITHER",
            "leaf_payload"
        ]
    );
    for (i, field) in catalogue.fields().iter().enumerate() {
        assert_eq!(field.ordinal as usize, i);
        assert_eq!(field.cardinality, if i < 2 { 2 } else { 1 });
        assert!(!field.role.is_empty());
    }
    assert_eq!(
        catalogue
            .signatures()
            .iter()
            .map(|s| (s.kind, s.value, s.name, s.signature))
            .collect::<Vec<_>>(),
        [
            (
                "action",
                0,
                "NO_EDIT",
                "NO_EDIT_ACTIVE_MASK NO_EDIT_NULL_MASK"
            ),
            ("action", 1, "APPLY", "OPCODE_SIGNATURE OPCODE_SIGNATURE"),
            ("opcode", 0, "NULL", "ONLY_UNDER_NO_EDIT"),
            (
                "opcode",
                1,
                "INSERT_SUPPORT",
                "INSERT_SUPPORT_ACTIVE_MASK INSERT_SUPPORT_NULL_MASK"
            )
        ]
    );
    let mut offset = 0;
    for (start, count, lane, kind, slot) in [
        (0, 32, 1, 0, 0),
        (32, 18, 1, 1, 0),
        (50, 18, 1, 1, 1),
        (68, 32, 2, 0, 0),
        (100, 18, 2, 1, 0),
        (118, 18, 2, 1, 1),
    ] {
        for field in 0..count {
            let c = &catalogue.components()[start + field];
            assert_eq!(
                (c.ordinal, c.lane, c.kind, c.slot, c.field, c.offset),
                (
                    (start + field) as u32,
                    lane,
                    kind,
                    slot,
                    field as u32,
                    offset
                )
            );
            offset += u64::from(c.cardinality);
        }
    }
    let bytes = std::array::from_fn(|i| (17 + i) as u8);
    let identity = xlog_cuda::Identity256::from_bytes(bytes);
    assert_eq!(identity.as_bytes(), &bytes);
    let mut reversed = bytes;
    reversed.reverse();
    assert_ne!(identity, xlog_cuda::Identity256::from_bytes(reversed));
    // Byte order, generation, domain, and logical field order all bind identity.
    for changed in [
        normalized.replacen("generation 3", "generation 4", 1),
        normalized.replacen("catalogue.v1", "catalogue.v2", 1),
        normalized.replacen("field 3 operand_0", "field 3 operand_1", 1),
    ] {
        let changed_digest: [u8; 32] = Sha256::digest(changed.as_bytes()).into();
        assert_ne!(digest, changed_digest);
    }
    assert_eq!(
        philox_reference([0; 4], [0; 2]),
        [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]
    );
}

#[test]
fn public_session_exact_projection_full_vocabulary_and_fp32_edges() {
    if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
        return;
    }
    let provider = CudaProviderBuilder::new(0, MemoryBudget::with_limit(256 * 1024 * 1024))
        .with_stream_capacity(1)
        .build()
        .unwrap();
    let provider = Arc::new(provider);
    let runtime = Arc::clone(provider.memory().runtime().unwrap());
    let id = runtime.stream_pool().acquire().unwrap();
    let stream = runtime.stream_pool().resolve(id).unwrap();
    let domain = provider
        .bind_resident_execution_domain(runtime, id, stream)
        .unwrap();
    let mut session = SemanticTransitionSession::new(&provider, &domain).unwrap();
    let mut inputs: Vec<_> = session
        .components()
        .iter()
        .map(|c| {
            let mut input = SemanticComponentInput {
                logits: vec![0.; c.cardinality as usize],
                product_support: vec![0; c.cardinality as usize],
            };
            input.product_support[0] = 1;
            input
        })
        .collect();
    // Whole dictionary exercises the true bound and original-category CDF order.
    inputs[0].product_support.fill(1);
    for (i, z) in inputs[0].logits.iter_mut().enumerate() {
        *z = (i % 19) as f32 / 32.;
    }
    let cases: &[&[f32]] = &[
        &[0., 0., 0.],
        &[0., -0., 0., -0.],
        &[0., 0.25, 0.5, -16.],
        &[f32::MAX, -f32::MAX, 0.],
        &[-f32::MAX, -f32::MAX],
        &[0., -16., -15.999999, -16.000002],
        &[0., -1., -2., -3., -4.],
        &[0., -0.5, -0.5],
        &[0., -1.0 + f32::EPSILON],
    ];
    for (ordinal, values) in (1..).zip(cases) {
        for (i, z) in values.iter().enumerate() {
            inputs[ordinal].logits[i * 7] = *z;
            inputs[ordinal].product_support[i * 7] = 1;
        }
    }
    // Exact subnormal differences matter below the CDF rounding scale, hence
    // check the 192-bit numerator itself rather than only the sampled category.
    inputs[11].logits[0] = -16.;
    inputs[11].logits[1] = 0.;
    inputs[11].logits[2] = f32::from_bits(1);
    inputs[11].product_support[..3].fill(1);
    for (ordinal, multiple) in [(12, 1.), (13, 3.)] {
        inputs[ordinal].logits[1] = multiple * 2.0f32.powi(-63);
        inputs[ordinal].product_support[1] = 1;
    }
    inputs[14].product_support[..128].fill(1);
    inputs[14].logits[1] = -(1. - 2.0f32.powi(-24));
    inputs[14].logits[2..128].fill(-16.);
    let rng = SemanticRngBinding {
        model_generation: 0x12345678,
        stream_serial: (1 << 56) - 1,
        family_id: 255,
        proposal: u32::MAX,
    };
    session
        .bind_inputs(session.binding(), rng, &inputs)
        .unwrap();
    session.capture().unwrap();
    let before = session.host_io_stats();
    session.launch().unwrap();
    assert_eq!(before, session.host_io_stats());
    let observation = session.observe(u32::MAX).unwrap().into_published().unwrap();
    for (i, r) in observation.components.iter().enumerate() {
        if !(11..=13).contains(&i) {
            check_receipt(r, &inputs[i]);
        }
    }
    let r = &observation.components[11];
    assert_eq!(
        observation.components[14].active_count, 1,
        "the exact zero projection mass is not active"
    );
    assert_eq!(r.active_count, 2);
    assert_eq!(r.legal_count, 3);
    assert_eq!(r.q, limbs_shift(2, 149));
    let p_base = limbs_shift((1u128 << 31) - 1, 118);
    let mut expected = p_base;
    match r.choice {
        0 => expected = limbs_shift(2, 118),
        1 => {
            expected[0] = u64::MAX;
            expected[1] -= 1;
        }
        2 => expected[0] += 1,
        other => panic!("unexpected category {other}"),
    }
    assert_eq!(r.p, expected);
    assert_eq!(r.random_words, philox_reference(r.counter, r.key));
    for (ordinal, multiple, boundary) in [(12, 1u64, 1u64 << 62), (13, 3u64, (1u64 << 62) - 2)] {
        let r = &observation.components[ordinal];
        let mut expected = limbs_shift(1, 149);
        if r.choice == 0 {
            expected[0] = 0;
            expected[1] = 0u64.wrapping_sub(multiple << 22);
            expected[2] -= 1;
        } else {
            assert_eq!(r.choice, 1);
            expected[1] += multiple << 22;
        }
        assert_eq!(r.p, expected);
        assert_eq!(r.q, limbs_shift(2, 149));
        assert_eq!(
            (r.cdf_start, r.cdf_end),
            if r.choice == 0 {
                (0, boundary)
            } else {
                (boundary, 1 << 63)
            }
        );
        assert_eq!(
            r.factor_denominator,
            limbs_shift(2 * u128::from(r.mass), 86)
        );
    }
    assert_eq!(observation.next_proposal, 1u64 << 32);
    assert!(matches!(
        session.launch(),
        Err(SemanticTransitionError::GenerationExhausted)
    ));
    assert!(!session.is_poisoned());
}

#[test]
fn public_session_preflight_overlap_and_observation_failure_prevent_reuse() {
    if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
        return;
    }
    let provider = CudaProviderBuilder::new(0, MemoryBudget::with_limit(256 * 1024 * 1024))
        .with_stream_capacity(1)
        .build()
        .unwrap();
    let provider = Arc::new(provider);
    let runtime = Arc::clone(provider.memory().runtime().unwrap());
    let id = runtime.stream_pool().acquire().unwrap();
    let stream = runtime.stream_pool().resolve(id).unwrap();
    let domain = provider
        .bind_resident_execution_domain(runtime, id, stream)
        .unwrap();
    let mut session = SemanticTransitionSession::new(&provider, &domain).unwrap();
    let mut inputs: Vec<_> = session
        .components()
        .iter()
        .map(|c| {
            let mut mask = vec![0; c.cardinality as usize];
            mask[0] = 1;
            SemanticComponentInput {
                logits: vec![0.; mask.len()],
                product_support: mask,
            }
        })
        .collect();
    let rng = SemanticRngBinding {
        model_generation: 5,
        stream_serial: 6,
        family_id: 7,
        proposal: 8,
    };
    assert!(matches!(
        session.observe(8),
        Err(SemanticTransitionError::NoPendingLaunch)
    ));
    assert!(matches!(
        session.capture(),
        Err(SemanticTransitionError::NotBound)
    ));
    assert!(matches!(
        session.launch(),
        Err(SemanticTransitionError::NotCaptured)
    ));
    let before = session.host_io_stats();
    let mut binding = session.binding();
    binding.generation += 1;
    assert!(matches!(
        session.bind_inputs(binding, rng, &inputs),
        Err(SemanticTransitionError::CatalogueMismatch)
    ));
    binding = session.binding();
    binding.digest = xlog_cuda::Identity256::from_bytes([0; 32]);
    assert!(matches!(
        session.bind_inputs(binding, rng, &inputs),
        Err(SemanticTransitionError::CatalogueMismatch)
    ));
    inputs[32].product_support = [1, 1].to_vec(); // APPLY is masked, not sampled.
    inputs[0].logits[0] = f32::NAN;
    assert!(matches!(
        session.bind_inputs(session.binding(), rng, &inputs),
        Err(SemanticTransitionError::InvalidInput { .. })
    ));
    inputs[0].logits[0] = 0.;
    inputs[0].product_support[0] = 0;
    assert!(matches!(
        session.bind_inputs(session.binding(), rng, &inputs),
        Err(SemanticTransitionError::InvalidInput { .. })
    ));
    inputs[0].product_support[0] = 2;
    assert!(matches!(
        session.bind_inputs(session.binding(), rng, &inputs),
        Err(SemanticTransitionError::InvalidInput { .. })
    ));
    inputs[0].product_support[0] = 1;
    let invalid_rng = SemanticRngBinding {
        stream_serial: 1 << 56,
        ..rng
    };
    assert!(matches!(
        session.bind_inputs(session.binding(), invalid_rng, &inputs),
        Err(SemanticTransitionError::InvalidInput { .. })
    ));
    assert_eq!(
        session.host_io_stats(),
        before,
        "preflight rejection must precede device writes"
    );
    session
        .bind_inputs(session.binding(), rng, &inputs)
        .unwrap();
    session.capture().unwrap();
    session.launch().unwrap();
    assert!(matches!(
        session.bind_inputs(session.binding(), rng, &inputs),
        Err(SemanticTransitionError::OverlappingLaunch)
    ));
    assert!(matches!(
        session.capture(),
        Err(SemanticTransitionError::OverlappingLaunch)
    ));
    assert!(matches!(
        session.root_snapshot(),
        Err(SemanticTransitionError::OverlappingLaunch)
    ));
    assert!(matches!(
        session.launch(),
        Err(SemanticTransitionError::OverlappingLaunch)
    ));
    assert!(matches!(
        session.observe(9),
        Err(SemanticTransitionError::ObservationMismatch)
    ));
    assert!(session.is_poisoned());
    assert!(matches!(
        session.launch(),
        Err(SemanticTransitionError::Poisoned)
    ));
    assert!(matches!(
        session.observe(8),
        Err(SemanticTransitionError::Poisoned)
    ));
    assert!(matches!(
        session.bind_inputs(session.binding(), rng, &inputs),
        Err(SemanticTransitionError::Poisoned)
    ));
}
