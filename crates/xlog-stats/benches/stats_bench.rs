//! Performance benchmarks for the xlog-stats crate.
//!
//! Run with: `cargo bench -p xlog-stats`
//!
//! These benchmarks measure the performance of:
//! - Relation registration
//! - Cardinality updates
//! - Join estimation
//! - Heat tracking

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use xlog_core::{RelId, ScalarType};
use xlog_stats::{ColumnStats, JoinSelectivity, RelationStats, StatsManager};

// =============================================================================
// StatsManager Benchmarks - Registration
// =============================================================================

fn bench_register_relation(c: &mut Criterion) {
    let mut group = c.benchmark_group("register_relation");

    for num_relations in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*num_relations as u64));
        group.bench_with_input(
            BenchmarkId::new("relations", num_relations),
            num_relations,
            |b, &num| {
                b.iter(|| {
                    let mut mgr = StatsManager::new();
                    for i in 0..num {
                        mgr.register_relation(black_box(RelId(i as u32)));
                    }
                    mgr
                });
            },
        );
    }

    group.finish();
}

fn bench_register_relation_idempotent(c: &mut Criterion) {
    let mut group = c.benchmark_group("register_relation_idempotent");

    // Measure cost of re-registering existing relations
    for num_relations in [100, 1000].iter() {
        let mut mgr = StatsManager::new();
        for i in 0..*num_relations {
            mgr.register_relation(RelId(i as u32));
        }

        group.throughput(Throughput::Elements(*num_relations as u64));
        group.bench_with_input(
            BenchmarkId::new("existing", num_relations),
            num_relations,
            |b, &num| {
                b.iter(|| {
                    for i in 0..num {
                        mgr.register_relation(black_box(RelId(i as u32)));
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_unregister_relation(c: &mut Criterion) {
    let mut group = c.benchmark_group("unregister_relation");

    for num_relations in [100, 500].iter() {
        group.bench_with_input(
            BenchmarkId::new("relations", num_relations),
            num_relations,
            |b, &num| {
                b.iter_batched(
                    || {
                        let mut mgr = StatsManager::new();
                        for i in 0..num {
                            mgr.register_relation(RelId(i as u32));
                        }
                        mgr
                    },
                    |mut mgr| {
                        for i in 0..num {
                            mgr.unregister_relation(black_box(RelId(i as u32)));
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// =============================================================================
// StatsManager Benchmarks - Cardinality Updates
// =============================================================================

fn bench_update_cardinality(c: &mut Criterion) {
    let mut group = c.benchmark_group("update_cardinality");

    for num_relations in [100, 1000].iter() {
        let mut mgr = StatsManager::new();
        for i in 0..*num_relations {
            mgr.register_relation(RelId(i as u32));
        }

        group.throughput(Throughput::Elements(*num_relations as u64));
        group.bench_with_input(
            BenchmarkId::new("relations", num_relations),
            num_relations,
            |b, &num| {
                b.iter(|| {
                    for i in 0..num {
                        mgr.update_cardinality(black_box(RelId(i as u32)), black_box(10000));
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_update_byte_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("update_byte_size");

    for num_relations in [100, 1000].iter() {
        let mut mgr = StatsManager::new();
        for i in 0..*num_relations {
            mgr.register_relation(RelId(i as u32));
        }

        group.throughput(Throughput::Elements(*num_relations as u64));
        group.bench_with_input(
            BenchmarkId::new("relations", num_relations),
            num_relations,
            |b, &num| {
                b.iter(|| {
                    for i in 0..num {
                        mgr.update_byte_size(black_box(RelId(i as u32)), black_box(1024 * 1024));
                    }
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// StatsManager Benchmarks - Join Estimation
// =============================================================================

fn bench_estimate_join_cardinality_no_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("estimate_join_cardinality_no_cache");

    for num_relations in [10, 50, 100].iter() {
        let mut mgr = StatsManager::new();
        for i in 0..*num_relations {
            mgr.register_relation(RelId(i as u32));
            mgr.update_cardinality(RelId(i as u32), (i as u64 + 1) * 1000);
        }

        // Number of possible join pairs
        let num_pairs = (*num_relations * (*num_relations - 1)) / 2;
        group.throughput(Throughput::Elements(num_pairs as u64));

        group.bench_with_input(
            BenchmarkId::new("pairs", num_pairs),
            num_relations,
            |b, &num| {
                b.iter(|| {
                    let mut total = 0u64;
                    for i in 0..num {
                        for j in (i + 1)..num {
                            total += mgr.estimate_join_cardinality(
                                black_box(RelId(i as u32)),
                                black_box(RelId(j as u32)),
                                black_box(&[0]),
                                black_box(&[0]),
                            );
                        }
                    }
                    total
                });
            },
        );
    }

    group.finish();
}

fn bench_estimate_join_cardinality_with_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("estimate_join_cardinality_with_cache");

    for num_relations in [10, 50, 100].iter() {
        let mut mgr = StatsManager::new();
        for i in 0..*num_relations {
            mgr.register_relation(RelId(i as u32));
            mgr.update_cardinality(RelId(i as u32), (i as u64 + 1) * 1000);
        }

        // Pre-populate join selectivity cache
        for i in 0..*num_relations {
            for j in (i + 1)..*num_relations {
                mgr.record_join_result(
                    RelId(i as u32),
                    RelId(j as u32),
                    vec![0],
                    vec![0],
                    1000 * 2000,
                    500,
                );
            }
        }

        let num_pairs = (*num_relations * (*num_relations - 1)) / 2;
        group.throughput(Throughput::Elements(num_pairs as u64));

        group.bench_with_input(
            BenchmarkId::new("pairs", num_pairs),
            num_relations,
            |b, &num| {
                b.iter(|| {
                    let mut total = 0u64;
                    for i in 0..num {
                        for j in (i + 1)..num {
                            total += mgr.estimate_join_cardinality(
                                black_box(RelId(i as u32)),
                                black_box(RelId(j as u32)),
                                black_box(&[0]),
                                black_box(&[0]),
                            );
                        }
                    }
                    total
                });
            },
        );
    }

    group.finish();
}

fn bench_estimate_join_with_column_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("estimate_join_with_column_stats");

    for num_relations in [10, 50].iter() {
        let mut mgr = StatsManager::new();
        for i in 0..*num_relations {
            mgr.register_relation(RelId(i as u32));
            mgr.update_cardinality(RelId(i as u32), (i as u64 + 1) * 1000);

            // Add column stats
            let mut col_stats = ColumnStats::new(0, ScalarType::I64);
            col_stats.update_distinct((i as u64 + 1) * 100);
            mgr.add_column_stats(RelId(i as u32), col_stats);
        }

        let num_pairs = (*num_relations * (*num_relations - 1)) / 2;
        group.throughput(Throughput::Elements(num_pairs as u64));

        group.bench_with_input(
            BenchmarkId::new("pairs", num_pairs),
            num_relations,
            |b, &num| {
                b.iter(|| {
                    let mut total = 0u64;
                    for i in 0..num {
                        for j in (i + 1)..num {
                            total += mgr.estimate_join_cardinality(
                                black_box(RelId(i as u32)),
                                black_box(RelId(j as u32)),
                                black_box(&[0]),
                                black_box(&[0]),
                            );
                        }
                    }
                    total
                });
            },
        );
    }

    group.finish();
}

fn bench_record_join_result(c: &mut Criterion) {
    let mut group = c.benchmark_group("record_join_result");

    for num_joins in [100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*num_joins as u64));
        group.bench_with_input(
            BenchmarkId::new("joins", num_joins),
            num_joins,
            |b, &num| {
                b.iter_batched(
                    || {
                        let mut mgr = StatsManager::new();
                        for i in 0..100 {
                            mgr.register_relation(RelId(i));
                        }
                        mgr
                    },
                    |mut mgr| {
                        for i in 0..num {
                            let left = RelId((i % 100) as u32);
                            let right = RelId(((i + 1) % 100) as u32);
                            mgr.record_join_result(
                                black_box(left),
                                black_box(right),
                                black_box(vec![0]),
                                black_box(vec![0]),
                                black_box(1000000),
                                black_box(5000),
                            );
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// =============================================================================
// StatsManager Benchmarks - Heat Tracking
// =============================================================================

fn bench_record_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("record_access");

    for num_relations in [100, 1000].iter() {
        let mut mgr = StatsManager::new();
        for i in 0..*num_relations {
            mgr.register_relation(RelId(i as u32));
        }

        group.throughput(Throughput::Elements(*num_relations as u64));
        group.bench_with_input(
            BenchmarkId::new("relations", num_relations),
            num_relations,
            |b, &num| {
                b.iter(|| {
                    for i in 0..num {
                        mgr.record_access(black_box(RelId(i as u32)));
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_decay_all_heat(c: &mut Criterion) {
    let mut group = c.benchmark_group("decay_all_heat");

    for num_relations in [100, 1000, 10000].iter() {
        let mut mgr = StatsManager::new();
        for i in 0..*num_relations {
            mgr.register_relation(RelId(i as u32));
            // Heat up each relation
            for _ in 0..10 {
                mgr.record_access(RelId(i as u32));
            }
        }

        group.throughput(Throughput::Elements(*num_relations as u64));
        group.bench_with_input(
            BenchmarkId::new("relations", num_relations),
            num_relations,
            |b, _| {
                b.iter(|| {
                    mgr.decay_all_heat(black_box(0.9));
                });
            },
        );
    }

    group.finish();
}

fn bench_hot_relations(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_relations");

    for num_relations in [100, 1000, 10000].iter() {
        let mut mgr = StatsManager::new();
        for i in 0..*num_relations {
            mgr.register_relation(RelId(i as u32));
            // Heat up ~20% of relations more than others
            if i % 5 == 0 {
                for _ in 0..20 {
                    mgr.record_access(RelId(i as u32));
                }
            } else {
                mgr.record_access(RelId(i as u32));
            }
        }

        group.throughput(Throughput::Elements(*num_relations as u64));
        group.bench_with_input(
            BenchmarkId::new("relations", num_relations),
            num_relations,
            |b, _| {
                b.iter(|| mgr.hot_relations(black_box(0.5)));
            },
        );
    }

    group.finish();
}

fn bench_cold_relations(c: &mut Criterion) {
    let mut group = c.benchmark_group("cold_relations");

    for num_relations in [100, 1000, 10000].iter() {
        let mut mgr = StatsManager::new();
        for i in 0..*num_relations {
            mgr.register_relation(RelId(i as u32));
            if i % 5 == 0 {
                for _ in 0..20 {
                    mgr.record_access(RelId(i as u32));
                }
            }
        }

        group.throughput(Throughput::Elements(*num_relations as u64));
        group.bench_with_input(
            BenchmarkId::new("relations", num_relations),
            num_relations,
            |b, _| {
                b.iter(|| mgr.cold_relations(black_box(0.1)));
            },
        );
    }

    group.finish();
}

// =============================================================================
// RelationStats Benchmarks
// =============================================================================

fn bench_relation_stats_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("relation_stats_creation");

    group.bench_function("new", |b| {
        b.iter(|| RelationStats::new(black_box(RelId(42))));
    });

    group.finish();
}

fn bench_relation_stats_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("relation_stats_operations");

    let mut stats = RelationStats::new(RelId(1));

    group.bench_function("update_cardinality", |b| {
        b.iter(|| stats.update_cardinality(black_box(10000)));
    });

    group.bench_function("update_byte_size", |b| {
        b.iter(|| stats.update_byte_size(black_box(1024 * 1024)));
    });

    group.bench_function("record_access", |b| {
        b.iter(|| stats.record_access());
    });

    group.bench_function("decay_heat", |b| {
        b.iter(|| stats.decay_heat(black_box(0.9)));
    });

    group.bench_function("estimate_selectivity", |b| {
        stats.update_cardinality(1000);
        b.iter(|| stats.estimate_selectivity(black_box(100)));
    });

    group.finish();
}

fn bench_relation_stats_column_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("relation_stats_column_operations");

    for num_columns in [5, 20, 100].iter() {
        let mut stats = RelationStats::new(RelId(1));
        for i in 0..*num_columns {
            stats.add_column(ColumnStats::new(i, ScalarType::I64));
        }

        group.bench_with_input(
            BenchmarkId::new("get_column", num_columns),
            num_columns,
            |b, &num| {
                b.iter(|| {
                    for i in 0..num {
                        black_box(stats.get_column(i));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("add_column", num_columns),
            num_columns,
            |b, &num| {
                b.iter_batched(
                    || RelationStats::new(RelId(1)),
                    |mut s| {
                        for i in 0..num {
                            s.add_column(ColumnStats::new(i, ScalarType::I64));
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// =============================================================================
// ColumnStats Benchmarks
// =============================================================================

fn bench_column_stats_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("column_stats_creation");

    let types = [
        ("U32", ScalarType::U32),
        ("U64", ScalarType::U64),
        ("I32", ScalarType::I32),
        ("I64", ScalarType::I64),
        ("F32", ScalarType::F32),
        ("F64", ScalarType::F64),
        ("Bool", ScalarType::Bool),
        ("Symbol", ScalarType::Symbol),
    ];

    for (name, dtype) in types.iter() {
        group.bench_with_input(BenchmarkId::new("type", name), dtype, |b, &dtype| {
            b.iter(|| ColumnStats::new(black_box(0), black_box(dtype)));
        });
    }

    group.finish();
}

fn bench_column_stats_selectivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("column_stats_selectivity");

    let mut col_stats = ColumnStats::new(0, ScalarType::I64);
    col_stats.update_distinct(1000);
    col_stats.update_range(0, 10000);

    group.bench_function("equality_selectivity", |b| {
        b.iter(|| col_stats.equality_selectivity(black_box(10000)));
    });

    group.bench_function("range_selectivity", |b| {
        b.iter(|| col_stats.range_selectivity(black_box(1000), black_box(5000)));
    });

    group.bench_function("value_size_bytes", |b| {
        b.iter(|| col_stats.value_size_bytes());
    });

    group.finish();
}

// =============================================================================
// JoinSelectivity Benchmarks
// =============================================================================

fn bench_join_selectivity_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("join_selectivity_creation");

    group.bench_function("new", |b| {
        b.iter(|| JoinSelectivity::new(black_box(RelId(1)), black_box(RelId(2))));
    });

    group.finish();
}

fn bench_join_selectivity_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("join_selectivity_operations");

    let mut js = JoinSelectivity::new(RelId(1), RelId(2));

    group.bench_function("set_keys", |b| {
        b.iter(|| js.set_keys(black_box(vec![0, 1]), black_box(vec![0, 1])));
    });

    group.bench_function("set_selectivity", |b| {
        b.iter(|| js.set_selectivity(black_box(0.01)));
    });

    group.bench_function("estimate_output_rows", |b| {
        js.set_selectivity(0.01);
        b.iter(|| js.estimate_output_rows(black_box(10000), black_box(5000)));
    });

    group.bench_function("estimate_output_rows_pk_fk", |b| {
        js.mark_pk_fk();
        b.iter(|| js.estimate_output_rows(black_box(10000), black_box(5000)));
    });

    group.bench_function("estimate_selectivity_from_stats", |b| {
        b.iter(|| {
            JoinSelectivity::estimate_selectivity_from_stats(black_box(1000), black_box(500))
        });
    });

    group.bench_function("update_from_observation", |b| {
        b.iter(|| js.update_from_observation(black_box(10000), black_box(5000), black_box(25000)));
    });

    group.finish();
}

// =============================================================================
// Aggregate Statistics Benchmarks
// =============================================================================

fn bench_aggregate_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("aggregate_stats");

    for num_relations in [100, 1000, 10000].iter() {
        let mut mgr = StatsManager::new();
        for i in 0..*num_relations {
            mgr.register_relation(RelId(i as u32));
            mgr.update_cardinality(RelId(i as u32), (i as u64 + 1) * 100);
            mgr.update_byte_size(RelId(i as u32), (i as u64 + 1) * 1024);
        }

        group.bench_with_input(
            BenchmarkId::new("total_byte_size", num_relations),
            num_relations,
            |b, _| {
                b.iter(|| mgr.total_byte_size());
            },
        );

        group.bench_with_input(
            BenchmarkId::new("total_cardinality", num_relations),
            num_relations,
            |b, _| {
                b.iter(|| mgr.total_cardinality());
            },
        );

        group.bench_with_input(
            BenchmarkId::new("relation_count", num_relations),
            num_relations,
            |b, _| {
                b.iter(|| mgr.relation_count());
            },
        );

        group.bench_with_input(
            BenchmarkId::new("relation_ids", num_relations),
            num_relations,
            |b, _| {
                b.iter(|| mgr.relation_ids().count());
            },
        );
    }

    group.finish();
}

// =============================================================================
// Criterion Groups
// =============================================================================

criterion_group!(
    registration_benches,
    bench_register_relation,
    bench_register_relation_idempotent,
    bench_unregister_relation,
);

criterion_group!(
    cardinality_benches,
    bench_update_cardinality,
    bench_update_byte_size,
);

criterion_group!(
    join_estimation_benches,
    bench_estimate_join_cardinality_no_cache,
    bench_estimate_join_cardinality_with_cache,
    bench_estimate_join_with_column_stats,
    bench_record_join_result,
);

criterion_group!(
    heat_benches,
    bench_record_access,
    bench_decay_all_heat,
    bench_hot_relations,
    bench_cold_relations,
);

criterion_group!(
    relation_stats_benches,
    bench_relation_stats_creation,
    bench_relation_stats_operations,
    bench_relation_stats_column_operations,
);

criterion_group!(
    column_stats_benches,
    bench_column_stats_creation,
    bench_column_stats_selectivity,
);

criterion_group!(
    join_selectivity_benches,
    bench_join_selectivity_creation,
    bench_join_selectivity_operations,
);

criterion_group!(aggregate_benches, bench_aggregate_stats);

criterion_main!(
    registration_benches,
    cardinality_benches,
    join_estimation_benches,
    heat_benches,
    relation_stats_benches,
    column_stats_benches,
    join_selectivity_benches,
    aggregate_benches
);
