#![allow(clippy::arc_with_non_send_sync)]
//! Real-World Integration Tests for XLOG
//!
//! These tests validate the xlog-logic subsystem against real-world problems:
//!
//! 1. SOCIAL NETWORK ANALYSIS
//!    - Transitive friend-of-friend relationships
//!    - Mutual friend detection
//!    - Friend recommendations
//!    - Influence propagation
//!
//! 2. ROLE-BASED ACCESS CONTROL (RBAC)
//!    - Role hierarchies with inheritance
//!    - Permission derivation
//!    - Conflict detection (separation of duties)
//!
//! 3. SUPPLY CHAIN / BILL OF MATERIALS
//!    - Recursive part dependencies
//!    - Component reachability
//!    - Circular dependency detection
//!
//! 4. PROGRAM ANALYSIS
//!    - Points-to analysis
//!    - Dataflow reachability
//!    - Call graph construction

use std::collections::HashSet;
use std::sync::Arc;

use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LoggingResource, LoggingSink,
    NullSink, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_logic::Compiler;
use xlog_runtime::Executor;

// =============================================================================
// Test Infrastructure
// =============================================================================

/// Default budget for both legacy and runtime fixtures. Sized to
/// the same 1 GiB the legacy path has always used, applied to the
/// local manager budget AND the runtime-side `GlobalDeviceBudget`.
const TEST_BUDGET_BYTES: usize = 1024 * 1024 * 1024;

/// Returns true when the test process should construct providers
/// through the v0.6 device runtime instead of the legacy cudarc
/// path. Controlled by `XLOG_USE_DEVICE_RUNTIME`: any non-empty
/// value other than `"0"` / `"false"` enables the runtime fixture.
///
/// The default is **off** so behavior with the env var unset is
/// bit-for-bit identical to pre-migration, matching the v0.6
/// ground rule of not flipping `CudaKernelProvider` defaults
/// globally.
fn use_device_runtime_fixture() -> bool {
    match std::env::var("XLOG_USE_DEVICE_RUNTIME") {
        Ok(v) => {
            let v = v.trim();
            !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false")
        }
        Err(_) => false,
    }
}

/// Single fixture all tests in this file use. Branches on
/// [`use_device_runtime_fixture`] so the entire test surface
/// runs against either the legacy cudarc allocator or the v0.6
/// runtime stack without per-test code changes.
fn create_test_executor() -> Option<(Executor, Arc<CudaKernelProvider>)> {
    if use_device_runtime_fixture() {
        create_test_executor_with_runtime()
    } else {
        create_test_executor_legacy()
    }
}

fn create_test_executor_legacy() -> Option<(Executor, Arc<CudaKernelProvider>)> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let budget = MemoryBudget::with_limit(TEST_BUDGET_BYTES as u64);
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    let provider = Arc::new(CudaKernelProvider::new(device, memory).ok()?);
    let executor = Executor::new(provider.clone());
    Some((executor, provider))
}

/// Opt-in v0.6 fixture: composes the recommended runtime stack —
/// `GlobalDeviceBudget(LoggingResource(AsyncCudaResource))` — and
/// builds the provider via `GpuMemoryManager::with_runtime` +
/// `CudaKernelProvider::with_runtime`. Returns the same shape as
/// [`create_test_executor_legacy`], so all call sites in this
/// file route through the runtime when the env var is set without
/// any per-test code change.
///
/// Uses `NullSink` rather than `InMemorySink`: this fixture is
/// rebuilt many times per stress loop iteration and the runtime
/// stack keeps the sink alive for as long as `LoggingResource`
/// exists, so retaining records would grow memory unbounded over a
/// 20× stress run with no test consuming them. `real_world_tests`
/// asserts on Datalog query results, not on allocation traces; a
/// discard sink keeps the runtime composition shape (`LoggingResource`
/// is still in the stack) without paying retention cost.
fn create_test_executor_with_runtime() -> Option<(Executor, Arc<CudaKernelProvider>)> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let sink: Arc<dyn LoggingSink> = Arc::new(NullSink::new());

    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
    );
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(LoggingResource::new(async_resource, sink));
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, TEST_BUDGET_BYTES));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));

    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(TEST_BUDGET_BYTES as u64),
        Arc::clone(&runtime),
    ));
    let provider =
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    let executor = Executor::new(provider.clone());
    Some((executor, provider))
}

fn create_edge_buffer(provider: &CudaKernelProvider, edges: &[(u32, u32)]) -> CudaBuffer {
    let schema = Schema::new(vec![
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
    ]);
    if edges.is_empty() {
        return provider
            .create_empty_buffer(schema)
            .expect("create empty buffer");
    }
    let col0: Vec<u8> = edges.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
    let col1: Vec<u8> = edges.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
    provider
        .create_buffer_from_slices(&[&col0, &col1], schema)
        .expect("create buffer")
}

fn create_node_buffer(provider: &CudaKernelProvider, nodes: &[u32]) -> CudaBuffer {
    let schema = Schema::new(vec![("c0".to_string(), ScalarType::U32)]);
    if nodes.is_empty() {
        return provider
            .create_empty_buffer(schema)
            .expect("create empty buffer");
    }
    let col: Vec<u8> = nodes.iter().flat_map(|n| n.to_le_bytes()).collect();
    provider
        .create_buffer_from_slices(&[&col], schema)
        .expect("create buffer")
}

fn create_triple_buffer(provider: &CudaKernelProvider, triples: &[(u32, u32, u32)]) -> CudaBuffer {
    let schema = Schema::new(vec![
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
        ("c2".to_string(), ScalarType::U32),
    ]);
    if triples.is_empty() {
        return provider
            .create_empty_buffer(schema)
            .expect("create empty buffer");
    }
    let col0: Vec<u8> = triples
        .iter()
        .flat_map(|(a, _, _)| a.to_le_bytes())
        .collect();
    let col1: Vec<u8> = triples
        .iter()
        .flat_map(|(_, b, _)| b.to_le_bytes())
        .collect();
    let col2: Vec<u8> = triples
        .iter()
        .flat_map(|(_, _, c)| c.to_le_bytes())
        .collect();
    provider
        .create_buffer_from_slices(&[&col0, &col1, &col2], schema)
        .expect("create buffer")
}

fn read_column_u32(provider: &CudaKernelProvider, buffer: &CudaBuffer, col: usize) -> Vec<u32> {
    if buffer.is_empty() || buffer.column(col).is_none() {
        return vec![];
    }
    provider
        .download_column::<u32>(buffer, col)
        .unwrap_or_default()
}

fn read_pairs(provider: &CudaKernelProvider, buffer: &CudaBuffer) -> Vec<(u32, u32)> {
    let c0 = read_column_u32(provider, buffer, 0);
    let c1 = read_column_u32(provider, buffer, 1);
    c0.into_iter().zip(c1).collect()
}

fn read_triples(provider: &CudaKernelProvider, buffer: &CudaBuffer) -> Vec<(u32, u32, u32)> {
    let c0 = read_column_u32(provider, buffer, 0);
    let c1 = read_column_u32(provider, buffer, 1);
    let c2 = read_column_u32(provider, buffer, 2);
    c0.into_iter()
        .zip(c1)
        .zip(c2)
        .map(|((a, b), c)| (a, b, c))
        .collect()
}

fn setup_facts(executor: &mut Executor, compiler: &Compiler, facts: Vec<(&str, CudaBuffer)>) {
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }
    for (name, buffer) in facts {
        executor.store_mut().put(name, buffer);
    }
}

// =============================================================================
// 1. SOCIAL NETWORK ANALYSIS
// =============================================================================

/// Real-world scenario: Social Network Friend Recommendations
///
/// Problem: Given a social network with friend relationships, compute:
/// - Direct friends
/// - Friends-of-friends (potential recommendations)
/// - Mutual friends between two users
///
/// Network topology:
///   Alice(1) -- Bob(2) -- Carol(3) -- Diana(4)
///       |         |
///       +--- Eve(5) --- Frank(6)
///
/// Expected results:
/// - friends_of_friends(1, 3) = Carol (via Bob)
/// - friends_of_friends(1, 6) = Frank (via Eve)
/// - mutual_friends(1, 3) = Bob (common friend of Alice and Carol)
#[test]
fn test_social_network_friend_recommendations() {
    let (mut executor, provider) = match create_test_executor() {
        Some(e) => e,
        None => {
            eprintln!("Skipping: no CUDA device");
            return;
        }
    };

    // User IDs: Alice=1, Bob=2, Carol=3, Diana=4, Eve=5, Frank=6
    // NOTE: GPU diff only supports single-column buffers, so we test
    // friends-of-friends without the negation filter (which would need
    // multi-column diff). The fof relation still demonstrates the core
    // recursive join functionality.
    let source = r#"
        friend(1, 2).
        friend(2, 1).
        friend(2, 3).
        friend(3, 2).
        friend(3, 4).
        friend(4, 3).
        friend(1, 5).
        friend(5, 1).
        friend(2, 5).
        friend(5, 2).
        friend(5, 6).
        friend(6, 5).

        // Friends-of-friends (excluding self-loops)
        fof(X, Z) :- friend(X, Y), friend(Y, Z), X != Z.

        // Mutual friends: M is a mutual friend of X and Y
        mutual(X, Y, M) :- friend(X, M), friend(Y, M), X != Y.
    "#;

    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("Compilation failed");

    let friends = vec![
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
        (3, 4),
        (4, 3),
        (1, 5),
        (5, 1),
        (2, 5),
        (5, 2),
        (5, 6),
        (6, 5),
    ];
    let friend_buffer = create_edge_buffer(&provider, &friends);
    setup_facts(&mut executor, &compiler, vec![("friend", friend_buffer)]);

    executor.execute_plan(&plan).expect("Execution failed");

    // Check friends-of-friends for Alice (user 1)
    if let Some(fof) = executor.store().get("fof") {
        let fofs = read_pairs(&provider, fof);
        let alice_fofs: HashSet<u32> = fofs
            .iter()
            .filter(|(from, _)| *from == 1)
            .map(|(_, to)| *to)
            .collect();

        println!("Alice's friends-of-friends: {:?}", alice_fofs);

        // Alice -> Bob -> Carol, so Carol (3) is fof
        assert!(
            alice_fofs.contains(&3),
            "Carol (3) should be Alice's friend-of-friend (via Bob)"
        );
        // Alice -> Eve -> Frank, so Frank (6) is fof
        assert!(
            alice_fofs.contains(&6),
            "Frank (6) should be Alice's friend-of-friend (via Eve)"
        );
        // Alice -> Bob -> Eve (back-link), so Eve appears in fof even though direct friend
        // This is expected behavior - fof doesn't exclude direct friends
    }

    // Check mutual friends
    if let Some(mutual) = executor.store().get("mutual") {
        let mutuals = read_triples(&provider, mutual);

        // Bob (2) should be a mutual friend of Alice (1) and Carol (3)
        let alice_carol_mutuals: HashSet<u32> = mutuals
            .iter()
            .filter(|(x, y, _)| (*x == 1 && *y == 3) || (*x == 3 && *y == 1))
            .map(|(_, _, m)| *m)
            .collect();

        println!(
            "Mutual friends of Alice and Carol: {:?}",
            alice_carol_mutuals
        );

        // Bob is mutual friend
        assert!(
            alice_carol_mutuals.contains(&2),
            "Bob should be mutual friend of Alice and Carol"
        );
    }
}

/// Real-world scenario: Influence Propagation in Social Network
///
/// Problem: Model how information or influence spreads through a network.
/// Using transitive closure to compute all nodes reachable from a source.
///
/// This is the classic transitive closure problem applied to influence.
#[test]
fn test_influence_propagation() {
    let (mut executor, provider) = match create_test_executor() {
        Some(e) => e,
        None => {
            eprintln!("Skipping: no CUDA device");
            return;
        }
    };

    // Compute transitive reachability from node 1
    // Network: 1 -> 2 -> 3 -> 4 -> 5
    let source = r#"
        // Network edges: 1 -> 2 -> 3 -> 4 -> 5
        follows(1, 2).
        follows(2, 3).
        follows(3, 4).
        follows(4, 5).

        // Reachability from any node to any other
        reaches(X, Y) :- follows(X, Y).
        reaches(X, Z) :- reaches(X, Y), follows(Y, Z).

        // All nodes that node 1 can reach
        from_one(Y) :- reaches(1, Y).
    "#;

    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("Compilation failed");

    let follows = vec![(1, 2), (2, 3), (3, 4), (4, 5)];
    let follows_buffer = create_edge_buffer(&provider, &follows);

    setup_facts(&mut executor, &compiler, vec![("follows", follows_buffer)]);

    executor.execute_plan(&plan).expect("Execution failed");

    // Check reachability from node 1
    if let Some(reaches) = executor.store().get("reaches") {
        let reach_pairs = read_pairs(&provider, reaches);

        // Node 1 should reach all others
        let from_one: HashSet<u32> = reach_pairs
            .iter()
            .filter(|(src, _)| *src == 1)
            .map(|(_, dst)| *dst)
            .collect();

        println!("Nodes reachable from 1: {:?}", from_one);

        // All users in the chain should be reachable from 1
        assert!(from_one.contains(&2), "Node 2 should be reachable from 1");
        assert!(from_one.contains(&3), "Node 3 should be reachable from 1");
        assert!(from_one.contains(&4), "Node 4 should be reachable from 1");
        assert!(from_one.contains(&5), "Node 5 should be reachable from 1");
        assert_eq!(
            from_one.len(),
            4,
            "Exactly 4 nodes should be reachable from 1"
        );
    }
}

// =============================================================================
// 2. ROLE-BASED ACCESS CONTROL (RBAC)
// =============================================================================

/// Real-world scenario: Enterprise Permission System
///
/// Problem: Model a hierarchical role-based access control system where:
/// - Users are assigned to roles
/// - Roles can inherit from other roles
/// - Roles grant permissions to resources
/// - Derived permissions flow through the hierarchy
///
/// Hierarchy:
///   Admin(1) -> Manager(2) -> Employee(3)
///                   |
///                   +-> Auditor(4)
///
/// Users:
///   Alice(10) -> Admin
///   Bob(11) -> Manager
///   Carol(12) -> Employee
///   Diana(13) -> Auditor
#[test]
fn test_rbac_permission_derivation() {
    let (mut executor, provider) = match create_test_executor() {
        Some(e) => e,
        None => {
            eprintln!("Skipping: no CUDA device");
            return;
        }
    };

    // Full RBAC: role hierarchy with permission inheritance
    // Role IDs: Admin=1, Manager=2, Employee=3, Auditor=4
    // User IDs: Alice=10, Bob=11, Carol=12, Diana=13
    // Resource IDs: Database=100, Reports=101, Logs=102
    // Permission IDs: Read=1, Write=2, Delete=3
    //
    // Hierarchy: Admin(1) <- Manager(2) <- Employee(3)
    //                     <- Auditor(4)
    let source = r#"
        // Role hierarchy: child inherits from parent
        inherits(2, 1).
        inherits(3, 2).
        inherits(4, 2).

        // Direct user-role assignments
        user_role(10, 1).
        user_role(11, 2).
        user_role(12, 3).
        user_role(13, 4).

        // Direct role permissions: role_perm(role, resource, permission)
        role_perm(1, 100, 3).
        role_perm(2, 100, 2).
        role_perm(3, 100, 1).
        role_perm(2, 101, 2).
        role_perm(3, 101, 1).
        role_perm(4, 102, 1).

        // Role hierarchy (transitive)
        has_role(Child, Parent) :- inherits(Child, Parent).
        has_role(Child, Ancestor) :- has_role(Child, Parent), inherits(Parent, Ancestor).

        // User's effective roles: direct role
        effective_role(U, R) :- user_role(U, R).
        // User's effective roles: inherited roles through hierarchy
        effective_role(U, ParentRole) :- user_role(U, R), has_role(R, ParentRole).

        // User has permission if any of their effective roles grants it
        user_perm(U, Res, Perm) :- effective_role(U, R), role_perm(R, Res, Perm).
    "#;

    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("Compilation failed");

    let inherits = vec![(2, 1), (3, 2), (4, 2)];
    let user_role = vec![(10, 1), (11, 2), (12, 3), (13, 4)];
    let role_perm = vec![
        (1, 100, 3), // Admin: Delete Database
        (2, 100, 2), // Manager: Write Database
        (3, 100, 1), // Employee: Read Database
        (2, 101, 2), // Manager: Write Reports
        (3, 101, 1), // Employee: Read Reports
        (4, 102, 1), // Auditor: Read Logs
    ];

    let inherits_buf = create_edge_buffer(&provider, &inherits);
    let user_role_buf = create_edge_buffer(&provider, &user_role);
    let role_perm_buf = create_triple_buffer(&provider, &role_perm);

    setup_facts(
        &mut executor,
        &compiler,
        vec![
            ("inherits", inherits_buf),
            ("user_role", user_role_buf),
            ("role_perm", role_perm_buf),
        ],
    );

    executor.execute_plan(&plan).expect("Execution failed");

    // Check effective roles
    if let Some(effective_role) = executor.store().get("effective_role") {
        let roles = read_pairs(&provider, effective_role);
        let alice_roles: HashSet<u32> = roles
            .iter()
            .filter(|(u, _)| *u == 10)
            .map(|(_, r)| *r)
            .collect();

        println!("Alice's effective roles: {:?}", alice_roles);

        // Alice (Admin) should have Admin role directly
        assert!(alice_roles.contains(&1), "Alice should have Admin role");

        // Bob (Manager) should have Manager (2) and Admin (1) through inheritance
        let bob_roles: HashSet<u32> = roles
            .iter()
            .filter(|(u, _)| *u == 11)
            .map(|(_, r)| *r)
            .collect();
        println!("Bob's effective roles: {:?}", bob_roles);
        assert!(bob_roles.contains(&2), "Bob should have Manager role");
        assert!(
            bob_roles.contains(&1),
            "Bob should have Admin role (inherited)"
        );

        // Carol (Employee) should have Employee (3), Manager (2), and Admin (1)
        let carol_roles: HashSet<u32> = roles
            .iter()
            .filter(|(u, _)| *u == 12)
            .map(|(_, r)| *r)
            .collect();
        println!("Carol's effective roles: {:?}", carol_roles);
        assert!(carol_roles.contains(&3), "Carol should have Employee role");
        assert!(
            carol_roles.contains(&2),
            "Carol should have Manager role (inherited)"
        );
        assert!(
            carol_roles.contains(&1),
            "Carol should have Admin role (inherited)"
        );
    } else {
        panic!("effective_role relation not found");
    }

    // Check user permissions
    if let Some(user_perm) = executor.store().get("user_perm") {
        let perms = read_triples(&provider, user_perm);

        // Alice (10) should have Delete (3) on Database (100) from Admin role
        let alice_db_perms: HashSet<u32> = perms
            .iter()
            .filter(|(u, res, _)| *u == 10 && *res == 100)
            .map(|(_, _, p)| *p)
            .collect();

        println!("Alice's Database permissions: {:?}", alice_db_perms);
        assert!(
            alice_db_perms.contains(&3),
            "Alice should have Delete on Database"
        );
    } else {
        panic!("user_perm relation not found");
    }
}

// =============================================================================
// 3. SUPPLY CHAIN / BILL OF MATERIALS
// =============================================================================

/// Real-world scenario: Bill of Materials (BOM) Explosion
///
/// Problem: Given a product composed of subassemblies and parts,
/// determine all components needed to build the product (recursive).
///
/// Product hierarchy:
///   Car(1)
///     |-- Engine(2)
///     |     |-- Piston(5)
///     |     +-- Crankshaft(6)
///     |-- Chassis(3)
///     |     +-- Frame(7)
///     +-- Wheels(4)
///           +-- Tire(8)
///           +-- Rim(9)
#[test]
fn test_bill_of_materials() {
    let (mut executor, provider) = match create_test_executor() {
        Some(e) => e,
        None => {
            eprintln!("Skipping: no CUDA device");
            return;
        }
    };

    let source = r#"
        // Direct component relationships: contains(parent, child)
        contains(1, 2).
        contains(1, 3).
        contains(1, 4).
        contains(2, 5).
        contains(2, 6).
        contains(3, 7).
        contains(4, 8).
        contains(4, 9).

        // All components (recursive BOM explosion)
        component(Parent, Child) :- contains(Parent, Child).
        component(Ancestor, Descendant) :- component(Ancestor, Mid), contains(Mid, Descendant).

        // All parts (anything that appears in contains)
        part(X) :- contains(X, Y).
        part(Y) :- contains(X, Y).

        // Parts that have children (are assemblies)
        has_children(X) :- contains(X, Y).

        // Leaf parts: parts that don't have children
        // Note: Using single-column negation which GPU supports
        leaf_part(X) :- part(X), not has_children(X).
    "#;

    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("Compilation failed");

    let contains = vec![
        (1, 2),
        (1, 3),
        (1, 4), // Car contains Engine, Chassis, Wheels
        (2, 5),
        (2, 6), // Engine contains Piston, Crankshaft
        (3, 7), // Chassis contains Frame
        (4, 8),
        (4, 9), // Wheels contains Tire, Rim
    ];
    let contains_buf = create_edge_buffer(&provider, &contains);

    setup_facts(&mut executor, &compiler, vec![("contains", contains_buf)]);

    executor.execute_plan(&plan).expect("Execution failed");

    // Check all components needed for Car (1)
    if let Some(component) = executor.store().get("component") {
        let comps = read_pairs(&provider, component);
        let car_components: HashSet<u32> = comps
            .iter()
            .filter(|(parent, _)| *parent == 1)
            .map(|(_, child)| *child)
            .collect();

        println!("Car components: {:?}", car_components);

        // Direct components
        assert!(car_components.contains(&2), "Car should contain Engine");
        assert!(car_components.contains(&3), "Car should contain Chassis");
        assert!(car_components.contains(&4), "Car should contain Wheels");

        // Indirect components (through recursion)
        assert!(
            car_components.contains(&5),
            "Car should contain Piston (via Engine)"
        );
        assert!(
            car_components.contains(&6),
            "Car should contain Crankshaft (via Engine)"
        );
        assert!(
            car_components.contains(&7),
            "Car should contain Frame (via Chassis)"
        );
        assert!(
            car_components.contains(&8),
            "Car should contain Tire (via Wheels)"
        );
        assert!(
            car_components.contains(&9),
            "Car should contain Rim (via Wheels)"
        );

        assert_eq!(
            car_components.len(),
            8,
            "Car should have 8 total components"
        );
    }

    // Check leaf parts
    if let Some(leaf_part) = executor.store().get("leaf_part") {
        let leaves = read_column_u32(&provider, leaf_part, 0);
        let leaf_set: HashSet<u32> = leaves.into_iter().collect();

        println!("Leaf parts: {:?}", leaf_set);

        // Piston(5), Crankshaft(6), Frame(7), Tire(8), Rim(9) are leaf parts
        assert!(leaf_set.contains(&5), "Piston should be a leaf part");
        assert!(leaf_set.contains(&6), "Crankshaft should be a leaf part");
        assert!(leaf_set.contains(&7), "Frame should be a leaf part");
        assert!(leaf_set.contains(&8), "Tire should be a leaf part");
        assert!(leaf_set.contains(&9), "Rim should be a leaf part");

        // Assemblies should NOT be leaf parts
        assert!(!leaf_set.contains(&2), "Engine should NOT be a leaf part");
        assert!(!leaf_set.contains(&3), "Chassis should NOT be a leaf part");
        assert!(!leaf_set.contains(&4), "Wheels should NOT be a leaf part");
    }
}

// =============================================================================
// 4. PROGRAM ANALYSIS
// =============================================================================

/// Real-world scenario: Points-To Analysis
///
/// Problem: Given a simple program with assignments and pointer operations,
/// compute what variables each pointer can point to.
///
/// Program being analyzed:
///   x = &a;      // x points to a
///   y = &b;      // y points to b
///   z = x;       // z copies x, so z points to a
///   w = z;       // w copies z, so w points to a
///   *x = y;      // store y into *x (a = y, so a points to b)
///
/// Variable IDs: a=1, b=2, x=3, y=4, z=5, w=6
#[test]
fn test_points_to_analysis() {
    let (mut executor, provider) = match create_test_executor() {
        Some(e) => e,
        None => {
            eprintln!("Skipping: no CUDA device");
            return;
        }
    };

    // Variable IDs: a=1, b=2, x=3, y=4, z=5, w=6
    // Simplified: testing copy propagation (basic points-to analysis)
    let source = r#"
        // addr_of(ptr, target): ptr = &target
        addr_of(3, 1).
        addr_of(4, 2).

        // copy(dest, src): dest = src
        copy(5, 3).
        copy(6, 5).

        // Base case: address-of creates points-to
        points_to(P, T) :- addr_of(P, T).

        // Copy propagation: if dest = src, dest points to whatever src points to
        points_to(Dest, T) :- copy(Dest, Src), points_to(Src, T).
    "#;

    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("Compilation failed");

    let addr_of = vec![(3, 1), (4, 2)]; // x = &a, y = &b
    let copy_rel = vec![(5, 3), (6, 5)]; // z = x, w = z

    let addr_of_buf = create_edge_buffer(&provider, &addr_of);
    let copy_buf = create_edge_buffer(&provider, &copy_rel);

    setup_facts(
        &mut executor,
        &compiler,
        vec![("addr_of", addr_of_buf), ("copy", copy_buf)],
    );

    executor.execute_plan(&plan).expect("Execution failed");

    if let Some(points_to) = executor.store().get("points_to") {
        let pts = read_pairs(&provider, points_to);
        let pts_set: HashSet<(u32, u32)> = pts.into_iter().collect();

        println!("Points-to relation: {:?}", pts_set);

        // x (3) points to a (1) from addr_of
        assert!(pts_set.contains(&(3, 1)), "x should point to a");

        // y (4) points to b (2) from addr_of
        assert!(pts_set.contains(&(4, 2)), "y should point to b");

        // z (5) points to a (1) from copy(z, x)
        assert!(
            pts_set.contains(&(5, 1)),
            "z should point to a (copied from x)"
        );

        // w (6) points to a (1) from copy(w, z)
        assert!(
            pts_set.contains(&(6, 1)),
            "w should point to a (copied from z)"
        );
    }
}

/// Real-world scenario: Call Graph Construction
///
/// Problem: Given a set of function definitions and call sites,
/// construct the complete call graph including indirect calls.
///
/// Functions:
///   main calls foo directly
///   main calls bar directly
///   foo calls baz directly
///   main has function pointer fp that could point to foo or bar
///   main calls through fp (indirect call)
#[test]
fn test_call_graph_construction() {
    let (mut executor, provider) = match create_test_executor() {
        Some(e) => e,
        None => {
            eprintln!("Skipping: no CUDA device");
            return;
        }
    };

    // Function IDs: main=1, foo=2, bar=3, baz=4
    // Simplified: test direct calls and transitive reachability
    let source = r#"
        // Direct call sites: calls(caller, callee)
        calls(1, 2).
        calls(1, 3).
        calls(2, 4).

        // Transitive call graph (reachability)
        reaches(A, B) :- calls(A, B).
        reaches(A, C) :- reaches(A, B), calls(B, C).
    "#;

    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("Compilation failed");

    // Facts are inline in the source, just need to register relations
    let calls_data = vec![(1, 2), (1, 3), (2, 4)];
    let calls_buf = create_edge_buffer(&provider, &calls_data);

    setup_facts(&mut executor, &compiler, vec![("calls", calls_buf)]);

    executor.execute_plan(&plan).expect("Execution failed");

    // Check calls relation has all expected edges
    if let Some(calls) = executor.store().get("calls") {
        let call_edges = read_pairs(&provider, calls);
        let call_set: HashSet<(u32, u32)> = call_edges.into_iter().collect();

        println!("Call graph: {:?}", call_set);

        // All call edges should be present
        assert!(call_set.contains(&(1, 2)), "main should call foo");
        assert!(call_set.contains(&(1, 3)), "main should call bar");
        assert!(call_set.contains(&(2, 4)), "foo should call baz");
    }

    // Check transitive reachability
    if let Some(reaches) = executor.store().get("reaches") {
        let reach_edges = read_pairs(&provider, reaches);
        let reach_set: HashSet<(u32, u32)> = reach_edges.into_iter().collect();

        println!("Reachability: {:?}", reach_set);

        // main (1) should reach all functions
        assert!(reach_set.contains(&(1, 2)), "main should reach foo");
        assert!(reach_set.contains(&(1, 3)), "main should reach bar");
        assert!(
            reach_set.contains(&(1, 4)),
            "main should reach baz (via foo)"
        );
    }
}

// =============================================================================
// 5. NETWORK ANALYSIS
// =============================================================================

/// Real-world scenario: Network Connectivity Analysis
///
/// Problem: Given a network topology, compute:
/// - All reachable nodes from each node
/// - Nodes that can reach the internet gateway
/// - Isolated network segments
#[test]
fn test_network_connectivity() {
    let (mut executor, provider) = match create_test_executor() {
        Some(e) => e,
        None => {
            eprintln!("Skipping: no CUDA device");
            return;
        }
    };

    // Network topology:
    // Gateway(1) -- Router(2) -- Server(3)
    //                  |
    //              Switch(4) -- Workstation(5)
    //                  |
    //              Workstation(6)
    //
    // Isolated segment: Printer(7) -- Scanner(8)
    let source = r#"
        // Bidirectional links: link(a, b) means a and b are connected
        link(1, 2).
        link(2, 1).
        link(2, 3).
        link(3, 2).
        link(2, 4).
        link(4, 2).
        link(4, 5).
        link(5, 4).
        link(4, 6).
        link(6, 4).

        // Isolated segment
        link(7, 8).
        link(8, 7).

        // All nodes
        node(1). node(2). node(3). node(4). node(5). node(6). node(7). node(8).

        // Gateway is the internet connection point
        gateway(1).

        // Reachability
        reachable(X, Y) :- link(X, Y).
        reachable(X, Z) :- reachable(X, Y), link(Y, Z).

        // Nodes that can reach the gateway (have internet)
        has_internet(X) :- gateway(X).
        has_internet(X) :- reachable(X, G), gateway(G).

        // Isolated nodes (cannot reach gateway)
        no_internet(X) :- node(X), not has_internet(X).
    "#;

    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("Compilation failed");

    let links = vec![
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
        (2, 4),
        (4, 2),
        (4, 5),
        (5, 4),
        (4, 6),
        (6, 4),
        (7, 8),
        (8, 7),
    ];
    let nodes = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let gateway_nodes = vec![1];

    let link_buf = create_edge_buffer(&provider, &links);
    let node_buf = create_node_buffer(&provider, &nodes);
    let gateway_buf = create_node_buffer(&provider, &gateway_nodes);

    setup_facts(
        &mut executor,
        &compiler,
        vec![
            ("link", link_buf),
            ("node", node_buf),
            ("gateway", gateway_buf),
        ],
    );

    executor.execute_plan(&plan).expect("Execution failed");

    // Check internet access
    if let Some(has_internet) = executor.store().get("has_internet") {
        let connected = read_column_u32(&provider, has_internet, 0);
        let connected_set: HashSet<u32> = connected.into_iter().collect();

        println!("Nodes with internet: {:?}", connected_set);

        // Main network should have internet
        assert!(connected_set.contains(&1), "Gateway should have internet");
        assert!(connected_set.contains(&2), "Router should have internet");
        assert!(connected_set.contains(&3), "Server should have internet");
        assert!(connected_set.contains(&4), "Switch should have internet");
        assert!(
            connected_set.contains(&5),
            "Workstation 5 should have internet"
        );
        assert!(
            connected_set.contains(&6),
            "Workstation 6 should have internet"
        );

        // Isolated segment should NOT have internet
        assert!(
            !connected_set.contains(&7),
            "Printer should NOT have internet"
        );
        assert!(
            !connected_set.contains(&8),
            "Scanner should NOT have internet"
        );
    }

    // Check isolated nodes
    if let Some(no_internet) = executor.store().get("no_internet") {
        let isolated = read_column_u32(&provider, no_internet, 0);
        let isolated_set: HashSet<u32> = isolated.into_iter().collect();

        println!("Isolated nodes: {:?}", isolated_set);

        assert!(isolated_set.contains(&7), "Printer should be isolated");
        assert!(isolated_set.contains(&8), "Scanner should be isolated");
        assert_eq!(
            isolated_set.len(),
            2,
            "Should have exactly 2 isolated nodes"
        );
    }
}

// =============================================================================
// 6. DATABASE QUERY OPTIMIZATION (COMPLEX JOIN PATTERNS)
// =============================================================================

/// Real-world scenario: Multi-way Join Query
///
/// Problem: Execute a complex query that joins multiple relations,
/// similar to a typical database analytics query.
///
/// Query: Find all orders where the customer is in the same city as the supplier
///
/// Schema:
///   customer(id, city)
///   supplier(id, city)
///   order(id, customer_id, supplier_id, product)
#[test]
fn test_complex_join_query() {
    let (mut executor, provider) = match create_test_executor() {
        Some(e) => e,
        None => {
            eprintln!("Skipping: no CUDA device");
            return;
        }
    };

    let source = r#"
        // Customers: (id, city)
        customer(1, 100).
        customer(2, 100).
        customer(3, 200).

        // Suppliers: (id, city)
        supplier(10, 100).
        supplier(11, 200).
        supplier(12, 300).

        // Orders: (id, customer_id, supplier_id)
        order(1001, 1, 10).
        order(1002, 1, 11).
        order(1003, 2, 10).
        order(1004, 3, 11).
        order(1005, 3, 12).

        // Find orders where customer and supplier are in the same city
        local_order(OrderId, CustId, SuppId, City) :-
            order(OrderId, CustId, SuppId),
            customer(CustId, City),
            supplier(SuppId, City).
    "#;

    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("Compilation failed");

    let customers = vec![(1, 100), (2, 100), (3, 200)];
    let suppliers = vec![(10, 100), (11, 200), (12, 300)];
    let orders = vec![
        (1001, 1, 10), // Customer 1 (city 100), Supplier 10 (city 100) - LOCAL
        (1002, 1, 11), // Customer 1 (city 100), Supplier 11 (city 200) - NOT local
        (1003, 2, 10), // Customer 2 (city 100), Supplier 10 (city 100) - LOCAL
        (1004, 3, 11), // Customer 3 (city 200), Supplier 11 (city 200) - LOCAL
        (1005, 3, 12), // Customer 3 (city 200), Supplier 12 (city 300) - NOT local
    ];

    let customer_buf = create_edge_buffer(&provider, &customers);
    let supplier_buf = create_edge_buffer(&provider, &suppliers);
    let order_buf = create_triple_buffer(&provider, &orders);

    setup_facts(
        &mut executor,
        &compiler,
        vec![
            ("customer", customer_buf),
            ("supplier", supplier_buf),
            ("order", order_buf),
        ],
    );

    executor.execute_plan(&plan).expect("Execution failed");

    // Check local orders (customer and supplier in same city)
    if let Some(local_order) = executor.store().get("local_order") {
        let results = read_column_u32(&provider, local_order, 0);
        let order_ids: HashSet<u32> = results.into_iter().collect();

        println!("Local orders: {:?}", order_ids);

        // Orders 1001, 1003, 1004 should be local
        assert!(order_ids.contains(&1001), "Order 1001 should be local");
        assert!(order_ids.contains(&1003), "Order 1003 should be local");
        assert!(order_ids.contains(&1004), "Order 1004 should be local");

        // Orders 1002, 1005 should NOT be local
        assert!(!order_ids.contains(&1002), "Order 1002 should NOT be local");
        assert!(!order_ids.contains(&1005), "Order 1005 should NOT be local");
    }
}

// =============================================================================
// 7. ARITHMETIC EXPRESSIONS
// =============================================================================

/// Test all basic arithmetic operations (+, -, *, /, %)
///
/// This test verifies that arithmetic is-expressions work correctly
/// for all basic operations. Uses only variable-to-variable operations
/// to match the current type inference behavior (which types integer
/// literals as I64 but fact integers as U32).
#[test]
fn test_arithmetic_all_ops() {
    // Test arithmetic using only variables (no integer literals in arithmetic)
    // This matches the type inference where variables from relations match their schema types
    let (mut executor, provider) = match create_test_executor() {
        Some(e) => e,
        None => {
            eprintln!("Skipping: no CUDA device");
            return;
        }
    };

    // Define pairs of values for addition, subtraction, multiplication, division, modulo
    let source = r#"
        // Use pairs for arithmetic: val(X, Y) means compute X op Y
        add_pair(10, 5).
        add_pair(20, 3).

        sub_pair(30, 7).
        sub_pair(15, 5).

        mul_pair(4, 5).
        mul_pair(6, 7).

        div_pair(20, 4).
        div_pair(15, 3).

        mod_pair(17, 5).
        mod_pair(23, 7).

        // Arithmetic operations using only variables
        add_result(X, Y, Z) :- add_pair(X, Y), Z is X + Y.
        sub_result(X, Y, Z) :- sub_pair(X, Y), Z is X - Y.
        mul_result(X, Y, Z) :- mul_pair(X, Y), Z is X * Y.
        div_result(X, Y, Z) :- div_pair(X, Y), Z is X / Y.
        mod_result(X, Y, Z) :- mod_pair(X, Y), Z is X % Y.

        ?- add_result(X, Y, Z).
        ?- sub_result(X, Y, Z).
        ?- mul_result(X, Y, Z).
        ?- div_result(X, Y, Z).
        ?- mod_result(X, Y, Z).
    "#;

    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("Compilation failed");

    // Create fact buffers
    let add_pair_buf = create_edge_buffer(&provider, &[(10, 5), (20, 3)]);
    let sub_pair_buf = create_edge_buffer(&provider, &[(30, 7), (15, 5)]);
    let mul_pair_buf = create_edge_buffer(&provider, &[(4, 5), (6, 7)]);
    let div_pair_buf = create_edge_buffer(&provider, &[(20, 4), (15, 3)]);
    let mod_pair_buf = create_edge_buffer(&provider, &[(17, 5), (23, 7)]);

    setup_facts(
        &mut executor,
        &compiler,
        vec![
            ("add_pair", add_pair_buf),
            ("sub_pair", sub_pair_buf),
            ("mul_pair", mul_pair_buf),
            ("div_pair", div_pair_buf),
            ("mod_pair", mod_pair_buf),
        ],
    );

    executor.execute_plan(&plan).expect("Execution failed");

    // Check add_result: (10, 5, 15), (20, 3, 23)
    if let Some(result) = executor.store().get("add_result") {
        let results = read_triples(&provider, result);
        println!("add_result: {:?}", results);
        assert_eq!(results.len(), 2, "add_result should have 2 results");
        let expected: HashSet<(u32, u32, u32)> = [(10, 5, 15), (20, 3, 23)].into_iter().collect();
        let actual: HashSet<(u32, u32, u32)> = results.into_iter().collect();
        assert_eq!(actual, expected, "add_result should match");
    } else {
        panic!("add_result relation not found");
    }

    // Check sub_result: (30, 7, 23), (15, 5, 10)
    if let Some(result) = executor.store().get("sub_result") {
        let results = read_triples(&provider, result);
        println!("sub_result: {:?}", results);
        assert_eq!(results.len(), 2, "sub_result should have 2 results");
        let expected: HashSet<(u32, u32, u32)> = [(30, 7, 23), (15, 5, 10)].into_iter().collect();
        let actual: HashSet<(u32, u32, u32)> = results.into_iter().collect();
        assert_eq!(actual, expected, "sub_result should match");
    } else {
        panic!("sub_result relation not found");
    }

    // Check mul_result: (4, 5, 20), (6, 7, 42)
    if let Some(result) = executor.store().get("mul_result") {
        let results = read_triples(&provider, result);
        println!("mul_result: {:?}", results);
        assert_eq!(results.len(), 2, "mul_result should have 2 results");
        let expected: HashSet<(u32, u32, u32)> = [(4, 5, 20), (6, 7, 42)].into_iter().collect();
        let actual: HashSet<(u32, u32, u32)> = results.into_iter().collect();
        assert_eq!(actual, expected, "mul_result should match");
    } else {
        panic!("mul_result relation not found");
    }

    // Check div_result: (20, 4, 5), (15, 3, 5)
    if let Some(result) = executor.store().get("div_result") {
        let results = read_triples(&provider, result);
        println!("div_result: {:?}", results);
        assert_eq!(results.len(), 2, "div_result should have 2 results");
        let expected: HashSet<(u32, u32, u32)> = [(20, 4, 5), (15, 3, 5)].into_iter().collect();
        let actual: HashSet<(u32, u32, u32)> = results.into_iter().collect();
        assert_eq!(actual, expected, "div_result should match");
    } else {
        panic!("div_result relation not found");
    }

    // Check mod_result: (17, 5, 2), (23, 7, 2)
    if let Some(result) = executor.store().get("mod_result") {
        let results = read_triples(&provider, result);
        println!("mod_result: {:?}", results);
        assert_eq!(results.len(), 2, "mod_result should have 2 results");
        let expected: HashSet<(u32, u32, u32)> = [(17, 5, 2), (23, 7, 2)].into_iter().collect();
        let actual: HashSet<(u32, u32, u32)> = results.into_iter().collect();
        assert_eq!(actual, expected, "mod_result should match");
    } else {
        panic!("mod_result relation not found");
    }
}

/// Test chained is-expressions (like distance calculation)
///
/// This test verifies that multiple is-expressions can be chained
/// in a single rule, with intermediate results used in subsequent expressions.
#[test]
fn test_arithmetic_chained() {
    let (mut executor, provider) = match create_test_executor() {
        Some(e) => e,
        None => {
            eprintln!("Skipping: no CUDA device");
            return;
        }
    };

    // Using U32 for point coordinates, computing squared values
    let source = r#"
        point(0, 0).
        point(3, 4).

        // Chained arithmetic: compute x^2 + y^2
        sum_squares(X, Y, Sum) :-
            point(X, Y),
            X2 is X * X,
            Y2 is Y * Y,
            Sum is X2 + Y2.

        ?- sum_squares(X, Y, Sum).
    "#;

    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("Compilation failed");

    // Create point buffer with U32 values
    let point_data: Vec<(u32, u32)> = vec![(0, 0), (3, 4)];
    let point_buffer = create_edge_buffer(&provider, &point_data);

    setup_facts(&mut executor, &compiler, vec![("point", point_buffer)]);

    executor.execute_plan(&plan).expect("Execution failed");

    // Check sum_squares: (0,0,0), (3,4,25)
    if let Some(sum_squares) = executor.store().get("sum_squares") {
        let c0 = read_column_u32(&provider, sum_squares, 0);
        let c1 = read_column_u32(&provider, sum_squares, 1);
        let c2 = read_column_u32(&provider, sum_squares, 2);
        let results: Vec<(u32, u32, u32)> = c0
            .into_iter()
            .zip(c1)
            .zip(c2)
            .map(|((a, b), c)| (a, b, c))
            .collect();

        println!("sum_squares results: {:?}", results);
        assert_eq!(results.len(), 2, "sum_squares should have 2 results");

        let expected: HashSet<(u32, u32, u32)> = [(0, 0, 0), (3, 4, 25)].into_iter().collect();
        let actual: HashSet<(u32, u32, u32)> = results.into_iter().collect();
        assert_eq!(actual, expected, "sum_squares results should match");
    } else {
        panic!("sum_squares relation not found");
    }
}

/// Test that type mismatch in arithmetic produces error
///
/// This test verifies that mixing incompatible types (i64 + u32)
/// produces a compile-time error.
#[test]
fn test_arithmetic_type_error() {
    // Use two separate predicates with different declared types
    // to trigger type mismatch in arithmetic
    let program = r#"
        pred int_val(i64).
        pred uint_val(u32).
        int_val(10).
        uint_val(20).
        // This should fail: X is i64, Y is u32 - type mismatch
        bad(Z) :- int_val(X), uint_val(Y), Z is X + Y.
        ?- bad(Z).
    "#;

    let mut compiler = Compiler::new();
    let result = compiler.compile(program);
    assert!(result.is_err(), "Should fail with type mismatch error");
    let err_msg = result.unwrap_err().to_string();
    println!("Type error message: {}", err_msg);
    // Check that the error mentions type mismatch in some way
    assert!(
        err_msg.to_lowercase().contains("type")
            || err_msg.to_lowercase().contains("mismatch")
            || err_msg.to_lowercase().contains("incompatible"),
        "Error should mention type issue: {}",
        err_msg
    );
}

/// Test that non-fresh variable in is-expression produces error
///
/// This test verifies that using an already-bound variable as the
/// target of an is-expression produces a compile-time error.
#[test]
fn test_arithmetic_fresh_var_error() {
    let program = r#"
        val(10).
        // Z is already bound from val, cannot be used as target of is
        bad(Z) :- val(Z), Z is Z + 1.
        ?- bad(Z).
    "#;

    let mut compiler = Compiler::new();
    let result = compiler.compile(program);
    assert!(result.is_err(), "Should fail with non-fresh variable error");
    let err_msg = result.unwrap_err().to_string();
    println!("Fresh var error message: {}", err_msg);
    // Check that the error mentions the variable being bound
    assert!(
        err_msg.to_lowercase().contains("bound")
            || err_msg.to_lowercase().contains("fresh")
            || err_msg.to_lowercase().contains("already"),
        "Error should mention variable binding issue: {}",
        err_msg
    );
}

// =============================================================================
// 8. REAL-WORLD ARITHMETIC PROBLEM SOLVING
// =============================================================================

/// Real-world scenario: Forward Value Computation
///
/// Problem: Compute a sequence where each value depends on previous values.
/// This tests forward chaining with arithmetic using explicit index generation.
///
/// Pattern: Generate steps (N, N+1) then join with values to compute next.
#[test]
fn test_forward_computation() {
    let (mut executor, provider) = match create_test_executor() {
        Some(e) => e,
        None => {
            eprintln!("Skipping: no CUDA device");
            return;
        }
    };

    // Approach: Explicit step relation for N -> N+1 transitions
    // Use V + V instead of V * 2 to avoid type issues with literals.
    let source = r#"
        pred base_val(u32, u32).
        pred step(u32, u32).
        pred val(u32, u32).

        // Base case
        base_val(0, 1).

        // Explicit step transitions
        step(0, 1).
        step(1, 2).
        step(2, 3).
        step(3, 4).

        // val includes base and computed
        val(N, V) :- base_val(N, V).

        // Forward computation: val(N+1) = val(N) + val(N) (doubling)
        val(N1, V2) :-
            val(N, V),
            step(N, N1),
            V2 is V + V.

        ?- val(N, V).
    "#;

    let mut compiler = Compiler::new();
    let plan = match compiler.compile(source) {
        Ok(p) => p,
        Err(e) => {
            println!("Compilation error: {}", e);
            panic!("Failed to compile forward computation program");
        }
    };

    // Create base fact buffers
    let base_val_data: Vec<(u32, u32)> = vec![(0, 1)];
    let step_data: Vec<(u32, u32)> = vec![(0, 1), (1, 2), (2, 3), (3, 4)];

    let base_val_buffer = create_edge_buffer(&provider, &base_val_data);
    let step_buffer = create_edge_buffer(&provider, &step_data);

    setup_facts(
        &mut executor,
        &compiler,
        vec![("base_val", base_val_buffer), ("step", step_buffer)],
    );

    match executor.execute_plan(&plan) {
        Ok(_) => println!("Forward computation completed"),
        Err(e) => {
            println!("Execution error: {}", e);
            panic!("Failed to execute forward computation program");
        }
    }

    // Check results: 0->1, 1->2, 2->4, 3->8, 4->16
    if let Some(val) = executor.store().get("val") {
        let results = read_pairs(&provider, val);
        println!("Forward values: {:?}", results);

        let expected: HashSet<(u32, u32)> = [(0, 1), (1, 2), (2, 4), (3, 8), (4, 16)]
            .into_iter()
            .collect();
        let actual: HashSet<(u32, u32)> = results.into_iter().collect();

        for (n, v) in expected.iter() {
            assert!(actual.contains(&(*n, *v)), "Missing val({})={}", n, v);
        }
    } else {
        panic!("val relation not found");
    }
}
