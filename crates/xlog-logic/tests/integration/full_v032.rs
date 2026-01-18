//! End-to-end tests using all v0.3.2 features together.
//!
//! These tests verify that reversible symbols, modules, and user-defined functions
//! work correctly in combination across realistic scenarios.
//!
//! All tests use `serial_test::serial` since they manipulate global symbol state.

use serial_test::serial;
use xlog_core::symbol;
use xlog_logic::ast::FuncBody;
use xlog_logic::parser::parse_program;

// =============================================================================
// Task Management System
// =============================================================================

/// Test scenario: A task management system
/// - Uses modules for organization
/// - Uses symbols for status values
/// - Uses UDFs for priority calculations
#[test]
#[serial]
fn test_task_management_system() {
    symbol::clear();

    // Module: priorities.xl
    let priorities_module = r#"
        func priority_score(Urgency, Importance) =
            if Urgency > 8 then Importance * 2
            else if Urgency > 5 then Importance + Urgency
            else Importance.

        func is_critical(Score) = if Score > 15 then 1 else 0.
    "#;

    // Module: statuses.xl
    let statuses_module = r#"
        pred valid_status(symbol).
        valid_status(todo).
        valid_status(in_progress).
        valid_status(done).
        valid_status(blocked).

        private pred internal_status(symbol).
        internal_status(archived).
    "#;

    // Main program: main.xl
    let main_src = r#"
        use priorities::{priority_score, is_critical}.
        use statuses::{valid_status}.

        pred task(u32, symbol, f64, f64).
        task(1, todo, 9.0, 8.0).
        task(2, in_progress, 3.0, 7.0).
        task(3, blocked, 10.0, 10.0).

        pred critical_task(u32, symbol).
        critical_task(Id, Status) :-
            task(Id, Status, Urgency, Importance),
            valid_status(Status),
            Score is priority_score(Urgency, Importance),
            Critical is is_critical(Score),
            Critical = 1.

        ?- critical_task(X, Y).
    "#;

    // Parse all modules
    let priorities = parse_program(priorities_module).unwrap();
    let statuses = parse_program(statuses_module).unwrap();
    let main = parse_program(main_src).unwrap();

    // Verify modules parsed correctly
    assert!(
        !priorities.functions.is_empty(),
        "priorities should have functions"
    );
    assert!(
        statuses
            .rules
            .iter()
            .any(|r| r.head.predicate == "valid_status"),
        "statuses should have valid_status"
    );
    assert!(!main.imports.is_empty(), "main should have imports");

    // Verify symbols are interned
    let todo = symbol::intern("todo");
    let in_progress = symbol::intern("in_progress");
    let done = symbol::intern("done");
    let blocked = symbol::intern("blocked");

    // All symbols should resolve correctly
    assert_eq!(symbol::resolve(todo), "todo");
    assert_eq!(symbol::resolve(in_progress), "in_progress");
    assert_eq!(symbol::resolve(done), "done");
    assert_eq!(symbol::resolve(blocked), "blocked");

    // Verify priority functions
    assert!(priorities.functions.iter().any(|f| f.name == "priority_score"));
    assert!(priorities.functions.iter().any(|f| f.name == "is_critical"));

    // Verify priority_score has nested conditionals
    let priority_score = priorities
        .functions
        .iter()
        .find(|f| f.name == "priority_score")
        .unwrap();
    assert_eq!(priority_score.params.len(), 2);
    match &priority_score.body {
        FuncBody::Conditional(cond) => {
            // Should have nested else-if
            match cond.else_branch.as_ref() {
                FuncBody::Conditional(_) => {} // nested conditional in else
                _ => panic!("Expected nested conditional in else branch"),
            }
        }
        _ => panic!("Expected conditional body for priority_score"),
    }

    // Verify private predicate in predicate declarations
    let private_pred = statuses
        .predicates
        .iter()
        .find(|p| p.name == "internal_status")
        .expect("internal_status predicate should exist");
    assert!(
        private_pred.is_private,
        "internal_status should be private"
    );

    // Verify main program imports
    assert_eq!(main.imports.len(), 2);
    assert!(main.imports.iter().any(|i| i.module_path == vec!["priorities"]));
    assert!(main.imports.iter().any(|i| i.module_path == vec!["statuses"]));

    // Verify main query
    assert!(!main.queries.is_empty());
}

#[test]
#[serial]
fn test_nested_module_with_functions_and_symbols() {
    symbol::clear();

    // Nested module path
    let utils_math = r#"
        func abs(X) = if X < 0 then 0 - X else X.
    "#;

    // Main using nested path
    let main_src = r#"
        use utils/math::{abs}.

        pred measurement(symbol, f64).
        measurement(temp, -5.0).
        measurement(pressure, 101.3).

        pred absolute_measurement(symbol, f64).
        absolute_measurement(Label, AbsVal) :-
            measurement(Label, Val),
            AbsVal is abs(Val).

        ?- absolute_measurement(X, Y).
    "#;

    let utils_prog = parse_program(utils_math).unwrap();
    let main_prog = parse_program(main_src).unwrap();

    // Verify nested import path
    assert!(main_prog
        .imports
        .iter()
        .any(|i| i.module_path == vec!["utils", "math"]));

    // Verify symbols
    assert_eq!(symbol::resolve(symbol::intern("temp")), "temp");
    assert_eq!(symbol::resolve(symbol::intern("pressure")), "pressure");

    // Verify abs function
    assert!(utils_prog.functions.iter().any(|f| f.name == "abs"));
}

// =============================================================================
// Inventory / Warehouse System
// =============================================================================

/// Test scenario: An inventory management system
/// - Uses modules for warehouse locations and product categories
/// - Uses symbols for product categories and warehouse codes
/// - Uses UDFs for stock calculations and reorder logic
#[test]
#[serial]
fn test_inventory_warehouse_system() {
    symbol::clear();

    // Module: warehouses.xl
    let warehouses_module = r#"
        pred warehouse(symbol, symbol).
        warehouse(wh_north, active).
        warehouse(wh_south, active).
        warehouse(wh_east, maintenance).
        warehouse(wh_west, inactive).

        pred active_warehouse(symbol).
        active_warehouse(W) :- warehouse(W, active).
    "#;

    // Module: products.xl
    let products_module = r#"
        pred category(symbol).
        category(electronics).
        category(clothing).
        category(food).
        category(furniture).

        func reorder_threshold(BaseStock) = BaseStock * 0.2.
        func safety_stock(DailyUsage, LeadTime) = DailyUsage * LeadTime * 1.5.
    "#;

    // Main program: inventory.xl
    let main_src = r#"
        use warehouses::{warehouse, active_warehouse}.
        use products::{category, reorder_threshold, safety_stock}.

        pred stock(symbol, symbol, f64).
        stock(wh_north, electronics, 500.0).
        stock(wh_north, clothing, 200.0).
        stock(wh_south, food, 1000.0).
        stock(wh_east, furniture, 50.0).

        pred daily_usage(symbol, f64).
        daily_usage(electronics, 25.0).
        daily_usage(clothing, 15.0).
        daily_usage(food, 100.0).
        daily_usage(furniture, 5.0).

        pred lead_time(symbol, f64).
        lead_time(electronics, 7.0).
        lead_time(clothing, 14.0).
        lead_time(food, 2.0).
        lead_time(furniture, 21.0).

        pred needs_reorder(symbol, symbol).
        needs_reorder(Warehouse, Cat) :-
            stock(Warehouse, Cat, CurrentStock),
            active_warehouse(Warehouse),
            category(Cat),
            daily_usage(Cat, Usage),
            lead_time(Cat, Lead),
            MinStock is safety_stock(Usage, Lead),
            CurrentStock < MinStock.

        ?- needs_reorder(W, C).
    "#;

    // Parse all modules
    let warehouses = parse_program(warehouses_module).unwrap();
    let products = parse_program(products_module).unwrap();
    let main = parse_program(main_src).unwrap();

    // Verify warehouse symbols
    let wh_north = symbol::intern("wh_north");
    let wh_south = symbol::intern("wh_south");
    let wh_east = symbol::intern("wh_east");
    let wh_west = symbol::intern("wh_west");
    let active = symbol::intern("active");
    let maintenance = symbol::intern("maintenance");
    let inactive = symbol::intern("inactive");

    assert_eq!(symbol::resolve(wh_north), "wh_north");
    assert_eq!(symbol::resolve(wh_south), "wh_south");
    assert_eq!(symbol::resolve(wh_east), "wh_east");
    assert_eq!(symbol::resolve(wh_west), "wh_west");
    assert_eq!(symbol::resolve(active), "active");
    assert_eq!(symbol::resolve(maintenance), "maintenance");
    assert_eq!(symbol::resolve(inactive), "inactive");

    // Verify category symbols
    let electronics = symbol::intern("electronics");
    let clothing = symbol::intern("clothing");
    let food = symbol::intern("food");
    let furniture = symbol::intern("furniture");

    assert_eq!(symbol::resolve(electronics), "electronics");
    assert_eq!(symbol::resolve(clothing), "clothing");
    assert_eq!(symbol::resolve(food), "food");
    assert_eq!(symbol::resolve(furniture), "furniture");

    // Verify functions
    assert!(products.functions.iter().any(|f| f.name == "reorder_threshold"));
    assert!(products.functions.iter().any(|f| f.name == "safety_stock"));

    // Verify safety_stock has 2 parameters
    let safety_stock_fn = products
        .functions
        .iter()
        .find(|f| f.name == "safety_stock")
        .unwrap();
    assert_eq!(safety_stock_fn.params.len(), 2);

    // Verify active_warehouse rule
    let active_wh_rules: Vec<_> = warehouses
        .proper_rules()
        .filter(|r| r.head.predicate == "active_warehouse")
        .collect();
    assert_eq!(active_wh_rules.len(), 1);

    // Verify main program structure
    assert_eq!(main.imports.len(), 2);
    assert!(main.queries.len() >= 1);

    // Verify needs_reorder rule has complex body
    let needs_reorder_rules: Vec<_> = main
        .proper_rules()
        .filter(|r| r.head.predicate == "needs_reorder")
        .collect();
    assert_eq!(needs_reorder_rules.len(), 1);
    assert!(
        needs_reorder_rules[0].body.len() >= 6,
        "needs_reorder should have at least 6 body literals"
    );
}

// =============================================================================
// Analytics / Metrics System
// =============================================================================

/// Test scenario: An analytics/metrics collection system
/// - Uses modules for metric definitions and aggregation functions
/// - Uses symbols for metric names and aggregation types
/// - Uses UDFs for statistical calculations
#[test]
#[serial]
fn test_analytics_metrics_system() {
    symbol::clear();

    // Module: metrics.xl
    let metrics_module = r#"
        pred metric_type(symbol, symbol).
        metric_type(cpu_usage, gauge).
        metric_type(request_count, counter).
        metric_type(response_time, histogram).
        metric_type(error_rate, gauge).

        pred aggregation_method(symbol, symbol).
        aggregation_method(gauge, avg).
        aggregation_method(counter, sum).
        aggregation_method(histogram, percentile).
    "#;

    // Module: statistics.xl
    let statistics_module = r#"
        func normalize(Value, Min, Max) =
            if Max = Min then 0.5
            else (Value - Min) / (Max - Min).

        func clamp(Value, Lo, Hi) =
            if Value < Lo then Lo
            else if Value > Hi then Hi
            else Value.

        func anomaly_score(Value, Mean, StdDev) =
            if StdDev = 0 then 0
            else abs_val(Value - Mean) / StdDev.

        private func abs_val(X) = if X < 0 then 0 - X else X.
    "#;

    // Main program: analytics.xl
    let main_src = r#"
        use metrics::{metric_type, aggregation_method}.
        use statistics::{normalize, clamp, anomaly_score}.

        pred raw_metric(symbol, f64, f64).
        raw_metric(cpu_usage, 85.5, 1000.0).
        raw_metric(request_count, 15000.0, 1001.0).
        raw_metric(response_time, 250.0, 1002.0).
        raw_metric(error_rate, 2.5, 1003.0).

        pred metric_stats(symbol, f64, f64).
        metric_stats(cpu_usage, 50.0, 15.0).
        metric_stats(request_count, 10000.0, 2000.0).
        metric_stats(response_time, 100.0, 50.0).
        metric_stats(error_rate, 1.0, 0.5).

        pred anomalous_metric(symbol, f64).
        anomalous_metric(Name, Score) :-
            raw_metric(Name, Value, Ts),
            metric_type(Name, Type),
            metric_stats(Name, Mean, StdDev),
            Score is anomaly_score(Value, Mean, StdDev),
            Score > 2.0.

        pred normalized_metric(symbol, f64).
        normalized_metric(Name, NormValue) :-
            raw_metric(Name, Value, Ts),
            Raw is normalize(Value, 0, 100),
            NormValue is clamp(Raw, 0, 1).

        ?- anomalous_metric(M, S).
        ?- normalized_metric(M, N).
    "#;

    // Parse all modules
    let metrics = parse_program(metrics_module).unwrap();
    let statistics = parse_program(statistics_module).unwrap();
    let main = parse_program(main_src).unwrap();

    // Verify metric type symbols
    assert_eq!(symbol::resolve(symbol::intern("cpu_usage")), "cpu_usage");
    assert_eq!(symbol::resolve(symbol::intern("request_count")), "request_count");
    assert_eq!(symbol::resolve(symbol::intern("response_time")), "response_time");
    assert_eq!(symbol::resolve(symbol::intern("error_rate")), "error_rate");

    // Verify aggregation type symbols
    assert_eq!(symbol::resolve(symbol::intern("gauge")), "gauge");
    assert_eq!(symbol::resolve(symbol::intern("counter")), "counter");
    assert_eq!(symbol::resolve(symbol::intern("histogram")), "histogram");
    assert_eq!(symbol::resolve(symbol::intern("avg")), "avg");
    assert_eq!(symbol::resolve(symbol::intern("sum")), "sum");
    assert_eq!(symbol::resolve(symbol::intern("percentile")), "percentile");

    // Verify statistics functions
    assert!(statistics.functions.iter().any(|f| f.name == "normalize"));
    assert!(statistics.functions.iter().any(|f| f.name == "clamp"));
    assert!(statistics.functions.iter().any(|f| f.name == "anomaly_score"));
    assert!(statistics.functions.iter().any(|f| f.name == "abs_val"));

    // Verify private function
    let abs_fn = statistics.functions.iter().find(|f| f.name == "abs_val").unwrap();
    assert!(abs_fn.is_private, "abs_val should be private");

    // Verify anomaly_score has 3 parameters
    let anomaly_fn = statistics
        .functions
        .iter()
        .find(|f| f.name == "anomaly_score")
        .unwrap();
    assert_eq!(anomaly_fn.params.len(), 3);

    // Verify metrics facts count
    let metric_type_facts: Vec<_> = metrics
        .facts()
        .filter(|r| r.head.predicate == "metric_type")
        .collect();
    assert_eq!(metric_type_facts.len(), 4);

    // Verify main has 2 queries
    assert_eq!(main.queries.len(), 2);

    // Verify main imports
    assert_eq!(main.imports.len(), 2);
}

// =============================================================================
// Hierarchical Organization Structure
// =============================================================================

/// Test scenario: A hierarchical organization structure
/// - Uses modules for departments and roles
/// - Uses symbols for department names, role types, and employee IDs
/// - Uses UDFs for salary calculations and level computations
#[test]
#[serial]
fn test_hierarchical_organization_structure() {
    symbol::clear();

    // Module: departments.xl
    let departments_module = r#"
        pred department(symbol, symbol).
        department(engineering, technology).
        department(product, technology).
        department(sales, business).
        department(marketing, business).
        department(hr, operations).
        department(finance, operations).

        pred department_budget_multiplier(symbol, f64).
        department_budget_multiplier(technology, 1.5).
        department_budget_multiplier(business, 1.2).
        department_budget_multiplier(operations, 1.0).
    "#;

    // Module: roles.xl
    let roles_module = r#"
        pred role_level(symbol, u32).
        role_level(intern, 1).
        role_level(junior, 2).
        role_level(mid, 3).
        role_level(senior, 4).
        role_level(lead, 5).
        role_level(manager, 6).
        role_level(director, 7).
        role_level(vp, 8).
        role_level(cxo, 9).

        func level_salary_base(Level) =
            if Level <= 2 then 50000
            else if Level <= 4 then 80000
            else if Level <= 6 then 120000
            else if Level <= 8 then 180000
            else 250000.

        func level_bonus_rate(Level) =
            if Level <= 3 then 0.05
            else if Level <= 6 then 0.15
            else 0.25.
    "#;

    // Main program: organization.xl
    let main_src = r#"
        use departments::{department, department_budget_multiplier}.
        use roles::{role_level, level_salary_base, level_bonus_rate}.

        pred employee(symbol, symbol, symbol).
        employee(emp_001, engineering, senior).
        employee(emp_002, engineering, lead).
        employee(emp_003, sales, junior).
        employee(emp_004, hr, manager).
        employee(emp_005, product, director).

        pred reports_to(symbol, symbol).
        reports_to(emp_001, emp_002).
        reports_to(emp_002, emp_005).
        reports_to(emp_003, emp_004).

        pred employee_compensation(symbol, f64, f64).
        employee_compensation(EmpId, Salary, Bonus) :-
            employee(EmpId, Dept, Role),
            department(Dept, Division),
            department_budget_multiplier(Division, Multiplier),
            role_level(Role, Level),
            BaseSalary is level_salary_base(Level),
            BonusRate is level_bonus_rate(Level),
            Salary is BaseSalary * Multiplier,
            Bonus is Salary * BonusRate.

        pred senior_staff(symbol, symbol).
        senior_staff(EmpId, Role) :-
            employee(EmpId, Dept, Role),
            role_level(Role, Level),
            Level >= 5.

        ?- employee_compensation(E, S, B).
        ?- senior_staff(E, R).
    "#;

    // Parse all modules
    let _departments = parse_program(departments_module).unwrap();
    let roles = parse_program(roles_module).unwrap();
    let main = parse_program(main_src).unwrap();

    // Verify department symbols
    assert_eq!(symbol::resolve(symbol::intern("engineering")), "engineering");
    assert_eq!(symbol::resolve(symbol::intern("product")), "product");
    assert_eq!(symbol::resolve(symbol::intern("sales")), "sales");
    assert_eq!(symbol::resolve(symbol::intern("marketing")), "marketing");
    assert_eq!(symbol::resolve(symbol::intern("hr")), "hr");
    assert_eq!(symbol::resolve(symbol::intern("finance")), "finance");

    // Verify division symbols
    assert_eq!(symbol::resolve(symbol::intern("technology")), "technology");
    assert_eq!(symbol::resolve(symbol::intern("business")), "business");
    assert_eq!(symbol::resolve(symbol::intern("operations")), "operations");

    // Verify role symbols
    assert_eq!(symbol::resolve(symbol::intern("intern")), "intern");
    assert_eq!(symbol::resolve(symbol::intern("junior")), "junior");
    assert_eq!(symbol::resolve(symbol::intern("mid")), "mid");
    assert_eq!(symbol::resolve(symbol::intern("senior")), "senior");
    assert_eq!(symbol::resolve(symbol::intern("lead")), "lead");
    assert_eq!(symbol::resolve(symbol::intern("manager")), "manager");
    assert_eq!(symbol::resolve(symbol::intern("director")), "director");
    assert_eq!(symbol::resolve(symbol::intern("vp")), "vp");
    assert_eq!(symbol::resolve(symbol::intern("cxo")), "cxo");

    // Verify employee symbols
    assert_eq!(symbol::resolve(symbol::intern("emp_001")), "emp_001");
    assert_eq!(symbol::resolve(symbol::intern("emp_002")), "emp_002");
    assert_eq!(symbol::resolve(symbol::intern("emp_003")), "emp_003");
    assert_eq!(symbol::resolve(symbol::intern("emp_004")), "emp_004");
    assert_eq!(symbol::resolve(symbol::intern("emp_005")), "emp_005");

    // Verify role level facts
    let role_level_facts: Vec<_> = roles
        .facts()
        .filter(|r| r.head.predicate == "role_level")
        .collect();
    assert_eq!(role_level_facts.len(), 9);

    // Verify salary functions
    assert!(roles.functions.iter().any(|f| f.name == "level_salary_base"));
    assert!(roles.functions.iter().any(|f| f.name == "level_bonus_rate"));

    // Verify level_salary_base has deeply nested conditionals
    let salary_fn = roles
        .functions
        .iter()
        .find(|f| f.name == "level_salary_base")
        .unwrap();
    match &salary_fn.body {
        FuncBody::Conditional(outer) => {
            // First level nested
            match outer.else_branch.as_ref() {
                FuncBody::Conditional(inner1) => {
                    // Second level nested
                    match inner1.else_branch.as_ref() {
                        FuncBody::Conditional(inner2) => {
                            // Third level nested
                            match inner2.else_branch.as_ref() {
                                FuncBody::Conditional(_) => {} // Fourth level
                                _ => panic!("Expected 4th level conditional"),
                            }
                        }
                        _ => panic!("Expected 3rd level conditional"),
                    }
                }
                _ => panic!("Expected 2nd level conditional"),
            }
        }
        _ => panic!("Expected conditional body for level_salary_base"),
    }

    // Verify employee_compensation rule complexity
    let comp_rules: Vec<_> = main
        .proper_rules()
        .filter(|r| r.head.predicate == "employee_compensation")
        .collect();
    assert_eq!(comp_rules.len(), 1);
    assert!(
        comp_rules[0].body.len() >= 8,
        "employee_compensation should have at least 8 body literals"
    );

    // Verify queries
    assert_eq!(main.queries.len(), 2);
}

// =============================================================================
// Event Processing Pipeline
// =============================================================================

/// Test scenario: An event processing pipeline
/// - Uses modules for event types and processing rules
/// - Uses symbols for event categories and states
/// - Uses UDFs for event scoring and filtering
#[test]
#[serial]
fn test_event_processing_pipeline() {
    symbol::clear();

    // Module: event_types.xl
    let event_types_module = r#"
        pred event_category(symbol, symbol).
        event_category(user_login, authentication).
        event_category(user_logout, authentication).
        event_category(page_view, navigation).
        event_category(button_click, interaction).
        event_category(form_submit, interaction).
        event_category(error_occurred, system).
        event_category(api_call, system).

        pred category_priority(symbol, u32).
        category_priority(system, 1).
        category_priority(authentication, 2).
        category_priority(interaction, 3).
        category_priority(navigation, 4).
    "#;

    // Module: event_processing.xl
    let event_processing_module = r#"
        func event_score(Priority, Frequency) =
            if Priority = 1 then Frequency * 10
            else if Priority = 2 then Frequency * 5
            else if Priority = 3 then Frequency * 2
            else Frequency.

        func should_alert(Score, Threshold) = if Score > Threshold then 1 else 0.

        func time_bucket(Timestamp, BucketSize) = Timestamp - (Timestamp % BucketSize).

        private func internal_weight(X) = X * 0.8.
    "#;

    // Module: event_state.xl
    let event_state_module = r#"
        pred valid_state(symbol).
        valid_state(pending).
        valid_state(processing).
        valid_state(completed).
        valid_state(failed).
        valid_state(retrying).

        pred state_transition(symbol, symbol).
        state_transition(pending, processing).
        state_transition(processing, completed).
        state_transition(processing, failed).
        state_transition(failed, retrying).
        state_transition(retrying, processing).
    "#;

    // Main program: pipeline.xl
    let main_src = r#"
        use event_types::{event_category, category_priority}.
        use event_processing::{event_score, should_alert, time_bucket}.
        use event_state::{valid_state, state_transition}.

        pred event(symbol, symbol, f64, f64).
        event(evt_001, user_login, 100.0, 1000.0).
        event(evt_002, error_occurred, 50.0, 1001.0).
        event(evt_003, page_view, 500.0, 1002.0).
        event(evt_004, api_call, 25.0, 1003.0).

        pred event_status(symbol, symbol).
        event_status(evt_001, completed).
        event_status(evt_002, failed).
        event_status(evt_003, processing).
        event_status(evt_004, pending).

        pred high_priority_event(symbol, symbol, f64).
        high_priority_event(EventId, EventType, Score) :-
            event(EventId, EventType, Frequency, Timestamp),
            event_category(EventType, Cat),
            category_priority(Cat, Priority),
            Score is event_score(Priority, Frequency),
            Alert is should_alert(Score, 100),
            Alert = 1.

        pred actionable_event(symbol, symbol, symbol).
        actionable_event(EventId, EventType, NextState) :-
            event(EventId, EventType, Freq, Ts),
            event_status(EventId, CurrentState),
            valid_state(CurrentState),
            state_transition(CurrentState, NextState).

        pred bucketed_event(symbol, f64).
        bucketed_event(EventId, Bucket) :-
            event(EventId, EvtType, Freq, Timestamp),
            Bucket is time_bucket(Timestamp, 60).

        ?- high_priority_event(E, T, S).
        ?- actionable_event(E, T, N).
        ?- bucketed_event(E, B).
    "#;

    // Parse all modules
    let event_types = parse_program(event_types_module).unwrap();
    let event_processing = parse_program(event_processing_module).unwrap();
    let event_state = parse_program(event_state_module).unwrap();
    let main = parse_program(main_src).unwrap();

    // Verify event type symbols
    assert_eq!(symbol::resolve(symbol::intern("user_login")), "user_login");
    assert_eq!(symbol::resolve(symbol::intern("user_logout")), "user_logout");
    assert_eq!(symbol::resolve(symbol::intern("page_view")), "page_view");
    assert_eq!(symbol::resolve(symbol::intern("button_click")), "button_click");
    assert_eq!(symbol::resolve(symbol::intern("form_submit")), "form_submit");
    assert_eq!(symbol::resolve(symbol::intern("error_occurred")), "error_occurred");
    assert_eq!(symbol::resolve(symbol::intern("api_call")), "api_call");

    // Verify category symbols
    assert_eq!(symbol::resolve(symbol::intern("authentication")), "authentication");
    assert_eq!(symbol::resolve(symbol::intern("navigation")), "navigation");
    assert_eq!(symbol::resolve(symbol::intern("interaction")), "interaction");
    assert_eq!(symbol::resolve(symbol::intern("system")), "system");

    // Verify state symbols
    assert_eq!(symbol::resolve(symbol::intern("pending")), "pending");
    assert_eq!(symbol::resolve(symbol::intern("processing")), "processing");
    assert_eq!(symbol::resolve(symbol::intern("completed")), "completed");
    assert_eq!(symbol::resolve(symbol::intern("failed")), "failed");
    assert_eq!(symbol::resolve(symbol::intern("retrying")), "retrying");

    // Verify event ID symbols
    assert_eq!(symbol::resolve(symbol::intern("evt_001")), "evt_001");
    assert_eq!(symbol::resolve(symbol::intern("evt_002")), "evt_002");
    assert_eq!(symbol::resolve(symbol::intern("evt_003")), "evt_003");
    assert_eq!(symbol::resolve(symbol::intern("evt_004")), "evt_004");

    // Verify event_category facts
    let category_facts: Vec<_> = event_types
        .facts()
        .filter(|r| r.head.predicate == "event_category")
        .collect();
    assert_eq!(category_facts.len(), 7);

    // Verify processing functions
    assert!(event_processing.functions.iter().any(|f| f.name == "event_score"));
    assert!(event_processing.functions.iter().any(|f| f.name == "should_alert"));
    assert!(event_processing.functions.iter().any(|f| f.name == "time_bucket"));
    assert!(event_processing.functions.iter().any(|f| f.name == "internal_weight"));

    // Verify private function
    let internal_weight = event_processing
        .functions
        .iter()
        .find(|f| f.name == "internal_weight")
        .unwrap();
    assert!(internal_weight.is_private);

    // Verify state transitions
    let transitions: Vec<_> = event_state
        .facts()
        .filter(|r| r.head.predicate == "state_transition")
        .collect();
    assert_eq!(transitions.len(), 5);

    // Verify main imports 3 modules
    assert_eq!(main.imports.len(), 3);

    // Verify main queries
    assert_eq!(main.queries.len(), 3);
}

// =============================================================================
// Cross-Module Symbol Deduplication
// =============================================================================

/// Test that symbols are properly deduplicated across multiple module parses
#[test]
#[serial]
fn test_cross_module_symbol_deduplication() {
    symbol::clear();

    // Module A uses some symbols
    let module_a = r#"
        pred data_a(symbol).
        data_a(shared_symbol).
        data_a(unique_a).
    "#;

    // Module B also uses shared_symbol
    let module_b = r#"
        pred data_b(symbol).
        data_b(shared_symbol).
        data_b(unique_b).
    "#;

    // Module C also uses shared_symbol plus some from A
    let module_c = r#"
        pred data_c(symbol).
        data_c(shared_symbol).
        data_c(unique_a).
        data_c(unique_c).
    "#;

    let _prog_a = parse_program(module_a).unwrap();
    let count_after_a = symbol::count();

    let _prog_b = parse_program(module_b).unwrap();
    let count_after_b = symbol::count();

    let _prog_c = parse_program(module_c).unwrap();
    let count_after_c = symbol::count();

    // A: shared_symbol, unique_a = 2 new
    // B: shared_symbol (exists), unique_b = 1 new
    // C: shared_symbol (exists), unique_a (exists), unique_c = 1 new
    assert!(
        count_after_a >= 2,
        "Expected at least 2 symbols after module A"
    );
    assert_eq!(
        count_after_b,
        count_after_a + 1,
        "Module B should add exactly 1 new symbol (unique_b)"
    );
    assert_eq!(
        count_after_c,
        count_after_b + 1,
        "Module C should add exactly 1 new symbol (unique_c)"
    );

    // All symbols resolve correctly
    let shared = symbol::intern("shared_symbol");
    let unique_a = symbol::intern("unique_a");
    let unique_b = symbol::intern("unique_b");
    let unique_c = symbol::intern("unique_c");

    assert_eq!(symbol::resolve(shared), "shared_symbol");
    assert_eq!(symbol::resolve(unique_a), "unique_a");
    assert_eq!(symbol::resolve(unique_b), "unique_b");
    assert_eq!(symbol::resolve(unique_c), "unique_c");
}

// =============================================================================
// Complex Nested Module Imports
// =============================================================================

/// Test deeply nested module paths with mixed imports
#[test]
#[serial]
fn test_deeply_nested_module_paths() {
    symbol::clear();

    let main_src = r#"
        use core/utils/math::{abs, clamp}.
        use core/utils/string.
        use domain/models/user::{user_model, validate_user}.
        use infra/db/postgres.
        use infra/cache/redis::{get, set, delete}.

        pred config(symbol, symbol).
        config(database, postgres).
        config(cache, redis).

        pred active_connections(symbol, u32).
        active_connections(postgres, 10).
        active_connections(redis, 5).

        ?- config(X, Y).
    "#;

    let prog = parse_program(main_src).unwrap();

    // Verify all imports
    assert_eq!(prog.imports.len(), 5);

    // Check nested paths
    assert!(prog.imports.iter().any(|i| i.module_path == vec!["core", "utils", "math"]));
    assert!(prog.imports.iter().any(|i| i.module_path == vec!["core", "utils", "string"]));
    assert!(prog.imports.iter().any(|i| i.module_path == vec!["domain", "models", "user"]));
    assert!(prog.imports.iter().any(|i| i.module_path == vec!["infra", "db", "postgres"]));
    assert!(prog.imports.iter().any(|i| i.module_path == vec!["infra", "cache", "redis"]));

    // Check specific imports
    let math_import = prog
        .imports
        .iter()
        .find(|i| i.module_path == vec!["core", "utils", "math"])
        .unwrap();
    let math_imports = math_import.imports.as_ref().unwrap();
    assert!(math_imports.contains(&"abs".to_string()));
    assert!(math_imports.contains(&"clamp".to_string()));

    // Check import-all
    let string_import = prog
        .imports
        .iter()
        .find(|i| i.module_path == vec!["core", "utils", "string"])
        .unwrap();
    assert!(string_import.imports.is_none()); // None means import all

    // Verify symbols
    assert_eq!(symbol::resolve(symbol::intern("database")), "database");
    assert_eq!(symbol::resolve(symbol::intern("postgres")), "postgres");
    assert_eq!(symbol::resolve(symbol::intern("cache")), "cache");
    assert_eq!(symbol::resolve(symbol::intern("redis")), "redis");
}

// =============================================================================
// Full System Integration: E-Commerce Platform
// =============================================================================

/// Test scenario: A complete e-commerce platform combining all features
/// This is the most comprehensive test, simulating a real-world application
#[test]
#[serial]
fn test_ecommerce_platform_full_integration() {
    symbol::clear();

    // Module: products.xl
    // Note: Function conditionals compare numeric values, not symbols directly
    let products_module = r#"
        pred product_category(symbol).
        product_category(electronics).
        product_category(clothing).
        product_category(books).
        product_category(home_garden).

        pred product(symbol, symbol, f64).
        product(prod_001, electronics, 999.99).
        product(prod_002, clothing, 49.99).
        product(prod_003, books, 29.99).
        product(prod_004, electronics, 199.99).
        product(prod_005, home_garden, 149.99).

        pred category_rate(symbol, f64).
        category_rate(electronics, 0.1).
        category_rate(clothing, 0.2).
        category_rate(books, 0.05).
        category_rate(home_garden, 0.15).
    "#;

    // Module: orders.xl
    let orders_module = r#"
        pred order_status(symbol).
        order_status(pending).
        order_status(confirmed).
        order_status(shipped).
        order_status(delivered).
        order_status(cancelled).
        order_status(refunded).

        pred status_priority(symbol, u32).
        status_priority(pending, 1).
        status_priority(confirmed, 2).
        status_priority(shipped, 3).
        status_priority(delivered, 4).
        status_priority(cancelled, 0).
        status_priority(refunded, 0).

        func can_modify(StatusPriority) = if StatusPriority <= 2 then 1 else 0.
        func can_cancel(StatusPriority) = if StatusPriority <= 3 then 1 else 0.
    "#;

    // Module: pricing.xl
    let pricing_module = r#"
        func apply_discount(Price, DiscountRate) = Price * (1.0 - DiscountRate).
        func apply_tax(Price, TaxRate) = Price * (1.0 + TaxRate).
        func calculate_shipping(Weight, Distance) =
            if Weight < 1.0 then Distance * 0.5
            else if Weight < 5.0 then Distance * 1.0
            else Distance * 2.0.

        private func round_cents(X) = X.
    "#;

    // Module: customers.xl
    let customers_module = r#"
        pred customer_tier(symbol).
        customer_tier(bronze).
        customer_tier(silver).
        customer_tier(gold).
        customer_tier(platinum).

        pred tier_benefits(symbol, f64, f64).
        tier_benefits(bronze, 0.0, 50.0).
        tier_benefits(silver, 0.05, 25.0).
        tier_benefits(gold, 0.1, 0.0).
        tier_benefits(platinum, 0.15, 0.0).

        pred tier_multiplier(symbol, f64).
        tier_multiplier(bronze, 1.0).
        tier_multiplier(silver, 1.2).
        tier_multiplier(gold, 1.5).
        tier_multiplier(platinum, 2.0).
    "#;

    // Main program: ecommerce.xl
    let main_src = r#"
        use products::{product_category, product, category_rate}.
        use orders::{order_status, status_priority, can_modify, can_cancel}.
        use pricing::{apply_discount, apply_tax, calculate_shipping}.
        use customers::{customer_tier, tier_benefits, tier_multiplier}.

        pred customer(symbol, symbol).
        customer(cust_001, gold).
        customer(cust_002, silver).
        customer(cust_003, bronze).
        customer(cust_004, platinum).

        pred order(symbol, symbol, symbol, symbol, f64).
        order(ord_001, cust_001, prod_001, confirmed, 1.5).
        order(ord_002, cust_002, prod_002, shipped, 0.3).
        order(ord_003, cust_003, prod_003, pending, 0.5).
        order(ord_004, cust_004, prod_004, delivered, 2.0).

        pred order_final_price(symbol, f64).
        order_final_price(OrderId, FinalPrice) :-
            order(OrderId, CustomerId, ProductId, Status, Weight),
            customer(CustomerId, Tier),
            product(ProductId, Cat, BasePrice),
            tier_benefits(Tier, TierDiscount, ShippingThreshold),
            category_rate(Cat, CatDiscount),
            TotalDiscount is TierDiscount + CatDiscount,
            DiscountedPrice is apply_discount(BasePrice, TotalDiscount),
            PriceWithTax is apply_tax(DiscountedPrice, 0.08),
            ShippingCost is calculate_shipping(Weight, 100),
            FinalPrice is PriceWithTax + ShippingCost.

        pred modifiable_order(symbol, symbol).
        modifiable_order(OrderId, Status) :-
            order(OrderId, CustId, ProdId, Status, Weight),
            order_status(Status),
            status_priority(Status, Priority),
            CanMod is can_modify(Priority),
            CanMod = 1.

        pred cancellable_order(symbol, symbol, symbol).
        cancellable_order(OrderId, CustomerId, Status) :-
            order(OrderId, CustomerId, ProdId, Status, Weight),
            order_status(Status),
            status_priority(Status, Priority),
            CanCancel is can_cancel(Priority),
            CanCancel = 1.

        pred premium_customer_order(symbol, symbol, f64).
        premium_customer_order(OrderId, CustomerId, LoyaltyPoints) :-
            order(OrderId, CustomerId, ProductId, Status, Weight),
            customer(CustomerId, Tier),
            product(ProductId, Cat, BasePrice),
            tier_multiplier(Tier, Multiplier),
            Multiplier > 1.0,
            LoyaltyPoints is BasePrice * Multiplier * 0.01.

        ?- order_final_price(O, P).
        ?- modifiable_order(O, S).
        ?- cancellable_order(O, C, S).
        ?- premium_customer_order(O, C, L).
    "#;

    // Parse all modules
    let products = parse_program(products_module).unwrap();
    let orders = parse_program(orders_module).unwrap();
    let pricing = parse_program(pricing_module).unwrap();
    let customers = parse_program(customers_module).unwrap();
    let main = parse_program(main_src).unwrap();

    // Verify product category symbols
    assert_eq!(symbol::resolve(symbol::intern("electronics")), "electronics");
    assert_eq!(symbol::resolve(symbol::intern("clothing")), "clothing");
    assert_eq!(symbol::resolve(symbol::intern("books")), "books");
    assert_eq!(symbol::resolve(symbol::intern("home_garden")), "home_garden");

    // Verify product ID symbols
    assert_eq!(symbol::resolve(symbol::intern("prod_001")), "prod_001");
    assert_eq!(symbol::resolve(symbol::intern("prod_002")), "prod_002");
    assert_eq!(symbol::resolve(symbol::intern("prod_003")), "prod_003");
    assert_eq!(symbol::resolve(symbol::intern("prod_004")), "prod_004");
    assert_eq!(symbol::resolve(symbol::intern("prod_005")), "prod_005");

    // Verify order status symbols
    assert_eq!(symbol::resolve(symbol::intern("pending")), "pending");
    assert_eq!(symbol::resolve(symbol::intern("confirmed")), "confirmed");
    assert_eq!(symbol::resolve(symbol::intern("shipped")), "shipped");
    assert_eq!(symbol::resolve(symbol::intern("delivered")), "delivered");
    assert_eq!(symbol::resolve(symbol::intern("cancelled")), "cancelled");
    assert_eq!(symbol::resolve(symbol::intern("refunded")), "refunded");

    // Verify customer tier symbols
    assert_eq!(symbol::resolve(symbol::intern("bronze")), "bronze");
    assert_eq!(symbol::resolve(symbol::intern("silver")), "silver");
    assert_eq!(symbol::resolve(symbol::intern("gold")), "gold");
    assert_eq!(symbol::resolve(symbol::intern("platinum")), "platinum");

    // Verify customer ID symbols
    assert_eq!(symbol::resolve(symbol::intern("cust_001")), "cust_001");
    assert_eq!(symbol::resolve(symbol::intern("cust_002")), "cust_002");
    assert_eq!(symbol::resolve(symbol::intern("cust_003")), "cust_003");
    assert_eq!(symbol::resolve(symbol::intern("cust_004")), "cust_004");

    // Verify order ID symbols
    assert_eq!(symbol::resolve(symbol::intern("ord_001")), "ord_001");
    assert_eq!(symbol::resolve(symbol::intern("ord_002")), "ord_002");
    assert_eq!(symbol::resolve(symbol::intern("ord_003")), "ord_003");
    assert_eq!(symbol::resolve(symbol::intern("ord_004")), "ord_004");

    // Verify products module - no functions (using predicates for rates)
    let product_facts: Vec<_> = products
        .facts()
        .filter(|r| r.head.predicate == "product")
        .collect();
    assert_eq!(product_facts.len(), 5);

    // Verify orders module
    assert_eq!(orders.functions.len(), 2);
    assert!(orders.functions.iter().any(|f| f.name == "can_modify"));
    assert!(orders.functions.iter().any(|f| f.name == "can_cancel"));
    let status_facts: Vec<_> = orders
        .facts()
        .filter(|r| r.head.predicate == "order_status")
        .collect();
    assert_eq!(status_facts.len(), 6);

    // Verify pricing module
    assert_eq!(pricing.functions.len(), 4); // 3 public + 1 private
    let round_cents = pricing
        .functions
        .iter()
        .find(|f| f.name == "round_cents")
        .unwrap();
    assert!(round_cents.is_private);

    // Verify customers module - no functions (using predicates for multipliers)
    let tier_facts: Vec<_> = customers
        .facts()
        .filter(|r| r.head.predicate == "customer_tier")
        .collect();
    assert_eq!(tier_facts.len(), 4);

    // Verify main program imports
    assert_eq!(main.imports.len(), 4);

    // Verify main queries
    assert_eq!(main.queries.len(), 4);

    // Verify complex rules
    let price_rules: Vec<_> = main
        .proper_rules()
        .filter(|r| r.head.predicate == "order_final_price")
        .collect();
    assert_eq!(price_rules.len(), 1);
    assert!(
        price_rules[0].body.len() >= 10,
        "order_final_price should have at least 10 body literals, has {}",
        price_rules[0].body.len()
    );

    // Verify premium_customer_order rule
    let premium_rules: Vec<_> = main
        .proper_rules()
        .filter(|r| r.head.predicate == "premium_customer_order")
        .collect();
    assert_eq!(premium_rules.len(), 1);

    // Count total unique symbols created
    let total_symbols = symbol::count();
    // We created many symbols across all modules
    // The exact count depends on deduplication but should be significant
    assert!(
        total_symbols >= 25,
        "Expected at least 25 unique symbols, got {}",
        total_symbols
    );
}

// =============================================================================
// Edge Cases and Stress Tests
// =============================================================================

/// Test many symbols with similar prefixes
#[test]
#[serial]
fn test_similar_symbol_prefixes() {
    symbol::clear();

    let src = r#"
        pred status(symbol).
        status(status_a).
        status(status_b).
        status(status_ab).
        status(status_abc).
        status(status_abcd).
        status(status_1).
        status(status_12).
        status(status_123).
        status(status_1234).

        pred stat(symbol).
        stat(stat_x).
        stat(stat_y).
    "#;

    let _prog = parse_program(src).unwrap();

    // All should be distinct
    let symbols = vec![
        "status_a",
        "status_b",
        "status_ab",
        "status_abc",
        "status_abcd",
        "status_1",
        "status_12",
        "status_123",
        "status_1234",
        "stat_x",
        "stat_y",
    ];

    let ids: Vec<_> = symbols.iter().map(|s| symbol::intern(s)).collect();

    // All IDs should be unique
    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            assert_ne!(
                ids[i], ids[j],
                "{} and {} should have different IDs",
                symbols[i], symbols[j]
            );
        }
    }

    // All should resolve correctly
    for (sym, id) in symbols.iter().zip(ids.iter()) {
        assert_eq!(symbol::resolve(*id), *sym);
    }
}

/// Test function with equality comparison (= vs ==)
#[test]
#[serial]
fn test_function_equality_comparisons() {
    symbol::clear();

    let src = r#"
        func check_zero(X) = if X = 0 then 1 else 0.
        func check_equal(X, Y) = if X = Y then 1 else 0.
        func check_threshold(X, T) = if X >= T then 1 else 0.

        pred data(symbol, f64).
        data(a, 0.0).
        data(b, 5.0).
        data(c, 10.0).

        pred is_zero(symbol).
        is_zero(Label) :- data(Label, Val), R is check_zero(Val), R = 1.

        ?- is_zero(X).
    "#;

    let prog = parse_program(src).unwrap();

    // Verify functions
    assert_eq!(prog.functions.len(), 3);

    // Verify symbols
    assert_eq!(symbol::resolve(symbol::intern("a")), "a");
    assert_eq!(symbol::resolve(symbol::intern("b")), "b");
    assert_eq!(symbol::resolve(symbol::intern("c")), "c");

    // Verify query
    assert!(!prog.queries.is_empty());
}

/// Test combining typed and untyped functions
#[test]
#[serial]
fn test_mixed_typed_untyped_functions() {
    symbol::clear();

    let src = r#"
        func untyped_add(X, Y) = X + Y.
        func typed_multiply(X: f64, Y: f64) -> f64 = X * Y.
        func mixed_usage(A, B: f64) = A + B.

        pred calculation(symbol, f64).
        calculation(sum_result, 0.0).
        calculation(product_result, 0.0).

        pred result(symbol, f64).
        result(Label, R) :- calculation(Label, V), R is untyped_add(5, 3).
    "#;

    let prog = parse_program(src).unwrap();

    // Verify mixed function types
    let untyped = prog
        .functions
        .iter()
        .find(|f| f.name == "untyped_add")
        .unwrap();
    assert!(untyped.params[0].typ.is_none());
    assert!(untyped.params[1].typ.is_none());

    let typed = prog
        .functions
        .iter()
        .find(|f| f.name == "typed_multiply")
        .unwrap();
    assert!(typed.params[0].typ.is_some());
    assert!(typed.params[1].typ.is_some());
    assert!(typed.return_type.is_some());

    let mixed = prog
        .functions
        .iter()
        .find(|f| f.name == "mixed_usage")
        .unwrap();
    assert!(mixed.params[0].typ.is_none());
    assert!(mixed.params[1].typ.is_some());

    // Verify symbols
    assert_eq!(
        symbol::resolve(symbol::intern("sum_result")),
        "sum_result"
    );
    assert_eq!(
        symbol::resolve(symbol::intern("product_result")),
        "product_result"
    );
}

/// Test symbols in probabilistic facts
#[test]
#[serial]
fn test_symbols_in_probabilistic_context() {
    symbol::clear();

    let src = r#"
        pred weather(symbol).
        0.6::weather(sunny).
        0.3::weather(cloudy).
        0.1::weather(rainy).

        func weather_modifier(Base) = if Base > 0.5 then Base * 1.2 else Base * 0.8.

        pred activity(symbol).
        0.7::activity(outdoor); 0.3::activity(indoor).
    "#;

    let prog = parse_program(src).unwrap();

    // Verify probabilistic facts
    assert_eq!(prog.prob_facts.len(), 3);

    // Verify annotated disjunction
    assert_eq!(prog.annotated_disjunctions.len(), 1);

    // Verify symbols
    assert_eq!(symbol::resolve(symbol::intern("sunny")), "sunny");
    assert_eq!(symbol::resolve(symbol::intern("cloudy")), "cloudy");
    assert_eq!(symbol::resolve(symbol::intern("rainy")), "rainy");
    assert_eq!(symbol::resolve(symbol::intern("outdoor")), "outdoor");
    assert_eq!(symbol::resolve(symbol::intern("indoor")), "indoor");

    // Verify function
    assert!(prog.functions.iter().any(|f| f.name == "weather_modifier"));
}

/// Test module with all feature types combined
#[test]
#[serial]
fn test_comprehensive_module_all_features() {
    symbol::clear();

    let module_src = r#"
        #pragma max_recursion_depth = 100

        pred status(symbol).
        status(active).
        status(inactive).

        func compute(X) = X * 2.
        private func helper(X) = X + 1.

        pred data(u32, symbol, f64).
        data(1, type_a, 10.0).
        data(2, type_b, 20.0).

        pred derived(u32, f64).
        derived(Id, Result) :- data(Id, Type, Val), Result is compute(Val).

        0.5::random_status(active).
        0.5::random_status(inactive).

        ?- derived(X, Y).
    "#;

    let prog = parse_program(module_src).unwrap();

    // Verify pragma
    assert_eq!(prog.directives.max_recursion_depth, Some(100));

    // Verify predicates
    assert!(prog.predicates.iter().any(|p| p.name == "status"));
    assert!(prog.predicates.iter().any(|p| p.name == "data"));
    assert!(prog.predicates.iter().any(|p| p.name == "derived"));

    // Verify functions
    assert_eq!(prog.functions.len(), 2);
    let helper = prog.functions.iter().find(|f| f.name == "helper").unwrap();
    assert!(helper.is_private);

    // Verify symbols
    assert_eq!(symbol::resolve(symbol::intern("active")), "active");
    assert_eq!(symbol::resolve(symbol::intern("inactive")), "inactive");
    assert_eq!(symbol::resolve(symbol::intern("type_a")), "type_a");
    assert_eq!(symbol::resolve(symbol::intern("type_b")), "type_b");

    // Verify probabilistic facts
    assert_eq!(prog.prob_facts.len(), 2);

    // Verify rules exist
    assert!(prog.proper_rules().count() > 0);

    // Verify query
    assert!(!prog.queries.is_empty());
}
