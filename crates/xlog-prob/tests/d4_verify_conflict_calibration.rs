//! Conflict-budget calibration harness (RunPod / manual only).
//!
//! Generator is a dense-correlated probabilistic reachability
//! (msg-20260614-170228); reproduction check: the CNF caps MUST match their
//! measured (n=6: 351/668, n=7: 654/1262) or the generators diverged.
//!
//! Run with the conflict budget set, e.g.:
//!   XLOG_DEBUG_VERIFY_SIZE=1 XLOG_D4_VERIFY_MAX_CONFLICTS=200000 \
//!     cargo test -p xlog-prob --test d4_verify_conflict_calibration \
//!     --features host-io --release -- --ignored --nocapture

use xlog_prob::exact::ExactDdnnfProgram;

fn dense_correlated_source(n: u32) -> String {
    let mut s = String::new();
    for i in 1..=n {
        for j in (i + 1)..=n {
            s.push_str(&format!("0.5::edge({i},{j}).\n"));
        }
    }
    s.push_str("reach(X,Y) :- edge(X,Y).\n");
    s.push_str("reach(X,Z) :- reach(X,Y), edge(Y,Z).\n");
    s.push_str(&format!("query(reach(1,{n})).\n"));
    s
}

/// Compile the dense-correlated program at `n`. Returns Ok(()) if the verify
/// COMPLETED (equivalence proven), Err(msg) if it DECLINED / errored. With
/// XLOG_D4_VERIFY_MAX_CONFLICTS set, a budget-exhausted verify returns
/// VerifyBudgetExceeded — the decline we are calibrating.
fn try_compile(n: u32) -> Result<(), String> {
    match ExactDdnnfProgram::compile_source(&dense_correlated_source(n)) {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("{e}")),
    }
}

#[test]
#[ignore = "RunPod/manual calibration; heavy verify"]
fn p03_conflict_budget_calibration() {
    // Set XLOG_DEBUG_VERIFY_SIZE=1 to print caps for the reproduction check
    // (must be n=6: 351/668, n=7: 654/1262). Set XLOG_D4_VERIFY_MAX_CONFLICTS
    // to the budget under test; the harness reports complete-vs-decline.
    let budget = std::env::var("XLOG_D4_VERIFY_MAX_CONFLICTS").unwrap_or_else(|_| "0".into());
    for n in [6u32, 7u32] {
        let outcome = try_compile(n);
        println!(
            "[CALIB] n={n} budget={budget} -> {}",
            match &outcome {
                Ok(()) => "COMPLETED (verify proved equivalence)".to_string(),
                Err(e) => format!("DECLINED/ERR: {e}"),
            }
        );
    }
}
