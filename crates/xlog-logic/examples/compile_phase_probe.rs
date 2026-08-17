//! Frontend phase-timing and plan-digest probe.
//!
//! Run with: cargo run -p xlog-logic --release --example compile_phase_probe
//!
//! Times parse / normalize / magic-sets / stratify / lower+optimize on three
//! synthetic workload shapes (N distinct-head rules, N facts + 2 rules,
//! N same-head candidate rules) and prints canonical plan digests (RelIds
//! renamed to predicate names, SCC ids replaced by sorted member sets) for
//! plan-equivalence comparison across compiler versions and processes.

use std::time::Instant;
use xlog_logic::compile::Compiler;
use xlog_logic::{
    normalize_list_builtins, normalize_meta_builtins, parse_program, rewrite_magic_sets, stratify,
};

fn ms<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = Instant::now();
    let r = f();
    (r, t.elapsed().as_secs_f64() * 1000.0)
}

fn profile(name: &str, src: &str) {
    let (parsed, t_parse) = ms(|| parse_program(src).expect("parse"));
    let (meta, t_meta) = ms(|| normalize_meta_builtins(&parsed).expect("meta"));
    let (list, t_list) = ms(|| normalize_list_builtins(&meta).expect("list"));
    let (magic, t_magic) = ms(|| rewrite_magic_sets(&list).expect("magic"));
    let (_strata, t_strat) = ms(|| stratify(&magic.program).expect("stratify"));
    // full pipeline from AST (desugar..promote), includes everything above except parse
    let mut c = Compiler::new();
    let (_plan, t_prog) = ms(|| c.compile_program(&parsed).expect("compile_program"));
    let t_rest = t_prog - (t_meta + t_list + t_magic + t_strat);
    println!(
        "{name}\tbytes={}\tparse={t_parse:.1}\tmeta={t_meta:.1}\tlist={t_list:.1}\tmagic={t_magic:.1}\tstratify={t_strat:.1}\tlower+opt={t_rest:.1}\tTOTAL={:.1}",
        src.len(),
        t_parse + t_prog
    );
}

fn rules_workload(n: usize) -> String {
    let mut s = String::new();
    for i in 0..n {
        s.push_str(&format!(
            "p{i}(X, Z) :- e{i}(X, Y), f{i}(Y, W), g{i}(W, Z).\n"
        ));
    }
    s.push_str("reach(X,Y) :- e0(X,Y).\nreach(X,Z) :- reach(X,Y), e0(Y,Z).\n");
    s
}

// ILP shape: many facts, few rules (like the kfold 10^4-events case)
fn facts_workload(n: usize) -> String {
    let mut s = String::new();
    for i in 0..n {
        s.push_str(&format!("event({i}, {}, {}).\n", i % 97, i % 13));
    }
    s.push_str("happens(E, T) :- event(E, T, _).\n");
    s.push_str("pair(A, B) :- event(A, T, K), event(B, T, K).\n");
    s
}

// same-predicate rule flood: N rules with the SAME head (ILP candidate set shape)
fn candidates_workload(n: usize) -> String {
    let mut s = String::new();
    s.push_str("event(1, 2, 3).\n");
    for i in 0..n {
        s.push_str(&format!(
            "cand(E, T) :- event(E, T, {}), event(E, {}, _).\n",
            i % 50,
            i % 20
        ));
    }
    s
}

fn digest(src: &str, tag: &str) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut c = Compiler::new();
    let plan = c.compile(src).expect("compile");
    let name_by_id: std::collections::HashMap<u32, &str> = c
        .rel_ids()
        .iter()
        .map(|(name, id)| (id.0, name.as_str()))
        .collect();
    let rename = |txt: &str| -> String {
        let mut out = txt.to_string();
        for (id, name) in &name_by_id {
            out = out.replace(&format!("RelId({id})"), &format!("Rel<{name}>"));
        }
        out
    };
    let mut arities: Vec<_> = plan
        .rel_arities
        .iter()
        .map(|(k, v)| (name_by_id.get(&k.0).copied().unwrap_or("?").to_string(), *v))
        .collect();
    arities.sort();
    let scc_key = |scc: &xlog_ir::Scc| -> String {
        let mut p = scc.predicates.clone();
        p.sort();
        format!("{:?}|{}", p, scc.is_recursive)
    };
    // SCC id-independent: pair each SCC's canonical key with its sorted rules.
    let mut scc_units: Vec<String> = plan
        .sccs
        .iter()
        .map(|scc| {
            let mut v: Vec<String> = plan.rules_by_scc[scc.id as usize]
                .iter()
                .map(|r| rename(&format!("{:?}", r)))
                .collect();
            v.sort();
            format!("{}::{:?}", scc_key(scc), v)
        })
        .collect();
    scc_units.sort();
    let scc_key_by_id: std::collections::HashMap<u32, String> =
        plan.sccs.iter().map(|s| (s.id, scc_key(s))).collect();
    let strata_canon: Vec<Vec<String>> = plan
        .strata
        .iter()
        .map(|s| {
            let mut v: Vec<String> = s
                .sccs
                .iter()
                .map(|x| scc_key_by_id.get(x).cloned().unwrap_or_default())
                .collect();
            v.sort();
            v
        })
        .collect();
    let repr = format!("{:?}|{:?}|{:?}", strata_canon, scc_units, arities);
    let mut h = DefaultHasher::new();
    repr.hash(&mut h);
    println!(
        "digest {tag}	strata={}	sccs={}	rules={}	hash={:016x}",
        plan.strata.len(),
        plan.sccs.len(),
        plan.rules_by_scc.iter().map(|r| r.len()).sum::<usize>(),
        h.finish()
    );
}

fn main() {
    println!("== workload: N distinct-head rules ==");
    for n in [100usize, 200, 400, 800, 1600] {
        profile(&format!("rules-{n}"), &rules_workload(n));
    }
    println!("== workload: N facts + 2 rules (ILP kfold shape) ==");
    for n in [1000usize, 2000, 4000, 8000, 16000] {
        profile(&format!("facts-{n}"), &facts_workload(n));
    }
    println!("== workload: N same-head candidate rules ==");
    for n in [100usize, 200, 400, 800] {
        profile(&format!("cands-{n}"), &candidates_workload(n));
    }
    println!("== plan digests ==");
    for n in [100usize, 800] {
        digest(&rules_workload(n), &format!("rules-{n}"));
        digest(&candidates_workload(n), &format!("cands-{n}"));
    }
    digest(&facts_workload(4000), "facts-4000");
}
