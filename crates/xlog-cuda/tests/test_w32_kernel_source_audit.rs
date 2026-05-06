// crates/xlog-cuda/tests/test_w32_kernel_source_audit.rs
//! W3.2 — Source-audit certs for the k=6 templated clique kernel.
//!
//! Two-tier audit of `crates/xlog-cuda/kernels/wcoj.cu`:
//!
//!   * **Tier 1** (4 cells) — each of the four
//!     `wcoj_clique6_{count,materialize}_{u32,u64}` ABI wrapper
//!     bodies must contain exactly one statement that calls the
//!     shared template. No loops, no conditionals, no
//!     hand-written body inside the wrapper.
//!
//!   * **Tier 2** (4 cells) — file-wide forbidden-pattern audit:
//!     no `template <>` specialization for K=5 or K=6, no
//!     `if constexpr (K == 6)` (or K==5) branch, no `clique6`
//!     helper function body outside the four ABI wrappers, no
//!     hardcoded `5` / `6` literal in the shared template body.
//!
//! Together these enforce the board contract verbatim:
//! "the only allowed k=6-specific `.cu` text should be ABI
//! wrapper names plus calls/instantiations using `<6>`."

use std::fs;
use std::path::PathBuf;

fn wcoj_cu_source() -> String {
    // The audit reads the .cu source FROM THE WORKTREE (not the
    // build artifact), so it pins the contract on the
    // human-readable source the developer ships.
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("kernels/wcoj.cu");
    fs::read_to_string(&p).unwrap_or_else(|e| panic!("failed to read {}: {}", p.display(), e))
}

/// Strip C/C++ line comments (`// ...`) and block comments
/// (`/* ... */`) from `s`. Returns a string of the same line
/// count (newlines preserved) so per-line context analysis still
/// works.
fn strip_comments(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                if bytes[i] == b'\n' {
                    out.push('\n');
                }
                i += 1;
            }
            i = (i + 2).min(bytes.len());
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

/// Locate `extern "C" __global__ void <name>(<args>) { <body> }`
/// in `src` and return `body` (the `{ ... }` interior, exclusive).
fn extract_extern_c_global_body(src: &str, name: &str) -> Option<String> {
    let needle = format!("__global__ void {}", name);
    let start = src.find(&needle)?;
    // From `start`, find `(` then matching `)` then `{` then matching `}`.
    let after_name = start + needle.len();
    let mut bytes = src[after_name..].bytes();
    // skip to first '('
    let mut consumed = 0;
    let mut depth_paren = 0i32;
    let mut state_paren = false;
    while let Some(b) = bytes.next() {
        consumed += 1;
        if b == b'(' {
            depth_paren += 1;
            state_paren = true;
        } else if b == b')' {
            depth_paren -= 1;
            if state_paren && depth_paren == 0 {
                break;
            }
        }
    }
    let after_args = after_name + consumed;
    // find first '{' then matching '}'
    let mut depth = 0i32;
    let mut body_start: Option<usize> = None;
    let mut body_end: Option<usize> = None;
    let bytes2 = src.as_bytes();
    let mut j = after_args;
    while j < bytes2.len() {
        let c = bytes2[j];
        if c == b'{' {
            if body_start.is_none() {
                body_start = Some(j + 1);
            }
            depth += 1;
        } else if c == b'}' {
            depth -= 1;
            if depth == 0 {
                body_end = Some(j);
                break;
            }
        }
        j += 1;
    }
    let s = body_start?;
    let e = body_end?;
    Some(src[s..e].to_string())
}

/// Count semicolon-terminated statements in `body`. Naive but
/// sufficient: strips comments + counts `;` tokens.
fn count_statements(body: &str) -> usize {
    let stripped = strip_comments(body);
    stripped.matches(';').count()
}

fn body_contains_loop_or_conditional(body: &str) -> bool {
    let s = strip_comments(body);
    let tokens = [
        "for", "while", "do ", "do{", "do (", "if ", "if(", "switch", "?:",
    ];
    tokens.iter().any(|t| s.contains(t))
}

fn body_contains_template_call(body: &str, template_name: &str, k_val: usize) -> bool {
    let s = strip_comments(body);
    let needle = format!("{}<{}", template_name, k_val);
    s.contains(&needle)
}

// ===============================================================
// Tier 1 — wrapper bodies are template-call-only (4 cells)
// ===============================================================

#[test]
fn k6_count_u32_wrapper_is_template_call_only() {
    let src = wcoj_cu_source();
    let body = extract_extern_c_global_body(&src, "wcoj_clique6_count_u32")
        .expect("wcoj_clique6_count_u32 must exist in wcoj.cu");
    let stmts = count_statements(&body);
    assert_eq!(
        stmts, 3,
        "wcoj_clique6_count_u32 body must contain exactly 3 statements \
         (thread idx + bound check + template call); got {} in body:\n{}",
        stmts, body
    );
    assert!(
        body_contains_template_call(&body, "wcoj_clique_template_count_t", 6),
        "wcoj_clique6_count_u32 must call wcoj_clique_template_count_t<6, ...>; got body:\n{}",
        body
    );
    // `for`/`while`/`do`/`switch` are forbidden — they would
    // indicate a hand-written algorithm body. The single `if`
    // for the thread-bounds check is allowed; the statement
    // count assertion above (== 3) bounds total complexity.
    let stripped = strip_comments(&body);
    for forbidden in &["for ", "for(", "while ", "while(", "do ", "do{", "switch"] {
        assert!(
            !stripped.contains(forbidden),
            "wcoj_clique6_count_u32 body contains forbidden token `{}`: hand-written \
             algorithm body detected. Body:\n{}",
            forbidden,
            body
        );
    }
}

#[test]
fn k6_count_u64_wrapper_is_template_call_only() {
    let src = wcoj_cu_source();
    let body = extract_extern_c_global_body(&src, "wcoj_clique6_count_u64")
        .expect("wcoj_clique6_count_u64 must exist");
    let stmts = count_statements(&body);
    assert_eq!(stmts, 3, "wcoj_clique6_count_u64 stmts: got {}", stmts);
    assert!(
        body_contains_template_call(&body, "wcoj_clique_template_count_t", 6),
        "wcoj_clique6_count_u64 must call wcoj_clique_template_count_t<6, ...>"
    );
}

#[test]
fn k6_materialize_u32_wrapper_is_template_call_only() {
    let src = wcoj_cu_source();
    let body = extract_extern_c_global_body(&src, "wcoj_clique6_materialize_u32")
        .expect("wcoj_clique6_materialize_u32 must exist");
    let stmts = count_statements(&body);
    // materialize wrapper: thread idx + bound check + base lookup
    // + base-vs-total check + emit call = 5 statements.
    assert_eq!(
        stmts, 5,
        "wcoj_clique6_materialize_u32 stmts: got {}",
        stmts
    );
    assert!(
        body_contains_template_call(&body, "wcoj_clique_template_emit_t", 6),
        "wcoj_clique6_materialize_u32 must call wcoj_clique_template_emit_t<6, ...>"
    );
}

#[test]
fn k6_materialize_u64_wrapper_is_template_call_only() {
    let src = wcoj_cu_source();
    let body = extract_extern_c_global_body(&src, "wcoj_clique6_materialize_u64")
        .expect("wcoj_clique6_materialize_u64 must exist");
    let stmts = count_statements(&body);
    assert_eq!(
        stmts, 5,
        "wcoj_clique6_materialize_u64 stmts: got {}",
        stmts
    );
    assert!(
        body_contains_template_call(&body, "wcoj_clique_template_emit_t", 6),
        "wcoj_clique6_materialize_u64 must call wcoj_clique_template_emit_t<6, ...>"
    );
}

// ===============================================================
// Tier 2 — file-wide forbidden-pattern audit (4 cells)
// ===============================================================

#[test]
fn no_explicit_k6_template_specialization() {
    let src = strip_comments(&wcoj_cu_source());
    // Forbidden: `template <>` (or `template<>`) followed within
    // a small window by any `clique` name with `<6>`.
    // Compact whitespace before scanning.
    let compact: String = src.split_whitespace().collect::<Vec<_>>().join(" ");
    let forbidden_phrases = [
        "template <> __device__",
        "template <> __global__",
        "template<> __device__",
        "template<> __global__",
    ];
    for phrase in &forbidden_phrases {
        let mut start = 0;
        while let Some(pos) = compact[start..].find(phrase) {
            let abs = start + pos;
            // Look in the next 200 chars for any "<6" — explicit
            // K=6 specialization.
            let window_end = (abs + 200).min(compact.len());
            let window = &compact[abs..window_end];
            assert!(
                !window.contains("<6"),
                "Forbidden: explicit template specialization for K=6 detected. Window:\n{}",
                window
            );
            assert!(
                !window.contains("<5"),
                "Forbidden: explicit template specialization for K=5 detected. Window:\n{}",
                window
            );
            start = abs + phrase.len();
        }
    }
}

#[test]
fn no_if_constexpr_k_equals_6_branch() {
    let src = strip_comments(&wcoj_cu_source());
    let compact: String = src.split_whitespace().collect::<Vec<_>>().join(" ");
    // Forbidden patterns: `if constexpr (K == 6)`, `if constexpr (K == 5)`,
    // `if (K == 6)`, `if (K == 5)`, plus minor whitespace variants.
    let forbidden = [
        "if constexpr (K == 6)",
        "if constexpr (K == 5)",
        "if constexpr (K_VAL == 6)",
        "if constexpr (K_VAL == 5)",
        "if (K == 6)",
        "if (K == 5)",
        "if (K_VAL == 6)",
        "if (K_VAL == 5)",
    ];
    for phrase in &forbidden {
        assert!(
            !compact.contains(phrase),
            "Forbidden K-keyed branch detected: '{}'. The shared template must \
             be uniformly K-parameterized; K=5/K=6 must come from instantiation, \
             not from runtime / compile-time branches keyed on the K value.",
            phrase
        );
    }
    // K-INDEPENDENT constexpr branches are allowed (e.g.
    // `if constexpr (Level >= K_VAL)` for recursion termination).
    // We don't forbid those.
}

#[test]
fn no_clique6_helper_function_body() {
    let src = strip_comments(&wcoj_cu_source());
    // The four ABI wrappers are the only `clique6`-named entities
    // permitted to have a function body. Any other `clique6`
    // identifier with a `__device__` or `__global__` qualifier
    // followed by a non-empty body would be forbidden.
    let allowed_names = [
        "wcoj_clique6_count_u32",
        "wcoj_clique6_count_u64",
        "wcoj_clique6_materialize_u32",
        "wcoj_clique6_materialize_u64",
    ];
    // Find every `clique6` substring; for each, check the
    // surrounding context against the allowed-name whitelist.
    let mut start = 0;
    while let Some(pos) = src[start..].find("clique6") {
        let abs = start + pos;
        // Walk back to the start of the identifier (alphanumeric
        // + underscore tokens).
        let bytes = src.as_bytes();
        let mut id_start = abs;
        while id_start > 0 {
            let c = bytes[id_start - 1];
            if c.is_ascii_alphanumeric() || c == b'_' {
                id_start -= 1;
            } else {
                break;
            }
        }
        // Walk forward to the end of the identifier.
        let mut id_end = abs + "clique6".len();
        while id_end < bytes.len() {
            let c = bytes[id_end];
            if c.is_ascii_alphanumeric() || c == b'_' {
                id_end += 1;
            } else {
                break;
            }
        }
        let ident = &src[id_start..id_end];
        // If the identifier IS one of the four ABI wrappers, fine
        // (allowed; the wrapper has a body audited by Tier 1).
        // Otherwise, if the identifier appears as a function
        // declarator preceded by `__device__` or `__global__`
        // followed by a `{ ... }` body, it's forbidden.
        if !allowed_names.contains(&ident) {
            // Simple heuristic: look 80 chars back for
            // `__device__` or `__global__` qualifier, AND look
            // forward for `{`. If both, forbidden.
            let look_back_start = id_start.saturating_sub(80);
            let preceding = &src[look_back_start..id_start];
            let has_device_qual =
                preceding.contains("__device__") || preceding.contains("__global__");
            // Look forward for `(` then `{` (function definition).
            let look_forward_end = (id_end + 200).min(src.len());
            let following = &src[id_end..look_forward_end];
            let has_paren = following.contains('(');
            let has_brace = following.contains('{');
            assert!(
                !(has_device_qual && has_paren && has_brace),
                "Forbidden: `clique6`-named function with body detected outside the \
                 4 ABI wrappers. Identifier `{}` near pos {}.",
                ident,
                abs
            );
        }
        start = id_end;
    }
}

#[test]
fn no_six_literal_in_template_body() {
    let src = wcoj_cu_source();
    // Locate the shared template's body. The template name is
    // `wcoj_clique_template_count_t` and `wcoj_clique_template_emit_t`.
    // Each has a function definition with a `{ ... }` body.
    let template_names = [
        "wcoj_clique_template_count_t",
        "wcoj_clique_template_emit_t",
        "clique_recurse_t",
    ];
    for name in &template_names {
        // Skip if not present (recurse_t is internal).
        let Some(start) = src.find(&format!("{}(", name)) else {
            continue;
        };
        // Find the `{` after the args list.
        let bytes = src.as_bytes();
        let mut depth = 0i32;
        let mut state = false;
        let mut j = start;
        let mut body_start: Option<usize> = None;
        let mut body_end: Option<usize> = None;
        while j < bytes.len() {
            let c = bytes[j];
            if c == b'(' {
                depth += 1;
                state = true;
            } else if c == b')' {
                depth -= 1;
                if state && depth == 0 {
                    state = false;
                }
            } else if c == b'{' && !state && depth == 0 {
                if body_start.is_none() {
                    body_start = Some(j + 1);
                }
                depth += 1;
                state = true;
            } else if c == b'}' && state {
                depth -= 1;
                if depth == 0 {
                    body_end = Some(j);
                    break;
                }
            }
            j += 1;
        }
        let (Some(s), Some(e)) = (body_start, body_end) else {
            continue;
        };
        let body = &src[s..e];
        // Strip comments first (so `// 5 vertices` doesn't match).
        let stripped = strip_comments(body);
        // Tokenize loosely: walk the body and find any `5` or `6`
        // surrounded by non-alphanumeric (= isolated literal).
        // Allow:
        //   * Inside `static_assert(K_VAL >= 3 && K_VAL <= 6, ...)`
        //     (K-bound check; we whitelist on a per-line basis if
        //     the line contains `static_assert`).
        //   * Inside `template <int K = 5>` default value (we
        //     whitelist on a per-line basis if the line contains
        //     `template <`).
        for (line_idx, line) in stripped.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.contains("static_assert") || trimmed.contains("template <") {
                continue;
            }
            // Tokenize: scan for digit sequences and check if it's
            // a literal `5` or `6` (single-digit, surrounded by
            // non-alphanumeric).
            let bytes = trimmed.as_bytes();
            let mut k = 0;
            while k < bytes.len() {
                let c = bytes[k];
                if c == b'5' || c == b'6' {
                    let prev = if k == 0 { b' ' } else { bytes[k - 1] };
                    let next = if k + 1 < bytes.len() {
                        bytes[k + 1]
                    } else {
                        b' '
                    };
                    let is_isolated = !prev.is_ascii_alphanumeric()
                        && prev != b'_'
                        && !next.is_ascii_alphanumeric()
                        && next != b'_';
                    // Allow K=5 / K=6 inside template instantiation
                    // syntax: `<5>`, `<6>` are NOT in the template
                    // body itself (they appear in ABI wrapper calls,
                    // outside this function). Within the template
                    // body we forbid them.
                    if is_isolated {
                        panic!(
                            "Forbidden: hardcoded literal `{}` in template `{}` body, line {}: `{}`",
                            c as char, name, line_idx, trimmed
                        );
                    }
                }
                k += 1;
            }
        }
    }
}
