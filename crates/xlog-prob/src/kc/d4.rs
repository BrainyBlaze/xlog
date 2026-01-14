//! D4 backend wrapper for compiling CNF to Decision-DNNF.

use std::path::{Path, PathBuf};
use std::process::Command;

use xlog_core::{Result, XlogError};

#[derive(Debug, Clone)]
pub struct D4Compiler {
    d4_path: PathBuf,
}

impl D4Compiler {
    pub fn new(d4_path: impl Into<PathBuf>) -> Self {
        Self {
            d4_path: d4_path.into(),
        }
    }

    pub fn bundled() -> Result<Self> {
        let path = option_env!("XLOG_PROB_D4_PATH").ok_or_else(|| {
            XlogError::Compilation(
                "Bundled D4 path not available: XLOG_PROB_D4_PATH not set at compile time"
                    .to_string(),
            )
        })?;
        let path = PathBuf::from(path);
        if !path.exists() {
            return Err(XlogError::Compilation(format!(
                "Bundled D4 binary missing at {}",
                path.display()
            )));
        }
        Ok(Self::new(path))
    }

    pub fn detect() -> Result<Self> {
        if let Ok(path) = std::env::var("XLOG_D4_PATH") {
            let trimmed = path.trim();
            if trimmed.is_empty() {
                return Err(XlogError::Compilation(
                    "XLOG_D4_PATH is set but empty".to_string(),
                ));
            }
            return Ok(Self::new(PathBuf::from(trimmed)));
        }
        Self::bundled()
    }

    pub fn d4_path(&self) -> &Path {
        &self.d4_path
    }

    pub fn compile_ddnnf(&self, cnf_path: &Path, out_path: &Path) -> Result<()> {
        if !cnf_path.exists() {
            return Err(XlogError::Compilation(format!(
                "D4 compile error: CNF file not found: {}",
                cnf_path.display()
            )));
        }

        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                XlogError::Execution(format!(
                    "D4 compile error: failed to create output directory {}: {}",
                    parent.display(),
                    e
                ))
            })?;
        }

        let out_arg = format!("-out={}", out_path.display());
        let mut output: Option<std::process::Output> = None;
        let mut last_spawn_err: Option<std::io::Error> = None;

        // Under some filesystems / concurrent build setups, spawning an executable can transiently
        // fail with ETXTBSY ("Text file busy") if the file is momentarily open for writing.
        // Retry with a short backoff to make D4 invocation robust.
        for attempt in 0..8u32 {
            match Command::new(&self.d4_path)
                .arg("-dDNNF")
                .arg(cnf_path)
                .arg(&out_arg)
                .output()
            {
                Ok(out) => {
                    output = Some(out);
                    break;
                }
                Err(e) => {
                    let is_text_file_busy = e.raw_os_error() == Some(26);
                    if is_text_file_busy && attempt < 7 {
                        last_spawn_err = Some(e);
                        std::thread::sleep(std::time::Duration::from_millis(10 * (attempt as u64 + 1)));
                        continue;
                    }
                    return Err(XlogError::Execution(format!(
                        "D4 compile error: failed to spawn {}: {}",
                        self.d4_path.display(),
                        e
                    )));
                }
            }
        }

        let Some(output) = output else {
            let err = last_spawn_err.expect("spawn error set when output missing");
            return Err(XlogError::Execution(format!(
                "D4 compile error: failed to spawn {} after retries: {}",
                self.d4_path.display(),
                err
            )));
        };

        if !output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(XlogError::Execution(format!(
                "D4 compile error: command failed (exit={})\nstdout:\n{}\nstderr:\n{}",
                output.status,
                stdout.trim_end(),
                stderr.trim_end()
            )));
        }

        if !out_path.exists() {
            return Err(XlogError::Execution(format!(
                "D4 compile error: command succeeded but output file missing: {}",
                out_path.display()
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
    mod tests {
        use super::*;
        use crate::kc::ddnnf::DecisionDnnf;
        use crate::xgcf::Xgcf;
        use std::fs;
        use std::fs::OpenOptions;
        use std::os::unix::fs::PermissionsExt;
        use std::path::{Path, PathBuf};
        use std::time::Duration;

    fn make_temp_dir(prefix: &str) -> PathBuf {
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("{}-{}-{}", prefix, pid, nanos));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_executable_script(path: &Path, script: &str) {
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, script).unwrap();
        fs::set_permissions(&tmp, fs::Permissions::from_mode(0o755)).unwrap();
        fs::rename(&tmp, path).unwrap();
    }

    #[test]
    fn test_d4_compile_invokes_binary_and_writes_ddnnf() {
        let dir = make_temp_dir("xlog-d4-test");

        let d4_path = dir.join("d4");
        write_executable_script(
            &d4_path,
            r#"#!/usr/bin/env bash
set -euo pipefail
out=""
for arg in "$@"; do
  case "$arg" in
    -out=*) out="${arg#-out=}" ;;
  esac
done
if [[ -z "$out" ]]; then
  echo "missing -out" >&2
  exit 2
fi
cat > "$out" <<'EOF'
o 1 0
t 2 0
f 3 0
1 2 1 0
1 3 -1 0
EOF
"#,
        );

        let cnf_path = dir.join("in.cnf");
        fs::write(&cnf_path, "c test\np cnf 1 1\n1 0\n").unwrap();
        let out_path = dir.join("out.nnf");

        let compiler = D4Compiler::new(d4_path.clone());
        compiler.compile_ddnnf(&cnf_path, &out_path).unwrap();

        let nnf = fs::read_to_string(&out_path).unwrap();
        let ddnnf = DecisionDnnf::parse_str(&nnf).unwrap();
        let xgcf = Xgcf::from_ddnnf(&ddnnf).unwrap();

        let p = 0.25_f64;
        let log_wmc = xgcf
            .eval_log_wmc(|var| match var {
                1 => (p.ln(), (1.0 - p).ln()),
                _ => panic!("unexpected var {}", var),
            })
            .unwrap();
        assert!((log_wmc - p.ln()).abs() < 1e-9, "log_wmc={}", log_wmc);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
        fn test_d4_compile_reports_failure() {
            let dir = make_temp_dir("xlog-d4-fail-test");

        let d4_path = dir.join("d4");
        write_executable_script(
            &d4_path,
            r#"#!/usr/bin/env bash
echo "boom" >&2
exit 7
"#,
        );

        let cnf_path = dir.join("in.cnf");
        fs::write(&cnf_path, "p cnf 1 1\n1 0\n").unwrap();
        let out_path = dir.join("out.nnf");

        let compiler = D4Compiler::new(d4_path);
        let err = compiler.compile_ddnnf(&cnf_path, &out_path).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("boom"), "msg={}", msg);

            fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn test_d4_compile_retries_text_file_busy() {
            let dir = make_temp_dir("xlog-d4-busy-test");

            let d4_path = dir.join("d4");
            write_executable_script(
                &d4_path,
                r#"#!/usr/bin/env bash
set -euo pipefail
out=""
for arg in "$@"; do
  case "$arg" in
    -out=*) out="${arg#-out=}" ;;
  esac
done
if [[ -z "$out" ]]; then
  echo "missing -out" >&2
  exit 2
fi
cat > "$out" <<'EOF'
o 1 0
t 2 0
f 3 0
1 2 1 0
1 3 -1 0
EOF
"#,
            );

            let cnf_path = dir.join("in.cnf");
            fs::write(&cnf_path, "c test\np cnf 1 1\n1 0\n").unwrap();
            let out_path = dir.join("out.nnf");

            let hold = OpenOptions::new().write(true).open(&d4_path).unwrap();
            let release = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(50));
                drop(hold);
            });

            let compiler = D4Compiler::new(d4_path);
            let result = compiler.compile_ddnnf(&cnf_path, &out_path);

            release.join().unwrap();
            result.unwrap();

            fs::remove_dir_all(&dir).ok();
        }
    }
