use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_PROGRAM_ID: AtomicU64 = AtomicU64::new(0);

struct TempProgram {
    path: PathBuf,
}

impl TempProgram {
    fn new(source: &str) -> Self {
        loop {
            let id = NEXT_PROGRAM_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "xlog-ground-term-encoding-{}-{id}.xlog",
                std::process::id()
            ));
            let mut file = match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create temporary XLOG program: {error}"),
            };
            if let Err(error) = file.write_all(source.as_bytes()) {
                drop(file);
                let _ = fs::remove_file(&path);
                panic!("write temporary XLOG program: {error}");
            }
            return Self { path };
        }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempProgram {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn run_xlog(source: &str) -> Output {
    let program = TempProgram::new(source);
    Command::new(env!("CARGO_BIN_EXE_xlog_run"))
        .arg(program.path())
        .arg("--memory-mb")
        .arg("256")
        .output()
        .expect("run xlog_run")
}

#[test]
fn xlog_run_uses_shared_ground_term_encoding() {
    let _device = match xlog_cuda::CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };

    let supported = run_xlog(
        r#"
            pred encoded(u32, u64, i32, i64, f32, f64, bool, bool, symbol, symbol).
            encoded(42, 43, -44, -45, 1.5, 2.25, true, 0, "hello", world).
            ?- encoded(A, B, C, D, E, F, G, H, I, J).
        "#,
    );
    assert!(
        supported.status.success(),
        "supported typed facts failed:\n{}",
        String::from_utf8_lossy(&supported.stderr)
    );
    let stdout = String::from_utf8_lossy(&supported.stdout);
    assert!(
        stdout
            .contains("A=42, B=43, C=-44, D=-45, E=1.5, F=2.25, G=true, H=false, I=hello, J=world"),
        "{stdout}"
    );

    let invalid = run_xlog(
        r#"
            pred invalid(u32).
            invalid(X).
        "#,
    );
    assert!(!invalid.status.success(), "invalid fact unexpectedly ran");
    let stderr = String::from_utf8_lossy(&invalid.stderr);
    assert!(
        stderr.contains(
            "Failed to encode fact for predicate invalid at column 0: Fact cannot contain variable X"
        ),
        "{stderr}"
    );
}
