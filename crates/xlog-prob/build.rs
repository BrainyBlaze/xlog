use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../../vendor/d4/VENDORED_COMMIT");
    println!("cargo:rerun-if-changed=../../vendor/boost/VENDORED_VERSION");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("xlog-prob expected to live under <workspace>/crates/xlog-prob")
        .to_path_buf();

    let vendor_dir = workspace_root.join("vendor").join("d4");
    if !vendor_dir.is_dir() {
        panic!(
            "Vendored D4 not found at {} (expected vendor/d4).",
            vendor_dir.display()
        );
    }

    let vendor_commit = vendor_dir.join("VENDORED_COMMIT");
    if !vendor_commit.is_file() {
        panic!(
            "Vendored D4 missing {} (expected snapshot metadata).",
            vendor_commit.display()
        );
    }

    let boost_dir = workspace_root.join("vendor").join("boost");
    let boost_headers = boost_dir.join("boost");
    if !boost_headers.is_dir() {
        panic!(
            "Vendored Boost headers not found at {} (expected vendor/boost/boost).",
            boost_headers.display()
        );
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let src_dir = out_dir.join("d4-src");
    let bin_dir = out_dir.join("d4-bin");
    let bin_path = bin_dir.join("d4");

    if bin_path.is_file() {
        println!("cargo:rustc-env=XLOG_PROB_D4_PATH={}", bin_path.display());
        return;
    }

    fs::create_dir_all(&bin_dir).expect("create d4-bin dir");

    let vendor_marker = fs::read_to_string(&vendor_commit).expect("read VENDORED_COMMIT");
    let needs_copy = match fs::read_to_string(src_dir.join("VENDORED_COMMIT")) {
        Ok(existing) => existing != vendor_marker,
        Err(_) => true,
    };
    if needs_copy {
        let _ = fs::remove_dir_all(&src_dir);
        copy_dir_recursive(&vendor_dir, &src_dir).expect("copy vendored D4 source tree");
    }

    let jobs: usize = env::var("NUM_JOBS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(1);

    let cflags = format!("-I{}", boost_dir.display());
    let lflags = "-lz patoh/libpatoh.a";

    let status = Command::new("make")
        .current_dir(&src_dir)
        .arg(format!("-j{}", jobs))
        .arg("r")
        .arg("UNAME=XLOG")
        .arg(format!("CFLAGS={}", cflags))
        .arg(format!("LFLAGS={}", lflags))
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn make for D4 build: {}", e));
    if !status.success() {
        panic!("vendored D4 build failed (exit={})", status);
    }

    let built = src_dir.join("d4_release");
    if !built.is_file() {
        panic!(
            "vendored D4 build succeeded but binary missing at {}",
            built.display()
        );
    }

    fs::copy(&built, &bin_path).expect("copy d4_release into OUT_DIR");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&bin_path)
            .expect("stat bundled d4")
            .permissions();
        perms.set_mode(perms.mode() | 0o111);
        fs::set_permissions(&bin_path, perms).expect("chmod bundled d4");
    }

    println!("cargo:rustc-env=XLOG_PROB_D4_PATH={}", bin_path.display());
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());

        if file_type.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else if file_type.is_file() {
            fs::copy(&src_path, &dst_path)?;
        } else {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "unsupported file type while copying D4: {}",
                    src_path.display()
                ),
            ));
        }
    }
    Ok(())
}
