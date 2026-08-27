use std::env;
use std::path::{Path, PathBuf};

/// Locates staged CUDA kernel artifacts in build, package, or override dirs.
#[derive(Debug, Clone, Default)]
pub struct KernelArtifactLocator {
    cubin_dir: Option<PathBuf>,
    package_kernels_dir: Option<PathBuf>,
    additional_kernel_dir: Option<PathBuf>,
}

impl KernelArtifactLocator {
    pub fn new(
        cubin_dir: Option<PathBuf>,
        package_kernels_dir: Option<PathBuf>,
        additional_kernel_dir: Option<PathBuf>,
    ) -> Self {
        Self {
            cubin_dir,
            package_kernels_dir,
            additional_kernel_dir,
        }
    }

    /// Build a locator from the current process environment.
    ///
    /// Precedence matches runtime loading:
    /// 1. `XLOG_CUBIN_DIR`
    /// 2. binary-adjacent `kernels/`
    ///
    /// Cargo's build output directory is intentionally excluded because embedding
    /// its absolute path makes otherwise identical release artifacts differ by
    /// build location. Source-tree consumers can use `XLOG_CUBIN_DIR`; packaged
    /// consumers discover their staged kernels through the package bootstrap.
    pub fn from_env() -> Self {
        let cubin_dir = env::var_os("XLOG_CUBIN_DIR").map(PathBuf::from);
        let package_kernels_dir = env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|dir| dir.join("kernels")));
        Self::new(cubin_dir, package_kernels_dir, None)
    }

    /// Resolve a module to a staged cubin or portable PTX.
    pub fn resolve_module_path(&self, name: &str, cc: u32) -> Option<(PathBuf, bool)> {
        self.resolve_module_paths(name, cc).into_iter().next()
    }

    /// Resolve all viable module artifacts in runtime precedence order.
    ///
    /// A matching cubin remains the first choice, but portable PTX is retained
    /// as a fallback when a target driver rejects a cubin image.
    pub fn resolve_module_paths(&self, name: &str, cc: u32) -> Vec<(PathBuf, bool)> {
        let cubin_name = format!("{name}.sm_{cc}.cubin");
        let ptx_name = format!("{name}.portable.ptx");
        let mut found_paths = Vec::new();

        for dir in [
            self.cubin_dir.as_ref(),
            self.package_kernels_dir.as_ref(),
            self.additional_kernel_dir.as_ref(),
        ] {
            let dir = match dir {
                Some(dir) => dir,
                None => continue,
            };

            if let Some(found) = Self::resolve_in_dir(dir, &cubin_name, true) {
                found_paths.push(found);
            }
            if let Some(found) = Self::resolve_in_dir(dir, &ptx_name, false) {
                found_paths.push(found);
            }
        }

        found_paths
    }

    fn resolve_in_dir(dir: &Path, file_name: &str, is_cubin: bool) -> Option<(PathBuf, bool)> {
        let path = dir.join(file_name);
        if path.exists() {
            Some((path, is_cubin))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::KernelArtifactLocator;
    use std::fs;

    #[test]
    fn runtime_locator_ignores_ephemeral_build_output_directory() {
        let module_name = format!("runtime_locator_ephemeral_output_{}", std::process::id());
        let generated_ptx =
            std::path::PathBuf::from(env!("OUT_DIR")).join(format!("{module_name}.portable.ptx"));
        fs::write(&generated_ptx, ".version 8.0\n").expect("write generated PTX sentinel");

        let locator = KernelArtifactLocator::from_env();
        let resolved = locator.resolve_module_path(&module_name, 999);
        fs::remove_file(&generated_ptx).expect("remove generated PTX sentinel");

        assert!(
            resolved.is_none(),
            "runtime kernel lookup must ignore Cargo's ephemeral OUT_DIR"
        );
    }

    #[test]
    fn resolves_in_precedence_order() {
        let root = std::env::temp_dir().join(format!(
            "xlog-kernel-paths-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before UNIX_EPOCH")
                .as_nanos()
        ));
        let cubin_dir = root.join("cubin");
        let package_dir = root.join("bin").join("kernels");
        let out_dir = root.join("out");
        fs::create_dir_all(&cubin_dir).expect("create cubin dir");
        fs::create_dir_all(&package_dir).expect("create package kernels dir");
        fs::create_dir_all(&out_dir).expect("create out dir");

        let name = "xlog_join";
        let cc = 75;
        let cubin_path = cubin_dir.join(format!("{name}.sm_{cc}.cubin"));
        let package_path = package_dir.join(format!("{name}.sm_{cc}.cubin"));
        let out_path = out_dir.join(format!("{name}.sm_{cc}.cubin"));
        fs::write(&cubin_path, b"cubin").expect("write cubin file");
        fs::write(&package_path, b"package").expect("write package file");
        fs::write(&out_path, b"out").expect("write out file");

        let locator = KernelArtifactLocator::new(
            Some(cubin_dir.clone()),
            Some(package_dir.clone()),
            Some(out_dir.clone()),
        );

        let (path, is_cubin) = locator
            .resolve_module_path(name, cc)
            .expect("expected a kernel artifact");
        assert_eq!(path, cubin_path);
        assert!(is_cubin);

        fs::remove_file(&cubin_path).expect("remove cubin file");
        let (path, is_cubin) = locator
            .resolve_module_path(name, cc)
            .expect("expected package kernel artifact");
        assert_eq!(path, package_path);
        assert!(is_cubin);

        fs::remove_file(&package_path).expect("remove package file");
        let (path, is_cubin) = locator
            .resolve_module_path(name, cc)
            .expect("expected out dir kernel artifact");
        assert_eq!(path, out_path);
        assert!(is_cubin);

        let _ = fs::remove_dir_all(&root);
    }
}
