//! CUDA device management
//!
//! This module keeps XLOG's historical single-stream device abstraction while
//! targeting cudarc's newer CUDA 13-capable context/stream APIs.

use std::collections::BTreeMap;
use std::ffi::{c_void, CString};
use std::path::Path;
use std::sync::{Arc, RwLock};

use cudarc::driver::result::{self, DriverError};
use cudarc::driver::{
    sys, CudaContext as CudarcContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr,
    HostSlice, LaunchConfig, ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;
use xlog_core::{Result, XlogError};

#[derive(Debug)]
struct LoadedModule {
    cu_module: sys::CUmodule,
    functions: BTreeMap<String, sys::CUfunction>,
}

unsafe impl Send for LoadedModule {}
unsafe impl Sync for LoadedModule {}

/// Kernel handle bound to XLOG's default CUDA stream.
#[derive(Debug, Clone)]
pub struct CudaFunction {
    cu_function: sys::CUfunction,
    context: Arc<CudarcContext>,
    stream: Arc<CudaStream>,
}

impl CudaFunction {
    pub(crate) unsafe fn launch_raw(
        &self,
        cfg: LaunchConfig,
        params: &mut [*mut c_void],
    ) -> std::result::Result<(), DriverError> {
        self.context.bind_to_thread()?;
        result::launch_kernel(
            self.cu_function,
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            self.stream.cu_stream(),
            params,
        )
    }

    pub(crate) unsafe fn launch_raw_on_stream(
        &self,
        stream: &CudaStream,
        cfg: LaunchConfig,
        params: &mut [*mut c_void],
    ) -> std::result::Result<(), DriverError> {
        self.context.bind_to_thread()?;
        result::launch_kernel(
            self.cu_function,
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            stream.cu_stream(),
            params,
        )
    }

    pub(crate) unsafe fn launch_raw_cooperative(
        &self,
        cfg: LaunchConfig,
        params: &mut [*mut c_void],
    ) -> std::result::Result<(), DriverError> {
        self.context.bind_to_thread()?;
        result::launch_cooperative_kernel(
            self.cu_function,
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            self.stream.cu_stream(),
            params,
        )
    }

    pub fn occupancy_available_dynamic_smem_per_block(
        &self,
        num_blocks: u32,
        block_size: u32,
    ) -> std::result::Result<usize, DriverError> {
        let mut dynamic_smem_size: usize = 0;
        unsafe {
            sys::cuOccupancyAvailableDynamicSMemPerBlock(
                &mut dynamic_smem_size,
                self.cu_function,
                num_blocks as std::ffi::c_int,
                block_size as std::ffi::c_int,
            )
            .result()?
        };
        Ok(dynamic_smem_size)
    }

    pub fn occupancy_max_active_blocks_per_multiprocessor(
        &self,
        block_size: u32,
        dynamic_smem_size: usize,
        flags: Option<sys::CUoccupancy_flags_enum>,
    ) -> std::result::Result<u32, DriverError> {
        let mut num_blocks: std::ffi::c_int = 0;
        let flags = flags.unwrap_or(sys::CUoccupancy_flags_enum::CU_OCCUPANCY_DEFAULT);
        unsafe {
            sys::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                &mut num_blocks,
                self.cu_function,
                block_size as std::ffi::c_int,
                dynamic_smem_size,
                flags as std::ffi::c_uint,
            )
            .result()?
        };
        Ok(num_blocks as u32)
    }

    pub fn occupancy_max_active_clusters(
        &self,
        config: LaunchConfig,
    ) -> std::result::Result<u32, DriverError> {
        let mut num_clusters: std::ffi::c_int = 0;
        let cfg = sys::CUlaunchConfig {
            gridDimX: config.grid_dim.0,
            gridDimY: config.grid_dim.1,
            gridDimZ: config.grid_dim.2,
            blockDimX: config.block_dim.0,
            blockDimY: config.block_dim.1,
            blockDimZ: config.block_dim.2,
            sharedMemBytes: config.shared_mem_bytes,
            hStream: self.stream.cu_stream(),
            attrs: std::ptr::null_mut(),
            numAttrs: 0,
        };
        unsafe {
            sys::cuOccupancyMaxActiveClusters(&mut num_clusters, self.cu_function, &cfg).result()?
        };
        Ok(num_clusters as u32)
    }

    pub fn occupancy_max_potential_block_size(
        &self,
        block_size_to_dynamic_smem_size: extern "C" fn(block_size: std::ffi::c_int) -> usize,
        dynamic_smem_size: usize,
        block_size_limit: u32,
        flags: Option<sys::CUoccupancy_flags_enum>,
    ) -> std::result::Result<(u32, u32), DriverError> {
        let mut min_grid_size: std::ffi::c_int = 0;
        let mut block_size: std::ffi::c_int = 0;
        let flags = flags.unwrap_or(sys::CUoccupancy_flags_enum::CU_OCCUPANCY_DEFAULT);
        unsafe {
            sys::cuOccupancyMaxPotentialBlockSizeWithFlags(
                &mut min_grid_size,
                &mut block_size,
                self.cu_function,
                Some(block_size_to_dynamic_smem_size),
                dynamic_smem_size,
                block_size_limit as std::ffi::c_int,
                flags as std::ffi::c_uint,
            )
            .result()?
        };
        Ok((min_grid_size as u32, block_size as u32))
    }

    pub fn occupancy_max_potential_cluster_size(
        &self,
        config: LaunchConfig,
    ) -> std::result::Result<u32, DriverError> {
        let mut cluster_size: std::ffi::c_int = 0;
        let cfg = sys::CUlaunchConfig {
            gridDimX: config.grid_dim.0,
            gridDimY: config.grid_dim.1,
            gridDimZ: config.grid_dim.2,
            blockDimX: config.block_dim.0,
            blockDimY: config.block_dim.1,
            blockDimZ: config.block_dim.2,
            sharedMemBytes: config.shared_mem_bytes,
            hStream: self.stream.cu_stream(),
            attrs: std::ptr::null_mut(),
            numAttrs: 0,
        };
        unsafe {
            sys::cuOccupancyMaxPotentialClusterSize(&mut cluster_size, self.cu_function, &cfg)
                .result()?
        };
        Ok(cluster_size as u32)
    }

    pub fn get_attribute(
        &self,
        attribute: sys::CUfunction_attribute_enum,
    ) -> std::result::Result<i32, DriverError> {
        self.context.bind_to_thread()?;
        unsafe { result::function::get_function_attribute(self.cu_function, attribute) }
    }

    pub fn num_regs(&self) -> std::result::Result<i32, DriverError> {
        self.get_attribute(sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NUM_REGS)
    }

    pub fn shared_size_bytes(&self) -> std::result::Result<i32, DriverError> {
        self.get_attribute(sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
    }

    pub fn const_size_bytes(&self) -> std::result::Result<i32, DriverError> {
        self.get_attribute(sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES)
    }

    pub fn local_size_bytes(&self) -> std::result::Result<i32, DriverError> {
        self.get_attribute(sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)
    }

    pub fn max_threads_per_block(&self) -> std::result::Result<i32, DriverError> {
        self.get_attribute(sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    }

    pub fn ptx_version(&self) -> std::result::Result<i32, DriverError> {
        self.get_attribute(sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PTX_VERSION)
    }

    pub fn binary_version(&self) -> std::result::Result<i32, DriverError> {
        self.get_attribute(sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_BINARY_VERSION)
    }

    pub fn set_attribute(
        &self,
        attribute: sys::CUfunction_attribute_enum,
        value: i32,
    ) -> std::result::Result<(), DriverError> {
        unsafe { result::function::set_function_attribute(self.cu_function, attribute, value) }
    }

    pub fn set_function_cache_config(
        &self,
        config: sys::CUfunc_cache,
    ) -> std::result::Result<(), DriverError> {
        unsafe { result::function::set_function_cache_config(self.cu_function, config) }
    }
}

#[derive(Debug)]
pub struct CudaDeviceInner {
    context: Arc<CudarcContext>,
    stream: Arc<CudaStream>,
    modules: RwLock<BTreeMap<String, LoadedModule>>,
}

impl Drop for CudaDeviceInner {
    fn drop(&mut self) {
        let _ = self.context.bind_to_thread();
        if let Ok(modules) = self.modules.get_mut() {
            for module in modules.values() {
                let _ = unsafe { result::module::unload(module.cu_module) };
            }
            modules.clear();
        }
    }
}

impl CudaDeviceInner {
    fn insert_module(
        &self,
        module_name: &str,
        cu_module: sys::CUmodule,
        kernels: &[&str],
    ) -> std::result::Result<(), DriverError> {
        let mut functions = BTreeMap::new();
        for &kernel in kernels {
            let name_c = CString::new(kernel).unwrap();
            let cu_function = unsafe { result::module::get_function(cu_module, name_c) }?;
            functions.insert(kernel.to_string(), cu_function);
        }
        let module = LoadedModule {
            cu_module,
            functions,
        };

        let mut modules = self.modules.write().unwrap();
        if let Some(prev) = modules.insert(module_name.to_string(), module) {
            unsafe { result::module::unload(prev.cu_module) }?;
        }
        Ok(())
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn has_func(&self, module_name: &str, func_name: &str) -> bool {
        let modules = self.modules.read().unwrap();
        modules
            .get(module_name)
            .is_some_and(|module| module.functions.contains_key(func_name))
    }

    pub fn get_func(&self, module_name: &str, func_name: &str) -> Option<CudaFunction> {
        let modules = self.modules.read().unwrap();
        let cu_function = modules
            .get(module_name)
            .and_then(|module| module.functions.get(func_name))
            .copied()?;
        Some(CudaFunction {
            cu_function,
            context: self.context.clone(),
            stream: self.stream.clone(),
        })
    }

    pub fn load_file(
        &self,
        path: &Path,
        module_name: &str,
        kernels: &[&str],
    ) -> std::result::Result<(), DriverError> {
        self.context.bind_to_thread()?;
        let name_c = CString::new(path.to_string_lossy().as_bytes()).unwrap();
        let cu_module = result::module::load(name_c)?;
        self.insert_module(module_name, cu_module, kernels)
    }

    pub fn load_ptx(
        &self,
        ptx: Ptx,
        module_name: &str,
        kernels: &[&str],
    ) -> std::result::Result<(), DriverError> {
        self.context.bind_to_thread()?;
        let cu_module = if let Some(bytes) = ptx.as_bytes() {
            unsafe { result::module::load_data(bytes.as_ptr() as *const _) }?
        } else {
            let src = CString::new(ptx.to_src()).unwrap();
            unsafe { result::module::load_data(src.as_ptr() as *const _) }?
        };
        self.insert_module(module_name, cu_module, kernels)
    }

    pub unsafe fn alloc<T: DeviceRepr>(
        &self,
        len: usize,
    ) -> std::result::Result<CudaSlice<T>, DriverError> {
        self.stream.alloc(len)
    }

    pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> std::result::Result<CudaSlice<T>, DriverError> {
        self.stream.alloc_zeros(len)
    }

    pub fn memset_zeros<T: DeviceRepr + ValidAsZeroBits, Dst: DevicePtrMut<T>>(
        &self,
        dst: &mut Dst,
    ) -> std::result::Result<(), DriverError> {
        self.stream.memset_zeros(dst)?;
        self.stream.synchronize()
    }

    pub fn htod_sync_copy_into<T: DeviceRepr, Dst: DevicePtrMut<T>, Src: HostSlice<T> + ?Sized>(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> std::result::Result<(), DriverError> {
        self.stream.memcpy_htod(src, dst)?;
        self.stream.synchronize()
    }

    pub fn dtoh_sync_copy_into<T: DeviceRepr, Src: DevicePtr<T>, Dst: HostSlice<T> + ?Sized>(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> std::result::Result<(), DriverError> {
        self.stream.memcpy_dtoh(src, dst)?;
        self.stream.synchronize()
    }

    pub fn htod_sync_copy<T: DeviceRepr, Src: HostSlice<T> + ?Sized>(
        &self,
        src: &Src,
    ) -> std::result::Result<CudaSlice<T>, DriverError> {
        let dst = self.stream.clone_htod(src)?;
        self.stream.synchronize()?;
        Ok(dst)
    }

    pub fn dtoh_sync_copy<T: DeviceRepr, Src: DevicePtr<T>>(
        &self,
        src: &Src,
    ) -> std::result::Result<Vec<T>, DriverError> {
        let dst = self.stream.clone_dtoh(src)?;
        self.stream.synchronize()?;
        Ok(dst)
    }

    pub fn dtod_copy<T, Src: DevicePtr<T>, Dst: DevicePtrMut<T>>(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> std::result::Result<(), DriverError> {
        self.stream.memcpy_dtod(src, dst)?;
        self.stream.synchronize()
    }

    pub unsafe fn upgrade_device_ptr<T>(
        &self,
        cu_device_ptr: sys::CUdeviceptr,
        len: usize,
    ) -> CudaSlice<T> {
        self.stream.upgrade_device_ptr(cu_device_ptr, len)
    }

    pub fn attribute(
        &self,
        attrib: sys::CUdevice_attribute,
    ) -> std::result::Result<i32, DriverError> {
        self.context.attribute(attrib)
    }

    pub fn synchronize(&self) -> std::result::Result<(), DriverError> {
        self.stream.synchronize()
    }

    pub fn ordinal(&self) -> usize {
        self.context.ordinal()
    }
}

/// CUDA device wrapper for GPU operations.
///
/// This keeps XLOG's historical "device with a built-in default stream" API,
/// but is backed by cudarc's newer `CudaContext` and `CudaStream`.
pub struct CudaDevice {
    device: Arc<CudaDeviceInner>,
}

impl CudaDevice {
    /// Create a new CUDA device on the specified GPU ordinal.
    pub fn new(ordinal: usize) -> Result<Self> {
        let context = std::panic::catch_unwind(|| CudarcContext::new(ordinal))
            .map_err(|_| {
                XlogError::Kernel(format!(
                    "Failed to create CUDA device {}: cudarc panicked during driver initialization",
                    ordinal
                ))
            })?
            .map_err(|e| {
                XlogError::Kernel(format!("Failed to create CUDA device {}: {}", ordinal, e))
            })?;

        let stream = context.default_stream();
        Ok(Self {
            device: Arc::new(CudaDeviceInner {
                context,
                stream,
                modules: RwLock::new(BTreeMap::new()),
            }),
        })
    }

    pub fn count() -> Result<i32> {
        std::panic::catch_unwind(|| {
            result::init()?;
            result::device::get_count()
        })
        .map_err(|_| {
            XlogError::Kernel(
                "Failed to count CUDA devices: cudarc panicked during driver initialization"
                    .to_string(),
            )
        })?
        .map_err(|e| XlogError::Kernel(format!("Failed to count CUDA devices: {}", e)))
    }

    pub fn synchronize(&self) -> Result<()> {
        self.device
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("Failed to synchronize device: {}", e)))
    }

    pub fn inner(&self) -> &Arc<CudaDeviceInner> {
        &self.device
    }

    pub fn ordinal(&self) -> usize {
        self.device.ordinal()
    }
}

// Compile-time assertion: CudaDevice must be Send so pyxlog can use py.allow_threads().
const _: () = {
    fn _assert_send<T: Send>() {}
    fn _check() {
        _assert_send::<CudaDevice>();
    }
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_creation() {
        let device = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
                return;
            }
        };
        drop(device);
    }

    #[test]
    fn test_device_synchronize() {
        let device = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
                return;
            }
        };
        let result = device.synchronize();
        assert!(result.is_ok(), "Failed to synchronize: {:?}", result.err());
    }

    #[test]
    fn test_device_ordinal() {
        let device = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
                return;
            }
        };
        assert_eq!(device.ordinal(), 0);
    }

    #[test]
    fn test_device_inner_access() {
        let device = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
                return;
            }
        };
        let inner = device.inner();
        assert_eq!(inner.ordinal(), 0);
    }

    #[test]
    fn test_invalid_device_ordinal() {
        let result = CudaDevice::new(9999);
        assert!(result.is_err(), "Should fail with invalid ordinal");

        if let Err(XlogError::Kernel(msg)) = result {
            assert!(msg.contains("9999"), "Error should mention device ordinal");
        } else {
            panic!("Expected XlogError::Kernel");
        }
    }
}
