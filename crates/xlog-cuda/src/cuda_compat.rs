use std::ffi::c_void;

use cudarc::driver::{self, SyncOnDrop};

pub use cudarc::driver::{
    sys, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr, DevicePtrMut, DeviceRepr,
    DeviceSlice, DriverError, LaunchConfig, ValidAsZeroBits,
};

pub use crate::device::CudaFunction;

mod sealed {
    pub trait KernelScalarSealed {}

    macro_rules! impl_kernel_scalar_sealed {
        ($($ty:ty),* $(,)?) => {
            $(impl KernelScalarSealed for $ty {})*
        };
    }

    impl_kernel_scalar_sealed!(
        bool, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64
    );
}

/// Stable host-side storage for a kernel argument pointer.
pub trait KernelParamStorage {
    fn as_kernel_param(&self) -> *mut c_void;
}

#[derive(Debug)]
pub struct ScalarParamStorage<T>(T);

impl<T> KernelParamStorage for ScalarParamStorage<T> {
    fn as_kernel_param(&self) -> *mut c_void {
        (&self.0 as *const T).cast_mut().cast()
    }
}

#[derive(Debug)]
pub struct DeviceParamStorage<'a> {
    ptr: driver::sys::CUdeviceptr,
    _sync: Option<SyncOnDrop<'a>>,
}

impl<'a> DeviceParamStorage<'a> {
    pub fn synced(ptr: driver::sys::CUdeviceptr, sync: SyncOnDrop<'a>) -> Self {
        Self {
            ptr,
            _sync: Some(sync),
        }
    }

    pub fn unsynced(ptr: driver::sys::CUdeviceptr) -> Self {
        Self { ptr, _sync: None }
    }
}

impl KernelParamStorage for DeviceParamStorage<'_> {
    fn as_kernel_param(&self) -> *mut c_void {
        (&self.ptr as *const driver::sys::CUdeviceptr)
            .cast_mut()
            .cast()
    }
}

/// Backwards-compatible `as_kernel_param()` helper for manual raw launch lists.
pub trait AsKernelParam {
    fn as_kernel_param(&self) -> *mut c_void;
}

/// Convert a launch argument into storage that lives until `cuLaunchKernel` runs.
pub trait IntoKernelParamStorage {
    type Storage: KernelParamStorage;

    fn into_kernel_param_storage(self) -> Self::Storage;
}

/// Scalar kernel parameters that can be copied directly into launch storage.
pub trait KernelScalar:
    sealed::KernelScalarSealed
    + cudarc::driver::DeviceRepr
    + Copy
    + 'static
    + AsKernelParam
    + IntoKernelParamStorage
{
}

macro_rules! impl_kernel_scalar {
    ($($ty:ty),* $(,)?) => {
        $(
            impl KernelScalar for $ty {}

            impl AsKernelParam for $ty {
                fn as_kernel_param(&self) -> *mut c_void {
                    (self as *const $ty).cast_mut().cast()
                }
            }

            impl AsKernelParam for &$ty {
                fn as_kernel_param(&self) -> *mut c_void {
                    (*self as *const $ty).cast_mut().cast()
                }
            }

            impl AsKernelParam for &mut $ty {
                fn as_kernel_param(&self) -> *mut c_void {
                    (*self as *const $ty).cast_mut().cast()
                }
            }

            impl IntoKernelParamStorage for $ty {
                type Storage = ScalarParamStorage<$ty>;

                fn into_kernel_param_storage(self) -> Self::Storage {
                    ScalarParamStorage(self)
                }
            }
        )*
    };
}

impl_kernel_scalar!(bool, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64);

impl<'a, T> IntoKernelParamStorage for &'a CudaSlice<T> {
    type Storage = DeviceParamStorage<'a>;

    fn into_kernel_param_storage(self) -> Self::Storage {
        let stream = self.stream();
        let (ptr, sync) = cudarc::driver::DevicePtr::device_ptr(self, stream);
        DeviceParamStorage::synced(ptr, sync)
    }
}

impl<'a, T> IntoKernelParamStorage for &'a mut CudaSlice<T> {
    type Storage = DeviceParamStorage<'static>;

    fn into_kernel_param_storage(self) -> Self::Storage {
        let stream = self.stream().clone();
        let (ptr, sync) = cudarc::driver::DevicePtrMut::device_ptr_mut(self, &stream);
        std::mem::forget(sync);
        DeviceParamStorage::unsynced(ptr)
    }
}

impl<'a, 'b, T> IntoKernelParamStorage for &'a CudaView<'b, T> {
    type Storage = DeviceParamStorage<'a>;

    fn into_kernel_param_storage(self) -> Self::Storage {
        let stream = self.stream();
        let (ptr, sync) = cudarc::driver::DevicePtr::device_ptr(self, stream);
        DeviceParamStorage::synced(ptr, sync)
    }
}

impl<'a, 'b, T> IntoKernelParamStorage for &'a CudaViewMut<'b, T> {
    type Storage = DeviceParamStorage<'a>;

    fn into_kernel_param_storage(self) -> Self::Storage {
        let stream = self.stream();
        let (ptr, sync) = cudarc::driver::DevicePtr::device_ptr(self, stream);
        DeviceParamStorage::synced(ptr, sync)
    }
}

impl<'a, 'b, T> IntoKernelParamStorage for &'a mut CudaViewMut<'b, T> {
    type Storage = DeviceParamStorage<'static>;

    fn into_kernel_param_storage(self) -> Self::Storage {
        let stream = self.stream().clone();
        let (ptr, sync) = cudarc::driver::DevicePtrMut::device_ptr_mut(self, &stream);
        std::mem::forget(sync);
        DeviceParamStorage::unsynced(ptr)
    }
}

/// Old cudarc-style launch trait reimplemented on top of CUDA 13-compatible
/// raw kernel launches.
pub unsafe trait LaunchAsync<Params> {
    unsafe fn launch(self, cfg: LaunchConfig, params: Params) -> Result<(), DriverError>;

    unsafe fn launch_on_stream(
        self,
        stream: &CudaStream,
        cfg: LaunchConfig,
        params: Params,
    ) -> Result<(), DriverError>;

    unsafe fn launch_cooperative(
        self,
        cfg: LaunchConfig,
        params: Params,
    ) -> Result<(), DriverError>;
}

unsafe impl LaunchAsync<&mut [*mut c_void]> for CudaFunction {
    unsafe fn launch(
        self,
        cfg: LaunchConfig,
        params: &mut [*mut c_void],
    ) -> Result<(), DriverError> {
        self.launch_raw(cfg, params)
    }

    unsafe fn launch_on_stream(
        self,
        stream: &CudaStream,
        cfg: LaunchConfig,
        params: &mut [*mut c_void],
    ) -> Result<(), DriverError> {
        self.launch_raw_on_stream(stream, cfg, params)
    }

    unsafe fn launch_cooperative(
        self,
        cfg: LaunchConfig,
        params: &mut [*mut c_void],
    ) -> Result<(), DriverError> {
        self.launch_raw_cooperative(cfg, params)
    }
}

unsafe impl LaunchAsync<&mut Vec<*mut c_void>> for CudaFunction {
    unsafe fn launch(
        self,
        cfg: LaunchConfig,
        params: &mut Vec<*mut c_void>,
    ) -> Result<(), DriverError> {
        self.launch_raw(cfg, params)
    }

    unsafe fn launch_on_stream(
        self,
        stream: &CudaStream,
        cfg: LaunchConfig,
        params: &mut Vec<*mut c_void>,
    ) -> Result<(), DriverError> {
        self.launch_raw_on_stream(stream, cfg, params)
    }

    unsafe fn launch_cooperative(
        self,
        cfg: LaunchConfig,
        params: &mut Vec<*mut c_void>,
    ) -> Result<(), DriverError> {
        self.launch_raw_cooperative(cfg, params)
    }
}

macro_rules! impl_launch_tuple {
    ([$($var:ident),*], [$($idx:tt),*]) => {
        #[allow(non_snake_case)]
        unsafe impl<$($var: IntoKernelParamStorage),*> LaunchAsync<($($var,)*)> for CudaFunction {
            unsafe fn launch(
                self,
                cfg: LaunchConfig,
                params: ($($var,)*),
            ) -> Result<(), DriverError> {
                let ($($var,)*) = params;
                $(let $var = $var.into_kernel_param_storage();)*
                let mut raw = [$( $var.as_kernel_param(), )*];
                self.launch_raw(cfg, &mut raw)
            }

            unsafe fn launch_on_stream(
                self,
                stream: &CudaStream,
                cfg: LaunchConfig,
                params: ($($var,)*),
            ) -> Result<(), DriverError> {
                let ($($var,)*) = params;
                $(let $var = $var.into_kernel_param_storage();)*
                let mut raw = [$( $var.as_kernel_param(), )*];
                self.launch_raw_on_stream(stream, cfg, &mut raw)
            }

            unsafe fn launch_cooperative(
                self,
                cfg: LaunchConfig,
                params: ($($var,)*),
            ) -> Result<(), DriverError> {
                let ($($var,)*) = params;
                $(let $var = $var.into_kernel_param_storage();)*
                let mut raw = [$( $var.as_kernel_param(), )*];
                self.launch_raw_cooperative(cfg, &mut raw)
            }
        }
    };
}

impl_launch_tuple!([A], [0]);
impl_launch_tuple!([A, B], [0, 1]);
impl_launch_tuple!([A, B, C], [0, 1, 2]);
impl_launch_tuple!([A, B, C, D], [0, 1, 2, 3]);
impl_launch_tuple!([A, B, C, D, E], [0, 1, 2, 3, 4]);
impl_launch_tuple!([A, B, C, D, E, F], [0, 1, 2, 3, 4, 5]);
impl_launch_tuple!([A, B, C, D, E, F, G], [0, 1, 2, 3, 4, 5, 6]);
impl_launch_tuple!([A, B, C, D, E, F, G, H], [0, 1, 2, 3, 4, 5, 6, 7]);
impl_launch_tuple!([A, B, C, D, E, F, G, H, I], [0, 1, 2, 3, 4, 5, 6, 7, 8]);
impl_launch_tuple!(
    [A, B, C, D, E, F, G, H, I, J],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
);
impl_launch_tuple!(
    [A, B, C, D, E, F, G, H, I, J, K],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
);
impl_launch_tuple!(
    [A, B, C, D, E, F, G, H, I, J, K, L],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
);
