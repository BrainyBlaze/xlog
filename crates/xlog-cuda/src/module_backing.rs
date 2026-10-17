//! Cold observations made by the original loaded CUDA module owner.
//!
//! These records preserve driver query facts. Mapping geometry and a physical
//! block identifier do not establish the full size of that physical allocation.

use std::collections::BTreeSet;
use std::ffi::CString;
use std::sync::Arc;

use cudarc::driver::{sys, CudaContext, DriverError};

use crate::device_runtime::{ResourceError, ResourceResult};
use crate::memory::{
    begin_gpu_backing_observation, retain_gpu_module_observation_failure, GpuBackingObserver,
};

pub const MODULE_LOADED: u32 = 1;
pub const MODULE_READ_STATIC: u32 = 2;
pub const MODULE_CONFIRMED: u32 = 3;
pub const MODULE_UNLOADED: u32 = 4;
pub const MODULE_UNKNOWN: u32 = 5;
pub const MODULE_DISCOVER_STATIC: u32 = 6;
pub const MODULE_INPUT_CUBIN: u32 = 1;
pub const MODULE_INPUT_PTX: u32 = 2;
pub const MODULE_INPUT_FILE: u32 = 3;
pub const MODULE_INPUT_DIGEST_VALID: u64 = 1;
pub const MODULE_ORIGINAL_LOAD_VALID: u64 = 1;
pub const MODULE_ORIGINAL_DELIVERY_COMPLETE: u64 = 2;
pub const MODULE_ORIGINAL_CUBIN_DIGEST_VALID: u64 = 4;
pub const MODULE_ORIGINAL_IMAGE_VALID: u64 = 8;
pub const MODULE_ORIGINAL_LIBRARY_HANDLE_VALID: u64 = 16;
pub const MODULE_ORIGINAL_UNLOAD_OBSERVED: u64 = 32;
pub const MODULE_ORIGINAL_RESET_OBSERVED: u64 = 64;
pub const STATIC_NAME_VALID: u64 = 1;
pub const STATIC_SOURCE_VALID: u64 = 2;
pub const STATIC_CONTEXT_VALID: u64 = 4;
pub const STATIC_CORRELATION_VALID: u64 = 8;

const QUERY_GLOBAL: usize = 0;
const QUERY_ALLOCATION: usize = 1;
const QUERY_CONTEXT: usize = 2;
const QUERY_DEVICE: usize = 3;
const QUERY_MEMORY_TYPE: usize = 4;
const QUERY_MAPPED: usize = 5;
const QUERY_MAPPING_BASE: usize = 6;
const QUERY_MAPPING_SIZE: usize = 7;
const QUERY_PHYSICAL_BLOCK: usize = 8;
const QUERY_BUFFER_ID: usize = 9;
const QUERY_COUNT: usize = 10;
const COMPLETE: u32 = 0;
const UNKNOWN: u32 = 2;
// CUpti_ActivityMemoryKind and CUpti_ActivityMemoryOperationType wire values.
const DEVICE_STATIC: u32 = 6;
const ALLOCATION: u32 = 1;

/// Native half of the installed observer's module lifecycle protocol.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ModuleObservation {
    pub struct_size: u32,
    pub operation: u32,
    pub owner_generation: u64,
    pub context_handle: u64,
    pub module_handle: u64,
    pub device_uuid: [u8; 16],
    pub device_ordinal: u32,
    pub input_kind: u32,
    pub flags: u64,
    pub input_sha256: [u8; 32],
    pub lease_id: u64,
    pub confirmations: *const StaticConfirmation,
    pub confirmation_count: u64,
    pub confirmation_size: u64,
}

/// Immutable original activity storage, pinned by the process collector lease.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ModuleOriginalView {
    pub struct_size: u32,
    pub state: u32,
    pub lease_id: u64,
    pub owner_generation: u64,
    pub context_handle: u64,
    pub module_handle: u64,
    pub original_load_start: u64,
    pub original_load_end: u64,
    pub original_context_id: u32,
    pub original_correlation_id: u32,
    pub original_module_id: u32,
    pub reserved: u32,
    pub flags: u64,
    pub actual_cubin_sha256: [u8; 32],
    pub records: *const OriginalStaticRecord,
    pub record_count: u64,
    pub record_size: u64,
    pub original_loader_callback_id: u32,
    pub reserved_tail: u32,
    pub library_handle: u64,
    pub original_image_pointer: u64,
    pub original_image_bytes: u64,
    pub functions: *const OriginalModuleFunction,
    pub function_count: u64,
    pub function_size: u64,
}

impl Default for ModuleOriginalView {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            state: UNKNOWN,
            lease_id: 0,
            owner_generation: 0,
            context_handle: 0,
            module_handle: 0,
            original_load_start: 0,
            original_load_end: 0,
            original_context_id: 0,
            original_correlation_id: 0,
            original_module_id: 0,
            reserved: 0,
            flags: 0,
            actual_cubin_sha256: [0; 32],
            records: std::ptr::null(),
            record_count: 0,
            record_size: std::mem::size_of::<OriginalStaticRecord>() as u64,
            original_loader_callback_id: 0,
            reserved_tail: 0,
            library_handle: 0,
            original_image_pointer: 0,
            original_image_bytes: 0,
            functions: std::ptr::null(),
            function_count: 0,
            function_size: std::mem::size_of::<OriginalModuleFunction>() as u64,
        }
    }
}

/// Original typed runtime function argument and nested driver lookup, copied by
/// the same immutable collector lease. These raw facts do not establish CUDA
/// ownership or grant a right to dereference the argument. The native module's
/// actual owner needs no runtime rows and does not dereference this array.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct OriginalModuleFunction {
    pub struct_size: u32,
    pub reserved: u32,
    pub record_id: u64,
    pub valid_fields: u64,
    pub runtime_callback_id: u32,
    pub runtime_correlation_id: u32,
    pub runtime_status: u32,
    pub lookup_callback_id: u32,
    pub lookup_correlation_id: u32,
    pub lookup_status: u32,
    pub runtime_function_argument: u64,
    pub function_handle: u64,
    pub module_handle: u64,
}

/// Exact original static activity fields; absent fields are never synthesized.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct OriginalStaticRecord {
    pub struct_size: u32,
    pub memory_kind: u32,
    pub operation: u32,
    pub device_id: u32,
    pub context_id: u32,
    pub correlation_id: u32,
    pub record_id: u64,
    pub address: u64,
    pub bytes: u64,
    pub timestamp: u64,
    pub valid_fields: u64,
    pub name: *const u8,
    pub name_size: u64,
    pub source: *const u8,
    pub source_size: u64,
}

/// Successful-query bits qualify values; GLOBAL also requires original-row
/// membership. Every attempted CUDA status is retained, including mismatches.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StaticConfirmation {
    pub struct_size: u32,
    pub reserved: u32,
    pub record_id: u64,
    pub valid_queries: u64,
    pub cuda_status: [u32; QUERY_COUNT],
    pub global_address: u64,
    pub global_bytes: u64,
    pub allocation_base: u64,
    pub allocation_bytes: u64,
    pub mapping_base: u64,
    pub mapping_bytes: u64,
    pub physical_block_id: u64,
    pub buffer_id: u64,
    pub context_handle: u64,
    pub device_ordinal: u32,
    pub memory_type: u32,
    pub mapped: u32,
    pub reserved_tail: u32,
}

impl StaticConfirmation {
    fn new(record_id: u64) -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            reserved: 0,
            record_id,
            valid_queries: 0,
            cuda_status: [u32::MAX; QUERY_COUNT],
            global_address: 0,
            global_bytes: 0,
            allocation_base: 0,
            allocation_bytes: 0,
            mapping_base: 0,
            mapping_bytes: 0,
            physical_block_id: 0,
            buffer_id: 0,
            context_handle: 0,
            device_ordinal: 0,
            memory_type: 0,
            mapped: 0,
            reserved_tail: 0,
        }
    }

    fn record_status(&mut self, query: usize, status: sys::CUresult) {
        self.cuda_status[query] = status as u32;
        if status == sys::CUresult::CUDA_SUCCESS {
            self.valid_queries |= 1 << query;
        }
    }

    fn has(&self, query: usize) -> bool {
        self.valid_queries & (1 << query) != 0
    }
}

const _: () = {
    assert!(std::mem::size_of::<ModuleObservation>() == 128);
    assert!(std::mem::size_of::<ModuleOriginalView>() == 192);
    assert!(std::mem::size_of::<OriginalModuleFunction>() == 72);
    assert!(std::mem::size_of::<OriginalStaticRecord>() == 96);
    assert!(std::mem::size_of::<StaticConfirmation>() == 152);
};

/// Part of LoadedModule, never a second CUDA module owner. The original observer
/// and lease remain here even when a typed loader can return only DriverError.
pub(crate) struct ModuleBacking {
    observer: Arc<dyn GpuBackingObserver>,
    observation: ModuleObservation,
    original: ModuleOriginalView,
    confirmations: Vec<StaticConfirmation>,
    first_failure: Option<Arc<ResourceError>>,
    unknown_entered: bool,
}

// SAFETY: the installed collector guarantees immutable lease storage and
// thread-safe recording through acknowledged unload. This capsule moves only
// with its actual module and retained CUDA context, including cold retirement.
unsafe impl Send for ModuleBacking {}

impl std::fmt::Debug for ModuleBacking {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ModuleBacking")
            .field("observation", &self.observation)
            .field("original", &self.original)
            .field("first_failure", &self.first_failure)
            .finish()
    }
}

impl ModuleBacking {
    /// Called before entering the original driver load, while capture is excluded.
    pub(crate) fn prepare(
        context: &Arc<CudaContext>,
        input_kind: u32,
        input_sha256: Option<[u8; 32]>,
    ) -> std::result::Result<Option<Self>, DriverError> {
        let (observer, generation) = begin_gpu_backing_observation().map_err(|error| {
            eprintln!("CUDA module observation admission failed: {error}");
            unknown_driver_error()
        })?;
        let Some(observer) = observer else {
            return Ok(None);
        };
        let mut uuid = sys::CUuuid { bytes: [0; 16] };
        // SAFETY: this is the bound original context before module allocation,
        // outside capture and every collector callback.
        unsafe { sys::cuDeviceGetUuid_v2(&mut uuid, context.cu_device()).result()? };
        let device_ordinal = u32::try_from(context.ordinal())
            .map_err(|_| DriverError(sys::CUresult::CUDA_ERROR_INVALID_DEVICE))?;
        Ok(Some(Self {
            observer,
            observation: ModuleObservation {
                struct_size: std::mem::size_of::<ModuleObservation>() as u32,
                operation: MODULE_LOADED,
                owner_generation: generation,
                context_handle: context.cu_ctx() as u64,
                module_handle: 0,
                device_uuid: uuid.bytes.map(|byte| byte as u8),
                device_ordinal,
                input_kind,
                flags: if input_sha256.is_some() {
                    MODULE_INPUT_DIGEST_VALID
                } else {
                    0
                },
                input_sha256: input_sha256.unwrap_or([0; 32]),
                lease_id: 0,
                confirmations: std::ptr::null(),
                confirmation_count: 0,
                confirmation_size: std::mem::size_of::<StaticConfirmation>() as u64,
            },
            original: ModuleOriginalView::default(),
            confirmations: Vec::new(),
            first_failure: None,
            unknown_entered: false,
        }))
    }

    pub(crate) fn loaded(&mut self, module: sys::CUmodule) -> std::result::Result<(), DriverError> {
        self.observation.module_handle = module as u64;
        self.observe(MODULE_LOADED)
    }

    pub(crate) fn read_static(
        &mut self,
        context: &CudaContext,
    ) -> std::result::Result<(), DriverError> {
        self.observe(MODULE_READ_STATIC)?;
        self.arm_failure("original module lease validation or static queries interrupted");
        if context.cu_ctx() as u64 != self.observation.context_handle {
            return Err(self.retain_failure(invalid("module observation lost its actual context")));
        }
        if let Err(error) = context.bind_to_thread() {
            return Err(self.retain_failure(error.into()));
        }
        if let Err(error) = self.confirm_original_records() {
            return Err(self.retain_failure(error));
        }
        self.first_failure = None;
        self.observe(MODULE_CONFIRMED)
    }

    fn arm_failure(&mut self, message: &str) {
        if self.first_failure.is_none() {
            self.first_failure = Some(Arc::new(ResourceError::Driver(message.into())));
        }
    }

    fn retain_failure(&mut self, error: ResourceError) -> DriverError {
        let cause = Arc::new(error);
        retain_gpu_module_observation_failure(Arc::clone(&cause));
        self.first_failure = Some(Arc::clone(&cause));
        eprintln!("CUDA module backing observation retained its original failure: {cause}");
        unknown_driver_error()
    }

    fn observe(&mut self, operation: u32) -> std::result::Result<(), DriverError> {
        if self.first_failure.is_some() && operation != MODULE_UNLOADED {
            return Err(unknown_driver_error());
        }
        let previous_failure = self.first_failure.is_some();
        self.arm_failure("original module observer acknowledgement interrupted");
        let mut observation = self.observation;
        observation.operation = operation;
        if operation == MODULE_CONFIRMED {
            observation.confirmations = self.confirmations.as_ptr();
            observation.confirmation_count = self.confirmations.len() as u64;
        }
        let result = self.observer.observe_module(
            &observation,
            if operation == MODULE_READ_STATIC {
                Some(&mut self.original)
            } else {
                None
            },
        );
        // Both the returned status and output are meaningful. An uncertain read
        // may already have handed out a lease; keep it with this actual owner.
        if operation == MODULE_READ_STATIC {
            self.observation.lease_id = self.original.lease_id;
        }
        match result {
            Ok(()) => {
                self.first_failure = None;
                Ok(())
            }
            Err(error) => {
                if !previous_failure {
                    return Err(self.retain_failure(error));
                }
                eprintln!("CUDA module unload acknowledgement remains incomplete: {error}");
                Err(unknown_driver_error())
            }
        }
    }

    pub(crate) fn ready(&mut self) -> std::result::Result<(), DriverError> {
        if self.first_failure.is_some() {
            self.unknown(None);
            return Err(unknown_driver_error());
        }
        Ok(())
    }

    /// Called only by retirement's post-success publication, never by Drop entry.
    pub(crate) fn unloaded(&mut self) -> std::result::Result<(), DriverError> {
        self.observe(MODULE_UNLOADED)
    }

    pub(crate) fn unknown(&mut self, error: Option<&DriverError>) {
        if self.first_failure.is_none() {
            self.first_failure = Some(Arc::new(ResourceError::Driver(format!(
                "original module unload is unconfirmed: {error:?}"
            ))));
        }
        if let Some(cause) = &self.first_failure {
            retain_gpu_module_observation_failure(Arc::clone(cause));
        }
        if self.unknown_entered {
            return;
        }
        self.unknown_entered = true;
        let mut observation = self.observation;
        observation.operation = MODULE_UNKNOWN;
        // The actual capsule is already retained by the retirement machinery;
        // even an error or unwind here cannot release its module or first cause.
        if let Err(error) = self.observer.observe_module(&observation, None) {
            eprintln!("CUDA module uncertainty publication incomplete: {error}");
        }
    }

    fn confirm_original_records(&mut self) -> ResourceResult<()> {
        let view = self.original;
        let required = MODULE_ORIGINAL_LOAD_VALID | MODULE_ORIGINAL_DELIVERY_COMPLETE;
        if view.struct_size as usize != std::mem::size_of::<ModuleOriginalView>()
            || view.state != COMPLETE
            || view.reserved != 0
            || view.reserved_tail != 0
            || view.lease_id == 0
            || view.owner_generation != self.observation.owner_generation
            || view.context_handle != self.observation.context_handle
            || view.module_handle != self.observation.module_handle
            || view.flags & required != required
            || view.flags
                & !(MODULE_ORIGINAL_LOAD_VALID
                    | MODULE_ORIGINAL_DELIVERY_COMPLETE
                    | MODULE_ORIGINAL_CUBIN_DIGEST_VALID
                    | MODULE_ORIGINAL_IMAGE_VALID
                    | MODULE_ORIGINAL_LIBRARY_HANDLE_VALID
                    | MODULE_ORIGINAL_UNLOAD_OBSERVED
                    | MODULE_ORIGINAL_RESET_OBSERVED)
                != 0
            || view.flags & (MODULE_ORIGINAL_UNLOAD_OBSERVED | MODULE_ORIGINAL_RESET_OBSERVED) != 0
            || view.original_load_start > view.original_load_end
            || view.record_size != std::mem::size_of::<OriginalStaticRecord>() as u64
            || view.function_size != std::mem::size_of::<OriginalModuleFunction>() as u64
        {
            return Err(invalid(
                "original module lease identity, state or layout is invalid",
            ));
        }
        if self.observation.input_kind == MODULE_INPUT_CUBIN
            && self.observation.flags & MODULE_INPUT_DIGEST_VALID != 0
            && view.flags & MODULE_ORIGINAL_CUBIN_DIGEST_VALID != 0
            && view.actual_cubin_sha256 != self.observation.input_sha256
        {
            return Err(invalid(
                "original loaded cubin differs from its exact cubin input",
            ));
        }
        let count = usize::try_from(view.record_count)
            .map_err(|_| invalid("original static record count exceeds usize"))?;
        let bytes = count
            .checked_mul(std::mem::size_of::<OriginalStaticRecord>())
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .ok_or_else(|| invalid("original static record span overflows"))?;
        if count != 0
            && (view.records.is_null()
                || !(view.records as usize)
                    .is_multiple_of(std::mem::align_of::<OriginalStaticRecord>())
                || (view.records as usize).checked_add(bytes).is_none())
        {
            return Err(invalid("original static record storage is invalid"));
        }
        self.confirmations
            .try_reserve(count)
            .map_err(|_| invalid("cannot retain original static query confirmations"))?;
        let mut record_ids = BTreeSet::new();
        for index in 0..count {
            // SAFETY: the collector's immutable lease owns the validated aligned
            // contiguous record span and strings until known unload acknowledgement.
            let record = unsafe { view.records.add(index).read() };
            if record.struct_size as usize != std::mem::size_of::<OriginalStaticRecord>()
                || !record_ids.insert(record.record_id)
                || record.valid_fields
                    & !(STATIC_NAME_VALID
                        | STATIC_SOURCE_VALID
                        | STATIC_CONTEXT_VALID
                        | STATIC_CORRELATION_VALID)
                    != 0
                || (record.valid_fields & STATIC_CONTEXT_VALID != 0
                    && record.context_id != view.original_context_id)
            {
                return Err(invalid(
                    "original static row identity, context or layout is invalid",
                ));
            }
            // Validate only fields which the original producer marked available.
            // A missing source or context is never repaired from a native query.
            if record.valid_fields & STATIC_SOURCE_VALID != 0 {
                unsafe { original_bytes(record.source, record.source_size)? };
            }
            let name = if record.valid_fields & STATIC_NAME_VALID != 0 {
                Some(
                    CString::new(unsafe { original_bytes(record.name, record.name_size)? })
                        .map_err(|_| invalid("original static name contains an interior NUL"))?,
                )
            } else {
                None
            };
            self.confirm_static_record(&record, name.as_deref())?;
        }
        Ok(())
    }

    fn confirm_static_record(
        &mut self,
        record: &OriginalStaticRecord,
        name: Option<&std::ffi::CStr>,
    ) -> ResourceResult<()> {
        // Allocate the owner's slot before any query. In particular, another
        // mapping can consume capacity reserved for later original rows.
        self.confirmations
            .try_reserve(1)
            .map_err(|_| invalid("cannot retain original static query facts"))?;
        let mut index = self.confirmations.len();
        self.confirmations
            .push(StaticConfirmation::new(record.record_id));
        let global = &mut self.confirmations[index];
        let Some(name) = name.filter(|name| !name.is_empty()) else {
            return Ok(());
        };
        // Only original device-static allocation rows describe live symbols.
        // Other rows remain present with unattempted query statuses.
        if record.memory_kind != DEVICE_STATIC
            || record.operation != ALLOCATION
            || record.address == 0
        {
            return Ok(());
        }
        let mut global_bytes = 0usize;
        // SAFETY: LoadedModule retains this module and its bound context; the
        // name is the exact length-delimited original symbol, made NUL-terminated.
        let status = unsafe {
            sys::cuModuleGetGlobal_v2(
                &mut global.global_address,
                &mut global_bytes,
                self.observation.module_handle as sys::CUmodule,
                name.as_ptr(),
            )
        };
        global.global_bytes = global_bytes as u64;
        global.record_status(QUERY_GLOBAL, status);
        if !global.has(QUERY_GLOBAL)
            || global.global_address != record.address
            || global.global_bytes != record.bytes
            || global.global_bytes == 0
        {
            // Keep the original successful status and raw outputs, but do not
            // certify this row's membership from a same-named foreign symbol.
            global.valid_queries &= !(1 << QUERY_GLOBAL);
            return Ok(());
        }
        let Some(end) = global.global_address.checked_add(global.global_bytes) else {
            global.valid_queries &= !(1 << QUERY_GLOBAL);
            return Ok(());
        };
        let global = *global;
        let mut cursor = global.global_address;
        loop {
            // Every actual status and output is written into retained storage,
            // including when a later slot allocation or callback fails.
            let confirmation = &mut self.confirmations[index];
            let mut allocation_bytes = 0usize;
            // SAFETY: cursor remains inside this exact live global span. No
            // address outside the symbol is probed to discover a larger extent.
            let status = unsafe {
                sys::cuMemGetAddressRange_v2(
                    &mut confirmation.allocation_base,
                    &mut allocation_bytes,
                    cursor,
                )
            };
            confirmation.allocation_bytes = allocation_bytes as u64;
            confirmation.record_status(QUERY_ALLOCATION, status);
            let mut context: sys::CUcontext = std::ptr::null_mut();
            let mut mapping_bytes = 0usize;
            macro_rules! attribute {
                ($slot:expr, $attribute:ident, $output:expr) => {{
                    // SAFETY: each output has the attribute's driver-defined
                    // scalar width; this call occurs cold under the live owner.
                    let status = unsafe {
                        sys::cuPointerGetAttribute(
                            std::ptr::from_mut($output).cast(),
                            sys::CUpointer_attribute::$attribute,
                            cursor,
                        )
                    };
                    confirmation.record_status($slot, status);
                }};
            }
            attribute!(QUERY_CONTEXT, CU_POINTER_ATTRIBUTE_CONTEXT, &mut context);
            confirmation.context_handle = context as u64;
            attribute!(
                QUERY_DEVICE,
                CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                &mut confirmation.device_ordinal
            );
            attribute!(
                QUERY_MEMORY_TYPE,
                CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                &mut confirmation.memory_type
            );
            attribute!(
                QUERY_MAPPED,
                CU_POINTER_ATTRIBUTE_MAPPED,
                &mut confirmation.mapped
            );
            attribute!(
                QUERY_MAPPING_BASE,
                CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR,
                &mut confirmation.mapping_base
            );
            attribute!(
                QUERY_MAPPING_SIZE,
                CU_POINTER_ATTRIBUTE_MAPPING_SIZE,
                &mut mapping_bytes
            );
            confirmation.mapping_bytes = mapping_bytes as u64;
            attribute!(
                QUERY_PHYSICAL_BLOCK,
                CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID,
                &mut confirmation.physical_block_id
            );
            attribute!(
                QUERY_BUFFER_ID,
                CU_POINTER_ATTRIBUTE_BUFFER_ID,
                &mut confirmation.buffer_id
            );
            let next = confirmation
                .mapping_base
                .checked_add(confirmation.mapping_bytes);
            if !confirmation.has(QUERY_MAPPING_BASE)
                || !confirmation.has(QUERY_MAPPING_SIZE)
                || !confirmation.has(QUERY_MAPPED)
                || confirmation.mapped == 0
                || confirmation.mapping_base > cursor
                || next.is_none_or(|next| next <= cursor || next >= end)
            {
                break;
            }
            cursor = next.expect("validated actual mapping boundary");
            // No CUDA query for the next piece precedes this fallible growth.
            // A failure leaves every previously observed piece in its owner.
            self.confirmations
                .try_reserve(1)
                .map_err(|_| invalid("cannot retain another original static mapping"))?;
            index = self.confirmations.len();
            self.confirmations.push(global);
        }
        Ok(())
    }
}

fn unknown_driver_error() -> DriverError {
    DriverError(sys::CUresult::CUDA_ERROR_UNKNOWN)
}

fn invalid(message: &str) -> ResourceError {
    ResourceError::Driver(message.into())
}

/// The pointer's allocation and lifetime are supplied by the immutable C lease.
unsafe fn original_bytes<'a>(pointer: *const u8, size: u64) -> ResourceResult<&'a [u8]> {
    let size =
        usize::try_from(size).map_err(|_| invalid("original static string size exceeds usize"))?;
    if size > isize::MAX as usize || (pointer as usize).checked_add(size).is_none() {
        return Err(invalid("original static string span overflows"));
    }
    if size == 0 {
        return Ok(&[]);
    }
    if pointer.is_null() {
        return Err(invalid("available original static string has null storage"));
    }
    // SAFETY: the original collector lease pins the complete supplied byte span;
    // this function validates the scalar geometry before forming the borrow.
    Ok(unsafe { std::slice::from_raw_parts(pointer, size) })
}
