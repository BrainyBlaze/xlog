//! CSR sparse matrix representation for CNF formulas.

use xlog_cuda::memory::TrackedCudaSlice;

/// GPU-resident CNF in CSR (Compressed Sparse Row) format.
pub(crate) struct GpuCsrCnf {
    /// Number of variables
    pub num_vars: u32,
    /// Number of clauses (rows)
    pub num_clauses: u32,
    /// CSR row pointers (length: num_clauses + 1)
    pub row_ptr: TrackedCudaSlice<u32>,
    /// CSR column indices (literal indices)
    pub col_idx: TrackedCudaSlice<u32>,
    /// CSR values (literal signs: +1 or -1)
    pub values: TrackedCudaSlice<i8>,
}
