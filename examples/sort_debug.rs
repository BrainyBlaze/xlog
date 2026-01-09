// Diagnostic test for large sort
use std::sync::Arc;
use xlog_core::{MemoryBudget, Schema, ScalarType};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

fn main() {
    let device = Arc::new(CudaDevice::new(0).unwrap());
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    let provider = CudaKernelProvider::new(device, memory).unwrap();
    
    // Test progressively larger sizes
    for size in [10, 50, 100, 256, 500, 1000, 5000, 10000] {
        // Create reverse-sorted input (worst case for sort)
        let input: Vec<u32> = (0..size).rev().collect();
        let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);
        
        let buffer = provider.create_buffer_from_u32_slice(&input, schema).unwrap();
        let sorted = provider.sort(&buffer, &[0]).unwrap();
        let result = provider.download_column_u32(&sorted, 0).unwrap();
        
        // Check if sorted correctly
        let expected: Vec<u32> = (0..size).collect();
        let is_correct = result == expected;
        
        // Count out-of-order pairs
        let mut out_of_order = 0;
        for i in 1..result.len() {
            if result[i] < result[i-1] {
                out_of_order += 1;
            }
        }
        
        println!("Size {}: {} (out of order pairs: {})", 
            size, 
            if is_correct { "PASS" } else { "FAIL" },
            out_of_order);
        
        if !is_correct && size <= 20 {
            println!("  Input:  {:?}", input);
            println!("  Got:    {:?}", result);
            println!("  Expect: {:?}", expected);
        } else if !is_correct {
            println!("  First 20 got: {:?}", &result[..20.min(result.len())]);
            println!("  Last 20 got:  {:?}", &result[result.len().saturating_sub(20)..]);
        }
    }
}
