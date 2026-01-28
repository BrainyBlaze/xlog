cmake_minimum_required(VERSION 3.18)
project(xlog_kernels CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Generate PTX for common architectures
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

# Compile to PTX only (no object files)
set(CUDA_PTXAS_FLAGS "-v")

# Kernel sources
set(KERNEL_SOURCES
    join.cu
    dedup.cu
    groupby.cu
    scan.cu
    filter.cu
    pack.cu
    pir.cu
    cnf.cu
    sort.cu
    set_ops.cu
    circuit.cu
    mc_sample.cu
)

# Create PTX for each kernel
foreach(KERNEL_SRC ${KERNEL_SOURCES})
    get_filename_component(KERNEL_NAME ${KERNEL_SRC} NAME_WE)
    add_library(${KERNEL_NAME}_ptx OBJECT ${KERNEL_SRC})
    set_target_properties(${KERNEL_NAME}_ptx PROPERTIES
        CUDA_PTX_COMPILATION ON
    )
endforeach()

# Install PTX files
install(FILES
    $<TARGET_OBJECTS:join_ptx>
    $<TARGET_OBJECTS:dedup_ptx>
    $<TARGET_OBJECTS:groupby_ptx>
    $<TARGET_OBJECTS:scan_ptx>
    $<TARGET_OBJECTS:filter_ptx>
    $<TARGET_OBJECTS:pack_ptx>
    $<TARGET_OBJECTS:pir_ptx>
    $<TARGET_OBJECTS:cnf_ptx>
    $<TARGET_OBJECTS:sort_ptx>
    $<TARGET_OBJECTS:set_ops_ptx>
    $<TARGET_OBJECTS:circuit_ptx>
    $<TARGET_OBJECTS:mc_sample_ptx>
    DESTINATION ptx
)
