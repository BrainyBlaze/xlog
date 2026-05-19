#[test]
fn default_clique_entries_do_not_upload_plan_order_arrays() {
    let provider = include_str!("../src/provider/wcoj.rs");
    let kernel = include_str!("../kernels/wcoj.cu");

    assert!(
        provider.contains("None,\n            None,\n            CliqueWidthClass::FourByte"),
        "default u32 clique entry should select the no-plan fast path"
    );
    assert!(
        provider.contains("None,\n            None,\n            CliqueWidthClass::EightByte"),
        "default u64 clique entry should select the no-plan fast path"
    );
    assert!(
        provider.contains("Some(edge_order),\n            Some(iteration_order),"),
        "planned clique entry should still pass explicit plan order arrays"
    );
    assert!(
        provider.contains("let null_order_ptr = 0_u64;"),
        "no-plan fast path should pass null order pointers instead of allocating arrays"
    );
    assert!(
        kernel.contains("return order == nullptr ? fallback : static_cast<int>(order[fallback]);"),
        "kernel should interpret null order pointers as identity order"
    );
}
