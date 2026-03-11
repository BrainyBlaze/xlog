mod common;
use common::setup_provider;

use xlog_cuda::provider::{pir_kernels, PIR_MODULE};

#[test]
fn test_provider_loads_pir_module_entrypoints() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let device = provider.device().inner();
    assert!(
        device
            .get_func(PIR_MODULE, pir_kernels::PIR_PACK_KEYS)
            .is_some(),
        "expected provider to load PIR module and expose pir_pack_keys"
    );
    assert!(
        device
            .get_func(PIR_MODULE, pir_kernels::PIR_HASH_KEYS)
            .is_some(),
        "expected provider to load PIR module and expose pir_hash_keys"
    );
    assert!(
        device
            .get_func(PIR_MODULE, pir_kernels::PIR_MARK_UNIQUE)
            .is_some(),
        "expected provider to load PIR module and expose pir_mark_unique"
    );
}
