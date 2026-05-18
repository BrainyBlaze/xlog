#![allow(dead_code)]

#[path = "../benches/fixtures/paper_class.rs"]
mod paper_class;

#[test]
fn dts_dlm_analog_fixture_is_registered_with_paper_class_harness() {
    let fixtures = paper_class::paper_class_fixtures(128);
    assert_eq!(
        fixtures.len(),
        4,
        "G_W39_DTSDLM extends the three paper-class fixtures with one DTS-DLM analog"
    );

    let dts = fixtures
        .iter()
        .find(|fixture| fixture.name == "dts_dlm_analog")
        .expect("dts_dlm_analog fixture is registered");

    assert!(
        dts.recursive,
        "DTS-DLM analog must exercise recursive Stage-4 set maintenance"
    );
    assert!(
        !dts.e_xy.is_empty() && !dts.e_yz.is_empty() && !dts.e_xz.is_empty(),
        "DTS-DLM analog must populate every relation in the triangle harness"
    );
    assert!(
        dts.e_yz.len() >= dts.e_xy.len(),
        "middle-key fanout should model DTS-DLM's chain-2 support expansion"
    );
    assert!(
        dts.bundle_path_status.contains("g_w66_cuda_graph=PASS")
            && dts.bundle_path_status.contains("invoked=7/7"),
        "DTS-DLM analog must expose its full bundle-path coverage through fixture metadata"
    );
}
