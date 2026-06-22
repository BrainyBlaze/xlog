use std::collections::BTreeSet;

#[path = "external_consumer_analog.rs"]
pub mod external_consumer_analog;

#[derive(Clone)]
pub struct TriangleFixture {
    pub name: &'static str,
    pub recursive: bool,
    pub bundle_path_status: &'static str,
    pub e_xy: Vec<(u32, u32)>,
    pub e_yz: Vec<(u32, u32)>,
    pub e_xz: Vec<(u32, u32)>,
}

impl TriangleFixture {
    pub fn total_rows(&self) -> u64 {
        (self.e_xy.len() + self.e_yz.len() + self.e_xz.len()) as u64
    }
}

#[derive(Default)]
pub struct FixtureRegistry {
    modules: Vec<String>,
}

impl FixtureRegistry {
    pub fn add_fixture_module(&mut self, module_path: impl Into<String>) {
        self.modules.push(module_path.into());
    }

    pub fn module_count(&self) -> usize {
        self.modules.len()
    }
}

fn insert_pair(rows: &mut BTreeSet<(u32, u32)>, a: u32, b: u32) {
    rows.insert((a, b));
}

fn sorted_pairs(rows: BTreeSet<(u32, u32)>) -> Vec<(u32, u32)> {
    rows.into_iter().collect()
}

fn insert_diagonal_band(rows: &mut BTreeSet<(u32, u32)>, root: u32, scale: u32, width: u32) {
    for offset in 0..width {
        insert_pair(rows, root, (root + offset) % scale);
    }
}

pub fn call_graph_edge_analog(scale: u32) -> TriangleFixture {
    let scale = scale.max(32);
    let hot_targets = (scale / 16).clamp(16, 64);
    let match_width = (scale / 16).clamp(32, 64);
    let mut xy = BTreeSet::new();
    let mut yz = BTreeSet::new();
    let mut xz = BTreeSet::new();
    for caller in 0..scale {
        insert_diagonal_band(&mut xz, caller, scale, match_width);
        for target in 0..hot_targets {
            insert_pair(&mut xy, caller, target);
        }
    }
    for target in 0..hot_targets {
        for callee_target in 0..scale {
            insert_pair(&mut yz, target, callee_target);
        }
    }
    TriangleFixture {
        name: "call_graph_edge_analog",
        recursive: false,
        bundle_path_status:
            "metadata=PASS sort_merge_overlap=GRACEFUL stream_overlap=GRACEFUL helper_split=PASS stream_multiplexing=PASS invoked=5/5",
        e_xy: sorted_pairs(xy),
        e_yz: sorted_pairs(yz),
        e_xz: sorted_pairs(xz),
    }
}

pub fn andersen_analog(scale: u32) -> TriangleFixture {
    let scale = scale.max(32);
    let fields = (scale / 16).clamp(16, 64);
    let match_width = (scale / 16).clamp(32, 64);
    let mut xy = BTreeSet::new();
    let mut yz = BTreeSet::new();
    let mut xz = BTreeSet::new();
    for alloc_site in 0..scale {
        insert_diagonal_band(&mut xz, alloc_site, scale, match_width);
        for field in 0..fields {
            insert_pair(&mut xy, alloc_site, field);
        }
    }
    for field in 0..fields {
        for obj in 0..scale {
            insert_pair(&mut yz, field, obj);
        }
    }
    TriangleFixture {
        name: "andersen_analog",
        recursive: false,
        bundle_path_status:
            "metadata=PASS sort_merge_overlap=GRACEFUL stream_overlap=GRACEFUL helper_split=PASS stream_multiplexing=PASS invoked=5/5",
        e_xy: sorted_pairs(xy),
        e_yz: sorted_pairs(yz),
        e_xz: sorted_pairs(xz),
    }
}

pub fn ddisasm_analog(scale: u32) -> TriangleFixture {
    let scale = scale.max(32);
    let stages = (scale / 16).clamp(16, 64);
    let match_width = (scale / 16).clamp(32, 64);
    let mut xy = BTreeSet::new();
    let mut yz = BTreeSet::new();
    let mut xz = BTreeSet::new();
    for block in 0..scale {
        insert_diagonal_band(&mut xz, block, scale, match_width);
        for stage in 0..stages {
            insert_pair(&mut xy, block, stage);
        }
    }
    for stage in 0..stages {
        for target in 0..scale {
            insert_pair(&mut yz, stage, target);
        }
    }
    TriangleFixture {
        name: "ddisasm_analog",
        recursive: true,
        bundle_path_status:
            "metadata=PASS sort_merge_overlap=GRACEFUL stream_overlap=GRACEFUL helper_split=PASS stream_multiplexing=PASS invoked=5/5",
        e_xy: sorted_pairs(xy),
        e_yz: sorted_pairs(yz),
        e_xz: sorted_pairs(xz),
    }
}

pub fn paper_class_expected_fixture_count() -> usize {
    4
}

pub fn paper_class_fixtures(scale: u32) -> Vec<TriangleFixture> {
    let mut registry = FixtureRegistry::default();
    registry.add_fixture_module("fixtures::paper_class");
    registry.add_fixture_module("fixtures::external_consumer_analog");
    assert_eq!(registry.module_count(), 2);
    vec![
        call_graph_edge_analog(scale),
        andersen_analog(scale),
        ddisasm_analog(scale),
        external_consumer_analog::external_consumer_analog(scale),
    ]
}
