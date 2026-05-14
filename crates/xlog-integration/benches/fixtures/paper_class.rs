use std::collections::BTreeSet;

#[derive(Clone)]
pub struct TriangleFixture {
    pub name: &'static str,
    pub recursive: bool,
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

#[inline]
fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

fn insert_pair(rows: &mut BTreeSet<(u32, u32)>, a: u32, b: u32) {
    rows.insert((a, b));
}

fn sorted_pairs(rows: BTreeSet<(u32, u32)>) -> Vec<(u32, u32)> {
    rows.into_iter().collect()
}

pub fn call_graph_edge_analog(scale: u32) -> TriangleFixture {
    let scale = scale.max(32);
    let hub_degree = (scale / 10).max(8);
    let mut xy = BTreeSet::new();
    let mut yz = BTreeSet::new();
    let mut xz = BTreeSet::new();
    for caller in 0..scale {
        let callee = caller % hub_degree;
        insert_pair(&mut xy, caller, callee);
        insert_pair(&mut yz, callee, caller);
        insert_pair(&mut xz, caller, caller);
        if caller % 3 == 0 {
            insert_pair(&mut yz, callee, scale + caller);
            insert_pair(&mut xz, caller, scale + caller);
        }
    }
    TriangleFixture {
        name: "call_graph_edge_analog",
        recursive: false,
        e_xy: sorted_pairs(xy),
        e_yz: sorted_pairs(yz),
        e_xz: sorted_pairs(xz),
    }
}

pub fn andersen_analog(scale: u32) -> TriangleFixture {
    let scale = scale.max(32);
    let fields = (scale / 8).max(4);
    let mut xy = BTreeSet::new();
    let mut yz = BTreeSet::new();
    let mut xz = BTreeSet::new();
    for obj in 0..scale {
        let alloc_site = obj % (scale / 2).max(1);
        let field = obj % fields;
        insert_pair(&mut xy, alloc_site, field);
        insert_pair(&mut yz, field, obj);
        insert_pair(&mut xz, alloc_site, obj);
        if obj % 5 == 0 {
            let alias = scale + obj;
            insert_pair(&mut yz, field, alias);
            insert_pair(&mut xz, alloc_site, alias);
        }
    }
    TriangleFixture {
        name: "andersen_analog",
        recursive: false,
        e_xy: sorted_pairs(xy),
        e_yz: sorted_pairs(yz),
        e_xz: sorted_pairs(xz),
    }
}

pub fn ddisasm_analog(scale: u32) -> TriangleFixture {
    let scale = scale.max(32);
    let mut state = 39_003_u64;
    let mut xy = BTreeSet::new();
    let mut yz = BTreeSet::new();
    let mut xz = BTreeSet::new();
    for block in 0..scale {
        let forward = (block + 1) % scale;
        let backward = (block + scale - 1) % scale;
        insert_pair(&mut xy, block, forward);
        insert_pair(&mut xy, block, backward);
        insert_pair(&mut yz, forward, block);
        insert_pair(&mut yz, backward, block);
        insert_pair(&mut xz, block, block);
        if block % 4 == 0 {
            let dataflow = (lcg_next(&mut state) % scale as u64) as u32;
            insert_pair(&mut yz, forward, dataflow);
            insert_pair(&mut xz, block, dataflow);
        }
    }
    TriangleFixture {
        name: "ddisasm_analog",
        recursive: true,
        e_xy: sorted_pairs(xy),
        e_yz: sorted_pairs(yz),
        e_xz: sorted_pairs(xz),
    }
}

pub fn paper_class_fixtures(scale: u32) -> Vec<TriangleFixture> {
    let mut registry = FixtureRegistry::default();
    registry.add_fixture_module("fixtures::paper_class");
    assert_eq!(registry.module_count(), 1);
    vec![
        call_graph_edge_analog(scale),
        andersen_analog(scale),
        ddisasm_analog(scale),
    ]
}
