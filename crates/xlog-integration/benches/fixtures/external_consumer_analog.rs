use std::collections::BTreeSet;

use super::TriangleFixture;

const EXTERNAL_CONSUMER_CODEBOOK_SIZE: u32 = 1024;

fn codebook_id(seed: u32) -> u32 {
    seed % EXTERNAL_CONSUMER_CODEBOOK_SIZE
}

fn insert_pair(rows: &mut BTreeSet<(u32, u32)>, a: u32, b: u32) {
    rows.insert((a, b));
}

fn sorted_pairs(rows: BTreeSet<(u32, u32)>) -> Vec<(u32, u32)> {
    rows.into_iter().collect()
}

pub fn external_consumer_analog(scale: u32) -> TriangleFixture {
    let scale = scale.max(32);
    let hot_middle = (scale / 16).clamp(16, 64);
    let output_band = (scale / 16).clamp(32, 64);

    let mut xy = BTreeSet::new();
    let mut yz = BTreeSet::new();
    let mut xz = BTreeSet::new();

    for doc_pos in 0..scale {
        let x = codebook_id(doc_pos);
        for rank in 0..hot_middle {
            insert_pair(&mut xy, x, codebook_id(rank));
        }
        for offset in 0..output_band {
            insert_pair(&mut xz, x, codebook_id(doc_pos + offset));
        }
    }

    for rank in 0..hot_middle {
        let y = codebook_id(rank);
        for tail in 0..scale {
            insert_pair(&mut yz, y, codebook_id(tail));
        }
    }

    TriangleFixture {
        name: "external_consumer_analog",
        recursive: true,
        bundle_path_status:
            "metadata=PASS first_branch=GRACEFUL second_branch=GRACEFUL helper_split=PASS stream_multiplexing=PASS chain_promoter=PASS cuda_graph=PASS invoked=7/7",
        e_xy: sorted_pairs(xy),
        e_yz: sorted_pairs(yz),
        e_xz: sorted_pairs(xz),
    }
}
