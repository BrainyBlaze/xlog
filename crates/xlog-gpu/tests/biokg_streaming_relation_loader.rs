use std::fs;

use xlog_gpu::biokg::{GraphInputFormat, StreamingGraphRelationLoader};

#[test]
fn xlog_biokg_001_streams_jsonl_csv_and_ntriples_with_hashes_and_histograms() {
    let dir = std::env::temp_dir().join(format!("xlog_biokg_001_{}", std::process::id()));
    fs::create_dir_all(&dir).expect("create temp dir");

    let jsonl = dir.join("edges.jsonl");
    fs::write(
        &jsonl,
        "{\"subject\":\"gene:A\",\"predicate\":\"treats\",\"object\":\"disease:B\",\"split\":\"train\"}\n\
         {\"subject\":\"drug:C\",\"predicate\":\"interacts_with\",\"object\":\"gene:A\",\"split\":\"test\"}\n",
    )
    .expect("write jsonl");

    let csv = dir.join("edges.csv");
    fs::write(
        &csv,
        "subject,predicate,object,split\n\
         gene:A,treats,disease:B,train\n\
         drug:C,interacts_with,gene:A,test\n",
    )
    .expect("write csv");

    let nt = dir.join("edges.nt");
    fs::write(
        &nt,
        "<gene:A> <treats> <disease:B> .\n\
         <drug:C> <interacts_with> <gene:A> .\n",
    )
    .expect("write ntriples");

    for (path, format) in [
        (&jsonl, GraphInputFormat::Jsonl),
        (&csv, GraphInputFormat::Csv),
        (&nt, GraphInputFormat::NTriples),
    ] {
        let mut streamed_edges = Vec::new();
        let report = StreamingGraphRelationLoader::new(format)
            .with_chunk_rows(1)
            .load_path_with_sink(path, |edge| streamed_edges.push(edge))
            .expect("load graph stream");

        assert_eq!(report.total_rows, 2);
        assert_eq!(report.edge_rows, 2);
        assert_eq!(streamed_edges.len(), 2);
        assert_eq!(streamed_edges[0].predicate, "treats");
        assert_eq!(streamed_edges[0].row_hash, report.row_hashes[0]);
        assert_eq!(report.relation_histogram.get("treats"), Some(&1));
        assert_eq!(report.relation_histogram.get("interacts_with"), Some(&1));
        if format == GraphInputFormat::NTriples {
            assert_eq!(report.split_histogram.get("unspecified"), Some(&2));
        } else {
            assert_eq!(report.split_histogram.get("train"), Some(&1));
            assert_eq!(report.split_histogram.get("test"), Some(&1));
        }
        assert_eq!(report.row_hashes.len(), 2);
        assert_ne!(report.row_hashes[0], report.row_hashes[1]);
        assert_eq!(report.bounded_memory.max_chunk_rows, 1);
        assert_eq!(report.bounded_memory.chunks, 2);
        assert_eq!(report.relation_columns, ["subject", "predicate", "object"]);
    }
}
