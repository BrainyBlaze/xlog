//! Streaming biomedical graph relation loading helpers.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use xlog_core::{Result, XlogError};

/// Supported graph stream formats for relation loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphInputFormat {
    /// One JSON object per line with subject, predicate, object, and optional split fields.
    Jsonl,
    /// CSV with subject, predicate, object, and optional split columns.
    Csv,
    /// RDF N-Triples with subject predicate object triples.
    NTriples,
}

/// Bounded-memory telemetry collected while streaming rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundedMemoryTelemetry {
    /// Maximum row count accepted per loader chunk.
    pub max_chunk_rows: usize,
    /// Number of chunks observed while streaming the input.
    pub chunks: usize,
}

/// One typed biomedical graph edge row emitted by the streaming loader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphEdgeRow {
    /// Subject node identifier.
    pub subject: String,
    /// Predicate/relation label.
    pub predicate: String,
    /// Object node identifier.
    pub object: String,
    /// Split/provenance label.
    pub split: String,
    /// Stable row hash over normalized subject, predicate, object, and split fields.
    pub row_hash: String,
}

/// Provenance and histogram summary for a streamed biomedical graph relation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphRelationLoadReport {
    /// Total non-empty rows read from the input stream.
    pub total_rows: usize,
    /// Rows that produced typed graph edges.
    pub edge_rows: usize,
    /// Edge count per predicate/relation label.
    pub relation_histogram: BTreeMap<String, usize>,
    /// Edge count per split label.
    pub split_histogram: BTreeMap<String, usize>,
    /// Stable per-row hashes over normalized subject, predicate, object, and split fields.
    pub row_hashes: Vec<String>,
    /// Streaming chunk telemetry.
    pub bounded_memory: BoundedMemoryTelemetry,
    /// Relation column names emitted by the loader.
    pub relation_columns: Vec<String>,
}

/// Streaming loader for typed biomedical graph relation rows.
#[derive(Debug, Clone)]
pub struct StreamingGraphRelationLoader {
    format: GraphInputFormat,
    chunk_rows: usize,
}

impl StreamingGraphRelationLoader {
    /// Create a loader for the given input format.
    pub fn new(format: GraphInputFormat) -> Self {
        Self {
            format,
            chunk_rows: 100_000,
        }
    }

    /// Set the maximum number of rows accounted to a single streaming chunk.
    pub fn with_chunk_rows(mut self, chunk_rows: usize) -> Self {
        self.chunk_rows = chunk_rows.max(1);
        self
    }

    /// Stream a graph file and return provenance and histogram telemetry.
    pub fn load_path(&self, path: impl AsRef<Path>) -> Result<GraphRelationLoadReport> {
        self.load_path_with_sink(path, |_| {})
    }

    /// Stream a graph file into a caller-provided edge sink and return telemetry.
    pub fn load_path_with_sink(
        &self,
        path: impl AsRef<Path>,
        mut sink: impl FnMut(GraphEdgeRow),
    ) -> Result<GraphRelationLoadReport> {
        let file = File::open(path.as_ref()).map_err(|e| {
            XlogError::Execution(format!(
                "Failed to open biomedical graph stream {}: {}",
                path.as_ref().display(),
                e
            ))
        })?;
        let mut reader = BufReader::new(file);
        match self.format {
            GraphInputFormat::Jsonl => self.load_jsonl(&mut reader, &mut sink),
            GraphInputFormat::Csv => self.load_csv(&mut reader, &mut sink),
            GraphInputFormat::NTriples => self.load_ntriples(&mut reader, &mut sink),
        }
    }

    fn empty_report(&self) -> GraphRelationLoadReport {
        GraphRelationLoadReport {
            total_rows: 0,
            edge_rows: 0,
            relation_histogram: BTreeMap::new(),
            split_histogram: BTreeMap::new(),
            row_hashes: Vec::new(),
            bounded_memory: BoundedMemoryTelemetry {
                max_chunk_rows: self.chunk_rows,
                chunks: 0,
            },
            relation_columns: vec![
                "subject".to_string(),
                "predicate".to_string(),
                "object".to_string(),
            ],
        }
    }

    fn record_edge(
        &self,
        report: &mut GraphRelationLoadReport,
        subject: String,
        predicate: String,
        object: String,
        split: String,
        sink: &mut dyn FnMut(GraphEdgeRow),
    ) {
        let row_hash = stable_row_hash(&subject, &predicate, &object, &split);
        report.total_rows += 1;
        report.edge_rows += 1;
        *report
            .relation_histogram
            .entry(predicate.clone())
            .or_insert(0) += 1;
        *report.split_histogram.entry(split.clone()).or_insert(0) += 1;
        report.row_hashes.push(row_hash.clone());
        report.bounded_memory.chunks = report.total_rows.div_ceil(self.chunk_rows);
        sink(GraphEdgeRow {
            subject,
            predicate,
            object,
            split,
            row_hash,
        });
    }

    fn load_jsonl(
        &self,
        reader: &mut dyn BufRead,
        sink: &mut dyn FnMut(GraphEdgeRow),
    ) -> Result<GraphRelationLoadReport> {
        let mut report = self.empty_report();
        for line in reader.lines() {
            let line = line.map_err(read_error)?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let subject = json_string_field(trimmed, "subject")?;
            let predicate = json_string_field(trimmed, "predicate")?;
            let object = json_string_field(trimmed, "object")?;
            let split =
                json_string_field(trimmed, "split").unwrap_or_else(|_| "unspecified".to_string());
            self.record_edge(&mut report, subject, predicate, object, split, sink);
        }
        Ok(report)
    }

    fn load_csv(
        &self,
        reader: &mut dyn BufRead,
        sink: &mut dyn FnMut(GraphEdgeRow),
    ) -> Result<GraphRelationLoadReport> {
        let mut report = self.empty_report();
        let mut lines = reader.lines();
        let header = match lines.next() {
            Some(line) => line.map_err(read_error)?,
            None => return Ok(report),
        };
        let headers: Vec<String> = header
            .split(',')
            .map(|item| item.trim().to_string())
            .collect();
        let subject_idx = csv_column(&headers, "subject")?;
        let predicate_idx = csv_column(&headers, "predicate")?;
        let object_idx = csv_column(&headers, "object")?;
        let split_idx = headers.iter().position(|item| item == "split");

        for line in lines {
            let line = line.map_err(read_error)?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let cells: Vec<&str> = trimmed.split(',').map(str::trim).collect();
            let split = split_idx
                .and_then(|idx| cells.get(idx))
                .filter(|value| !value.is_empty())
                .copied()
                .unwrap_or("unspecified");
            self.record_edge(
                &mut report,
                csv_cell(&cells, subject_idx, "subject")?.to_string(),
                csv_cell(&cells, predicate_idx, "predicate")?.to_string(),
                csv_cell(&cells, object_idx, "object")?.to_string(),
                split.to_string(),
                sink,
            );
        }
        Ok(report)
    }

    fn load_ntriples(
        &self,
        reader: &mut dyn BufRead,
        sink: &mut dyn FnMut(GraphEdgeRow),
    ) -> Result<GraphRelationLoadReport> {
        let mut report = self.empty_report();
        for line in reader.lines() {
            let line = line.map_err(read_error)?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() < 4 || parts[3] != "." {
                return Err(XlogError::Execution(format!(
                    "Invalid N-Triples row: {}",
                    trimmed
                )));
            }
            self.record_edge(
                &mut report,
                trim_iri(parts[0]).to_string(),
                trim_iri(parts[1]).to_string(),
                trim_iri(parts[2]).to_string(),
                "unspecified".to_string(),
                sink,
            );
        }
        Ok(report)
    }
}

fn json_string_field(line: &str, key: &str) -> Result<String> {
    let needle = format!("\"{}\"", key);
    let key_pos = line
        .find(&needle)
        .ok_or_else(|| XlogError::Execution(format!("Missing JSONL field {}", key)))?;
    let after_key = &line[key_pos + needle.len()..];
    let colon_pos = after_key
        .find(':')
        .ok_or_else(|| XlogError::Execution(format!("Missing ':' for JSONL field {}", key)))?;
    let after_colon = after_key[colon_pos + 1..].trim_start();
    if !after_colon.starts_with('"') {
        return Err(XlogError::Execution(format!(
            "JSONL field {} must be a string",
            key
        )));
    }
    let rest = &after_colon[1..];
    let end = rest
        .find('"')
        .ok_or_else(|| XlogError::Execution(format!("Unterminated JSONL field {}", key)))?;
    Ok(rest[..end].to_string())
}

fn csv_column(headers: &[String], name: &str) -> Result<usize> {
    headers
        .iter()
        .position(|item| item == name)
        .ok_or_else(|| XlogError::Execution(format!("Missing CSV column {}", name)))
}

fn csv_cell<'a>(cells: &'a [&str], index: usize, name: &str) -> Result<&'a str> {
    cells
        .get(index)
        .copied()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| XlogError::Execution(format!("Missing CSV value for {}", name)))
}

fn trim_iri(value: &str) -> &str {
    value.trim_start_matches('<').trim_end_matches('>')
}

fn stable_row_hash(subject: &str, predicate: &str, object: &str, split: &str) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in format!("{subject}|{predicate}|{object}|{split}").bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn read_error(error: std::io::Error) -> XlogError {
    XlogError::Execution(format!("Failed to read biomedical graph stream: {}", error))
}
