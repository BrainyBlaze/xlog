# Knowledge Graph Example

Scientific ontology and research network demonstrating XLOG's capabilities for semantic reasoning and graph analytics.

## Domain Model

This example models a scientific research ecosystem with:

- **Entities**: Researchers, institutions, publications, concepts
- **Type Hierarchy**: Ontology classes with inheritance (Thing → Agent → Person → Researcher)
- **Relations**: Authorship, citations, topics, co-authorship
- **Analytics**: Citation counts, publication metrics, semantic inference

## Features Demonstrated

| Feature | Usage |
|---------|-------|
| **symbol type** | Entity IDs, labels, concept names |
| **Recursive rules** | Type inheritance (`is_a`), citation chains (`cites_transitively`) |
| **count aggregation** | Citation counts, publication counts per researcher |
| **Comparisons** | Filtering prolific authors (Count >= 3), deep citations (Depth > 1) |
| **Arithmetic** | Citation depth tracking |

## Key Predicates

### Base Data
```xlog
pred entity(symbol, symbol).           // entity_id, entity_type
pred subclass_of(symbol, symbol).      // child_type, parent_type
pred researcher(symbol, symbol, symbol). // id, name, institution
pred publication(symbol, symbol, u32). // id, title, year
pred cites(symbol, symbol).            // citing_pub, cited_pub
```

### Derived Relations
```xlog
// Type hierarchy (transitive closure)
pred is_a(symbol, symbol).
is_a(Child, Parent) :- subclass_of(Child, Parent).
is_a(Child, Ancestor) :-
    subclass_of(Child, Parent),
    is_a(Parent, Ancestor).

// Citation analysis
pred citation_count(symbol, u64).
citation_count(Pub, count(Citing)) :- cites(Citing, Pub).

// Multi-hop citations
pred cites_transitively(symbol, symbol, u32).
cites_transitively(A, B, 1) :- cites(A, B).
cites_transitively(A, C, Depth) :-
    cites(A, B),
    cites_transitively(B, C, PrevDepth),
    Depth is PrevDepth + cast(1, u32).
```

## Queries

1. **Type hierarchy**: All types that inherit from 'thing'
2. **Prolific authors**: Researchers with 3+ publications
3. **Citation counts**: Per-publication citation statistics
4. **Deep citations**: Multi-hop citation chains (depth > 1)
5. **Co-authorship**: Researcher collaboration network
6. **Major institutions**: Institutions with 5+ publications
7. **Related concepts**: Strongly related concept pairs

## Running

```bash
cargo run --release -- run examples/xlog/80-v032-showcase/02-knowledge-graph/main.xlog
```

## Sample Output

```
__xlog_query_0 (Types inheriting from thing)
+------------------+
| col_0            |
+------------------+
| agent            |
| creative_work    |
| person           |
| researcher       |
| publication      |
| ...              |
+------------------+

__xlog_query_1 (Prolific authors)
+----------------+-------+-------+
| col_0          | col_1 | col_2 |
+----------------+-------+-------+
| Alice Chen     | mit   | 3     |
| Carol Williams | mit   | 3     |
+----------------+-------+-------+
```

## Data Statistics

- 8 researchers across 4 institutions
- 10 publications with citation network
- 8 research concepts with similarity scores
- 16 ontology classes in type hierarchy
