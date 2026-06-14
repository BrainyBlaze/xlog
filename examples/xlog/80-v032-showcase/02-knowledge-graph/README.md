# Knowledge Graph Example

Movie ontology and analytics example demonstrating XLOG's semantic reasoning
and graph-analytics capabilities.

## Domain Model

This example models a movie-domain knowledge graph with:

- **Entities**: Movies, people, genres, studios, awards, and ontology concepts
- **Type hierarchy**: Ontology classes with recursive inheritance from `thing`
- **Relations**: Directing, acting, genres, studios, awards, and nominations
- **Analytics**: Rating tiers, return on investment, filmography counts,
  collaboration pairs, and decade-level summaries

## Features Demonstrated

| Feature | Usage |
|---------|-------|
| **symbol type** | Entity identifiers, labels, concept names |
| **Recursive rules** | Type inheritance through `is_subclass` |
| **count aggregation** | Filmography counts, genre counts, studio counts |
| **sum and average aggregation** | Box-office totals and rating summaries |
| **Comparisons** | Filtering high-return movies, acclaimed movies, and major studios |
| **Arithmetic** | Decade calculation, productivity scores, return on investment |

## Key Predicates

### Base Data

```xlog
pred entity_type(symbol, symbol).       // entity, direct ontology type
pred movie(symbol, symbol, u32, u32, u32, u32).
pred person(symbol, symbol, u32, symbol).
pred studio(symbol, symbol, u32).
pred directed(symbol, symbol).          // director, movie
pred acted_in(symbol, symbol).          // actor, movie
```

### Derived Relations

```xlog
// Type hierarchy traversal.
pred is_subclass(symbol, symbol).
is_subclass(Child, Parent) :- subclass_of(Child, Parent).
is_subclass(Child, Ancestor) :-
    subclass_of(Child, Parent),
    is_subclass(Parent, Ancestor).

// Movie return on investment.
pred movie_roi(symbol, symbol, u32).
movie_roi(Movie, Title, ReturnOnInvestment) :-
    movie(Movie, Title, _, _, Budget, BoxOffice),
    ReturnOnInvestment is roi_pct(BoxOffice, Budget).

// Director filmography counts.
pred director_movie_count(symbol, symbol, u64).
director_movie_count(Director, Name, Count) :-
    person(Director, Name, _, _),
    entity_type(Director, director),
    director_movie_count_raw(Director, Count).
```

## Queries

1. **Type hierarchy**: Direct and inherited ontology classes
2. **Movie analytics**: Decades, rating tiers, box-office classes, and return
   on investment
3. **Director analytics**: Filmography counts, career spans, productivity, and
   box-office totals
4. **Actor analytics**: Age-at-release, career spans, and productivity
5. **Genre and studio analytics**: Counts, ratings, box-office totals, and
   awards
6. **Collaboration analytics**: Director-actor collaborations and repeated
   collaborations
7. **Entity type inference**: Direct and inherited entity classifications

## Running

From this example directory:

```bash
cargo run -p xlog-cli -- run main.xlog
```

## Data Statistics

- 35 movies with rating, budget, and box-office metadata
- 30 people across actor and director roles
- 12 genres and 8 studios
- Multi-level ontology for movies, people, concepts, organizations, and awards
