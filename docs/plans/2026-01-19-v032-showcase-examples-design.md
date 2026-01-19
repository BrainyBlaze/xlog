# v0.3.2 Showcase Examples Design

> **Goal:** Create comprehensive real-world .xlog programs demonstrating ALL v0.3.2 features
> **Features:** Symbols, User-Defined Functions, Module System

---

## Overview

Four complete multi-module applications across different domains, each demonstrating:
- **Symbols**: Real string values instead of numeric IDs
- **UDFs**: Domain-specific calculation functions
- **Modules**: Logical separation with public APIs and private helpers

## Directory Structure

```
examples/xlog/80-v032-showcase/
├── README.md                           # Overview of all examples
├── 01-enterprise/
│   ├── README.md
│   ├── main.xlog
│   ├── hr/employees.xlog
│   ├── finance/compensation.xlog
│   └── org/hierarchy.xlog
├── 02-knowledge-graph/
│   ├── README.md
│   ├── main.xlog
│   ├── ontology/schema.xlog
│   ├── entities/movies.xlog
│   └── inference/reasoning.xlog
├── 03-game-analytics/
│   ├── README.md
│   ├── main.xlog
│   ├── players/profiles.xlog
│   ├── matches/history.xlog
│   ├── achievements/system.xlog
│   └── ranking/elo.xlog
└── 04-supply-chain/
    ├── README.md
    ├── main.xlog
    ├── inventory/stock.xlog
    ├── shipping/routes.xlog
    └── cost/calculator.xlog
```

---

## Domain 1: Enterprise/Business Analytics

**Scenario**: Company with departments, employees, skills, and compensation.

### Modules

#### hr/employees.xlog
```prolog
pred employee(symbol, symbol, symbol).      // id, name, department
pred skill(symbol, symbol, u32).            // employee_id, skill_name, level(1-5)
pred hire_date(symbol, u32, u32, u32).      // employee_id, year, month, day
```

#### finance/compensation.xlog
```prolog
pred base_salary(symbol, u32).              // employee_id, annual_base
pred bonus_tier(symbol, symbol).            // department, tier

func years_of_service(HireYear, CurrentYear) = CurrentYear - HireYear.

func bonus_multiplier(Tier) =
    if Tier = 1 then 20      // gold = 20%
    else if Tier = 2 then 10 // silver = 10%
    else 5.                  // bronze = 5%

func calculate_bonus(BaseSalary, Multiplier) = BaseSalary * Multiplier / 100.

func tax_bracket(Salary) =
    if Salary > 150000 then 35
    else if Salary > 80000 then 25
    else 15.

func net_salary(Gross, TaxRate) = Gross - (Gross * TaxRate / 100).
```

#### org/hierarchy.xlog
```prolog
pred reports_to(symbol, symbol).            // employee_id, manager_id
pred team(symbol, symbol).                  // team_name, employee_id

// Management chain (recursive)
management_chain(Emp, Mgr, 1) :- reports_to(Emp, Mgr).
management_chain(Emp, TopMgr, Level) :-
    reports_to(Emp, Mgr),
    management_chain(Mgr, TopMgr, PrevLevel),
    Level is PrevLevel + 1.
```

### Data Volume
- 50 employees across 5 departments
- 150 skill assignments
- Full org hierarchy with 3 levels

---

## Domain 2: Knowledge Graph / Semantic Web

**Scenario**: Movie database with actors, directors, genres, and semantic reasoning.

### Modules

#### ontology/schema.xlog
```prolog
pred subclass_of(symbol, symbol).           // child_type, parent_type
pred domain(symbol, symbol).                // property, subject_type
pred range(symbol, symbol).                 // property, object_type
```

#### entities/movies.xlog
```prolog
pred entity(symbol, symbol).                // id, type
pred label(symbol, symbol).                 // id, human_readable_name
pred released(symbol, u32).                 // movie_id, year
pred directed_by(symbol, symbol).           // movie_id, person_id
pred acted_in(symbol, symbol).              // person_id, movie_id
pred has_genre(symbol, symbol).             // movie_id, genre_id
pred rating(symbol, f64).                   // movie_id, imdb_rating
```

#### inference/reasoning.xlog
```prolog
func decade(Year) = (Year / 10) * 10.

func rating_category(Rating) =
    if Rating >= 8.0 then 1      // excellent
    else if Rating >= 6.5 then 2 // good
    else if Rating >= 5.0 then 3 // average
    else 4.                      // poor

func jaccard_score(Common, Total1, Total2) =
    (Common * 100) / (Total1 + Total2 - Common).

// Type inference
has_type(E, ParentType) :-
    entity(E, ChildType),
    subclass_of(ChildType, ParentType).

// Collaborators
collaborators(P1, P2) :-
    acted_in(P1, M), acted_in(P2, M), P1 != P2.
```

### Data Volume
- 100 movies across 4 decades
- 80 people (actors + directors)
- 12 genres, 300+ relationships

---

## Domain 3: Game Analytics / Leaderboard System

**Scenario**: Gaming platform with players, matches, achievements, and ELO rankings.

### Modules

#### players/profiles.xlog
```prolog
pred player(symbol, symbol).                // player_id, display_name
pred registered(symbol, u32, u32, u32).     // player_id, year, month, day
pred country(symbol, symbol).               // player_id, country_code
pred status(symbol, symbol).                // player_id, active/banned/inactive
```

#### matches/history.xlog
```prolog
pred game(symbol, symbol).                  // game_id, game_title
pred match(symbol, symbol, u32).            // match_id, game_id, timestamp
pred participant(symbol, symbol, u32).      // match_id, player_id, score
pred match_winner(symbol, symbol).          // match_id, player_id
```

#### achievements/system.xlog
```prolog
pred achievement(symbol, symbol, symbol).   // ach_id, game_id, title
pred ach_requirement(symbol, symbol, u32).  // ach_id, stat_type, threshold
pred unlocked(symbol, symbol, u32).         // player_id, ach_id, timestamp

func achievement_points(Rarity) =
    if Rarity = 1 then 100          // legendary
    else if Rarity = 2 then 50      // epic
    else if Rarity = 3 then 25      // rare
    else 10.                        // common
```

#### ranking/elo.xlog
```prolog
pred elo_rating(symbol, symbol, u32).       // player_id, game_id, rating
pred tier(symbol, symbol, symbol).          // player_id, game_id, tier_name

func elo_k_factor(GamesPlayed) =
    if GamesPlayed < 30 then 40
    else if GamesPlayed < 100 then 20
    else 10.

func elo_expected(RatingA, RatingB) =
    100 / (100 + pow(10, (RatingB - RatingA) / 400)).

func elo_change(K, Actual, Expected) =
    K * (Actual * 100 - Expected) / 100.

func tier_from_elo(Rating) =
    if Rating >= 2400 then 1        // grandmaster
    else if Rating >= 2000 then 2   // master
    else if Rating >= 1600 then 3   // diamond
    else if Rating >= 1200 then 4   // gold
    else if Rating >= 800 then 5    // silver
    else 6.                         // bronze

func win_rate(Wins, Total) = (Wins * 100) / Total.
```

### Data Volume
- 200 players across 10 countries
- 5 games, 1000 matches
- 50 achievements, 500+ unlocks

---

## Domain 4: Supply Chain / Logistics

**Scenario**: Logistics network with warehouses, products, shipments, and cost optimization.

### Modules

#### inventory/stock.xlog
```prolog
pred product(symbol, symbol, symbol).       // sku, name, category
pred warehouse(symbol, symbol, symbol).     // wh_id, city, region
pred stock(symbol, symbol, u32).            // wh_id, sku, quantity
pred reorder_point(symbol, u32).            // sku, min_quantity
pred unit_weight(symbol, f64).              // sku, weight_kg
pred unit_value(symbol, u32).               // sku, value_cents
```

#### shipping/routes.xlog
```prolog
pred lane(symbol, symbol, symbol).          // lane_id, origin_wh, dest_wh
pred carrier(symbol, symbol).               // carrier_id, carrier_name
pred lane_carrier(symbol, symbol, u32).     // lane_id, carrier_id, transit_days
pred distance(symbol, symbol, u32).         // wh1, wh2, distance_km

func estimated_arrival(ShipDay, TransitDays) = ShipDay + TransitDays.

// Reachable warehouses (transitive)
can_ship(Origin, Dest) :- lane(_, Origin, Dest).
can_ship(Origin, Dest) :-
    lane(_, Origin, Mid),
    can_ship(Mid, Dest).
```

#### cost/calculator.xlog
```prolog
pred carrier_rate(symbol, symbol, u32).     // carrier_id, rate_type, cents_per_kg
pred zone_surcharge(symbol, symbol, u32).   // origin_region, dest_region, pct
pred fuel_surcharge(u32).                   // current fuel surcharge pct

func base_shipping_cost(Weight, RatePerKg) =
    cast(cast(Weight, u32) * RatePerKg, u32).

func apply_surcharge(BaseCost, SurchargePct) =
    BaseCost + (BaseCost * SurchargePct / 100).

func volume_discount(Quantity) =
    if Quantity >= 1000 then 20
    else if Quantity >= 100 then 10
    else if Quantity >= 10 then 5
    else 0.

func total_cost(Base, ZoneSurcharge, FuelSurcharge, Discount) =
    Base
    + (Base * ZoneSurcharge / 100)
    + (Base * FuelSurcharge / 100)
    - (Base * Discount / 100).

func days_of_stock(CurrentQty, DailyDemand) = CurrentQty / DailyDemand.

func reorder_urgency(DaysOfStock) =
    if DaysOfStock < 3 then 1      // critical
    else if DaysOfStock < 7 then 2 // urgent
    else if DaysOfStock < 14 then 3
    else 4.                        // comfortable

// Best carrier (uses negation)
private pred cheaper_option(symbol, u32).
cheaper_option(Lane, Cost) :-
    lane_shipping_cost(Lane, _, OtherCost),
    OtherCost < Cost.

lane_best_carrier(Lane, Carrier, Cost) :-
    lane_carrier(Lane, Carrier, _),
    lane_shipping_cost(Lane, Carrier, Cost),
    not cheaper_option(Lane, Cost).
```

### Data Volume
- 100 products across 8 categories
- 15 warehouses in 5 regions
- 500 stock records, 50 shipping lanes

---

## Feature Coverage Matrix

| Feature | Enterprise | Knowledge Graph | Game Analytics | Supply Chain |
|---------|------------|-----------------|----------------|--------------|
| **Symbols** | names, depts, skills | entity IDs, labels | player names, tiers | SKUs, cities |
| **UDFs (arithmetic)** | salary, bonus | decade, jaccard | ELO, win rate | shipping cost |
| **UDFs (conditional)** | tax brackets | rating category | tier assignment | urgency levels |
| **Modules (import)** | hr→finance→org | ontology→inference | players→ranking | inventory→cost |
| **Modules (private)** | internal helpers | schema internals | stat calcs | cheaper_option |
| **Aggregates** | dept totals | genre counts | leaderboards | inventory sums |
| **Recursion** | mgmt chain | type hierarchy | - | can_ship |
| **Negation** | - | - | not banned | not cheaper |

---

## Deliverables

- **16 .xlog files** across 4 domains
- **5 README files** with documentation
- **~1,200 lines** of XLOG code
- **22+ UDFs** with arithmetic, conditionals, composition
- **~2,500 facts** with realistic data
- **Full v0.3.2 coverage**: symbols, UDFs, modules
