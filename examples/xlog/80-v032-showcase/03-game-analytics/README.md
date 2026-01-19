# Game Analytics Example

Multiplayer game analytics platform demonstrating XLOG's capabilities for player statistics, social networks, and leaderboard calculations.

## Domain Model

This example models a gaming platform with:

- **Players**: Profiles, statistics, regional distribution
- **Matches**: Results, kills/deaths/assists, game modes
- **Achievements**: Unlockable achievements with prerequisite chains
- **Social**: Friends network, guilds with ranks
- **Items**: Inventory system with rarity tiers

## Features Demonstrated

| Feature | Usage |
|---------|-------|
| **symbol type** | Player names, item names, region names, achievement IDs |
| **Recursive rules** | Achievement prerequisites, friend-of-friend connections |
| **count aggregation** | Match participation, friend counts, guild sizes |
| **sum aggregation** | Total kills, XP, guild power, inventory value |
| **Comparisons** | Top fraggers (kills >= 20), high achievers (points >= 100) |
| **Arithmetic** | KDA ratio calculations, inventory value |

## Key Predicates

### Base Data
```xlog
pred player(symbol, symbol, u32).         // id, username, level
pred player_stats(symbol, u32, u32, u32). // id, total_xp, games_played, hours
pred match_result(symbol, symbol, u32, u32, u32). // match, player, kills, deaths, assists
pred achievement(symbol, symbol, u32).    // id, name, points
pred achievement_requires(symbol, symbol). // achievement, prerequisite
pred guild(symbol, symbol, symbol).       // id, name, leader
pred friend(symbol, symbol).              // player1, player2
```

### Derived Relations
```xlog
// Achievement chain (transitive)
pred all_prerequisites(symbol, symbol).
all_prerequisites(AchId, PrereqId) :- achievement_requires(AchId, PrereqId).
all_prerequisites(AchId, TransPrereq) :-
    achievement_requires(AchId, DirectPrereq),
    all_prerequisites(DirectPrereq, TransPrereq).

// Player statistics
pred total_kills(symbol, u64).
total_kills(PlayerId, sum(Kills)) :- match_result(_, PlayerId, Kills, _, _).

pred player_achievement_points(symbol, u64).
player_achievement_points(PlayerId, sum(Points)) :-
    player_achievement(PlayerId, AchId),
    achievement(AchId, _, Points).

// Guild analytics
pred guild_power(symbol, u64).
guild_power(GuildId, sum(Level)) :-
    guild_member(GuildId, PlayerId),
    player(PlayerId, _, Level).
```

## Queries

1. **Top fraggers**: Players with 20+ total kills by region
2. **High achievers**: Players with 100+ achievement points
3. **Top guilds**: Guilds with power rating >= 100
4. **Active regions**: Regions with 50000+ total XP
5. **Wealthy players**: Players with inventory value >= 1000
6. **Rare item owners**: Players owning rarity 3+ items
7. **Achievement chains**: Prerequisites for Champion achievement

## Running

```bash
cargo run --release -- run examples/xlog/80-v032-showcase/03-game-analytics/main.xlog
```

## Sample Output

```
__xlog_query_0 (Top fraggers by region)
+----------------+---------------+-------+
| col_0          | col_1         | col_2 |
+----------------+---------------+-------+
| DragonSlayer99 | North America | 79    |
| NightHawk      | North America | 70    |
| VoidWalker     | Europe        | 73    |
| StormBringer   | Asia Pacific  | 68    |
+----------------+---------------+-------+

__xlog_query_2 (Top guilds)
+----------------+----------------+-------+-------+
| col_0          | col_1          | col_2 | col_3 |
+----------------+----------------+-------+-------+
| Dragon Knights | DragonSlayer99 | 4     | 177   |
| Shadow Legion  | StormBringer   | 3     | 143   |
| Frost Wolves   | VoidWalker     | 3     | 132   |
+----------------+----------------+-------+-------+
```

## Data Statistics

- 12 players across 4 regions
- 10 matches with detailed results
- 10 achievements with prerequisite chains
- 3 guilds with membership hierarchy
- 10 items with rarity tiers
- Friendship network with bidirectional links
