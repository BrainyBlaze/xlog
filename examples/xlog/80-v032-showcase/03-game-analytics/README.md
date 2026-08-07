# Game Analytics Example

Multiplayer game analytics platform demonstrating XLOG's capabilities for player statistics, social networks, and leaderboard calculations.

## Domain Model

This example models a gaming platform with:

- **Players**: Profiles, statistics, regional distribution
- **Matches**: Results, kills/deaths/assists, game modes
- **Achievements**: Unlockable achievements with prerequisite chains
- **Social**: Friends network, guilds with power rankings
- **Ranking**: Elo ratings, tiers, win rates

## Features Demonstrated

| Feature | Usage |
|---------|-------|
| **symbol type** | Player names, usernames, region names, achievement identifiers |
| **Recursive rules** | Achievement prerequisites, friend-of-friend connections |
| **count aggregation** | Achievement counts, tier distribution |
| **sum aggregation** | Total kills, achievement points, guild Elo totals |
| **Comparisons** | Elite players (Elo >= 2200), grandmasters (Elo >= 2400) |
| **Arithmetic** | Elo tier calculation, kill/death/assist ratio, win rate |

## Key Predicates

### Base Data
```xlog
pred player(symbol, symbol, symbol, symbol).   // player, username, country, status
pred player_stats(symbol, u32, u32, u32).      // player, total experience, games played, hours played
pred match_result(symbol, symbol, u32, u32, u32). // match, player, kills, deaths, assists
pred achievement(symbol, symbol, u32).         // achievement, name, rarity
pred achievement_requires(symbol, symbol).     // achievement, prerequisite
pred guild(symbol, symbol, symbol).            // guild, name, leader
pred friend(symbol, symbol).                   // player1, player2
pred player_elo(symbol, u32).                  // player, elo_rating
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
    achievement(AchId, _, Rarity),
    Points is achievement_points(Rarity).

// Guild power ranking
pred guild_power_ranking(symbol, symbol, u64, u64).
guild_power_ranking(GuildName, LeaderName, TotalEloRating, CompletedAchievementMembers) :-
    guild(Guild, GuildName, Leader),
    player(Leader, LeaderName, _, _),
    guild_total_elo_rating(Guild, TotalEloRating),
    guild_total_achievements(Guild, CompletedAchievementMembers).
```

## Queries

`main.xlog` cross-references all four modules (`players/profiles`, `matches/history`, `achievements/system`, `ranking/elo`) with 15 queries:

1. **Elite players**: Active players with Elo >= 2200 and 300+ achievement points
2. **Regional leaderboard**: Active players with Elo >= 2000, by region
3. **Top fraggers by region**: Players with 100+ total kills, with rating and win rate
4. **Guild power rankings**: Leader, total Elo rating, and achievement-completing members per guild
5. **Grandmaster players**: Players with Elo >= 2400
6. **Tier distribution**: Player count per calculated Elo tier
7. **High win rate players**: Players with a 60%+ win rate
8. **Achievement progress**: Achievement count and total points per player
9. **Legendary achievement holders**: Players holding rarity-1 (legendary) achievements
10. **Achievement prerequisite chain**: All prerequisites for the Perfectionist achievement (recursive)
11. **Friend network analysis**: Elo comparison between friends (close match vs. skill gap)
12. **Potential friend introductions**: Friend-of-friend pairs who aren't already friends
13. **Popular game modes**: Match count per game mode
14. **Match performance**: Kill/death/assist ratio per player per match
15. **Incomplete achievement chains**: Players missing prerequisite achievements

## Running

From this example directory:

```bash
cargo run -p xlog-cli -- run main.xlog
```

## Data Statistics

- 55 players across 5 regions
- 110 matches with detailed kill/death/assist results
- 25 achievements with prerequisite chains (combat, teamwork, progression, winning, social), plus standalone skill achievements
- 8 guilds with 39 memberships
- Friendship network with bidirectional links
