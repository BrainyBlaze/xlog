# Enterprise Analytics Example

Demonstrates module, symbol, user-defined function, recursion, and aggregation
features in a corporate human-resources, finance, and organization context.

## Modules

| Module | Purpose | Key Features |
|--------|---------|--------------|
| `hr/employees.xlog` | Employee data | **symbols** for names, departments, skills |
| `finance/compensation.xlog` | Salary calculations | **user-defined functions** for bonus, tax, net pay |
| `org/hierarchy.xlog` | Organization structure | **recursive** management chains |

## Features Demonstrated

| Feature | Usage |
|---------|-------|
| `symbol` type | Employee names (`"Alice Chen"`), department identifiers (`eng`), skill names (`rust`) |
| `func` (arithmetic) | `years_of_service`, `calculate_bonus`, `net_after_tax` |
| `func` (conditional) | `bonus_multiplier`, `tax_bracket`, `seniority_bonus` |
| `use` imports | Main imports all three modules |
| `private` predicate | `current_year` helper |
| Recursion | `management_chain` for org traversal |
| Aggregation | `count`, `sum`, `max` for analytics |

## User-Defined Functions

```prolog
func years_of_service(HireYear, CurrentYear) = CurrentYear - HireYear.

func bonus_multiplier(Tier) =
    if Tier = cast(1, u32) then cast(20, u32)      // gold
    else if Tier = cast(2, u32) then cast(15, u32) // silver
    else if Tier = cast(3, u32) then cast(10, u32) // bronze
    else cast(5, u32).

func calculate_bonus(BaseSalary, BonusPct) = BaseSalary * BonusPct / cast(100, u32).

func tax_bracket(AnnualSalary) =
    if AnnualSalary > cast(20000000, u32) then cast(37, u32)
    else if AnnualSalary > cast(15000000, u32) then cast(32, u32)
    else if AnnualSalary > cast(10000000, u32) then cast(24, u32)
    else if AnnualSalary > cast(5000000, u32) then cast(22, u32)
    else cast(12, u32).

func net_after_tax(Gross, TaxPct) = Gross - (Gross * TaxPct / cast(100, u32)).

func seniority_bonus(YearsService) =
    if YearsService > cast(5, u32) then (YearsService - cast(5, u32)) * cast(2, u32)
    else cast(0, u32).
```

## Running

From this example directory:

```bash
cargo run -p xlog-cli -- run main.xlog
```

## Queries

| Query | Description |
|-------|-------------|
| `senior_engineer(Name, Skill, Level)` | Engineers with 5+ years and skill level 4+ |
| `high_earner(Name, Department, Total)` | Employees earning > $150k total |
| `dept_total_comp(Department, Total)` | Total compensation by department |
| `expert_skill(Name, Skill)` | Employees with level-5 skills |
| `management_chain(e009, Manager, Level)` | Management chain for employee e009 |
| `large_team_manager(Name, Department, Size)` | Managers with 3+ direct reports |
| `team_size(Team, Size)` | Size of each team |

## Data Volume

- 46 employees across 5 departments
- 60+ skill assignments
- Full org hierarchy (4 levels deep)
- Compensation data for all employees
