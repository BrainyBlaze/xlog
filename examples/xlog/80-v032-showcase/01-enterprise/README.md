# Enterprise Analytics Example

Demonstrates v0.3.2 features in a corporate HR/Finance/Org context.

## Modules

| Module | Purpose | Key Features |
|--------|---------|--------------|
| `hr/employees.xlog` | Employee data | **symbols** for names, departments, skills |
| `finance/compensation.xlog` | Salary calculations | **UDFs** for bonus, tax, net pay |
| `org/hierarchy.xlog` | Org structure | **recursive** management chains |

## Features Demonstrated

| Feature | Usage |
|---------|-------|
| `symbol` type | Employee names (`"Alice Chen"`), department IDs (`eng`), skill names (`rust`) |
| `func` (arithmetic) | `years_of_service`, `calculate_bonus`, `net_after_tax` |
| `func` (conditional) | `bonus_multiplier`, `tax_bracket`, `seniority_bonus` |
| `use` imports | Main imports all three modules |
| `private` predicate | `current_year` helper |
| Recursion | `management_chain` for org traversal |
| Aggregation | `count`, `sum`, `max` for analytics |

## UDFs Defined

```prolog
func years_of_service(HireYear, CurrentYear) = CurrentYear - HireYear.

func bonus_multiplier(Tier) =
    if Tier = 1 then 20      // gold
    else if Tier = 2 then 15 // silver
    else if Tier = 3 then 10 // bronze
    else 5.

func calculate_bonus(BaseSalary, BonusPct) = BaseSalary * BonusPct / 100.

func tax_bracket(AnnualSalary) =
    if AnnualSalary > 20000000 then 37
    else if AnnualSalary > 15000000 then 32
    else if AnnualSalary > 10000000 then 24
    else if AnnualSalary > 5000000 then 22
    else 12.

func net_after_tax(Gross, TaxPct) = Gross - (Gross * TaxPct / 100).

func seniority_bonus(YearsService) =
    if YearsService > 5 then (YearsService - 5) * 2
    else 0.
```

## Running

```bash
cargo run -p xlog-cli -- run examples/xlog/80-v032-showcase/01-enterprise/main.xlog
```

## Queries

| Query | Description |
|-------|-------------|
| `senior_engineer(Name, Skill, Level)` | Engineers with 5+ years and skill level 4+ |
| `high_earner(Name, Dept, Total)` | Employees earning > $150k total |
| `dept_total_comp(Dept, Total)` | Total compensation by department |
| `expert_skill(Name, Skill)` | Employees with level-5 skills |
| `management_chain(e009, Mgr, Level)` | Management chain for employee e009 |
| `large_team_manager(Name, Dept, Size)` | Managers with 3+ direct reports |
| `team_size(Team, Size)` | Size of each team |

## Data Volume

- 46 employees across 5 departments
- 60+ skill assignments
- Full org hierarchy (4 levels deep)
- Compensation data for all employees
