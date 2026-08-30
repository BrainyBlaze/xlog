# Architecture debt ledger — XLOG. Аудит 2 (волна 3)

Дата: 2026-08-24. Продолжение файла `docs/architecture-debt.md` (аудит 1, 2026-08-10, волны 1–2,
записи SD-001…SD-039). Этот документ предназначен для дописывания в тот же файл: нумерация
`SD-nnn` сквозная и не переиспользуется.

Аудит только диагностирует. Код не менялся, ничего не собиралось и не запускалось.

---

## Рамка аудита — прочесть до использования любой записи

Аудит выполнен **против релизного рефа `origin/main` @ `6478c884`** (2026-08-24, версия
`0.12.0` в `Cargo.toml`), выгруженного в изолированный клон вне репозитория. Рабочее дерево
разработчика (`whitepaper-source` @ `9516a1c1`) отстаёт от него на **121 коммит** и было
использовано только для чтения прошлого ledger'а.

С момента аудита 1 (`a9c2ed17`) на `main` пришло **48 коммитов, +123 345 / −12 452 строк**.
Ключевое отличие от аудита 1: **whitepaper теперь живёт на `main`** (`paper/`, 14 секций,
PR #274/#275/#276), поэтому его больше не нужно проверять на отдельной ветке.

Что покрыто: все 12 измерений `dimensions.md` — либо находками, либо явным
`проверено, ниже порога`. Что не покрыто — в разделе «Заявление о покрытии» в конце.

Исключено из измерений: `examples/neural/baseline/**` (вендоренные DeepProbLog и Scallop),
`target/`, `docs/whitepaper/artifacts/**`.

---

## Строка тренда

| дата | scope line | del/add (окно) | open | accepted | fixed с прошлого | new | worse |
|---|---|---|---|---|---|---|---|
| 2026-08-10 (аудит 1) | all tracked source \| ext=rs,py \| since=1 year ago \| vendored excluded | 0.174 | 18 → 39 | 0 | — | 39 | — |
| 2026-08-24 (аудит 2) | all tracked source \| ext=rs,py,cu \| since=1 year ago \| vendored excluded | **0.162** | 41 | 2 | **4** | 12 | 6 |

⚠️ Scope line изменился (добавлен `cu`), поэтому 0.174 и 0.162 **строго говоря несопоставимы**.
Сопоставим помесячный профиль при одном и том же scope:

```
2026-02  0.02   2026-03  0.66   2026-04  0.06   2026-05  0.15
2026-06  0.18   2026-07  0.59   2026-08  0.13
```

Профиль **пунктирный, а не плоский** — то есть удаление здесь кому-то поручают, хотя бы
эпизодически. Март (0.66) и июль (0.59) — кампании. Август 0.13 при +101 748 строк: это окно,
про которое спрашивает пользователь («много нового добавляли»), и в нём же прошла
**настоящая кампания прополки** — семь коммитов с явным «remove»:

`2c0bf5dc remove unreachable exact GPU compiler` · `4ea98a5a remove dormant type and name
scaffolding` · `801f640d remove unused CSR CNF module` · `699fbda4 remove unselectable dense D4
frontier` · `dd669a28 remove dormant epistemic export` · `a19dca15 remove dormant RawCudaView
recorder contract` · плюс `916a74a3`/`9d424980 share ...` (дедупликация).

**Это редкость и это правильно.** Низкое 0.13 объясняется не отсутствием прополки, а тем, что
рядом приземлились 38 тыс. строк нового движка и 46 тыс. строк закоммиченных артефактов
экспериментов. Метрика не приговор; приговор — то, что прополка не покрыла ни одну из открытых
записей ledger'а (см. таблицу статусов).

---

## Статус записей аудита 1

Перепроверено у источника лично: SD-001…SD-010, SD-012, SD-013, SD-015, SD-016, SD-017,
SD-025, SD-026, SD-031, SD-034, SD-036, SD-039. Остальные — механическая проверка или
`не перепроверено`.

| ID | статус на `origin/main` @ 6478c884 | что изменилось |
|---|---|---|
| SD-001 | **open, лучше** | `cuda-ci.yml` теперь триггерится и на `pull_request` (job `python-wheel` на self-hosted CUDA-раннере, ~15 адресных `cargo test`). Но `cargo test --workspace --all-targets` остался в job `rust-tests`, а тот под `if: github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main'` — **полный набор тестов не запускается автоматически никогда** |
| SD-002 | **worse** | было 2 копии `make_provider` без `XlogDeviceRuntime`, стало **3** (добавился `provider/ilp_exact_nary.rs`); всего копий 15 (было 13) |
| SD-003 | open, confirmed | без изменений: `mc_resident.cu:541` пишет флаг, `:543-556` считает queries/evidence из `cur` безусловно; единственный читатель `resident_status_flags` — тест `mc_resident.rs:104` |
| SD-004 | open, **повышено до confirmed** | `provider/mod.rs:1490` (`!v.is_empty() && v != "0"`) и `wcoj_dispatch.rs:106` (`v=="1" \|\| eq_ignore_ascii_case("true")`) прочитаны лично. Появился **четвёртый** парсер — см. SD-048 |
| SD-005 | open, confirmed | `git grep 'validators::'` = 0 |
| SD-006 | **частично fixed → расщеплено** | см. SD-049 |
| SD-007 | open, confirmed | `cudarc` в 8 манифестах крейтов |
| SD-008 | open, confirmed | единственный `impl KernelProvider` — `MockProvider` в `xlog-core/src/traits.rs:86` |
| SD-009 | open, **повышено до high** | `/CLAUDE.md` и `/AGENTS.md` в `.gitignore:17-18`, в дереве отсутствуют. Появилось первое наблюдаемое последствие — SD-044 |
| SD-010 | open, confirmed | `XLOG_DISABLE_WCOJ_TRIANGLE`: 0 попаданий в `crates/*/src`, 2 doc-страницы описывают его как «hard kill switch» |
| SD-011 | open, suspected | не перепроверено у источника. Подтверждено смежное: `compilation/mod.rs:517` явно верифицирует **pre-smoothing** контур, `:580` сглаживает после (SD-031) |
| SD-012 | **worse → заменено на SD-040** | оказалось шире: Python отбрасывает **все пять** `#pragma prob_*`, не только `prob_samples` |
| SD-013 | open, **повышено до confirmed** | `wcoj_paper_class.rs:420,428`: `VRAM_GATE_BYTES` только печатается через `eprintln!`, ни одного `assert` |
| SD-014 | open, suspected | не перепроверено |
| SD-015 | open, confirmed, **worse** | `XlogDeviceRuntime::try_get` (`runtime.rs:389`) по-прежнему без продакшн-вызовов (все вызовы с `:753` внутри `#[cfg(test)] mod tests`, начинается на `:749`). Добавилась **новая точка сборки рантайма мимо синглтона** — `xlog-gpu/src/logic.rs:2794-2807` |
| SD-016 | open, **worse** | Python-тестов, читающих `.rs`/`.pyi` как текст: **21** (было 15) |
| SD-017 | **частично fixed** | PR #253 добавил 12 `py.detach(...)` в `epistemic.rs`, `ilp.rs`, `neural.rs` (pyo3 0.29 переименовал `allow_threads` → `detach`). Но именно те точки входа, что названы в записи, GIL по-прежнему держат: `program.rs:552 evaluate(&self, _py: Python<'_>, …)` (параметр так и назван `_py`) и `logic.rs:158,367` |
| SD-018 | open, suspected | `trainable_rule` — 28 попаданий в `pyxlog/python/pyxlog/ilp/neurosymbolic.py`, 0 в `xlog-logic/src/grammar.pest` |
| SD-019 | **fixed** | `paper/artifacts/runtime-optimization/{persistent_hash_index,chain_shared_memory_scorer}.json` закоммичены; `paper/artifacts/head-to-head/README.md` документирует протокол каждой ячейки |
| SD-020 | open, не перепроверено | |
| SD-021 | open, не перепроверено | |
| SD-022 | open, suspected | смежно подтверждено: `require_accepted_gpu_tuple_evidence_trace` и `require_accepted_gpu_world_view_evidence_trace` не имеют вызовов вне своего файла (SD-045) |
| SD-023 | open | `epistemic_workspace.rs` 6 950 строк против `xlog-logic/src/epistemic.rs` 6 419 — разрыв сохраняется |
| SD-024 | open, confirmed | подтверждено механизмом: единственный job, который запустил бы этот файл, — `rust-tests`, под `workflow_dispatch` |
| SD-025 | open, confirmed | `xlog-cuda-tests` не является зависимостью ни одного другого крейта |
| SD-026 | **worse** | `Ok(None)` в `wcoj_dispatch.rs`: **184** (было ~155) |
| SD-027 | open, не перепроверено | 11 упоминаний `sparse_overflow` во всём дереве |
| SD-028 | open, не перепроверено | |
| SD-029 | **accepted** | `paper/artifacts/head-to-head/README.md:52-56` теперь **явно публикует** выпавшую ячейку: «queries=4 even measures −8.9%», «not a monotone climb». Замалчивания больше нет; одиночность измерений остаётся, но это осознанный протокол |
| SD-030 | open, confirmed | `paper/sections/05_probabilistic.tex:74` дословно сохраняет «the solver never reads its status back to the host»; `xlog-solve/src/gpu_cdcl.rs:204,208` — два `dtoh_scalar_untracked` |
| SD-031 | open, **повышено до confirmed** | `compilation/mod.rs:517` («Verify equivalence on the *base* circuit (pre-smoothing)») против `:580` `smooth_random_vars_device` |
| SD-032 | fixed (закрыто в аудите 1) | |
| SD-033 | open, не перепроверено | |
| SD-034 | open, **переформулировано → SD-042** | претензия не «цифра неверна», а «артефакт старше исправления отчётности» |
| SD-035 | open, не перепроверено | |
| SD-036 | open, **повышено до confirmed** | `recursive.rs:17 MAX_FIXPOINT_ITERATIONS = 1000` используется в `execute_fixpoint` (`:1016`), тогда как `execute_recursive_scc` (`:447`) берёт `self.config.max_iterations` (`:613`). Два цикла неподвижной точки в одном файле с разными лимитами; путь `RirNode::Fixpoint` (`node_dispatch.rs:287`) идёт в первый |
| SD-037 | open, не перепроверено | |
| SD-038 | open, не перепроверено | |
| SD-039 | open, **уточнено** | файл **уменьшился**: 29 236 строк (было 30 474), 202 теста. Подтверждено, что это **burst-churn, а не sustained**: 194 из 202 коммитов приходятся на 2026-05-19/20. По правилу census §3 это «кто-то запустил генератор», а не «дизайн не устоялся» — топ-1 хотспот по churn×size **не** является местом, где долг стоит денег на каждое изменение |

**Итого: fixed 4 (SD-019, SD-032, плюс частично SD-006 и SD-017), accepted 1 (SD-029),
worse 6 (SD-002, SD-009, SD-012, SD-015, SD-016, SD-026), open остальные.**

Ни одна из шести записей `high` аудита 1 не была закрыта.

---

# Новые записи

### SD-040 — на Python-пути отбрасываются **все** директивы `#pragma prob_*`; на CLI-пути они работают
status: open   severity: high   first-seen: 2026-08-24   confidence: confirmed
(заменяет и расширяет SD-012)

```
Cost:      один и тот же `.xlog` даёт разные ответы из CLI и из Python, без предупреждения.
           `#pragma prob_samples = 1000000` → CLI считает 1 000 000 сэмплов, `program.evaluate()`
           из Python — 10 000. То же для `prob_seed` (0), `prob_confidence` (0.95),
           `prob_max_nonmonotone_iterations` (1024) и `prob_method` (None). Каждый воспроизводимый
           прогон, опубликованный из Python-ноутбука, не воспроизводится из CLI и наоборот.
           Следующая добавленная директива по умолчанию попадёт в ту же яму: правильный образец
           лежит в том же workspace.
Evidence:  `crates/pyxlog/src/program.rs:597-601` — `let mut cfg = McEvalConfig::default();`
           затем `cfg.samples = samples.unwrap_or(10000); cfg.seed = seed.unwrap_or(0);
           cfg.confidence = confidence; cfg.max_nonmonotone_iterations = …;` — безусловные
           присваивания. `McEvalConfig::from_directives`
           (`crates/xlog-prob/src/mc/mod.rs:162-180`) из `pyxlog` **не вызывается вообще**
           (`git grep from_directives -- crates/pyxlog` = 0).
           Правильный образец: `crates/xlog-cli/src/main.rs:1368-1382`
           `apply_mc_cli_overrides` — каждое поле под `if let Some(...)`, поверх конфигурации,
           уже построенной из директив. Второй экземпляр той же ошибки: `program.rs:753`.
Verified:  confirmed — я прочёл оба места сам и проверил отсутствие вызова `from_directives`
           из `pyxlog` грепом по всему дереву.
Remedy:    в `program.rs:597` и `:753` заменить `McEvalConfig::default()` на
           `McEvalConfig::from_directives(&program.directives)?`, а пять присваиваний обернуть
           в `if let Some(...)` по образцу CLI. Сигнатуры `confidence: f64` и
           `max_nonmonotone_iterations: usize` придётся сделать `Option<...>`, иначе pyo3-дефолт
           неотличим от явного аргумента — это ломающее изменение Python API, но именно оно
           и есть суть бага. ~1 сессия.
Leave it:  расхождение CLI/Python остаётся неотличимым от шума в любом отчёте, который не
           печатает фактический `samples`.
```

### SD-041 — типизированная ошибка `ResourceExhausted` лжёт о своей единице измерения в 74 местах из 94
status: open   severity: high   first-seen: 2026-08-24   confidence: confirmed

```
Cost:      сообщение «Resource exhausted: …, estimated N bytes, budget M bytes» печатается
           там, где N и M — не байты, а количество кандидатов, столбцов, запусков ядра или
           фильтров строк, а «бюджет» — `u32::MAX`. Оператор, читающий такую ошибку, не может
           отличить настоящее исчерпание VRAM от отказа по арности, и предпримет неверное
           действие («увеличить memory_mb» на ошибку «слишком много выходных столбцов»).
           Ровно этот механизм делает нечитаемым закоммитированный артефакт статьи — см. SD-042.
           Любая ретрай-логика вокруг этого варианта ошибки некорректна по построению.
Evidence:  определение и Display: `crates/xlog-core/src/error.rs:22-30` («GPU memory budget
           exceeded», формат «estimated {estimated_bytes} bytes, budget {budget_bytes} bytes»).
           94 реальных места конструирования (`grep -rn "XlogError::ResourceExhausted *{"
           crates/*/src`), из них 74 — в `crates/xlog-runtime/src/executor/epistemic_workspace.rs`:
           `:692` `estimated_bytes: candidate_count`, `:867` `output_column_count` при
           `budget_bytes: u32::MAX`, `:892` `kernel_launches`, `:934` `negated_row_filter_count`,
           `:1441` `candidate_count` при `budget_bytes: rejection_reason_slots`.
           Ни одно из 74 не упоминает `current_bytes`/`used_bytes`/`prior_peak`.
           Правильная конструкция существует и единственна: `crates/xlog-cuda/src/memory.rs:238-258`
           `MemoryPressure::into_error` — `estimated_bytes = current + requested`, плюс полный
           контекст. Она добавлена коммитом `867f48a1` (#254, 2026-08-17) и покрывает 6 мест из 94.
Verified:  confirmed — я прочёл определение, `MemoryPressure::into_error` и пять мест
           в `epistemic_workspace.rs` лично; счётчик 94 получен приведённой командой.
Remedy:    два действия, оба механические. (1) Ввести отдельный вариант
           `XlogError::CapacityExceeded { context, actual: u64, limit: u64, unit: &'static str }`
           и перевести на него 74 не-байтовых места — компилятор укажет каждое. (2) Оставшиеся
           байтовые места провести через `MemoryPressure`. ~1–2 сессии.
           Это gate, а не список: после разделения тип перестаёт допускать ошибку.
Leave it:  каждый отчёт об OOM из эпистемического воркспейса остаётся нечитаемым, и следующий
           артефакт для статьи унаследует ту же неоднозначность.
```

### SD-042 — артефакт, подкрепляющий главное внешнее сравнение статьи, старше исправления отчётности, которое он использует
status: open   severity: high   first-seen: 2026-08-24   confidence: confirmed
(переформулировка SD-034: утверждение не «неверно», а **неподкреплено**)

```
Cost:      `paper/sections/10_evaluation.tex:124` и `00_abstract.tex:3` утверждают, что
           материализация промежуточного соединения «needs 3–7 GB and exhausts the budget (OOM)
           at 23M triangles». Рецензент, открывший закоммитированный JSON, видит
           `ResourceExhausted { estimated_bytes: 3234973092, budget_bytes: 18874368000 }` —
           то есть отказ при оценке в 3.23 ГБ против бюджета 18.87 ГБ, что читается как
           прямое противоречие тексту. На самом деле это старый формат ошибки, который
           печатал только инкремент запроса без уже занятых байт; исправление пришло позже.
           Пока артефакт не переснят, никто — включая авторов — не может сказать, держится
           ли фраза «3–7 GB and exhausts the budget».
Evidence:  артефакт `paper/artifacts/head-to-head/triangle_counting_vs_souffle.json`,
           последняя запись данных — коммит `59f4a103`, **2026-07-10**;
           перемещён без изменения содержимого коммитом `f72d298a` (2026-07-13).
           Исправление отчётности — `867f48a1` «fix(cuda): report exact memory pressure and
           peak usage (#254)», **2026-08-17**, добавившее `current_bytes`/`required_bytes`/
           `prior_peak_bytes` в `crates/xlog-cuda/src/memory.rs:238-258`.
           Записанные пики арма `enum_then_count`: 3 048 МБ и 7 351 МБ; у третьей ячейки
           (`h80_e500000`) полей `wall_s`/`peak_mb` нет вообще — только `rc: 1` и строка `err`.
Verified:  confirmed — я прочёл артефакт, его историю в git и коммит #254 сам.
           **Не проверено:** реальный пик на текущем `main`. Закрыть можно только перезапуском.
Remedy:    перезапустить head-to-head против Soufflé на текущем `main` (RunPod RTX 4090,
           протокол уже описан в `paper/artifacts/head-to-head/README.md`), закоммитить новый
           артефакт с полным `context` из `MemoryPressure`, затем либо оставить фразу, либо
           заменить её на то, что покажет запись. Половина сессии плюс один прогон на GPU.
           Одновременно закоммитить раннер этого сравнения — сейчас в `runners/` лежат только
           три скрипта для xlog-only изоляций.
Leave it:  главное внешнее сравнение статьи опирается на артефакт, чьё сообщение об ошибке
           внутренне противоречиво. Это ровно тот класс, который рецензент проверяет первым.
```

### SD-043 — 41 тыс. строк, включая крупнейшую подсистему проекта и два breaking-изменения, пришли на `main` без pull request
status: open   severity: high   first-seen: 2026-08-24   confidence: confirmed

```
Cost:      единственные автоматические предмёржевые ворота проекта — `ci.yml` на
           `pull_request` (fmt, clippy, build, python-contract, caviar-examples) и job
           `python-wheel` в `cuda-ci.yml` на `pull_request` — на этих коммитах не запускались,
           потому что pull request'а не было. Ревью человеком тоже не было. Именно в этом
           наборе оказались: новый резидентный движок (54 файла, +38 159/−487), поддержка
           joint-constraint произвольной арности (+2 454/−595) и помеченное `!` ломающее
           `perf(cuda)!: skip redundant certified unions`. Второе последствие — SD-044.
           На `main` нет защиты от прямого push, поэтому следующий крупный коммит может
           повторить это по умолчанию, а не по решению.
Evidence:  из последних 60 коммитов `origin/main` пять не содержат ссылки `(#N)`:
           `3919424f` (2026-08-18, 54 файла, +38 159/−487, «execute dependency-closed plans in
           one resident graph»), `3321371f` (+2 454/−595) и его merge `af9c71bd`,
           `0dbd006e` (`perf(cuda)!`, 25 файлов), `deb4cd0e`.
           Все пять и только они авторизованы идентичностью `levi770 <levi@brainyblaze.com>`;
           остальные 55 — под noreply-идентичностями GitHub, то есть прошли через PR.
           Команда: `git log --format="%h|%p|%s" -60 origin/main`.
Verified:  confirmed — я прогнал перечисление и сверил авторов и размеры сам.
           **Не проверено:** настроена ли на GitHub защита ветки. Из репозитория это не видно;
           наблюдаемое поведение говорит, что нет.
Remedy:    включить branch protection на `main`: require pull request, require status checks
           (`ci.yml` job'ы + `cuda-ci / python-wheel`). Полчаса в настройках репозитория,
           ноль строк кода. Это gate, а не список — он делает класс невозможным.
Leave it:  предмёржевые ворота остаются опциональными, и их обход не оставляет следа нигде,
           кроме отсутствующего `(#N)` в теме коммита.
```

### SD-044 — `CHANGELOG.md` отредактирован вручную вопреки правилу, которое лежит только в неотслеживаемом файле
status: open   severity: high   first-seen: 2026-08-24   confidence: confirmed
(прямое следствие SD-009 и SD-043; первый наблюдаемый ущерб от обоих)

```
Cost:      контракт проекта гласит: «Никогда не тегать релизы, не править
           `[workspace.package].version` и `CHANGELOG.md` вручную — этим владеет release-plz».
           Коммит `3919424f` дописал в `CHANGELOG.md` секцию `### Changed` от руки.
           Правило нарушено, потому что оно физически недоступно из чистого клона: `CLAUDE.md`
           и `AGENTS.md` перечислены в `.gitignore:17-18`. Ревью не поймало нарушение,
           потому что PR не было (SD-043). Три механизма отказали одновременно, и каждый
           из них уже описан в ledger'е отдельной записью.
           Второе следствие: `[Unreleased]` в `CHANGELOG.md` содержит ровно одну запись —
           про привязку `LogicMaterializedStore`. Резидентного движка, joint-constraint
           произвольной арности и `0dbd006e perf(cuda)!` там нет. Их нет и в `ROADMAP.md`
           (2 326 строк), и в `docs/architecture/roadmap.mdx` (104 строки):
           `grep -in "resident conditional\|resident graph"` по всем трём даёт ноль.
           То есть release notes следующей версии сейчас умолчат о крупнейшей фиче цикла
           и о двух ломающих изменениях.
Evidence:  `git show 3919424f -- CHANGELOG.md` (+4 строки, ручная секция);
           `git log -3 -- CHANGELOG.md` показывает, что последним файл трогал именно этот
           коммит, а до него — `chore: release v0.12.0 (#196)`, то есть release-plz;
           `release-plz.toml:5 changelog_path = "CHANGELOG.md"`;
           `.gitignore:17-18`; `CHANGELOG.md:5-11`.
Verified:  confirmed — прочитал diff, историю файла, конфиг release-plz и `.gitignore` сам.
Remedy:    три шага, каждый мал. (1) Вернуть `CLAUDE.md` под git, `AGENTS.md` сделать
           заглушкой-ссылкой, убрать две строки из `.gitignore` (это SD-009, ~1 сессия).
           (2) Добавить в `ci.yml` шаг: PR, изменяющий `CHANGELOG.md` и не помеченный
           лейблом `release`, падает. ~15 строк YAML. (3) Дописать в `[Unreleased]` три
           пропущенные позиции **через conventional-коммит**, а не руками, либо позволить
           release-plz пересобрать секцию.
Leave it:  release notes выйдут неполными, а правило, запрещающее ручную правку, продолжит
           отсутствовать в репозитории, где его должны читать.
```

### SD-045 — целая ось публичного API эпистемического продакшн-адаптера не имеет ни одного потребителя три месяца
status: open   severity: high   first-seen: 2026-08-24   confidence: confirmed

```
Cost:      `crates/xlog-prob/src/epistemic_production.rs` — 4 081 строка, 63 различных
           `pub fn`. Имена образуют декартово произведение четырёх независимых осей:
           глагол (`compile` / `compile_and_evaluate` / `encode_*_pir_cnf` / `evaluate`) ×
           вход (`source` / `program`) × `conditioned` × `with_grads` ×
           носитель улик (`with_accepted_world_view` / `with_gpu_execution_result` /
           `for_gpu_execution_results` / `for_gpu_batch_execution_result`).
           **Вся ось `_with_accepted_world_view` мертва: 11 методов из 11, ноль вызовов
           вне собственного файла, с 2026-05-18.** Плюс два стража
           `require_accepted_gpu_{tuple,world_view}_evidence_trace` — тоже ноль.
           Python-мост вызывает соседнюю ось: `pyxlog/src/epistemic.rs:158`
           `compile_and_evaluate_conditioned_source_with_gpu_execution_result`.
           Цена платится каждый раз: добавление пятого носителя улик или ещё одного флага
           снова умножает поверхность; 13 методов надо держать компилируемыми, покрытыми
           и понятыми при каждом рефакторинге типов.
Evidence:  определения `:340`, `:458`, `:1552`, `:1822`, `:1909`, `:2000`, `:2103`, `:2328`,
           `:2585`, `:2731`, `:2882`, `:2987`, `:3137`.
           Проверка: для каждого из 63 имён `grep -rn "\b<name>\b" crates/ python/ examples/
           --include=*.rs --include=*.py --include=*.pyi | grep -v epistemic_production.rs`
           даёт 0 ровно для этих 13.
           Введены коммитами `9e012421` (2026-05-18) и `84906de6` (2026-05-19).
Verified:  confirmed — счётчик получен приведённой командой, и я лично проверил, что
           `pyxlog` вызывает `_with_gpu_execution_result`, а не `_with_accepted_world_view`.
           **Оговорка:** проверка токенная. Она бы не увидела вызов, собранный макросом
           или через строковую диспетчеризацию; ни того, ни другого в этом файле нет.
Remedy:    сначала удалить, не абстрагировать. Спросить владельца, планируется ли потребитель
           у оси `accepted_world_view`. Если нет — удалить 13 методов (компилятор укажет
           на всё, что развалится: ничего). Если да — назвать дату. ~0.5 сессии на решение.
           Отдельно и позже: свернуть оставшиеся 50 методов в `EpistemicRequest`-структуру
           с полями вместо имён — это и есть лекарство от произведения, но это отдельная работа.
Leave it:  файл продолжает расти как произведение осей, и следующая ось прибавит ещё 8–11
           методов, из которых половина снова не найдёт потребителя.
```

---

## Компактные записи — волна 3

| ID | Заголовок | Sev | Confidence | Опора | Цена в одну строку |
|---|---|---|---|---|---|
| SD-046 | нет ворот против мёртвого кода, и идиома `pub fn` в lib-крейтах отключает встроенные | med | confirmed | 129 `pub fn` из `crates/*/src`, чьё имя встречается **ровно один раз** во всём Rust-исходнике (команда в записи ниже); история `fc2693fc` в SD-005 | мёртвая функция неотличима от живой ни для компилятора, ни для ревьюера; следующая мёртвая ось (SD-045) обнаружится ещё через три месяца |
| SD-047 | линт-ворота шириной в три правила на 400 тыс. строк, и никто не измерял цену полных | med | confirmed | `.github/workflows/ci.yml:226-234` (`-D dbg_macro -D todo -D unimplemented`, `-A approx_constant`); `[workspace.lints]` в `Cargo.toml` отсутствует, `lints.workspace = true` — 0 из 15 крейтов | комментарий в CI признаёт «broad repo-wide debt», но объём этого долга не измерен ни разу, поэтому решение «оставить три правила» — не компромисс, а бессрочная отсрочка |
| SD-048 | конфигурационная поверхность ~110 переменных окружения, 94 прямых `env::var` в продакшн-коде, ≥4 несовместимых булевых конвенции | med | confirmed | `xlog-gpu/src/logic.rs:50-53` (4-я копия `!value.is_empty() && value != "0"`); `provider/mod.rs:1490`; `wcoj_dispatch.rs:106`; `harness/provider.rs:110-115` (`"1"\|"true"\|"TRUE"\|"True"`) | `XLOG_USE_RECORDED_OPS=false` включает путь; новейшая подсистема воспроизвела ту же копию через две недели после того, как SD-004 её описала |
| SD-049 | CPU и GPU теперь согласованы в порядке float — и одинаково самопротиворечивы; агрегаты по-прежнему расходятся | med | confirmed | `xlog-logic/src/arithmetic_eval.rs:545-559` (`Eq/Ne` — IEEE, `Lt/Le/Gt/Ge` — `total_cmp`) зеркалит `kernels/filter.cu:135-140`; но `xlog-prob/src/aggregates.rs:127-129` бросает на NaN, а `kernels/groupby.cu:238` `if (val <= old_val) break;` для NaN ложно, и NaN выигрывает атомарный max | на обеих сторонах `X == 0.0` и `X < 0.0` истинны одновременно для `X = -0.0`, и это нигде не задокументировано; NaN в `logsumexp` — ошибка на CPU и ответ группы на GPU. Оба нарушают заявленный контракт «row-set parity» и «fail-closed» (`docs/architecture/overview.mdx:163-170`) |
| SD-050 | вторая реализация реляционных операторов, выключенная по умолчанию, охраняется 13 игрушечными кейсами | med | confirmed | `kernels/resident_{relational,filter_project,schedule}.cu` — 26 `__global__`, включая `resident_join_build/probe_inner/probe_semi`, `resident_filter_*`, `resident_project_*`, `resident_set_insert`, дублирующие `join.cu`/`filter.cu`/`pack.cu`/`set_ops.cu`; `ResidentSelectionMode::from_env` (`xlog-gpu/src/logic.rs:60-69`) при пустом окружении → `Disabled` | ~25 тыс. строк второго исполнителя, который в продакшне не выполняется; дифференциальный тест `resident_semantic_acceptance_matrix` (`logic.rs:8340`) сравнивает ответы обеих трасс, но на 13 программах в 3–8 строк |
| SD-051 | резидентный путь строит собственный `XlogDeviceRuntime` на каждый вызов, мимо назначенного синглтона | low | confirmed | `xlog-gpu/src/logic.rs:2794-2807` создаёт `StreamPool::with_defaults` + `AsyncCudaResource` + `GlobalDeviceBudget` + `XlogDeviceRuntime::with_resource` внутри `resident_provider_view`, вызываемого из `logic.rs:2293` на каждой оценке | усугубляет SD-015: назначенный единственный владелец `XlogDeviceRuntime::try_get` по-прежнему без продакшн-вызовов, а точек создания стало больше. Локальный бюджет при этом разделяется корректно (см. ND-014), поэтому severity низкая |

Команда для SD-046 (воспроизводима без сборки):

```sh
grep -rhno "pub fn [a-z_][a-z_0-9]*" crates/*/src --include=*.rs | sed 's/.*pub fn //' | sort -u > pubfns.txt
grep -rhoE "\b[a-z_][a-z_0-9]*\b" crates/ --include=*.rs | sort | uniq -c > allids.txt
awk 'NR==FNR{w[$1]=1;next} ($2 in w) && $1==1 {print $2}' pubfns.txt allids.txt | wc -l   # → 129
```

Распределение 129 по крейтам: `xlog-cuda-tests` 42, `xlog-cuda` 30, `pyxlog` 21, `xlog-prob` 11,
`xlog-logic` 11, `xlog-core` 4, `xlog-solve` 3, `xlog-gpu` 3, `xlog-runtime` 2, `xlog-neural` 1,
`xlog-ir` 1. Из 21 в `pyxlog` часть может вызываться из Python по имени — это оговорка,
а не опровержение: остальные 108 в Python не экспортируются.

---

## Не долг — проверено и снято

Эти пункты удешевляют следующий аудит. Они не `accepted` — это снятые сигналы.

- **ND-012 — дисковый кеш контуров версионирован правильно.** `crates/xlog-prob/src/compilation/disk_cache.rs:14-15`
  `MAGIC = 0x584C4743` и `FORMAT_VERSION = 1`, проверяются при чтении (`:360-369`), ключ включает
  `cnf_hash, config_hash, random_vars_hash, sm, FORMAT_VERSION` (`:71`), есть тест на порчу magic
  (`:783`). Это ровно тот класс (персистентный формат без маркера версии), который в других
  проектах приводил к тихой выдаче устаревшего объекта после исправления. Здесь его нет.
- **ND-013 — дифференциальный тест резидентного движка настоящий.** `assert_required_resident_semantics`
  (`xlog-gpu/src/logic.rs:7422-7476`) прогоняет одну и ту же программу под
  `XLOG_DISABLE_RESIDENT_RECURSION=1` и под `XLOG_REQUIRE_RESIDENT_RECURSION=1` и сравнивает
  снимки результатов запросов, а не телеметрию. Ранний выход без GPU закрыт: провайдер строится
  через `finish_test_provider_setup(..., XLOG_REQUIRE_CUDA == "1")` (`:5968-5982`), а workflow
  выставляет `XLOG_REQUIRE_CUDA: "1"` и дополнительно требует точного
  `N passed; 0 failed; 0 ignored` и нуля строк `Skipping test:` (`cuda-ci.yml:224-276`).
  Претензия только к покрытию (SD-050), не к конструкции.
- **ND-014 — overlay рантайма не создаёт второго бюджета.** `GpuMemoryManager::with_runtime_overlay`
  (`memory.rs:677-714`) клонирует `Arc` на тот же `accounting` и тот же `budget`, поэтому
  родительские и overlay-аллокации не могут превысить лимит совместно. Отдельный
  `GlobalDeviceBudget` внутри свежего рантайма инициализируется тем же `budget_limit` и может
  быть только не строже — переподписки нет. Гипотеза о двойном учёте проверена и опровергнута.
- **ND-015 — фоновой работы в продакшне по-прежнему нет.** Все `thread::spawn` в `crates/*/src`
  находятся внутри `mod tests`; единственное исключение вне тестов — `std::thread::sleep`
  для debounce в `xlog-cli/src/main.rs:321`. ND-006 из аудита 1 держится.
- **ND-016 — артефакты статьи стали самодокументируемыми.** `paper/artifacts/head-to-head/README.md`
  публикует таблицу `comparison_acceptable` по каждому файлу, объясняет, почему у сравнения
  с Soufflé он `false` намеренно, и печатает ту самую ячейку `−8.9%`, замалчивание которой было
  предметом SD-029. Это прямой ответ на находки аудита 1 и правильный образец для остальных
  артефактов.
- **Измерение 5 (доступ к данным и схемы)** — проверено, ниже порога: постоянного хранилища
  с запросами в проекте нет, единственная персистентность — дисковый кеш контуров (ND-012)
  и кеш cubin. Составных ключей, миграций схемы и неиндексированных фильтров не существует.
- **Измерение 11 (клиентский слой)** — проверено, ниже порога: `docs/` — статический
  Mintlify-сайт (`docs/docs.json`), без опроса, состояния и логики. Единственное замечание —
  два документа роадмапа (`ROADMAP.md` 2 326 строк и `docs/architecture/roadmap.mdx` 104 строки)
  описывают одно; учтено в SD-044, отдельной записи не заводится.

---

## Топ-5 к действию сейчас

Ранжировано по (цена × уверенность) ÷ усилие. Всё остальное остаётся в ledger'е со статусом.

1. **SD-043 — включить branch protection на `main`.** Полчаса в настройках, ноль строк кода,
   закрывает целый класс и является предусловием для того, чтобы любые другие ворота
   что-то значили.
2. **SD-040 — починить отбрасывание `#pragma prob_*` в Python.** ~1 сессия, правильный образец
   лежит в соседнем крейте, следствие — воспроизводимость результатов между двумя поверхностями.
3. **SD-042 — переснять head-to-head против Soufflé на текущем `main` и закоммитить раннер.**
   Полсессии плюс один прогон на RunPod; это единственная запись, блокирующая подачу статьи.
4. **SD-041 — расщепить `ResourceExhausted` на байтовый и небайтовый варианты.** 1–2 сессии,
   компилятор укажет все 94 места, делает класс невозможным и попутно чинит читаемость
   артефактов для (3).
5. **SD-045 — принять решение по мёртвой оси `_with_accepted_world_view`.** 0.5 сессии на решение,
   1 на удаление 13 методов. Самое дешёвое сокращение в списке и единственный пункт,
   уменьшающий кодовую базу.

Осознанно **не** в списке: SD-001 (полный `cargo test` в CI) — правильный шаг, но он даст
красное на неизвестном объёме и требует отдельного бюджета на триаж; сначала (1).
SD-050 (второй исполнитель) — не долг, пока путь выключен; долгом он станет в тот день,
когда его включат по умолчанию, и тогда цена — покрытие, а не удаление.

---

## Заявление о покрытии — чего аудит 2 не сделал

- **Ничего не собиралось и не запускалось.** Ни `cargo build/test/clippy`, ни `pytest`,
  ни `maturin`, ни GPU. Все утверждения прочитаны из исходников. Ни одно расхождение,
  описанное здесь, не воспроизведено экспериментально.
- **Объём линт-долга (SD-047) не измерен.** Измерить его без CUDA-хоста нельзя;
  это единственное число, которое превратило бы SD-047 из наблюдения в решение.
- **Не перепроверены у источника 12 записей аудита 1:** SD-011, SD-014, SD-020, SD-021,
  SD-023, SD-027, SD-028, SD-033, SD-035, SD-037, SD-038 и часть SD-018. Для каждой в таблице
  статусов указано, что осталось непроверенным.
- **Не открыт вообще:** содержимое `crates/xlog-induce` и `xlog-neural`; `provider/wcoj_metadata.rs`
  (5 208 строк); `executor/rewrite.rs`; `provider/resident_schedule.rs` (8 265 строк — прочитаны
  только сигнатуры и точка требования рантайма); тела CUDA-ядер, кроме `filter.cu`, `groupby.cu`,
  `mc_resident.cu` и списка `__global__` в резидентных; `crates/xlog-cuda-tests/src/categories/**`;
  `fuzz/`, `tools/`, `build.rs`.
- **Не проверены заявления статьи, кроме четырёх.** Прочитаны и сверены SD-029, SD-030, SD-034/042
  и часть SD-031. Остальные девять секций переписанной статьи (она выросла с 705 строк до 14
  секций через PR #274/#275/#276) против кода не сверялись. Это самое дорогое оставшееся
  измерение — то же, что и после аудита 1.
- **История запусков GitHub Actions недоступна из репозитория.** Настроена ли защита ветки
  и являются ли какие-либо проверки required — определить нельзя. SD-043 исходит из
  наблюдаемого поведения (пять коммитов без PR), а не из конфигурации.
- **Проверка «мёртвости» (SD-045, SD-046) токенная,** а не по графу вызовов. Она не увидела бы
  вызов, собранный макросом. Для `epistemic_production.rs` и пяти проверенных вручную
  кандидатов из census §9 такого не обнаружено.
- **Не смотрели 20+ активных feature-веток** и другие worktree'ы (`xlog-mixed-bodies`,
  `xlog-plasticity`, `xlog-stageb`, `.worktrees/log-z-e`).

---

## Исправления, внесённые позже

- **2026-08-30, строка тренда.** В ней стояло `new = 11`, тогда как записи идут
  SD-040…SD-051 — двенадцать. Исправлено на 12. Ошибка была замечена внешней сверкой
  (`xlog-audit2-handoff-2026-08-27.md`), которая посчитала все двенадцать и оказалась права.
- **2026-08-30, счёт тихих отказов.** SD-026 и последующие записи приводят «184 `Ok(None)`
  в `executor/wcoj_dispatch.rs`». Это счёт **строк**: одна строка (`:1803`) содержит
  `Ok(None) => Ok(None)`, поэтому вхождений 185. Ограничение в
  `python/tests/test_structural_debt_ceilings.py` записано по вхождениям, то есть 185.
  На существо находки это не влияет, на воспроизводимость числа — влияет.
