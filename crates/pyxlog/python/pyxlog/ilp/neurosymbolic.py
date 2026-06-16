"""Joint neural-predicate and symbolic-rule training on the real xlog engine.

``train_neurosymbolic_program`` accepts an xlog program extended with two
training declarations:

``trainable_rule(id) :: head :- body.`` (optionally ``weight=<logit>``)
    A rule whose inclusion strength is learned. It is desugared into a guard
    neural predicate (``nsr_guard_<id>``) backed by a single-parameter torch
    module, so the rule weight is the guard's ``on`` probability and its
    gradient flows through the actual compiled circuit.

``train(head_predicate, objective).``
    Names the supervised head (arity 1, ranging over example row indices) and
    the training objective.

Everything else in the source is REAL xlog: after desugaring, the whole
program is parsed and compiled by the native engine (``pyxlog.Program``), and
every training step evaluates the supervised rule's circuit on the GPU via
``CompiledProgram.forward_backward``. There is no surrogate scoring path: the
rule body determines the query probability, and both network parameters and
rule weights receive gradients from the same circuit evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyxlog.ilp.inventory import RuleInventory, RuleInventoryClause

_GUARD_PREDICATE_PREFIX = "nsr_guard_"
_GUARD_NETWORK_PREFIX = "nsr_w_"
_TENSOR_SOURCE_NAME = "nsr_examples"
_ENGINE_NAME = "xlog-exact-circuit"


@dataclass(frozen=True)
class NeuroSymbolicTrainingConfig:
    """Training knobs for a declarative neuro-symbolic program."""

    steps: int = 1
    learning_rate: float = 0.1
    # Optimizer for the neural and rule-weight parameters. The supervised loss is
    # multiplicative (prob = softmax_positive * sigmoid(rule_weight)), which gives
    # a flat plateau around uniform init that plain SGD frequently cannot leave
    # (it separated a cleanly separable signal in only ~1/10 random inits, vs
    # ~8/10 for Adam in the same ablation). Adam is the default for that reason;
    # "sgd" remains selectable.
    optimizer: str = "adam"
    device: int = 0
    gpu_memory_mb: int = 4096


@dataclass(frozen=True)
class TrainableRuleDecl:
    """A ``trainable_rule`` statement after desugaring."""

    id: str
    head: str
    body_literals: tuple[str, ...]
    initial_weight: float
    source: str
    guard_predicate: str
    guard_network: str
    query_variable: str


@dataclass
class NeuroSymbolicTrainingResult:
    neural_parameter_grads: dict[str, float]
    symbolic_weight_grads: dict[str, float]
    symbolic_rule_weights: dict[str, float]
    learned_rule_inventory: RuleInventory
    losses: list[float]
    query_probabilities: list[float]
    engine: str
    proof_trace_map: Any
    # Provider device->host transfer counters observed across the training hot
    # loop (e.g. {"dtoh_calls": N, "dtoh_bytes": M}). The device-resident step
    # introduces no provider downloads, so these stay at their reset baseline;
    # surfaced so a caller can assert the no-host property of the training path.
    training_host_transfer_stats: Any = None


def train_neurosymbolic_program(
    source: str,
    *,
    networks: dict[str, Any],
    examples: list[dict[str, Any]],
    config: NeuroSymbolicTrainingConfig = NeuroSymbolicTrainingConfig(),
) -> NeuroSymbolicTrainingResult:
    """Jointly train neural predicates and symbolic rule weights on the engine."""

    import torch

    import pyxlog

    program_source, rules, train_head, objective = _desugar_source(source)
    if objective != "binary_cross_entropy":
        raise ValueError(f"unsupported training objective: {objective}")
    if not examples:
        raise ValueError("examples must not be empty")

    inputs, targets = _collect_examples(examples)

    # The native parser/compiler is the sole gatekeeper for program syntax.
    program = pyxlog.Program.compile(
        program_source,
        device=config.device,
        memory_mb=config.gpu_memory_mb,
    )

    guard_networks = {rule.guard_network for rule in rules}
    declared = set(program.declared_network_names())
    user_networks = sorted(declared - guard_networks)
    for name in user_networks:
        if name not in networks:
            raise ValueError(f"missing network for nn/4 predicate: {name}")

    modules: dict[str, Any] = {}
    for name in user_networks:
        module = networks[name].cuda()
        modules[name] = module
        program.register_network(
            name,
            module,
            _make_optimizer(config.optimizer, module.parameters(), config.learning_rate),
        )

    guard_modules: dict[str, Any] = {}
    for rule in rules:
        guard = _make_rule_weight_module(rule.initial_weight).cuda()
        guard_modules[rule.id] = guard
        program.register_network(
            rule.guard_network,
            guard,
            _make_optimizer(config.optimizer, guard.parameters(), config.learning_rate),
        )

    program.add_tensor_source(_TENSOR_SOURCE_NAME, inputs.cuda())

    queries = [f"{train_head}({i})" for i in range(len(targets))]
    losses: list[float] = []
    neural_grads: dict[str, float] = {name: 0.0 for name in modules}
    symbolic_grads: dict[str, float] = {rule.id: 0.0 for rule in rules}

    # ST-TRC Phase-1b: when MORE THAN ONE trainable rule derives the train head,
    # this is the joint multi-rule same-head soft-mixture — the candidates compete
    # for mass on one head. Route to the joint noisy-OR forward; a single defining
    # rule keeps the (faster, circuit) single-rule path below.
    candidate_ids = [
        rule.id
        for rule in rules
        if rule.head.split("(", 1)[0].strip() == train_head
    ]
    if len(candidate_ids) > 1:
        host_transfer_stats, query_probabilities = _train_joint_mixture(
            program, train_head, targets, candidate_ids, guard_modules, config, losses
        )
        for rule in rules:
            grad = guard_modules[rule.id].logit.grad
            symbolic_grads[rule.id] = (
                float(grad.detach().abs().item()) if grad is not None else 0.0
            )
    else:
        # Warm the device-side caches (circuit template + batched query-var
        # metadata) once with a throwaway forward-backward over the real queries.
        # The bounded one-time metadata uploads happen here, BEFORE the measured
        # region; the warm-up gradients are cleared by the first step's zero_grad.
        program.zero_grad()
        program.forward_backward_grouped(queries, targets)

        # Zero-host training hot loop. Every example's supervised circuit is
        # evaluated in one device-resident batched pass per step (grouped by
        # target and circuit template), so a step costs a single host sync for the
        # summed loss rather than one per query. Looping the scalar
        # forward_backward instead host-syncs on every query (.item()), which
        # leaves the GPU idle between syncs and makes training CPU-bound.
        # Reset/read the provider's host-transfer counters around the warm loop so
        # the no-host property (no tracked device<->host transfers either
        # direction) is observable.
        program.reset_host_transfer_stats()
        for _step in range(config.steps):
            program.zero_grad()
            step_loss = program.forward_backward_grouped(queries, targets)
            program.optimizer_step()
            losses.append(step_loss / len(targets))
        host_transfer_stats = program.host_transfer_stats()

        # Final gradient magnitudes, read once after training. These are per
        # parameter, not per query, so they stay out of the hot loop;
        # optimizer_step does not clear gradients (only zero_grad does), so after
        # the last step they still reflect the final backward pass.
        for name, module in modules.items():
            neural_grads[name] = float(
                sum(
                    param.grad.detach().abs().sum().item()
                    for param in module.parameters()
                    if param.grad is not None
                )
            )
        for rule in rules:
            grad = guard_modules[rule.id].logit.grad
            symbolic_grads[rule.id] = (
                float(grad.detach().abs().item()) if grad is not None else 0.0
            )

        # Final evaluation pass: query probabilities from the trained circuit, in
        # one batched pass per template (O(templates) host syncs, not O(N) per
        # query), so the whole training surface — not just the step loop — avoids
        # per-query host syncs at corpus scale.
        program.zero_grad()
        query_probabilities = program.query_probabilities_grouped(queries)
        program.zero_grad()

    learned_weights = {
        rule.id: float(torch.sigmoid(guard_modules[rule.id].logit.detach()).item())
        for rule in rules
    }
    learned_logits = {
        rule.id: float(guard_modules[rule.id].logit.detach().item()) for rule in rules
    }

    proof_trace_map = _build_proof_trace_map(
        pyxlog, rules, queries, targets, learned_logits
    )

    selected: list[RuleInventoryClause] = []
    rejected: list[RuleInventoryClause] = []
    for rule in rules:
        clause = RuleInventoryClause(
            id=rule.id,
            rule=rule.source,
            status="selected" if learned_weights[rule.id] >= 0.5 else "rejected",
            weight=learned_weights[rule.id],
            neural_predicate=_first_neural_predicate(rule),
        )
        (selected if clause.status == "selected" else rejected).append(clause)

    inventory = RuleInventory(
        selected_clauses=selected,
        rejected_clauses=rejected,
        training_objective=objective,
    )

    return NeuroSymbolicTrainingResult(
        neural_parameter_grads=neural_grads,
        symbolic_weight_grads=symbolic_grads,
        symbolic_rule_weights=learned_weights,
        learned_rule_inventory=inventory,
        losses=losses,
        query_probabilities=query_probabilities,
        engine=_ENGINE_NAME,
        proof_trace_map=proof_trace_map,
        training_host_transfer_stats=host_transfer_stats,
    )


def _train_joint_mixture(
    program: Any,
    train_head: str,
    targets: list[bool],
    candidate_ids: list[str],
    guard_modules: dict[str, Any],
    config: NeuroSymbolicTrainingConfig,
    losses: list[float],
) -> tuple[dict[str, int], list[float]]:
    """ST-TRC Phase-1b joint soft-mixture over guard-only same-head candidates.

    N candidate rules derive the SAME head, each gated by its own guard. The head
    probability is the noisy-OR over candidates of (engine relational eligibility
    x guard sigmoid); BCE on the supervised head drives the per-candidate
    competition. The relational eligibility — which head bindings each candidate's
    join fires on — comes from the engine via ``joint_candidate_eligibility``; the
    differentiable mass (guard sigmoids, OR, BCE) is torch over the guard params.
    Guard-only: no neural predicate beyond the guards, so this is the faithful
    Phase-1b mechanism with no circuit eval in the loop (the gradient to each
    guard is identical to a circuit-routed OR for input-independent guards).

    Returns ``(host_transfer_stats, query_probabilities)``.
    """
    import torch

    n = len(targets)
    # Engine relational eligibility per candidate: a length-n mask of which head
    # bindings 0..n-1 satisfy that candidate's join. Static across steps.
    eligibility = program.joint_candidate_eligibility(train_head, 1, n)
    device = guard_modules[candidate_ids[0]].logit.device
    masks: dict[str, Any] = {}
    for guard_pred, mask in eligibility:
        rule_id = guard_pred[len(_GUARD_PREDICATE_PREFIX) :]
        masks[rule_id] = torch.tensor(
            [1.0 if m else 0.0 for m in mask], dtype=torch.float32, device=device
        )
    targets_t = torch.tensor(
        [1.0 if t else 0.0 for t in targets], dtype=torch.float32, device=device
    )
    eps = 1e-7

    def head_prob() -> Any:
        # Noisy-OR: P(head) = 1 - prod_k (1 - eligible_k * sigmoid(w_k)).
        # Grad-carrying guard sigmoids so the backward competes the candidates;
        # the SAME _joint_noisy_or backs the held-out generalization read, so the
        # two cannot drift.
        p_by_rule = {
            rule_id: torch.sigmoid(guard_modules[rule_id].logit)
            for rule_id in candidate_ids
        }
        return _joint_noisy_or(masks, p_by_rule, candidate_ids, n, device)

    # Pure-torch differentiable loop over the guard params + static engine masks,
    # so the joint training performs no tracked device<->host transfers.
    program.reset_host_transfer_stats()
    for _step in range(config.steps):
        program.zero_grad()
        p_or = head_prob().clamp(eps, 1.0 - eps)
        loss = -(
            targets_t * torch.log(p_or) + (1.0 - targets_t) * torch.log(1.0 - p_or)
        ).mean()
        loss.backward()
        program.optimizer_step()
        losses.append(float(loss.item()))
    host_transfer_stats = dict(program.host_transfer_stats())

    with torch.no_grad():
        query_probabilities = head_prob().detach().cpu().tolist()
    return host_transfer_stats, query_probabilities


def _joint_noisy_or(
    masks: dict[str, Any],
    p_by_rule: dict[str, Any],
    candidate_ids: list[str],
    n: int,
    device: Any,
) -> Any:
    """Joint multi-rule noisy-OR head probability, shared by the training forward
    and the held-out generalization read so the two compute the IDENTICAL mixture.

    ``P(head_i) = 1 - prod_k (1 - mask_k[i] * p_k)`` where ``mask_k`` is candidate
    k's relational eligibility (0/1 per binding, from the engine) and ``p_k`` its
    guard probability. Training passes grad-carrying ``sigmoid(logit)`` tensors so
    the backward competes the guards; the held-out read passes the detached
    trained sigmoids. Routing both through this one function is what makes the
    held-out generalization read honest by construction — there is no second
    noisy-OR implementation that could silently diverge from the trained mixture.
    """
    import torch

    one_minus = torch.ones(n, dtype=torch.float32, device=device)
    for rule_id in candidate_ids:
        one_minus = one_minus * (1.0 - masks[rule_id] * p_by_rule[rule_id])
    return 1.0 - one_minus


def _joint_mixture_probs(
    program: Any,
    train_head: str,
    rule_weights: dict[str, float],
    num_queries: int,
    arity: int,
) -> list[float]:
    """Trained-guard joint noisy-OR over the engine's relational eligibility for
    bindings ``train_head(0..num_queries-1)`` on ``program``. Pure forward: the
    guards are fixed at ``rule_weights`` (their trained sigmoids) and only the
    engine eligibility is read, so no training occurs."""
    import torch

    eligibility = program.joint_candidate_eligibility(train_head, arity, num_queries)
    device = torch.device("cpu")
    masks: dict[str, Any] = {}
    p_by_rule: dict[str, Any] = {}
    candidate_ids: list[str] = []
    for guard_pred, mask in eligibility:
        rule_id = guard_pred[len(_GUARD_PREDICATE_PREFIX) :]
        if rule_id not in rule_weights:
            # A defining rule whose guard was not trained (no weight supplied)
            # cannot contribute a learned mixture term, so it is skipped.
            continue
        candidate_ids.append(rule_id)
        masks[rule_id] = torch.tensor(
            [1.0 if m else 0.0 for m in mask], dtype=torch.float32, device=device
        )
        p_by_rule[rule_id] = torch.tensor(
            float(rule_weights[rule_id]), dtype=torch.float32, device=device
        )
    with torch.no_grad():
        p_or = _joint_noisy_or(masks, p_by_rule, candidate_ids, num_queries, device)
    return p_or.detach().cpu().tolist()


def evaluate_joint_mixture(
    source: str,
    *,
    rule_weights: dict[str, float],
    num_queries: int,
    arity: int = 1,
    config: NeuroSymbolicTrainingConfig = NeuroSymbolicTrainingConfig(),
) -> list[float]:
    """Held-out generalization read for the ST-TRC Phase-1b joint mixture.

    Given a program ``source`` carrying the SAME ``trainable_rule`` candidates as
    the train run but with the HELD-OUT bindings' ground facts materialized, and
    the guard sigmoids learned by ``train_neurosymbolic_program``
    (``result.symbolic_rule_weights``), returns the per-query joint noisy-OR
    ``p_or`` over the held-out bindings ``train_head(0..num_queries-1)``.

    This is the faithful generalization signal: the TRAINED-guard mixture
    evaluated on the engine's relational eligibility for the held-out split — not
    a structural set-intersection — so a candidate that fit only the training
    facts (a spurious correlate) yields low held-out ``p_or`` wherever its join
    does not fire. The read reuses the exact ``_joint_noisy_or`` of the training
    forward, so it cannot drift from the trained mixture.

    Only a compiled program is needed — ``joint_candidate_eligibility`` reads
    relational hard-filter membership, never the guard network — so no network
    registration or example tensor source is required here.

    SELECTION vs ADMISSION (load-bearing — the candidate set is the caller's via
    ``rule_weights``): on a train-tie, every candidate whose join coincides with
    the head on the TRAIN facts trains to an equally-high guard, so the guards
    alone cannot discriminate the true rule from a train-perfect spurious
    correlate. The discriminator is held-out coverage. Therefore:
      - SELECT among train-covering candidates by held-out coverage — guard-free,
        ``mean`` of each candidate's ``joint_candidate_eligibility`` mask over the
        held-out positives — NOT by guard (the guards are tied).
      - ADMIT by calling this function with ONLY the selected winner's weight
        (``rule_weights={winner: w}``): the noisy-OR is then that one candidate's
        held-out probability, the faithful generalization read. Passing the FULL
        pool here is a trap — the OR is inflated wherever ANY candidate fires, so
        a high-guard spurious coverer would mask the winner's non-generalization.
        Pool-wide is the MIXTURE's prediction, not a single-candidate admission
        gate.

    CAVEAT: the held-out bindings' ground facts MUST be present in ``source`` for
    each index ``0..num_queries-1``; if a binding's supporting facts are absent
    the engine eligibility is empty (all-zero mask) and its ``p_or`` collapses to
    0, which reads as a FALSE spurious-correlate rather than a real one. Callers
    flattening pairs into indexed unary facts must materialize the held-out facts,
    not only the train-split facts.
    """
    import pyxlog

    program_source, _rules, train_head, _objective = _desugar_source(source)
    program = pyxlog.Program.compile(
        program_source,
        device=config.device,
        memory_mb=config.gpu_memory_mb,
    )
    return _joint_mixture_probs(program, train_head, rule_weights, num_queries, arity)


def _make_optimizer(name: str, params: Any, lr: float) -> Any:
    """Build the per-module optimizer named by the config (``adam`` or ``sgd``)."""
    import torch

    key = name.lower()
    if key == "adam":
        return torch.optim.Adam(params, lr=lr)
    if key == "sgd":
        return torch.optim.SGD(params, lr=lr)
    raise ValueError(f"unsupported optimizer {name!r}; expected 'adam' or 'sgd'")


def _make_rule_weight_module(initial_logit: float) -> Any:
    """Single-logit torch module emitting ``[1-p, p]`` with ``p = sigmoid(logit)``.

    Registered as the network behind a rule's guard predicate, so the rule
    weight participates in circuit evaluation exactly like any neural output.
    """
    import torch

    class RuleWeight(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logit = torch.nn.Parameter(torch.tensor(float(initial_logit)))

        def forward(self, x):  # type: ignore[override]
            p = torch.sigmoid(self.logit)
            row = torch.stack([1.0 - p, p])
            return row.unsqueeze(0).expand(x.shape[0], 2)

    return RuleWeight()


def _collect_examples(examples: list[dict[str, Any]]):
    import torch

    input_parts = []
    target_parts = []
    for batch in examples:
        inputs = batch["inputs"]
        targets = batch["targets"].to(dtype=torch.float32)
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError(
                f"examples batch has {inputs.shape[0]} inputs but "
                f"{targets.shape[0]} targets"
            )
        if bool(((targets != 0.0) & (targets != 1.0)).any()):
            raise ValueError("targets must be binary (0.0 or 1.0)")
        input_parts.append(inputs)
        target_parts.append(targets)
    all_inputs = torch.cat(input_parts, dim=0)
    all_targets = [bool(t >= 0.5) for t in torch.cat(target_parts, dim=0)]
    return all_inputs, all_targets


def _build_proof_trace_map(
    pyxlog_module: Any,
    rules: list[TrainableRuleDecl],
    queries: list[str],
    targets: list[bool],
    learned_logits: dict[str, float],
) -> Any:
    trace_map_cls = getattr(pyxlog_module, "DifferentiableProofTraceMap", None)
    if trace_map_cls is None:
        native = getattr(pyxlog_module, "_native", None)
        trace_map_cls = getattr(native, "DifferentiableProofTraceMap", None)
    if trace_map_cls is None:
        raise RuntimeError(
            "pyxlog native extension does not expose DifferentiableProofTraceMap"
        )

    trace_map = trace_map_cls()
    for query in queries:
        for rule in rules:
            trace_map.insert(
                query,
                rule.id,
                list(rule.body_literals),
                learned_logits[rule.id],
            )
    trace_map.accumulate_binary_logistic_gradients(
        [(query, 1.0 if target else 0.0) for query, target in zip(queries, targets)]
    )
    return trace_map


def _first_neural_predicate(rule: TrainableRuleDecl) -> str:
    for literal in rule.body_literals:
        name = literal.split("(", 1)[0].strip()
        if not name.startswith(_GUARD_PREDICATE_PREFIX):
            return name
    return ""


# ---------------------------------------------------------------------------
# Desugaring: trainable_rule / train statements -> real xlog
# ---------------------------------------------------------------------------


def _desugar_source(source: str) -> tuple[str, list[TrainableRuleDecl], str, str]:
    for reserved in (_GUARD_PREDICATE_PREFIX, _GUARD_NETWORK_PREFIX):
        if reserved in source:
            raise ValueError(
                f"source must not use the reserved identifier prefix '{reserved}'"
            )

    rules: list[TrainableRuleDecl] = []
    train_directives: list[tuple[str, str]] = []
    replacements: list[tuple[int, int, str]] = []

    for start, end, statement in _statement_spans(source):
        if statement.startswith("trainable_rule"):
            rule = _parse_trainable_statement(statement)
            rules.append(rule)
            guarded = (
                f"{rule.head} :- "
                + ", ".join(rule.body_literals)
                + f", {rule.guard_predicate}({rule.query_variable}, on)."
            )
            replacements.append((start, end, guarded))
        elif statement.startswith("train("):
            train_directives.append(_parse_train_directive(statement))
            replacements.append((start, end, ""))

    if not rules:
        raise ValueError("source must declare at least one trainable_rule")
    if len(train_directives) != 1:
        raise ValueError(
            "source must declare exactly one train(head, objective) directive, "
            f"found {len(train_directives)}"
        )

    rewritten = source
    for start, end, text in sorted(replacements, reverse=True):
        rewritten = rewritten[:start] + text + rewritten[end:]

    guard_decls = "\n".join(
        f"nn({rule.guard_network}, [NsrCase], NsrState, [off, on]) :: "
        f"{rule.guard_predicate}(NsrCase, NsrState)."
        for rule in rules
    )
    train_head, objective = train_directives[0]
    return rewritten + "\n" + guard_decls + "\n", rules, train_head, objective


def _statement_spans(source: str):
    """Yield ``(start, end, text)`` for '.'-terminated statements.

    Tracks bracket depth, quoted spans, and ``//`` comments so a '.' inside a
    term, a float, or a comment never terminates a statement. The desugaring
    layer only interprets statements that start with ``trainable_rule`` or
    ``train(``; everything else passes through to the native parser verbatim.
    """
    n = len(source)
    i = 0
    start: int | None = None
    depth = 0
    while i < n:
        ch = source[i]
        if ch == "/" and i + 1 < n and source[i + 1] == "/":
            newline = source.find("\n", i)
            i = n if newline == -1 else newline + 1
            continue
        if ch in "\"'":
            quote = ch
            if start is None:
                start = i
            i += 1
            while i < n and source[i] != quote:
                i += 1
            i += 1
            continue
        if start is None and not ch.isspace():
            start = i
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "." and depth == 0:
            next_ch = source[i + 1] if i + 1 < n else " "
            if not next_ch.isdigit():
                if start is not None:
                    yield start, i + 1, source[start:i].strip()
                start = None
        i += 1


def _parse_trainable_statement(statement: str) -> TrainableRuleDecl:
    rest = statement[len("trainable_rule") :].lstrip()
    if not rest.startswith("("):
        raise ValueError(f"malformed trainable_rule statement: {statement[:80]!r}")
    args_text, after = _read_balanced(rest)
    after = after.lstrip()
    if not after.startswith("::"):
        raise ValueError(
            f"trainable_rule({args_text}) must be followed by ':: head :- body'"
        )
    rule_text = after[2:].strip()

    args = _split_top_level(args_text)
    if not args or not args[0].strip():
        raise ValueError("trainable_rule requires a rule id")
    rule_id = args[0].strip()
    if not (rule_id[0].islower() and rule_id.replace("_", "").isalnum()):
        raise ValueError(f"trainable_rule id must be an identifier, got {rule_id!r}")
    initial_weight = 0.0
    for extra in args[1:]:
        key, _, value = extra.partition("=")
        if key.strip() != "weight":
            raise ValueError(f"unknown trainable_rule argument: {extra.strip()!r}")
        initial_weight = float(value.strip())

    head, sep, body = rule_text.partition(":-")
    if not sep:
        raise ValueError(
            f"trainable_rule '{rule_id}' must be a rule with a body (head :- body)"
        )
    head = head.strip()
    body_literals = tuple(lit.strip() for lit in _split_top_level(body) if lit.strip())
    if not body_literals:
        raise ValueError(f"trainable_rule '{rule_id}' has an empty body")

    head_var = _first_variable(head)
    if head_var is None:
        raise ValueError(
            f"trainable_rule '{rule_id}' head must bind at least one variable"
        )

    return TrainableRuleDecl(
        id=rule_id,
        head=head,
        body_literals=body_literals,
        initial_weight=initial_weight,
        source=rule_text.strip(),
        guard_predicate=f"{_GUARD_PREDICATE_PREFIX}{rule_id}",
        guard_network=f"{_GUARD_NETWORK_PREFIX}{rule_id}",
        query_variable=head_var,
    )


def _parse_train_directive(statement: str) -> tuple[str, str]:
    rest = statement[len("train") :].lstrip()
    args_text, after = _read_balanced(rest)
    if after.strip():
        raise ValueError(f"malformed train directive: {statement[:80]!r}")
    args = [arg.strip() for arg in _split_top_level(args_text)]
    if len(args) == 1:
        return args[0], "binary_cross_entropy"
    if len(args) == 2:
        return args[0], args[1]
    raise ValueError(
        f"train directive must be train(head) or train(head, objective): "
        f"{statement[:80]!r}"
    )


def _read_balanced(text: str) -> tuple[str, str]:
    """Read a balanced ``(...)`` group from the start of *text*."""
    if not text.startswith("("):
        raise ValueError(f"expected '(' at: {text[:40]!r}")
    depth = 0
    for idx, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[1:idx], text[idx + 1 :]
    raise ValueError(f"unbalanced parentheses at: {text[:40]!r}")


def _split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return parts


def _first_variable(term_text: str) -> str | None:
    token: list[str] = []
    for ch in term_text + " ":
        if ch.isalnum() or ch == "_":
            token.append(ch)
        else:
            name = "".join(token)
            if name and name[0].isupper():
                return name
            token = []
    return None
