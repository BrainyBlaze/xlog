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

import math
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
            name, module, torch.optim.SGD(module.parameters(), lr=config.learning_rate)
        )

    guard_modules: dict[str, Any] = {}
    for rule in rules:
        guard = _make_rule_weight_module(rule.initial_weight).cuda()
        guard_modules[rule.id] = guard
        program.register_network(
            rule.guard_network,
            guard,
            torch.optim.SGD(guard.parameters(), lr=config.learning_rate),
        )

    program.add_tensor_source(_TENSOR_SOURCE_NAME, inputs.cuda())

    queries = [f"{train_head}({i})" for i in range(len(targets))]
    losses: list[float] = []
    neural_grads: dict[str, float] = {name: 0.0 for name in modules}
    symbolic_grads: dict[str, float] = {rule.id: 0.0 for rule in rules}

    # Zero-host training hot loop. Every example's supervised circuit is
    # evaluated in one device-resident batched pass per step (grouped by target
    # and circuit template), so a step costs a single host sync for the summed
    # loss rather than one per query. Looping the scalar forward_backward
    # instead host-syncs on every query (.item()), which leaves the GPU idle
    # between syncs and makes training CPU-bound. Reset/read the provider's
    # device->host counter around the loop so the no-host property is observable.
    program.reset_host_transfer_stats()
    for _step in range(config.steps):
        program.zero_grad()
        step_loss = program.forward_backward_grouped(queries, targets)
        program.optimizer_step()
        losses.append(step_loss / len(targets))
    host_transfer_stats = program.host_transfer_stats()

    # Final gradient magnitudes, read once after training. These are per
    # parameter, not per query, so they stay out of the hot loop; optimizer_step
    # does not clear gradients (only zero_grad does), so after the last step they
    # still reflect the final backward pass.
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

    # Final evaluation pass: query probabilities from the trained circuit.
    program.zero_grad()
    query_probabilities = [
        math.exp(-program.forward_backward(query, True)) for query in queries
    ]
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
