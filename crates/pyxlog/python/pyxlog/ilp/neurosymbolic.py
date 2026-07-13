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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

from pyxlog.ilp.inventory import RuleInventory, RuleInventoryClause
from pyxlog.ilp.join_bodies import (
    domain_row_index,
    mentions_neural_on_nonhead_var,
    noisy_or_from_index,
    parse_join_body,
    prepare_extension,
    read_join_extension,
    translate_extension_to_rows,
)

_GUARD_PREDICATE_PREFIX = "nsr_guard_"
_GUARD_NETWORK_PREFIX = "nsr_w_"
_TENSOR_SOURCE_NAME = "nsr_examples"
# Stage-B existential join: per-event feature batch the join neural predicate is
# forwarded over. The driver owns this name and passes it to the engine via
# `register_domain_tensor_source`; the engine reads whichever source it was given
# (no hardcoded source name on the Rust side), so this is the single definition.
_DOMAIN_SOURCE_NAME = "nsr_domain"
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
class NeuralBodySpec:
    """A neural conjunct ``g_theta(phi(x)) >= tau`` attached to a candidate's body
    (the neural-bodied candidate shape: a relational body plus one learned gate).

    The candidate's eligibility becomes its relational grounding mask AND the
    straight-through-thresholded neural gate, so a head ``H(x) :- r_i(x) ^
    [g_theta(phi(x)) >= tau]`` competes in the SAME noisy-OR mixture as a
    guard-only relational candidate — its guard ``sigma(w_k)`` and the held-out
    selector are unchanged; only the eligibility gains the learned gate. Gradient
    flows to ``theta`` (and the guard), never to ``phi`` (an external entity
    feature; detached by default, so no gradient reaches phi -- coupling phi to a
    trainable backbone via LoRA is a separate downstream task).

    ``features`` is the per-binding entity feature ``phi(x)`` as a fixed-width
    tensor ``[num_queries, width]`` (the contract default is the mean-pooled
    pre-quantization VQ-RB feature); ``width`` is fixed at head construction.
    """

    features: Any  # torch.Tensor [num_queries, width] — phi(x) per binding
    threshold: float = 0.5  # tau_k: gate fires when sigmoid(g_theta) >= threshold
    head_depth: int = 1  # 1 = linear->scalar; >1 inserts tanh hidden layers
    hidden_dim: int = 16  # hidden width when head_depth > 1
    gumbel_temperature: float = 1.0  # straight-through softening temperature
    gumbel_noise: bool = False  # add Gumbel exploration noise during training
    # (default off: deterministic straight-through; on: ST-Gumbel exploration)
    train_within_set_norm: bool = False  # when set, train the gate's BACKWARD via
    # within_set_norm (offset-invariant within-comparison-set RANK) instead of the
    # absolute per-entity sigmoid; the FORWARD hard gate is unchanged. Default off:
    # byte-identical to the absolute _st_neural_gate path.


@dataclass
class NeuralBodyState:
    """The trained neural conjunct, serialized so the driver can rebuild the
    PARAMETRIC HardenedClause (it carries ``theta`` + the phi-extraction spec and
    re-evaluates ``g_theta`` per entity at apply time) and so the held-out read can
    reconstruct ``g_theta`` on held-out features."""

    state_dict: dict[str, Any]
    width: int
    threshold: float
    head_depth: int
    hidden_dim: int


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
    # Trained neural-body conjuncts, keyed by candidate rule id (only for
    # neural-bodied candidates). Each NeuralBodyState carries the learned g_theta
    # params + the phi/threshold spec so the driver can build the parametric
    # HardenedClause and the held-out read can reconstruct the gate.
    neural_body_state: dict[str, "NeuralBodyState"] = None


def train_neurosymbolic_program(
    source: str,
    *,
    networks: dict[str, Any],
    examples: list[dict[str, Any]],
    config: NeuroSymbolicTrainingConfig = NeuroSymbolicTrainingConfig(),
    neural_bodies: dict[str, "NeuralBodySpec"] | None = None,
    domain_inputs: dict[str, Any] | None = None,
    domain_ids: dict[str, Sequence[int]] | None = None,
    candidate_masses: dict[str, Any] | None = None,
) -> NeuroSymbolicTrainingResult:
    """Jointly train neural predicates and symbolic rule weights on the engine.

    ``neural_bodies`` attaches a neural conjunct
    ``g_theta_k(phi(x)) >= tau_k`` to a same-head candidate (keyed by its
    ``trainable_rule`` id) for the joint multi-rule mixture: that candidate's
    eligibility becomes its relational grounding AND the ST-thresholded gate, and
    ``g_theta_k`` trains jointly with the guards under the same held-out selector.
    Only meaningful in the multi-rule joint path (>1 same-head candidate).

    ``candidate_masses`` supplies graded per-binding confidences for joint-mixture
    candidates: ``{trainable_rule id: tensor[len(targets)]}`` with values in
    [0, 1]. A candidate's eligibility becomes its relational grounding TIMES its
    graded mass, so the head probability is
    ``P(head_i) = 1 - prod_k (1 - rel_k[i] * mass_k[i] * sigmoid(w_k))`` — the
    noisy-OR over graded evidence masses rather than binary grounding alone.
    This is the trajectory-supervision channel: mapping head bindings to world
    steps and masses to a fact's per-step confidence (e.g. a version-chain walk)
    trains the guards against an evolving world. Omitted candidates keep their
    binary relational mask; omitting the argument entirely leaves every code
    path identical to before. Only meaningful in the multi-rule joint path.

    ``domain_inputs`` is the Stage-B existential-join channel: a per-network
    ``{net_name: features}`` map where ``features`` is a ``[n_domain_constants, k]``
    tensor — one row per join-domain constant. When a ``trainable_rule`` joins a
    neural predicate to an ordinary relation on an existential variable, the
    predicate is grounded over this real domain and OR-aggregated at the head; the
    ``examples`` then carry ONLY per-head-binding ``targets`` (no per-query
    ``inputs``). The head-binding (e.g. edge) ids must be ``0..len(targets)-1``,
    row-aligned with ``targets``. Currently supports a single join network.

    ``domain_ids`` says WHICH CONSTANT EACH ROW HOLDS: ``domain_ids[net][j]`` is the
    domain constant whose feature vector is row ``j`` of ``domain_inputs[net]``. The
    tensor carries no labels, so that correspondence is a convention, and it must be
    stated rather than inferred — inferring it is exactly what went wrong before: the
    exact d-DNNF circuit read row ``j`` as the ``j``-th constant of the join
    relation's OWN domain (rank indexing) while the torch join path read the row the
    caller assigned to that constant. The ids are therefore handed to the ENGINE too,
    and BOTH paths now resolve a constant to its row through this one list. It is the
    only map; there is no ordering left for either side to infer.

    The ids must be strictly increasing (one row per constant, in a stable ascending
    layout) — a requirement on the caller's tensor, not the thing that keeps the two
    paths in step.

    Omitting ``domain_ids`` defaults it to ``[0, 1, ..., D-1]`` per network — the
    dense identity, i.e. exactly the behaviour every existing caller already relies
    on. Every constant the join relation's extension mentions must appear in the
    network's ids; one that does not is named in a ``ValueError`` (it has no feature
    row at all). The ids may cover MORE constants than the relation joins (features
    for every event, though only some are joined): the row is looked up by constant,
    so the unjoined ones simply go unread. ``domain_ids`` never supplies the join
    STRUCTURE: which constants a head binding joins is read from the engine's
    relation, always.
    """

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

    if inputs is not None:
        program.add_tensor_source(_TENSOR_SOURCE_NAME, inputs.cuda())

    # Stage-B existential join: register the per-event feature batch the join
    # neural predicate is grounded over. A single join network is supported.
    domain_inputs = domain_inputs or {}
    if len(domain_inputs) > 1:
        raise ValueError(
            "domain_inputs currently supports a single join network; "
            f"got {sorted(domain_inputs)}"
        )
    # Which ROW of domain_inputs[net] holds which domain CONSTANT. Stated by the
    # caller, never inferred; defaulted to the dense identity, which is what every
    # caller written before this parameter existed already meant.
    domain_ids = _resolve_domain_ids(domain_ids, domain_inputs)

    for name, feats in domain_inputs.items():
        if name not in declared:
            raise ValueError(
                f"domain_inputs names network '{name}' which is not declared by any nn/4"
            )
        # The ids go DOWN TO THE ENGINE, so the exact circuit resolves a joined
        # constant to a feature row through the very same list the torch-side mixture
        # uses. One map, two engines: they cannot disagree about which row is whose.
        program.register_domain_tensor_source(
            _DOMAIN_SOURCE_NAME, feats.cuda(), domain_ids[name]
        )

    # Neural-body conjuncts: one small g_theta head per neural-bodied
    # candidate, over its fixed-width phi(x). Trained torch-side (not a circuit
    # predicate), so the heads carry their own optimizers stepped alongside the
    # guards. phi width is fixed here at construction.
    neural_bodies = neural_bodies or {}
    # ``neural_parameter_grads`` is a single flat map keyed BOTH by nn/4 network name
    # (join-body / gate networks) and by candidate rule id (NeuralBodySpec heads). A
    # network named exactly like a neural-bodied candidate would therefore have its
    # gradient silently overwritten by that candidate's. The two namespaces are the
    # user's to choose, so refuse the overlap rather than report a wrong number.
    grad_key_collisions = sorted(set(modules) & set(neural_bodies))
    if grad_key_collisions:
        raise ValueError(
            "neural_parameter_grads is keyed by both nn/4 network name and "
            "neural-bodied candidate id, so these names must not collide; "
            f"rename one of: {grad_key_collisions}"
        )
    neural_modules: dict[str, Any] = {}
    neural_optims: dict[str, Any] = {}
    for rule_id, spec in neural_bodies.items():
        width = int(spec.features.shape[-1])
        head = _make_neural_body_head(width, spec.head_depth, spec.hidden_dim).cuda()
        neural_modules[rule_id] = head
        neural_optims[rule_id] = _make_optimizer(
            config.optimizer, head.parameters(), config.learning_rate
        )

    queries = [f"{train_head}({i})" for i in range(len(targets))]
    losses: list[float] = []
    neural_grads: dict[str, float] = {name: 0.0 for name in modules}
    symbolic_grads: dict[str, float] = {rule.id: 0.0 for rule in rules}

    # Joint multi-rule path: when MORE THAN ONE trainable rule derives the train head,
    # this is the joint multi-rule same-head soft-mixture — the candidates compete
    # for mass on one head. Route to the joint noisy-OR forward; a single defining
    # rule keeps the (faster, circuit) single-rule path below.
    candidate_ids = [
        rule.id
        for rule in rules
        if rule.head.split("(", 1)[0].strip() == train_head
    ]
    if candidate_masses and len(candidate_ids) <= 1:
        raise ValueError(
            "candidate_masses requires the multi-rule joint mixture (more than "
            "one same-head trainable_rule candidate)"
        )
    if candidate_masses:
        unknown = sorted(set(candidate_masses) - set(candidate_ids))
        if unknown:
            raise ValueError(
                f"candidate_masses names unknown candidates: {unknown}"
            )
    if len(candidate_ids) > 1:
        host_transfer_stats, query_probabilities = _train_joint_mixture(
            program,
            train_head,
            targets,
            candidate_ids,
            guard_modules,
            config,
            losses,
            neural_modules=neural_modules,
            neural_specs=neural_bodies,
            neural_optims=neural_optims,
            candidate_masses=candidate_masses,
            rules=rules,
            modules=modules,
            domain_inputs=domain_inputs,
            domain_ids=domain_ids,
            join_read_source=_read_only_source(source),
        )
        for rule in rules:
            grad = guard_modules[rule.id].logit.grad
            symbolic_grads[rule.id] = (
                float(grad.detach().abs().item()) if grad is not None else 0.0
            )
        # nn/4 network gradients. A neural-JOIN candidate's network is forwarded
        # inside the mixture's own head_prob, so its parameters receive gradient
        # from the joint loss exactly as in the single-rule circuit path; a network
        # no candidate uses keeps its 0.0 baseline (grad is None).
        for name, module in modules.items():
            neural_grads[name] = float(
                sum(
                    param.grad.detach().abs().sum().item()
                    for param in module.parameters()
                    if param.grad is not None
                )
            )
        # Neural-head gradient magnitudes, read once after training (proof the
        # neural conjunct received gradient through the ST gate).
        for rule_id, head in neural_modules.items():
            neural_grads[rule_id] = float(
                sum(
                    param.grad.detach().abs().sum().item()
                    for param in head.parameters()
                    if param.grad is not None
                )
            )
    elif neural_bodies:
        raise ValueError(
            "neural_bodies requires the multi-rule joint mixture (more than one "
            "same-head trainable_rule candidate); a single defining rule has no "
            "joint competition to attach a neural conjunct to"
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

    # Serialize each trained neural conjunct so the driver can rebuild the
    # parametric HardenedClause (theta + phi/threshold spec) and the held-out read
    # can reconstruct g_theta.
    neural_body_state: dict[str, NeuralBodyState] = {}
    for rule_id, head in neural_modules.items():
        spec = neural_bodies[rule_id]
        neural_body_state[rule_id] = NeuralBodyState(
            state_dict={k: v.detach().cpu() for k, v in head.state_dict().items()},
            width=int(spec.features.shape[-1]),
            threshold=spec.threshold,
            head_depth=spec.head_depth,
            hidden_dim=spec.hidden_dim,
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
        neural_body_state=neural_body_state or None,
    )


def _train_joint_mixture(
    program: Any,
    train_head: str,
    targets: list[bool],
    candidate_ids: list[str],
    guard_modules: dict[str, Any],
    config: NeuroSymbolicTrainingConfig,
    losses: list[float],
    neural_modules: dict[str, Any] | None = None,
    neural_specs: dict[str, "NeuralBodySpec"] | None = None,
    neural_optims: dict[str, Any] | None = None,
    candidate_masses: dict[str, Any] | None = None,
    rules: list[TrainableRuleDecl] | None = None,
    modules: dict[str, Any] | None = None,
    domain_inputs: dict[str, Any] | None = None,
    domain_ids: dict[str, Sequence[int]] | None = None,
    join_read_source: str | None = None,
) -> tuple[dict[str, int], list[float]]:
    """Joint soft-mixture over same-head candidates (guard-only, neural-bodied
    when a candidate carries a neural conjunct, or neural-JOIN when a candidate
    puts a neural predicate on an existential join variable).

    N candidate rules derive the SAME head, each gated by its own guard. The head
    probability is the noisy-OR over candidates of (eligible_k x guard sigmoid);
    BCE on the supervised head drives the per-candidate competition. A candidate's
    ``eligible_k`` is its engine relational eligibility (``joint_candidate_eligibility``,
    static) AND — for a neural-bodied candidate — the ST-thresholded neural gate
    ``[g_theta_k(phi_k) >= tau_k]`` (recomputed each step, gradient to theta_k).
    The differentiable mass (guard sigmoids, neural gates, OR, BCE) is torch over
    the guard + head params; phi is external + detached, so the loop performs no
    tracked device<->host transfers. The hard threshold keeps the neural conjunct
    a derivation GATE (not soft truth-mass), so it composes in the noisy-OR without
    a circuit leaf.

    A neural-JOIN candidate (``plastic(E) :- saliency(Ev, strengthen),
    pre_before_post(Ev, E).``) has NO boolean eligibility to speak of: the engine's
    ``joint_candidate_eligibility`` emits no mask for it, because ``Ev`` is
    existential and the candidate's relational truth is the OR, over the join
    extension, of the network's PER-EVENT probability. Such a candidate's mask is
    therefore ``1 - prod_{e in ext_k(h)} (1 - p_net(e))`` — see the join-body block
    below and :func:`join_bodies.noisy_or_from_index`.

    Returns ``(host_transfer_stats, query_probabilities)``.
    """
    import torch

    neural_modules = neural_modules or {}
    neural_specs = neural_specs or {}
    neural_optims = neural_optims or {}
    rules = rules or []
    modules = modules or {}
    domain_inputs = domain_inputs or {}
    domain_ids = _resolve_domain_ids(domain_ids, domain_inputs)

    n = len(targets)

    # --- neural JOIN bodies (Stage-B shape) ---------------------------------
    # A candidate whose neural predicate sits on an EXISTENTIAL join variable has no
    # boolean relational eligibility to speak of: its relational truth is the OR, over
    # the join extension, of the network's per-event probability. Recognize that from
    # the RULE (the rule names the join relation; nn/4 names the network) — never from
    # the caller, and never from what the engine did or did not hand back.
    #
    # ROUTING IS DECIDED BY THE RULE, NOT BY THE MASK, and that is load-bearing. Today
    # such a candidate happens to get NO mask out of joint_candidate_eligibility (its
    # mask is keyed by the first neural group's predicate, which for a join candidate
    # yields a key the guard-prefix strip turns to junk), so "no mask" alone would
    # appear to be a sufficient trigger. But that is a property of the engine's current
    # keying, not of the rule: were the keying fixed upstream, a join candidate would
    # arrive WITH a hard-filters-only (all-True) mask and be trained as an always-true
    # relational candidate — no gradient to the detector, no error, no failing test.
    # So: any candidate that puts a declared nn/4 predicate on a non-head variable MUST
    # be owned by the join path or be rejected here; it is never a plain relational one.
    #
    # The supported shape is EXACTLY {one neural atom, one join relation} — two BODY
    # LITERALS, each a bare positive atom. Anything else (a further conjunct, a
    # negation, a comparison, an `is`, a modal) is a DIFFERENT rule for which this
    # module has no mask, and is rejected with a typed error rather than silently
    # reduced to the shape it resembles. This check runs BEFORE the engine is asked for
    # eligibility, so the diagnostic names the offending candidate and its body rather
    # than surfacing whatever the engine says about the part it cannot compile.
    join_bodies: dict[str, Any] = {}
    join_index: dict[str, Any] = {}
    join_label_index: dict[str, int] = {}
    rule_by_id = {rule.id: rule for rule in rules}
    neural_predicates = _neural_predicate_networks(program, rules)

    def _rule(rule_id: str) -> TrainableRuleDecl:
        rule = rule_by_id.get(rule_id)
        if rule is None:      # candidate_ids are derived FROM rules; belt and braces
            raise ValueError(
                f"candidate '{rule_id}' has no trainable_rule declaration"
            )
        return rule

    def _reject(rule: TrainableRuleDecl) -> NoReturn:
        raise ValueError(
            f"trainable_rule '{rule.id}' cannot be trained as a plain relational "
            "candidate (it puts a neural predicate on a non-head variable, and/or it "
            "has no relational eligibility mask), and its body is not the supported "
            "neural-join shape (EXACTLY two literals, each a bare positive atom: one "
            "neural atom on an existential variable, plus the one ordinary relation "
            "that joins that variable to the head; a negation, a comparison, an 'is' "
            "expression, a modal literal or any further conjunct is a different rule). "
            "An unsupported body is rejected, not silently reduced to a shorter one: "
            + ", ".join(rule.body_literals)
        )

    parsed_join: dict[str, Any] = {}
    for rule_id in candidate_ids:
        rule = _rule(rule_id)
        if not mentions_neural_on_nonhead_var(
            list(rule.body_literals), neural_predicates, rule.query_variable
        ):
            continue                     # not a join candidate; the engine's mask rules
        jb = parse_join_body(
            list(rule.body_literals), neural_predicates, rule.query_variable
        )
        if jb is None:
            _reject(rule)                # a join candidate we cannot mask -> refused
        parsed_join[rule_id] = jb

    # Engine relational eligibility per candidate: a length-n mask of which head
    # bindings 0..n-1 satisfy that candidate's relational grounding. Static.
    eligibility = program.joint_candidate_eligibility(train_head, 1, n)
    device = guard_modules[candidate_ids[0]].logit.device
    rel_masks: dict[str, Any] = {}
    for guard_pred, mask in eligibility:
        rule_id = guard_pred[len(_GUARD_PREDICATE_PREFIX) :]
        rel_masks[rule_id] = torch.tensor(
            [1.0 if m else 0.0 for m in mask], dtype=torch.float32, device=device
        )

    # The join path owns (a) every candidate the rule says is a join candidate, whatever
    # mask the engine emitted for it, and (b) every candidate the engine left unmasked —
    # which, not being a join candidate by (a), has no mask at all and is rejected.
    needs_join = [
        rule_id
        for rule_id in candidate_ids
        if rule_id in parsed_join or rule_id not in rel_masks
    ]
    if needs_join:
        ilp_read = None
        for rule_id in needs_join:
            rule = _rule(rule_id)
            jb = parsed_join.get(rule_id)
            if jb is None:
                _reject(rule)
            if jb.network not in modules:
                raise ValueError(
                    f"trainable_rule '{rule_id}' joins neural predicate "
                    f"'{jb.neural_predicate}' (network '{jb.network}'), which has no "
                    "registered network"
                )
            if jb.network not in domain_inputs:
                raise ValueError(
                    f"trainable_rule '{rule_id}' puts neural predicate "
                    f"'{jb.neural_predicate}' on the existential join variable "
                    f"'{jb.join_var}', so network '{jb.network}' must be forwarded over "
                    f"the join domain: pass domain_inputs={{'{jb.network}': features}} "
                    f"(and domain_ids={{'{jb.network}': ids}} unless the constants are "
                    "the dense range 0..D-1)"
                )
            if ilp_read is None:
                ilp_read = _open_join_read_handle(join_read_source, config)
            join_bodies[rule_id] = jb
            # The engine hands the extension back in RAW domain constants; the network's
            # per-constant probabilities are ROWS of domain_inputs. `domain_ids` states
            # which row holds which constant, and this is the ONE place the two are
            # reconciled — everything downstream speaks rows. A constant with no row is
            # named in a typed error here (it used to be a CUDA out-of-bounds index, or,
            # on a padded tensor, silently the wrong row).
            extension = translate_extension_to_rows(
                read_join_extension(ilp_read, jb, n),
                domain_ids[jb.network],
                network=jb.network,
                rule_id=rule_id,
            )
            # The extension is STATIC: flatten it into device tensors ONCE, here,
            # outside the step loop, so the per-step OR is a single gather +
            # segmented sum with no host->device copy (see JoinExtensionIndex).
            join_index[rule_id] = prepare_extension(extension, device)
            # Which output column is the network's "positive" probability is not
            # ours to guess: the rule's neural atom names the label
            # (``saliency(Ev, strengthen)``) and the ENGINE resolves it against the
            # nn/4 label list.
            join_label_index[rule_id] = int(
                program.label_to_index(
                    jb.neural_predicate, _neural_atom_label(rule, jb.neural_predicate)
                )
            )
            # The binding-level relational mask: 1 where the join extension is
            # non-empty. The OR over the extension is 0 there anyway (an OR over
            # nothing is false); keeping the mask separate is what lets
            # candidate_masses fold graded evidence onto a join candidate too. If the
            # engine DID hand back an eligibility mask for this candidate (it does not
            # today; it would after the upstream keying fix), that mask carries the
            # hard filters and is kept — multiplied, never overwritten and never used
            # on its own.
            extension_mask = torch.tensor(
                [1.0 if events else 0.0 for events in extension],
                dtype=torch.float32,
                device=device,
            )
            engine_mask = rel_masks.get(rule_id)
            rel_masks[rule_id] = (
                extension_mask if engine_mask is None else engine_mask * extension_mask
            )

    # Graded per-binding confidences fold multiplicatively onto the relational
    # eligibility: a candidate contributes mass_k[i] * sigmoid(w_k) where it is
    # relationally grounded, zero elsewhere. Values must already be in [0, 1];
    # candidates absent from the map keep their binary mask.
    for rule_id, masses in (candidate_masses or {}).items():
        masses_t = masses.detach().to(device=device, dtype=torch.float32)
        if masses_t.shape != (n,):
            raise ValueError(
                f"candidate_masses['{rule_id}'] must have one value per head "
                f"binding ({n}); got shape {tuple(masses_t.shape)}"
            )
        if bool((masses_t < 0.0).any()) or bool((masses_t > 1.0).any()):
            raise ValueError(
                f"candidate_masses['{rule_id}'] values must lie in [0, 1]"
            )
        rel_masks[rule_id] = rel_masks[rule_id] * masses_t
    targets_t = torch.tensor(
        [1.0 if t else 0.0 for t in targets], dtype=torch.float32, device=device
    )
    # Move each neural candidate's phi(x) to device ONCE (it is static; only
    # theta changes across steps). Detached: no backbone gradient by default.
    device_phi = {
        rule_id: neural_specs[rule_id].features.detach().to(
            device=device, dtype=torch.float32
        )
        for rule_id in neural_modules
    }
    # The join domain (per-event features) moved to device ONCE: it is static, only
    # the network's theta changes across steps.
    device_domain = {
        name: feats.detach().to(device=device, dtype=torch.float32)
        for name, feats in domain_inputs.items()
    }
    eps = 1e-7

    def head_prob(training: bool) -> Any:
        # Noisy-OR: P(head) = 1 - prod_k (1 - eligible_k * sigmoid(w_k)).
        # eligible_k = relational grounding mask, AND the ST neural gate for a
        # neural-bodied candidate (recomputed each call so theta_k gets gradient
        # and the gate tracks the current head). The SAME _joint_noisy_or backs the
        # held-out generalization read, so training and read cannot drift.
        masks: dict[str, Any] = {}
        for rule_id in candidate_ids:
            if rule_id in join_bodies:
                # Neural JOIN body. The network scores EVERY event of the domain;
                # the LOGIC's join extension (read from the engine) says which
                # events belong to this head binding; the OR aggregates them. This
                # is not a gate — it IS the candidate's relational truth, so its
                # gradient reaches the per-event detector.
                jb = join_bodies[rule_id]
                p_event = modules[jb.network](device_domain[jb.network])[
                    :, join_label_index[rule_id]
                ]
                masks[rule_id] = rel_masks[rule_id] * noisy_or_from_index(
                    p_event, join_index[rule_id]
                )
            elif rule_id in neural_modules:
                spec = neural_specs[rule_id]
                gate = _neural_gate_for(
                    neural_modules[rule_id](device_phi[rule_id]),
                    spec,
                    n,
                    device,
                    training,
                )
                masks[rule_id] = rel_masks[rule_id] * gate
            else:
                masks[rule_id] = rel_masks[rule_id]
        p_by_rule = {
            rule_id: torch.sigmoid(guard_modules[rule_id].logit)
            for rule_id in candidate_ids
        }
        return _joint_noisy_or(masks, p_by_rule, candidate_ids, n, device)

    # Pure-torch differentiable loop over guard + neural-head params + static
    # engine masks, so the joint training performs no tracked device<->host
    # transfers. The neural-head optimizers are stepped alongside the guards'
    # (the program owns the guard optimizers; the heads are torch-side).
    program.reset_host_transfer_stats()
    for _step in range(config.steps):
        program.zero_grad()
        for opt in neural_optims.values():
            opt.zero_grad()
        p_or = head_prob(training=True).clamp(eps, 1.0 - eps)
        loss = -(
            targets_t * torch.log(p_or) + (1.0 - targets_t) * torch.log(1.0 - p_or)
        ).mean()
        loss.backward()
        program.optimizer_step()
        for opt in neural_optims.values():
            opt.step()
        losses.append(float(loss.item()))
    host_transfer_stats = dict(program.host_transfer_stats())

    with torch.no_grad():
        query_probabilities = head_prob(training=False).detach().cpu().tolist()
    return host_transfer_stats, query_probabilities


def _read_only_source(source: str) -> str:
    """The program with ONLY the training sugar (``trainable_rule`` / ``train``)
    removed. Facts, ``pred`` declarations, ``nn`` declarations and any ordinary
    rules are real xlog and stay — this is the logic program itself, minus the two
    statements the native parser does not know."""
    spans = [
        (start, end)
        for start, end, statement in _statement_spans(source)
        if statement.startswith("trainable_rule") or statement.startswith("train(")
    ]
    out = source
    for start, end in sorted(spans, reverse=True):
        out = out[:start] + out[end:]
    return out


def _open_join_read_handle(join_read_source: str | None, config: Any) -> Any:
    """A SECOND, read-only handle on the same program, used ONLY to enumerate the
    deterministic ground join relation.

    This is a deliberate, load-bearing choice, not an accident. The object the
    mixture trains on is ``pyxlog.Program.compile(...)`` -> ``CompiledProgram``,
    whose entire read surface is ``evaluate*``: it does not expose facts at all
    (``EvalResult.atoms`` comes back empty for ground facts). ``IlpProgramFactory``
    -> ``CompiledIlpProgram`` does, via ``relation_facts(rel)``. So to let the
    ENGINE answer "which events does this head binding join?" — rather than accept
    an edge->events map from the caller, which would make the OR Python's and the
    claim that the LOGIC aggregates the join hollow — we compile the same source a
    second time through the ILP factory and read it. Nothing is ever trained on
    this handle; it is evaluated once and only enumerated.

    The desugared source cannot be used here (its guard-carrying rules do not
    survive the ILP plan builder's schema unification), which is why the handle is
    compiled from :func:`_read_only_source` — the user's program with the two
    training statements removed and everything else, including the facts and any
    ordinary rules that DERIVE the join relation, intact.
    """
    import pyxlog

    if join_read_source is None:
        raise ValueError(
            "a neural-join candidate needs the program source to read its join "
            "extension from the engine, but none was threaded through"
        )
    handle = pyxlog.IlpProgramFactory.compile(
        join_read_source,
        device=config.device,
        memory_mb=config.gpu_memory_mb,
    )
    handle.evaluate()
    return handle


def _neural_predicate_networks(
    program: Any, rules: list[TrainableRuleDecl]
) -> dict[str, str]:
    """``neural predicate -> network`` for every predicate a candidate body mentions,
    asked of the ENGINE (``neural_predicate_info``, populated from the nn/4
    declarations it compiled) rather than re-parsed out of the source text.
    Non-neural body predicates simply are not in the registry and are skipped."""
    mapping: dict[str, str] = {}
    for rule in rules:
        for literal in rule.body_literals:
            name = literal.split("(", 1)[0].strip()
            if name in mapping:
                continue
            try:
                info = program.neural_predicate_info(name)
            except ValueError:
                continue  # not a neural predicate
            mapping[name] = str(info["network"])
    return mapping


def _resolve_domain_ids(
    domain_ids: dict[str, Sequence[int]] | None, domain_inputs: dict[str, Any]
) -> dict[str, list[int]]:
    """Validate the caller's ``domain_ids`` and default the ones it omits.

    ``domain_ids[net][j]`` is the domain constant whose feature vector is row ``j`` of
    ``domain_inputs[net]``. The default is the dense identity ``0..D-1`` — which is
    what a caller written before this parameter existed already meant, so omitting it
    reproduces the previous behaviour exactly.

    What makes the two engines agree is that the resolved ids are registered WITH the
    domain tensor, so the exact circuit looks a joined constant up in the same list the
    torch-side mixture does. The strictly-increasing check (:func:`domain_row_index`) is
    a layout requirement on the caller's tensor, not the reconciliation.
    """
    supplied = dict(domain_ids or {})
    unknown = sorted(set(supplied) - set(domain_inputs))
    if unknown:
        raise ValueError(
            f"domain_ids names network(s) {unknown} with no domain_inputs; "
            "domain_ids says which row of domain_inputs[net] holds which domain "
            "constant, so every key must also be a key of domain_inputs"
        )
    resolved: dict[str, list[int]] = {}
    for name, feats in domain_inputs.items():
        rows = int(feats.shape[0])
        ids = [int(c) for c in supplied[name]] if name in supplied else list(range(rows))
        if len(ids) != rows:
            raise ValueError(
                f"domain_ids['{name}'] has {len(ids)} id(s) but domain_inputs['{name}'] "
                f"has {rows} row(s); there must be exactly one id per row (id j names the "
                "domain constant whose feature vector is row j)"
            )
        domain_row_index(ids, name)      # strictly increasing, or a typed refusal
        resolved[name] = ids
    return resolved


def _neural_atom_label(rule: TrainableRuleDecl, predicate: str) -> str:
    """The label term the rule's neural atom asks for (``saliency(Ev, strengthen)``
    -> ``strengthen``): the LAST argument, since an nn/4 predicate is
    ``p(Inputs..., Label)``."""
    for literal in rule.body_literals:
        if literal.split("(", 1)[0].strip() != predicate:
            continue
        args = _split_top_level(literal.split("(", 1)[1].rsplit(")", 1)[0])
        if len(args) >= 2:
            return args[-1].strip()
    raise ValueError(
        f"trainable_rule '{rule.id}' has no labelled '{predicate}' atom in its body"
    )


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


def within_set_norm(g_theta, group_id, *, mode, temp=1.0):
    """Within-comparison-set normalization of a head logit into a [0,1] WMC mass.

    The within-comparison-set normalization circuit substrate. Each entity's
    raw head logit ``g_theta`` is normalized RELATIVE TO its comparison set (the
    entities sharing its ``group_id``), de-saturating the graded gate and cancelling
    the per-set absolute offset that does NOT transfer train->held-out. Only the
    within-set RANK transfers (proven), and both realizations below are monotone in
    ``g_theta`` within a group, hence RANK-IDENTICAL (a surrogate<->exact pair; only
    the numeric mass differs):

    - ``mode="train"``: z-norm then ``sigmoid(z/temp)`` -- differentiable, so the
      gradient reaches theta. The set mean is subtracted, so an additive set-wide
      offset (the non-transferring component) cancels EXACTLY and the bias receives
      no gradient through this op; only the within-set rank-bearing signal drives
      training. This is the substrate/train realization.
    - ``mode="eval"``: within-group rank-percentile in (0,1) (tie-averaged) --
      exact, bounded, robust to outliers; non-differentiable (read/eval realization).

    The result is a length-N mass in [0,1] suitable as a noisy-OR / WMC leaf
    (``masks_graded = rel * within_set_norm`` -> the unchanged :func:`_joint_noisy_or`).

    ``group_id`` is a per-entity comparison-set index: for admission the whole
    held-out query set is one group; for context-relative firing the grouping comes
    from the per-context ``context_id`` sidecar. Degenerate groups (``|group| <= 1`` or zero
    within-set spread) carry no within-set signal and return the neutral ``0.5`` --
    the cardinality fence (``n < 16`` fail-closed for firing) is a CALLER concern,
    not this pure-math helper. The reduction is order-invariant (set statistics);
    ordered comparison-set emission for recompute-from-raw is the read-side's job.
    ``tau`` is NOT used here: within-set rank + the top-k% firing rule supersede the
    ``sigmoid(g_theta - tau_logit)`` threshold (the absolute quantity that does not
    transfer).
    """
    import torch

    if mode not in ("train", "eval"):
        raise ValueError(
            f"within_set_norm mode must be 'train' or 'eval', got {mode!r}"
        )
    g = g_theta.reshape(-1).to(dtype=torch.float32)
    groups = group_id.reshape(-1)
    if g.shape[0] != groups.shape[0]:
        raise ValueError(
            "within_set_norm: g_theta and group_id must have equal length, got "
            f"{g.shape[0]} vs {groups.shape[0]}"
        )
    n = g.shape[0]
    neutral = torch.full_like(g, 0.5)
    if n == 0:
        return neutral

    if mode == "train":
        uniq, inv = torch.unique(groups, return_inverse=True)
        ngrp = uniq.shape[0]
        ones = torch.ones(n, dtype=torch.float32, device=g.device)
        count = torch.zeros(ngrp, dtype=torch.float32, device=g.device).scatter_add(
            0, inv, ones
        )
        gsum = torch.zeros(ngrp, dtype=torch.float32, device=g.device).scatter_add(
            0, inv, g
        )
        mean_e = (gsum / count)[inv]
        var_e = (
            torch.zeros(ngrp, dtype=torch.float32, device=g.device).scatter_add(
                0, inv, (g - mean_e) ** 2
            )
            / count
        )[inv]
        std_e = torch.sqrt(var_e + 1e-12)
        count_e = count[inv]
        mass = torch.sigmoid((g - mean_e) / (std_e * max(float(temp), 1e-6)))
        valid = (count_e > 1.0) & (std_e > 1e-6)
        return torch.where(valid, mass, neutral)

    # mode == "eval": within-group rank-percentile in (0,1), tie-averaged. Exact,
    # bounded, robust; non-differentiable (read-side only). O(N^2) over the small
    # per-cell N typical of this domain.
    same = groups.reshape(-1, 1) == groups.reshape(1, -1)
    gi = g.reshape(-1, 1)
    gj = g.reshape(1, -1)
    less = ((gj < gi) & same).sum(dim=1).to(dtype=torch.float32)
    equal = ((gj == gi) & same).sum(dim=1).to(dtype=torch.float32)
    size = same.sum(dim=1).to(dtype=torch.float32)
    rank_pct = (less + 0.5 * equal) / size
    return torch.where(size > 1.0, rank_pct, neutral)


def _joint_mixture_probs(
    program: Any,
    train_head: str,
    rule_weights: dict[str, float],
    num_queries: int,
    arity: int,
    neural_heldout: dict[str, Any] | None = None,
) -> list[float]:
    """Trained-guard joint noisy-OR over the engine's relational eligibility for
    bindings ``train_head(0..num_queries-1)`` on ``program``. Pure forward: the
    guards are fixed at ``rule_weights`` (their trained sigmoids) and only the
    engine eligibility is read, so no training occurs.

    For a neural-bodied candidate, ``neural_heldout[rule_id] = (NeuralBodyState,
    held_out_features)``: its eligibility is the relational grounding mask AND the
    HARD (deterministic, no Gumbel) ST gate of the trained ``g_theta`` over the
    held-out features — so an overfit neural predicate fails the held-out gate
    exactly as a spurious relational correlate fails its join, inheriting the same
    vigilance safety net."""
    import torch

    neural_heldout = neural_heldout or {}
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
        rel = torch.tensor(
            [1.0 if m else 0.0 for m in mask], dtype=torch.float32, device=device
        )
        if rule_id in neural_heldout:
            state, features = neural_heldout[rule_id]
            head = _make_neural_body_head(state.width, state.head_depth, state.hidden_dim)
            head.load_state_dict(state.state_dict)
            head.eval()
            with torch.no_grad():
                phi = features.detach().to(device=device, dtype=torch.float32)
                gate = _st_neural_gate(
                    head(phi), state.threshold, 1.0, gumbel=False, training=False
                )
            rel = rel * gate
        masks[rule_id] = rel
        p_by_rule[rule_id] = torch.tensor(
            float(rule_weights[rule_id]), dtype=torch.float32, device=device
        )
    with torch.no_grad():
        p_or = _joint_noisy_or(masks, p_by_rule, candidate_ids, num_queries, device)
    return p_or.detach().cpu().tolist()


def _graded_admission_evidence(
    eligibility: list[Any],
    rule_weights: dict[str, float],
    num_queries: int,
    neural_heldout: dict[str, Any] | None = None,
    labels: list[bool] | None = None,
    within_set_norm_fn: Any | None = None,
) -> dict[str, Any]:
    """Graded (de-saturated) admission evidence for the neural-bodied candidate.

    The GRADED analog of ``_joint_mixture_probs``: instead of the HARD ST gate it
    consumes the graded gate ``g_tilde = sigmoid((g_theta - logit(tau)))`` (the
    canonical temperature-1 graded support — the WMC-facing mass), and returns the
    DECOMPOSED per-query evidence the locked two-axis rubric consumes.

    Both ``hard_head_prob`` and ``graded_mass`` are the noisy-OR over the SAME
    ``rel_mask * . * sigma(guard)`` structure — only the gate kind differs — so the
    hard-vs-graded divergence audit isolates exactly the gate effect (a checker can
    never compare graded head-mass against a bare neural gate and miss the
    rel_mask/guard layers).

    ``graded_mass`` is the WMC-facing graded SUPPORT, NOT a calibrated truth
    probability, and carries NO production-firing certification. It is a monotone
    transform of ``g_theta`` (rank-preserving in exact arithmetic), but under a
    SATURATED head (large ``|g_theta|``) it floors to a near-constant and loses the
    rank NUMERICALLY — so the admission-retention rank (retention_auc / strict
    vigilance / axis1_margin) is read from the raw ``g_theta`` logit, the lossless
    rank carrier, NOT from ``graded_mass``. (The checker recomputes that rank from
    ``g_theta``.) ``production_firing_mass`` is the per-entity, offset-dependent
    gate the rubric's local per-entity transfer reads (the quantity an earlier
    probe showed does NOT transfer) — exposed so the checker can READ
    firing-transfer, never to assert it passes. ``axis1_margin`` is a convenience
    scalar; the checker recomputes the retention rank from raw per-query
    ``(graded_mass, label, g_theta)`` (anti-gaming).
    """
    import math

    import torch

    neural_heldout = neural_heldout or {}
    device = torch.device("cpu")
    masks_hard: dict[str, Any] = {}
    masks_graded: dict[str, Any] = {}
    p_by_rule: dict[str, Any] = {}
    candidate_ids: list[str] = []
    per_cand: dict[str, dict[str, Any]] = {}
    for guard_pred, mask in eligibility:
        rule_id = guard_pred[len(_GUARD_PREDICATE_PREFIX) :]
        if rule_id not in rule_weights:
            continue
        candidate_ids.append(rule_id)
        rel = torch.tensor(
            [1.0 if m else 0.0 for m in mask], dtype=torch.float32, device=device
        )
        if rule_id in neural_heldout:
            state, features = neural_heldout[rule_id]
            head = _make_neural_body_head(state.width, state.head_depth, state.hidden_dim)
            head.load_state_dict(state.state_dict)
            head.eval()
            with torch.no_grad():
                phi = features.detach().to(device=device, dtype=torch.float32)
                logit = head(phi).reshape(-1)
            tau = min(max(state.threshold, 1e-6), 1.0 - 1e-6)
            tau_logit = math.log(tau / (1.0 - tau))
            hard = (logit >= tau_logit).float()  # hard ST gate forward (eligibility)
            if within_set_norm_fn is not None:
                # Context-relative (set-relative) graded mass: the within-set normalization of
                # g_theta over the comparison set (admission = one group). It
                # de-saturates AND preserves the g_theta rank where the per-entity
                # sigmoid floors to a near-constant. The hard ST gate above is
                # untouched (eligibility fence); only the GRADED mass is set-relative.
                group_id = torch.zeros(num_queries, dtype=torch.long, device=device)
                with torch.no_grad():
                    within_set = within_set_norm_fn(logit, group_id, mode="eval")
                graded = within_set
            else:
                within_set = None
                graded = torch.sigmoid(logit - tau_logit)  # per-entity temp-1 gate
            per_cand[rule_id] = {
                "rel": rel, "logit": logit, "tau_logit": tau_logit,
                "graded": graded, "hard": hard, "within_set": within_set,
            }
            masks_hard[rule_id] = rel * hard
            masks_graded[rule_id] = rel * graded
        else:
            ones = torch.ones(num_queries, dtype=torch.float32, device=device)
            per_cand[rule_id] = {
                "rel": rel, "logit": None, "tau_logit": None,
                "graded": ones, "hard": ones, "within_set": None,
            }
            masks_hard[rule_id] = rel
            masks_graded[rule_id] = rel
        p_by_rule[rule_id] = torch.tensor(
            float(rule_weights[rule_id]), dtype=torch.float32, device=device
        )

    with torch.no_grad():
        hard_head = _joint_noisy_or(
            masks_hard, p_by_rule, candidate_ids, num_queries, device
        )
        graded_head = _joint_noisy_or(
            masks_graded, p_by_rule, candidate_ids, num_queries, device
        )

    # The admitted candidate: the neural-bodied one if present (the single-winner
    # admission case), else the first candidate. Its per-entity gate/logit are the
    # decomposed scalar fields; the head probs are the noisy-OR over the pool.
    selected = next(
        (rid for rid in candidate_ids if rid in neural_heldout),
        candidate_ids[0] if candidate_ids else None,
    )
    sel = per_cand.get(selected, {})
    sel_within = sel.get("within_set")
    per_query: list[dict[str, Any]] = []
    for i in range(num_queries):
        logit_i = sel.get("logit")
        per_query.append(
            {
                "query_index": i,
                "selected_rule_id": selected,
                "relational_mask": float(sel["rel"][i]) if selected else None,
                "hard_gate": float(sel["hard"][i]) if selected else None,
                "graded_gate": float(sel["graded"][i]) if selected else None,
                # within_set_norm: the operator's set-relative output (context-relative mode),
                # None in per-entity mode. Cross-check only — the checker
                # recomputes the within-set rank from raw g_theta (anti-gaming).
                "within_set_norm": (
                    float(sel_within[i]) if sel_within is not None else None
                ),
                "g_theta": float(logit_i[i]) if logit_i is not None else None,
                "tau_logit": sel.get("tau_logit"),
                "hard_head_prob": float(hard_head[i]),
                "graded_mass": float(graded_head[i]),
                "production_firing_mass": (
                    float(sel["graded"][i]) if selected else None
                ),
                "label": (bool(labels[i]) if labels is not None else None),
            }
        )

    # axis1_margin: the LOGIT-space separation margin
    # ``min g_theta(pos) - max g_theta(neg)`` — matching the locked rubric (whose
    # LOW_MARGIN annotation is "< 1.0 logit") and the separation-margin definition. A
    # convenience cross-check only; the checker recomputes it from raw
    # ``(g_theta, label)`` (anti-gaming). ``None`` when there is no neural logit.
    axis1_margin = None
    sel_logit = sel.get("logit")
    if labels is not None and sel_logit is not None:
        pos = [float(sel_logit[i]) for i in range(num_queries) if labels[i]]
        neg = [float(sel_logit[i]) for i in range(num_queries) if not labels[i]]
        if pos and neg:
            axis1_margin = min(pos) - max(neg)

    return {"mode": "graded", "per_query": per_query, "axis1_margin": axis1_margin}


def _graded_firing_evidence(
    eligibility: list[Any],
    rule_weights: dict[str, float],
    neural_heldout: dict[str, Any],
    *,
    context_ids: list[Any],
    firing_rule: dict[str, Any],
    split: str = "heldout",
    labels: list[bool] | None = None,
    within_set_norm_fn: Any,
) -> dict[Any, dict[str, Any]]:
    """Within-context firing evidence (read-side).

    Emits the per-``context_id`` comparison-set the firing-evidence checker consumes. The
    within-context firing decision is RANK-based (the only property that transfers):
    per context, the selected neural candidate's ``g_theta`` is normalized within its
    comparison set (``within_set_norm_fn(.., mode="eval")``) and the top-k% by that
    within-context rank FIRE — restricted to admissible entities (the eligibility
    fence: the hard ST gate ``g_theta >= tau_logit``; a non-admissible entity never fires).

    ``production_firing_mass`` is the within-set firing mass (NOT ``sigmoid(g-tau)``,
    the non-transferring absolute quantity); ``tau_logit`` is emitted as provenance
    only. Raw ``g_theta`` and the ordered comparison set are emitted so the checker
    RECOMPUTES the within-set rank / firing from raw (anti-gaming). The cardinality
    fence (``|set| < 16`` fail-closed) and the cross-context production gate are the
    checker's to enforce on this evidence, not this emit.
    """
    import math

    import torch

    neural_heldout = neural_heldout or {}
    n = len(context_ids)

    selected = None
    sel: dict[str, Any] = {}
    for guard_pred, mask in eligibility:
        rule_id = guard_pred[len(_GUARD_PREDICATE_PREFIX) :]
        if rule_id not in rule_weights or rule_id not in neural_heldout:
            continue
        selected = rule_id
        rel = torch.tensor([1.0 if m else 0.0 for m in mask], dtype=torch.float32)
        state, features = neural_heldout[rule_id]
        head = _make_neural_body_head(state.width, state.head_depth, state.hidden_dim)
        head.load_state_dict(state.state_dict)
        head.eval()
        with torch.no_grad():
            logit = head(features.detach().to(dtype=torch.float32)).reshape(-1)
        tau = min(max(state.threshold, 1e-6), 1.0 - 1e-6)
        tau_logit = math.log(tau / (1.0 - tau))
        admissible = (logit >= tau_logit)  # hard ST eligibility (the eligibility fence)
        sel = {"rel": rel, "logit": logit, "tau_logit": tau_logit, "admissible": admissible}
        break
    if selected is None:
        return {}

    # ordered distinct context ids -> integer group index for the within-set helper.
    uniq = list(dict.fromkeys(context_ids))
    cid_to_idx = {c: i for i, c in enumerate(uniq)}
    group_id = torch.tensor([cid_to_idx[c] for c in context_ids], dtype=torch.long)
    with torch.no_grad():
        within = within_set_norm_fn(sel["logit"], group_id, mode="eval")

    # fired: top ceil(k * |admissible-in-context|) by within-context rank, per context.
    k = float(firing_rule.get("k", 0.5))
    fired = [False] * n
    for c in uniq:
        adm_idx = [
            i for i in range(n) if context_ids[i] == c and bool(sel["admissible"][i])
        ]
        if not adm_idx:
            continue
        adm_idx.sort(key=lambda i: float(within[i]), reverse=True)
        n_fire = max(1, math.ceil(k * len(adm_idx)))
        for i in adm_idx[:n_fire]:
            fired[i] = True

    out: dict[Any, dict[str, Any]] = {}
    for c in uniq:
        comparison_set = [
            {
                "x": i,
                "label": (bool(labels[i]) if labels is not None else None),
                "g_theta": float(sel["logit"][i]),
                "tau_logit": sel["tau_logit"],
                "relational_mask": float(sel["rel"][i]),
                "axis1_admissible": bool(sel["admissible"][i]),
                "within_set_norm": float(within[i]),
                "production_firing_mass": float(within[i]),
                "fired": fired[i],
            }
            for i in range(n)
            if context_ids[i] == c
        ]
        out[c] = {
            "comparison_set": comparison_set,
            "firing_rule": dict(firing_rule),
            "split": split,
        }
    return out


def evaluate_joint_mixture(
    source: str,
    *,
    rule_weights: dict[str, float],
    num_queries: int,
    arity: int = 1,
    config: NeuroSymbolicTrainingConfig = NeuroSymbolicTrainingConfig(),
    neural_heldout: dict[str, Any] | None = None,
    mode: str = "hard",
    heldout_labels: list[bool] | None = None,
    set_relative: bool = False,
) -> list[float] | dict[str, Any]:
    """Held-out generalization read for the joint mixture.

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

    NEURAL-BODIED candidates: pass ``neural_heldout[rule_id] =
    (result.neural_body_state[rule_id], held_out_features)``. That candidate's
    held-out eligibility is its relational grounding AND the trained ``g_theta``'s
    hard gate over the HELD-OUT entity features — so an overfit neural predicate
    yields low held-out ``p_or`` exactly as a spurious relational correlate does;
    the guard-free held-out selector is the same vigilance net for both.

    MODE (graded admission):
      - ``"hard"`` (default): the byte-unchanged behavior above — returns the
        per-query hard-gate noisy-OR ``list[float]``. Production firing semantics.
      - ``"graded"``: returns the DECOMPOSED graded admission evidence
        (``_graded_admission_evidence``) — per-query hard_gate / hard_head_prob /
        graded_gate / graded_mass / production_firing_mass / g_theta / tau_logit /
        (optional) ``heldout_labels`` + a convenience ``axis1_margin``. This is the
        graded admission read; graded_mass is the WMC-facing graded SUPPORT, NOT
        calibrated truth, and carries NO production-firing certification. The
        admission-retention rank is read from the raw ``g_theta`` logit (the lossless
        rank carrier), NOT from ``graded_mass`` — which saturates to a near-constant
        under a saturated head. Pass ``heldout_labels`` (the held-out supervision) to
        populate ``label`` and the ``axis1_margin`` cross-check; the rubric checker
        recomputes the retention rank (retention_auc / strict vigilance /
        axis1_margin) from the raw per-query ``(g_theta, label)``.
    """
    if mode not in ("hard", "graded"):
        raise ValueError(
            f"unsupported evaluate_joint_mixture mode {mode!r}; expected "
            "'hard' or 'graded'"
        )
    import pyxlog

    program_source, _rules, train_head, _objective = _desugar_source(source)
    program = pyxlog.Program.compile(
        program_source,
        device=config.device,
        memory_mb=config.gpu_memory_mb,
    )
    if mode == "graded":
        eligibility = program.joint_candidate_eligibility(
            train_head, arity, num_queries
        )
        # set_relative (context-relative admission): the de-saturating within-set normalization
        # of g_theta over the held-out comparison set (one group). Default off ->
        # per-entity graded gate, byte-unchanged.
        return _graded_admission_evidence(
            eligibility,
            rule_weights,
            num_queries,
            neural_heldout,
            heldout_labels,
            within_set_norm_fn=(within_set_norm if set_relative else None),
        )
    return _joint_mixture_probs(
        program, train_head, rule_weights, num_queries, arity, neural_heldout
    )


def _make_neural_body_head(width: int, head_depth: int, hidden_dim: int) -> Any:
    """The learned predicate ``g_theta`` over an entity feature vector: a small
    head mapping ``phi(x)`` (width) to a scalar logit. ``head_depth == 1`` is a
    single linear->scalar (the contract default, matching the guard's
    minimalism); ``head_depth > 1`` inserts tanh hidden layers (the config knob,
    so capacity grows without a surface re-spin)."""
    import torch

    if head_depth <= 1:
        return torch.nn.Sequential(torch.nn.Linear(width, 1))
    layers: list[Any] = [torch.nn.Linear(width, hidden_dim), torch.nn.Tanh()]
    for _ in range(head_depth - 2):
        layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        layers.append(torch.nn.Tanh())
    layers.append(torch.nn.Linear(hidden_dim, 1))
    return torch.nn.Sequential(*layers)


def _st_neural_gate(
    logit: Any, threshold: float, temperature: float, gumbel: bool, training: bool
) -> Any:
    """Straight-through neural gate: the ST-Gumbel discretization already used
    for relation-selection weights, applied to ``g_theta``'s activation.

    Forward is the HARD Boolean ``sigmoid(g_theta) >= tau`` (so the eligibility is
    a hard derivation gate, not soft truth-mass — which is exactly why it composes
    in the noisy-OR without being a WMC circuit leaf). Backward is the
    temperature-sigmoid (optionally Gumbel-perturbed during training), so theta
    receives gradient. Returns a length-n vector of {0,1} (forward) carrying the
    soft gradient.
    """
    import torch

    tau = min(max(threshold, 1e-6), 1.0 - 1e-6)
    # gate fires when sigmoid(logit) >= tau  <=>  logit >= logit(tau).
    tau_logit = torch.log(torch.tensor(tau / (1.0 - tau), device=logit.device))
    centered = logit.reshape(-1) - tau_logit
    if gumbel and training:
        u = torch.rand_like(centered).clamp(1e-6, 1.0 - 1e-6)
        centered = centered + (torch.log(u) - torch.log(1.0 - u))
    soft = torch.sigmoid(centered / max(temperature, 1e-6))
    hard = (soft >= 0.5).float()
    # Straight-through: hard value forward, soft gradient backward.
    return hard + (soft - soft.detach())


def _within_set_st_gate(
    logit: Any, threshold: float, group_id: Any, temperature: float
) -> Any:
    """Straight-through gate whose backward trains on within-comparison-set rank.

    FORWARD is the SAME hard derivation gate as ``_st_neural_gate``
    (``sigmoid(logit) >= tau``), so the neural conjunct stays a hard Boolean gate,
    not soft truth-mass (thinking proposes, the engine disposes), and the held-out
    read is unaffected. BACKWARD flows through ``within_set_norm(mode="train")`` --
    an offset-invariant within-set z-norm -- so when ``phi`` carries gradient (its
    backbone feature is not detached) the head/backbone receive gradient on the
    transferable within-set RANK rather than the non-transferable absolute level
    that the per-entity sigmoid trains. Returns a length-n vector of {0,1}
    (forward) carrying the within-set soft gradient.
    """
    import torch

    tau = min(max(threshold, 1e-6), 1.0 - 1e-6)
    # gate fires when sigmoid(logit) >= tau  <=>  logit >= logit(tau).
    tau_logit = torch.log(torch.tensor(tau / (1.0 - tau), device=logit.device))
    flat = logit.reshape(-1)
    hard = (flat >= tau_logit).float()
    soft = within_set_norm(flat, group_id, mode="train", temp=temperature)
    # Straight-through: hard value forward, within-set z-norm gradient backward.
    return hard + (soft - soft.detach())


def _neural_gate_for(
    logit: Any, spec: "NeuralBodySpec", n: int, device: Any, training: bool
) -> Any:
    """Select a neural-bodied candidate's gate for the joint noisy-OR forward.

    When ``spec.train_within_set_norm`` is set, the TRAINING backward is routed
    through the offset-invariant within-set z-norm (``_within_set_st_gate``; the
    train comparison set is the whole candidate binding-set); otherwise the
    absolute per-entity ``_st_neural_gate``. Both share the IDENTICAL hard forward
    gate, so the held-out read (``training=False``) and the noisy-OR composition
    are unchanged regardless of the flag.
    """
    import torch

    if spec.train_within_set_norm and training:
        group_id = torch.zeros(n, dtype=torch.long, device=device)
        return _within_set_st_gate(
            logit, spec.threshold, group_id, spec.gumbel_temperature
        )
    return _st_neural_gate(
        logit,
        spec.threshold,
        spec.gumbel_temperature,
        gumbel=spec.gumbel_noise and training,
        training=training,
    )


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
    """Collect ``(inputs, targets)`` from the example batches.

    ``inputs`` is ``None`` when the batches carry only ``targets`` — the Stage-B
    existential-join contract, where the join neural predicate is forwarded over
    the ``domain_inputs`` event domain rather than over per-query input rows.
    """
    import torch

    input_parts = []
    target_parts = []
    have_inputs = True
    for batch in examples:
        targets = batch["targets"].to(dtype=torch.float32)
        if bool(((targets != 0.0) & (targets != 1.0)).any()):
            raise ValueError("targets must be binary (0.0 or 1.0)")
        if "inputs" in batch:
            inputs = batch["inputs"]
            if inputs.shape[0] != targets.shape[0]:
                raise ValueError(
                    f"examples batch has {inputs.shape[0]} inputs but "
                    f"{targets.shape[0]} targets"
                )
            input_parts.append(inputs)
        else:
            have_inputs = False
        target_parts.append(targets)
    all_targets = [bool(t >= 0.5) for t in torch.cat(target_parts, dim=0)]
    all_inputs = torch.cat(input_parts, dim=0) if (have_inputs and input_parts) else None
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
