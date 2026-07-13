"""Neural JOIN bodies: a neural predicate on an EXISTENTIAL join variable.

    plastic(E) :- saliency(Ev, strengthen), pre_before_post(Ev, E).
                  \\____ neural on Ev ____/  \\__ ordinary relation __/

``Ev`` is not in the head, so it ranges over a domain; the head binding's truth is
the OR over that domain. This module owns three things and nothing else:

  1. reading that shape OUT OF THE RULE (the rule is the single source of truth --
     the join relation is never supplied by the caller). The shape is EXACTLY
     ``{one neural atom, one join relation}``; a body with any further conjunct is
     NOT this shape and is refused, never silently reduced to it;
  2. reading the join extension FROM THE ENGINE (:func:`read_join_extension`), by
     enumerating the relation's tuples (never from a Python side-channel: if the
     caller handed us the edge->events map, the OR would be Python's, not the
     logic's, and the whole claim would be hollow);
  3. the OR itself, in log space (a naive product underflows on a large domain),
     over a PRECOMPUTED index of that static extension so the training hot loop
     does no per-step host->device copies (:class:`JoinExtensionIndex`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_ATOM = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)")


@dataclass(frozen=True)
class JoinBody:
    neural_predicate: str
    network: str
    join_var: str
    relation: str
    event_arg: int   # position of the join var in the relation
    head_arg: int    # position of the head var in the relation


def _atoms(body: str) -> list[tuple[str, list[str]]]:
    return [
        (m.group(1), [a.strip() for a in m.group(2).split(",") if a.strip()])
        for m in _ATOM.finditer(body)
    ]


def parse_join_body(
    body: str, neural_predicates: dict[str, str], head_var: str
) -> JoinBody | None:
    """Recognize the Stage-B shape. Returns None for any other body (the caller then
    keeps its existing behaviour) -- this function never guesses.

    The shape is EXACTLY two literals: the one neural atom and the one relation that
    joins the existential variable to the head. A longer body (e.g. an extra
    ``high_degree(E)`` conjunct) is a DIFFERENT rule, and this module has no mask for
    it -- returning the two-literal mask anyway would silently drop the conjunct and
    train a rule nobody wrote. Out of scope must mean rejected, so anything but the
    exact shape returns None and the caller's typed rejection stands.
    """
    atoms = _atoms(body)
    if len(atoms) != 2:
        return None
    neural = [(p, args) for p, args in atoms if p in neural_predicates]
    if len(neural) != 1:
        return None
    pred, nargs = neural[0]
    if not nargs:
        return None
    join_var = nargs[0]                      # nn(net, [Input], Label, ...) -> arg 0
    if join_var == head_var:
        return None                          # head-bound gate, not an existential join
    if not join_var[:1].isupper():
        return None                          # a constant, not a variable

    relation = [(p, args) for p, args in atoms if p not in neural_predicates]
    if len(relation) != 1:
        return None
    p, args = relation[0]
    if join_var not in args or head_var not in args:
        return None
    return JoinBody(
        neural_predicate=pred,
        network=neural_predicates[pred],
        join_var=join_var,
        relation=p,
        event_arg=args.index(join_var),
        head_arg=args.index(head_var),
    )


def read_join_extension(
    ilp_program: Any, jb: JoinBody, num_bindings: int
) -> list[list[int]]:
    """head binding -> [event ids], read FROM THE ENGINE.

    The engine owns the relation; we only ask it to enumerate the tuples that
    hold, via ``CompiledIlpProgram.relation_facts(name) -> list[list[int]]``. This
    is deliberately NOT solved by taking an edge->events map from Python: that
    would move the aggregation structure out of the logic, and the claim that
    "the logic performs the OR" would be false -- it would be Python's OR over a
    caller-supplied hint, not the engine's own relation.

    This is O(|extension|) -- it enumerates only the tuples that actually hold,
    not O(num_bindings * domain_size) membership probes. Tuples whose head
    binding falls outside ``0 .. num_bindings - 1`` are ignored. Bindings with no
    joined events get ``[]``. Events are sorted ascending within each bucket, and
    the traversal itself is a plain forward scan of ``relation_facts``, so the
    result is deterministic for a given compiled program.
    """
    buckets: list[list[int]] = [[] for _ in range(num_bindings)]
    for t in ilp_program.relation_facts(jb.relation):
        h = t[jb.head_arg]
        if 0 <= h < num_bindings:
            buckets[h].append(t[jb.event_arg])
    for bucket in buckets:
        bucket.sort()
    return buckets


@dataclass(frozen=True)
class JoinExtensionIndex:
    """The STATIC join extension, flattened once into device tensors.

    ``event_ids`` is every ``(binding, event)`` pair's event id, concatenated across
    bindings; ``binding_ids`` is the same length and says which head binding each
    entry belongs to (a segment id). Both live on the device from construction, so a
    training step's OR is one gather + one segmented sum -- no host->device copy, no
    per-binding Python loop, no stack of n scalars. The extension never changes
    across steps (that is the whole premise of reading it from the engine once), so
    this is built ONCE, outside the hot loop.
    """

    event_ids: Any       # [total] long, on device
    binding_ids: Any     # [total] long, on device (segment id per entry)
    num_bindings: int
    device: Any


def prepare_extension(extension: list[list[int]], device: Any) -> JoinExtensionIndex:
    """Flatten ``head binding -> [event ids]`` into the device-resident index."""
    import torch

    event_ids: list[int] = []
    binding_ids: list[int] = []
    for h, events in enumerate(extension):
        event_ids.extend(events)
        binding_ids.extend([h] * len(events))
    return JoinExtensionIndex(
        event_ids=torch.as_tensor(event_ids, device=device, dtype=torch.long),
        binding_ids=torch.as_tensor(binding_ids, device=device, dtype=torch.long),
        num_bindings=len(extension),
        device=device,
    )


def noisy_or_from_index(p: Any, index: JoinExtensionIndex) -> Any:
    """1 - PROD_{e in ext(h)} (1 - p_e), in log space, over a prepared index.

    The ONE implementation of the math. Vectorized: gather log(1 - p) at every
    (binding, event) entry, segmented-sum it into a [num_bindings] accumulator, and
    exponentiate. A binding with an empty extension contributes no entry, so its
    accumulator stays 0 and it yields exactly ``1 - exp(0) = 0`` -- an OR over
    nothing is false, which is also what the provenance circuit does for an edge with
    no joined events.
    """
    import torch

    eps = 1e-7
    logq = torch.log1p(-p.clamp(eps, 1.0 - eps))     # log(1 - p_e)
    acc = torch.zeros(index.num_bindings, device=logq.device, dtype=logq.dtype)
    acc = acc.index_add(0, index.binding_ids, logq[index.event_ids])
    return 1.0 - torch.exp(acc)


def noisy_or_over_extension(p: Any, extension: list[list[int]], device: Any) -> Any:
    """1 - PROD_{e in ext(h)} (1 - p_e), for a raw (unprepared) extension.

    Convenience wrapper: builds the index and delegates, so there is exactly one
    implementation of the OR. Callers in a hot loop must build the index ONCE with
    :func:`prepare_extension` and call :func:`noisy_or_from_index` per step instead.
    """
    return noisy_or_from_index(p, prepare_extension(extension, device))
