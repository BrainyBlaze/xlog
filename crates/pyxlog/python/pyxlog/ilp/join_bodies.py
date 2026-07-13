"""Neural JOIN bodies: a neural predicate on an EXISTENTIAL join variable.

    plastic(E) :- saliency(Ev, strengthen), pre_before_post(Ev, E).
                  \\____ neural on Ev ____/  \\__ ordinary relation __/

``Ev`` is not in the head, so it ranges over a domain; the head binding's truth is
the OR over that domain. This module owns three things and nothing else:

  1. reading that shape OUT OF THE RULE (the rule is the single source of truth --
     the join relation is never supplied by the caller);
  2. materializing the join extension FROM THE ENGINE (never from a Python
     side-channel: if the caller handed us the edge->events map, the OR would be
     Python's, not the logic's, and the whole claim would be hollow);
  3. the OR itself, in log space (a naive product underflows on a large domain).
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
    keeps its existing behaviour) -- this function never guesses."""
    atoms = _atoms(body)
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

    for p, args in atoms:
        if p == pred or p in neural_predicates:
            continue
        if join_var in args and head_var in args:
            return JoinBody(
                neural_predicate=pred,
                network=neural_predicates[pred],
                join_var=join_var,
                relation=p,
                event_arg=args.index(join_var),
                head_arg=args.index(head_var),
            )
    return None


def materialize_extension(
    program: Any, jb: JoinBody, num_bindings: int, domain_size: int
) -> list[list[int]]:
    """edge -> [event ids], read FROM THE ENGINE.

    The engine owns the relation; we only ask it which tuples hold. This is
    O(num_bindings * domain_size) membership probes. That is negligible at demo
    scale and is a known wart at dataset scale -- the follow-up is an extension
    ENUMERATION api. It is deliberately NOT solved by taking the map from Python:
    that would move the aggregation structure out of the logic.
    """
    tuples: list[list[int]] = []
    for h in range(num_bindings):
        for e in range(domain_size):
            t = [0, 0]
            t[jb.event_arg] = e
            t[jb.head_arg] = h
            tuples.append(t)
    flat = program.batch_fact_membership(jb.relation, tuples)
    ext: list[list[int]] = []
    for h in range(num_bindings):
        base = h * domain_size
        ext.append([e for e in range(domain_size) if flat[base + e]])
    return ext


def noisy_or_over_extension(p: Any, extension: list[list[int]], device: Any) -> Any:
    """1 - PROD_{e in ext(h)} (1 - p_e), in log space.

    An empty extension yields 0 (an OR over nothing is false), which is also what the
    provenance circuit does for an edge with no joined events.
    """
    import torch

    eps = 1e-7
    logq = torch.log1p(-p.clamp(eps, 1.0 - eps))     # log(1 - p_e)
    out = []
    for ev in extension:
        if not ev:
            out.append(torch.zeros((), device=device, dtype=logq.dtype))
        else:
            idx = torch.as_tensor(ev, device=device, dtype=torch.long)
            out.append(1.0 - torch.exp(logq[idx].sum()))
    return torch.stack(out)
