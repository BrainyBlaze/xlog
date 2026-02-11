"""Runtime utilities for reproducible DeepProbLog baseline execution."""

from __future__ import annotations

from typing import Iterable, Any, Optional

import torch
from deepproblog.network import Network as DeepProbLogNetwork


def _resolve_device(device: Any) -> torch.device:
    if device is None:
        return torch.device("cuda")
    return torch.device(device)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def patch_network_cuda_path() -> None:
    """Patch DeepProbLog network evaluation so batched inputs really move to CUDA."""
    if getattr(DeepProbLogNetwork, "_xlog_cuda_patch_applied", False):
        return

    def _patched_call(self, to_evaluate: list) -> list:
        if self.batching:
            inputs = (self.function(*e) for e in to_evaluate)
            stacked_inputs = []
            for group in zip(*inputs):
                try:
                    tensor_group = torch.stack(group)
                    if self.is_cuda:
                        tensor_group = tensor_group.to(
                            _resolve_device(self.device), non_blocking=True
                        )
                    stacked_inputs.append(tensor_group)
                except TypeError:
                    list_group = list(group)
                    if self.is_cuda:
                        device = _resolve_device(self.device)
                        list_group = [
                            _move_to_device(value, device) for value in list_group
                        ]
                    stacked_inputs.append(list_group)
            return self.network_module(*stacked_inputs)

        outputs = []
        for element in to_evaluate:
            args = self.function(*element)
            if self.is_cuda:
                device = _resolve_device(self.device)
                args = tuple(_move_to_device(value, device) for value in args)
            outputs.append(self.network_module(*args))
        return outputs

    DeepProbLogNetwork.__call__ = _patched_call
    DeepProbLogNetwork._xlog_cuda_patch_applied = True


def require_cuda(networks: Iterable[DeepProbLogNetwork]) -> None:
    """Enable CUDA execution for all provided DeepProbLog networks."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this baseline run but is unavailable.")
    patch_network_cuda_path()
    for network in networks:
        network.cuda()


def _predict_label(answer: Any, gt_query: Any, eps: Optional[float]) -> Any:
    if len(answer.result) == 0:
        return "no_answer"
    max_ans = max(answer.result, key=lambda x: answer.result[x])
    if eps is None:
        return str(max_ans.args[gt_query.output_ind[0]])
    predicted = float(max_ans.args[gt_query.output_ind[0]])
    actual = float(gt_query.output_values()[0])
    if abs(actual - predicted) < eps:
        return actual
    return predicted


def held_out_accuracy_batched(
    model: Any, dataset: Any, batch_size: int = 256, eps: Optional[float] = None
) -> float:
    """Compute held-out accuracy on the full dataset via batched model.solve calls."""
    queries = dataset.to_queries()
    if len(queries) == 0:
        return 0.0

    model.eval()
    correct = 0
    total = 0
    for start in range(0, len(queries), batch_size):
        batch_queries = queries[start : start + batch_size]
        test_queries = [query.variable_output() for query in batch_queries]
        answers = model.solve(test_queries)
        for gt_query, answer in zip(batch_queries, answers):
            if eps is None:
                actual = str(gt_query.output_values()[0])
            else:
                actual = float(gt_query.output_values()[0])
            predicted = _predict_label(answer, gt_query, eps)
            if predicted == actual:
                correct += 1
            total += 1
    return float(correct) / float(total)
