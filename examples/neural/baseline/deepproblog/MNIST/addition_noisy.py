from random import randint
import os
import sys

import torch
from problog.logic import Constant

from deepproblog.dataset import DataLoader
from deepproblog.dataset import NoiseMutatorDecorator, MutatingDataset, Subset
from deepproblog.engines import ExactEngine
from deepproblog.examples.MNIST.data import (
    MNISTOperator,
    MNIST_train,
    MNIST_test,
    datasets as mnist_datasets,
)
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.optimizer import SGD
from deepproblog.query import Query
from deepproblog.train import train_model
from _xlog_runtime import require_cuda, held_out_accuracy_batched


def noise(_, query: Query):
    new_query = query.replace_output([Constant(randint(0, 18))])
    return new_query


def held_out_accuracy_closed_form(network, dataset, batch_size: int):
    """Exact closed-form evaluator for noisy addition program.

    Program semantics:
      P(Z=z) = 0.1*(1/19) + 0.9*sum_{d1+d2=z} P(d1|X)*P(d2|Y)
    """
    queries = dataset.to_queries()
    if len(queries) == 0:
        return 0.0, 0, 0

    device = next(network.parameters()).device
    network.eval()
    correct = 0
    total = 0
    uniform_mass = 1.0 / 19.0

    with torch.no_grad():
        for start in range(0, len(queries), batch_size):
            batch_queries = queries[start : start + batch_size]
            idx_x = []
            idx_y = []
            labels = []
            for query in batch_queries:
                # Stable variable order: p0_0 (X), p1_0 (Y)
                vars_sorted = sorted(query.substitution.items(), key=lambda kv: str(kv[0]))
                tx = vars_sorted[0][1]
                ty = vars_sorted[1][1]
                idx_x.append(int(tx.args[0].args[0]))
                idx_y.append(int(ty.args[0].args[0]))
                labels.append(int(float(query.output_values()[0])))

            x_batch = torch.stack([mnist_datasets["test"][i][0] for i in idx_x]).to(
                device, non_blocking=True
            )
            y_batch = torch.stack([mnist_datasets["test"][i][0] for i in idx_y]).to(
                device, non_blocking=True
            )
            px = network(x_batch)
            py = network(y_batch)

            # Convolution over digit probabilities for sums 0..18.
            sum_probs = torch.zeros((len(batch_queries), 19), device=device)
            for d1 in range(10):
                for d2 in range(10):
                    sum_probs[:, d1 + d2] += px[:, d1] * py[:, d2]

            z_probs = 0.1 * uniform_mass + 0.9 * sum_probs
            pred = torch.argmax(z_probs, dim=1).cpu()
            label_tensor = torch.tensor(labels)
            correct += int((pred == label_tensor).sum().item())
            total += len(batch_queries)

    return float(correct) / float(total), correct, total


dataset = MNISTOperator(
    dataset_name="train",
    function_name="addition_noisy",
    operator=sum,
    size=1,
)
noisy_dataset = MutatingDataset(dataset, NoiseMutatorDecorator(0.2, noise))
train_limit = int(os.getenv("XLOG_TRAIN_LIMIT", "0"))
eval_batch_size = int(os.getenv("XLOG_EVAL_BATCH_SIZE", "256"))
train_batch_size = int(os.getenv("XLOG_TRAIN_BATCH_SIZE", "1024"))
run_mode = os.getenv("XLOG_RUN_MODE", "train_eval")
state_path = os.getenv("XLOG_STATE_PATH", "snapshot/noisy_addition.pth")
eval_start = int(os.getenv("XLOG_EVAL_START", "0"))
eval_end_raw = os.getenv("XLOG_EVAL_END")
if train_limit > 0:
    noisy_dataset = Subset(noisy_dataset, 0, train_limit)
queries = DataLoader(noisy_dataset, train_batch_size)

network = MNIST_Net()
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
require_cuda([net])
model = Model("models/noisy_addition.pl", [net])

model.add_tensor_source("train", MNIST_train)
model.add_tensor_source("test", MNIST_test)

engine_cache = os.getenv("XLOG_ENGINE_CACHE", "0") == "1"
cache_root = os.getenv("XLOG_CACHE_ROOT")
sdd_auto_gc = os.getenv("XLOG_SDD_AUTO_GC", "0") == "1"
solver_kwargs = {"cache": engine_cache}
if cache_root:
    solver_kwargs["cache_root"] = cache_root
if sdd_auto_gc:
    solver_kwargs["sdd_auto_gc"] = True
model.set_engine(ExactEngine(model), **solver_kwargs)
model.optimizer = SGD(model, 1e-3)

if run_mode not in {"train_eval", "train_only", "eval_only"}:
    raise ValueError(
        "XLOG_RUN_MODE must be one of train_eval, train_only, eval_only "
        f"(got {run_mode})"
    )

if run_mode in {"train_eval", "train_only"}:
    train_model(model, queries, 1, log_iter=100)
    model.save_state(state_path)
else:
    model.load_state(state_path)

if run_mode == "train_only":
    print("Training complete; state saved to", state_path)
    sys.exit(0)

held_out_dataset = MNISTOperator(
    dataset_name="test", function_name="addition_noisy", operator=sum, size=1
)
eval_end = len(held_out_dataset) if eval_end_raw is None else int(eval_end_raw)
eval_end = min(eval_end, len(held_out_dataset))
if eval_end <= eval_start:
    raise ValueError(f"Invalid evaluation slice: start={eval_start}, end={eval_end}")

eval_dataset = (
    held_out_dataset
    if eval_start == 0 and eval_end == len(held_out_dataset)
    else Subset(held_out_dataset, eval_start, eval_end)
)
eval_method = os.getenv("XLOG_EVAL_METHOD", "solver")
if eval_method == "closed_form":
    held_out_accuracy, correct, total = held_out_accuracy_closed_form(
        network, eval_dataset, batch_size=eval_batch_size
    )
else:
    held_out_accuracy = held_out_accuracy_batched(
        model, eval_dataset, batch_size=eval_batch_size
    )
    total = len(eval_dataset)
    correct = int(round(held_out_accuracy * total))
print("Held-out Accuracy", held_out_accuracy)
print("Held-out Correct/Total", correct, total)
print("Held-out Slice", eval_start, eval_end)
