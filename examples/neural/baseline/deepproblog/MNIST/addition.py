from json import dumps
import os
import sys

import torch

from deepproblog.dataset import DataLoader, Subset
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.examples.MNIST.data import (
    MNIST_train,
    MNIST_test,
    addition,
    datasets as mnist_datasets,
)
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from _xlog_runtime import require_cuda, held_out_accuracy_batched


def held_out_accuracy_closed_form(network, dataset, batch_size: int):
    queries = dataset.to_queries()
    if len(queries) == 0:
        return 0.0, 0, 0

    device = next(network.parameters()).device
    network.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for start in range(0, len(queries), batch_size):
            batch_queries = queries[start : start + batch_size]
            idx_x = []
            idx_y = []
            labels = []
            for query in batch_queries:
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

            sum_probs = torch.zeros((len(batch_queries), 19), device=device)
            for d1 in range(10):
                for d2 in range(10):
                    sum_probs[:, d1 + d2] += px[:, d1] * py[:, d2]

            pred = torch.argmax(sum_probs, dim=1).cpu()
            label_tensor = torch.tensor(labels)
            correct += int((pred == label_tensor).sum().item())
            total += len(batch_queries)

    return float(correct) / float(total), correct, total


method = "exact"
N = 1

name = "addition_{}_{}".format(method, N)

train_set = addition(N, "train")
test_set = addition(N, "test")
train_limit = int(os.getenv("XLOG_TRAIN_LIMIT", "0"))
eval_batch_size = int(os.getenv("XLOG_EVAL_BATCH_SIZE", "256"))
train_batch_size = int(os.getenv("XLOG_TRAIN_BATCH_SIZE", "1024"))
epochs = int(os.getenv("XLOG_EPOCHS", "1"))
run_mode = os.getenv("XLOG_RUN_MODE", "train_eval")
state_path = os.getenv("XLOG_STATE_PATH", "snapshot/" + name + ".pth")
eval_start = int(os.getenv("XLOG_EVAL_START", "0"))
eval_end_raw = os.getenv("XLOG_EVAL_END")
eval_method = os.getenv("XLOG_EVAL_METHOD", "solver")
if train_limit > 0:
    train_set = Subset(train_set, 0, train_limit)

network = MNIST_Net()

pretrain = 0
if pretrain is not None and pretrain > 0:
    network.load_state_dict(torch.load("models/pretrained/all_{}.pth".format(pretrain)))
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
require_cuda([net])

model = Model("models/addition.pl", [net])
if method == "exact":
    model.set_engine(ExactEngine(model), cache=True)
elif method == "geometric_mean":
    model.set_engine(
        ApproximateEngine(model, 1, ApproximateEngine.geometric_mean, exploration=False)
    )

model.add_tensor_source("train", MNIST_train)
model.add_tensor_source("test", MNIST_test)

loader = DataLoader(train_set, train_batch_size, False)
if run_mode not in {"train_eval", "train_only", "eval_only"}:
    raise ValueError(
        "XLOG_RUN_MODE must be one of train_eval, train_only, eval_only "
        f"(got {run_mode})"
    )

if run_mode in {"train_eval", "train_only"}:
    train = train_model(model, loader, epochs, log_iter=100, profile=0)
    model.save_state(state_path)
    train.logger.comment(dumps(model.get_hyperparameters()))
else:
    model.load_state(state_path)
    train = None

if run_mode == "train_only":
    print("Training complete; state saved to", state_path)
    sys.exit(0)

eval_end = len(test_set) if eval_end_raw is None else int(eval_end_raw)
eval_end = min(eval_end, len(test_set))
if eval_end <= eval_start:
    raise ValueError(f"Invalid evaluation slice: start={eval_start}, end={eval_end}")

eval_dataset = (
    test_set
    if eval_start == 0 and eval_end == len(test_set)
    else Subset(test_set, eval_start, eval_end)
)
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
if train is not None:
    train.logger.comment("Accuracy {}".format(held_out_accuracy))
    train.logger.write_to_file("log/" + name)
