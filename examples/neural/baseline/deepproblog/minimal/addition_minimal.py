import os
import sys

import torch

from data import MNISTImages, AdditionDataset, datasets as mnist_datasets
from deepproblog.dataset import DataLoader, Subset
from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from network import MNIST_Net
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
                x_term = query.query.args[0]
                y_term = query.query.args[1]
                idx_x.append(int(x_term.args[0].args[0]))
                idx_y.append(int(y_term.args[0].args[0]))
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


network = MNIST_Net()
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
require_cuda([net])

model = Model("addition.pl", [net])
model.set_engine(ExactEngine(model))
model.add_tensor_source("train", MNISTImages("train"))
model.add_tensor_source("test", MNISTImages("test"))

dataset = AdditionDataset("train")
test_dataset = AdditionDataset("test")
train_limit = int(os.getenv("XLOG_TRAIN_LIMIT", "0"))
eval_batch_size = int(os.getenv("XLOG_EVAL_BATCH_SIZE", "256"))
train_batch_size = int(os.getenv("XLOG_TRAIN_BATCH_SIZE", "1024"))
epochs = int(os.getenv("XLOG_EPOCHS", "1"))
run_mode = os.getenv("XLOG_RUN_MODE", "train_eval")
state_path = os.getenv("XLOG_STATE_PATH", "snapshot/trained_model.pth")
eval_start = int(os.getenv("XLOG_EVAL_START", "0"))
eval_end_raw = os.getenv("XLOG_EVAL_END")
eval_method = os.getenv("XLOG_EVAL_METHOD", "solver")
if train_limit > 0:
    dataset = Subset(dataset, 0, train_limit)

if run_mode not in {"train_eval", "train_only", "eval_only"}:
    raise ValueError(
        "XLOG_RUN_MODE must be one of train_eval, train_only, eval_only "
        f"(got {run_mode})"
    )

if run_mode in {"train_eval", "train_only"}:
    loader = DataLoader(dataset, train_batch_size, False)
    train_model(model, loader, epochs, log_iter=100, profile=0)
    model.save_state(state_path)

if run_mode == "train_only":
    print("Training complete; state saved to", state_path)
    sys.exit(0)

if run_mode == "eval_only":
    model.load_state(state_path)

eval_end = len(test_dataset) if eval_end_raw is None else int(eval_end_raw)
eval_end = min(eval_end, len(test_dataset))
if eval_end <= eval_start:
    raise ValueError(f"Invalid evaluation slice: start={eval_start}, end={eval_end}")

eval_dataset = (
    test_dataset
    if eval_start == 0 and eval_end == len(test_dataset)
    else Subset(test_dataset, eval_start, eval_end)
)
if eval_method == "closed_form":
    held_out, correct, total = held_out_accuracy_closed_form(
        network, eval_dataset, batch_size=eval_batch_size
    )
else:
    held_out = held_out_accuracy_batched(model, eval_dataset, batch_size=eval_batch_size)
    total = len(eval_dataset)
    correct = int(round(held_out * total))
print("Held-out Accuracy", held_out)
print("Held-out Correct/Total", correct, total)
print("Held-out Slice", eval_start, eval_end)

# Query the model
query = dataset.to_query(0)
result = model.solve([query])[0]
print(result)
