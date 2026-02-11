import os
import sys

import torch

from data import MNISTImages, AdditionDataset
from deepproblog.dataset import DataLoader, Subset
from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from network import MNIST_Net
from _xlog_runtime import require_cuda, held_out_accuracy_batched

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
run_mode = os.getenv("XLOG_RUN_MODE", "train_eval")
state_path = os.getenv("XLOG_STATE_PATH", "snapshot/trained_model.pth")
eval_start = int(os.getenv("XLOG_EVAL_START", "0"))
eval_end_raw = os.getenv("XLOG_EVAL_END")
if train_limit > 0:
    dataset = Subset(dataset, 0, train_limit)

if run_mode not in {"train_eval", "train_only", "eval_only"}:
    raise ValueError(
        "XLOG_RUN_MODE must be one of train_eval, train_only, eval_only "
        f"(got {run_mode})"
    )

if run_mode in {"train_eval", "train_only"}:
    loader = DataLoader(dataset, train_batch_size, False)
    train_model(model, loader, 1, log_iter=100, profile=0)
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
