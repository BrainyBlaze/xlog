from json import dumps
import os
import sys

import torch

from deepproblog.dataset import DataLoader, Subset
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.examples.MNIST.data import MNIST_train, MNIST_test, addition
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from _xlog_runtime import require_cuda, held_out_accuracy_batched

method = "exact"
N = 1

name = "addition_{}_{}".format(method, N)

train_set = addition(N, "train")
test_set = addition(N, "test")
train_limit = int(os.getenv("XLOG_TRAIN_LIMIT", "0"))
eval_batch_size = int(os.getenv("XLOG_EVAL_BATCH_SIZE", "256"))
train_batch_size = int(os.getenv("XLOG_TRAIN_BATCH_SIZE", "1024"))
run_mode = os.getenv("XLOG_RUN_MODE", "train_eval")
state_path = os.getenv("XLOG_STATE_PATH", "snapshot/" + name + ".pth")
eval_start = int(os.getenv("XLOG_EVAL_START", "0"))
eval_end_raw = os.getenv("XLOG_EVAL_END")
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
    train = train_model(model, loader, 1, log_iter=100, profile=0)
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
