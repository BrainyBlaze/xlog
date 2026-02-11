import os

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
if train_limit > 0:
    dataset = Subset(dataset, 0, train_limit)

# Train the model
loader = DataLoader(dataset, train_batch_size, False)
train_model(model, loader, 1, log_iter=100, profile=0)
model.save_state("snapshot/trained_model.pth")

held_out = held_out_accuracy_batched(model, test_dataset, batch_size=eval_batch_size)
print("Held-out Accuracy", held_out)

# Query the model
query = dataset.to_query(0)
result = model.solve([query])[0]
print(result)
