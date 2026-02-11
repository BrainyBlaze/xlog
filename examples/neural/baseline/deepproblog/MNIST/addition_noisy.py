from random import randint
import os

import torch
from problog.logic import Constant

from deepproblog.dataset import DataLoader
from deepproblog.dataset import NoiseMutatorDecorator, MutatingDataset, Subset
from deepproblog.engines import ExactEngine
from deepproblog.examples.MNIST.data import MNISTOperator, MNIST_train, MNIST_test
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

model.set_engine(ExactEngine(model))
model.optimizer = SGD(model, 1e-3)

train = train_model(model, queries, 1, log_iter=100)
held_out_dataset = MNISTOperator(
    dataset_name="test", function_name="addition_noisy", operator=sum, size=1
)
held_out_accuracy = held_out_accuracy_batched(
    model, held_out_dataset, batch_size=eval_batch_size
)
print("Held-out Accuracy", held_out_accuracy)
