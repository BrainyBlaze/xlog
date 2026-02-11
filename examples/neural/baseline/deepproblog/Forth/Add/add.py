import os

import torch

from deepproblog.dataset import DataLoader, QueryDataset
from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from Forth import EncodeModule
from _xlog_runtime import require_cuda, held_out_accuracy_batched

train = 2
test = 8

train_queries = QueryDataset("data/train{}_test{}_train.txt".format(train, test))
test_queries = QueryDataset("data/train{}_test{}_test.txt".format(train, test))
val = QueryDataset("data/train{}_test{}_dev.txt".format(train, test))
train_limit = int(os.getenv("XLOG_TRAIN_LIMIT", "0"))
eval_batch_size = int(os.getenv("XLOG_EVAL_BATCH_SIZE", "256"))
train_batch_size = int(os.getenv("XLOG_TRAIN_BATCH_SIZE", "50"))
if train_limit > 0:
    train_queries = train_queries.subset(0, train_limit)


net1 = EncodeModule(30, 50, 10, "tanh")
network1 = Network(net1, "neural1")
network1.optimizer = torch.optim.Adam(net1.parameters(), lr=0.02)

net2 = EncodeModule(22, 10, 2, "tanh")
network2 = Network(net2, "neural2")
network2.optimizer = torch.optim.Adam(net2.parameters(), lr=0.02)

model = Model("choose.pl", [network1, network2])
test_network1 = Network(net1, "neural1", k=1)
test_network2 = Network(net2, "neural2", k=1)
test_model = Model(
    "choose.pl", [test_network1, test_network2]
)
require_cuda([network1, network2, test_network1, test_network2])
model.set_engine(ExactEngine(model), cache=True)
test_model.set_engine(ExactEngine(test_model), cache=False)
train_obj = train_model(
    model,
    DataLoader(train_queries, train_batch_size),
    40,
    log_iter=20,
    test=lambda x: [
        ("Accuracy", held_out_accuracy_batched(test_model, val, batch_size=eval_batch_size))
    ],
    test_iter=100,
)

held_out_accuracy = held_out_accuracy_batched(
    test_model, test_queries, batch_size=eval_batch_size
)
print("Held-out Accuracy", held_out_accuracy)
