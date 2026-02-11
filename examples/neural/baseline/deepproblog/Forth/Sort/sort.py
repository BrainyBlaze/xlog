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
dev_queries = QueryDataset("data/train{}_test{}_dev.txt".format(train, test))
test_queries = QueryDataset("data/train{}_test{}_test.txt".format(train, test))
train_limit = int(os.getenv("XLOG_TRAIN_LIMIT", "0"))
eval_batch_size = int(os.getenv("XLOG_EVAL_BATCH_SIZE", "256"))
train_batch_size = int(os.getenv("XLOG_TRAIN_BATCH_SIZE", "16"))
if train_limit > 0:
    train_queries = train_queries.subset(0, train_limit)

fc1 = EncodeModule(20, 20, 2)

model = Model(
    "compare.pl",
    [Network(fc1, "swap_net", optimizer=torch.optim.Adam(fc1.parameters(), 1.0))],
)
require_cuda(model.networks.values())
model.set_engine(ExactEngine(model), cache=True)

test_model = Model("compare.pl", [Network(fc1, "swap_net", k=1)])
require_cuda(test_model.networks.values())
test_model.set_engine(ExactEngine(test_model), cache=False)

train_obj = train_model(
    model,
    DataLoader(train_queries, train_batch_size),
    40,
    log_iter=50,
    test_iter=len(train_queries),
    test=lambda x: [
        (
            "Accuracy",
            held_out_accuracy_batched(test_model, dev_queries, batch_size=eval_batch_size),
        )
    ],
)

held_out_accuracy = held_out_accuracy_batched(
    test_model, test_queries, batch_size=eval_batch_size
)
print("Held-out Accuracy", held_out_accuracy)
