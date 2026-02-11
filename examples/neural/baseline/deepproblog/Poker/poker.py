import torch
import os

from deepproblog.dataset import DataLoader, Dataset
from deepproblog.engines import ExactEngine
from deepproblog.examples.Poker import PokerSeparate
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.optimizer import SGD
from deepproblog.train import train_model
from deepproblog.utils.standard_networks import smallnet
from _xlog_runtime import require_cuda, held_out_accuracy_batched

batch_size = int(os.getenv("XLOG_TRAIN_BATCH_SIZE", "500"))
epochs = int(os.getenv("XLOG_EPOCHS", "10"))
datasets = {
    "unfair": PokerSeparate(
        "unfair", probs=[0.2, 0.4, 0.15, 0.25], extra_supervision=True
    ),
    "fair_test": PokerSeparate("fair_test"),
}

dataset = "unfair"
train_limit = int(os.getenv("XLOG_TRAIN_LIMIT", "0"))
eval_batch_size = int(os.getenv("XLOG_EVAL_BATCH_SIZE", "256"))


class QuerySliceDataset(Dataset):
    def __init__(self, queries):
        self._queries = queries

    def __len__(self):
        return len(self._queries)

    def to_query(self, i):
        return self._queries[i]


train_dataset = datasets[dataset]
if train_limit > 0:
    train_dataset = QuerySliceDataset(train_dataset.to_queries()[:train_limit])

net = Network(
    smallnet(pretrained=True, num_classes=4, size=(100, 150)), "net1", batching=True
)
net.optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
require_cuda([net])
loader = DataLoader(train_dataset, batch_size)

model = Model("model.pl", [net])
model.set_engine(ExactEngine(model), cache=True)
model.optimizer = SGD(model, 5e-2)
model.add_tensor_source(dataset, datasets[dataset])
model.add_tensor_source("fair_test", datasets["fair_test"])

train_obj = train_model(
    model,
    loader,
    epochs,
    loss_function_name="mse",
    log_iter=max(1, len(train_dataset) // batch_size),
    test_iter=max(1, 5 * len(train_dataset) // batch_size),
    test=lambda x: [
        (
            "Accuracy",
            held_out_accuracy_batched(
                model, datasets["fair_test"], batch_size=eval_batch_size
            ),
        )
    ],
    infoloss=0.5,
)  # ,

held_out_accuracy = held_out_accuracy_batched(
    model, datasets["fair_test"], batch_size=eval_batch_size
)
print("Held-out Accuracy", held_out_accuracy)
