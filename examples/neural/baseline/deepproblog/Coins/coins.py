import torch
import os

from deepproblog.dataset import DataLoader, Subset
from deepproblog.engines import ExactEngine
from deepproblog.examples.Coins.data.dataset import train_dataset, test_dataset
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.utils.standard_networks import smallnet
from deepproblog.utils.stop_condition import Threshold, StopOnPlateau
from _xlog_runtime import require_cuda, held_out_accuracy_batched

batch_size = int(os.getenv("XLOG_TRAIN_BATCH_SIZE", "50"))
train_limit = int(os.getenv("XLOG_TRAIN_LIMIT", "0"))
eval_batch_size = int(os.getenv("XLOG_EVAL_BATCH_SIZE", "256"))
if train_limit > 0:
    train_dataset = Subset(train_dataset, 0, train_limit)
loader = DataLoader(train_dataset, batch_size)
lr = 1e-4
coin_network1 = smallnet(num_classes=2, pretrained=True)
coin_net1 = Network(coin_network1, "net1", batching=True)
coin_net1.optimizer = torch.optim.Adam(coin_network1.parameters(), lr=lr)
coin_network2 = smallnet(num_classes=2, pretrained=True)
coin_net2 = Network(coin_network2, "net2", batching=True)
coin_net2.optimizer = torch.optim.Adam(coin_network2.parameters(), lr=lr)
require_cuda([coin_net1, coin_net2])

model = Model("model.pl", [coin_net1, coin_net2])
model.add_tensor_source("train", train_dataset)
model.add_tensor_source("test", test_dataset)
model.set_engine(ExactEngine(model), cache=True)
train_obj = train_model(
    model,
    loader,
    StopOnPlateau("Accuracy", warm_up=10, patience=10)
    | Threshold("Accuracy", 1.0, duration=2),
    log_iter=100 // batch_size,
    test_iter=100 // batch_size,
    test=lambda x: [
        ("Accuracy", held_out_accuracy_batched(x, test_dataset, batch_size=eval_batch_size))
    ],
    infoloss=0.25,
)

held_out_accuracy = held_out_accuracy_batched(
    model, test_dataset, batch_size=eval_batch_size
)
print("Held-out Accuracy", held_out_accuracy)
