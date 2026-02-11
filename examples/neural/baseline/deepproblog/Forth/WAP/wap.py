import os

from deepproblog.dataset import DataLoader, QueryDataset
from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from Forth.WAP.wap_network import get_networks
from _xlog_runtime import require_cuda, held_out_accuracy_batched

train_queries = QueryDataset("data/train.pl")
dev_queries = QueryDataset("data/dev.pl")
test_queries = QueryDataset("data/test.pl")
train_limit = int(os.getenv("XLOG_TRAIN_LIMIT", "0"))
eval_batch_size = int(os.getenv("XLOG_EVAL_BATCH_SIZE", "256"))
train_batch_size = int(os.getenv("XLOG_TRAIN_BATCH_SIZE", "200"))
if train_limit > 0:
    train_queries = train_queries.subset(0, train_limit)

networks = get_networks(0.005, 0.5)

train_networks = [Network(x[0], x[1], x[2]) for x in networks]
test_networks = [Network(networks[0][0], networks[0][1])] + [
    Network(x[0], x[1], k=1) for x in networks[1:]
]
require_cuda(train_networks + test_networks)

model = Model("wap.pl", train_networks)
model.set_engine(ExactEngine(model), cache=True)

test_model = Model("wap.pl", test_networks)
test_model.set_engine(ExactEngine(test_model), cache=False)

train_obj = train_model(
    model,
    DataLoader(train_queries, train_batch_size),
    40,
    log_iter=10,
    test=lambda x: [
        (
            "Accuracy",
            held_out_accuracy_batched(
                test_model, test_queries, batch_size=eval_batch_size
            ),
        )
    ],
    test_iter=30,
)

held_out_accuracy = held_out_accuracy_batched(
    test_model, test_queries, batch_size=eval_batch_size
)
print("Held-out Accuracy", held_out_accuracy)
