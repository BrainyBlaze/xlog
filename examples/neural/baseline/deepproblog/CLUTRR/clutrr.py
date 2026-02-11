from json import dumps
import os

from deepproblog.engines import ApproximateEngine
from deepproblog.network import Network
from deepproblog.model import Model
from deepproblog.dataset import DataLoader
from deepproblog.train import TrainObject
from deepproblog.utils.stop_condition import Threshold, StopOnPlateau
from CLUTRR.architecture import Encoder, RelNet, GenderNet
from CLUTRR.data import CLUTRR, CLUTRR_Dataset
from _xlog_runtime import require_cuda, held_out_accuracy_batched
import torch

name = 'sys_gen_0'
clutrr = CLUTRR(name)

embed_size = 32
lstm = Encoder(clutrr.get_vocabulary(), embed_size, p_drop=0.0)
lstm_net = Network(lstm, "encoder", optimizer=torch.optim.Adam(lstm.parameters(), lr=1e-2))
rel_net = Network(RelNet(embed_size, 2 * embed_size), "rel_extract")
rel_net.optimizer = torch.optim.Adam(rel_net.parameters(), lr=1e-2)
gender_net = GenderNet(clutrr.get_vocabulary(), embed_size)
gender_net = Network(gender_net, "gender_net", optimizer=torch.optim.Adam(gender_net.parameters(), lr=1e-2))
require_cuda([rel_net, lstm_net, gender_net])

model_filename = "model_forward.pl"
model = Model(model_filename, [rel_net, lstm_net, gender_net])
model.set_engine(ApproximateEngine(model, 1, ApproximateEngine.geometric_mean, exploration=True))

dataset: CLUTRR_Dataset = clutrr.get_dataset(".*train", gender=True, query_type="split")
train_limit = int(os.getenv("XLOG_TRAIN_LIMIT", "0"))
eval_batch_size = int(os.getenv("XLOG_EVAL_BATCH_SIZE", "256"))
train_batch_size = int(os.getenv("XLOG_TRAIN_BATCH_SIZE", "64"))
if train_limit > 0:
    dataset = dataset.subset(0, train_limit)
val_dataset = dataset.subset(100)
test_datasets = clutrr.get_dataset(".*test", gender=True, query_type="split", separate=True)

loader = DataLoader(dataset, train_batch_size)

train_log = TrainObject(model)
train_log.train(
    loader,
    Threshold("Accuracy", 1.0) + StopOnPlateau("Accuracy", patience=5, warm_up=10),
    initial_test=False,
    test=lambda x: [
        ("Accuracy", held_out_accuracy_batched(x, val_dataset, batch_size=eval_batch_size),)
    ],
    log_iter=50,
    test_iter=250,
)

model.save_state("models/" + name + ".pth")

for dataset in test_datasets:
    final_acc = held_out_accuracy_batched(
        model, test_datasets[dataset], batch_size=eval_batch_size
    )
    train_log.logger.comment("{}\t{}".format(dataset, final_acc))
    print("Held-out Accuracy", dataset, final_acc)
train_log.logger.comment(dumps(model.get_hyperparameters()))
train_log.write_to_file("log/" + name)
