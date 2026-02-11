from json import dumps
import os
from torch.optim import Adam

from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.examples.HWF.data import HWFDataset, hwf_images
from deepproblog.examples.HWF.network import SymbolEncoder, SymbolClassifier
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from _xlog_runtime import require_cuda, held_out_accuracy_batched

N = 1
method = 'exact'
name = "hwf_{}_{}".format(method, N)
curriculum = False
eval_batch_size = int(os.getenv("XLOG_EVAL_BATCH_SIZE", "256"))
train_batch_size = int(os.getenv("XLOG_TRAIN_BATCH_SIZE", "32"))

print("Training HWF with N={} and curriculum={}".format(N, curriculum))

encoder = SymbolEncoder()
network1 = SymbolClassifier(encoder, 10)
network2 = SymbolClassifier(encoder, 4)

net1 = Network(network1, "net1", Adam(network1.parameters(), lr=3e-3), batching=True)
net2 = Network(network2, "net2", Adam(network2.parameters(), lr=3e-3), batching=True)
require_cuda([net1, net2])

model = Model("model.pl", [net1, net2])
model.add_tensor_source("hwf", hwf_images)

if method == "exact":
    model.set_engine(ExactEngine(model), cache=True)
elif method == "approximate":
    heuristic = ApproximateEngine.geometric_mean
    model.set_engine(ApproximateEngine(model, 1, heuristic, timeout=30, ignore_timeout=True, exploration=True))

try:
    if curriculum:
        dataset = HWFDataset("train2", lambda x: x <= N)
        val_dataset = HWFDataset("val", lambda x: x <= N)
        test_dataset = HWFDataset("test", lambda x: x <= N)
    else:
        dataset = HWFDataset("train2", lambda x: x == N)
        val_dataset = HWFDataset("val", lambda x: x == N)
        test_dataset = HWFDataset("test", lambda x: x == N)
except FileNotFoundError:
    print('The HWD dataset has not been downloaded. See the README.md for info on how to download it.')
    dataset, val_dataset, test_dataset = None, None, None
    exit(1)

loader = DataLoader(dataset, train_batch_size, shuffle=True)

print("Training on size {}".format(N))
train_log = train_model(
    model,
    loader,
    50,
    log_iter=50,
    inital_test=False,
    test_iter=100,
    test=lambda x: [
        (
            "Val_accuracy",
            held_out_accuracy_batched(
                x, val_dataset, batch_size=eval_batch_size, eps=1e-6
            ),
        ),
        (
            "Test_accuracy",
            held_out_accuracy_batched(
                x, test_dataset, batch_size=eval_batch_size, eps=1e-6
            ),
        ),
    ],
)

model.save_state("models/" + name + ".pth")
final_acc = held_out_accuracy_batched(
    model, test_dataset, batch_size=eval_batch_size, eps=1e-6
)
print("Held-out Accuracy", final_acc)
train_log.logger.comment("Accuracy {}".format(final_acc))
train_log.logger.comment(dumps(model.get_hyperparameters()))
train_log.write_to_file("log/" + name)
