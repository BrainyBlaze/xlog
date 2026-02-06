import torch


class RelNet(torch.nn.Module):
    def __init__(self, vocab_size: int, out_dim: int):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, 64)
        self.fc = torch.nn.Linear(64, out_dim)

    def forward(self, x):
        h = self.emb(x).mean(dim=1)
        return torch.softmax(self.fc(h), dim=-1)
