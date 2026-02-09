import torch


class RelNet(torch.nn.Module):
    def __init__(self, vocab_size: int, out_dim: int):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, 128, padding_idx=0)
        self.encoder = torch.nn.GRU(
            input_size=128,
            hidden_size=96,
            batch_first=True,
            bidirectional=True,
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(192, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, out_dim),
        )

    def forward(self, x):
        emb = self.emb(x)
        _, h = self.encoder(emb)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return torch.softmax(self.head(h), dim=-1)
