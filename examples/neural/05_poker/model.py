import torch


class CardNet(torch.nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 96, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = torch.nn.Linear(96, out_dim)

    def forward(self, x):
        x = self.net(x).view(x.size(0), -1)
        return torch.softmax(self.fc(x), dim=-1)
