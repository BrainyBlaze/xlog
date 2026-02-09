import torch


class DigitNet(torch.nn.Module):
    def __init__(self, side: str = "full"):
        super().__init__()
        if side not in {"left", "right", "full"}:
            raise ValueError("side must be one of: left, right, full")
        self.side = side
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 96, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = torch.nn.Linear(96, 10)

    def _select_region(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] >= 6:
            if self.side == "left":
                return x[:, :3, :, :]
            if self.side == "right":
                return x[:, 3:6, :, :]
            return x[:, :3, :, :]
        return x

    def forward(self, x):
        x = self._select_region(x)
        x = self.net(x).view(x.size(0), -1)
        return torch.softmax(self.fc(x), dim=-1)
