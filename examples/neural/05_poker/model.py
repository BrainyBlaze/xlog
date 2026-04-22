import torch
import torch.nn.functional as F

try:
    from torchvision import models as tv_models
except ImportError:
    tv_models = None


class _FallbackCardBackbone(torch.nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = torch.nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class CardNet(torch.nn.Module):
    def __init__(self, out_dim: int, pretrained: bool = True, focus: str = "full"):
        super().__init__()
        self.focus = focus
        if tv_models is None:
            self.backbone = _FallbackCardBackbone(out_dim)
        else:
            weights = None
            if pretrained:
                try:
                    weights = tv_models.ResNet18_Weights.DEFAULT
                except Exception:
                    weights = None
            self.backbone = tv_models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = torch.nn.Linear(in_features, out_dim)

            # Keep fine-tuning focused and stable on relatively small card corpora.
            for name, parameter in self.backbone.named_parameters():
                if name.startswith("layer4") or name.startswith("fc"):
                    parameter.requires_grad = True
                else:
                    parameter.requires_grad = False

    def forward(self, x):
        if self.focus == "rank":
            h = x.shape[2]
            w = x.shape[3]
            x = x[:, :, : max(8, int(h * 0.5)), : max(8, int(w * 0.45))]
            x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        elif self.focus == "suit":
            h = x.shape[2]
            w = x.shape[3]
            x = x[:, :, : max(8, int(h * 0.55)), : max(8, int(w * 0.5))]
            x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        logits = self.backbone(x)
        return torch.softmax(logits, dim=-1)
