import torch
import torch.nn.functional as F
from torchvision import models


class CardNet(torch.nn.Module):
    def __init__(self, out_dim: int, pretrained: bool = True, focus: str = "full"):
        super().__init__()
        self.focus = focus
        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except Exception:
                weights = None
        self.backbone = models.resnet18(weights=weights)
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
