import torch.nn as nn
import timm

def build_model(name="resnet50", num_classes=5, pretrained=True):
    """Build a baseline classifier; RETFound integration is isolated here."""
    if name == "resnet50":
        return timm.create_model("resnet50", pretrained=pretrained, num_classes=num_classes)
    if name == "retfound":
        raise NotImplementedError("Load the official RETFound checkpoint before training.")
    raise ValueError(f"Unknown model: {name}")

class AnyDRHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.classifier = nn.Linear(in_features, 1)
    def forward(self, features):
        return self.classifier(features).squeeze(-1)
