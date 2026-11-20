"""R(2+1)D-18 EF regressor: Kinetics-400 backbone, fc -> 1 scalar (percent 0-100)."""
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


def build_model(pretrained=True, dropout=0.0):
    weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
    model = r2plus1d_18(weights=weights)
    in_features = model.fc.in_features  # 512
    layers = []
    if dropout and dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(in_features, 1))
    model.fc = nn.Sequential(*layers) if len(layers) > 1 else layers[0]
    # Small init on the new head (percent-scale target).
    with __import__("torch").no_grad():
        head = model.fc[-1] if isinstance(model.fc, nn.Sequential) else model.fc
        nn.init.normal_(head.weight, std=0.01)
        nn.init.zeros_(head.bias)
    return model
