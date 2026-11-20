"""Video EF regressors: Kinetics-400 backbones, fc -> 1 scalar (percent 0-100)."""
import torch.nn as nn
from torchvision.models.video import (
    MC3_18_Weights,
    R2Plus1D_18_Weights,
    R3D_18_Weights,
    mc3_18,
    r2plus1d_18,
    r3d_18,
)

BACKBONES = {
    "r2plus1d_18": (r2plus1d_18, R2Plus1D_18_Weights.KINETICS400_V1),
    "mc3_18": (mc3_18, MC3_18_Weights.KINETICS400_V1),
    "r3d_18": (r3d_18, R3D_18_Weights.KINETICS400_V1),
}


def build_model(name="r2plus1d_18", pretrained=True, dropout=0.0):
    ctor, w = BACKBONES[name]
    model = ctor(weights=w if pretrained else None)
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
