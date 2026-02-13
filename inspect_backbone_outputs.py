import torch
from torchvision import models

# -----------------------------
# DEVICE
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD BACKBONE (DO NOT NAME IT)
# -----------------------------
backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
backbone = backbone.to(DEVICE)
backbone.eval()

# -----------------------------
# DUMMY INPUT
# -----------------------------
x = torch.randn(1, 3, 224, 224).to(DEVICE)

# -----------------------------
# FORWARD STEP-BY-STEP
# -----------------------------
outputs = {}

with torch.no_grad():
    # Input
    outputs["Input Image"] = x

    # Initial Feature Extraction
    x = backbone.conv1(x)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    outputs["Convolutional Stem"] = x

    x = backbone.maxpool(x)
    outputs["Downsampling Block"] = x

    # Residual Learning Stages
    x = backbone.layer1(x)
    outputs["Residual Feature Block 1"] = x

    x = backbone.layer2(x)
    outputs["Residual Feature Block 2"] = x

    x = backbone.layer3(x)
    outputs["Residual Feature Block 3"] = x

    x = backbone.layer4(x)
    outputs["Residual Feature Block 4"] = x

    # Global Feature Aggregation
    x = backbone.avgpool(x)
    x = torch.flatten(x, 1)
    outputs["Global Average Pooling"] = x

# -----------------------------
# DISPLAY OUTPUT SHAPES
# -----------------------------
print("\n📊 Backbone Feature Map Outputs\n" + "-" * 45)

for name, tensor in outputs.items():
    print(f"{name:30s} → shape: {tuple(tensor.shape)}")
