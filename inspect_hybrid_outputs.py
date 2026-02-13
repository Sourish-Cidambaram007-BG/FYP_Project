import torch
import torch.nn as nn
from torchvision import models

# -----------------------------
# DEVICE
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LIGHTWEIGHT CNN BRANCH
# (Efficient-style)
# -----------------------------
class LightweightCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.features(x)


# -----------------------------
# RESIDUAL CNN BRANCH
# (DO NOT NAME IT)
# -----------------------------
residual_backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
residual_backbone = nn.Sequential(*list(residual_backbone.children())[:-2])
residual_backbone = residual_backbone.to(DEVICE)
residual_backbone.eval()


# -----------------------------
# HYBRID MODEL
# -----------------------------
class HybridCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lightweight = LightweightCNN()
        self.residual = residual_backbone

        self.fusion = nn.Sequential(
            nn.Linear(64 + 512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )

    def forward(self, x):
        outputs = {}

        # Input
        outputs["Input Image"] = x

        # Lightweight branch
        lw = self.lightweight(x)
        outputs["Lightweight Feature Maps"] = lw

        lw = lw.view(lw.size(0), -1)
        outputs["Lightweight Global Features"] = lw

        # Residual branch
        rb = self.residual(x)
        outputs["Residual Feature Maps"] = rb

        rb = torch.nn.functional.adaptive_avg_pool2d(rb, (1, 1))
        rb = rb.view(rb.size(0), -1)
        outputs["Residual Global Features"] = rb

        # Feature fusion
        fused = torch.cat([lw, rb], dim=1)
        outputs["Feature Fusion Vector"] = fused

        fused = self.fusion(fused)
        outputs["Post-Fusion Features"] = fused

        return outputs


# -----------------------------
# RUN INSPECTION
# -----------------------------
model = HybridCNN().to(DEVICE)
model.eval()

x = torch.randn(1, 3, 224, 224).to(DEVICE)

with torch.no_grad():
    outputs = model(x)

print("\n📊 HYBRID MODEL – BOX-WISE OUTPUT SHAPES\n" + "-" * 55)

for name, tensor in outputs.items():
    print(f"{name:35s} → shape: {tuple(tensor.shape)}")
