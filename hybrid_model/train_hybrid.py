import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from models.hybrid_model import HybridEffV2ResNet
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# TRANSFORMS
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# DATASET (BIGGER SUBSET)
# -------------------------------
full_ds = datasets.ImageFolder(
    "../data/plantnet_300K/train",
    transform=transform
)

# 🔥 50K images (good overnight load)
indices = list(range(len(full_ds)))
random.shuffle(indices)
subset_indices = indices[:50000]

train_ds = Subset(full_ds, subset_indices)

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=0,   # WINDOWS SAFE
    pin_memory=True
)

# -------------------------------
# MODEL (FROZEN BACKBONES)
# -------------------------------
model = HybridEffV2ResNet(
    num_classes=1081,
    freeze_backbones=True
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=5e-4   # stable overnight LR
)

# -------------------------------
# TRAINING
# -------------------------------
EPOCHS = 12   # ~6–7 hours

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f}")

# -------------------------------
# SAVE
# -------------------------------
torch.save(model.state_dict(), "hybrid_model_overnight.pth")
print("✅ Hybrid model saved (OVERNIGHT)")
