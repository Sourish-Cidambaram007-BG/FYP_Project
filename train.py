import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.hybrid_model import HybridEffV2ResNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# DATA
# -------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_ds = datasets.ImageFolder(
    "data/plantnet_300K/images_train",
    transform=train_transform
)

val_ds = datasets.ImageFolder(
    "data/plantnet_300K/images_val",
    transform=val_transform
)

train_loader = DataLoader(
    train_ds, batch_size=32, shuffle=True, num_workers=8, pin_memory=True
)

val_loader = DataLoader(
    val_ds, batch_size=32, shuffle=False, num_workers=8, pin_memory=True
)

# -------------------------------
# MODEL
# -------------------------------
model = HybridEffV2ResNet(num_classes=len(train_ds.classes))
model.to(DEVICE)

# Freeze backbones (FAST TRAIN)
for p in model.effnet.parameters():
    p.requires_grad = False
for p in model.resnet.parameters():
    p.requires_grad = False

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(
    model.classifier.parameters(), lr=1e-3
)

scaler = torch.cuda.amp.GradScaler()

# -------------------------------
# TRAIN
# -------------------------------
EPOCHS = 5   # fast & enough for demo + report

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "hybrid_model.pth")
print("✅ Hybrid model saved successfully")
