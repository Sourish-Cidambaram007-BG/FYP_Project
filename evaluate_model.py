import sys
import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

# =====================================================
# PATH FIX
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLANTNET_DIR = os.path.join(BASE_DIR, "PlantNet-300K")
sys.path.append(PLANTNET_DIR)

from utils import load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# DATA
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

test_ds = datasets.ImageFolder(
    os.path.join(BASE_DIR, "data", "plantnet_300K", "test"),
    transform=transform
)

test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False
)

# -------------------------------
# MODEL (ResNet-18 backend)
# -------------------------------
model = models.resnet18(num_classes=1081)
load_model(
    model,
    os.path.join(
        PLANTNET_DIR,
        "results",
        "fast_resnet18",
        "fast_resnet18_weights_best_acc.tar"
    ),
    use_gpu=torch.cuda.is_available()
)

model.eval().to(DEVICE)

# -------------------------------
# METRICS
# -------------------------------
y_true = []
y_pred = []
top5_correct = 0
total = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)

        _, pred = torch.max(out, 1)
        y_pred.extend(pred.cpu().numpy())
        y_true.extend(y.cpu().numpy())

        # Top-5 accuracy
        top5 = torch.topk(out, k=5, dim=1).indices
        for i in range(y.size(0)):
            if y[i] in top5[i]:
                top5_correct += 1
        total += y.size(0)

# -------------------------------
# RESULTS
# -------------------------------
top1_acc = sum(p == t for p, t in zip(y_pred, y_true)) / total
top5_acc = top5_correct / total

precision = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
f1 = f1_score(y_true, y_pred, average="macro")

print("\n📊 FINAL EVALUATION RESULTS")
print("--------------------------------------------------")
print(f"Top-1 Accuracy   : {top1_acc:.4f}")
print(f"Top-5 Accuracy   : {top5_acc:.4f}")
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro)   : {recall:.4f}")
print(f"F1-Score (Macro) : {f1:.4f}")
print("--------------------------------------------------")
