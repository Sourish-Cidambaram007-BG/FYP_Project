import json
import torch
from torchvision import transforms
from PIL import Image
from models.hybrid_model import HybridEffV2ResNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# LOAD CLASS → SPECIES NAME MAP
# -------------------------------
with open("../PlantNet-300K/class_idx_to_species_name.json", "r") as f:
    id_to_name = json.load(f)

# keys come as strings → convert to int
id_to_name = {int(k): v for k, v in id_to_name.items()}

# -------------------------------
# LOAD HYBRID MODEL (OVERNIGHT)
# -------------------------------
model = HybridEffV2ResNet(
    num_classes=1081,
    freeze_backbones=True
)

model.load_state_dict(
    torch.load("hybrid_model_overnight.pth", map_location=DEVICE)
)

model.eval().to(DEVICE)

# -------------------------------
# IMAGE TRANSFORM
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
# LOAD IMAGE
# -------------------------------
IMAGE_PATH = "../carrot.jpg"   # change to neem.jpg if needed
img = Image.open(IMAGE_PATH).convert("RGB")

x = transform(img).unsqueeze(0).to(DEVICE)

# -------------------------------
# INFERENCE
# -------------------------------
with torch.no_grad():
    logits = model(x)
    pred_id = torch.argmax(logits, dim=1).item()

plant_name = id_to_name.get(pred_id, "Unknown species")

print("🌿 Hybrid Prediction :", plant_name)
