import json
import torch
from torchvision import transforms
from PIL import Image
from models.hybrid_model import HybridEffV2ResNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load mapping
with open("../PlantNet-300K/class_idx_to_species_name.json") as f:
    id_to_name = json.load(f)
id_to_name = {int(k): v for k, v in id_to_name.items()}

# Load model
model = HybridEffV2ResNet(num_classes=1081)
model.load_state_dict(torch.load("hybrid_model.pth", map_location=DEVICE))
model.eval().to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

img = Image.open("../neem.jpg").convert("RGB")
x = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    pred = torch.argmax(model(x), dim=1).item()

print("🌿 Hybrid Prediction:", id_to_name[pred])
