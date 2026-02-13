import pandas as pd
import torch
from module3_semantic.embedder import embed

# Load dataset
df = pd.read_excel("data/🍀Dataset - FYP.xlsx")

# Fill NaNs to avoid issues
df = df.fillna("")

# Build semantic text using ALL relevant attributes
df["semantic_text"] = (
    "Plant Name: " + df["English Name"] + ". "
    "Botanical Name: " + df["Botanical name \n"] + ". "
    "Botanical Description: " + df["Botanical description"] + ". "
    "Usages: " + df["Usages"] + ". "
    "Pharmacological Activity: " + df["Pharmacological activity"] + ". "
    "Active Constituents: " + df["Active constituents"] + ". "
    "Tamil Explanation: " + df["Tamil explanation"] + ". "
    "English Explanation: " + df["English Explanation"]
)

# Create embeddings (offline)
embeddings = embed(df["semantic_text"].tolist()).cpu()

# Save embeddings
torch.save(embeddings, "models/embeddings/plant.pt")

print("✅ Embeddings built using ALL dataset attributes")
