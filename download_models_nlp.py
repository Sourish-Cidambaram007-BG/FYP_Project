from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import torch

print("🟡 Device:", "CUDA" if torch.cuda.is_available() else "CPU")

# IndicTrans2
print("⬇ Downloading IndicTrans2...")
AutoTokenizer.from_pretrained(
    "ai4bharat/indictrans2-indic-en-1B",
    trust_remote_code=True
)
AutoModelForSeq2SeqLM.from_pretrained(
    "ai4bharat/indictrans2-indic-en-1B",
    trust_remote_code=True
)

# FLAN-T5
print("⬇ Downloading FLAN-T5...")
AutoTokenizer.from_pretrained("google/flan-t5-base")
AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Sentence Embeddings
print("⬇ Downloading SentenceTransformer...")
SentenceTransformer("all-MiniLM-L6-v2")

print("✅ ALL NLP MODELS DOWNLOADED SUCCESSFULLY")
