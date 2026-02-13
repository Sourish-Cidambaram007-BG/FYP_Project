from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import torch

# Create FastAPI instance
app = FastAPI()

# ===============================
# Request Schema
# ===============================

class TextRequest(BaseModel):
    text: str


# ===============================
# Root Check
# ===============================

@app.get("/")
def health_check():
    return {"status": "NLP Service Running"}


# ===============================
# Text Processing Endpoint
# ===============================

@app.post("/process")
def process_text(request: TextRequest):
    user_text = request.text

    # ---- Replace this block with your real NLP pipeline ----
    # For now simple demo logic
    response_text = f"You said: {user_text}"

    return {
        "input": user_text,
        "response": response_text
    }
