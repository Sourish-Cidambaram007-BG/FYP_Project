import os
import torch
from sentence_transformers import util

# =====================================================
# RESOLVE PROJECT ROOT SAFELY
# =====================================================
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

EMBEDDING_PATH = os.path.join(
    PROJECT_ROOT, "models", "embeddings", "plant.pt"
)

# =====================================================
# LOAD DATABASE EMBEDDINGS (CPU SAFE)
# =====================================================
if not os.path.exists(EMBEDDING_PATH):
    raise FileNotFoundError(
        f"❌ plant.pt not found at: {EMBEDDING_PATH}"
    )

db_embeddings = torch.load(
    EMBEDDING_PATH,
    map_location="cpu"
)

# =====================================================
# SEMANTIC SEARCH (TOP-K SUPPORT)
# =====================================================
def semantic_search(query_embedding, top_k=1):
    """
    Args:
        query_embedding (torch.Tensor): shape (1, dim)
        top_k (int): number of top matches

    Returns:
        list[int] if top_k > 1
        int       if top_k == 1
    """

    query_embedding = query_embedding.cpu()

    # Cosine similarity
    scores = util.cos_sim(query_embedding, db_embeddings)[0]

    # Top-K results
    top_results = torch.topk(scores, k=top_k)

    if top_k == 1:
        return top_results.indices.item()

    return top_results.indices.tolist()
