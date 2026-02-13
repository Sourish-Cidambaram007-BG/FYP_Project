import pandas as pd
from module3_semantic.embedder import embed
from module3_semantic.search import semantic_search

# Load dataset once
df = pd.read_excel("data/🍀Dataset - FYP.xlsx").fillna("")

def detect_plant_semantic(query: str):
    """
    Detect plant entity from user query using semantic similarity.

    Args:
        query (str): Cleaned English user query

    Returns:
        plant_id (int): index of matched plant
        plant_name (str): English name of plant
    """

    # Convert query to embedding
    query_embedding = embed([query]).cpu()

    # Semantic search (TOP-1)
    plant_id = semantic_search(query_embedding)

    plant_name = df.iloc[plant_id]["English Name"]

    return plant_id, plant_name
