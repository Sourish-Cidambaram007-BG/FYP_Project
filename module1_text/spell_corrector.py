from difflib import get_close_matches
import pandas as pd

# Load plant names once
df = pd.read_excel("data/🍀Dataset - FYP.xlsx").fillna("")
PLANT_NAMES = df["English Name"].str.lower().tolist()

def correct_plant_spelling(text: str):
    """
    Correct misspelled plant names using fuzzy matching.
    """
    words = text.lower().split()
    corrected_words = []

    for word in words:
        match = get_close_matches(word, PLANT_NAMES, n=1, cutoff=0.75)
        if match:
            corrected_words.append(match[0])
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)
