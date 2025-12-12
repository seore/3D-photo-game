from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MOVIES_COMBINED_PATH = DATA_DIR / "movies_combined.csv"
MOVIES_PATH = MOVIES_COMBINED_PATH if MOVIES_COMBINED_PATH.exists() else (DATA_DIR / "movies.csv")
SBERT_OUT_PATH = MODELS_DIR / "sbert_embeddings.npz"


def main():
    print(f"Loading movies from {MOVIES_PATH}")
    movies = pd.read_csv(MOVIES_PATH)
    print(f"Loaded {len(movies)} movies")

    # Build descriptive text: genres + overview + title
    genres = movies["genres"].fillna("") if "genres" in movies.columns else ""
    if "overview" in movies.columns:
        overview = movies["overview"].fillna("")
    elif "description" in movies.columns:
        overview = movies["description"].fillna("")
    else:
        overview = movies["title"].fillna("")

    text = (
        genres.astype(str)
        + " | "
        + movies["title"].fillna("").astype(str)
        + " - "
        + overview.astype(str)
    ).tolist()

    print("Loading SBERT model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Encoding movie descriptions...")
    embeddings = model.encode(
        text,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True, 
    )

    print("Embeddings shape:", embeddings.shape)
    np.savez(SBERT_OUT_PATH, embeddings=embeddings)
    print(f"Saved SBERT embeddings to {SBERT_OUT_PATH}")


if __name__ == "__main__":
    main()
