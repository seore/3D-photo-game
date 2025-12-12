from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

RATINGS_PATH = DATA_DIR / "ratings.csv"
NCF_ITEM_EMB_PATH = MODELS_DIR / "ncf_item_embeddings.npz"


class RatingsDataset(Dataset):
    def __init__(self, user_idx, movie_idx, ratings):
        self.user_idx = user_idx
        self.movie_idx = movie_idx
        self.ratings = ratings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (
            self.user_idx[idx],
            self.movie_idx[idx],
            self.ratings[idx],
        )


class NCF(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, user_idx, item_idx):
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        x = torch.cat([u, i], dim=-1)
        out = self.mlp(x).squeeze(-1)
        return out


def main():
    print(f"Loading ratings from {RATINGS_PATH}")
    ratings = pd.read_csv(RATINGS_PATH)
    required = {"userId", "movieId", "rating"}
    if not required.issubset(ratings.columns):
        raise ValueError(f"ratings.csv must contain {required}")

    user_enc = LabelEncoder()
    movie_enc = LabelEncoder()

    ratings["user_idx"] = user_enc.fit_transform(ratings["userId"])
    ratings["movie_idx"] = movie_enc.fit_transform(ratings["movieId"])

    num_users = ratings["user_idx"].nunique()
    num_items = ratings["movie_idx"].nunique()
    print("Users:", num_users, "Movies:", num_items)

    # Normalise rating to ~0–1
    r = ratings["rating"].astype("float32").values
    r_norm = (r - r.min()) / (r.max() - r.min() + 1e-6)

    ds = RatingsDataset(
        user_idx=torch.tensor(ratings["user_idx"].values, dtype=torch.long),
        movie_idx=torch.tensor(ratings["movie_idx"].values, dtype=torch.long),
        ratings=torch.tensor(r_norm, dtype=torch.float32),
    )

    dl = DataLoader(ds, batch_size=4096, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NCF(num_users, num_items, emb_dim=32).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print("Training NCF (a few epochs for demo)...")
    model.train()
    epochs = 3  
    for epoch in range(epochs):
        total_loss = 0.0
        for user_batch, item_batch, rating_batch in dl:
            user_batch = user_batch.to(device)
            item_batch = item_batch.to(device)
            rating_batch = rating_batch.to(device)

            optim.zero_grad()
            preds = model(user_batch, item_batch)
            loss = loss_fn(preds, rating_batch)
            loss.backward()
            optim.step()
            total_loss += loss.item() * len(user_batch)

        avg_loss = total_loss / len(ds)
        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    # Extract item embeddings
    item_emb = model.item_emb.weight.detach().cpu().numpy()
    movie_ids = movie_enc.inverse_transform(np.arange(num_items)).astype(int)

    np.savez(NCF_ITEM_EMB_PATH, movie_ids=movie_ids, embeddings=item_emb)
    print(f"Saved NCF item embeddings to {NCF_ITEM_EMB_PATH}")


if __name__ == "__main__":
    main()
