# Movie Recommender - Hybrid AI/ML System (SBERT + Neural CF)
AI powered, modern fullstack system using semantic embeddings, 
neural collaborative filtering and a MovieLens + TMDB enriched dataset.

## Features
### 🎥 Hybrid Recommendation Engine
- combines content-based and collaborative filtering signals using SBERT (Sentence-Bert) and Neural Collaborative filtering.
### 🧠 Advanced Search System
- Autocomplete with live results
- Regex-safe literal substring matching
- Auto-update dropdown features
### 🔎 Explore Movies
### 📀 MovieLens + TMDB Combined Dataset
### 🎨 Modern Frontend
### ⚡ FastAPI Backend
- /recommend: hybrid SBERT + NCF recommendations
- /search-movies: fast literal substring search
- /movies: paginated catalog
- /poster: TMDB poster lookup
- /explore: filtered movie browsing

## Tech Stack
### Backend
- FastAPI
- Python 3
- PyTorch
- TMDB API
- NumPy / Pandas / SciPy

### Frontend
- HTML 5
- CSS 3
- JavaScript
  
### DevOps
- Dockerfile

## 🏛 Architecture
 ┌───────────────────┐        ┌─────────────────────────┐
 │   Frontend (JS)    │──────▶│ FastAPI Backend          │
 │  - search UI        │       │ - recommend endpoint     │
 │  - explore grid     │       │ - search_movies endpoint │
 └───────────────────┘        └──────────┬──────────────┘
                                          │
                          ┌───────────────▼────────────────┐
                          │ Hybrid Engine (SBERT + NCF)     │
                          │ - SBERT embeddings               │
                          │ - Neural CF user-item matrix     │
                          └───────────────┬────────────────┘
                                          │
                              ┌───────────▼──────────────┐
                              │ MovieLens + TMDB Dataset  │
                              └───────────────────────────┘

## 🚀 Getting Started
- Clone repository and navigate to folder: cd movie-recommender/backend
- Install all dependencies: pip install -r requirements.txt
- Build datasets: python build_combined_movies.py
- Train models: python train_sbert.py & python train_ncf.py
- Start backend & frontend: uvicorn app.main:app --reload  &  python -m http.server 5500 (http://localhost:5500)
                        
## 📌 Future Enhancements
- User accounts + personalised profiles
- Search using SBERT embeddings
- Similarity clusters for genres
- Deployment on Render

## 🤝 Contributing
Pull requests are welcome!
Open an issue for bugs or feature requests.
