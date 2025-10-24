"""
Movie Recommendation System
Content-based + Collaborative Filtering
- Uses TF-IDF + cosine similarity for content-based recommendations (plot, genres, directors, cast)
- Uses user-item rating matrix + KNN (cosine) for simple collaborative recommendations
- Fetches movie data directly from OMDb API using search functionality
- Generates synthetic ratings for demonstration purposes

Requirements:
- python 3.8+
- pandas, numpy, scikit-learn, requests, matplotlib, seaborn, scipy, python-dotenv

Example usage:
>>> python MRS.py --demo

This single-module implementation is intended as a base for a small project or prototype.
Feel free to split into modules, add caching, unit tests, or a web front-end.

"""

import argparse
import json
import os
import time
import random
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from dotenv import load_dotenv

# Load .env variables if present
load_dotenv()

# ---------- OMDb API functions ----------

def search_movies_omdb(search_term: str, api_key: str, page: int = 1) -> Optional[dict]:
    """Search for movies using OMDb API"""
    url = "http://www.omdbapi.com/"
    params = {"s": search_term, "apikey": api_key, "type": "movie", "page": page}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("Response") == "True":
            return data
        else:
            print(f"Search failed: {data.get('Error', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"OMDb search request failed: {e}")
        return None

def fetch_omdb_by_imdb_id(imdb_id: str, api_key: str, sleep: float = 0.1) -> Optional[dict]:
    """Fetch movie metadata from OMDb for a given IMDb ID.
    Returns None on failure.
    Use a local cache to avoid repeated requests when iterating.
    """
    url = "http://www.omdbapi.com/"
    params = {"i": imdb_id, "apikey": api_key, "plot": "full", "r": "json"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("Response") == "True":
            time.sleep(sleep)
            return data
        else:
            return None
    except Exception as e:
        print(f"OMDb request failed for {imdb_id}: {e}")
        return None

def fetch_movie_collection(api_key: str, search_terms: List[str] = None, max_movies: int = 100) -> pd.DataFrame:
    """Fetch a collection of movies from OMDb API"""
    if search_terms is None:
        # Default search terms for popular movies
        search_terms = [
            "Marvel", "Batman", "Superman", "Star Wars", "Lord of the Rings", 
            "Harry Potter", "Fast and Furious", "Mission Impossible", "James Bond",
            "Avengers", "Spider-Man", "Iron Man", "Thor", "Captain America",
            "Transformers", "Jurassic", "Indiana Jones", "Terminator", "Matrix",
            "Godfather", "Pulp Fiction", "Forrest Gump", "Titanic", "Avatar",
            "Comedy", "Horror", "Action", "Drama", "Romance", "Thriller"
        ]
    
    all_movies = []
    cache_path = 'omdb_movie_cache.json'
    cache = {}
    
    # Load existing cache
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cache = json.load(f)
    
    print(f"Fetching movies from OMDb API...")
    
    for search_term in search_terms:
        if len(all_movies) >= max_movies:
            break
            
        print(f"Searching for: {search_term}")
        
        # Search for movies
        search_results = search_movies_omdb(search_term, api_key)
        if not search_results or 'Search' not in search_results:
            continue
            
        for movie_basic in search_results['Search']:
            if len(all_movies) >= max_movies:
                break
                
            imdb_id = movie_basic.get('imdbID')
            if not imdb_id:
                continue
                
            # Check cache first
            if imdb_id in cache:
                movie_data = cache[imdb_id]
            else:
                # Fetch detailed movie data
                movie_data = fetch_omdb_by_imdb_id(imdb_id, api_key)
                if movie_data:
                    cache[imdb_id] = movie_data
                else:
                    continue
            
            # Process movie data
            movie_info = {
                'movieId': len(all_movies) + 1,
                'imdbId': imdb_id,
                'title': movie_data.get('Title', ''),
                'year': movie_data.get('Year', ''),
                'genres': movie_data.get('Genre', '').replace(', ', '|'),
                'director': movie_data.get('Director', ''),
                'actors': movie_data.get('Actors', ''),
                'plot': movie_data.get('Plot', ''),
                'rating': movie_data.get('imdbRating', ''),
                'runtime': movie_data.get('Runtime', ''),
                'language': movie_data.get('Language', ''),
                'country': movie_data.get('Country', ''),
                'poster': movie_data.get('Poster', '')
            }
            
            # Avoid duplicates
            if not any(m['imdbId'] == imdb_id for m in all_movies):
                all_movies.append(movie_info)
                print(f"Added: {movie_info['title']} ({movie_info['year']})")
    
    # Save updated cache
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    
    print(f"Fetched {len(all_movies)} movies total")
    return pd.DataFrame(all_movies)

def generate_synthetic_ratings(movies_df: pd.DataFrame, num_users: int = 50, ratings_per_user: int = 20) -> pd.DataFrame:
    """Generate synthetic rating data for demonstration"""
    print(f"Generating synthetic ratings for {num_users} users...")
    
    ratings_data = []
    movie_ids = movies_df['movieId'].tolist()
    
    for user_id in range(1, num_users + 1):
        # Each user rates a random subset of movies
        user_movies = random.sample(movie_ids, min(ratings_per_user, len(movie_ids)))
        
        for movie_id in user_movies:
            # Generate ratings based on movie's IMDb rating with some noise
            movie_info = movies_df[movies_df['movieId'] == movie_id].iloc[0]
            base_rating = movie_info.get('rating', '7.0')
            
            try:
                base_rating = float(base_rating)
                # Convert IMDb 10-point scale to 5-point scale with noise
                rating = (base_rating / 2) + random.uniform(-1, 1)
                rating = max(1, min(5, rating))  # Clamp between 1-5
            except:
                rating = random.uniform(2, 4.5)
            
            ratings_data.append({
                'userId': user_id,
                'movieId': movie_id,
                'rating': round(rating, 1),
                'timestamp': int(time.time()) + random.randint(-1000000, 0)
            })
    
    return pd.DataFrame(ratings_data)

# ---------- Data preparation utilities ----------

def load_movies_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print(f"CSV file {path} not found. Will fetch from OMDb API instead.")
        return pd.DataFrame()

def load_ratings_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print(f"CSV file {path} not found. Will generate synthetic ratings.")
        return pd.DataFrame()

def normalize_imdb_id(imdb_raw: str) -> str:
    if pd.isna(imdb_raw):
        return ""
    s = str(imdb_raw).strip()
    if s.startswith("tt"):
        return s
    s = s.zfill(7)
    return f"tt{s}"

# ---------- Feature engineering (Content) ----------

def build_content_field(df: pd.DataFrame, fields: List[str]) -> pd.Series:
    def safe_join(row):
        pieces = []
        for f in fields:
            val = row.get(f, "")
            if pd.isna(val):
                continue
            if isinstance(val, (list, tuple)):
                pieces.append(" ".join([str(x) for x in val]))
            else:
                pieces.append(str(val))
        return " ".join(pieces)

    return df.apply(safe_join, axis=1)

def compute_tfidf_matrix(texts: pd.Series, max_features: int = 20000) -> Tuple[TfidfVectorizer, csr_matrix]:
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(texts.fillna(""))
    return vectorizer, tfidf_matrix

# ---------- Content-based recommender ----------

def get_content_recommendations(movie_title: str, movies_df: pd.DataFrame, tfidf_matrix: csr_matrix, top_n: int = 10) -> List[Tuple[int, str, float]]:
    matches = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
    if matches.empty:
        matches = movies_df[movies_df['title'].str.lower().str.contains(movie_title.lower())]
    if matches.empty:
        raise ValueError(f"Movie title '{movie_title}' not found in dataset.")

    idx = matches.index[0]
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sims[idx] = -1
    top_idx = np.argsort(sims)[::-1][:top_n]
    results = []
    for i in top_idx:
        results.append((int(movies_df.loc[i, 'movieId']), movies_df.loc[i, 'title'], float(sims[i])))
    return results

# ---------- Collaborative filtering (user-based KNN) ----------

def build_user_item_matrix(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> Tuple[pd.DataFrame, csr_matrix, dict, dict]:
    merged = ratings_df.merge(movies_df[['movieId', 'title']], on='movieId', how='left')
    pivot = merged.pivot_table(index='userId', columns='movieId', values='rating')
    sparse = csr_matrix(pivot.fillna(0).values)
    user_to_index = {u: i for i, u in enumerate(pivot.index)}
    item_to_index = {mid: j for j, mid in enumerate(pivot.columns)}
    return pivot, sparse, user_to_index, item_to_index

def get_user_based_recommendations(user_id: int, pivot_df: pd.DataFrame, sparse_matrix: csr_matrix, movies_df: pd.DataFrame, user_to_index: dict, item_to_index: dict, n_neighbors: int = 5, top_n: int = 10) -> List[Tuple[int, str, float]]:
    if user_id not in user_to_index:
        raise ValueError(f"User {user_id} not found in rating data.")

    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='cosine', algorithm='brute')
    knn.fit(sparse_matrix)
    uid = user_to_index[user_id]
    distances, indices = knn.kneighbors(sparse_matrix[uid], return_distance=True)
    neigh_idx = indices.flatten()[1:]
    neigh_dist = distances.flatten()[1:]

    neighbor_users = [pivot_df.index[i] for i in neigh_idx]
    neighbor_weights = 1 - neigh_dist

    neighbor_ratings = pivot_df.loc[neighbor_users].fillna(0)
    weighted = neighbor_ratings.T.dot(neighbor_weights)
    weighted = weighted / (np.sum(neighbor_weights) + 1e-9)

    user_rated = pivot_df.loc[user_id].dropna().index
    candidate_scores = weighted.drop(index=user_rated, errors='ignore')
    candidate_scores = candidate_scores.sort_values(ascending=False)
    results = []
    for movieId, score in candidate_scores.head(top_n).items():
        title = movies_df.loc[movies_df['movieId'] == int(movieId), 'title'].values
        t = title[0] if len(title) > 0 else "Unknown"
        results.append((int(movieId), t, float(score)))
    return results

# ---------- Visualization ----------

def plot_user_similarity_heatmap(pivot_df: pd.DataFrame, sample_users: Optional[List[int]] = None, figsize=(10, 8)) -> None:
    from sklearn.metrics.pairwise import cosine_similarity as cos

    df = pivot_df
    if sample_users is not None:
        df = df.loc[df.index.isin(sample_users)]
    filled = df.fillna(0)
    sim = cos(filled.values)
    plt.figure(figsize=figsize)
    sns.heatmap(sim, xticklabels=df.index.astype(str), yticklabels=df.index.astype(str), cmap='viridis')
    plt.title('User-User Similarity (Cosine)')
    plt.xlabel('User')
    plt.ylabel('User')
    plt.tight_layout()
    plt.show()

# ---------- Enhanced demo pipeline ----------

def demo_pipeline(movies_csv: str, ratings_csv: str, omdb_api_key: Optional[str] = None):
    print("=== Movie Recommendation System Demo ===")
    
    # Get API key
    if not omdb_api_key:
        omdb_api_key = os.getenv("OMDB_API_KEY")
    
    if not omdb_api_key:
        print("ERROR: OMDb API key not found!")
        print("Please set OMDB_API_KEY in .env file or pass via --omdb-key")
        return
    
    print(f"Using OMDb API key: {omdb_api_key[:8]}...")
    
    # Try to load existing CSVs, otherwise fetch from API
    movies = load_movies_csv(movies_csv)
    ratings = load_ratings_csv(ratings_csv)
    
    if movies.empty:
        print("Fetching movie data from OMDb API...")
        movies = fetch_movie_collection(omdb_api_key, max_movies=100)
        if movies.empty:
            print("Failed to fetch movies from API!")
            return
        # Save for future use
        movies.to_csv(movies_csv, index=False)
        print(f"Saved {len(movies)} movies to {movies_csv}")
    
    if ratings.empty:
        print("Generating synthetic ratings...")
        ratings = generate_synthetic_ratings(movies, num_users=50, ratings_per_user=15)
        if ratings.empty:
            print("Failed to generate ratings!")
            return
        # Save for future use
        ratings.to_csv(ratings_csv, index=False)
        print(f"Saved {len(ratings)} ratings to {ratings_csv}")
    
    print(f"Dataset: {len(movies)} movies, {len(ratings)} ratings from {ratings['userId'].nunique()} users")
    
    # Build content features
    print("\nBuilding content features...")
    combine_fields = ['title', 'genres', 'plot', 'director', 'actors']
    movies['content'] = build_content_field(movies, combine_fields)
    vec, tfidf = compute_tfidf_matrix(movies['content'], max_features=5000)
    print(f"TF-IDF matrix shape: {tfidf.shape}")
    
    # Content-based recommendations demo
    print("\n=== Content-Based Recommendations Demo ===")
    sample_movies = movies['title'].head(3).tolist()
    print(f"Available movies (first 3): {sample_movies}")
    
    sample_title = sample_movies[0]
    print(f"\nContent-based recommendations for '{sample_title}':")
    try:
        recs = get_content_recommendations(sample_title, movies, tfidf, top_n=5)
        for i, (mid, title, score) in enumerate(recs, 1):
            print(f"{i}. {title} (similarity: {score:.3f})")
    except Exception as e:
        print(f"Error: {e}")
    
    # Collaborative filtering demo
    print("\n=== Collaborative Filtering Demo ===")
    print("Building user-item matrix...")
    pivot_df, sparse_mat, user_to_idx, item_to_idx = build_user_item_matrix(ratings, movies)
    print(f"User-item matrix shape: {pivot_df.shape}")
    
    # Show user similarity heatmap
    sample_users = list(pivot_df.index[:10])
    print(f"\nPlotting user similarity heatmap (first 10 users)...")
    plot_user_similarity_heatmap(pivot_df, sample_users=sample_users, figsize=(8, 6))
    
    # Generate collaborative recommendations
    if len(pivot_df.index) > 0:
        demo_user = pivot_df.index[0]
        print(f"\nCollaborative recommendations for User {demo_user}:")
        try:
            coll_recs = get_user_based_recommendations(
                int(demo_user), pivot_df, sparse_mat, movies, 
                user_to_idx, item_to_idx, n_neighbors=5, top_n=5
            )
            for i, (mid, title, score) in enumerate(coll_recs, 1):
                print(f"{i}. {title} (score: {score:.3f})")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n=== Demo Complete ===")
    print("Data files saved for future runs:")
    print(f"- {movies_csv}")
    print(f"- {ratings_csv}")
    print("- omdb_movie_cache.json")

def parse_args():
    p = argparse.ArgumentParser(description='Movie Recommender Demo')
    p.add_argument('--movies', type=str, default='movies.csv', help='movies CSV (will be created if not exists)')
    p.add_argument('--ratings', type=str, default='ratings.csv', help='ratings CSV (will be created if not exists)')
    p.add_argument('--omdb-key', type=str, default=None, help='OMDb API key for fetching movie data')
    p.add_argument('--demo', action='store_true', help='Run demo pipeline')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.demo:
        demo_pipeline(args.movies, args.ratings, args.omdb_key)
    else:
        print("Run with --demo to see the pipeline.")
        print("The system will fetch movie data directly from OMDb API using your API key.")