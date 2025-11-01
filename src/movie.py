import os
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv
import streamlit as st
import numpy as np


load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"


EMO_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


MOOD_GENRES: Dict[str, Dict[str, List[str]]] = {
    "Happy":   {
        "match": ["Comedy", "Romance", "Family", "Animation"],
        "lift":  ["Adventure", "Sci-Fi", "Fantasy"],
    },
    "Sad":     {
        "match": ["Drama", "Music", "Biography"],
        "lift":  ["Comedy", "Family", "Animation", "Romance"],
    },
    "Angry":   {
        "match": ["Action", "Crime", "Thriller"],
        "lift":  ["Comedy", "Sports"],
    },
    "Fear":    {
        "match": ["Horror", "Thriller", "Mystery"],
        "lift":  ["Animation", "Adventure", "Fantasy"],
    },
    "Disgust": {
        "match": ["Documentary", "Crime", "War"],
        "lift":  ["Drama", "Biography", "History"],
    },
    "Surprise":{
        "match": ["Mystery", "Sci-Fi", "Adventure", "Fantasy"],
        "lift":  ["Comedy", "Romance"],
    },
    "Neutral": {
        "match": ["Drama", "Documentary", "Comedy"],
        "lift":  ["Comedy", "Adventure"],
    },
}


def tmdb_get(path: str, params: Optional[dict] = None) -> dict:
    if not TMDB_API_KEY:
        raise RuntimeError("TMDB_API_KEY not set")
    params = dict(params or {})
    params["api_key"] = TMDB_API_KEY
    r = requests.get(f"{TMDB_BASE}{path}", params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def get_genre_id_map(language: str = "en-US") -> Dict[str, int]:
    data = tmdb_get("/genre/movie/list", {"language": language})
    return {g["name"]: g["id"] for g in data.get("genres", [])}


def _discover(params: dict) -> list:
    data = tmdb_get("/discover/movie", params)
    return data.get("results", []) or []


def discover_movies_for_emotion(
    emotion: str,
    mode: str = "match",           
    language: str = "en-US",
    region: Optional[str] = None,
    page: int = 1,
    min_votes: int = 100,
    include_adult: bool = False,
    recent_gte: Optional[str] = None,  
) -> List[dict]:
    
    emo_cfg = MOOD_GENRES.get(emotion, MOOD_GENRES["Neutral"])
    genre_names = emo_cfg.get(mode, emo_cfg["match"])

    
    genre_id_map = get_genre_id_map(language)
    with_genres = ",".join(str(genre_id_map[g]) for g in genre_names if g in genre_id_map)

    base = {
        "language": language,
        "sort_by": "popularity.desc",
        "page": page,
        "include_adult": str(include_adult).lower(),
        "vote_count.gte": min_votes,
    }
    if region:
        base["region"] = region
    if recent_gte:
        base["primary_release_date.gte"] = recent_gte

    
    p1 = dict(base)
    if with_genres:
        p1["with_genres"] = with_genres
    results = _discover(p1)
    if results:
        return results

    
    p2 = dict(p1)
    p2["vote_count.gte"] = max(0, min_votes // 2)
    results = _discover(p2)
    if results:
        return results

    
    if "primary_release_date.gte" in p2:
        p3 = dict(p2)
        p3.pop("primary_release_date.gte", None)
        results = _discover(p3)
        if results:
            return results



def recommend_from_probs(
    y_prob: np.ndarray,         
    class_names: List[str] = EMO_CLASSES,
    mode: str = "match",         
    k: int = 2,                   
    per_emotion: int = 10,
    language: str = "en-US",
    region: Optional[str] = None,
    min_votes: int = 100,
    include_adult: bool = False,
    recent_gte: Optional[str] = None,
) -> List[dict]:
    top_idxs = np.argsort(y_prob)[::-1][:k]
    pool = []
    for idx in top_idxs:
        emo = class_names[idx]
        recs = discover_movies_for_emotion(
            emo, mode=mode, language=language, region=region, page=1,
            min_votes=min_votes, include_adult=include_adult, recent_gte=recent_gte
        )[:per_emotion]
        for m in recs:
            score = float(y_prob[idx]) * float(m.get("popularity", 0.0))
            m["_mood"] = emo
            m["_score"] = score
            pool.append(m)

    
    seen = set()
    ranked = []
    for m in sorted(pool, key=lambda x: x.get("_score", 0.0), reverse=True):
        mid = m.get("id")
        if mid in seen:
            continue
        seen.add(mid)
        ranked.append(m)
    return ranked[:20]


def show_movies(movies: List[dict], ncols: int = 4, img_base: str = "https://image.tmdb.org/t/p/w342"):
    cols = st.columns(ncols)
    for i, m in enumerate(movies):
        with cols[i % ncols]:
            poster = m.get("poster_path")
            if poster:
                st.image(f"{img_base}{poster}", use_container_width=True)
            st.markdown(f"**{m.get('title','(no title)')}**")
            st.caption(f"{m.get('release_date','')} • ⭐ {m.get('vote_average',0):.1f} • {m.get('_mood','')}")
            overview = (m.get("overview") or "")
            st.write((overview[:160] + "…") if len(overview) > 160 else overview)
