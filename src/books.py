import requests
from typing import List, Dict, Optional
import numpy as np
import streamlit as st

OL_BASE = "https://openlibrary.org"
COV_BASE = "https://covers.openlibrary.org"

#  model’s label order
EMO_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


BOOK_MOOD_SUBJECTS: Dict[str, Dict[str, List[str]]] = {
    "Happy": {
        "match": ["Humor", "Romance", "Family life", "Comics & graphic novels"],
        "lift":  ["Adventure stories", "Fantasy fiction", "Travel"],
    },
    "Sad": {
        "match": ["Domestic fiction", "Psychological fiction", "Biographies", "Music"],
        "lift":  ["Humor", "Romance", "Inspiration", "Friendship"],
    },
    "Angry": {
        "match": ["Thrillers", "Crime", "Revenge", "Dystopias"],
        "lift":  ["Sports stories", "Humor"],
    },
    "Fear": {
        "match": ["Horror", "Ghost stories", "Supernatural", "Mystery fiction"],
        "lift":  ["Fantasy fiction", "Adventure stories", "Young adult fiction"],
    },
    "Disgust": {
        "match": ["True crime", "War", "Corruption", "Social psychology"],
        "lift":  ["History", "Biography", "Inspirational"],
    },
    "Surprise": {
        "match": ["Mystery fiction", "Time travel", "Heist", "Science fiction"],
        "lift":  ["Romance", "Humor"],
    },
    "Neutral": {
        "match": ["Coming of age", "Slice of life", "Essays", "Documentary films"],  # essays/biogs/non-fic vibes
        "lift":  ["Humor", "Travel", "Adventure stories"],
    },
}

def _ol_search_by_subjects(subjects: List[str], language: Optional[str], limit: int = 24, page: int = 1) -> dict:
    """
    Use Open Library /search.json with multiple 'subject' filters.
    Fields of interest: title, author_name, first_publish_year, cover_i, ratings_average, edition_count, key
    """
    params = [("limit", str(limit)), ("page", str(page))]
    # /search.json supports multiple subject params
    for s in subjects:
        params.append(("subject", s))
    if language:
        params.append(("language", language)) 
    
    params.append(("fields", "title,author_name,first_publish_year,cover_i,ratings_average,edition_count,key"))

    r = requests.get(f"{OL_BASE}/search.json", params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def _rank_books(docs: List[dict]) -> List[dict]:
    """
    Rank by ratings_average (if present) then edition_count, fallback to recent first_publish_year.
    """
    def score(d):
        ra = d.get("ratings_average", None)
        ec = d.get("edition_count", 0) or 0
        yr = d.get("first_publish_year", 0) or 0
        
        return (ra if isinstance(ra, (int, float)) else -1, ec, yr)
    return sorted(docs, key=score, reverse=True)

def _cover_url(doc: dict, size: str = "L") -> Optional[str]:
    """
    Build a cover URL. Prefer cover_i; fallback to None if not available.
    Sizes: S/M/L
    """
    cover_i = doc.get("cover_i", None)
    if cover_i:
        return f"{COV_BASE}/b/id/{cover_i}-{size}.jpg"
    return None

def discover_books_for_emotion(
    emotion: str,
    mode: str = "match",        
    language_ol: Optional[str] = "eng", 
    per_page: int = 24,
    page: int = 1,
) -> List[dict]:
    emo_cfg = BOOK_MOOD_SUBJECTS.get(emotion, BOOK_MOOD_SUBJECTS["Neutral"])
    subjects = emo_cfg.get(mode, emo_cfg["match"])

    
    for k in range(len(subjects), 0, -1):
        try_subjects = subjects[:k]
        data = _ol_search_by_subjects(try_subjects, language_ol, limit=per_page, page=page)
        docs = data.get("docs", []) or []
        if docs:
            ranked = _rank_books(docs)
            # Normalize a bit for display
            for d in ranked:
                d["_mood"] = emotion
                d["_subjects"] = ", ".join(try_subjects)
                d["_cover"] = _cover_url(d)
                d["_work_url"] = f"{OL_BASE}{d['key']}" if d.get("key") else None
            return ranked

    params = [("limit", str(per_page)), ("page", str(page)), ("fields", "title,author_name,first_publish_year,cover_i,edition_count,key")]
    if language_ol:
        params.append(("language", language_ol))
    r = requests.get(f"{OL_BASE}/search.json", params=params, timeout=15)
    r.raise_for_status()
    docs = r.json().get("docs", []) or []
    ranked = sorted(docs, key=lambda d: (d.get("edition_count", 0) or 0, d.get("first_publish_year", 0) or 0), reverse=True)
    for d in ranked:
        d["_mood"] = emotion
        d["_subjects"] = "—"
        d["_cover"] = _cover_url(d)
        d["_work_url"] = f"{OL_BASE}{d['key']}" if d.get("key") else None
    return ranked

def recommend_books_from_probs(
    y_prob: np.ndarray,           
    class_names: List[str] = EMO_CLASSES,
    mode: str = "match",
    k: int = 2,
    per_emotion: int = 8,
    language_ol: Optional[str] = "eng",
) -> List[dict]:
    top_idxs = np.argsort(y_prob)[::-1][:k]
    pool = []
    for idx in top_idxs:
        emo = class_names[idx]
        docs = discover_books_for_emotion(emo, mode=mode, language_ol=language_ol, per_page=per_emotion, page=1)[:per_emotion]
        for d in docs:
            score = float(y_prob[idx]) * float(d.get("edition_count", 0) or 1)
            d["_score"] = score
            pool.append(d)
    # sort & dedupe by work key
    seen = set()
    ranked = []
    for d in sorted(pool, key=lambda x: x.get("_score", 0.0), reverse=True):
        key = d.get("key")
        if key in seen:
            continue
        seen.add(key)
        ranked.append(d)
    return ranked[:20]

def show_books(books: List[dict]):
    st.markdown("""
    <style>
    .books-section {
        width: 100%;
        margin-top: 20px;
    }

    .section-title {
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 8px;
        color: #222;
    }

    .section-divider {
        height: 2px;
        width: 100%;
        background: linear-gradient(to right, #28a74540, #28a74590, #28a74540);
        margin-bottom: 20px;
        border-radius: 2px;
    }

    .book-list {
        display: flex;
        flex-direction: column;
        gap: 25px;
        padding: 10px;
        width: 100%;
    }

    .book-card {
        display: flex;
        flex-direction: row;
        background-color: #f9f9f9;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        overflow: hidden;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        width: 100%;
    }

    .book-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }

    .book-cover {
        flex: 0 0 100px;         /* cover width */
        height: 200px;           /* cover height */
        object-fit: cover;
    }

    .book-content {
        flex: 1;
        padding: 10px 14px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .book-title {
        font-weight: 700;
        font-size: 16px;
        color: #111;
        margin-bottom: 4px;
    }

    .book-info {
        color: #444;
        font-size: 13px;
        margin-bottom: 8px;
    }

    .book-subjects {
        font-size: 13px;
        color: #555;
        line-height: 1.5;
        max-height: 110px;
        overflow-y: auto;
        text-align: justify;
        padding-right: 6px;
    }

    .book-subjects::-webkit-scrollbar {
        width: 6px;
    }
    .book-subjects::-webkit-scrollbar-thumb {
        background-color: #ccc;
        border-radius: 10px;
    }

    .book-footer {
        text-align: right;
        margin-top: 6px;
        font-size: 13px;
        color: #28a745;
        font-weight: 500;
    }

    @media (max-width: 700px) {
        .book-card {
            flex-direction: column;
        }
        .book-cover {
            width: 100%;
            height: 220px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="books-section">
        <div class="section-title">Recommended Books</div>
        <div class="section-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    # Book list
    st.markdown('<div class="book-list">', unsafe_allow_html=True)

    for b in books:
        img_url = b.get("_cover") or "https://via.placeholder.com/200x300?text=No+Cover"
        title = b.get("title", "(no title)")
        authors = ", ".join(b.get("author_name", [])[:2])
        year = b.get("first_publish_year", "N/A")
        subjects = b.get("_subjects", "—")
        mood = b.get("_mood", "")
        link = b.get("_work_url", "")

        st.markdown(f"""
        <div class="book-card">
            <img src="{img_url}" class="book-cover" alt="{title}">
            <div class="book-content">
                <div>
                    <div class="book-title">{title}</div>
                    <div class="book-info">{authors}<br>{year} • {mood} mood</div>
                    <div class="book-subjects">{subjects}</div>
                </div>
                {'<div class="book-footer"><a href="'+link+'" target="_blank" style="color:#28a745;text-decoration:none;">View on Open Library</a></div>' if link else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)