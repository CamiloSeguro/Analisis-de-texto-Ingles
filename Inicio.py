# app.py — TF-IDF Q&A (English) · robusto y configurable
import re
import unicodedata
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────
# Stemming inglés (sin dependencias extra de corpus)
try:
    from nltk.stem import SnowballStemmer
    STEMMER = SnowballStemmer("english")
    def stem_en(t: str) -> str: return STEMMER.stem(t)
except Exception:
    # Fallback: identidad (sin stemming) si NLTK no está disponible
    def stem_en(t: str) -> str: return t

# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="TF-IDF Q&A (English)", page_icon="🔎", layout="wide")
st.title("🔎 TF-IDF Q&A (English) — Mini IR Lab")

st.write("""
Each line is treated as a **document**.  
This demo expects **English** text (stopwords/normalization tuned for EN).
You can tweak preprocessing, n-grams, and df thresholds in the sidebar.
""")

# Ejemplo inicial en inglés
DEFAULT_DOCS = """The dog barks loudly.
The cat meows at night.
The dog and the cat play together in the garden.
Kids run and have fun in the park.
Music is very loud at the party.
Birds sing beautiful melodies at dawn."""

SUGGESTED = [
    "Who is playing?",
    "Where do kids have fun?",
    "When do birds sing?",
    "Where is the music loud?",
    "Which animal meows at night?",
]

# ─────────────────────────────────────────────────────────────
# Prepro y tokenización
def strip_accents(s: str) -> str:
    nkfd = unicodedata.normalize("NFKD", s)
    return "".
