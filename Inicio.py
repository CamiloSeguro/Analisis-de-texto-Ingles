# app.py — TF-IDF Q&A (English) – enhanced
import re
import unicodedata
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional stemming (Snowball English). If NLTK not present, fall back to identity.
try:
    from nltk.stem import SnowballStemmer
    STEMMER = SnowballStemmer("english")
    def stem_en(t: str) -> str: return STEMMER.stem(t)
except Exception:
    def stem_en(t: str) -> str: return t

st.set_page_config(page_title="TF-IDF Q&A (English)", page_icon="🔍", layout="wide")
st.title("🔍 TF-IDF Q&A Demo (English) – Lab")

st.write("""
Each line is treated as a **document**.  
This demo is configured for **English** (stopwords & stemming).  
You can tweak vectorizer parameters in the sidebar to see how retrieval changes.
""")

# ---------------------------------------------------------------------
# Defaults & suggested questions
# ---------------------------------------------------------------------
DEFAULT_DOCS = """The dog barks loudly in the park.
The cat meows softly during the night.
The dog and the cat play together in the garden.
Children run and have fun in the park.
The music is very loud at the party.
Birds sing beautiful melodies at dawn."""

SUGGESTED = [
    "Where do the dog and the cat play?",
    "What do children do in the park?",
    "When do birds sing?",
    "Where is the music loud?",
    "Which animal meows at night?"
]

# ---------------------------------------------------------------------
# Cleaning & tokenization
# ---------------------------------------------------------------------
def strip_accents(s: str) -> str:
    nkfd = unicodedata.normalize("NFKD", s)
    return "".join([c for c in nkfd if not unicodedata.combining(c)])

def tokenize_en(text: str, use_stemming: bool, remove_stopwords: bool) -> List[str]:
    # lower + remove URLs/@/# + keep only letters/spaces
    text = text.lower()
    text = re.sub(r"https?://\S+|[@#]\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = strip_accents(text)
    tokens = [t for t in text.split() if len(t) > 1]
    if remove_stopwords:
        # Light built-in English stopword removal via TfidfVectorizer (we'll still pass stop_words there)
        pass
    if use_stemming:
        tokens = [stem_en(t) for t in tokens]
    return tokens

def highlight_terms(text: str, terms: List[str]) -> str:
    """Highlight (mark) matched terms in original text (case-insensitive)."""
    if not terms: return text
    safe = [re.escape(t) for t in sorted(set(terms), key=len, reverse=True)]
    if not safe: return text
    pattern = r"\b(" + "|".join(safe) + r")\b"
    return re.sub(pattern, lambda m: f"<mark>{m.group(0)}</mark>", text, flags=re.IGNORECASE)

def build_vectorizer(ngram_max, min_df, max_df, use_stemming, remove_stopwords, sublinear, norm):
    # We use a custom tokenizer (stemming), so we must disable token_pattern
    stop_words = "english" if remove_stopwords else None
    return TfidfVectorizer(
        tokenizer=lambda s: tokenize_en(s, use_stemming=use_stemming, remove_stopwords=remove_stopwords),
        token_pattern=None,
        stop_words=stop_words,
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear,
        norm=norm,
        use_idf=True,
    )

# ---------------------------------------------------------------------
# Sidebar – parameters (safe defaults)
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Vectorizer parameters")

    ngram_max = st.select_slider("Max n-gram", options=[1, 2, 3], value=2)

    mode_df = st.radio("df mode", ["Count (int)", "Proportion (0–1)"], index=0, horizontal=True)
    if mode_df == "Count (int)":
        min_df_ui = st.number_input("min_df (count)", value=1, min_value=1, step=1,
                                    help="≥1 = appears in at least this many docs. Start with 1–2.")
        max_df_ui = st.number_input("max_df (count)", value=1000, min_value=1, step=1,
                                    help="Filter terms that appear in too many docs. Keep high to avoid filtering.")
        min_df = int(min_df_ui); max_df = int(max_df_ui)
    else:
        min_df_ui = st.number_input("min_df (proportion 0–1)", value=0.0, min_value=0.0, max_value=1.0, step=0.05,
                                    help="0.0–1.0. 0.0 = no lower filter. Avoid 1.0 (100%).")
        max_df_ui = st.number_input("max_df (proportion 0–1)", value=1.0, min_value=0.0, max_value=1.0, step=0.05,
                                    help="Upper filter for very frequent terms. 1.0 = none.")
        min_df = float(min_df_ui); max_df = float(max_df_ui)

    remove_stop = st.checkbox("Remove English stopwords", True)
    use_stem = st.checkbox("Apply stemming (Snowball)", True)
    sublinear = st.checkbox("Sublinear TF (log(1+tf))", True)
    norm = st.selectbox("Normalization", ["l2", None], index=0)

    st.markdown("---")
    topk_matrix = st.slider("Top-K terms to show in matrix", 5, 50, 20, 1)
    show_all_tokens = st.checkbox("Show ALL columns (can be large)", False)

# ---------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------
col1, col2 = st.columns([2, 1])
with col1:
    text_input = st.text_area("📝 Documents (one per line, English):", DEFAULT_DOCS, height=180)
    question = st.text_input("❓ Query (English):", "Where do the dog and the cat play?")
with col2:
    st.markdown("### 💡 Suggested queries")
    for q in SUGGESTED:
        if st.button(q, use_container_width=True):
            st.session_state.question = q
            st.rerun()
if "question" in st.session_state:
    question = st.session_state.question

# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------
if st.button("🔎 Compute TF-IDF & retrieve", type="primary"):
    docs = [d.strip() for d in text_input.split("\n") if d.strip()]
    if not docs:
        st.error("Please provide at least one document.")
        st.stop()
    if not question.strip():
        st.error("Please type a query.")
        st.stop()

    # Pre-check: ensure docs won't become empty after tokenization
    token_lists = [tokenize_en(d, use_stemming=use_stem, remove_stopwords=remove_stop) for d in docs]
    empty_docs = [i for i, toks in enumerate(token_lists) if len(toks) == 0]
    if empty_docs:
        st.error(
            "These documents ended up EMPTY after cleaning/tokenization: "
            + ", ".join([f"Doc {i+1}" for i in empty_docs])
            + ". Try disabling stopwords/stemming, lowering min_df/max_df filtering, or editing the text."
        )
        st.stop()

    vec = build_vectorizer(
        ngram_max=ngram_max,
        min_df=min_df,
        max_df=max_df,
        use_stemming=use_stem,
        remove_stopwords=remove_stop,
        sublinear=sublinear,
        norm=norm,
    )

    try:
        X = vec.fit_transform(docs)  # (n_docs, n_terms)
    except ValueError:
        st.error("After pruning, no terms remain.")
        st.info(
            "- Use **min_df=1** (count) or lower proportion (e.g., 0.0–0.1).\n"
            "- Increase **max_df** (1.0 or a large count).\n"
            "- Temporarily disable **stopwords** and/or **stemming**.\n"
            "- Increase **n-grams** (2–3)."
        )
        st.stop()

    feature_names = np.array(vec.get_feature_names_out())
    q_vec = vec.transform([question])
    sims = cosine_similarity(q_vec, X).flatten()
    order = np.argsort(-sims)

    # --- Similarity table
    st.markdown("### 📈 Document similarity")
    sim_table = pd.DataFrame({
        "Document": [f"Doc {i+1}" for i in range(len(docs))],
        "Text": docs,
        "Cosine similarity": sims
    }).sort_values("Cosine similarity", ascending=False)
    st.dataframe(sim_table.style.format({"Cosine similarity": "{:.3f}"}), use_container_width=True, hide_index=True)

    best_idx = int(order[0])
    best_doc = docs[best_idx]
    best_score = float(sims[best_idx])

    # --- TF-IDF matrix (trimmed)
    st.markdown("### 📊 TF-IDF matrix")
    Xdense = X.toarray()
    if show_all_tokens:
        df_tfidf = pd.DataFrame(
            Xdense, columns=feature_names, index=[f"Doc {i+1}" for i in range(len(docs))]
        )
        st.dataframe(df_tfidf.round(3), use_container_width=True)
    else:
        variances = Xdense.var(axis=0)
        topk_idx = np.argsort(-variances)[:topk_matrix]
        df_tfidf = pd.DataFrame(
            Xdense[:, topk_idx],
            columns=feature_names[topk_idx],
            index=[f"Doc {i+1}" for i in range(len(docs))]
        )
        st.dataframe(df_tfidf.round(3), use_container_width=True)

    # --- Top terms of best doc
    st.markdown("### 🏷️ Top terms in the best document")
    row = X[best_idx, :].toarray().ravel()
    top_doc_idx = np.argsort(-row)[:10]
    top_doc_terms = feature_names[top_doc_idx]
    top_doc_scores = row[top_doc_idx]
    df_top_terms = pd.DataFrame({"Term": top_doc_terms, "TF-IDF": top_doc_scores})
    st.dataframe(df_top_terms.style.format({"TF-IDF": "{:.3f}"}), use_container_width=True, hide_index=True)

    # --- Highlight matches (stems/ngrams) present in best doc
    q_tokens = tokenize_en(question, use_stemming=use_stem, remove_stopwords=remove_stop)
    vocab_set = set(feature_names)
    to_highlight = [t for t in q_tokens if t in vocab_set]
    highlighted = highlight_terms(best_doc, to_highlight)

    st.markdown("### 🎯 Answer")
    st.markdown(f"**Your query:** {question}")
    if best_score > 0.05:
        st.success(f"**Most similar: Doc {best_idx+1}** — similarity: **{best_score:.3f}**")
    else:
        st.warning(f"**Most similar: Doc {best_idx+1}** — low similarity: **{best_score:.3f}**")
    st.markdown(
        f"<div style='padding:.6rem 1rem;border-radius:12px;border:1px solid #e5e7eb;background:#f8fafc'>{highlighted}</div>",
        unsafe_allow_html=True
    )

    with st.expander("ℹ️ How does this work?"):
        st.markdown("""
- **Clean** text → lowercase, remove URLs/punctuation, optional stopwords.
- **Tokenize** + optional **stemming** (Snowball).
- Build a TF-IDF space with configurable **n-grams**.
- Compute **cosine similarity** between your query and each document.
- Show the most similar document + its **strongest TF-IDF terms**.
        """)
