# --- Imports ---

import sys
import os

# Add parent directory to path for custom modules
# FIXME: Ideally do not do this
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import preprocessing

import re

import pandas as pd
import pickle
import arxiv

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

import streamlit as st
import html



# --- Constants ---

DEFAULT_NUM_RECOMMENDATIONS = 5

MODEL_LOCAL_DIR = "models"
MODEL_NAME = "nadrajak/allenai-specter-ft2"

CORPUS_LOCAL_DIR = "corpus"
CORPUS_NAME = "demo1k"



# --- Styling ---

st.markdown("""
<style>
    /* Input button -- align with text field */
    .stButton > button {
        height: 2.5rem;
        margin-top: 1.7rem;
    }
            
    /* Expander -- remove border and set background */
    [data-testid="stExpander"] details {
        border-style: none;
        background-color: #f0f2f6;
        /* border-radius: 0.5rem; */
        padding: 0rem;
        margin: 0rem;
    }

    /* Expander -- keep header text white even when expanded */
    [data-testid="stExpander"] summary {
        background-color: #f0f2f6 !important;
}
</style>
""", unsafe_allow_html=True)



# --- Helpers (arXiv) ---

def fix_arxiv_id(id_value):
    """Fixes the broken IDs introduced by the arXiv dataset"""
    # Ensure it's a string
    if not isinstance(id_value, str):
        id_value = str(id_value)

    # Old arXiv IDs (like "math.GT/0309136") shouldn’t be split
    if "." not in id_value or "/" in id_value:
        return id_value  

    try:
        id_a, id_b = id_value.split(".")
        return f"{id_a.rjust(4, '0')}.{id_b.ljust(4, '0')}"
    except ValueError:
        # Fallback: just return as-is if something unexpected happens
        return id_value


def id_to_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


def input_to_id(input_text: str) -> str:
    return input_text.split("/")[-1].split(".pdf")[0]


def valid_id(arxiv_id: str) -> bool:
    patterns = [
        r'^\d{4}\.\d{4,5}$',                 # 2301.07041
        r'^\d{4}\.\d{4,5}v\d+$',             # 2301.07041v1
        r'^[a-z-]+(\.[A-Z]{2})?/\d{7}$',     # math.GT/0309136
        r'^[a-z-]+(\.[A-Z]{2})?/\d{7}v\d+$'  # with version
    ]
    return any(re.match(p, arxiv_id) for p in patterns)


@st.cache_data(show_spinner=False)
def fetch_paper_info(arxiv_id: str) -> pd.DataFrame:
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    result = next(client.results(search))

    df = pd.DataFrame({
        "id": [arxiv_id],
        "title": [result.title],
        "abstract": [result.summary],
        "categories": [" ".join(result.categories)],
        "authors_parsed": [[a.name for a in result.authors]],
    })

    df = preprocessing.normalize_whitespace(df)
    df = preprocessing.normalize_abstracts(df)
    df = preprocessing.truncate_categories(df)
    return df.iloc[0]



# --- (Helpers) Streamlit --- 

def display_input_field():
    col1, col2 = st.columns([3, 1])

    with col1:
        input_text = st.text_input(
            "Enter arXiv URL or ID:",
            placeholder="e.g., https://arxiv.org/abs/2301.07041 or 2301.07041",
            help="You can paste a full arXiv URL or just the paper ID"
        )
    with col2:
        get_recommendations = st.button("Recommend", type="primary", use_container_width=True)

    return input_text, get_recommendations


def display_paper(paper, score=0):
    # Extract data from pd.Series
    title = paper["title"]
    abstract = paper["abstract"] 
    category = paper["category"]
    arxiv_id = paper["id"]
    url = id_to_url(arxiv_id) #  if arxiv_id else None
    try:
        authors_text = paper["authors"]
    except Exception:
        authors = paper["authors_parsed"]
        authors_text = ", ".join([a for a in authors])

    # Escape HTML
    title = html.escape(title)
    abstract = html.escape(abstract)
    category = html.escape(category)
    authors_text = html.escape(authors_text)

    # TODO: Handle inline math
    # TODO: Handle latex escape characters

    score_html = ""
    if score:
        score_html = f'<div style="font-size:0.85rem; color:#555; font-family: monospace;">Score: {score:.3f}</div>'

    with st.container(border=True):
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.25rem;">
                <div style="display:flex; align-items:center; gap:1rem;">
                    <a href="{url}" target="_blank" style="font-family: monospace; color:#0366d6; text-decoration:none;">
                        [arXiv:{arxiv_id}]
                    </a>
                    <span style="background-color:#0366d6; color:#ffffff; font-size:0.75rem; padding:0.15rem 0.5rem; border-radius:0.5rem; font-family: monospace;">
                        {category}
                    </span>
                </div>
                {score_html}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Title and authors
        st.markdown(f'<div style="font-weight:600; font-size:1.05rem; margin-bottom:0.25rem;">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color:#666; font-size:0.9rem; margin-bottom:0.5rem;">{authors_text}</div>', unsafe_allow_html=True)

        # Abstract
        with st.expander("Abstract"):
            st.write(abstract)



# --- Helpers (Recommender) ---

@st.cache_data(show_spinner=False)
def load_corpus():
    corpus_path = os.path.join(CORPUS_LOCAL_DIR, CORPUS_NAME)
    with open(f"{corpus_path}.pickle", "rb") as f:
        corpus = pickle.load(f)
    with open(f"{corpus_path}_embedding.pickle", "rb") as f:
        embeddings = pickle.load(f)
    return corpus, embeddings


@st.cache_resource(show_spinner=False)
def load_model(name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(name, device=device)



# --- App ---

def main():
    # Load model
    try:
        with st.spinner("Loading model..."):
            model = load_model(os.path.join(MODEL_LOCAL_DIR, MODEL_NAME))
    except Exception as e:
        alert = st.warning(f"Failed to load model: {str(e)}")
        try:
            with st.spinner("Downloading model...", show_time=True):
                model = load_model(MODEL_NAME)
                model.save(os.path.join(MODEL_LOCAL_DIR, MODEL_NAME))
        except Exception as e:
            st.warning(f"Failed to download sentence embedding model: {str(e)}")
            st.stop()
        alert.empty()
    

    # Load corpus
    with st.spinner("Loading corpus..."):
        try:
            corpus, corpus_embeddings = load_corpus()

            # FIXME: Ideally handle this elsewhere (like when generating the corpus)
            corpus["id"] = corpus["id"].apply(fix_arxiv_id)
            
        except Exception as e:
            st.error(f"Failed to load corpus: {str(e)}")
            st.stop()

    # Title
    st.title("arXiv-recommender")

    with st.sidebar:
        st.title("Sample IDs")
        
        # Get top 7 largest categories and one paper from each
        top_categories = corpus['category'].value_counts().head(7).index
        sample_papers = corpus[corpus['category'].isin(top_categories)].groupby('category').first().reset_index()
        # Sort by category size (descending)
        sample_papers = sample_papers.set_index('category').reindex(top_categories).reset_index()
        
        # TODO: Present this better
        for _, paper in sample_papers.iterrows():
            st.code(paper['id'])


    # Input
    input_text, get_recommendations = display_input_field()
    if input_text or get_recommendations:
        # Set placeholder input if no input given
        if not input_text:
            input_text = "2301.07041"

        arxiv_id = input_to_id(input_text)
        if not valid_id(arxiv_id):
            st.warning("Invalid arXiv ID.")
            st.stop()

        # Fetch & display input paper
        with st.spinner("Fetching paper metadata..."):
            paper_info = fetch_paper_info(arxiv_id)

        st.subheader("Input Paper")
        display_paper(paper_info)

        # Encode & recommend
        with st.spinner("Recommending..."):
            embedding = model.encode(paper_info["abstract"], convert_to_tensor=True)
            results = semantic_search(embedding, corpus_embeddings, top_k=DEFAULT_NUM_RECOMMENDATIONS)

        st.subheader("Recommended Papers")
        for r in results[0]:
            display_paper(corpus.iloc[r["corpus_id"]], r['score'])



if __name__ == "__main__":
    main()
