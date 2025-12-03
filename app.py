#!/usr/bin/env python3
"""Streamlit app: select a poem and find the closest similar poem by embedding."""
import json
import numpy as np
import streamlit as st
from pathlib import Path

# Load embeddings and metadata
PROCESSED_DIR = Path("processed")
RAW_DIR = Path("raw_data") / "authors"
EMBEDDINGS_FILE = PROCESSED_DIR / "embeddings.npy"
METADATA_FILE = PROCESSED_DIR / "embeddings_metadata.json"

# Cache loading to avoid re-reading on every interaction
@st.cache_data
def load_data():
    if not EMBEDDINGS_FILE.exists() or not METADATA_FILE.exists():
        st.error("Embeddings or metadata files not found. Run embed_poems.py first.")
        st.stop()
    
    embeddings = np.load(EMBEDDINGS_FILE)
    with open(METADATA_FILE, "r", encoding="utf8") as f:
        metadata = json.load(f)
    
    # Load poems from processed/poems.jsonl
    poems_file = PROCESSED_DIR / "poems.jsonl"
    poems = []
    if poems_file.exists():
        with open(poems_file, "r", encoding="utf8") as f:
            for line in f:
                if line.strip():
                    poems.append(json.loads(line))
    
    # Load raw poems from raw_data/authors/*.json
    raw_poems = {}
    if RAW_DIR.exists():
        for json_file in RAW_DIR.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for poem in data:
                            key = (poem.get("author", ""), poem.get("title", ""))
                            raw_poems[key] = poem
            except Exception:
                pass
    
    return embeddings, metadata, poems, raw_poems


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def get_raw_poem_text(poem, raw_poems):
    """Get raw poem text with proper formatting, fall back to processed text."""
    raw_key = (poem['author'], poem['title'])
    if raw_key in raw_poems:
        raw_poem = raw_poems[raw_key]
        lines = raw_poem.get('lines', [])
        return '\n'.join(lines)
    return poem['text']


def main():
    st.set_page_config(page_title="Poem Similarity Finder", layout="wide")
    st.title("📖 Poem Similarity Finder")
    
    embeddings, metadata, poems, raw_poems = load_data()
    
    # Sidebar: Author selection
    with st.sidebar:
        st.header("📚 Select Poem")
        
        # Get unique authors sorted
        authors = sorted(set(p['author'] for p in poems))
        selected_author = st.selectbox("Select Author:", authors)
        
        # Get poems by selected author
        author_poems = [p for p in poems if p['author'] == selected_author]
        poem_titles = [p['title'] for p in author_poems]
        
        selected_title = st.selectbox("Select Poem:", poem_titles)
        
        # Find selected poem index
        selected_poem = next(p for p in poems if p['author'] == selected_author and p['title'] == selected_title)
        selected_idx = poems.index(selected_poem)
        
        st.write("---")
        
        # Find top 5 similar poems
        selected_embedding = embeddings[selected_idx]
        similarities = np.array([cosine_similarity(selected_embedding, embeddings[i]) for i in range(len(embeddings))])
        
        # Exclude the selected poem itself
        similarities[selected_idx] = -np.inf
        
        # Get top 5 indices
        top_5_indices = np.argsort(-similarities)[:5]
        
        st.subheader("Top 5 Similar Poems")
        selected_similar_idx = None
        for i, idx in enumerate(top_5_indices):
            col1, col2 = st.columns([3, 1])
            with col1:
                poem_str = f"{poems[idx]['author']} - {poems[idx]['title']}"
                if st.button(poem_str, key=f"similar_{i}"):
                    selected_similar_idx = idx
            with col2:
                st.metric("Score", f"{similarities[idx]:.3f}")
        
        # If a similar poem is selected, update selected_idx
        if selected_similar_idx is not None:
            selected_idx = selected_similar_idx
            selected_poem = poems[selected_idx]
    
    # Main panel: Display poems
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Selected Poem")
        st.write(f"**Author:** {selected_poem['author']}")
        st.write(f"**Title:** {selected_poem['title']}")
        st.write("---")
        poem_text = get_raw_poem_text(selected_poem, raw_poems)
        st.text(poem_text)
    
    with col2:
        st.subheader("Most Similar Poem")
        # Find the closest poem again (in case selection changed)
        selected_embedding = embeddings[selected_idx]
        similarities = np.array([cosine_similarity(selected_embedding, embeddings[i]) for i in range(len(embeddings))])
        similarities[selected_idx] = -np.inf
        closest_idx = np.argmax(similarities)
        closest_poem = poems[closest_idx]
        closest_similarity = similarities[closest_idx]
        
        st.metric("Similarity Score", f"{closest_similarity:.4f}")
        st.write(f"**Author:** {closest_poem['author']}")
        st.write(f"**Title:** {closest_poem['title']}")
        st.write("---")
        closest_poem_text = get_raw_poem_text(closest_poem, raw_poems)
        st.text(closest_poem_text)


if __name__ == "__main__":
    main()
