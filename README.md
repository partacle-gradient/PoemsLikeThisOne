README

This workspace contains scripts to fetch poems from PoetryDB, process them, compute embeddings, and run a small Streamlit dashboard to explore poem similarity.

Files of interest
- `fetch_poetrydb.py` — Fetches poems from PoetryDB and saves per-author JSON and text into `raw_data/authors/`.
- `process_poems.py` — Cleans raw poems and writes `processed/poems.jsonl` and per-author cleaned text files under `processed/authors/`.
- `embed_poems.py` — Computes embeddings for poems in `processed/poems.jsonl` and saves `processed/embeddings.npy` and `processed/embeddings_metadata.json`. Default model is `BAAI/bge-m3`.
- `app.py` — Streamlit app that lets you pick an author and poem, shows top-5 similar poems (by cosine similarity) in the sidebar, and displays the selected poem and the chosen similar poem in the main panel.
- `requirements.txt` / `pyproject.toml` — Dependency manifests. The project expects `streamlit`, `sentence-transformers`, `numpy`, `tqdm`, and `requests`.

Quick setup (PowerShell)

1. Sync/install dependencies using `uv` (the project uses `uv` for environment/task management in this workspace):

```powershell
cd C:\PoemsLikeThisOne
uv sync
```

2. Fetch a subset (for a quick test) or all poems from PoetryDB:

```powershell
# quick test (5 authors)
uv run python fetch_poetrydb.py --outdir raw_data --limit 5
# full fetch (may take a while)
uv run python fetch_poetrydb.py --outdir raw_data
```

3. Process raw poems into a JSONL suitable for embedding:

```powershell
uv run python process_poems.py
```

4. Generate embeddings (uses `BAAI/bge-m3` by default). This will download the model and may take time and disk space:

```powershell
uv run python embed_poems.py
```

5. Run the Streamlit app and explore:

```powershell
uv run streamlit run app.py
```

App behavior
- Use the sidebar to select an **Author** and then a **Poem** by that author.
- The sidebar shows the top-5 similar poems with similarity scores and clickable buttons — clicking a similar poem will make it the selected poem.
- The main panel always displays the currently selected poem (original/raw text with proper line breaks) and the most similar poem's raw text.

Notes & tips
- If you change the embedding model (in `embed_poems.py`) or re-process poems, re-run `embed_poems.py` to regenerate `processed/embeddings.npy`.
- The raw poem text is read from `raw_data/authors/<sanitized_author>.json` and displayed with original line breaks. The processed text (lowercased/collapsed) is only used to compute embeddings.
- If you're behind a firewall or have restricted network access, model downloads (sentence-transformers / Hugging Face models) may fail — consider pre-downloading or using a local model cache.

Questions or next steps
- Want me to include author/title when computing embeddings (so author signal is baked into vectors)? I can update `embed_poems.py` to concatenate author/title with poem text.
- Want the dashboard to persist embeddings in a vector DB or add nearest-neighbor indexing for speed? I can add FAISS or Annoy integration.
