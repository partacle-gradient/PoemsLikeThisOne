#!/usr/bin/env python3
"""Compute embeddings for processed poems using BAAI/bge-m3.

Requires `sentence-transformers` and `numpy`.

Input: `processed/poems.jsonl` (created by `process_poems.py`)
Output: `processed/embeddings.npy` and `processed/embeddings_metadata.json`
"""
import json
from pathlib import Path
import numpy as np
  

def load_poems(jl_path):
    poems = []
    with open(jl_path, 'r', encoding='utf8') as fh:
        for line in fh:
            if not line.strip():
                continue
            poems.append(json.loads(line))
    return poems


def main(jl='processed/poems.jsonl', model_name='BAAI/bge-m3', out_dir='processed'):
    jl_path = Path(jl)
    if not jl_path.exists():
        raise SystemExit(f'{jl} not found. Run process_poems.py first')
    poems = load_poems(jl_path)
    texts = [p['text'] for p in poems]

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise SystemExit('sentence-transformers not installed. Run `pip install -r requirements.txt`')

    model = SentenceTransformer(model_name,device='gpu')
    print(f"Using model: {model_name}")
    print(f"Computing embeddings for {len(texts)} poems...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64, convert_to_numpy=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / 'embeddings.npy'
    meta_path = out_dir / 'embeddings_metadata.json'

    np.save(emb_path, embeddings)
    # save metadata
    metadata = [{'author': p['author'], 'title': p['title']} for p in poems]
    with open(meta_path, 'w', encoding='utf8') as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    print(f'Saved embeddings ({embeddings.shape}) to {emb_path} and metadata to {meta_path}')


if __name__ == '__main__':
    print("in embed_poems.py")
    main()
