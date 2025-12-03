#!/usr/bin/env python3
"""Process raw poem JSONs into cleaned text and a JSONL of poems.

Reads `raw_data/authors/*.json` created by `fetch_poetrydb.py`, cleans poem text
(collapse whitespace, remove empty lines, lower-case), and writes:

- `processed/authors/<author>.txt` : cleaned poems per author
- `processed/poems.jsonl` : JSON Lines file with {author, title, text}
"""
import glob
import json
import os
import re
from pathlib import Path


def sanitize(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]', '_', name).strip('_')


def clean_text_from_lines(lines):
    # join lines with space, remove multiple spaces, strip, lowercase
    joined = ' '.join([ln.strip() for ln in lines if ln is not None])
    # remove repeated spaces/tabs
    cleaned = re.sub(r'\s+', ' ', joined).strip().lower()
    return cleaned


def main(raw_dir='raw_data', out_dir='processed'):
    authors_dir = Path(raw_dir) / 'authors'
    out_authors = Path(out_dir) / 'authors'
    out_authors.mkdir(parents=True, exist_ok=True)
    poems_jl = Path(out_dir) / 'poems.jsonl'

    files = sorted(glob.glob(str(authors_dir / '*.json')))
    total = 0
    with poems_jl.open('w', encoding='utf8') as jl:
        for fp in files:
            with open(fp, 'r', encoding='utf8') as fh:
                try:
                    arr = json.load(fh)
                except Exception:
                    continue
            if not isinstance(arr, list):
                continue
            # per-author output file
            author = arr[0].get('author') if arr else Path(fp).stem
            safe = sanitize(author or Path(fp).stem)
            out_fp = out_authors / f"{safe}.txt"
            with out_fp.open('w', encoding='utf8') as outf:
                for p in arr:
                    title = p.get('title', '').strip()
                    lines = p.get('lines') or []
                    cleaned = clean_text_from_lines(lines)
                    if not cleaned:
                        continue
                    # write header and poem
                    outf.write(f"### {title}\n")
                    outf.write(cleaned + '\n\n')
                    # write to jsonl
                    rec = {'author': p.get('author', author), 'title': title, 'text': cleaned}
                    jl.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    total += 1

    print(f'Processed {total} poems into {poems_jl} and per-author files in {out_authors}')


if __name__ == '__main__':
    main()
