# PoetryDB extractor

This repository contains a small Python script to fetch all poems from PoetryDB and dump them per-author into a raw data folder.

Files created
- `fetch_poetrydb.py`: main script. Saves JSON and plain-text files per author into `<outdir>/authors/`.
- `requirements.txt`: Python dependencies.

Quick start (PowerShell)

```powershell
python -m pip install -r requirements.txt
python fetch_poetrydb.py --outdir raw_data
```

If you want to test quickly with only a few authors:

```powershell
python fetch_poetrydb.py --outdir raw_data --limit 5
```

Output layout
- `raw_data/authors/<sanitized_author>.json` — JSON array of poems for the author
- `raw_data/authors/<sanitized_author>.txt` — human-readable text file with titles and poem lines

Notes
- The script uses `https://poetrydb.org` by default. If you're behind a proxy or mirror, pass `--base`.
- Author names are sanitized for filenames. Characters outside [A-Za-z0-9._-] are replaced with `_`.
