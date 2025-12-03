#!/usr/bin/env python3
"""Fetch all poems from PoetryDB per author and save into raw data folder.

Saves two files per author under `<outdir>/authors/`:
- `<sanitized_author>.json` : full JSON response from PoetryDB for that author
- `<sanitized_author>.txt`  : plain text with titles and poem lines

Usage:
  python fetch_poetrydb.py --outdir raw_data
"""
import argparse
import json
import logging
import os
import re
import time
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_session(retries: int = 5, backoff: float = 0.5) -> requests.Session:
    s = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def sanitize(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]', '_', name).strip('_')


def get_authors(session: requests.Session, base: str):
    # try known endpoints until one returns a list
    for ep in ('/authors', '/author'):
        url = base.rstrip('/') + ep
        try:
            r = session.get(url, timeout=30)
        except Exception:
            continue
        if not r.ok:
            continue
        try:
            data = r.json()
        except Exception:
            continue
        if isinstance(data, dict) and 'authors' in data:
            return data['authors']
        if isinstance(data, list):
            return data
    raise RuntimeError('Could not fetch authors from PoetryDB')


def get_poems_by_author(session: requests.Session, base: str, author: str):
    url = f"{base.rstrip('/')}/author/{quote(author)}"
    r = session.get(url, timeout=60)
    if not r.ok:
        r.raise_for_status()
    return r.json()


def save_author(outdir: str, author: str, poems) -> tuple:
    authors_dir = os.path.join(outdir, 'authors')
    os.makedirs(authors_dir, exist_ok=True)
    safe = sanitize(author)
    json_path = os.path.join(authors_dir, f"{safe}.json")
    with open(json_path, 'w', encoding='utf8') as fh:
        json.dump(poems, fh, ensure_ascii=False, indent=2)

    txt_path = os.path.join(authors_dir, f"{safe}.txt")
    with open(txt_path, 'w', encoding='utf8') as fh:
        for p in poems:
            title = p.get('title', '(untitled)')
            fh.write(f"### {title} - {p.get('author','')}\n")
            lines = p.get('lines') or []
            fh.write('\n'.join(lines))
            fh.write('\n\n')

    return json_path, txt_path


def main():
    parser = argparse.ArgumentParser(description='Fetch poems from PoetryDB per author')
    parser.add_argument('--outdir', '-o', default='raw_data', help='Output directory')
    parser.add_argument('--base', '-b', default='https://poetrydb.org', help='PoetryDB base URL')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of authors (0=all)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    session = create_session()

    logging.info('Fetching authors list...')
    authors = get_authors(session, args.base)
    logging.info(f'Found {len(authors)} authors')

    if args.limit and args.limit > 0:
        authors = authors[: args.limit]

    os.makedirs(args.outdir, exist_ok=True)

    for i, author in enumerate(authors, start=1):
        try:
            logging.info(f'[{i}/{len(authors)}] Fetching poems for {author}')
            poems = get_poems_by_author(session, args.base, author)
            json_path, txt_path = save_author(args.outdir, author, poems)
            logging.info(f'  saved: {json_path} and {txt_path}')
        except Exception as exc:
            logging.warning(f'  failed for {author}: {exc}')
            time.sleep(1)

    logging.info('All done.')


if __name__ == '__main__':
    main()
