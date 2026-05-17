import re
import requests
from pathlib import Path

BOOK_BASE = "https://raw.githubusercontent.com/rust-lang/book/main/src/"


def get_chapter_files():
    summary = requests.get(BOOK_BASE + "SUMMARY.md").text
    files = re.findall(r'\(([^)]+\.md)\)', summary)
    return list(dict.fromkeys(files))  # deduplicate, preserve order


def fetch_chapter(filepath):
    resp = requests.get(BOOK_BASE + filepath)
    if resp.status_code != 200:
        return None
    content = resp.text
    lines = content.split('\n')
    title = next((l.lstrip('# ').strip() for l in lines if l.startswith('#')), filepath)
    return {"filepath": filepath, "title": title, "content": content}


def load_book(cache_dir="data"):
    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True)

    chapters = []
    for fp in get_chapter_files():
        cache_file = cache / fp.replace('/', '_')
        if cache_file.exists():
            content = cache_file.read_text(encoding='utf-8')
            title = content.split('\n')[0].lstrip('# ').strip()
            chapters.append({"filepath": fp, "title": title, "content": content})
        else:
            print(f"  Fetching {fp}...")
            chapter = fetch_chapter(fp)
            if chapter:
                cache_file.write_text(chapter["content"], encoding='utf-8')
                chapters.append(chapter)

    return chapters
