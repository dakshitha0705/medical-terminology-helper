"""Milestone 1: download the public-domain books and build data/corpus.txt, data/train.txt, data/val.txt.

    python scripts/prepare_data.py            # downloads once into data/raw/, then cleans
    python scripts/prepare_data.py --offline  # re-clean from data/raw/ only

Downloads come from Project Gutenberg's own mirror (gutenberg.pglaf.org), as PG asks automated
clients to do, with www.gutenberg.org as a fallback. Every source is listed in DATA.md.
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.core import DATA_DIR, SOURCES, clean_book, corpus_stats, milestone_1  # noqa: E402

UA = {"User-Agent": "medical-terminology-helper/0.1 (educational, non-commercial)"}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


def download(book_id: int) -> str:
    """Return the book's raw text, from the data/raw cache if present."""
    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cached = raw_dir / f"pg{book_id}.txt"
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    sub = "/".join(str(book_id)[:-1])
    base = f"https://gutenberg.pglaf.org/{sub}/{book_id}/"
    candidates = []
    try:
        listing = _get(base).decode("utf-8", "replace")
        candidates = [base + f for f in re.findall(rf'href="({book_id}(?:-0|-8)?\.txt)"', listing)]
    except OSError as e:
        print(f"  mirror listing failed for {book_id}: {e}")
    candidates.append(f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt")
    for url in candidates:
        try:
            data = _get(url)
        except OSError as e:
            print(f"  {url} failed: {e}")
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")          # older "-8" files are ISO-8859-1
        if "PROJECT GUTENBERG" not in text.upper():
            print(f"  {url} is not a Gutenberg text (maintenance page?), skipping")
            continue
        cached.write_text(text, encoding="utf-8", newline="")
        print(f"  downloaded {url} ({len(data):,} bytes)")
        return text
    raise SystemExit(f"Could not download book {book_id}. Try again later.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--offline", action="store_true", help="use data/raw/ only, no downloads")
    args = ap.parse_args()

    raw = {}
    for src in SOURCES:
        print(f"[{src.id}] {src.title} — {src.author} ({src.year})")
        if args.offline:
            raw[src.id] = (DATA_DIR / "raw" / f"pg{src.id}.txt").read_text(encoding="utf-8")
        else:
            raw[src.id] = download(src.id)

    corpus, train, val = milestone_1(raw)
    for name, text in [("corpus.txt", corpus), ("train.txt", train), ("val.txt", val)]:
        (DATA_DIR / name).write_text(text, encoding="utf-8", newline="\n")

    print("\nPer book (raw chars -> clean chars, paragraphs):")
    per_book = {}
    for src in SOURCES:
        paras = clean_book(raw[src.id], src)
        n = sum(len(p) + 1 for p in paras)
        per_book[src.id] = {"raw_chars": len(raw[src.id]), "clean_chars": n, "paragraphs": len(paras)}
        print(f"  {src.id:>6}  {src.author:<16} {len(raw[src.id]):>9,} -> {n:>9,}  ({len(paras):,} paragraphs)")

    stats = {"corpus": corpus_stats(corpus), "train": corpus_stats(train), "val": corpus_stats(val),
             "books": per_book}
    (DATA_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    c = stats["corpus"]
    print(f"\nCorpus: {c['chars']:,} chars, {c['words']:,} words, {c['paragraphs']:,} paragraphs, "
          f"{c['distinct_words']:,} distinct words, {c['distinct_chars']} distinct characters")
    print(f"Split:  train {stats['train']['chars']:,} chars | val {stats['val']['chars']:,} chars "
          f"({stats['val']['chars'] / c['chars']:.1%})")
    print(f"Characters: {''.join(sorted(set(corpus)))!r}")
    print(f"Saved data/corpus.txt, data/train.txt, data/val.txt, data/stats.json")


if __name__ == "__main__":
    main()
