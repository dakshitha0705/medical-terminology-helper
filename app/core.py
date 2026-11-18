"""Your code. One stub per milestone; build them in order and keep each one small.

Medical Terminology Helper: a GPT built from scratch (as in my "Build GPT from Scratch" drill),
trained on public-domain anatomy/physiology books. Everything here runs on a laptop CPU.
"""
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"          # never committed (.gitignore)
MODEL_DIR = ROOT / "models"       # never committed (.gitignore)
RESULTS_DIR = ROOT / "results"    # committed: loss plot, train log, samples


# --- Milestone 1: Curate a small corpus of public domain medical texts (e.g., anatomy, physiology).
# Hint: Use text processing techniques to clean and prepare your corpus.

@dataclass(frozen=True)
class Source:
    """One Project Gutenberg book, plus where its real text starts and stops."""
    id: int
    title: str
    author: str
    year: int
    start: str            # regex for the line where the body text begins ...
    start_nth: int = 1    # ... its nth match (the first "CHAPTER I." is usually the table of contents)
    end: str = r"$^"      # regex for the first line after the body (index, adverts, notes)
    glossary: str = ""    # regex for the line where a glossary starts (kept, reformatted as "Term: definition")

    @property
    def url(self) -> str:
        return f"https://www.gutenberg.org/ebooks/{self.id}"


SOURCES = [
    Source(64754, "Structure and Functions of the Body: A Hand-Book of Anatomy and Physiology for Nurses",
           "Annette Fiske", 1911, start=r"^CHAPTER I\.$", end=r"^INDEX\.$"),
    Source(30541, "A Treatise on Anatomy, Physiology, and Hygiene (Revised Edition)",
           "Calvin Cutter", 1870, start=r"^CHAPTER I\.$", start_nth=1, end=r"^INDEX\.$", glossary=r"^GLOSSARY$"),
    Source(52775, "Ellis's Primary Physiology, or Good Health for Boys and Girls",
           "Edward S. Ellis", 1900, start=r"^\s*CHAPTER I\.$", start_nth=2, end=r"^Transcriber's note:"),
    Source(53347, "Physiology (Science Primers)", "Michael Foster", 1880,
           start=r"^INTRODUCTION\. § I\.$", end=r"NICHOLSON’S GEOLOGY"),
]

# Characters that NFKD can't turn into ASCII on its own.
_CHAR_MAP = {"œ": "oe", "Œ": "Oe", "æ": "ae", "Æ": "Ae", "’": "'", "‘": "'", "“": '"', "”": '"',
             "—": "--", "–": "-", "⁄": "/", "×": "x", "°": " degrees", "´": "", "§": "", "£": "",
             " ": " ", "\t": " "}


def strip_gutenberg(text: str) -> str:
    """Keep only what lies between the '*** START OF ...' and '*** END OF ...' licence markers."""
    text = re.sub(r"\r+\n", "\n", text).replace("\r", "\n")
    start = re.search(r"^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", text, re.M | re.I)
    end = re.search(r"^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", text, re.M | re.I)
    if start:
        text = text[start.end():] if not end else text[start.end():end.start()]
    elif end:
        text = text[:end.start()]
    return re.split(r"^End of (the )?Project Gutenberg", text, flags=re.M | re.I)[0]


def normalize_chars(text: str) -> str:
    """Map typographic characters to plain ASCII (œ -> oe, curly quotes -> straight, ½ -> 1/2 ...)."""
    for a, b in _CHAR_MAP.items():
        text = text.replace(a, b)
    text = unicodedata.normalize("NFKD", text).replace("⁄", "/")   # ½ -> 1⁄2 -> 1/2
    return text.encode("ascii", "ignore").decode("ascii")


def _remove_bracketed(text: str, tag: str) -> str:
    """Remove '[Illustration: ...]' / '[Footnote ...]' blocks, which may span lines and nest brackets."""
    out, i = [], 0
    while True:
        j = text.find("[" + tag, i)
        if j < 0:
            return "".join(out) + text[i:]
        out.append(text[i:j])
        depth, k = 0, j
        while k < len(text):
            depth += {"[": 1, "]": -1}.get(text[k], 0)
            k += 1
            if depth == 0:
                break
        i = k


def _keep_line(line: str) -> bool:
    """Body text is flush-left prose. Indented lines are contents, tables, centred headings or
    question lists; lines without lowercase letters are headings, page marks or '* * *' rules."""
    if not line.strip():
        return True                         # blank lines separate paragraphs
    if line[0].isspace() or "   " in line or "...." in line:
        return False
    return any(c.islower() for c in line)


def _paragraphs(lines: list[str]) -> list[str]:
    paras, buf = [], []
    for line in lines + [""]:
        if line.strip():
            buf.append(line.strip())
        elif buf:
            paras.append(" ".join(buf))
            buf = []
    return paras


def _clean_paragraph(p: str) -> str:
    p = re.sub(r"[_=]", "", p)                                   # _italics_ and =bold= markup
    p = re.sub(r"\{pg ?\d+\}|\[\d+\]|\[[A-Z]\]", "", p)            # page and footnote anchors
    p = re.sub(r"\s*\((see )?figs?\.[^)]*\)", "", p, flags=re.I)  # "(Fig. 12)" references
    p = re.sub(r"^\d+\.\s+", "", p)                              # numbered paragraphs "12. ..."
    p = re.sub(r"\^\{(\w+)\}|\^(\w)", r"\1\2", p)                 # superscripts "A^{2}" -> "A2"
    p = normalize_chars(p)
    p = re.sub(r"\s+", " ", p).strip()
    p = re.sub(r"\s+([,.;:?!])", r"\1", p)
    return p


def _looks_like_prose(p: str) -> bool:
    words = p.split()
    if len(words) < 6 or p.count("?") >= 2 or "{" in p or "}" in p:  # fragments, questions, layout
        return False
    letters = sum(c.isalpha() for c in p)
    return letters / max(len(p), 1) > 0.6             # tables of numbers fail this


def clean_glossary(lines: list[str]) -> list[str]:
    """'AB-DUC´TOR. [L. abduco, to lead away.] A muscle which ...' -> 'Abductor: A muscle which ...'"""
    entries = []
    for p in _paragraphs(lines):
        p = _remove_bracketed(p, "")  # etymologies in [...]
        m = re.match(r"^([A-Z][A-Z\-´' ]+?)\.\s*(.+)$", p)
        if not m:
            continue
        head = re.sub(r"[-´']", "", m.group(1)).strip().capitalize()
        body = _clean_paragraph(m.group(2))
        if head and len(body.split()) >= 2 and "}" not in p:
            entries.append(f"{normalize_chars(head)}: {body}")
    return entries


def clean_book(raw: str, source: Source) -> list[str]:
    """Raw Gutenberg text -> list of clean prose paragraphs (and glossary entries)."""
    text = strip_gutenberg(raw)
    text = _remove_bracketed(text, "Illustration")
    text = _remove_bracketed(text, "Footnote")
    lines = text.split("\n")

    starts = [i for i, ln in enumerate(lines) if re.search(source.start, ln)]
    lo = starts[source.start_nth - 1] + 1 if len(starts) >= source.start_nth else 0
    hi = next((i for i, ln in enumerate(lines) if i > lo and re.search(source.end, ln)), len(lines))
    gl = next((i for i, ln in enumerate(lines) if source.glossary and lo < i < hi
               and re.search(source.glossary, ln)), hi)
    body, glossary = lines[lo:gl], lines[gl + 1:hi]

    # Cutter puts review questions between '-=-=-=' rules: drop them.
    kept, in_questions = [], False
    for ln in body:
        if ln.strip().startswith("-=-=-"):
            in_questions = not in_questions
            continue
        if not in_questions and _keep_line(ln):
            kept.append(ln)

    paras = [_clean_paragraph(p) for p in _paragraphs(kept)]
    paras = [p for p in paras if _looks_like_prose(p)]
    return paras + clean_glossary(glossary)


def split_train_val(paragraphs: list[str], val_every: int = 10, block: int = 20) -> tuple[str, str]:
    """Every 10th block of 20 paragraphs goes to validation, so val covers every book and every topic
    without sharing paragraphs with train."""
    train, val = [], []
    for b in range(0, len(paragraphs), block):
        (val if (b // block) % val_every == val_every - 1 else train).extend(paragraphs[b:b + block])
    return "\n".join(train) + "\n", "\n".join(val) + "\n"


def corpus_stats(text: str) -> dict:
    words = re.findall(r"[A-Za-z]+", text)
    return {"chars": len(text), "words": len(words), "paragraphs": text.count("\n"),
            "distinct_words": len({w.lower() for w in words}), "distinct_chars": len(set(text))}


def milestone_1(raw_books: dict[int, str]) -> tuple[str, str, str]:
    """Clean every downloaded book; return (corpus, train, val). `scripts/prepare_data.py` does the I/O."""
    paragraphs = []
    for src in SOURCES:
        if src.id in raw_books:
            paragraphs += clean_book(raw_books[src.id], src)
    train, val = split_train_val(paragraphs)
    return "\n".join(paragraphs) + "\n", train, val


# --- Milestone 2: Implement the byte-pair tokenizer and generate embeddings from the medical corpus.
# Hint: Apply the byte-pair encoding (BPE) and embedding layer concepts.
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
# Hint: Implement the masked self-attention mechanism and feed-forward network.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Set up the training loop and train the GPT model on the medical text data.
# Hint: Set up your optimizer, loss function, and training loop.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Implement a text generation function to complete medical phrases or definitions.
# Hint: Utilize the model's prediction capabilities to generate sequences.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# --- Milestone 6: Create a CLI for students to enter partial phrases and get completions.
# Hint: Design a simple command-line interface for user interaction.
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
