"""Your code. One stub per milestone; build them in order and keep each one small.

Medical Terminology Helper: a GPT built from scratch (as in my "Build GPT from Scratch" drill),
trained on public-domain anatomy/physiology books. Everything here runs on a laptop CPU.
"""
import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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
#
# The drill used one token per character (65 tokens). Here BPE starts from the corpus characters and
# repeatedly merges the most frequent adjacent pair into a new token, so frequent pieces such as
# " the", "ation", " card" or " muscle" become single tokens. Text is first split into chunks
# (a word with its leading space, a number, punctuation, whitespace) and merges never cross chunks.

CHUNK_RE = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+")


class UnknownCharacterError(ValueError):
    """Raised when text contains characters the tokenizer never saw (after normalisation)."""
    def __init__(self, chars: list[str]):
        self.chars = chars
        super().__init__(f"unknown characters: {' '.join(repr(c) for c in chars)}")


class BPETokenizer:
    """Byte-pair encoding written from scratch: train / encode / decode / save / load."""

    def __init__(self, chars: list[str] | None = None, merges: list[tuple[int, int]] | None = None):
        self.chars = list(chars or [])
        self.merges = [tuple(m) for m in (merges or [])]
        self._build()

    def _build(self) -> None:
        self.vocab = list(self.chars)                       # id -> token string
        self.ranks = {}                                     # (a, b) -> merge rank
        for rank, (a, b) in enumerate(self.merges):
            self.ranks[(a, b)] = rank
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.merged = {pair: len(self.chars) + r for pair, r in self.ranks.items()}
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # -- training -------------------------------------------------------------------------------
    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        chars = sorted(set(text))
        if vocab_size < len(chars):
            raise ValueError(f"vocab_size {vocab_size} is smaller than the {len(chars)} base characters")
        stoi = {c: i for i, c in enumerate(chars)}
        chunk_counts = Counter(CHUNK_RE.findall(text))
        words = [[stoi[c] for c in w] for w in chunk_counts]          # each distinct chunk once ...
        freqs = list(chunk_counts.values())                          # ... weighted by its count
        pair_counts: Counter = Counter()
        where: dict[tuple[int, int], set[int]] = defaultdict(set)    # pair -> chunks containing it
        for wi, w in enumerate(words):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += freqs[wi]
                where[pair].add(wi)

        merges = []
        for new_id in range(len(chars), vocab_size):
            if not pair_counts:
                break
            best = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))
            if pair_counts[best] < 2:
                break
            merges.append(best)
            for wi in list(where[best]):                             # only chunks that contain it
                w, f = words[wi], freqs[wi]
                for pair in zip(w, w[1:]):                           # remove old pair counts
                    pair_counts[pair] -= f
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    where[pair].discard(wi)
                words[wi] = w = cls._merge(w, best, new_id)
                for pair in zip(w, w[1:]):                           # add the new ones
                    pair_counts[pair] += f
                    where[pair].add(wi)
            if verbose and len(merges) % 250 == 0:
                tok = cls(chars, merges)
                print(f"  merge {len(merges):5d}: {tok.vocab[new_id]!r}")
        return cls(chars, merges)

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out, i = [], 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    # -- encode / decode ------------------------------------------------------------------------
    def unknown_chars(self, text: str) -> list[str]:
        return sorted(set(text) - set(self.stoi))

    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk in self._cache:
            return self._cache[chunk]
        ids = [self.stoi[c] for c in chunk]
        while len(ids) >= 2:                                         # apply merges in training order
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = self._merge(ids, pair, self.merged[pair])
        if len(self._cache) < 100_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        bad = self.unknown_chars(text)
        if bad:
            raise UnknownCharacterError(bad)
        return [i for chunk in CHUNK_RE.findall(text) for i in self._encode_chunk(chunk)]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids)

    def tokens(self, text: str) -> list[str]:
        """How a piece of text is split, e.g. ' cardiomyopathy' -> [' card', 'i', 'om', ...]."""
        return [self.vocab[i] for i in self.encode(text)]

    # -- save / load ----------------------------------------------------------------------------
    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"type": "bpe", "chars": self.chars,
                                          "merges": [list(m) for m in self.merges]}))

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        d = json.loads(Path(path).read_text())
        return cls(d["chars"], [tuple(m) for m in d["merges"]])


class Embeddings(nn.Module):
    """Token embedding + learned positional embedding, exactly as in the drill's GPT:
    x = token_embedding(idx) + position_embedding(arange(T))."""

    def __init__(self, vocab_size: int, block_size: int, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)
        return self.drop(self.token_embedding(idx) + self.position_embedding(pos))   # (B, T, n_embd)


def milestone_2(train_text: str, vocab_size: int = 1000, verbose: bool = False) -> BPETokenizer:
    """Train the BPE tokenizer on the training split (`scripts/tokenize_corpus.py` reports on it)."""
    return BPETokenizer.train(train_text, vocab_size, verbose=verbose)


# --- Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
# Hint: Implement the masked self-attention mechanism and feed-forward network.
#
# The drill's Head / MultiHeadAttention / FeedForward / Block / GPT (mini_gpt.py, notebooks 04-07),
# almost as-is. Changes: sizes come from a GPTConfig instead of globals, dropout, GPT-2 style
# N(0, 0.02) init (so the untrained loss sits at log(vocab)), and `last_attention` kept for tests.


@dataclass
class GPTConfig:
    vocab_size: int = 1000
    block_size: int = 128     # context length (tokens)
    n_embd: int = 128         # embedding size
    n_heads: int = 4          # attention heads
    n_layers: int = 4         # transformer blocks
    dropout: float = 0.1


class Head(nn.Module):
    """One head of masked scaled dot-product self-attention (drill lesson 4 + 5)."""

    def __init__(self, cfg: GPTConfig, head_size: int):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.drop = nn.Dropout(cfg.dropout)
        self.last_attention = None

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5            # (B, T, T) scaled scores
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))   # causal mask: no peeking ahead
        wei = F.softmax(wei, dim=-1)
        self.last_attention = wei.detach()
        return self.drop(wei) @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_heads
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg.n_heads)])
        self.proj = nn.Linear(cfg.n_heads * head_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.ReLU(),
                                 nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Communicate (attention) then think (feed-forward); pre-norm with residual connections."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.sa, self.ff = MultiHeadAttention(cfg), FeedForward(cfg)
        self.ln1, self.ln2 = nn.LayerNorm(cfg.n_embd), nn.LayerNorm(cfg.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ff(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = Embeddings(cfg.vocab_size, cfg.block_size, cfg.n_embd, cfg.dropout)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.ln_f(self.blocks(self.embed(idx)))
        logits = self.lm_head(x)                                       # (B, T, vocab)
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def milestone_3(cfg: GPTConfig | None = None) -> GPT:
    """Build the GPT for a given config."""
    return GPT(cfg or GPTConfig())


# --- Milestone 4: Set up the training loop and train the GPT model on the medical text data.
# Hint: Set up your optimizer, loss function, and training loop.
#
# Same loop as the drill's lesson 7 (random windows, AdamW, cross-entropy, estimate_loss), plus a
# cosine learning-rate decay, gradient clipping and a checkpoint that bundles weights + config + tokenizer.


def get_batch(data: torch.Tensor, block_size: int, batch_size: int):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model: GPT, splits: dict, batch_size: int, n_batches: int = 20) -> dict:
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = [model(*get_batch(data, model.cfg.block_size, batch_size))[1].item() for _ in range(n_batches)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def bigram_baseline(train_ids: list[int], val_ids: list[int], vocab_size: int) -> float:
    """Validation cross-entropy of a count-based bigram model (add-one smoothing): the bar to beat."""
    counts = torch.ones(vocab_size, vocab_size)
    a, b = torch.tensor(train_ids[:-1]), torch.tensor(train_ids[1:])
    counts.index_put_((a, b), torch.ones(len(a)), accumulate=True)
    logp = (counts / counts.sum(1, keepdim=True)).log()
    va, vb = torch.tensor(val_ids[:-1]), torch.tensor(val_ids[1:])
    return -logp[va, vb].mean().item()


def train_model(model: GPT, train_data: torch.Tensor, val_data: torch.Tensor, max_iters: int,
                batch_size: int = 32, lr: float = 1e-3, eval_every: int | None = None,
                eval_batches: int = 20, log=print) -> list[dict]:
    """AdamW + cross-entropy; returns the loss history [{step, train, val, secs}, ...]."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, max_iters) / max(max_iters, 1))))
    eval_every = eval_every or max(max_iters // 10, 1)
    splits = {"train": train_data, "val": val_data}
    history, t0 = [], time.time()
    model.train()
    for step in range(max_iters + 1):
        if step % eval_every == 0 or step == max_iters:
            losses = estimate_loss(model, splits, batch_size, eval_batches)
            history.append({"step": step, **losses, "secs": round(time.time() - t0, 1)})
            log(f"step {step:5d}  train {losses['train']:.3f}  val {losses['val']:.3f}  ({time.time() - t0:.0f}s)")
        if step == max_iters:
            break
        _, loss = model(*get_batch(train_data, model.cfg.block_size, batch_size))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    return history


class ModelNotFoundError(FileNotFoundError):
    pass


def save_checkpoint(path, model: GPT, tok: BPETokenizer, extra: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict(),
                "tokenizer": {"chars": tok.chars, "merges": [list(m) for m in tok.merges]},
                "extra": extra or {}}, path)


def load_checkpoint(path) -> tuple[GPT, BPETokenizer, dict]:
    if not Path(path).exists():
        raise ModelNotFoundError(
            f"No trained model at {path}.\nTrain one first (about 10 minutes on a laptop CPU):\n"
            "    python scripts/prepare_data.py\n    python scripts/train.py")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**ck["config"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, BPETokenizer(ck["tokenizer"]["chars"], ck["tokenizer"]["merges"]), ck.get("extra", {})


def milestone_4(train_text: str, val_text: str, tok: BPETokenizer, cfg: GPTConfig, max_iters: int,
                batch_size: int = 32, **kw) -> tuple[GPT, list[dict]]:
    """Tokenize the splits, build the GPT and train it (`scripts/train.py` handles timing, files, plots)."""
    train_data = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_data = torch.tensor(tok.encode(val_text), dtype=torch.long)
    model = GPT(cfg)
    return model, train_model(model, train_data, val_data, max_iters, batch_size, **kw)


# --- Milestone 5: Implement a text generation function to complete medical phrases or definitions.
# Hint: Utilize the model's prediction capabilities to generate sequences.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# --- Milestone 6: Create a CLI for students to enter partial phrases and get completions.
# Hint: Design a simple command-line interface for user interaction.
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
