"""Milestone 2: train the BPE tokenizer on data/train.txt and report what it learned.

    python scripts/tokenize_corpus.py [--vocab-size 1500]

Saves models/tokenizer.json (not committed) and results/tokenizer_report.md.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.core import DATA_DIR, MODEL_DIR, RESULTS_DIR, Embeddings, milestone_2  # noqa: E402

WORDS = ["cardiomyopathy", "tachycardia", "pericardium", "hemoglobin", "gastrocnemius",
         "osteoporosis", "femur", "the heart", "vertebrae", "diaphragm", "epithelium", "neuralgia"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=1000)
    args = ap.parse_args()
    train = (DATA_DIR / "train.txt").read_text(encoding="utf-8")
    val = (DATA_DIR / "val.txt").read_text(encoding="utf-8")

    t0 = time.time()
    tok = milestone_2(train, args.vocab_size, verbose=True)
    secs = time.time() - t0
    tok.save(MODEL_DIR / "tokenizer.json")

    lines = [f"# Tokenizer report\n",
             f"BPE trained from scratch on `data/train.txt` in {secs:.1f}s: "
             f"{len(tok.chars)} base characters + {len(tok.merges)} merges = **{tok.vocab_size} tokens**.\n",
             "## Compression\n", "| split | characters | BPE tokens | chars / token |", "|---|---|---|---|"]
    for name, text in [("train", train), ("val", val)]:
        n = len(tok.encode(text))
        lines.append(f"| {name} | {len(text):,} | {n:,} | {len(text) / n:.2f} |")
        assert tok.decode(tok.encode(text)) == text, "round trip failed"
    lines += ["", "Round trip `decode(encode(text)) == text`: **True** for both splits "
              "(a character-level tokenizer like the drill's would be 1.00 chars/token).\n",
              "## How medical words split\n", "| word | tokens |", "|---|---|"]
    for w in WORDS:
        lines.append(f"| {w} | {' · '.join(repr(t) for t in tok.tokens(' ' + w))} |")
    longest = sorted(tok.vocab, key=len, reverse=True)[:25]
    lines += ["", "## Longest learned tokens\n", ", ".join(f"`{t!r}`" for t in longest),
              "", "## First 30 merges\n", ", ".join(f"`{tok.vocab[len(tok.chars) + i]!r}`" for i in range(30))]

    # Embeddings: look up token + position vectors for one sentence (an untrained layer).
    torch.manual_seed(0)
    emb = Embeddings(tok.vocab_size, block_size=128, n_embd=128)
    ids = torch.tensor([tok.encode("The femur is the longest bone of the body.")])
    x = emb(ids)
    lines += ["", "## Embeddings\n",
              f"`Embeddings(vocab={tok.vocab_size}, block_size=128, n_embd=128)` turns the sentence "
              f"\"The femur is the longest bone of the body.\" ({ids.shape[1]} tokens) into a tensor of shape "
              f"`{tuple(x.shape)}` = token vector + position vector. "
              f"Parameters: {sum(p.numel() for p in emb.parameters()):,} "
              f"({tok.vocab_size}×128 token table + 128×128 position table). "
              "These vectors start random and are learned during training (milestone 4)."]

    RESULTS_DIR.mkdir(exist_ok=True)
    report = "\n".join(lines) + "\n"
    (RESULTS_DIR / "tokenizer_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
