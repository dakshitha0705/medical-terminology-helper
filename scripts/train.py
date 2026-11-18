"""Milestone 4: train the medical GPT on the CPU within a time budget.

    python scripts/train.py                  # ~8 minutes, default budget
    python scripts/train.py --minutes 3      # quicker, weaker model

Times 20 steps first, then picks max_iters so the whole run fits the budget.
Saves models/medgpt.pt + models/tokenizer.json (never committed), results/loss.png, results/train_log.json.
"""
import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.core import (DATA_DIR, MODEL_DIR, RESULTS_DIR, GPT, BPETokenizer, GPTConfig, bigram_baseline,  # noqa: E402
                      get_batch, save_checkpoint, train_model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=8.0, help="total training time budget")
    ap.add_argument("--vocab-size", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    if not (DATA_DIR / "train.txt").exists():
        raise SystemExit("data/train.txt is missing. Run: python scripts/prepare_data.py")
    train_text = (DATA_DIR / "train.txt").read_text(encoding="utf-8")
    val_text = (DATA_DIR / "val.txt").read_text(encoding="utf-8")
    tok_path = MODEL_DIR / "tokenizer.json"
    tok = BPETokenizer.load(tok_path) if tok_path.exists() else BPETokenizer.train(train_text, args.vocab_size)
    tok.save(tok_path)
    train_ids, val_ids = tok.encode(train_text), tok.encode(val_text)
    train_data, val_data = torch.tensor(train_ids), torch.tensor(val_ids)
    print(f"tokens: train {len(train_ids):,}  val {len(val_ids):,}  vocab {tok.vocab_size}")

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=128, n_embd=128, n_heads=4, n_layers=4, dropout=0.1)
    model = GPT(cfg)
    print(f"model: {model.num_params():,} parameters  {asdict(cfg)}")

    # Time 20 steps on a throwaway copy, then size the run (evals cost ~11 x 2 x 20 forward passes).
    probe = GPT(cfg)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    t0 = time.time()
    for _ in range(20):
        _, loss = probe(*get_batch(train_data, cfg.block_size, args.batch_size))
        opt.zero_grad(); loss.backward(); opt.step()
    step_secs = (time.time() - t0) / 20
    eval_secs = 11 * 2 * 20 * step_secs / 3            # a forward pass is ~1/3 of a training step
    max_iters = max(100, int((args.minutes * 60 * 0.85 - eval_secs) / step_secs) // 100 * 100)
    print(f"timing: {step_secs * 1000:.0f} ms/step -> max_iters {max_iters} for a {args.minutes:.0f}-minute budget")
    del probe, opt

    baseline = bigram_baseline(train_ids, val_ids, tok.vocab_size)
    print(f"baselines: uniform log(V) = {math.log(tok.vocab_size):.3f}   bigram val loss = {baseline:.3f}")

    t0 = time.time()
    history = train_model(model, train_data, val_data, max_iters, args.batch_size, lr=1e-3)
    minutes = (time.time() - t0) / 60
    final = history[-1]
    log = {"config": asdict(cfg), "params": model.num_params(), "max_iters": max_iters,
           "batch_size": args.batch_size, "ms_per_step": round(step_secs * 1000, 1),
           "train_minutes": round(minutes, 2), "train_tokens": len(train_ids), "val_tokens": len(val_ids),
           "uniform_loss": round(math.log(tok.vocab_size), 4), "bigram_val_loss": round(baseline, 4),
           "bigram_val_perplexity": round(math.exp(baseline), 1),
           "final_train_loss": round(final["train"], 4), "final_val_loss": round(final["val"], 4),
           "final_val_perplexity": round(math.exp(final["val"]), 1), "history": history}
    save_checkpoint(MODEL_DIR / "medgpt.pt", model, tok, extra={k: v for k, v in log.items() if k != "history"})
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "train_log.json").write_text(json.dumps(log, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = [h["step"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, [h["train"] for h in history], label="train", color="#2a6fdb")
    ax.plot(steps, [h["val"] for h in history], label="validation", color="#e07b00")
    ax.axhline(baseline, ls="--", color="#777", label=f"bigram baseline ({baseline:.2f})")
    ax.set_xlabel("step"); ax.set_ylabel("cross-entropy loss"); ax.set_title("Medical GPT training loss")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(RESULTS_DIR / "loss.png", dpi=120)

    print(f"\ndone in {minutes:.1f} min: val loss {final['val']:.3f} (perplexity {math.exp(final['val']):.1f}) "
          f"vs bigram {baseline:.3f} (perplexity {math.exp(baseline):.1f})")
    print("saved models/medgpt.pt, models/tokenizer.json, results/train_log.json, results/loss.png")


if __name__ == "__main__":
    main()
