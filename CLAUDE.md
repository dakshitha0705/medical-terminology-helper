# Medical Terminology Helper — project plan

A command-line assistant that completes medical phrases and definitions with a small GPT built
from scratch (BPE tokenizer + transformer), trained on public-domain anatomy/physiology books,
running on a laptop CPU. It builds on the "Build GPT from Scratch" drill
(`E:\projects\Build-GPT-from-Scratch\work\mini_gpt.py`, notebooks 04–09).

## Layout (keep it — fill in, don't restructure)
- `app/core.py` — all the ML code, one section per milestone: corpus cleaning, BPE tokenizer,
  embeddings, causal self-attention + transformer block + GPT, training loop, generation.
  The starter's `milestone_N()` stubs become small entry points that run each milestone.
- `app/llm.py` — the "LLM" is now our own local GPT: save/load the checkpoint + tokenizer and
  complete text. No API, no key, no `.env` needed.
- `app/storage.py` — the starter's JSON store, used for the CLI's recent-prompt history.
- `app/cli.py` — `complete`, `define`, interactive loop, `train`, `info`, `history`.
- `scripts/prepare_data.py` — download + clean the corpus (milestone 1); `scripts/train.py` (milestone 4);
  `scripts/make_samples.py` (milestone 5).
- `tests/` — run on CI with **no dataset, no trained model, no key**: tiny random text/models only.

## Milestones (askThili titles; commit + push after each)
1. Curate a small corpus of public domain medical texts (e.g., anatomy, physiology).
2. Implement the byte-pair tokenizer and generate embeddings from the medical corpus.
3. Build the causal self-attention mechanism and assemble the transformer block.
4. Set up the training loop and train the GPT model on the medical text data.
5. Implement a text generation function to complete medical phrases or definitions.
6. Create a CLI for students to enter partial phrases and get completions.

## Rules
- Never commit `data/`, `models/` (weights/tokenizer), `.env`, or `.venv/`. Commit files by name only.
- `requirements.txt` uses CPU torch (`--extra-index-url https://download.pytorch.org/whl/cpu`).
- Hand-written attention only (no `nn.MultiheadAttention`). Training must finish in ≤ 15 min on CPU.
- Downloads use Project Gutenberg's own mirror (gutenberg.pglaf.org), as PG asks of automated clients.
- Commit message: `Milestone N: <title> — <result>` + Co-Authored-By trailer.

## Commands
```bash
py -3.13 -m venv .venv && .venv/Scripts/activate      # Windows (CI uses 3.12)
pip install -r requirements.txt
python scripts/prepare_data.py     # -> data/corpus.txt, data/train.txt, data/val.txt
python scripts/train.py            # -> models/medgpt.pt, models/tokenizer.json, results/
python -m app.cli complete "The femur is"
python -m app.cli define "tachycardia"
python -m app.cli interactive
pytest -q
```
