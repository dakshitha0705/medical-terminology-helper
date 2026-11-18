# Medical Terminology Helper

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/xka8hhccyz/badge.svg)](https://ask.thili.ai/b/xka8hhccyz)

This project develops a command-line assistant to help medical students reinforce terminology. By training a GPT model on public domain medical texts, the tool provides intelligent completions for medical phrases and definitions, making learning more interactive and efficient.

## What you are building

**Goal.** You will build a command-line assistant that completes medical phrases and definitions, leveraging a GPT model trained from scratch on public domain medical texts.

**Done when.** The project is complete when you have a trained GPT model capable of generating coherent medical text completions, accessible through a command-line interface where students can input partial phrases and receive relevant medical terminology suggestions.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Curate a small corpus of public domain medical texts (e.g., anatomy, physiology).**  
      Hint: Use text processing techniques to clean and prepare your corpus.
- [ ] **2. Implement the byte-pair tokenizer and generate embeddings from the medical corpus.**  
      Hint: Apply the byte-pair encoding (BPE) and embedding layer concepts.
- [ ] **3. Build the causal self-attention mechanism and assemble the transformer block.**  
      Hint: Implement the masked self-attention mechanism and feed-forward network.
- [ ] **4. Set up the training loop and train the GPT model on the medical text data.**  
      Hint: Set up your optimizer, loss function, and training loop.
- [ ] **5. Implement a text generation function to complete medical phrases or definitions.**  
      Hint: Utilize the model's prediction capabilities to generate sequences.
- [ ] **6. Create a CLI for students to enter partial phrases and get completions.**  
      Hint: Design a simple command-line interface for user interaction.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow the user to specify the length of the generated completion.
- Implement a feature to save and load the trained model weights, avoiding retraining.
- Add basic error handling for user input in the CLI.
