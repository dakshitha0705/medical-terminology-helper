"""Milestone 5: complete 10 fixed medical prompts at two settings -> results/samples.md."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.core import RESULTS_DIR  # noqa: E402
from app.llm import MedicalGPT  # noqa: E402

PROMPTS = ["The femur is", "The heart is a hollow muscular organ", "The function of the kidneys is",
           "Blood is composed of", "The lungs are", "The skin consists of",
           "The spinal cord", "Digestion begins in the mouth, where", "The pericardium", "Muscles are attached to"]
DEFINE = ["pericardium", "aorta", "larynx", "tachycardia", "diaphragm"]
SETTINGS = {"focused (T=0.5, top-k 20)": dict(temperature=0.5, top_k=20, top_p=None),
            "creative (T=1.0, top-p 0.9)": dict(temperature=1.0, top_k=None, top_p=0.9)}


def main() -> None:
    gpt = MedicalGPT()
    log = json.loads((RESULTS_DIR / "train_log.json").read_text())
    out = ["# Sample completions\n",
           f"Model: {log['params']:,}-parameter GPT, val loss {log['final_val_loss']} "
           f"(perplexity {log['final_val_perplexity']}) vs bigram {log['bigram_val_loss']}. "
           "40 new tokens per completion, seed 0, prompt in **bold**.\n"]
    for name, kw in SETTINGS.items():
        out.append(f"## {name}\n")
        for i, p in enumerate(PROMPTS, 1):
            text = gpt.complete(p, length=40, seed=0, **kw).replace("\n", " ")
            out.append(f"{i}. **{p}**{text}")
        out.append("")
    out.append("## define mode (glossary style `Term:`, stops at the end of the sentence)\n")
    for term in DEFINE:
        out.append(f"- **{term}** → {gpt.define(term, seed=0)}")
    out += ["", "## Honest note on quality\n",
            "- **What it learned:** the books' vocabulary and register — real terms in plausible places "
            "(ventricle, capillaries, pulmonary veins, cerebrum, ligaments), sentence rhythm, and the "
            "glossary's definition opener \"A term applied to …\".",
            "- **What it did not learn:** facts. Completions are fluent-looking but usually wrong or "
            "meaningless, with invented words (\"calseale\", \"concleule\") and run-on lists.",
            "- **Low temperature** repeats safe words (\"the body … the system\"); **high temperature** "
            "invents more words. 0.6–0.8 with top-k/top-p is the best trade-off.",
            "- **define** produces the same generic glossary-style sentence for every term: 428 "
            "definitions are far too few for a 1M-parameter model to tie terms to meanings.",
            "- Odd ALL-CAPS fragments come from capitalised headwords in Cutter's text.",
            "- Same seed = same random draws, so different prompts can drift into the same continuation.",
            "",
            "Use it to practise spotting and correcting terminology, never as a source of facts."]
    (RESULTS_DIR / "samples.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
