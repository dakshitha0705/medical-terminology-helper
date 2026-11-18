"""Command-line entry point: `python -m app.cli --help`.

    python -m app.cli complete "The femur is" --length 40 --temperature 0.7
    python -m app.cli define "tachycardia"
    python -m app.cli interactive          # type partial phrases, get completions
    python -m app.cli info | history
"""
import argparse
import sys

from app import core
from app.storage import Store

NOTICE = ("Educational use only - NOT medical advice. Completions come from a small model trained on "
          "1870-1911 textbooks;\nthey can be wrong, outdated or made up. Check a current source.")
HELP_LOOP = """Type a partial phrase and press Enter to complete it.
  define <term>      define a term (e.g. define pericardium)
  :len N             completion length in tokens (now {length})
  :temp T            temperature, > 0 (now {temperature})
  :topk K / :topp P  sampling filters (now {top_k} / {top_p}); 0 turns one off
  :help              this help            :quit  leave (or Ctrl-C)"""


def _positive_int(s: str) -> int:
    v = int(s)
    if not 1 <= v <= 500:
        raise argparse.ArgumentTypeError("must be between 1 and 500")
    return v


def _temperature(s: str) -> float:
    v = float(s)
    if not 0 < v <= 5:
        raise argparse.ArgumentTypeError("must be > 0 and <= 5")
    return v


def _top_p(s: str) -> float:
    v = float(s)
    if not 0 <= v <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1 (0 = off)")
    return v


def _top_k(s: str) -> int:
    v = int(s)
    if v < 0:
        raise argparse.ArgumentTypeError("must be >= 0 (0 = off)")
    return v


def _show(chars: list[str]) -> str:
    """'@' stays '@'; non-ASCII characters also get their code point, e.g. '♥ (U+2665)'."""
    return " ".join(c if c.isascii() else f"{c} (U+{ord(c):04X})" for c in chars)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medical-terminology-helper", description="Medical students get a command-line assistant completing medical phrases and definitions, trained on public domain medical texts to reinforce terminology learning.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")

    knobs = argparse.ArgumentParser(add_help=False)
    knobs.add_argument("--length", "-n", type=_positive_int, default=None, help="tokens to generate (1-500)")
    knobs.add_argument("--temperature", "-t", type=_temperature, default=0.8, help="randomness, > 0 (default 0.8)")
    knobs.add_argument("--top-k", "-k", type=_top_k, default=40, help="sample from the k likeliest tokens (0 = off)")
    knobs.add_argument("--top-p", "-p", type=_top_p, default=0.95, help="nucleus sampling mass (0 = off)")
    knobs.add_argument("--seed", type=int, default=None, help="fix the random seed for repeatable output")
    knobs.add_argument("--model", default=None, help="checkpoint path (default models/medgpt.pt)")

    c = sub.add_parser("complete", parents=[knobs], help="complete a partial medical phrase")
    c.add_argument("prompt", help='e.g. "The femur is"')
    c.add_argument("--sentence", action="store_true", help="stop at the end of the first sentence")
    d = sub.add_parser("define", parents=[knobs], help="ask the model to define a term")
    d.add_argument("term", help='e.g. "tachycardia"')
    sub.add_parser("interactive", parents=[knobs], help="type partial phrases in a loop")
    sub.add_parser("info", parents=[knobs], help="show the trained model's size and scores")
    sub.add_parser("history", help="show your recent prompts")
    return parser


def _load(args):
    from app.llm import MedicalGPT
    return MedicalGPT(args.model)


def _knobs(args, length_default: int) -> dict:
    return dict(length=args.length or length_default, temperature=args.temperature,
                top_k=args.top_k or None, top_p=args.top_p or None, seed=args.seed)


def _remember(kind: str, text: str, output: str) -> None:
    try:
        store = Store()
        hist = store.get("history", [])[-49:]
        store.put("history", hist + [{"kind": kind, "input": text, "output": output}])
    except OSError:
        pass                                            # history is a nice-to-have


def _answer(gpt, line: str, knobs: dict) -> str:
    if line.lower().startswith("define "):
        term = line[7:].strip()
        out = gpt.define(term, **{**knobs, "length": min(knobs["length"], 60)})
        _remember("define", term, out)
        return f"{core.define_prompt(term)} {out}"
    out = gpt.complete(line, **knobs)
    _remember("complete", line, out)
    return line + out


def interactive(gpt, args, input_fn=input) -> None:
    knobs = _knobs(args, 40)
    print(HELP_LOOP.format(**knobs))
    while True:
        try:
            line = input_fn("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if not line:
            print("Type the start of a phrase, e.g. 'The femur is' (or :help).")
            continue
        if line in (":q", ":quit", "quit", "exit"):
            print("Goodbye.")
            return
        if line.startswith(":"):
            cmd, _, val = line[1:].partition(" ")
            try:
                if cmd == "help":
                    print(HELP_LOOP.format(**knobs))
                elif cmd == "len":
                    knobs["length"] = _positive_int(val)
                elif cmd == "temp":
                    knobs["temperature"] = _temperature(val)
                elif cmd == "topk":
                    knobs["top_k"] = _top_k(val) or None
                elif cmd == "topp":
                    knobs["top_p"] = _top_p(val) or None
                else:
                    print(f"Unknown command :{cmd} (try :help)")
                    continue
                if cmd != "help":
                    print(f"ok: length {knobs['length']}, temperature {knobs['temperature']}, "
                          f"top-k {knobs['top_k']}, top-p {knobs['top_p']}")
            except (ValueError, argparse.ArgumentTypeError) as e:
                print(f"Bad value for :{cmd}: {e}")
            continue
        try:
            print(_answer(gpt, line, knobs))
        except core.UnknownCharacterError as e:
            print(f"Sorry, the model has never seen these characters: {_show(e.chars)}. Try plain English letters.")
        except ValueError as e:
            print(f"Sorry: {e}")


def main(argv: list[str] | None = None, input_fn=input) -> int:
    for stream in (sys.stdout, sys.stderr):           # Windows consoles can't print every character
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except (ValueError, OSError):
                pass
    args = build_parser().parse_args(argv)
    from app.llm import load_env
    load_env()
    if args.command == "hello":
        print("Medical Terminology Helper is ready. Open README.md for your milestones.")
        return 0
    if args.command == "history":
        hist = Store().get("history", [])
        if not hist:
            print("No history yet. Try: python -m app.cli complete \"The femur is\"")
        for h in hist[-20:]:
            print(f"[{h['kind']}] {h['input']} -> {h['output'][:80]}")
        return 0

    try:
        if args.command in ("complete", "define") and not (getattr(args, "prompt", None) or
                                                           getattr(args, "term", None) or "").strip():
            raise ValueError("please type a phrase or term (the input was empty)")
        gpt = _load(args)
        if args.command == "info":
            i = gpt.info
            print(f"Model: {gpt.path}\n  parameters {i.get('params', 0):,}, vocab {gpt.tok.vocab_size}, "
                  f"context {gpt.model.cfg.block_size} tokens\n  val loss {i.get('final_val_loss')} "
                  f"(perplexity {i.get('final_val_perplexity')}) vs bigram {i.get('bigram_val_loss')} "
                  f"(perplexity {i.get('bigram_val_perplexity')}), trained {i.get('train_minutes')} min")
            return 0
        print(NOTICE + "\n")
        if args.command == "complete":
            out = gpt.complete(args.prompt, stop_at_sentence=args.sentence, **_knobs(args, 60))
            _remember("complete", args.prompt, out)
            print(args.prompt + out)
        elif args.command == "define":
            out = gpt.define(args.term, **_knobs(args, 40))
            _remember("define", args.term, out)
            print(f"{core.define_prompt(args.term)} {out}")
        else:
            interactive(gpt, args, input_fn)
        return 0
    except core.ModelNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
    except core.UnknownCharacterError as e:
        print(f"Error: the model has never seen these characters: {_show(e.chars)}. "
              "Use plain English letters, digits and punctuation.", file=sys.stderr)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
