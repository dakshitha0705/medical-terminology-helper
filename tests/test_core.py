"""Tests for each milestone. They use tiny made-up text and tiny random models only,
so they pass on CI with no dataset, no trained model and no API key."""
from app import core
from app.core import Source

# A miniature Gutenberg book with everything the cleaner has to cope with.
FAKE_BOOK = """The Project Gutenberg eBook of A Fake Anatomy
This eBook is for the use of anyone anywhere in the United States.
*** START OF THE PROJECT GUTENBERG EBOOK 99999 ***

CONTENTS.

                    CHAPTER I.
    THE BONES                                    11

CHAPTER I.

THE BONES.

1. The _femur_ is the longest and strongest bone of the body, and it
reaches from the hip to the knee.

[Illustration: FIG. 1.--The femur
(after Gray).]

=The Heart.=--The heart is a hollow muscular organ (Fig. 2) which pumps
the blood through the arteries; it beats about seventy times a minute.

  -=-=-=-=-=-=-=-=-=-=-=-=

1. What is the femur? 2. Where does it reach?

  -=-=-=-=-=-=-=-=-=-=-=-=

 Potatoes                 boiled       3       30

The cœliac artery arises from the aorta, just below the diaphragm—close to it.

GLOSSARY

AB-DUC´TOR. [L. _abduco_, to lead away.] A muscle which moves certain
parts away from the axis of the body.

INDEX.

    Femur, 11
    Heart, 12

*** END OF THE PROJECT GUTENBERG EBOOK 99999 ***
Licence text that must disappear.
"""
FAKE_SOURCE = Source(99999, "A Fake Anatomy", "Nobody", 1900, start=r"^CHAPTER I\.$",
                     end=r"^INDEX\.$", glossary=r"^GLOSSARY$")


# Milestone 1: Curate a small corpus of public domain medical texts (e.g., anatomy, physiology).
def test_milestone_1_strips_the_gutenberg_licence():
    body = core.strip_gutenberg(FAKE_BOOK)
    assert "START OF" not in body and "END OF" not in body
    assert "Licence text" not in body and "for the use of anyone" not in body
    assert "femur" in body


def test_milestone_1_normalises_characters_to_ascii():
    assert core.normalize_chars("cœliac “heart”—½ in, 37°") == 'coeliac "heart"--1/2 in, 37 degrees'
    assert core.normalize_chars("café").isascii()


def test_milestone_1_cleans_a_book_into_prose_paragraphs():
    paras = core.clean_book(FAKE_BOOK, FAKE_SOURCE)
    text = "\n".join(paras)
    assert paras[0].startswith("The femur is the longest")          # paragraph number + italics gone
    assert "The Heart.--The heart is a hollow muscular organ which pumps" in text   # bold + (Fig. 2) gone
    assert "coeliac artery" in text and "diaphragm--close" in text
    assert "Illustration" not in text and "Gray" not in text           # figure caption dropped
    assert "What is the femur" not in text                             # review questions dropped
    assert "CONTENTS" not in text and "Potatoes" not in text           # contents + tables dropped
    assert "Femur, 11" not in text                                     # index dropped
    assert paras[-1] == "Abductor: A muscle which moves certain parts away from the axis of the body."
    assert text.isascii() and all("\n" not in p for p in paras)


def test_milestone_1_split_keeps_train_and_val_apart():
    paras = [f"paragraph number {i} about the heart and lungs." for i in range(400)]
    train, val = core.split_train_val(paras)
    t, v = set(train.splitlines()), set(val.splitlines())
    assert not (t & v) and len(t) + len(v) == 400
    assert len(v) == 40                                               # 10% held out


def test_milestone_1_corpus_stats():
    s = core.corpus_stats("The heart beats.\nThe lungs breathe.\n")
    assert s["words"] == 6 and s["paragraphs"] == 2 and s["distinct_words"] == 5


def test_milestone_1_runs_end_to_end_without_downloads(monkeypatch):
    monkeypatch.setattr(core, "SOURCES", [FAKE_SOURCE])
    corpus, train, val = core.milestone_1({99999: FAKE_BOOK})
    assert corpus.startswith("The femur") and train and corpus.endswith("\n")


# Milestone 2: Implement the byte-pair tokenizer and generate embeddings from the medical corpus.
import torch  # noqa: E402

TINY = ("The heart pumps blood. The heart has four chambers. The pericardium surrounds the heart.\n"
        "Cardiac muscle contracts; the cardiac cycle repeats about 72 times a minute.\n"
        'Heart: The "pump" of the blood.\n') * 5


def test_milestone_2_bpe_round_trips_and_compresses():
    tok = core.BPETokenizer.train(TINY, vocab_size=80)
    assert tok.vocab_size == 80 and len(tok.merges) == 80 - len(set(TINY))
    ids = tok.encode(TINY)
    assert tok.decode(ids) == TINY
    assert len(ids) < len(TINY) / 2                       # merges really compress
    assert " heart" in tok.vocab                          # a frequent word became one token
    assert "".join(tok.tokens(" cardiac")) == " cardiac"
    unseen = "The blood of the cardiac heart."
    assert tok.decode(tok.encode(unseen)) == unseen


def test_milestone_2_most_frequent_pair_is_merged_first():
    tok = core.BPETokenizer.train("ab ab ab ab cd", vocab_size=len(set("ab ab ab ab cd")) + 1)
    assert tok.vocab[-1] == "ab"


def test_milestone_2_save_and_load_give_the_same_tokenizer(tmp_path):
    tok = core.BPETokenizer.train(TINY, vocab_size=70)
    tok.save(tmp_path / "tok.json")
    tok2 = core.BPETokenizer.load(tmp_path / "tok.json")
    assert tok2.vocab == tok.vocab and tok2.encode(TINY) == tok.encode(TINY)


def test_milestone_2_unknown_characters_raise_a_clear_error():
    tok = core.BPETokenizer.train(TINY, vocab_size=60)
    try:
        tok.encode("The heart @ #")
        raise AssertionError("expected UnknownCharacterError")
    except core.UnknownCharacterError as e:
        assert e.chars == ["#", "@"]


def test_milestone_2_vocab_smaller_than_alphabet_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        core.BPETokenizer.train(TINY, vocab_size=5)


def test_milestone_2_embeddings_add_token_and_position_vectors():
    emb = core.Embeddings(vocab_size=50, block_size=16, n_embd=8)
    idx = torch.randint(0, 50, (3, 10))
    x = emb(idx)
    assert x.shape == (3, 10, 8)
    expected = emb.token_embedding(idx) + emb.position_embedding(torch.arange(10))
    assert torch.allclose(x, expected)


# Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
import math  # noqa: E402

SMALL = core.GPTConfig(vocab_size=50, block_size=16, n_embd=32, n_heads=4, n_layers=2, dropout=0.0)


def test_milestone_3_causal_mask_blocks_the_future():
    torch.manual_seed(0)
    model = core.GPT(SMALL).eval()
    x = torch.randint(0, 50, (1, 12))
    logits, _ = model(x)
    w = model.blocks[0].sa.heads[0].last_attention[0]
    assert torch.all(w.triu(1) == 0) and torch.allclose(w.sum(-1), torch.ones(12))
    x2 = x.clone(); x2[0, 8:] = (x2[0, 8:] + 1) % 50                # change only the future
    logits2, _ = model(x2)
    assert torch.allclose(logits[0, :8], logits2[0, :8], atol=1e-6)


def test_milestone_3_shapes_and_untrained_loss_near_log_vocab():
    torch.manual_seed(0)
    model = core.milestone_3(SMALL)
    x, y = torch.randint(0, 50, (4, 16)), torch.randint(0, 50, (4, 16))
    logits, loss = model(x, y)
    assert logits.shape == (4, 16, 50)
    assert model.blocks[0](torch.randn(4, 16, 32)).shape == (4, 16, 32)
    assert abs(loss.item() - math.log(50)) < 0.1


# Milestone 4: Set up the training loop and train the GPT model on the medical text data.
def _tiny_trained(steps=60):
    torch.manual_seed(0)
    tok = core.BPETokenizer.train(TINY, vocab_size=60)
    cfg = core.GPTConfig(vocab_size=60, block_size=16, n_embd=32, n_heads=2, n_layers=1, dropout=0.0)
    model, hist = core.milestone_4(TINY, TINY, tok, cfg, max_iters=steps, batch_size=8, lr=3e-3,
                                   eval_batches=2, log=lambda *_: None)
    return model, tok, hist


def test_milestone_4_training_lowers_the_loss():
    model, tok, hist = _tiny_trained()
    assert hist[0]["step"] == 0 and hist[-1]["step"] == 60 and len(hist) == 11
    assert hist[-1]["train"] < hist[0]["train"] - 1.0


def test_milestone_4_bigram_baseline_learns_a_deterministic_sequence():
    ids = [0, 1, 2, 3] * 200
    assert core.bigram_baseline(ids, ids, 4) < 0.05 < math.log(4)


def test_milestone_4_checkpoint_save_and_load(tmp_path):
    model, tok, _ = _tiny_trained(steps=5)
    core.save_checkpoint(tmp_path / "m.pt", model, tok, extra={"val": 1.0})
    model2, tok2, extra = core.load_checkpoint(tmp_path / "m.pt")
    x = torch.tensor([tok.encode("The heart")])
    assert torch.allclose(model.eval()(x)[0], model2(x)[0]) and tok2.vocab == tok.vocab
    assert extra == {"val": 1.0}


def test_milestone_4_missing_model_says_how_to_train(tmp_path):
    import pytest
    with pytest.raises(core.ModelNotFoundError, match="scripts/train.py"):
        core.load_checkpoint(tmp_path / "nope.pt")


# Milestone 5: Implement a text generation function to complete medical phrases or definitions.
def test_milestone_5_top_k_and_top_p_keep_only_likely_tokens():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]])
    assert torch.isfinite(core.filter_logits(logits, top_k=2)).sum() == 2
    p = F_softmax = torch.softmax(logits, -1)[0]
    kept = torch.isfinite(core.filter_logits(logits, top_p=0.7)).sum().item()
    assert p[:kept].sum() >= 0.7 and p[:kept - 1].sum() < 0.7
    assert torch.isfinite(core.filter_logits(logits, top_p=0.01)).sum() == 1   # always keeps the best
    sharp = torch.softmax(core.filter_logits(logits, temperature=0.5), -1)
    assert sharp[0, 0] > F_softmax[0]                                            # low T sharpens


def test_milestone_5_complete_respects_length_and_seed():
    model, tok, _ = _tiny_trained(steps=5)
    a = core.complete(model, tok, "The heart", max_new_tokens=7, seed=1)
    b = core.complete(model, tok, "The heart", max_new_tokens=7, seed=1)
    assert a == b and len(tok.encode("The heart" + a)) >= len(tok.encode("The heart")) + 1
    assert len(core.generate(model, torch.tensor([tok.encode("The heart")]), 7)[0]) == \
        len(tok.encode("The heart")) + 7


def test_milestone_5_define_uses_glossary_style_and_rejects_bad_input():
    import pytest
    assert core.define_prompt("  tachyCARDIA ") == "Tachycardia:"
    model, tok, _ = _tiny_trained(steps=5)
    assert "\n" not in core.define(model, tok, "heart", max_new_tokens=10, seed=0)
    for bad in [dict(prompt=" "), dict(prompt="The", max_new_tokens=0), dict(prompt="The", temperature=0)]:
        with pytest.raises(ValueError):
            core.complete(model, tok, **bad)


# Milestone 6: Create a CLI for students to enter partial phrases and get completions.
import pytest  # noqa: E402

from app import cli  # noqa: E402


@pytest.fixture
def tiny_model(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "app.json"))      # history goes to a temp file
    model, tok, _ = _tiny_trained(steps=5)
    path = tmp_path / "m.pt"
    core.save_checkpoint(path, model, tok, extra={"params": model.num_params()})
    return str(path)


def test_milestone_6_parses_commands_and_knobs():
    a = cli.build_parser().parse_args(["complete", "The femur is", "-n", "20", "-t", "0.5", "-k", "10", "-p", "0.9"])
    assert (a.command, a.prompt, a.length, a.temperature, a.top_k, a.top_p) == ("complete", "The femur is", 20, 0.5, 10, 0.9)
    assert cli.build_parser().parse_args(["define", "tachycardia"]).term == "tachycardia"
    for bad in (["complete", "x", "-t", "0"], ["complete", "x", "-n", "0"], ["complete", "x", "-p", "2"]):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(bad)


def test_milestone_6_one_shot_complete_and_define(tiny_model, capsys):
    assert cli.main(["complete", "The heart", "-n", "5", "--seed", "0", "--model", tiny_model]) == 0
    out = capsys.readouterr().out
    assert "not medical advice" in out.lower() and "The heart" in out
    assert cli.main(["define", "heart", "--seed", "0", "--model", tiny_model]) == 0
    assert "Heart:" in capsys.readouterr().out
    assert cli.main(["history"]) == 0 and "[define] heart" in capsys.readouterr().out


def test_milestone_6_friendly_errors(tiny_model, tmp_path, capsys):
    assert cli.main(["complete", "The femur", "--model", str(tmp_path / "missing.pt")]) == 1
    assert "python scripts/train.py" in capsys.readouterr().err
    assert cli.main(["complete", "   ", "--model", tiny_model]) == 1
    assert "empty" in capsys.readouterr().err
    assert cli.main(["complete", "The heart @", "--model", tiny_model]) == 1
    assert "never seen these characters: @" in capsys.readouterr().err
    assert cli.main(["complete", "The “heart”", "-n", "3", "--model", tiny_model]) == 0   # curly quotes normalised


def test_milestone_6_interactive_loop(tiny_model, capsys):
    lines = iter(["", ":len 3", ":temp 0", "The heart", "define heart", "The # heart", ":bogus", ":quit"])
    assert cli.main(["interactive", "--seed", "0", "--model", tiny_model], input_fn=lambda _: next(lines)) == 0
    out = capsys.readouterr().out
    assert "Type the start of a phrase" in out and "ok: length 3" in out and "Bad value for :temp" in out
    assert "Heart:" in out and "never seen these characters: #" in out and "Unknown command" in out
    assert out.rstrip().endswith("Goodbye.")


def test_milestone_6_symbols_that_cannot_be_normalised_are_reported(tiny_model, capsys):
    assert cli.main(["complete", "The heart ♥", "--model", tiny_model]) == 1
    assert "U+2665" in capsys.readouterr().err
