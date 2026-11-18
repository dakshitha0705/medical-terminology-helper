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
        "Cardiac muscle contracts; the cardiac cycle repeats about 72 times a minute.\n") * 5


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
