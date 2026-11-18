# Data sources

The training corpus is built by `python scripts/prepare_data.py` from four books on
[Project Gutenberg](https://www.gutenberg.org). The files land in `data/`, which is **never committed**.

| # | Book | Author | Published | Gutenberg URL | Licence |
|---|------|--------|-----------|---------------|---------|
| 1 | *Structure and Functions of the Body: A Hand-Book of Anatomy and Physiology for Nurses* | Annette Fiske | 1911 | https://www.gutenberg.org/ebooks/64754 | Public domain in the USA |
| 2 | *A Treatise on Anatomy, Physiology, and Hygiene* (Revised Edition), incl. its glossary | Calvin Cutter | 1870 | https://www.gutenberg.org/ebooks/30541 | Public domain in the USA |
| 3 | *Ellis's Primary Physiology, or Good Health for Boys and Girls* | Edward S. Ellis | 1900 | https://www.gutenberg.org/ebooks/52775 | Public domain in the USA |
| 4 | *Physiology* (Science Primers) | Michael Foster | 1880 | https://www.gutenberg.org/ebooks/53347 | Public domain in the USA |

**Why these are public domain.** All four were published in the USA before 1930, so their US
copyright has expired; Project Gutenberg lists each as "Public domain in the USA". Project
Gutenberg's own licence covers only its trademark and the eBook packaging (header/footer text),
which the cleaning step removes. If you are outside the USA, check your local copyright law
before redistributing the texts.

**How they are fetched.** From Project Gutenberg's own mirror (`gutenberg.pglaf.org`) with
`www.gutenberg.org` as a fallback, once, into `data/raw/` — Gutenberg asks automated clients to use
mirrors rather than the main site.

## Cleaning (`app/core.py`, milestone 1)

1. Keep only the text between the `*** START OF …` and `*** END OF …` markers (drops the licence).
2. Cut each book to its body: from the first chapter to the index / publisher's adverts /
   transcriber's notes (front matter, prefaces' tables of contents and indexes are dropped).
3. Remove `[Illustration: …]` figure captions, `[Footnote …]` blocks, `(Fig. 12)` references,
   page and footnote anchors, `_italic_` / `=bold=` markup and paragraph numbers.
4. Drop indented lines (contents pages, tables, centred headings, review-question lists),
   all-caps headings, and Cutter's question blocks between `-=-=-` rules.
5. Normalise characters to plain ASCII (`œ`→`oe`, curly quotes → straight, `—`→`--`, `½`→`1/2`).
6. Join lines into one paragraph per line; drop fragments under 6 words and number-heavy lines.
7. Cutter's glossary is kept and reformatted as `Term: definition`
   (e.g. `Pericardium: A membrane that encloses the heart.`), which gives the model a definition style.

Split: every 10th block of 20 paragraphs is held out for validation (~9%), so validation text comes
from every book but never overlaps training text.

## Caveat

These are 1870–1911 textbooks. Some physiology and much of the "hygiene" advice is outdated or
wrong by modern standards. The model is for practising terminology, **not** medical advice.
