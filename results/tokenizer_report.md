# Tokenizer report

BPE trained from scratch on `data/train.txt` in 1.1s: 80 base characters + 920 merges = **1000 tokens**.

## Compression

| split | characters | BPE tokens | chars / token |
|---|---|---|---|
| train | 997,881 | 327,927 | 3.04 |
| val | 100,598 | 33,190 | 3.03 |

Round trip `decode(encode(text)) == text`: **True** for both splits (a character-level tokenizer like the drill's would be 1.00 chars/token).

## How medical words split

| word | tokens |
|---|---|
| cardiomyopathy | ' c' · 'ard' · 'i' · 'om' · 'y' · 'op' · 'ath' · 'y' |
| tachycardia | ' t' · 'ach' · 'y' · 'c' · 'ard' · 'ia' |
| pericardium | ' per' · 'ic' · 'ard' · 'ium' |
| hemoglobin | ' he' · 'm' · 'o' · 'g' · 'lo' · 'b' · 'in' |
| gastrocnemius | ' g' · 'ast' · 'ro' · 'c' · 'n' · 'em' · 'i' · 'us' |
| osteoporosis | ' o' · 'st' · 'e' · 'op' · 'or' · 'os' · 'is' |
| femur | ' f' · 'em' · 'ur' |
| the heart | ' the' · ' heart' |
| vertebrae | ' v' · 'ert' · 'eb' · 'ra' · 'e' |
| diaphragm | ' diaphragm' |
| epithelium | ' e' · 'p' · 'it' · 'he' · 'l' · 'ium' |
| neuralgia | ' ne' · 'ur' · 'al' · 'g' · 'ia' |

## Longest learned tokens

`' capillaries'`, `' contraction'`, `' temperature'`, `'Observation'`, `' different'`, `' diaphragm'`, `' substance'`, `' important'`, `' condition'`, `' generally'`, `' membrane'`, `' arteries'`, `' muscular'`, `' function'`, `' together'`, `' contract'`, `' exercise'`, `' internal'`, `' quantity'`, `' external'`, `' anterior'`, `' carbonic'`, `' pressure'`, `' muscles'`, `' through'`

## First 30 merges

`' t'`, `'he'`, `' a'`, `' the'`, `'in'`, `' o'`, `'re'`, `' s'`, `'er'`, `' b'`, `' of'`, `'nd'`, `' c'`, `'on'`, `'is'`, `' w'`, `'es'`, `'at'`, `' f'`, `'it'`, `' in'`, `'en'`, `' p'`, `' and'`, `'ed'`, `' m'`, `'ar'`, `'ou'`, `'al'`, `'or'`

## Embeddings

`Embeddings(vocab=1000, block_size=128, n_embd=128)` turns the sentence "The femur is the longest bone of the body." (13 tokens) into a tensor of shape `(1, 13, 128)` = token vector + position vector. Parameters: 144,384 (1000×128 token table + 128×128 position table). These vectors start random and are learned during training (milestone 4).
