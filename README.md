# Nepali BPE Tokenizer

A small, educational **Byte Pair Encoding (BPE) tokenizer for Nepali**
(Devanagari script), implemented **from scratch in pure Python standard
library** — no Hugging Face `tokenizers`, no SentencePiece, no third-party
packages at all.

```text
Nepali corpus
→ Unicode normalization (NFC)
→ vocabulary initialization (words → Unicode symbols + </w>)
→ pair-frequency counting
→ BPE merge learning
→ learned vocabulary
→ encode Nepali text into token IDs
→ decode token IDs back into Nepali text
```

## Quick start

```bash
python train.py            # train a tokenizer on corpus.txt (target vocab 2048)
python test_tokenizer.py   # unit tests + evaluation metrics
```

```bash
python train.py --target-vocab 8192 --verbose     # grow to ~8192 tokens, print every merge
python train.py --merges 1500 --max-words 5000    # explicit budget + word cap
```

A trained tokenizer can be saved and reused:

```python
from npltokenizer import NepaliBPETokenizer

tok = NepaliBPETokenizer.load("tokenizer.json")

text = "काठमाडौँ नेपालको राजधानी हो।"
ids = tok.encode(text)        # [12, 47, ...]
print(tok.tokenize(text))     # ['काठमा', 'डौँ</w>', ' ', ...]
print(tok.decode(ids))        # काठमाडौँ नेपालको राजधानी हो।
```

---

## 1. What is BPE?

Byte Pair Encoding started as a data-compression trick (from the original
"byte pair" papers) and was repurposed by Sennrich et al. (2016, *Neural
Machine Translation of Rare Words with Subword Units*) to subdivide text.

The idea is simple:

1. Split every word of a corpus into its raw symbols (here: Unicode code points).
2. Count how often each **adjacent pair** of symbols appears, weighted by word frequency.
3. Find the **most frequent pair**, fuse it into one new symbol everywhere, and record the rule.
4. Repeat — the vocabulary grows by one token per merge, always for the pair that pays off most.

You end up with a **subword vocabulary**: frequent sequences become whole
tokens (`ने`, `पाल</w>`), while rare strings stay split into small pieces.
This is the same algorithm used (in variant form) inside GPT, LLaMA, and
most modern language models.

## 2. Why BPE is useful for Nepali

Like many languages, Nepali has **no spaces inside words** — but unlike
English it has **very productive morphology**:

- नेपाल → नेपाली, नेपालको, नेपालमा, नेपालले, नेपालीहरू, नेपालीभाषी…
- हिमाल → हिमालहरू, हिमाली, हिमालय, हिमालीभेग…

A word-level tokenizer explodes because the number of inflected forms is
huge. A character-level tokenizer is small but slow to learn from (long
sequences). BPE sits in the middle:

- Frequent whole words (`नेपाल</w>`) become single tokens.
- Common pieces (`को`, `मा`, `हरू`, `ले`) become reusable tokens.
- Anything unseen is still encodable as smaller subword pieces.

So BPE gives a **fixed vocabulary** that can represent any Nepali text
(with a reasonable unknown-symbol fallback), while staying **compact and
fast** for training — exactly what a language model needs.

## 3. How the algorithm works

Training loop (see `learn_bpe` in `src/npltokenizer/tokenizer.py`):

```text
count pairs   get_pair_counts(): every adjacent symbol pair, weighted by word frequency
   ↓
find best     the most frequent pair (deterministic tie-break: first seen wins)
   ↓
merge all     merge_pair(): fuse that pair everywhere it appears in every word
   ↓
save rule     append the pair to self.merges (ORDER MATTERS)
   ↓
repeat
```

Example. Start with the word `नेपाल` represented as its code points plus a
word-end marker:

```text
न े प ा ल </w>
```

which produces the adjacent pairs

```text
(न, े)  (े, प)  (प, ा)  (ा, ल)  (ल, </w>)
```

After learning, if `(न, े)` is the most frequent pair, every occurrence
becomes one symbol:

```text
ने प ा ल </w>
```

Encoding applies every learned rule in **the same order they were
learned**, so encoding and training never disagree about what merges.

## 4. Why `</w>` is used

`</w>` is an explicit **word-end marker**. Real words are stored as

```python
symbols + ["</w>"]
```

This lets BPE learn *awareness of word boundaries*:

- यहाँ `ल` at a word end is a different *context* from `ल` in the middle.
- `नेपाल</w>` and `पालती` do not merge the same way, so the model can tell
  "this subword ends the word" (`नेपाल</w>`) from "this subword continues" (`पालत`).

Without `</w>`, a token like `पाल` would be ambiguous — decoder output
would lose the boundary between words. With it, the decoder strips
`</w>` cleanly and the original text is reconstructed.

Whitespace is treated as **literal tokens** rather than synthesized
spaces, so the round trip is exact even for repeated whitespace:

```python
encode("नेपाल   काठमाडौँ") → [ने, पाल</w>, " " , " ", " ", काठमा, डौँ</w>]
decode(...)                → नेपाल   काठमाडौँ
```

## 5. How Unicode normalization affects Nepali

Nepali is written in **Devanagari**, and Devanagari is a complex script:

- `ने` = `न` (U+0928) + `े` (U+0947) — **two code points, one visual syllable**.
- `क्ष` = `क` + `्` + `ष` — three code points that render as one glyph.
- `काठमाडौँ` ends in `औ` + `ँ` (chandrabindu).

The tokenizer NFC-normalizes all text first
(`unicodedata.normalize("NFC", text)`), so a canonically-decomposed
spelling cannot produce a second, different tokenization. For Devanagari
NFC is mostly an identity operation (Nepali has few canonical
decompositions), but it is still essential for robustness.

**Important:** this implementation does *not* claim that code points are
visual characters. It documents three symbolization layers:

| Layer | Symbol = | Notes |
|---|---|---|
| **code-point BPE** *(this repo)* | individual Unicode code points | simplest to understand; `ने` is 2 symbols until a merge fuses it |
| **grapheme-aware BPE** | extended grapheme clusters (`\X`) | `ने` stays one symbol forever; closer to human reading, needs a grapheme-segmentation library |
| **byte-level BPE** | UTF-8 bytes | every text representable, no unknowns ever; used by GPT-2/3 |

The whole design funnels symbolization through **one method,
`symbolize_word`**, so you can swap layers without touching the BPE core.

## 6. How training works

`train.py`:

1. Reads `corpus.txt` (UTF-8), folds CRLF → LF, NFC-normalizes.
2. Pre-tokenizes into word runs and whitespace runs and counts them.
3. `set_training_words` builds word→symbols and the sorted **base vocabulary**
   (every distinct code point plus `</w>`).
4. `learn_bpe(N)` learns N merges, showing progress:

```text
Corpus: corpus.txt
Unique words: 12450
Initial vocabulary: 183
Special tokens: 4 (<PAD>, <UNK>, <BOS>, <EOS>)
Target vocabulary: 2048  (1861 merges requested)

Learning BPE merges ...
  ... 186 merge rules learned (last: 'न', frequency 4123)
  ...

Merge 1/200:            <- with --verbose
    ('न', 'े') → 'ने'
    frequency: 12453
```

5. Builds deterministic IDs: special tokens first, then base symbols
   (sorted), then merged symbols in learned order.
6. Saves to `tokenizer.json`.

## 7. How encoding works

```python
def encode(text):
    text        = normalize(text)          # NFC
    pieces      = split words + whitespace
    for each piece:
        symbols = symbolize(piece) + ["</w>"] if it is a word
        apply every merge rule in learned order
    map symbols → IDs, unknown symbols → <UNK>
```

A word never seen in training still gets a sensible encoding, because the
merge rules generalize: unknown words are decomposed into the best
available subword pieces.

## 8. How decoding works

```python
def decode(ids):
    token  = id_to_token[id]
    token.endswith("</w>")  → strip the marker (word boundary)
    whitespace token        → its literal characters
    <UNK>                   → U+FFFD replacement character
    <PAD>/<BOS>/<EOS>       → skipped by default
    join everything         → text
```

Because the original whitespace is stored as literal tokens (not
synthesized), the reconstructed text matches the input **exactly** —
this is what the round-trip assertions in the test suite verify.

## 9. Limitations of this first implementation

- **Code points ≠ graphemes.** A visual syllable such as `ने` is
  initially two symbols; it is fused by a merge only if it occurs
  frequently. Rare syllables stay split. Design is reflected in the
  swappable `symbolize_word` layer, but the shipped layer is code points.
- **No byte fallback.** Symbols not in the training data decode as
  `<UNK>` / U+FFFD. A production tokenizer would add a byte-level
  fallback underneath so every string is encodable.
- **Speed.** Pair counting recomputes from scratch on every merge
  (O(merges × corpus)). Fine for learning a small tokenizer, but a
  production trainer would use a priority queue and incremental stats.
- **Order-sensitivity.** BPE is greedy; a different corpus or a different
  tie-break produces a different (usually similar-quality) vocabulary.
- **Whitespace policy.** Whitespace is preserved byte-for-byte. Some
  production tokenizers deliberately collapse repeats; ours does not,
  because lossless round trips are pedagogically valuable.
- **No cache / no on-the-fly O(n·v) prune.** Encoding naively tries every
  merge rule on every word.

## 10. Road map: from here to a GPT-style Nepali model tokenizer

1. **Byte-level fallback** in `symbolize_word`: pair UTF-8 bytes so that
   truly unknown input is still tokenizable (no `<UNK>`).
2. **Production pre-tokenization** (a deliberate regex, e.g. numbers,
   whitespace runs, punctuation classes) instead of the `/S+/|\s+/` split.
3. **Faster learning**: incremental pair statistics + a priority queue,
   and a vocabulary-pruning step that drops merge rules that only fire on
   single words.
4. **Deterministic special-token strategy**: reserve IDs for
   `<|endoftext|>`, `<|startoftext|>`, etc., matching the model's training
   corpus format.
5. **Vocab-aware training**: train the tokenizer on a large Nepali corpus
   (the `</w>` markers already teach it word boundaries), then fine-tune or
   pre-train a transformer on tokenized text.
6. **Metal-layer abstraction**: keep `encode`/`decode` stable while the
   inner layers change, so swapping in SentencePiece-style byte trie
   encoding is invisible to the model code.

The BPE core here is the same family of algorithm used by GPT; everything
else (byte fallback, pre-tokenization, performance) is incremental
engineering on top of it.

## Layout

```text
.
├── corpus.txt                  # small Nepali corpus (UTF-8)
├── train.py                    # thin launcher -> npltokenizer.train
├── test_tokenizer.py           # unit tests + evaluation
├── tokenizer.json              # generated when training
├── requirements.txt            # standard library only
├── pyproject.toml              # package metadata / script entry points
└── src/npltokenizer/
    ├── __init__.py             # package exports (main entry prints help)
    ├── tokenizer.py            # the BPE implementation (heart of the repo)
    └── train.py                # training CLI with progress + demo
```