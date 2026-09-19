# -*- coding: utf-8 -*-
"""
NepaliBPETokenizer
==================

A small, educational Byte Pair Encoding (BPE) tokenizer for the Nepali
language (Devanagari script), implemented from scratch using ONLY the
Python standard library.

No Hugging Face ``tokenizers``, no SentencePiece, no third-party code.

The pipeline that this module implements is:

    Nepali corpus
      -> Unicode normalization (NFC)
      -> pre-tokenization into words (and whitespace runs)
      -> symbolization (one Unicode code point per symbol, by default)
      -> pair-frequency counting
      -> BPE merge learning  (get_pair_counts -> merge_pair -> learn_bpe)
      -> a learned vocabulary (token_to_id / id_to_token)
      -> encode(nepali_text)  -> token IDs
      -> decode(token_ids)    -> nepali text

Why code points and not graphemes/bytes?
----------------------------------------
Nepali uses the Devanagari script. A *visual* Devanagari character (a
syllable such as ने) is usually composed of several Unicode code points:
    न  +  े          (consonant U+0928 + vowel sign U+0947)
So an individual code point is NOT a visual character.  The first
implementation in this file therefore works on code points on purpose:
it keeps the BPE algorithm easy to understand.  All symbolization happens
in ``symbolize_word``; you can swap that single method to get a
grapheme-aware layer (e.g. ``\\X`` clusters) or a byte-level layer
(UTF-8 bytes) without touching the BPE core.  See the README.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Marker appended to the end of every real (non-whitespace) word.
#: ``</w>`` stands for "end of word".  It tells the tokenizer where a word
#: finishes so that a subword token that ends a word (e.g. ``पाल</w>``)
#: is distinguished from the same subword in the middle of a word.
WORD_END = "</w>"

#: Reserved tokens.  IDs are deterministic and fixed:
#:   <PAD> = 0, <UNK> = 1, <BOS> = 2, <EOS> = 3
DEFAULT_SPECIAL_TOKENS: Tuple[str, ...] = ("<PAD>", "<UNK>", "<BOS>", "<EOS>")
UNK_TOKEN = "<UNK>"

#: Pre-tokenizer.  Splits text into maximal runs of non-whitespace
#: characters ("words" such as नेपाल or हो।) and maximal runs of
#: whitespace ("नेपाल काठमाडौँ" -> ["नेपाल", " ", "काठमाडौँ"]).
#: Keeping the whitespace runs as literal tokens (instead of deleting
#: them) lets encode/decode round-trip reproduce the original text,
#: including repeated spaces, tabs and newlines, exactly.
_PRETOKEN_RE = re.compile(r"\S+|\s+")


# ---------------------------------------------------------------------------
# Low-level helper: count adjacent symbol pairs
# ---------------------------------------------------------------------------
def count_pairs(splits: Dict[str, List[str]], frequencies: Dict[str, int]) -> Counter:
    """
    Count adjacent symbol pairs across a "vocabulary".

    ``splits`` maps every word (string) to its current symbol list, and
    ``frequencies`` maps every word to how often it occurs in the corpus.

    Each pair is counted once per *occurrence*, so a pair inside a word
    that occurs N times contributes N to the pair's frequency.  This
    weighting by word frequency is what makes BPE merge the pairs that
    help the most.

    Example word ``नेपाल`` (symbols ``["न", "े", "प", "ा", "ल", "</w>"]``)
    contributes one count to each of:
        (न, े), (े, प), (प, ा), (ा, ल), (ल, </w>)
    checking the first index0/1 box: yes, those five adjacent pairs.
    """
    pair_counts: Counter = Counter()
    for word, symbols in splits.items():
        word_frequency = frequencies[word]
        # Walk over every slice of two neighbouring symbols.
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_counts[pair] += word_frequency
    return pair_counts


# ---------------------------------------------------------------------------
# The tokenizer
# ---------------------------------------------------------------------------
class NepaliBPETokenizer:
    """
    A from-scratch BPE tokenizer for Nepali (Devanagari) text.

    Two stages of life:

    1. TRAINING:
         tok = NepaliBPETokenizer()
         tok.set_training_words(word_frequencies)   # build word -> symbols
         tok.learn_bpe(num_merges)                  # learn merge rules
         tok.build_vocab()                          # build id maps
       (or use the shortcut classmethod ``NepaliBPETokenizer.train``)

    2. USING:
         ids = tok.encode("नेपाल सुन्दर देश हो।")   # -> list[int]
         text = tok.decode(ids)                      # -> "नेपाल सुन्दर देश हो।"
         tok.save("tokenizer.json")                  # persist
         tok2 = NepaliBPETokenizer.load("tokenizer.json")
    """

    def __init__(
        self,
        *,
        special_tokens: Optional[Sequence[str]] = None,
        base_tokens: Optional[Sequence[str]] = None,
        merges: Optional[Sequence[Sequence[str]]] = None,
        symbol_mode: str = "unicode",
    ) -> None:
        # Special (reserved) tokens always get the first IDs in fixed order.
        self.special_tokens: List[str] = list(
            special_tokens if special_tokens is not None else DEFAULT_SPECIAL_TOKENS
        )
        # The base vocabulary: every distinct symbol seen in the corpus,
        # e.g. न, े, ा, ँ, </w>, " ", "0", "b", "।" ... (kept sorted so the
        # IDs are deterministic regardless of corpus reading order).
        self.base_tokens: List[str] = list(base_tokens or [])

        # Merge rules in the *order they were learned*.  Encoding MUST
        # apply them in this exact order because every merge was learned
        # on top of the results of the previous ones.
        self.merges: List[Tuple[str, str]] = [tuple(m) for m in merges] if merges else []
        # The new symbols produced by each merge (''.join of the pair).
        # These join ``base_tokens`` to form the full vocabulary.
        self.merged_tokens: List[str] = [a + b for a, b in self.merges]

        # Symbolization layer. ``unicode`` = one Unicode code point per
        # symbol. Future layers: ``grapheme``, ``byte``.
        self.symbol_mode: str = symbol_mode

        # ---- training-state attributes --------------------------------------
        # word -> frequency (only used while learning merges)
        self.vocab: Dict[str, int] = {}
        # word -> current list of symbols (updated as merges are applied)
        self.splits: Dict[str, List[str]] = {}

        # ---- id maps ---------------------------------------------------------
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.build_vocab()

    # ------------------------------------------------------------------
    # Unicode normalization
    # ------------------------------------------------------------------
    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize text to Unicode NFC.

        NFC composes characters that have a canonical decomposition.
        For Devanagari this is mostly a no-op (ने stays न + े), but it
        guarantees that text which entered the system in a decomposed
        form (for example ``a + U+0301``) is folded into its canonical
        composed form before tokenization, so one spelling can never
        receive two different tokenizations.

        NOTE: NFC is NOT the same as grapheme clustering.  A visual
        Devanagari syllable may still be several code points even after
        NFC.  That is exactly why the default symbol layer is documented
        as "code points", not "characters".
        """
        return unicodedata.normalize("NFC", text)

    # ------------------------------------------------------------------
    # Symbolization layer (the ONE place to swap strategies later)
    # ------------------------------------------------------------------
    def symbolize_word(self, word: str) -> List[str]:
        """
        Turn a word string into a list of atomic "symbols".

        This is the symbolization LAYER.  The whole BPE machinery above
        it only ever sees these little strings, so this single method
        defines what a symbol is:

        * ``unicode``   -> list(word)                 (current default)
        * ``grapheme``  -> re.findall(r"\\X", word)   (needs regex module)
        * ``byte``      -> [hex byte] for word.encode("utf-8")

        The current layer uses Unicode code points on purpose: it is the
        simplest thing that lets the BPE algorithm (pair counting, merge
        application) be studied without distractions.

        Remember: a code point is not a visual Devanagari character.
        ने is TWO code points; क्ष is THREE (क + ् + ष).  Multi-code-point
        pieces like that are recombined later by BPE merges when they are
        frequent, so the tokenizer behaves correctly even though its
        "atoms" are finer than visual syllables.
        """
        return list(word)

    # ------------------------------------------------------------------
    # Pre-tokenization and corpus word counting
    # ------------------------------------------------------------------
    def _pretokenize(self, text: str) -> List[str]:
        """Split text into alternating word runs and whitespace runs."""
        return _PRETOKEN_RE.findall(text)

    def count_words(self, text: str) -> Counter:
        """
        Normalize text and count every pre-token piece.

        Both real words (नेपाल) and whitespace runs ("  ", "\n") are
        counted: the whitespace pieces also end up in the base
        vocabulary (as the " ", "\t", "\n" symbols) so that they can be
        encoded and decoded losslessly later.
        """
        normalized = self.normalize(text)
        return Counter(self._pretokenize(normalized))

    # ------------------------------------------------------------------
    # Vocabulary initialization ("word -> symbols")
    # ------------------------------------------------------------------
    def _word_symbols(self, word: str) -> List[str]:
        """
        Initial symbol list of one word.

        * real word      -> symbols + [WORD_END]
        * whitespace run -> symbols only

        The ``</w>`` end marker allows the tokenizer to know where a word
        finishes.  Merges therefore learn word-end-aware subwords such as
        ``ल</w>``.  Without the marker, ``पाल`` and ``पाली`` would merge
        identically and decoding could not re-insert the word boundary.
        """
        symbols = self.symbolize_word(word)
        if word.isspace():
            # Whitespace runs are NOT words: they need no end marker.
            return symbols
        return symbols + [WORD_END]

    def set_training_words(self, word_counts: Dict[str, int]) -> None:
        """
        Initialize the training vocabulary from a word->frequency dict.

        Every word is expanded into its initial symbol list and the base
        vocabulary is set to the sorted set of all distinct symbols
        (including ``</w>``).
        """
        # Keep insertion order; it makes pair counting (and therefore
        # tie-breaking between equally frequent pairs) deterministic.
        self.vocab = dict(word_counts)
        self.splits = {}
        all_symbols = set()
        for word in self.vocab:
            symbols = self._word_symbols(word)
            self.splits[word] = symbols
            all_symbols.update(symbols)
        self.base_tokens = sorted(all_symbols)
        # base_tokens changed -> id maps must be rebuilt.
        self.build_vocab()

    # ------------------------------------------------------------------
    # Pair counting, merging and BPE learning (the heart of the algorithm)
    # ------------------------------------------------------------------
    def get_pair_counts(self) -> Counter:
        """
        Count every adjacent symbol pair in the current training state,
        weighted by word frequency.
        """
        return count_pairs(self.splits, self.vocab)

    @staticmethod
    def merge_pair(pair: Tuple[str, str], symbols: Sequence[str]) -> List[str]:
        """
        Replace every occurrence of ``pair`` inside ``symbols`` with a
        single merged symbol ``left + right``.

        The scan is greedy, left to right, and non-overlapping:
            merge_pair(("a", "a"), ["a", "a", "a", "b"])
                -> ["aa", "a", "b"]          (first pair merged, middle 'a' left)
        If the pair does not occur, the ORIGINAL list object is returned
        unchanged so callers can cheaply detect "nothing changed".
        """
        left, right = pair
        n = len(symbols)
        result: List[str] = []
        merged_any = False
        i = 0
        while i < n:
            if i + 1 < n and symbols[i] == left and symbols[i + 1] == right:
                # Found the pair: fuse the two symbols into one.
                result.append(left + right)
                i += 2
                merged_any = True
            else:
                result.append(symbols[i])
                i += 1
        if not merged_any:
            # Nothing changed: return the very same list object so that
            # callers can use ``is`` to detect "no change" for free.
            return symbols
        return result

    def learn_bpe(self, num_merges: int, verbose: bool = False,
                  progress_every: Optional[int] = None) -> int:
        """
        Learn ``num_merges`` BPE merge rules.

        At every iteration:
            count pairs
            -> find the single most frequent pair (weighted by word freq)
            -> merge that pair everywhere it occurs in the training data
            -> save the merge rule (in learned order)
            -> repeat

        Merge rules are appended to ``self.merges`` in the order they are
        learned; encoding applies them in that same order.

        Returns the number of merges actually performed (fewer than
        requested only if no pairs remain, i.e. every word collapsed to
        a single symbol).
        """
        if num_merges <= 0:
            return 0
        if progress_every is None:
            progress_every = max(1, num_merges // 10)

        performed = 0
        for step in range(1, num_merges + 1):
            # 1) count pairs in the whole training vocabulary
            pair_counts = self.get_pair_counts()
            if not pair_counts:
                break

            # 2) find the most frequent pair.
            #    We scan manually (instead of Counter.most_common) so the
            #    tie-break is explicit: among equally frequent pairs the
            #    first one to be discovered wins.  That is deterministic.
            best_pair: Optional[Tuple[str, str]] = None
            best_frequency = -1
            for pair, frequency in pair_counts.items():
                if frequency > best_frequency:
                    best_frequency = frequency
                    best_pair = pair
            assert best_pair is not None

            # 3) apply the merge to every word that contains it
            for word in list(self.splits):
                symbols = self.splits[word]
                merged = self.merge_pair(best_pair, symbols)
                if merged is not symbols:  # this word actually contained the pair
                    self.splits[word] = merged

            # 4) save the rule, order matters for encoding
            self.merges.append(best_pair)
            self.merged_tokens.append(best_pair[0] + best_pair[1])
            performed += 1

            if verbose:
                print(f"\nMerge {performed}/{num_merges}:")
                print(f"    {best_pair} -> '{best_pair[0] + best_pair[1]}'")
                print(f"    frequency: {best_frequency}")
            elif progress_every and (
                performed == num_merges or performed % progress_every == 0
            ):
                merged_symbol = best_pair[0] + best_pair[1]
                print(f"  ... {performed} merge rules learned "
                      f"(last: '{merged_symbol}', frequency {best_frequency})")

        return performed

    # ------------------------------------------------------------------
    # Vocabulary / id maps
    # ------------------------------------------------------------------
    def build_vocab(self) -> None:
        """
        Build ``token_to_id`` and ``id_to_token``.

        Ordering (deterministic):
            0..k-1  special tokens     (<PAD> <UNK> <BOS> <EOS>)
            then    base symbols       (sorted code points of the corpus)
            then    merged symbols     (in the order the merges were learned)
        """
        tokens: List[str] = []
        seen = set()
        for token in self.special_tokens:
            if token not in seen:
                seen.add(token)
                tokens.append(token)
        for token in self.base_tokens:
            if token not in seen:
                seen.add(token)
                tokens.append(token)
        for token in self.merged_tokens:
            if token not in seen:
                seen.add(token)
                tokens.append(token)

        self.token_to_id = {token: i for i, token in enumerate(tokens)}
        self.id_to_token = {i: token for i, token in enumerate(tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"NepaliBPETokenizer(vocab_size={self.vocab_size}, "
            f"merges={len(self.merges)}, symbol_mode='{self.symbol_mode}')"
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode_word(self, piece: str) -> List[str]:
        """
        Turn one pre-tokenized piece into subword symbols.

            1. symbolize the piece (code points)
            2. append </w> if it is a real (non-whitespace) word
            3. apply every learned BPE merge, in learned order
        """
        symbols = self._word_symbols(piece)
        for pair in self.merges:
            symbols = self.merge_pair(pair, symbols)
        return symbols

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        """
        Encode Nepali text into a list of token IDs.

            text -> normalize -> split words/whitespace
                 -> symbolize each piece -> apply merges
                 -> map symbols to IDs (<UNK> for anything unseen)
        """
        unk_id = self.token_to_id[UNK_TOKEN]
        ids: List[int] = []
        if bos:
            ids.append(self.token_to_id["<BOS>"])
        for piece in self._pretokenize(self.normalize(text)):
            for token in self.encode_word(piece):
                ids.append(self.token_to_id.get(token, unk_id))
        if eos:
            ids.append(self.token_to_id["<EOS>"])
        return ids

    def tokenize(self, text: str) -> List[str]:
        """Convenience: encode ``text`` and return the token strings."""
        return [self.id_to_token[i] for i in self.encode(text)]

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True,
               unk_replacement: str = "\ufffd") -> str:
        """
        Rebuild text from token IDs.

        Reconstruction rules:
            * a token ending in ``</w>``  -> its text with ``</w>`` removed
            * a lone ``</w>``             -> nothing
            * a whitespace token (" ", "  ") -> its literal characters
            * ``<UNK>``                   -> the U+FFFD replacement character
            * ``<PAD>/<BOS>/<EOS>``       -> skipped by default
            * unknown / out-of-range ID   -> the replacement character

        Because the original whitespace between words is stored as plain
        literal tokens (not synthesized from ``</w>``), joining the pieces
        reproduces the input text exactly, including repeated spaces.
        """
        parts: List[str] = []
        for raw_id in ids:
            token = self.id_to_token.get(raw_id)
            if token is None:
                parts.append(unk_replacement)
                continue
            if token in self.special_tokens:
                if token == UNK_TOKEN:
                    parts.append(unk_replacement)
                elif not skip_special_tokens:
                    parts.append(token)
                # PAD / BOS / EOS contribute nothing to the surface text.
                continue
            if token.endswith(WORD_END):
                # Subword that ends a word: drop the word-end marker only.
                parts.append(token[: -len(WORD_END)])
            else:
                parts.append(token)
        return "".join(parts)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "type": "NepaliBPETokenizer",
            "version": 1,
            "symbol_mode": self.symbol_mode,
            "special_tokens": self.special_tokens,
            "base_tokens": self.base_tokens,
            "merges": [[a, b] for a, b in self.merges],
        }

    def save(self, path: str) -> None:
        """Persist the tokenizer to a JSON file."""
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "NepaliBPETokenizer":
        """Reconstruct a tokenizer from a JSON file written by ``save``."""
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(
            special_tokens=data.get("special_tokens"),
            base_tokens=data.get("base_tokens"),
            merges=data.get("merges", []),
            symbol_mode=data.get("symbol_mode", "unicode"),
        )

    # ------------------------------------------------------------------
    # Training shortcut
    # ------------------------------------------------------------------
    @classmethod
    def train(cls, corpus_text: str, num_merges: int, verbose: bool = False) -> "NepaliBPETokenizer":
        """
        Full training pipeline in one call:

            count words -> initialize vocabulary -> learn BPE merges
            -> build id maps
        """
        tokenizer = cls()
        word_counts = tokenizer.count_words(corpus_text)
        tokenizer.set_training_words(word_counts)
        tokenizer.learn_bpe(num_merges, verbose=verbose)
        tokenizer.build_vocab()
        return tokenizer


__all__ = ["NepaliBPETokenizer", "count_pairs", "WORD_END", "UNK_TOKEN",
           "DEFAULT_SPECIAL_TOKENS"]