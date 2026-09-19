# -*- coding: utf-8 -*-
"""
Training driver for NepaliBPETokenizer.

Runnable three ways (all equivalent):

    python train.py                       # from the repo root
    python -m npltokenizer.train
    nplt-train                            # after ``pip install .`` / ``uv sync``

Flags:

    --corpus PATH        corpus file (default: corpus.txt)
    --merges N           learn exactly N merges
    --target-vocab N     instead, grow until the vocabulary reaches ~N tokens
    --max-words N        cap the number of unique training words
    --verbose            print EVERY merge (can be many lines)
    --output PATH        where to save tokenizer.json (default: tokenizer.json)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

from npltokenizer.tokenizer import NepaliBPETokenizer

# The four reserved tokens are always present, so they do not count
# towards the merge budget.
DEFAULT_TARGET_VOCAB = 2048

PIPELINE = (
    "Training\n"
    "   |\n"
    "   v\n"
    "Learned merges\n"
    "   |\n"
    "   v\n"
    "Vocabulary\n"
    "   |\n"
    "   v\n"
    "Encoding\n"
    "   |\n"
    "   v\n"
    "Token IDs\n"
    "   |\n"
    "   v\n"
    "Decoding\n"
    "   |\n"
    "   v\n"
    "Original Nepali text"
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a from-scratch BPE tokenizer on a Nepali corpus."
    )
    parser.add_argument("--corpus", default="corpus.txt",
                        help="UTF-8 Nepali corpus file (default: corpus.txt)")
    parser.add_argument("--output", "-o", default="tokenizer.json",
                        help="where to write the trained tokenizer (default: tokenizer.json)")
    parser.add_argument("--max-words", type=int, default=None,
                        help="limit training to the N most frequent unique words")
    parser.add_argument("--verbose", action="store_true",
                        help="print every individual merge rule (many lines)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--merges", type=int, default=None,
                       help="number of BPE merges to learn")
    group.add_argument("--target-vocab", type=int, default=DEFAULT_TARGET_VOCAB,
                       help="grow vocabulary until ~N tokens "
                            "(default %(default)s; merges = N - specials - base)")
    return parser


def load_corpus(path: str) -> str:
    """
    Read a UTF-8 Nepali corpus.

    * Unicode is NFC-normalized (idempotent with the tokenizer's own
      normalization, but makes the byte->text load explicit here too).
    * Windows CRLF and stray carriage returns are folded to ``\\n`` so
      that line endings never become two different whitespace tokens.
    """
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return NepaliBPETokenizer.normalize(text)


def compute_merge_budget(tokenizer: NepaliBPETokenizer,
                         requested_merges: Optional[int],
                         target_vocab: int) -> int:
    """Decide how many merges to learn from the CLI flags."""
    if requested_merges is not None:
        if requested_merges < 0:
            raise ValueError("--merges must be >= 0")
        return requested_merges

    # merges = target vocab - reserved tokens - base symbols already present
    budget = target_vocab - len(tokenizer.special_tokens) - len(tokenizer.base_tokens)
    if budget <= 0:
        print(f"warning: target vocabulary {target_vocab} is already reached by "
              f"{len(tokenizer.special_tokens)} special + {len(tokenizer.base_tokens)} "
              f"base tokens; learning 0 merges.", file=sys.stderr)
        return 0
    return budget


def demo_round_trip(tokenizer: NepaliBPETokenizer) -> None:
    """Small end-to-end demonstration used at the very end of training."""
    sample = "नेपाल एउटा सुन्दर देश हो।"
    user_text_style = "input"
    ids = tokenizer.encode(sample)
    tokens = tokenizer.tokenize(sample)
    decoded = tokenizer.decode(ids)
    ok = decoded == sample

    print("\nEnd-to-end demonstration")
    print("=" * 62)
    print(f"input text : {sample}")
    print(f"tokens     : {tokens}")
    print(f"ids        : {ids}")
    print(f"decoded    : {decoded}")
    print(f"round-trip : {'OK' if ok else 'MISMATCH (see above)'}")

    if ok:
        print()
        print(PIPELINE)
        print(f"'{user_text_style} नेपाल एउटा सुन्दर देश हो।' -> "
              f"{len(tokenizer.encode(sample))} token ids -> "
              f"'{decoded}'")


def main(argv: Optional[List[str]] = None) -> int:
    # Make sure Devanagari prints cleanly even on legacy Windows consoles.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    args = build_argument_parser().parse_args(argv)

    if not os.path.exists(args.corpus):
        print(f"error: corpus file not found: {args.corpus}", file=sys.stderr)
        print(f"  hint: run this from the project root, or use --corpus PATH.",
              file=sys.stderr)
        return 1

    started = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Corpus -> words
    # ------------------------------------------------------------------
    print(f"Reading corpus: {args.corpus}")
    corpus_text = load_corpus(args.corpus)
    print(f"Corpus characters (NFC-normalized): {len(corpus_text)}")

    tokenizer = NepaliBPETokenizer()
    word_counts = tokenizer.count_words(corpus_text)
    if args.max_words:
        word_counts = word_counts.most_common(args.max_words)
        word_counts = dict(word_counts)

    # ------------------------------------------------------------------
    # 2. Vocabulary initialization
    # ------------------------------------------------------------------
    tokenizer.set_training_words(dict(word_counts))
    unique_words = len(tokenizer.vocab)
    base_vocab_size = len(tokenizer.base_tokens)
    num_merges = compute_merge_budget(tokenizer, args.merges, args.target_vocab)

    print()
    print(f"Corpus: {os.path.basename(args.corpus)}")
    print(f"Unique words: {unique_words}")
    print(f"Initial vocabulary: {base_vocab_size}")
    print(f"Special tokens: {len(tokenizer.special_tokens)} "
          f"({', '.join(tokenizer.special_tokens)})")
    print(f"Target vocabulary: {len(tokenizer.special_tokens) + base_vocab_size + num_merges}"
          f"  ({num_merges} merges requested)")

    # ------------------------------------------------------------------
    # 3/4. BPE pair counting + merging
    # ------------------------------------------------------------------
    print()
    print("Learning BPE merges (count pair -> merge most frequent -> repeat)...")
    performed = tokenizer.learn_bpe(num_merges, verbose=args.verbose)
    tokenizer.build_vocab()
    elapsed = time.perf_counter() - started

    print(f"\nLearned merges: {performed}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Training wall time: {elapsed:.1f}s")
    print(f"Merges per second: {performed / elapsed:.0f}")

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    tokenizer.save(args.output)
    print(f"Saved tokenizer  -> {args.output}")

    # ------------------------------------------------------------------
    # 6. Demonstrate encoding -> ids -> decoding
    # ------------------------------------------------------------------
    demo_round_trip(tokenizer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())