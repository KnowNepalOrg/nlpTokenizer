# -*- coding: utf-8 -*-
"""A from-scratch Byte Pair Encoding (BPE) tokenizer for Nepali (Devanagari)."""

from npltokenizer.tokenizer import (
    DEFAULT_SPECIAL_TOKENS,
    UNK_TOKEN,
    WORD_END,
    NepaliBPETokenizer,
    count_pairs,
)

__version__ = "0.1.0"

__all__ = [
    "NepaliBPETokenizer",
    "count_pairs",
    "WORD_END",
    "UNK_TOKEN",
    "DEFAULT_SPECIAL_TOKENS",
    "__version__",
]


def main() -> None:
    """Console entry point that points the user at the real scripts."""
    print("npltokenizer - a from-scratch Nepali BPE tokenizer")
    print()
    print("  python train.py            # train a tokenizer on corpus.txt")
    print("  python test_tokenizer.py   # run tests and evaluation")
    print()
    print("  nplt-train                 # same training, installed as a script")