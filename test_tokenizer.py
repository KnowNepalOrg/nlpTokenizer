# -*- coding: utf-8 -*-
"""
Unit tests + evaluation for NepaliBPETokenizer.

Run:

    python test_tokenizer.py

(No third-party test framework required; the functions are plain
``test_*`` asserts so ``pytest`` also picks them up if installed.)

The evaluation section reports, on the shipped ``corpus.txt`` tokenizer:

    * number of unique words
    * vocabulary size
    * number of learned merges
    * tokens per sentence
    * tokens per word
    * unknown-token rate
    * encode/decode round-trip accuracy
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import tempfile  # noqa: E402

from npltokenizer import NepaliBPETokenizer  # noqa: E402
from npltokenizer.tokenizer import WORD_END  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(REPO_ROOT, "corpus.txt")

# Shared tokenizer: trained ONCE on the real corpus and reused by every test.
SHARED_MERGES = 800
_SHARED_TOKENIZER = None


def shared_tokenizer() -> NepaliBPETokenizer:
    """Lazily train one tokenizer on corpus.txt and cache it."""
    global _SHARED_TOKENIZER
    if _SHARED_TOKENIZER is None:
        with open(CORPUS_PATH, encoding="utf-8") as handle:
            corpus_text = handle.read()
        _SHARED_TOKENIZER = NepaliBPETokenizer.train(corpus_text, SHARED_MERGES)
    return _SHARED_TOKENIZER


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
def test_special_tokens_have_fixed_deterministic_ids() -> None:
    tok = shared_tokenizer()
    for name, expected_id in [("<PAD>", 0), ("<UNK>", 1), ("<BOS>", 2), ("<EOS>", 3)]:
        assert tok.token_to_id[name] == expected_id, name


def test_nfc_normalization() -> None:
    tok = NepaliBPETokenizer()
    # NFC composes a base letter + combining acute accent into "é".
    assert tok.normalize("e\u0301") == "\u00e9"
    assert tok.normalize("\u00e9") == "\u00e9"
    # And therefore both spellings encode identically.
    assert tok.encode("e\u0301") == tok.encode("\u00e9")
    # Devanagari 'कि' has no precomposed form: NFC keeps क + ि as 2 code
    # points.  This is expected and is NOT a bug of the code-point layer.
    assert list(tok.normalize("कि")) == ["क", "ि"]


def test_get_pair_counts_weights_by_frequency() -> None:
    tok = NepaliBPETokenizer()
    tok.set_training_words({"नेपाल": 2, "नेपाली": 1, "हो": 3})
    counts = tok.get_pair_counts()
    # ('न','े') appears once in नेपाल(×2) and once in नेपाली(×1) -> 3.
    assert counts[("न", "े")] == 3
    # 'ल' followed by word-end appears in नेपाल(×2) + नेपाली(×1) -> 3.
    assert counts[("ल", WORD_END)] == 3
    assert counts[("ह", "ो")] == 3
    # Word-internal symbols are code points; the 'े' of 'हो' is 'ो'.
    assert counts[("ो", WORD_END)] == 3


def test_merge_pair() -> None:
    tok = NepaliBPETokenizer()
    symbols = ["न", "े", "प", "ा", "ल", WORD_END]
    merged = tok.merge_pair(("न", "े"), symbols)
    assert merged == ["ने", "प", "ा", "ल", WORD_END]
    assert merged is not symbols
    # Greedy, left-to-right, non-overlapping.
    assert tok.merge_pair(("a", "a"), ["a", "a", "a", "b"]) == ["aa", "a", "b"]
    # An unchanged pair returns the SAME object (cheap "did anything change").
    unchanged = ["x", "y"]
    assert tok.merge_pair(("a", "b"), unchanged) is unchanged


def test_learn_bpe_records_ordered_deterministic_merges() -> None:
    corpus = "नेपाल नेपाल नेपाल हो हो हो"
    a = NepaliBPETokenizer.train(corpus, 2)
    b = NepaliBPETokenizer.train(corpus, 2)
    # With all pair frequencies tied, the first-seen pair wins:
    assert a.merges[0] == ("न", "े")
    assert a.merges[1] == ("ने", "प")
    assert a.merges == b.merges
    assert a.token_to_id == b.token_to_id


def test_merged_tokens_appear_in_vocabulary() -> None:
    tok = shared_tokenizer()
    assert tok.merges  # some merges must have been learned
    for merged in tok.merged_tokens[:5]:
        assert merged in tok.token_to_id
    # The base symbols (code points) are all in the vocabulary too.
    for base in tok.base_tokens[:5]:
        assert base in tok.token_to_id


def assert_round_trips(text: str, tok: NepaliBPETokenizer) -> None:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text, f"round-trip failed for {text!r} -> {decoded!r}"


def test_round_trip_spec_example() -> None:
    text = "नेपाल सुन्दर देश हो।"
    tok = shared_tokenizer()
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_round_trip_spec_sentences() -> None:
    tok = shared_tokenizer()
    sentences = [
        "नेपाल एउटा सुन्दर देश हो।",
        "नेपाली भाषा नेपालको राष्ट्रिय भाषा हो।",
        "म नेपाली भाषा सिक्दै छु।",
        "काठमाडौँ नेपालको राजधानी हो।",
        "नेपालमा धेरै हिमालहरू छन्।",
    ]
    for sentence in sentences:
        assert_round_trips(sentence, tok)


def test_round_trip_punctuation_and_numbers() -> None:
    tok = shared_tokenizer()
    samples = [
        "खै, कहिले आउँछु? हुनसक्छ!",
        "सन् २०२४ मा ७७ जिल्लामा काम भयो।",
        "टिकटको मूल्य २५० रूपैयाँ र 3.14% छ।",
        "गणितमा १ + १ = २ हुन्छ।",
        "नेपाली संख्या १२३४५",
    ]
    for sample in samples:
        assert_round_trips(sample, tok)


def test_round_trip_english_mixed_with_nepali() -> None:
    tok = shared_tokenizer()
    samples = [
        "Nepali BPE tokenizer र GPT मोडेल बारे पढ्दै छु।",
        "Unicode ले Devanagari अक्षरलाई सहयोग गर्छ।",
        "The quick brown fox jumps over the lazy dog.",
        "Nepal is a beautiful country and I love it.",
    ]
    for sample in samples:
        assert_round_trips(sample, tok)


def test_round_trip_repeated_whitespace() -> None:
    tok = shared_tokenizer()
    samples = [
        "नेपाल   काठमाडौँ   पोखरा",
        "पोखरा र   काठमाडौँबीच हवाई सेवा छ।",
        "नेपाल\tकाठमाडौँ",
        "बाह्रै महिना   खेती   गरिन्छ  ।",
    ]
    for sample in samples:
        assert_round_trips(sample, tok)


def test_round_trip_uncommon_words() -> None:
    tok = shared_tokenizer()
    samples = [
        "भाषाविज्ञानको शब्दावलीमा स्वनिमको अभिलेख छ।",
        "दुर्लभ पाण्डुलिपिको सङ्कट अनुसन्धानमा छ।",
        "प्रज्ञा र अनुशासनले उन्नति हुन्छ।",
    ]
    for sample in samples:
        assert_round_trips(sample, tok)


def test_unknown_symbols_map_to_unk() -> None:
    tok = shared_tokenizer()
    unk_id = tok.token_to_id["<UNK>"]
    # The rocket emoji is NOT part of the corpus's base vocabulary.
    ids = tok.encode("नेपाल 🚀")
    assert unk_id in ids
    assert "नेपाल " in tok.decode(ids)
    assert "\ufffd" in tok.decode(ids)


def test_encode_is_deterministic() -> None:
    tok = shared_tokenizer()
    text = "काठमाडौँ नेपालको राजधानी हो।"
    assert tok.encode(text) == tok.encode(text)


def test_save_and_load_round_trip() -> None:
    tok = shared_tokenizer()
    with tempfile.TemporaryDirectory(prefix="nplt_bpe_") as directory:
        path = os.path.join(directory, "tokenizer.json")
        tok.save(path)

        restored = NepaliBPETokenizer.load(path)

        assert restored.token_to_id == tok.token_to_id
        assert restored.merges == tok.merges
        assert restored.special_tokens == tok.special_tokens
        assert restored.symbol_mode == tok.symbol_mode == "unicode"

        for text in ["नेपाल सुन्दर देश हो।",
                     "काठमाडौँ नेपालको राजधानी हो।"]:
            assert restored.encode(text) == tok.encode(text)
            assert restored.tokenize(text) == tok.tokenize(text)
            assert restored.decode(tok.encode(text)) == text


def test_whitespace_is_not_a_word() -> None:
    tok = NepaliBPETokenizer()
    tok.set_training_words({"नेपाल": 1, " ": 1, "  ": 1})
    assert WORD_END not in tok.splits[" "]
    assert WORD_END in tok.splits["नेपाल"]


def test_every_special_decode_is_lossless_in_practice() -> None:
    tok = shared_tokenizer()
    text = "नेपाल सुन्दर देश हो।"
    assert tok.decode(tok.encode(text, bos=True, eos=True)) == text


# ---------------------------------------------------------------------------
# Evaluation (also doubles as a smoke test that the trained artifact works)
# ---------------------------------------------------------------------------
EVAL_SENTENCES = [
    "नेपाल एउटा सुन्दर देश हो।",
    "नेपाली भाषा नेपालको राष्ट्रिय भाषा हो।",
    "म नेपाली भाषा सिक्दै छु।",
    "काठमाडौँ नेपालको राजधानी हो।",
    "नेपालमा धेरै हिमालहरू छन्।",
    "खै, कहिले आउँछु? हुनसक्छ!",
    "सन् २०२४ मा ७७ जिल्लामा काम भयो।",
    "टिकटको मूल्य २५० रूपैयाँ र 3.14% छ।",
    "गणितमा १ + १ = २ हुन्छ।",
    "The quick brown fox jumps over the lazy dog.",
    "Nepali BPE tokenizer र GPT मोडेल बारे पढ्दै छु।",
    "Unicode ले Devanagari अक्षरलाई सहयोग गर्छ।",
    "नेपाल   काठमाडौँ   पोखरा",
    "भाषाविज्ञानको शब्दावलीमा स्वनिमको अभिलेख छ।",
    "दुर्लभ पाण्डुलिपिको सङ्कट अनुसन्धानमा छ।",
]


def run_evaluation(tok: NepaliBPETokenizer) -> None:
    unk_id = tok.token_to_id["<UNK>"]
    total_ids = 0
    total_unks = 0
    total_words = 0
    n_sentences = len(EVAL_SENTENCES)
    n_exact = 0
    rows = []

    for sentence in EVAL_SENTENCES:
        ids = tok.encode(sentence)
        decoded = tok.decode(ids)
        words = [w for w in tok._pretokenize(tok.normalize(sentence)) if not w.isspace()]
        exact = decoded == sentence
        n_exact += int(exact)
        total_ids += len(ids)
        total_unks += ids.count(unk_id)
        total_words += len(words)
        rows.append((sentence, len(ids), exact, ids.count(unk_id)))

    tokens_per_sentence = total_ids / n_sentences
    tokens_per_word = total_ids / total_words
    unk_rate = total_unks / total_ids
    accuracy = n_exact / n_sentences

    print("Evaluation on evaluation sentences")
    print("-" * 72)
    print(f"{'sentence':<60} {'tokens':>7} {'RT':>4} {'UNK':>4}")
    print("-" * 72)
    for sentence, n_ids, exact, n_unk in rows:
        display = sentence if len(sentence) <= 58 else sentence[:57] + "…"
        print(f"{display:<60} {n_ids:>7} {'OK' if exact else 'FAIL':>4} {n_unk:>4}")
    print("-" * 72)

    print("\nMetric summary")
    print("-" * 72)
    unique_words = len(tok.vocab) if tok.vocab else _count_unique_words(tok)
    summary = [
        ("Unique words (training corpus)", f"{unique_words}"),
        ("Vocabulary size", f"{tok.vocab_size}"),
        ("Learned merges", f"{len(tok.merges)}"),
        ("Tokens per sentence", f"{tokens_per_sentence:.2f}"),
        ("Tokens per word", f"{tokens_per_word:.2f}"),
        ("Unknown-token rate", f"{unk_rate:.2%}"),
        ("Round-trip accuracy", f"{accuracy:.1%}"),
    ]
    for label, value in summary:
        print(f"  {label:<34} {value}")
    print("-" * 72)


def _count_unique_words(tok: NepaliBPETokenizer) -> int:
    """Count unique words directly from corpus when not in training state."""
    if tok.vocab:
        return len(tok.vocab)
    with open(CORPUS_PATH, encoding="utf-8") as handle:
        text = handle.read()
    return len(tok.count_words(text))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_all_tests() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    tests = [
        ("special token ids", test_special_tokens_have_fixed_deterministic_ids),
        ("NFC normalization", test_nfc_normalization),
        ("pair counting weights", test_get_pair_counts_weights_by_frequency),
        ("merge_pair", test_merge_pair),
        ("learn_bpe order/determinism", test_learn_bpe_records_ordered_deterministic_merges),
        ("merged tokens in vocab", test_merged_tokens_appear_in_vocabulary),
        ("round trip: spec example", test_round_trip_spec_example),
        ("round trip: spec sentences", test_round_trip_spec_sentences),
        ("round trip: punctuation/numbers", test_round_trip_punctuation_and_numbers),
        ("round trip: English mixed", test_round_trip_english_mixed_with_nepali),
        ("round trip: repeated whitespace", test_round_trip_repeated_whitespace),
        ("round trip: uncommon words", test_round_trip_uncommon_words),
        ("unknown -> <UNK>", test_unknown_symbols_map_to_unk),
        ("encode deterministic", test_encode_is_deterministic),
        ("save/load round trip", test_save_and_load_round_trip),
        ("whitespace not a word", test_whitespace_is_not_a_word),
        ("BOS/EOS decode lossless", test_every_special_decode_is_lossless_in_practice),
    ]

    print("Training shared tokenizer on corpus.txt "
          f"({SHARED_MERGES} merges)...")
    tok = shared_tokenizer()
    print(f"Shared tokenizer: vocab_size={tok.vocab_size}, "
          f"merges={len(tok.merges)}\n")

    failures = 0
    for name, func in tests:
        try:
            func()
            print(f"  PASS  {name}")
        except AssertionError as error:
            failures += 1
            print(f"  FAIL  {name}: {error}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}: {type(error).__name__}: {error}")

    print()
    if failures:
        print(f"{len(tests) - failures}/{len(tests)} tests passed "
              f"({failures} failed)")
    else:
        print(f"All {len(tests)} tests passed.")

    print()
    run_evaluation(tok)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run_all_tests())