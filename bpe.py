import logging
from collections import defaultdict
from pathlib import Path

import numpy as np


def _get_log_path() -> Path:
    n = 1
    while (p := Path(f"log_{n}.txt")).exists():
        n += 1
    return p


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_get_log_path(), encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def get_vocab(corpus: str) -> dict[str, int]:
    """Split each word into individual characters and count frequencies."""
    vocab: dict[str, int] = defaultdict(int)
    for word in corpus.split():
        # Space-separated characters + end-of-word marker
        vocab[" ".join(list(word)) + " </w>"] += 1
    return dict(vocab)


def get_pair_freqs(vocab: dict[str, int]) -> dict[tuple[str, str], int]:
    """Count the frequency of every adjacent token pair across the vocabulary."""
    pairs: dict[tuple[str, str], int] = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return dict(pairs)


def merge_pair(pair: tuple[str, str], vocab: dict[str, int]) -> dict[str, int]:
    """Replace every occurrence of the given pair with a single merged token."""
    new_vocab: dict[str, int] = {}
    bigram = " ".join(pair)
    merged = "".join(pair)
    for word, freq in vocab.items():
        new_vocab[word.replace(bigram, merged)] = freq
    return new_vocab


def train_bpe(corpus: str, vocab_size: int) -> tuple[list[tuple[str, str]], set[str]]:
    """
    Train BPE until the vocabulary reaches vocab_size.
    Returns: (list of merge rules, final vocabulary set)
    """
    vocab = get_vocab(corpus)

    # Base vocabulary: all unique characters present in the corpus + </w>
    base_vocab: set[str] = set()
    for word in vocab:
        base_vocab.update(word.split())

    merges: list[tuple[str, str]] = []
    current_vocab_size = len(base_vocab)

    log.info(f"Initial vocabulary size: {current_vocab_size}")

    while current_vocab_size < vocab_size:
        pairs = get_pair_freqs(vocab)
        if not pairs:
            break

        best_pair = max(pairs, key=lambda p: pairs[p])
        best_freq = pairs[best_pair]

        vocab = merge_pair(best_pair, vocab)
        merges.append(best_pair)

        new_token = "".join(best_pair)
        base_vocab.add(new_token)
        current_vocab_size += 1

        log.info(
            f"  [{current_vocab_size}/{vocab_size}] "
            f"'{best_pair[0]}' + '{best_pair[1]}' => '{new_token}'  (freq={best_freq})"
        )

    log.info(f"\nTraining complete. Total merges: {len(merges)}")
    return merges, base_vocab


def encode(text: str, merges: list[tuple[str, str]]) -> list[str]:
    """Tokenize text using the learned merge rules."""
    word_list = text.split()
    unique_words = sorted(set(word_list))
    total = len(unique_words)

    cache: dict[str, list[str]] = {}
    for i, word in enumerate(unique_words, 1):
        tokenized = " ".join(list(word)) + " </w>"
        for pair in merges:
            tokenized = tokenized.replace(" ".join(pair), "".join(pair))
        cache[word] = tokenized.split()

        if i % max(1, total // 20) == 0 or i == total:
            log.info(f"  Encoding unique words: {i}/{total} ({100*i//total}%)")

    tokens: list[str] = []
    for word in word_list:
        tokens.extend(cache[word])
    return tokens


def build_token_to_id(vocab: set[str]) -> dict[str, int]:
    """Assign a stable integer id to each token (sorted for determinism)."""
    return {tok: i for i, tok in enumerate(sorted(vocab))}


def encode_to_ids(
    text: str,
    merges: list[tuple[str, str]],
    token_to_id: dict[str, int],
) -> list[int]:
    """Return integer token ids for the given text."""
    tokens = encode(text, merges)
    return [token_to_id[t] for t in tokens if t in token_to_id]


if __name__ == "__main__":
    corpus_path = "corpus.txt"

    try:
        with open(corpus_path, encoding="utf-8") as f:
            corpus = f.read().lower()
    except FileNotFoundError:
        log.error(f"Error: '{corpus_path}' not found.")
        raise

    TARGET_VOCAB_SIZE = 8192

    merges, final_vocab = train_bpe(corpus, TARGET_VOCAB_SIZE)

    test_sentences = ["koşmak ve atlamak", "istanbullulaştıramadıklarımızdan mısınız"]
    log.info("\n--- Tokenization Test ---")
    for sentence in test_sentences:
        tokens = encode(sentence, merges)
        log.info(f"  '{sentence}' => {tokens}")

    # --- Encode full corpus and save as train.bin ---
    log.info("\n--- Encoding full corpus ---")
    token_to_id = build_token_to_id(final_vocab)
    ids = encode_to_ids(corpus, merges, token_to_id)

    arr = np.array(ids, dtype=np.uint16)
    out_path = Path("train.bin")
    arr.tofile(out_path)

    log.info(f"  Tokens : {len(ids):,}")
    log.info(f"  Vocab  : {len(token_to_id):,} unique tokens")
    log.info(f"  Saved  : {out_path} ({out_path.stat().st_size:,} bytes, dtype=uint16)")
