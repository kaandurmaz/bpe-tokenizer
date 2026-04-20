import logging
from collections import defaultdict
from pathlib import Path


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
    words = [" ".join(list(w)) + " </w>" for w in text.split()]

    for pair in merges:
        bigram = " ".join(pair)
        merged = "".join(pair)
        words = [w.replace(bigram, merged) for w in words]

    tokens: list[str] = []
    for word in words:
        tokens.extend(word.split())
    return tokens


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
