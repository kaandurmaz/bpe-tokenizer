# BPE Tokenizer

A Byte Pair Encoding (BPE) tokenizer implemented in Python.

BPE is a subword tokenization algorithm used in modern language models (GPT, etc.). It iteratively merges the most frequent adjacent token pairs in a corpus until reaching a target vocabulary size, then uses those learned merge rules to tokenize new text.

## How it works

1. **Initialize**: split every word in the corpus into characters, append an end-of-word marker `</w>`
2. **Find**: the most frequent adjacent pair across all words
3. **Merge**: that pair into a single token, update the vocabulary
4. **Repeat**: until the vocabulary reaches the target size
5. **Encode**: new text by applying the learned merge rules in order

## Usage

```
python bpe.py
```

No arguments needed. The script reads `corpus.txt`, trains to `TARGET_VOCAB_SIZE = 8192`, encodes two test sentences, and writes a log file.

To change the target vocabulary size or test sentences, edit the constants at the bottom of [bpe.py](bpe.py).

## Output

Training progress and tokenization results are logged to the console and to a text file.

Example merges learned from the included Turkish corpus:

```
'şim' + 'diye</w>' => 'şimdiye</w>'  (freq=100)
'seb' + 'ze</w>' => 'sebze</w>'  (freq=100)
'seç' + 'ti.</w>' => 'seçti.</w>'  (freq=100)
'tarz' + 'ında</w>' => 'tarzında</w>'  (freq=100)
'kış' + 'ları</w>' => 'kışları</w>'  (freq=100)
```

## Requirements

- Python 3.10+
- No external packages — standard library only
