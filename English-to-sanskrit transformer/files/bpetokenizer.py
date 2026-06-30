
import re
import json
import os
import unicodedata
from collections import Counter
from typing import List, Dict, Tuple


# ── special tokens ─────────────────────────────────────────────────────────
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]

PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3


# ── text cleaning ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    # NFC normalisation is critical for Sanskrit: many Devanagari characters
    # have NFD / NFC variants that are byte-different but visually identical.
    text = unicodedata.normalize('NFC', text.strip())
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    # remove zero-width characters common in Devanagari text
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    # give dandas their own token so BPE never merges them into words
    text = re.sub(r'([\u0964\u0965])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── BPE core ───────────────────────────────────────────────────────────────

def _get_word_vocab(texts: List[str]) -> Dict[Tuple, int]:
    vocab: Dict[Tuple, int] = {}
    for text in texts:
        for word in clean_text(text).split():
            chars = list(word)
            if not chars:
                continue
            chars[0] = '▁' + chars[0]
            key = tuple(chars)
            vocab[key] = vocab.get(key, 0) + 1
    return vocab


def _get_pairs(vocab: Dict[Tuple, int]) -> Dict[Tuple, int]:
    pairs: Dict[Tuple, int] = {}
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            p = (word[i], word[i + 1])
            pairs[p] = pairs.get(p, 0) + freq
    return pairs


def _merge_vocab(pair: Tuple, vocab: Dict[Tuple, int]) -> Dict[Tuple, int]:
    merged = ''.join(pair)
    new_vocab: Dict[Tuple, int] = {}
    for word, freq in vocab.items():
        new_word, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_vocab[tuple(new_word)] = freq
    return new_vocab


def learn_bpe(texts: List[str], num_merges: int) -> List[Tuple[str, str]]:
    print(f"  Learning {num_merges} BPE merges …")
    vocab = _get_word_vocab(texts)
    merges = []
    for i in range(num_merges):
        pairs = _get_pairs(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = _merge_vocab(best, vocab)
        merges.append(best)
        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{num_merges} merges done")
    print(f"  Learned {len(merges)} merges")
    return merges


# ── Vocabulary class ───────────────────────────────────────────────────────

class Vocabulary:
    def __init__(self):
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.merges:   List[Tuple[str, str]] = []
        for tok in SPECIAL_TOKENS:
            self._add(tok)

    def _add(self, token: str) -> int:
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
        return self.token2id[token]

    # ── build ────────────────────────────────────────────────────────────────

    def build(self, texts: List[str], num_merges: int = 6000, min_freq: int = 1):
        self.merges = learn_bpe(texts, num_merges)

        # apply BPE to corpus and collect all resulting tokens
        print("  Collecting final token set …")
        counter: Counter = Counter()

        # always add every individual character first (guarantees full coverage)
        for text in texts:
            for ch in clean_text(text):
                if ch != ' ':
                    counter[ch] += 1
            # add word-boundary versions of first chars
            for word in clean_text(text).split():
                if word:
                    counter['▁' + word[0]] += 1

        # add BPE-merged tokens
        for text in texts:
            for tok in self._bpe_encode_text(text):
                counter[tok] += 1

        for token, freq in counter.items():
            if freq >= min_freq:
                self._add(token)

        print(f"  Vocabulary size: {len(self):,}")

    def _bpe_encode_word(self, word: str) -> List[str]:
        if not word:
            return []
        chars = list(word)
        chars[0] = '▁' + chars[0]
        tokens = list(chars)

        for merge_a, merge_b in self.merges:
            merged, new_tokens, i = merge_a + merge_b, [], 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == merge_a and tokens[i + 1] == merge_b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def _bpe_encode_text(self, text: str) -> List[str]:
        tokens = []
        for word in clean_text(text).split():
            tokens.extend(self._bpe_encode_word(word))
        return tokens

    # ── public API ───────────────────────────────────────────────────────────

    def encode(self, text: str, add_sos: bool = False, add_eos: bool = False) -> List[int]:
        tokens = self._bpe_encode_text(text)
        ids = []
        for tok in tokens:
            if tok in self.token2id:
                ids.append(self.token2id[tok])
            else:
                # character-level fallback for unseen subwords
                for ch in tok.replace('▁', ''):
                    ids.append(self.token2id.get(ch, UNK_ID))
        if add_sos:
            ids = [SOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        special = {PAD_ID, UNK_ID, SOS_ID, EOS_ID}
        pieces = []
        for i in ids:
            if skip_special and i in special:
                continue
            pieces.append(self.id2token.get(i, UNK_TOKEN))
        # '▁' = word boundary → space
        text = ''.join(pieces).replace('▁', ' ').strip()
        return re.sub(r'\s+', ' ', text)

    def __len__(self):
        return len(self.token2id)

    # ── save / load ──────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"token2id": self.token2id, "merges": self.merges},
                      f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        obj = cls.__new__(cls)
        obj.token2id = {}
        obj.id2token = {}
        obj.merges   = []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj.token2id = data["token2id"]
        obj.id2token = {int(i): t for t, i in obj.token2id.items()}
        obj.merges   = [tuple(m) for m in data["merges"]]
        return obj


# ── convenience builder ────────────────────────────────────────────────────

def build_vocabs(train_en_path: str, train_sa_path: str,
                 save_dir: str = "checkpoints",
                 num_merges_en: int = 4000,
                 num_merges_sa: int = 6000,
                 min_freq: int = 1):
    with open(train_en_path, encoding="utf-8") as f:
        en_lines = f.read().splitlines()
    with open(train_sa_path, encoding="utf-8") as f:
        sa_lines = f.read().splitlines()

    print("Building English BPE vocabulary …")
    src_vocab = Vocabulary()
    src_vocab.build(en_lines, num_merges=num_merges_en, min_freq=min_freq)
    print("\nBuilding Sanskrit BPE vocabulary …")
    tgt_vocab = Vocabulary()
    tgt_vocab.build(sa_lines, num_merges=num_merges_sa, min_freq=min_freq)

    os.makedirs(save_dir, exist_ok=True)
    src_vocab.save(os.path.join(save_dir, "src_vocab.json"))
    tgt_vocab.save(os.path.join(save_dir, "tgt_vocab.json"))
    print(f"\nSaved to {save_dir}/")
    return src_vocab, tgt_vocab