"""
dataset.py  –  PyTorch Dataset and DataLoader for English → Sanskrit translation.
"""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple

from bpetokenizer import Vocabulary, PAD_ID, SOS_ID, EOS_ID


class TranslationDataset(Dataset):
    """
    Reads parallel text files (one sentence per line) and produces
    (src_ids, tgt_ids) integer tensors.
    """

    def __init__(self,
                 src_path: str,
                 tgt_path: str,
                 src_vocab: Vocabulary,
                 tgt_vocab: Vocabulary,
                 max_len: int = 150):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        self.pairs: List[Tuple[List[int], List[int]]] = []

        with open(src_path, encoding="utf-8") as fs, \
             open(tgt_path, encoding="utf-8") as ft:
            for src_line, tgt_line in zip(fs, ft):
                src_ids = src_vocab.encode(src_line.strip(), add_sos=True, add_eos=True)
                tgt_ids = tgt_vocab.encode(tgt_line.strip(), add_sos=True, add_eos=True)
                # skip sentences that are too long
                if len(src_ids) <= max_len and len(tgt_ids) <= max_len:
                    self.pairs.append((src_ids, tgt_ids))

        print(f"  Loaded {len(self.pairs)} sentence pairs from {src_path}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.pairs[idx]
        return torch.tensor(src_ids, dtype=torch.long), \
               torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch):
    """
    Pads a batch of (src, tgt) pairs to the same length.
    Returns:
        src          (B, src_len)
        tgt_input    (B, tgt_len-1)  – decoder input  (<sos> … last-1)
        tgt_output   (B, tgt_len-1)  – expected output (first+1 … <eos>)
        src_mask     (B, 1, 1, src_len)
        tgt_mask     (B, 1, tgt_len-1, tgt_len-1)
    """
    src_batch, tgt_batch = zip(*batch)

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_ID)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_ID)

    tgt_input  = tgt_padded[:, :-1]   # drop last token
    tgt_output = tgt_padded[:, 1:]    # drop first token (<sos>)

    # src padding mask: 1 where real token, 0 where pad
    src_mask = (src_padded != PAD_ID).unsqueeze(1).unsqueeze(2)  # (B,1,1,S)

    # tgt causal mask: padding mask AND causal mask
    tgt_len = tgt_input.shape[1]
    tgt_pad_mask = (tgt_input != PAD_ID).unsqueeze(1).unsqueeze(2)  # (B,1,1,T)
    causal_mask  = torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool)) \
                        .unsqueeze(0).unsqueeze(0)                   # (1,1,T,T)
    tgt_mask = tgt_pad_mask & causal_mask                           # (B,1,T,T)

    return src_padded, tgt_input, tgt_output, src_mask, tgt_mask


class BucketBatchSampler:
    """
    Yields batches whose sentences are close in source length.
    This minimises padding waste — typically cuts padding tokens by ~30-40%.
    """
    def __init__(self, lengths: List[int], batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        sorted_idx = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batches = [sorted_idx[i:i + batch_size]
                        for i in range(0, len(sorted_idx), batch_size)]

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            random.shuffle(order)
        for i in order:
            yield self.batches[i]

    def __len__(self):
        return len(self.batches)


def get_dataloaders(data_dir: str,
                    src_vocab: Vocabulary,
                    tgt_vocab: Vocabulary,
                    batch_size: int = 32,
                    max_len: int = 150):
    """Build train / val / test DataLoaders."""
    train_ds = TranslationDataset(
        os.path.join(data_dir, "train.en"),
        os.path.join(data_dir, "train.sa"),
        src_vocab, tgt_vocab, max_len
    )
    val_ds = TranslationDataset(
        os.path.join(data_dir, "dev.en"),
        os.path.join(data_dir, "dev.sa"),
        src_vocab, tgt_vocab, max_len
    )
    test_ds = TranslationDataset(
        os.path.join(data_dir, "test.en"),
        os.path.join(data_dir, "test.sa"),
        src_vocab, tgt_vocab, max_len
    )

    # bucket sampler groups similar-length sentences → much less padding
    train_sampler = BucketBatchSampler(
        [len(p[0]) for p in train_ds.pairs], batch_size, shuffle=True
    )
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    return train_loader, val_loader, test_loader
