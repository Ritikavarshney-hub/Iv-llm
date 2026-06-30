# English → Sanskrit Neural Machine Translator

A Transformer-based sequence-to-sequence model trained on the **Saamayik** parallel corpus.

---

## Project Structure

```
.
├── model.py          ← Transformer architecture (fixed)
├── tokenizer.py      ← Word-level tokenizer + Vocabulary
├── dataset.py        ← PyTorch Dataset & DataLoader
├── train.py          ← Training loop
├── inference.py      ← Greedy & beam-search decoding
├── checkpoints/      ← Saved model weights & vocabularies (auto-created)
└── data/
    └── final_data/
        ├── train.en / train.sa   (43 k pairs)
        ├── dev.en   / dev.sa     (2.4 k pairs)
        └── test.en  / test.sa    (2.4 k pairs)
```

---

## Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# (use the CUDA wheel if you have a GPU)
```

No other external libraries are required.

---

## Training

```bash
python train.py
```

Edit the `CONFIG` dict at the top of `train.py` to adjust hyperparameters:

| Key | Default | Description |
|-----|---------|-------------|
| `d_model` | 256 | Model dimension |
| `d_ff` | 1024 | Feed-forward inner dimension |
| `num_heads` | 8 | Attention heads |
| `num_layers` | 4 | Encoder/decoder layers |
| `dropout` | 0.1 | Dropout rate |
| `batch_size` | 32 | Training batch size |
| `num_epochs` | 30 | Number of epochs |
| `max_len` | 150 | Max token sequence length |
| `label_smoothing` | 0.1 | Label smoothing for cross-entropy |

The best model (lowest validation loss) is saved to `checkpoints/best_model.pt`.

---

## Inference

**Interactive mode:**
```bash
python inference.py
```

**Translate a file:**
```bash
python inference.py --input my_sentences.txt --output translations.txt
```

**Options:**
```
--checkpoint   Path to model checkpoint  (default: checkpoints/best_model.pt)
--method       greedy | beam             (default: beam)
--beam_size    Beam width                (default: 5)
```

---

## Bugs Fixed in `model.py`

| Location | Bug | Fix |
|----------|-----|-----|
| `PositionalEncoding.forward` | `.requires_grad(False)` called on tensor — invalid syntax | Changed to `.detach()` |
| `MultiHeadAttention.forward` | Query reshape used `query.shape[1]` as batch dim (wrong) | Changed to `query.shape[0]` (batch) and `-1` for seq dim |
| `MultiHeadAttention.attention` | Called as instance method `self.attention(...)` instead of static | Changed to `MultiHeadAttention.attention(...)` with explicit `@staticmethod` decorator |
| `ProjectionLayer.forward` | Applied `log_softmax` before `CrossEntropyLoss` (double-applies softmax) | Return raw logits; `CrossEntropyLoss` handles softmax internally |
| `build_transformer` defaults | d_model=512, d_ff=2048, layers=6 is very large for a ~43 k sentence dataset | Reduced defaults to d_model=256, d_ff=1024, layers=4 to prevent overfitting |

---

## Tips for Better Results

- **More data**: Consider adding the Bible and Gitasopanam sub-corpora from the zip.
- **BPE tokenisation**: Replace the word-level tokenizer with `sentencepiece` (BPE) for better handling of rare Sanskrit words.
- **Longer training**: Try 50–100 epochs with early stopping.
- **Larger model**: On a GPU, you can restore d_model=512, num_layers=6.
