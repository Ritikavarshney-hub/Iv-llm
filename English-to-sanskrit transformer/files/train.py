
import os
import math
import time
import contextlib
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from model import build_transformer
from bpetokenizer import build_vocabs, Vocabulary
from dataset import get_dataloaders

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = {
    # paths
    "data_dir":    "data/final_data",
    "save_dir":    "checkpoints",
    "best_model":  "checkpoints/best_model.pt",
    # vocab (BPE merge counts — more merges for Sanskrit due to morphology)
    "num_merges_en": 4000,
    "num_merges_sa": 6000,
    "min_freq": 1,
    # model
    "d_model":    256,
    "d_ff":      1024,
    "num_heads":    8,
    "num_layers":   4,
    "dropout":    0.1,
    "max_len":    150,
    # training
    "batch_size":  32,
    "accum_steps":  4,      # effective batch = batch_size × accum_steps = 128
    "num_epochs":  50,
    "warmup_steps": 4000,
    "label_smoothing": 0.1,
    "clip_grad":   1.0,
    "patience":     7,      # early-stopping: stop if val loss doesn't improve for this many epochs
    "mixed_precision": True,  # fp16 on CUDA — no-op on CPU
    # misc
    "seed": 42,
}
# ─────────────────────────────────────────────────────────────────────────────


def get_lr_scheduler(optimizer, d_model: int, warmup_steps: int):
    def lr_lambda(step):
        step = max(step, 1)
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)
    return LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, optimizer, scheduler, criterion, device,
              scaler=None, train: bool = True, accum_steps: int = 1):
    model.train() if train else model.eval()
    total_loss, total_tokens = 0.0, 0

    use_amp = scaler is not None
    grad_ctx = torch.enable_grad() if train else torch.no_grad()

    with grad_ctx:
        if train:
            optimizer.zero_grad()

        for step, (src, tgt_input, tgt_output, src_mask, tgt_mask) in enumerate(loader):
            src        = src.to(device)
            tgt_input  = tgt_input.to(device)
            tgt_output = tgt_output.to(device)
            src_mask   = src_mask.to(device)
            tgt_mask   = tgt_mask.to(device)

            amp_ctx = torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext()
            with amp_ctx:
                memory = model.encode(src, src_mask)
                out    = model.decode(tgt_input, memory, src_mask, tgt_mask)
                logits = model.project(out)             # (B, T, vocab)
                B, T, V = logits.shape
                # divide loss by accum_steps so gradients scale correctly
                loss = criterion(logits.view(B * T, V),
                                 tgt_output.reshape(B * T)) / accum_steps

            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # update weights every accum_steps batches (or at the last batch)
                if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
                    if use_amp:
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_grad"])
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            non_pad = (tgt_output != 0).sum().item()
            # undo the accum_steps division for loss tracking
            total_loss   += loss.item() * accum_steps * non_pad
            total_tokens += non_pad

    return total_loss / max(total_tokens, 1)


def train():
    torch.manual_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    use_amp = CONFIG["mixed_precision"] and device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("Mixed-precision (fp16) training enabled.")

    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    # ── vocabularies ──────────────────────────────────────────────────────────
    src_vocab_path = os.path.join(CONFIG["save_dir"], "src_vocab.json")
    tgt_vocab_path = os.path.join(CONFIG["save_dir"], "tgt_vocab.json")

    vocab_loaded = False
    if os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path):
        try:
            print("Loading cached vocabularies …")
            src_vocab = Vocabulary.load(src_vocab_path)
            tgt_vocab = Vocabulary.load(tgt_vocab_path)
            # sanity check: BPE vocab has a merges list
            assert hasattr(src_vocab, "merges") and src_vocab.merges
            print(f"  EN vocab size: {len(src_vocab)}   SA vocab size: {len(tgt_vocab)}")
            vocab_loaded = True
        except (KeyError, AssertionError):
            print("  Cached vocab is not BPE format — rebuilding …")

    if not vocab_loaded:
        print("Building BPE vocabularies …")
        src_vocab, tgt_vocab = build_vocabs(
            os.path.join(CONFIG["data_dir"], "train.en"),
            os.path.join(CONFIG["data_dir"], "train.sa"),
            save_dir=CONFIG["save_dir"],
            num_merges_en=CONFIG["num_merges_en"],
            num_merges_sa=CONFIG["num_merges_sa"],
            min_freq=CONFIG["min_freq"],
        )

    # ── data ──────────────────────────────────────────────────────────────────
    print("\nLoading datasets …")
    train_loader, val_loader, _ = get_dataloaders(
        CONFIG["data_dir"], src_vocab, tgt_vocab,
        batch_size=CONFIG["batch_size"], max_len=CONFIG["max_len"]
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=CONFIG["d_model"],
        d_ff=CONFIG["d_ff"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
        seq_len=CONFIG["max_len"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {num_params:,}")

    # ── loss / optimiser ──────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        ignore_index=0,
        label_smoothing=CONFIG["label_smoothing"]
    )
    optimizer = Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = get_lr_scheduler(optimizer, CONFIG["d_model"], CONFIG["warmup_steps"])

    best_val_loss  = float("inf")
    patience_count = 0
    print(f"\nStarting training for up to {CONFIG['num_epochs']} epochs …\n")

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, scheduler,
                               criterion, device, scaler=scaler, train=True,
                               accum_steps=CONFIG["accum_steps"])
        val_loss   = run_epoch(model, val_loader, optimizer, scheduler,
                               criterion, device, scaler=None, train=False,
                               accum_steps=1)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{CONFIG['num_epochs']} | "
              f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
              f"Val PPL: {math.exp(val_loss):.2f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Time: {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": val_loss,
                "config": CONFIG,
            }, CONFIG["best_model"])
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= CONFIG["patience"]:
                print(f"\nEarly stopping: val loss did not improve for "
                      f"{CONFIG['patience']} epochs.")
                break

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.4f}  (PPL: {math.exp(best_val_loss):.2f})")


if __name__ == "__main__":
    train()
