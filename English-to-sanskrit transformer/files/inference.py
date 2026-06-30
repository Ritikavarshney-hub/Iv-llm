

import os
import argparse
import torch

from model import build_transformer, Transformer
from bpetokenizer import Vocabulary, SOS_ID, EOS_ID, PAD_ID


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]

    src_vocab = Vocabulary.load(os.path.join(cfg["save_dir"], "src_vocab.json"))
    tgt_vocab = Vocabulary.load(os.path.join(cfg["save_dir"], "tgt_vocab.json"))

    model = build_transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg["d_model"],
        d_ff=cfg["d_ff"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        dropout=0.0,            # no dropout at inference
        seq_len=cfg["max_len"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']}  "
          f"(val_loss={ckpt['val_loss']:.4f})")
    return model, src_vocab, tgt_vocab, cfg["max_len"]


def make_src_mask(src: torch.Tensor) -> torch.Tensor:
    """(B, S) → (B, 1, 1, S)"""
    return (src != PAD_ID).unsqueeze(1).unsqueeze(2)


@torch.no_grad()
def greedy_decode(model: Transformer, src_ids: list,
                  tgt_vocab: Vocabulary, max_len: int,
                  device: torch.device) -> list:
    src = torch.tensor([src_ids], dtype=torch.long).to(device)
    src_mask = make_src_mask(src)

    memory = model.encode(src, src_mask)
    ys = [SOS_ID]

    for _ in range(max_len):
        tgt = torch.tensor([ys], dtype=torch.long).to(device)
        tgt_len = tgt.shape[1]
        causal = torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool)) \
                      .unsqueeze(0).unsqueeze(0).to(device)

        out    = model.decode(tgt, memory, src_mask, causal)
        logits = model.project(out[:, -1, :])   # (1, vocab)
        next_id = logits.argmax(dim=-1).item()
        ys.append(next_id)
        if next_id == EOS_ID:
            break

    return ys[1:]  

@torch.no_grad()
def beam_search_decode(model: Transformer, src_ids: list,
                       tgt_vocab: Vocabulary, max_len: int,
                       device: torch.device, beam_size: int = 5) -> list:
    import math

    src = torch.tensor([src_ids], dtype=torch.long).to(device)
    src_mask = make_src_mask(src)
    memory = model.encode(src, src_mask)

    beams = [(0.0, [SOS_ID])]
    completed = []

    for _ in range(max_len):
        all_candidates = []
        for score, seq in beams:
            if seq[-1] == EOS_ID:
                completed.append((score, seq))
                continue

            tgt = torch.tensor([seq], dtype=torch.long).to(device)
            tgt_len = tgt.shape[1]
            causal = torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool)) \
                          .unsqueeze(0).unsqueeze(0).to(device)

            out    = model.decode(tgt, memory, src_mask, causal)
            logits = model.project(out[:, -1, :])
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

            topk = log_probs.topk(beam_size)
            for log_p, token_id in zip(topk.values, topk.indices):
                new_score = score + log_p.item()
                all_candidates.append((new_score, seq + [token_id.item()]))

        if not all_candidates:
            break

        all_candidates.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
        beams = all_candidates[:beam_size]

        if all(seq[-1] == EOS_ID for _, seq in beams):
            completed.extend(beams)
            break

    completed.extend(beams)
    completed.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
    best_seq = completed[0][1]

    if best_seq and best_seq[0] == SOS_ID:
        best_seq = best_seq[1:]
    if best_seq and best_seq[-1] == EOS_ID:
        best_seq = best_seq[:-1]
    return best_seq

def translate(sentence: str, model, src_vocab, tgt_vocab, max_len, device,
              method: str = "beam", beam_size: int = 5) -> str:
    src_ids = src_vocab.encode(sentence, add_sos=True, add_eos=True)
    if len(src_ids) > max_len:
        src_ids = src_ids[:max_len - 1] + [EOS_ID]

    if method == "greedy":
        ids = greedy_decode(model, src_ids, tgt_vocab, max_len, device)
    else:
        ids = beam_search_decode(model, src_ids, tgt_vocab, max_len, device,
                                 beam_size=beam_size)

    return tgt_vocab.decode(ids, skip_special=True)


def main():
    parser = argparse.ArgumentParser(description="English → Sanskrit Translator")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--method",     default="beam", choices=["greedy", "beam"])
    parser.add_argument("--beam_size",  default=5, type=int)
    parser.add_argument("--input",      default=None, help="Input file (one sentence per line)")
    parser.add_argument("--output",     default=None, help="Output file for translations")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, src_vocab, tgt_vocab, max_len = load_model(args.checkpoint, device)

    if args.input:
        # batch translation from file
        with open(args.input, encoding="utf-8") as f:
            sentences = [l.strip() for l in f if l.strip()]

        results = []
        for i, sent in enumerate(sentences, 1):
            translation = translate(sent, model, src_vocab, tgt_vocab,
                                    max_len, device, args.method, args.beam_size)
            results.append(translation)
            print(f"[{i}/{len(sentences)}] {sent}")
            print(f"  → {translation}\n")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write("\n".join(results))
            print(f"Translations saved to {args.output}")
    else:
        # interactive mode
        print(f"\nEnglish → Sanskrit Translator  (method={args.method})")
        print("Type 'quit' to exit.\n")
        while True:
            try:
                sentence = input("English: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if sentence.lower() in ("quit", "exit", "q"):
                break
            if not sentence:
                continue
            translation = translate(sentence, model, src_vocab, tgt_vocab,
                                    max_len, device, args.method, args.beam_size)
            print(f"Sanskrit: {translation}\n")


if __name__ == "__main__":
    main()
