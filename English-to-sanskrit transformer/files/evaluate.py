"""
evaluate.py  –  Compute BLEU and BERTScore for the English → Sanskrit transformer.

Loads a trained checkpoint, translates a source test file, and scores the
hypotheses against the reference Sanskrit translations.

Metrics
-------
* BLEU       – corpus-level BLEU via `sacrebleu` (language-agnostic tokeniser).
* BERTScore  – precision / recall / F1 via `bert_score` using a multilingual
               BERT model (Devanagari is covered by `bert-base-multilingual-cased`).

Example
-------
    # from inside the `files/` directory
    python evaluate.py \
        --checkpoint checkpoints/best_model.pt \
        --src data/final_data/test.en \
        --ref data/final_data/test.sa \
        --method beam --beam_size 5 \
        --save_hyp checkpoints/test_hyp.sa

Install deps if needed:
    pip install sacrebleu bert-score
"""

import os
import time
import argparse

import torch

from inference import load_model, translate


def read_lines(path: str) -> list:
    """Read a text file as a list of stripped, non-empty-preserving lines."""
    with open(path, encoding="utf-8") as f:
        # keep line alignment with the reference file: strip only trailing newline
        return [line.rstrip("\n") for line in f]


def generate_hypotheses(src_lines, model, src_vocab, tgt_vocab, max_len, device,
                        method: str, beam_size: int) -> list:
    """Translate every source sentence and return the list of hypotheses."""
    hyps = []
    n = len(src_lines)
    t0 = time.time()
    for i, sent in enumerate(src_lines, 1):
        sent = sent.strip()
        if not sent:
            hyps.append("")
            continue
        hyp = translate(sent, model, src_vocab, tgt_vocab, max_len, device,
                        method=method, beam_size=beam_size)
        hyps.append(hyp)
        if i % 25 == 0 or i == n:
            rate = i / max(time.time() - t0, 1e-9)
            print(f"  translated {i}/{n}  ({rate:.1f} sent/s)")
    return hyps


def compute_bleu(hyps: list, refs: list) -> float:
    """Corpus BLEU using sacrebleu. References must be a list of one ref each."""
    try:
        import sacrebleu
    except ImportError as e:
        raise SystemExit(
            "sacrebleu is not installed.  Install it with:  pip install sacrebleu"
        ) from e

    # sacrebleu expects: list of hypotheses, and a list of reference-streams.
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    print(f"\nBLEU = {bleu.score:.2f}")
    print(f"  {bleu}")
    return bleu.score


def compute_bertscore(hyps: list, refs: list, model_type: str,
                       lang: str, device: torch.device) -> dict:
    """BERTScore P/R/F1 averaged over the corpus."""
    try:
        from bert_score import score as bert_score_fn
    except ImportError as e:
        raise SystemExit(
            "bert-score is not installed.  Install it with:  pip install bert-score"
        ) from e

    kwargs = dict(
        cands=hyps,
        refs=refs,
        verbose=True,
        device=str(device),
        rescale_with_baseline=False,
    )
    # `model_type` takes precedence; otherwise fall back to a language code.
    if model_type:
        kwargs["model_type"] = model_type
    else:
        kwargs["lang"] = lang

    P, R, F1 = bert_score_fn(**kwargs)
    result = {
        "precision": P.mean().item(),
        "recall":    R.mean().item(),
        "f1":        F1.mean().item(),
    }
    print(f"\nBERTScore  (model={model_type or lang})")
    print(f"  Precision = {result['precision']:.4f}")
    print(f"  Recall    = {result['recall']:.4f}")
    print(f"  F1        = {result['f1']:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compute BLEU and BERTScore for English → Sanskrit translations."
    )
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt",
                        help="Path to the trained model checkpoint.")
    parser.add_argument("--src", default="data/final_data/test.en",
                        help="Source (English) test file, one sentence per line.")
    parser.add_argument("--ref", default="data/final_data/test.sa",
                        help="Reference (Sanskrit) file, aligned with --src.")
    parser.add_argument("--method", default="beam", choices=["greedy", "beam"],
                        help="Decoding strategy.")
    parser.add_argument("--beam_size", default=5, type=int)
    parser.add_argument("--limit", default=0, type=int,
                        help="Only evaluate the first N sentences (0 = all).")
    parser.add_argument("--save_hyp", default=None,
                        help="Optional file to write the generated hypotheses.")
    parser.add_argument("--hyp_file", default=None,
                        help="Skip decoding and score these pre-generated hypotheses.")
    # BERTScore config
    parser.add_argument("--bertscore_model", default="bert-base-multilingual-cased",
                        help="HuggingFace model for BERTScore (multilingual covers "
                             "Devanagari). Set empty to use --bertscore_lang instead.")
    parser.add_argument("--bertscore_lang", default="hi",
                        help="Language code for BERTScore when no model is given "
                             "(Hindi 'hi' shares the Devanagari script with Sanskrit).")
    parser.add_argument("--no_bleu", dest="no_bleu",
                        action="store_true", help="Skip BLEU.")
    parser.add_argument("--no_bertscore", action="store_true",
                        help="Skip BERTScore.")
    args = parser.parse_args()

    # ── references ──────────────────────────────────────────────────────────
    refs = read_lines(args.ref)

    # ── hypotheses (decode or load) ─────────────────────────────────────────
    if args.hyp_file:
        print(f"Loading pre-generated hypotheses from {args.hyp_file}")
        hyps = read_lines(args.hyp_file)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        model, src_vocab, tgt_vocab, max_len = load_model(args.checkpoint, device)

        src_lines = read_lines(args.src)
        if args.limit > 0:
            src_lines = src_lines[:args.limit]
            refs = refs[:args.limit]

        if len(src_lines) != len(refs):
            print(f"WARNING: src ({len(src_lines)}) and ref ({len(refs)}) line "
                  f"counts differ; truncating to the shorter.")
            n = min(len(src_lines), len(refs))
            src_lines, refs = src_lines[:n], refs[:n]

        print(f"\nTranslating {len(src_lines)} sentences "
              f"(method={args.method}, beam_size={args.beam_size}) …")
        hyps = generate_hypotheses(src_lines, model, src_vocab, tgt_vocab,
                                   max_len, device, args.method, args.beam_size)

        if args.save_hyp:
            with open(args.save_hyp, "w", encoding="utf-8") as f:
                f.write("\n".join(hyps))
            print(f"\nHypotheses saved to {args.save_hyp}")

    # align lengths defensively before scoring
    n = min(len(hyps), len(refs))
    hyps, refs = hyps[:n], refs[:n]
    print(f"\nScoring {n} sentence pairs.")

    # ── metrics ─────────────────────────────────────────────────────────────
    summary = {}
    if not args.no_bleu:
        summary["bleu"] = compute_bleu(hyps, refs)

    if not args.no_bertscore:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bs = compute_bertscore(hyps, refs, args.bertscore_model,
                               args.bertscore_lang, device)
        summary["bertscore_f1"] = bs["f1"]

    # ── final summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    if "bleu" in summary:
        print(f"BLEU          : {summary['bleu']:.2f}")
    if "bertscore_f1" in summary:
        print(f"BERTScore F1  : {summary['bertscore_f1']:.4f}")


if __name__ == "__main__":
    main()
