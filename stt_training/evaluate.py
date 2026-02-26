"""
WATZS STT Evaluation Script
=============================
Benchmarks stock Whisper-small vs. your fine-tuned model on
keyword recognition accuracy.

Computes:
  - Word Error Rate (WER) on the validation set
  - Keyword-specific recall and precision
  - Side-by-side comparison table

Usage:
    python evaluate.py                              # Compare both models
    python evaluate.py --model-path ../models/watzs-whisper  # Evaluate one model
    python evaluate.py --test-keywords              # Only test keyword sentences
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_dependencies():
    """Verify required libraries."""
    missing = []
    for lib in ["transformers", "torch", "jiwer", "datasets"]:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)

    if missing:
        print(f"ERROR: Missing: {', '.join(missing)}")
        print("Run: pip install transformers torch jiwer datasets")
        sys.exit(1)


def load_keywords():
    """Load all keywords from keywords.json."""
    keywords_path = PROJECT_ROOT / "config" / "keywords.json"
    with open(keywords_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    all_keywords = []
    for level_key, level_data in config.get("levels", {}).items():
        for kw in level_data.get("keywords", []):
            all_keywords.append(kw.lower())

    secret = config.get("secret_code", {}).get("phrase", "")
    if secret:
        all_keywords.append(secret.lower())

    return all_keywords


def load_model(model_path):
    """Load a Whisper model for evaluation."""
    from transformers import pipeline
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"  Loading model: {model_path}")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=str(model_path),
        device=device,
        torch_dtype=torch_dtype,
    )
    return pipe


def evaluate_model(pipe, dataset, keywords, label="Model"):
    """
    Evaluate a model on the validation set.

    Returns:
        dict with WER, keyword_recall, keyword_precision, per_keyword stats
    """
    import soundfile as sf
    import numpy as np
    from jiwer import wer as compute_wer

    print(f"\n  Evaluating {label}...")

    predictions = []
    references = []
    keyword_stats = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})

    # Build WAV file lookup for resolving relative paths
    data_dir = Path(__file__).parent / "data"
    wav_lookup = {}
    for wav_file in data_dir.rglob("*.wav"):
        wav_lookup[wav_file.name] = str(wav_file)

    total = len(dataset)
    for i in range(total):
        audio_path = dataset[i]["audio_path"]
        reference = dataset[i]["sentence"].lower().strip()

        # Resolve path if needed
        if not Path(audio_path).exists():
            basename = Path(audio_path).name
            if basename in wav_lookup:
                audio_path = wav_lookup[basename]
            else:
                print(f"    ⚠️  Skipping missing file: {audio_path}")
                continue

        # Load audio with soundfile (bypasses torchcodec)
        audio_array, sr = sf.read(audio_path, dtype="float32")
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != 16000:
            try:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            except ImportError:
                ratio = 16000 / sr
                new_len = int(len(audio_array) * ratio)
                audio_array = np.interp(
                    np.linspace(0, len(audio_array), new_len),
                    np.arange(len(audio_array)),
                    audio_array
                )

        # Run inference
        result = pipe(audio_array)
        prediction = result["text"].lower().strip()

        predictions.append(prediction)
        references.append(reference)

        # Check keyword detection
        for kw in keywords:
            kw_in_ref = kw in reference
            kw_in_pred = kw in prediction

            if kw_in_ref and kw_in_pred:
                keyword_stats[kw]["tp"] += 1  # True positive
            elif kw_in_ref and not kw_in_pred:
                keyword_stats[kw]["fn"] += 1  # False negative (missed)
            elif not kw_in_ref and kw_in_pred:
                keyword_stats[kw]["fp"] += 1  # False positive

        if (i + 1) % 50 == 0:
            print(f"    Processed {i+1}/{total}...")

    # Compute overall WER
    overall_wer = compute_wer(references, predictions) * 100

    # Compute keyword metrics
    total_tp = sum(s["tp"] for s in keyword_stats.values())
    total_fn = sum(s["fn"] for s in keyword_stats.values())
    total_fp = sum(s["fp"] for s in keyword_stats.values())

    keyword_recall = (
        total_tp / (total_tp + total_fn) * 100
        if (total_tp + total_fn) > 0 else 0
    )
    keyword_precision = (
        total_tp / (total_tp + total_fp) * 100
        if (total_tp + total_fp) > 0 else 0
    )

    results = {
        "label": label,
        "wer": overall_wer,
        "keyword_recall": keyword_recall,
        "keyword_precision": keyword_precision,
        "total_samples": total,
        "keyword_stats": dict(keyword_stats),
        "predictions": predictions,
        "references": references,
    }

    return results


def print_results(results):
    """Print evaluation results in a formatted table."""
    print(f"\n  {'='*50}")
    print(f"  Results: {results['label']}")
    print(f"  {'='*50}")
    print(f"  Overall WER:       {results['wer']:.2f}%")
    print(f"  Keyword Recall:    {results['keyword_recall']:.1f}%")
    print(f"  Keyword Precision: {results['keyword_precision']:.1f}%")
    print(f"  Total samples:     {results['total_samples']}")

    # Per-keyword breakdown
    kw_stats = results["keyword_stats"]
    if kw_stats:
        print(f"\n  {'Keyword':<25} {'Recall':>8} {'TP':>5} {'FN':>5} {'FP':>5}")
        print(f"  {'-'*48}")
        for kw in sorted(kw_stats.keys()):
            s = kw_stats[kw]
            recall = s["tp"] / (s["tp"] + s["fn"]) * 100 if (s["tp"] + s["fn"]) > 0 else 0
            marker = " ⚠️" if recall < 80 else " ✅" if recall >= 90 else ""
            print(f"  {kw:<25} {recall:>7.1f}% {s['tp']:>5} {s['fn']:>5} {s['fp']:>5}{marker}")


def print_comparison(stock_results, finetuned_results):
    """Print side-by-side comparison."""
    print(f"\n{'='*60}")
    print(f"  COMPARISON: Stock vs Fine-tuned")
    print(f"{'='*60}")

    metrics = [
        ("WER ↓", "wer", True),
        ("Keyword Recall ↑", "keyword_recall", False),
        ("Keyword Precision ↑", "keyword_precision", False),
    ]

    print(f"\n  {'Metric':<25} {'Stock':>12} {'Fine-tuned':>12} {'Delta':>10}")
    print(f"  {'-'*59}")

    for name, key, lower_better in metrics:
        stock_val = stock_results[key]
        ft_val = finetuned_results[key]
        delta = ft_val - stock_val
        if lower_better:
            marker = "✅" if delta < 0 else "⚠️"
            delta_str = f"{delta:+.2f}%"
        else:
            marker = "✅" if delta > 0 else "⚠️"
            delta_str = f"{delta:+.2f}%"

        print(f"  {name:<25} {stock_val:>11.2f}% {ft_val:>11.2f}% {delta_str:>8} {marker}")


def show_error_examples(results, n=10):
    """Show examples where the model made errors."""
    print(f"\n  Sample errors ({results['label']}):")
    print(f"  {'-'*50}")

    shown = 0
    for ref, pred in zip(results["references"], results["predictions"]):
        if ref.strip() != pred.strip() and shown < n:
            print(f"  REF:  \"{ref}\"")
            print(f"  PRED: \"{pred}\"")
            print()
            shown += 1


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Whisper models for WATZS keyword detection"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to fine-tuned model (default: ../models/watzs-whisper)"
    )
    parser.add_argument(
        "--stock-model", type=str, default="openai/whisper-small",
        help="Stock model to compare against (default: openai/whisper-small)"
    )
    parser.add_argument(
        "--compare", action="store_true", default=True,
        help="Compare stock vs fine-tuned (default: True)"
    )
    parser.add_argument(
        "--no-compare", action="store_true",
        help="Only evaluate the fine-tuned model"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit evaluation samples (for quick testing)"
    )
    parser.add_argument(
        "--show-errors", action="store_true",
        help="Show example transcription errors"
    )
    args = parser.parse_args()

    check_dependencies()

    from datasets import DatasetDict, Dataset

    print("\n📊 WATZS — STT Model Evaluation")
    print("=" * 60)

    # Load dataset
    dataset_path = Path(__file__).parent / "dataset"
    if not dataset_path.exists():
        print(f"  ❌ Dataset not found at {dataset_path}")
        print(f"  Run build_manifest.py first.")
        sys.exit(1)

    raw_dataset = DatasetDict.load_from_disk(str(dataset_path))

    # Extract paths from Arrow table (bypass torchcodec)
    def rebuild_split(split_dataset):
        table = split_dataset.data
        audio_col = table.column("audio")
        paths = []
        for i in range(len(audio_col)):
            item = audio_col[i].as_py()
            if isinstance(item, dict):
                paths.append(item.get("path", ""))
            elif isinstance(item, str):
                paths.append(item)
            else:
                paths.append(str(item))
        sentences = [table.column("sentence")[i].as_py() for i in range(len(table))]
        return Dataset.from_dict({"audio_path": paths, "sentence": sentences})

    val_dataset = rebuild_split(raw_dataset["validation"])

    if args.max_samples and args.max_samples < len(val_dataset):
        val_dataset = val_dataset.select(range(args.max_samples))

    print(f"  Validation samples: {len(val_dataset)}")

    # Load keywords
    keywords = load_keywords()
    print(f"  Keywords to track: {len(keywords)}")

    # Determine model path
    model_path = args.model_path or str(PROJECT_ROOT / "models" / "watzs-whisper")
    model_exists = Path(model_path).exists()

    if not model_exists and args.no_compare:
        print(f"\n  ❌ Fine-tuned model not found at {model_path}")
        print(f"  Run fine_tune_whisper.py first.")
        sys.exit(1)

    # Evaluate stock model
    stock_results = None
    if not args.no_compare:
        stock_pipe = load_model(args.stock_model)
        stock_results = evaluate_model(
            stock_pipe, val_dataset, keywords,
            label=f"Stock ({args.stock_model})"
        )
        print_results(stock_results)
        if args.show_errors:
            show_error_examples(stock_results)
        del stock_pipe  # Free memory

    # Evaluate fine-tuned model
    finetuned_results = None
    if model_exists:
        finetuned_pipe = load_model(model_path)
        finetuned_results = evaluate_model(
            finetuned_pipe, val_dataset, keywords,
            label=f"Fine-tuned ({model_path})"
        )
        print_results(finetuned_results)
        if args.show_errors:
            show_error_examples(finetuned_results)
    else:
        print(f"\n  ⚠️  Fine-tuned model not found at {model_path}")
        print(f"  Showing stock model results only.")
        print(f"  Run fine_tune_whisper.py to create the fine-tuned model.")

    # Comparison
    if stock_results and finetuned_results:
        print_comparison(stock_results, finetuned_results)

    print(f"\n{'='*60}")
    print(f"✅ Evaluation complete!")
    print(f"{'='*60}")

    if finetuned_results:
        wer = finetuned_results["wer"]
        recall = finetuned_results["keyword_recall"]
        if wer <= 20 and recall >= 90:
            print(f"  🎉 Model meets targets! (WER: {wer:.1f}% ≤ 20%, Recall: {recall:.1f}% ≥ 90%)")
        elif wer <= 20:
            print(f"  ✅ WER target met ({wer:.1f}% ≤ 20%)")
            print(f"  ⚠️  Keyword recall below target ({recall:.1f}% < 90%)")
            print(f"  Tip: Record more keyword utterances and retrain.")
        else:
            print(f"  ⚠️  WER above target ({wer:.1f}% > 20%)")
            print(f"  Tip: Add more training data or increase max-steps.")


if __name__ == "__main__":
    main()
