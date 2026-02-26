"""
WATZS Dataset Builder
=====================
Merges all prepared data (Common Voice, custom keywords, augmented)
into a single HuggingFace DatasetDict for Whisper fine-tuning.

Creates:
  - A HuggingFace Dataset saved to stt_training/dataset/
  - Summary statistics of the final dataset

Usage:
    python build_manifest.py
"""

import os
import sys
import csv
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_manifest(manifest_path):
    """Load a manifest CSV and validate paths exist."""
    rows = []
    missing = 0

    if not manifest_path.exists():
        return rows, 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = Path(row["path"])
            if path.exists():
                rows.append(row)
            else:
                missing += 1

    return rows, missing


def build_dataset():
    """
    Merge all data sources into a HuggingFace DatasetDict.
    """
    try:
        from datasets import Dataset, DatasetDict, Audio
    except ImportError:
        print("ERROR: 'datasets' library not installed.")
        print("Run: pip install datasets")
        sys.exit(1)

    data_dir = Path(__file__).parent / "data"
    output_dir = Path(__file__).parent / "dataset"

    # Collect all manifests
    sources = {
        "Train (Common Voice)": data_dir / "train" / "manifest.csv",
        "Validation": data_dir / "val" / "manifest.csv",
        "Custom Keywords": data_dir / "custom" / "manifest.csv",
        "Augmented": data_dir / "augmented" / "manifest.csv",
    }

    # Load all data
    train_rows = []
    val_rows = []
    stats = {}

    for name, manifest_path in sources.items():
        rows, missing = load_manifest(manifest_path)
        stats[name] = {"loaded": len(rows), "missing": missing}

        if "Validation" in name:
            val_rows.extend(rows)
        else:
            train_rows.extend(rows)

    # Print summary
    print("\n  Data sources:")
    print(f"  {'Source':<30} {'Loaded':>8} {'Missing':>8}")
    print(f"  {'-'*46}")
    for name, s in stats.items():
        print(f"  {name:<30} {s['loaded']:>8} {s['missing']:>8}")

    total_train = len(train_rows)
    total_val = len(val_rows)
    print(f"\n  Total train: {total_train}")
    print(f"  Total val:   {total_val}")
    print(f"  Grand total: {total_train + total_val}")

    if total_train == 0:
        print("\n  ❌ No training data found! Run prepare_data.py first.")
        sys.exit(1)

    if total_val == 0:
        print("\n  ⚠️  No validation data found. Splitting 10% from train.")
        # Take 10% for validation
        import random
        random.seed(42)
        random.shuffle(train_rows)
        split_idx = max(1, int(len(train_rows) * 0.1))
        val_rows = train_rows[:split_idx]
        train_rows = train_rows[split_idx:]

    # Create HuggingFace Dataset
    print(f"\n  Building HuggingFace DatasetDict...")

    def rows_to_dict(rows):
        return {
            "audio": [row["path"] for row in rows],
            "sentence": [row["sentence"] for row in rows],
        }

    train_dataset = Dataset.from_dict(rows_to_dict(train_rows))
    val_dataset = Dataset.from_dict(rows_to_dict(val_rows))

    # Cast audio column to Audio type (auto-loads and resamples)
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=16000))

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
    })

    # Save to disk
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))

    print(f"  ✅ DatasetDict saved to: {output_dir}")
    print(f"\n  Dataset structure:")
    print(f"    train:      {len(train_dataset)} samples")
    print(f"    validation: {len(val_dataset)} samples")
    print(f"    Columns:    {train_dataset.column_names}")

    return dataset_dict


def analyze_keyword_coverage(dataset_dict):
    """
    Analyze how well keywords are represented in the training data.
    """
    import json

    keywords_path = PROJECT_ROOT / "config" / "keywords.json"
    with open(keywords_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    all_keywords = []
    for level_data in config.get("levels", {}).values():
        all_keywords.extend(level_data.get("keywords", []))

    secret = config.get("secret_code", {}).get("phrase")
    if secret:
        all_keywords.append(secret)

    # Count keyword occurrences in training transcripts
    train_sentences = dataset_dict["train"]["sentence"]
    print(f"\n  Keyword coverage in training data:")
    print(f"  {'Keyword':<30} {'Count':>6}")
    print(f"  {'-'*36}")

    low_coverage = []
    for keyword in sorted(all_keywords):
        count = sum(1 for s in train_sentences if keyword.lower() in s.lower())
        marker = " ⚠️" if count < 10 else ""
        print(f"  {keyword:<30} {count:>6}{marker}")
        if count < 10:
            low_coverage.append(keyword)

    if low_coverage:
        print(f"\n  ⚠️  Low coverage keywords (< 10 occurrences):")
        for kw in low_coverage:
            print(f"    - \"{kw}\"")
        print(f"  Consider recording more utterances:")
        print(f"    python record_keywords.py --keyword \"{low_coverage[0]}\" --count 20")


def main():
    parser = argparse.ArgumentParser(
        description="Build HuggingFace dataset for Whisper fine-tuning"
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Analyze keyword coverage in the dataset"
    )
    args = parser.parse_args()

    print("\n📦 WATZS — Dataset Builder")
    print("=" * 60)

    dataset_dict = build_dataset()

    # Always analyze keyword coverage
    analyze_keyword_coverage(dataset_dict)

    print(f"\n{'='*60}")
    print(f"✅ Dataset build complete!")
    print(f"{'='*60}")
    print(f"\nNext step:")
    print(f"  python fine_tune_whisper.py  (start fine-tuning)")


if __name__ == "__main__":
    main()
