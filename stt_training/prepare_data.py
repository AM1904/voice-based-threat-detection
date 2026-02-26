"""
WATZS STT Data Preparation
===========================
Downloads and prepares Common Voice Indian English data for Whisper fine-tuning.

Steps:
  1. Downloads Common Voice 13.0 (en-IN subset) via HuggingFace datasets
  2. Filters clips by duration (1-30 seconds)
  3. Resamples to 16kHz mono WAV
  4. Cleans transcripts (lowercase, remove punctuation)
  5. Splits into train/val sets (90/10)

Usage:
    python prepare_data.py
    python prepare_data.py --max-samples 500   # Limit for quick testing
"""

import os
import sys
import re
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_data_dirs():
    """Create and return data directory paths."""
    base = Path(__file__).parent / "data"
    dirs = {
        "raw": base / "raw",
        "custom": base / "custom",
        "augmented": base / "augmented",
        "train": base / "train",
        "val": base / "val",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def clean_transcript(text):
    """
    Clean transcript for STT training.
    - Lowercase
    - Remove punctuation (keep apostrophes in contractions)
    - Collapse whitespace
    """
    text = text.lower().strip()
    # Keep letters, digits, spaces, and apostrophes
    text = re.sub(r"[^a-z0-9\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def download_speech_data(max_samples=None):
    """
    Download English speech data for Whisper fine-tuning.

    Tries multiple sources in order of preference:
      1. Google FLEURS (Indian English subset — best for Indian accents)
      2. Community mirror of Common Voice 22.0
      3. LibriSpeech clean-100 (fallback, US English)

    Returns a HuggingFace Dataset with 'audio' and 'sentence' columns.
    """
    try:
        from datasets import load_dataset, Audio
    except ImportError:
        print("ERROR: 'datasets' library not installed.")
        print("Run: pip install datasets")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Step 1: Downloading English speech data")
    print("=" * 60)
    print("This may take a while on the first run...")
    print("The dataset is cached for future runs.\n")

    dataset = None
    transcript_col = "sentence"  # Column name varies by dataset

    # ── Attempt 1: Google FLEURS (Indian English) ───────────────
    try:
        print("  Trying Google FLEURS (English)...")
        dataset = load_dataset(
            "google/fleurs",
            "en_us",
            split="train",
            trust_remote_code=True,
        )
        transcript_col = "transcription"
        print(f"  ✅ Downloaded {len(dataset)} samples from Google FLEURS (en_us)")
    except Exception as e:
        print(f"  ⚠️  FLEURS not available: {e}")

    # ── Attempt 2: Community Common Voice mirror ────────────────
    if dataset is None:
        try:
            print("  Trying Common Voice 22.0 community mirror...")
            dataset = load_dataset(
                "fsicoli/common_voice_22_0",
                "en",
                split="train",
                trust_remote_code=True,
            )
            transcript_col = "sentence"
            print(f"  ✅ Downloaded {len(dataset)} samples from Common Voice 22.0 mirror")
        except Exception as e:
            print(f"  ⚠️  Common Voice mirror not available: {e}")

    # ── Attempt 3: LibriSpeech clean-100 ────────────────────────
    if dataset is None:
        try:
            print("  Trying LibriSpeech clean-100...")
            dataset = load_dataset(
                "librispeech_asr",
                "clean",
                split="train.100",
                trust_remote_code=True,
            )
            transcript_col = "text"
            print(f"  ✅ Downloaded {len(dataset)} samples from LibriSpeech clean-100")
        except Exception as e:
            print(f"  ⚠️  LibriSpeech not available: {e}")

    if dataset is None:
        print("\n  ❌ Could not download any speech dataset.")
        print("  Manual alternative:")
        print("  1. Download audio from https://commonvoice.mozilla.org/en/datasets")
        print("  2. Extract WAV files to: stt_training/data/raw/")
        print("  3. Create a manifest CSV with columns: path, sentence")
        sys.exit(1)

    # Normalise the transcript column name to 'sentence'
    if transcript_col != "sentence":
        dataset = dataset.rename_column(transcript_col, "sentence")

    # Resample audio to 16kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        print(f"  Limited to {max_samples} samples for testing.")

    return dataset


def filter_and_clean(dataset):
    """
    Filter clips by duration and clean transcripts.
    - Keep clips between 1 and 30 seconds
    - Remove clips with empty transcripts
    - Clean transcript text
    """
    print("\n" + "=" * 60)
    print("Step 2: Filtering and cleaning data")
    print("=" * 60)

    original_count = len(dataset)

    def is_valid(example):
        audio = example["audio"]
        duration = len(audio["array"]) / audio["sampling_rate"]
        sentence = example.get("sentence", "").strip()
        return 1.0 <= duration <= 30.0 and len(sentence) > 0

    dataset = dataset.filter(is_valid, desc="Filtering by duration")

    def clean_example(example):
        example["sentence"] = clean_transcript(example["sentence"])
        return example

    dataset = dataset.map(clean_example, desc="Cleaning transcripts")

    # Remove empty transcripts after cleaning
    dataset = dataset.filter(
        lambda x: len(x["sentence"].strip()) > 0,
        desc="Removing empty transcripts"
    )

    kept = len(dataset)
    print(f"  Original: {original_count} clips")
    print(f"  After filtering: {kept} clips ({kept/original_count*100:.1f}%)")

    return dataset


def save_to_wav(dataset, output_dir, split_name="train"):
    """
    Save dataset audio to individual WAV files and create a manifest CSV.
    """
    import soundfile as sf
    import csv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"

    print(f"\n  Saving {len(dataset)} clips to {output_dir}/...")

    rows = []
    for i, example in enumerate(dataset):
        audio = example["audio"]
        wav_path = output_dir / f"{split_name}_{i:06d}.wav"

        sf.write(
            str(wav_path),
            audio["array"],
            audio["sampling_rate"],
            subtype="PCM_16"
        )

        rows.append({
            "path": str(wav_path.resolve()),
            "sentence": example["sentence"],
        })

        if (i + 1) % 500 == 0:
            print(f"    Saved {i+1}/{len(dataset)} clips...")

    # Write manifest CSV
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "sentence"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved {len(rows)} clips and manifest to {manifest_path}")
    return manifest_path


def split_and_save(dataset, dirs):
    """
    Split dataset into train/val (90/10) and save as WAV + manifest CSV.
    """
    print("\n" + "=" * 60)
    print("Step 3: Splitting into train/val and saving WAV files")
    print("=" * 60)

    # Split 90/10
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    val_ds = split["test"]

    print(f"  Train: {len(train_ds)} clips")
    print(f"  Val:   {len(val_ds)} clips")

    train_manifest = save_to_wav(train_ds, dirs["train"], split_name="train")
    val_manifest = save_to_wav(val_ds, dirs["val"], split_name="val")

    return train_manifest, val_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare Common Voice data for Whisper fine-tuning"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of samples (for quick testing)"
    )
    args = parser.parse_args()

    print("\n🎙️  WATZS — STT Data Preparation")
    print("=" * 60)

    dirs = get_data_dirs()

    # Check if data already exists
    train_manifest = dirs["train"] / "manifest.csv"
    if train_manifest.exists():
        print(f"\n⚠️  Training data already exists at {dirs['train']}")
        response = input("   Overwrite? (y/N): ").strip().lower()
        if response != "y":
            print("   Skipping download. Run augment_data.py next.")
            return

    # Download
    dataset = download_speech_data(max_samples=args.max_samples)

    # Filter and clean
    dataset = filter_and_clean(dataset)

    # Split and save
    train_manifest, val_manifest = split_and_save(dataset, dirs)

    print("\n" + "=" * 60)
    print("✅ Data preparation complete!")
    print("=" * 60)
    print(f"  Train manifest: {train_manifest}")
    print(f"  Val manifest:   {val_manifest}")
    print(f"\nNext steps:")
    print(f"  1. Record custom keywords: python record_keywords.py")
    print(f"  2. Augment data:           python augment_data.py")
    print(f"  3. Build final dataset:    python build_manifest.py")


if __name__ == "__main__":
    main()
