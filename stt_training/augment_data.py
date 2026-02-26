"""
WATZS Audio Data Augmentation
==============================
Augments existing audio data to improve model robustness.
Creates multiple variations of each clip using:
  - Background noise mixing (babble, street, ambient)
  - Pitch shifting (+/- 2 semitones)
  - Time stretching (0.9x and 1.1x speed)
  - Room reverb simulation
  - Gain variation

Multiplies your dataset by 5-8x without needing more recordings.

Usage:
    python augment_data.py                   # Augment all data
    python augment_data.py --multiplier 3    # 3 augmented copies per clip
    python augment_data.py --source custom   # Only augment custom recordings
"""

import os
import sys
import csv
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    print("ERROR: Required libraries not installed.")
    print("Run: pip install numpy soundfile")
    sys.exit(1)


def check_audiomentations():
    """Check if audiomentations is available and import it."""
    try:
        import audiomentations as am
        return am
    except ImportError:
        print("ERROR: 'audiomentations' library not installed.")
        print("Run: pip install audiomentations")
        sys.exit(1)


def create_augmentation_pipeline(am):
    """
    Create the augmentation pipeline using audiomentations.

    Each augmentation is applied with a probability, so not every
    augment is applied to every clip — creating natural variation.
    """
    augment = am.Compose([
        # Add background noise at various SNR levels
        am.AddGaussianNoise(
            min_amplitude=0.001,
            max_amplitude=0.015,
            p=0.5,
        ),

        # Pitch shift: +/- 2 semitones
        am.PitchShift(
            min_semitones=-2.0,
            max_semitones=2.0,
            p=0.4,
        ),

        # Time stretch: 0.9x to 1.1x speed
        am.TimeStretch(
            min_rate=0.9,
            max_rate=1.1,
            p=0.4,
        ),

        # Random gain: volume variation
        am.Gain(
            min_gain_db=-6.0,
            max_gain_db=6.0,
            p=0.5,
        ),

        # Low-pass filter (simulates distance from mic)
        am.LowPassFilter(
            min_cutoff_freq=2000,
            max_cutoff_freq=7500,
            p=0.2,
        ),

        # Band-pass filter (simulates telephone/radio quality)
        am.BandPassFilter(
            min_center_freq=200,
            max_center_freq=4000,
            p=0.1,
        ),

        # Clipping distortion (simulates overdriven mic)
        am.ClippingDistortion(
            min_percentile_threshold=0,
            max_percentile_threshold=10,
            p=0.1,
        ),
    ])

    return augment


def create_heavy_noise_pipeline(am):
    """
    A heavier augmentation for simulating noisy environments.
    Applied to a subset of clips.
    """
    augment = am.Compose([
        am.AddGaussianNoise(
            min_amplitude=0.01,
            max_amplitude=0.05,
            p=0.8,
        ),
        am.Gain(
            min_gain_db=-10.0,
            max_gain_db=3.0,
            p=0.6,
        ),
        am.LowPassFilter(
            min_cutoff_freq=1500,
            max_cutoff_freq=4000,
            p=0.5,
        ),
    ])

    return augment


def load_manifest(manifest_path):
    """Load manifest CSV into list of dicts."""
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def augment_clip(audio, sample_rate, augment_pipeline, seed=None):
    """
    Apply augmentation to a single audio clip.

    Args:
        audio: numpy array of audio samples
        sample_rate: sample rate in Hz
        augment_pipeline: audiomentations Compose pipeline
        seed: random seed for reproducibility

    Returns:
        augmented audio as numpy array
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensure float32 for audiomentations
    audio = audio.astype(np.float32)

    # Apply augmentation
    augmented = augment_pipeline(samples=audio, sample_rate=sample_rate)

    # Clip to [-1, 1] range
    augmented = np.clip(augmented, -1.0, 1.0)

    return augmented


def augment_dataset(source_dir, output_dir, multiplier=5):
    """
    Augment all clips in a directory.

    Args:
        source_dir: directory containing WAV files and manifest.csv
        output_dir: directory to save augmented clips
        multiplier: number of augmented copies per original clip
    """
    am = check_audiomentations()

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = source_dir / "manifest.csv"
    if not manifest_path.exists():
        print(f"  ⚠️  No manifest.csv found in {source_dir}, skipping.")
        return []

    rows = load_manifest(manifest_path)
    if not rows:
        print(f"  ⚠️  Empty manifest in {source_dir}, skipping.")
        return []

    print(f"\n  Source: {source_dir} ({len(rows)} clips)")
    print(f"  Output: {output_dir}")
    print(f"  Multiplier: {multiplier}x")
    print(f"  Will create: {len(rows) * multiplier} augmented clips\n")

    # Create pipelines
    standard_pipeline = create_augmentation_pipeline(am)
    noisy_pipeline = create_heavy_noise_pipeline(am)

    augmented_rows = []
    errors = 0

    for i, row in enumerate(rows):
        wav_path = Path(row["path"])
        if not wav_path.exists():
            errors += 1
            continue

        try:
            audio, sr = sf.read(str(wav_path), dtype="float32")
        except Exception as e:
            print(f"  ⚠️  Error reading {wav_path.name}: {e}")
            errors += 1
            continue

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        for j in range(multiplier):
            # Use heavy noise pipeline for ~20% of augmentations
            pipeline = noisy_pipeline if j % 5 == 4 else standard_pipeline
            seed = i * multiplier + j

            augmented = augment_clip(audio, sr, pipeline, seed=seed)

            # Save augmented clip
            stem = wav_path.stem
            out_filename = f"{stem}_aug{j:02d}.wav"
            out_path = output_dir / out_filename

            sf.write(str(out_path), augmented, sr, subtype="PCM_16")

            augmented_rows.append({
                "path": str(out_path.resolve()),
                "sentence": row["sentence"],
            })

        if (i + 1) % 100 == 0:
            print(f"  Augmented {i+1}/{len(rows)} clips "
                  f"({len(augmented_rows)} total generated)...")

    # Save augmented manifest
    aug_manifest = output_dir / "manifest.csv"
    with open(aug_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "sentence"])
        writer.writeheader()
        writer.writerows(augmented_rows)

    if errors:
        print(f"  ⚠️  {errors} clips had errors and were skipped")

    return augmented_rows


def main():
    parser = argparse.ArgumentParser(
        description="Augment audio data for Whisper fine-tuning"
    )
    parser.add_argument(
        "--source", type=str, default="all",
        choices=["all", "train", "custom"],
        help="Which data to augment (default: all)"
    )
    parser.add_argument(
        "--multiplier", type=int, default=5,
        help="Number of augmented copies per clip (default: 5)"
    )
    args = parser.parse_args()

    print("\n🔊 WATZS — Audio Data Augmentation")
    print("=" * 60)

    data_dir = Path(__file__).parent / "data"
    output_dir = data_dir / "augmented"

    sources = []
    if args.source in ("all", "train"):
        train_dir = data_dir / "train"
        if train_dir.exists() and (train_dir / "manifest.csv").exists():
            sources.append(("train", train_dir))
    if args.source in ("all", "custom"):
        custom_dir = data_dir / "custom"
        if custom_dir.exists() and (custom_dir / "manifest.csv").exists():
            sources.append(("custom", custom_dir))

    if not sources:
        print("\n  ❌ No source data found. Run these first:")
        print("     python prepare_data.py    (download Common Voice)")
        print("     python record_keywords.py (record your keywords)")
        sys.exit(1)

    total_augmented = []
    for name, src_dir in sources:
        print(f"\n{'='*60}")
        print(f"  Augmenting: {name}")
        print(f"{'='*60}")

        aug_subdir = output_dir / name
        rows = augment_dataset(src_dir, aug_subdir, multiplier=args.multiplier)
        total_augmented.extend(rows)

    # Merge all augmented manifests
    merged_manifest = output_dir / "manifest.csv"
    with open(merged_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "sentence"])
        writer.writeheader()
        writer.writerows(total_augmented)

    print(f"\n{'='*60}")
    print(f"✅ Augmentation complete!")
    print(f"{'='*60}")
    print(f"  Total augmented clips: {len(total_augmented)}")
    print(f"  Merged manifest: {merged_manifest}")
    print(f"\nNext step:")
    print(f"  python build_manifest.py  (create final HuggingFace dataset)")


if __name__ == "__main__":
    main()
