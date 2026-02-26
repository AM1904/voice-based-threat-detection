"""
WATZS Keyword Recording Utility
================================
Interactive CLI tool to record yourself saying sentences containing
each WATZS threat keyword. These custom recordings dramatically
improve fine-tuned Whisper accuracy on your specific keywords and accent.

Target: 50-100 utterances per keyword for best results.

Usage:
    python record_keywords.py                # Record all keywords
    python record_keywords.py --keyword gun  # Record a specific keyword
    python record_keywords.py --count 10     # 10 utterances per keyword
"""

import os
import sys
import json
import time
import argparse
import csv
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import sounddevice as sd
    import numpy as np
    import soundfile as sf
except ImportError:
    print("ERROR: Required libraries not installed.")
    print("Run: pip install sounddevice numpy soundfile")
    sys.exit(1)


SAMPLE_RATE = 16000
CHANNELS = 1
KEYWORDS_PATH = PROJECT_ROOT / "config" / "keywords.json"


def load_keywords():
    """Load all keywords from keywords.json."""
    with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    keywords = []
    levels = config.get("levels", {})
    for level_key in ["L1", "L2", "L3"]:
        level_data = levels.get(level_key, {})
        for kw in level_data.get("keywords", []):
            keywords.append({"keyword": kw, "level": level_key})

    # Add secret code phrase
    secret = config.get("secret_code", {})
    if secret.get("phrase"):
        keywords.append({
            "keyword": secret["phrase"],
            "level": "SECRET"
        })

    return keywords


def generate_prompts(keyword, count=5):
    """
    Generate sentence prompts that naturally contain the keyword.
    These guide you on what to say while recording.
    """
    templates = [
        "I heard someone say {kw}",
        "There is a {kw} situation here",
        "Please {kw} right now",
        "{kw}",
        "Someone is shouting {kw}",
        "I think there is {kw}",
        "We need to report {kw}",
        "Did you hear that {kw}",
        "{kw} {kw}",
        "Alert there is {kw} happening",
        "Quick call for {kw}",
        "Somebody said {kw} over there",
        "Can you hear the {kw}",
        "There was a loud {kw}",
        "The situation involves {kw}",
    ]
    prompts = []
    for i in range(count):
        template = templates[i % len(templates)]
        sentence = template.format(kw=keyword)
        prompts.append(sentence)
    return prompts


def record_audio(duration_seconds=4.0):
    """
    Record audio from the default microphone.

    Returns:
        numpy array of audio samples (float32, 16kHz, mono)
    """
    print(f"    🔴 Recording ({duration_seconds}s)...", end="", flush=True)

    audio = sd.rec(
        int(duration_seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.float32,
    )
    sd.wait()  # Wait until recording is complete

    # Trim silence from end
    audio = audio.flatten()
    rms_threshold = 0.01
    # Find last non-silent sample
    for end_idx in range(len(audio) - 1, 0, -1):
        window = audio[max(0, end_idx - 1600):end_idx]
        if np.sqrt(np.mean(window ** 2)) > rms_threshold:
            break
    audio = audio[:end_idx + 1600]  # Keep a small tail

    print(f" Done ({len(audio)/SAMPLE_RATE:.1f}s captured)")
    return audio


def save_recording(audio, output_path):
    """Save audio to 16kHz mono WAV file."""
    sf.write(str(output_path), audio, SAMPLE_RATE, subtype="PCM_16")


def record_keyword_set(keyword_info, count, output_dir, existing_count=0):
    """
    Record multiple utterances of a single keyword.

    Args:
        keyword_info: dict with 'keyword' and 'level'
        count: number of utterances to record
        output_dir: where to save WAV files
        existing_count: number of existing recordings (for numbering)

    Returns:
        list of (wav_path, transcript) tuples
    """
    keyword = keyword_info["keyword"]
    level = keyword_info["level"]
    prompts = generate_prompts(keyword, count)
    recordings = []

    print(f"\n{'='*60}")
    print(f"  Keyword: \"{keyword}\" ({level})")
    print(f"  Recording {count} utterances")
    print(f"{'='*60}")
    print(f"  Tips:")
    print(f"  - Vary your distance from the mic (close, arm's length, far)")
    print(f"  - Vary your volume (whisper, normal, loud)")
    print(f"  - Vary your speed (slow, normal, fast)")
    print(f"  - Try with some background noise too")
    print()

    for i, prompt in enumerate(prompts):
        idx = existing_count + i
        print(f"  [{i+1}/{count}] Say: \"{prompt}\"")

        input("    Press Enter when ready to record...")

        audio = record_audio(duration_seconds=5.0)

        # Check if audio has content
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.005:
            print("    ⚠️  Very quiet recording. Redo? (y/N): ", end="")
            if input().strip().lower() == "y":
                audio = record_audio(duration_seconds=5.0)

        # Save
        safe_kw = keyword.replace(" ", "_").replace("'", "")
        filename = f"kw_{safe_kw}_{idx:04d}.wav"
        wav_path = output_dir / filename
        save_recording(audio, wav_path)

        transcript = prompt.lower().strip()
        recordings.append((str(wav_path.resolve()), transcript))
        print(f"    ✅ Saved: {filename}")

    return recordings


def main():
    parser = argparse.ArgumentParser(
        description="Record keyword utterances for Whisper fine-tuning"
    )
    parser.add_argument(
        "--keyword", type=str, default=None,
        help="Record only this specific keyword"
    )
    parser.add_argument(
        "--count", type=int, default=5,
        help="Number of utterances per keyword (default: 5, recommended: 50-100)"
    )
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="Max recording duration in seconds (default: 5.0)"
    )
    args = parser.parse_args()

    print("\n🎙️  WATZS — Keyword Recording Utility")
    print("=" * 60)

    # Setup output directory
    output_dir = Path(__file__).parent / "data" / "custom"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"

    # Load existing manifest if any
    existing_recordings = []
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_recordings = list(reader)
        print(f"  Found {len(existing_recordings)} existing recordings")

    # Load keywords
    keywords = load_keywords()
    print(f"  Loaded {len(keywords)} keywords from keywords.json")

    # Filter to specific keyword if requested
    if args.keyword:
        keywords = [k for k in keywords if k["keyword"].lower() == args.keyword.lower()]
        if not keywords:
            print(f"\n  ERROR: Keyword '{args.keyword}' not found in keywords.json")
            sys.exit(1)

    print(f"\n  Will record {args.count} utterances for {len(keywords)} keywords")
    print(f"  Total recordings: {len(keywords) * args.count}")
    print(f"  Output: {output_dir}/")

    # Test microphone
    print(f"\n  Testing microphone...")
    try:
        test = sd.rec(int(0.5 * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                      channels=CHANNELS, dtype=np.float32)
        sd.wait()
        rms = np.sqrt(np.mean(test ** 2))
        print(f"  ✅ Microphone working (RMS: {rms:.4f})")
    except Exception as e:
        print(f"  ❌ Microphone error: {e}")
        sys.exit(1)

    input("\n  Press Enter to start recording...\n")

    # Record each keyword (skip already-completed ones)
    all_recordings = list(existing_recordings)
    new_count = 0
    skipped = 0

    for keyword_info in keywords:
        # Count existing recordings for this keyword
        kw_lower = keyword_info["keyword"].lower()
        existing_for_kw = len([
            r for r in existing_recordings
            if kw_lower in r.get("sentence", "").lower()
        ])

        # Skip if already have enough recordings
        if existing_for_kw >= args.count:
            print(f"\n  ⏭️  Skipping \"{keyword_info['keyword']}\" "
                  f"— already has {existing_for_kw} recordings")
            skipped += 1
            continue

        remaining = args.count - existing_for_kw
        if existing_for_kw > 0:
            print(f"\n  ↩️  Resuming \"{keyword_info['keyword']}\" "
                  f"— {existing_for_kw} done, {remaining} remaining")

        recordings = record_keyword_set(
            keyword_info,
            count=remaining,
            output_dir=output_dir,
            existing_count=existing_for_kw,
        )

        for wav_path, transcript in recordings:
            all_recordings.append({
                "path": wav_path,
                "sentence": transcript,
            })
            new_count += 1

    # Save manifest
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "sentence"])
        writer.writeheader()
        writer.writerows(all_recordings)

    print(f"\n{'='*60}")
    print(f"✅ Recording complete!")
    print(f"{'='*60}")
    print(f"  New recordings: {new_count}")
    print(f"  Total recordings: {len(all_recordings)}")
    print(f"  Manifest: {manifest_path}")
    print(f"\nNext steps:")
    print(f"  1. Record more: python record_keywords.py --count 10")
    print(f"  2. Augment data: python augment_data.py")


if __name__ == "__main__":
    main()
