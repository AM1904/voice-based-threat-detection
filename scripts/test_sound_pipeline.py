"""
Sound Classifier Pipeline Diagnostic Test
==========================================
Tests the full classification pipeline with known audio samples to identify
where detection is breaking (RMS gate, YAMNet speech filter, classifier).

Usage:
    python scripts/test_sound_pipeline.py

Requires: tensorflow, tensorflow_hub, numpy, datasets (>=3.6.0)
"""

import os
import sys
import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_engine.sound_classifier import (
    SoundClassifier,
    CLASSES,
    IDX_TO_CLASS,
    NUM_CLASSES,
    RMS_NOISE_GATE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    CONFIDENCE_PER_CLASS,
    YAMNET_SPEECH_INDICES,
    YAMNET_SPEECH_THRESHOLD,
)


def load_esc50_samples():
    """Load a few ESC-50 samples for each target class."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets>=3.6.0")
        sys.exit(1)

    ds = load_dataset("ashraq/esc50", split="train", trust_remote_code=True)

    # ESC-50 categories we use for testing (string names)
    target_map = {
        "glass_breaking": "glass_breaking",
        "fireworks": "fireworks",       # proxy for gunshot
        "crying_baby": "crying_baby",   # proxy for scream
        "engine": "engine",             # normal sound
        "laughing": "laughing",         # normal sound (speech-adjacent)
    }

    results = {}
    for row in ds:
        cat = row["category"]
        if cat in target_map:
            label = target_map[cat]
            if label not in results:
                results[label] = []
            if len(results[label]) < 3:  # 3 samples per class
                audio = row["audio"]
                waveform = np.array(audio["array"], dtype=np.float32)
                sr = audio["sampling_rate"]
                results[label].append((waveform, sr))

        # Stop early if we have enough for all target classes
        if (len(results) == len(target_map) and
                all(len(v) >= 3 for v in results.values())):
            break

    return results


def analyze_audio(classifier, waveform, sr, label, idx):
    """Run a single audio sample through the full pipeline and report."""
    import tensorflow_hub as hub

    # Resample to 16kHz if needed
    if sr != 16000:
        from scipy.signal import resample
        num_samples = int(len(waveform) * 16000 / sr)
        waveform = resample(waveform, num_samples).astype(np.float32)

    # Ensure at least 0.975s
    min_samples = 15600
    if len(waveform) < min_samples:
        waveform = np.pad(waveform, (0, min_samples - len(waveform)))

    # Simulate the int16 path (process_audio) for RMS check
    int16_data = (waveform * 32768.0).clip(-32768, 32767).astype(np.int16)
    rms = float(np.sqrt(np.mean(int16_data.astype(np.float64) ** 2)))
    rms_pass = rms >= RMS_NOISE_GATE

    print(f"\n--- [{label}] Sample {idx+1} ---")
    print(f"  Duration: {len(waveform)/16000:.2f}s | RMS(int16): {rms:.0f} "
          f"{'PASS' if rms_pass else 'BLOCKED'} (gate={RMS_NOISE_GATE})")

    if not rms_pass:
        print(f"  ** Would be SKIPPED by RMS noise gate **")
        # Still run classifier for diagnostic
        print(f"  (Running classifier anyway for diagnostic...)")

    # YAMNet analysis
    yamnet = classifier._yamnet
    scores, embeddings, _ = yamnet(waveform)
    scores_np = scores.numpy()
    mean_scores = scores_np.mean(axis=0)

    top_yamnet_idx = int(mean_scores.argmax())
    top_yamnet_score = float(mean_scores.max())
    is_speech = top_yamnet_idx in YAMNET_SPEECH_INDICES
    speech_blocked = is_speech and top_yamnet_score >= YAMNET_SPEECH_THRESHOLD

    # Top 5 YAMNet classes
    top5_idx = mean_scores.argsort()[-5:][::-1]
    print(f"  YAMNet top-5:")
    for i in top5_idx:
        marker = " <-- SPEECH" if int(i) in YAMNET_SPEECH_INDICES else ""
        print(f"    class {int(i):3d}: {float(mean_scores[i]):.4f}{marker}")

    print(f"  Speech filter: top_idx={top_yamnet_idx} is_speech={is_speech} "
          f"score={top_yamnet_score:.3f} thresh={YAMNET_SPEECH_THRESHOLD} "
          f"-> {'BLOCKED' if speech_blocked else 'PASS'}")

    if speech_blocked:
        print(f"  ** Would be SKIPPED by speech pre-filter **")
        # Still run classifier for diagnostic

    # Custom classifier
    embedding = embeddings.numpy().mean(axis=0, keepdims=True)
    prediction = classifier._classifier.predict(embedding, verbose=0)[0]

    print(f"  Custom classifier predictions:")
    for i in range(NUM_CLASSES):
        class_name = CLASSES[i]
        class_thresh = CONFIDENCE_PER_CLASS.get(class_name, DEFAULT_CONFIDENCE_THRESHOLD)
        marker = ""
        if i == int(np.argmax(prediction)):
            marker = " <-- TOP"
        if float(prediction[i]) >= class_thresh:
            marker += f" (above thresh {class_thresh})"
        print(f"    {class_name:15s}: {float(prediction[i]):.4f}{marker}")

    top_class = IDX_TO_CLASS[int(np.argmax(prediction))]
    top_conf = float(prediction[int(np.argmax(prediction))])
    effective_thresh = CONFIDENCE_PER_CLASS.get(top_class, DEFAULT_CONFIDENCE_THRESHOLD)
    would_alert = (top_class != "normal" and
                   top_conf >= effective_thresh and
                   rms_pass and
                   not speech_blocked)

    verdict = "ALERT" if would_alert else "SILENT"
    reasons = []
    if not rms_pass:
        reasons.append("RMS too low")
    if speech_blocked:
        reasons.append("speech filter")
    if top_class == "normal":
        reasons.append("classified as normal")
    if top_conf < effective_thresh:
        reasons.append(f"confidence {top_conf:.2f} < {effective_thresh} ({top_class} thresh)")

    print(f"  VERDICT: {verdict} (predicted={top_class} @ {top_conf:.1%})"
          + (f" -- blocked by: {', '.join(reasons)}" if reasons else ""))

    return {
        "label": label,
        "predicted": top_class,
        "confidence": top_conf,
        "rms": rms,
        "rms_pass": rms_pass,
        "speech_blocked": speech_blocked,
        "would_alert": would_alert,
    }


def main():
    print("=" * 60)
    print("Sound Classifier Pipeline Diagnostic Test")
    print("=" * 60)

    print("\nLoading sound classifier...")
    sc = SoundClassifier()
    print("Classifier loaded.\n")

    print("Loading ESC-50 test samples...")
    samples = load_esc50_samples()
    print(f"Loaded: {', '.join(f'{k}({len(v)})' for k, v in samples.items())}")

    all_results = []
    for label, audio_list in samples.items():
        for idx, (waveform, sr) in enumerate(audio_list):
            result = analyze_audio(sc, waveform, sr, label, idx)
            all_results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for label in ["glass_breaking", "fireworks", "crying_baby", "engine", "laughing"]:
        group = [r for r in all_results if r["label"] == label]
        if not group:
            continue
        alerts = sum(1 for r in group if r["would_alert"])
        blocked_rms = sum(1 for r in group if not r["rms_pass"])
        blocked_speech = sum(1 for r in group if r["speech_blocked"])
        predicted_normal = sum(1 for r in group if r["predicted"] == "normal")
        avg_conf = np.mean([r["confidence"] for r in group])
        top_preds = [r["predicted"] for r in group]

        print(f"\n  {label}:")
        print(f"    Alerts: {alerts}/{len(group)} | Avg conf: {avg_conf:.1%}")
        print(f"    Blocked by RMS: {blocked_rms} | Blocked by speech: {blocked_speech}")
        print(f"    Predicted normal: {predicted_normal}")
        print(f"    Predictions: {top_preds}")


if __name__ == "__main__":
    main()
