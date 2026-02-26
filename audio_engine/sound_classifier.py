"""
Sound Classifier Module
=======================
Classifies audio for abnormal/threat sounds using YAMNet
(a pre-trained TensorFlow model for 521 sound event classes).

Detects sounds like screams, gunshots, explosions, glass breaking,
crashes, and other threat-related audio events.

Usage:
    from audio_engine.sound_classifier import SoundClassifier

    def on_alert(alert):
        print(f"SOUND ALERT: {alert['metadata']['sound_class']}")

    classifier = SoundClassifier()
    classifier.on_alert = on_alert
    # Feed audio from AudioCapture:
    classifier.process_audio(audio_bytes, frame_count)
"""

import os
import uuid
import time
import numpy as np
from datetime import datetime, timezone, timedelta


# ─── Threat Sound Categories ────────────────────────────────────────────
# YAMNet class names that indicate potential threats
THREAT_SOUNDS = {
    # Weapons
    "Gunshot, gunfire": "gunshot",
    "Machine gun": "gunshot",
    "Explosion": "explosion",
    "Burst, pop": "explosion",

    # Human distress
    "Screaming": "scream",
    "Scream": "scream",
    "Crying, sobbing": "distress",
    "Whimper": "distress",
    "Shout": "shout",
    "Yell": "shout",

    # Breaking / impacts
    "Glass": "glass_breaking",
    "Shatter": "glass_breaking",
    "Breaking": "breaking",
    "Crash": "crash",
    "Smash, crash": "crash",
    "Slam": "impact",
    "Thump, thud": "impact",

    # Alarms
    "Alarm": "alarm",
    "Siren": "siren",
    "Fire alarm": "fire_alarm",
    "Smoke detector, smoke alarm": "fire_alarm",

    # Emergency
    "Emergency vehicle": "emergency",
    "Police car (siren)": "emergency",
    "Ambulance (siren)": "emergency",
    "Fire engine, fire truck (siren)": "emergency",
}

# Alert levels for sound categories
SOUND_ALERT_LEVELS = {
    "gunshot": 3,
    "explosion": 3,
    "scream": 2,
    "distress": 2,
    "shout": 1,
    "glass_breaking": 2,
    "breaking": 2,
    "crash": 2,
    "impact": 1,
    "alarm": 2,
    "siren": 2,
    "fire_alarm": 2,
    "emergency": 2,
}

DEFAULT_CONFIDENCE_THRESHOLD = 0.5
CONSECUTIVE_DETECTIONS_REQUIRED = 2  # To reduce false positives


class SoundClassifier:
    """
    Classifies audio chunks using YAMNet for threat sound detection.

    Requires TensorFlow and tensorflow-hub. The YAMNet model is
    downloaded automatically on first use.

    Attributes:
        confidence_threshold (float): Minimum confidence to trigger (0.0–1.0)
        consecutive_required (int): Number of consecutive detections needed
        on_alert (callable): Callback for alert events
    """

    def __init__(self, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
                 consecutive_required=CONSECUTIVE_DETECTIONS_REQUIRED):
        self.confidence_threshold = confidence_threshold
        self.consecutive_required = consecutive_required

        # Callbacks
        self.on_alert = None
        self.on_classification = None  # Called with all classifications

        # State
        self._is_active = True
        self._consecutive_detections = {}  # category → count
        self._last_detection_time = {}     # category → timestamp
        self._model = None
        self._class_names = None
        self._model_loaded = False

        # Lazy-load model
        self._load_model()

    def _load_model(self):
        """Load YAMNet model from TensorFlow Hub."""
        try:
            import tensorflow_hub as hub
            import tensorflow as tf
            import csv

            print("[SoundClassifier] Loading YAMNet model from TensorFlow Hub...")

            # Load the YAMNet model
            self._model = hub.load("https://tfhub.dev/google/yamnet/1")

            # Load class names from the model's asset
            class_map_path = self._model.class_map_path().numpy().decode("utf-8")
            self._class_names = []
            with open(class_map_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self._class_names.append(row["display_name"])

            self._model_loaded = True
            print(f"[SoundClassifier] YAMNet loaded — "
                  f"{len(self._class_names)} sound classes available.")

        except ImportError:
            print("[SoundClassifier] WARNING: TensorFlow not installed!")
            print("  Install with: pip install tensorflow tensorflow-hub")
            print("  Running in MOCK mode — will not classify real audio.")
            self._model_loaded = False

        except Exception as e:
            print(f"[SoundClassifier] WARNING: Failed to load YAMNet: {e}")
            print("  Running in MOCK mode.")
            self._model_loaded = False

    # ─── Audio Processing ───────────────────────────────────────────
    def process_audio(self, data, frame_count=None):
        """
        Classify an audio chunk for threat sounds.

        Designed to be used as an AudioCapture listener:
            capture.add_listener(classifier.process_audio)

        Args:
            data (bytes): Raw audio data (16-bit PCM, mono, 16kHz)
            frame_count (int): Number of frames (unused)
        """
        if not self._is_active:
            return

        if not self._model_loaded:
            return

        try:
            import tensorflow as tf

            # Convert to float32 waveform normalized to [-1, 1]
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            waveform = samples / 32768.0

            # YAMNet expects at least 0.975s of audio at 16kHz
            min_samples = 15600  # ~0.975s at 16kHz
            if len(waveform) < min_samples:
                # Pad with zeros if needed
                waveform = np.pad(waveform, (0, min_samples - len(waveform)))

            # Run YAMNet inference
            scores, embeddings, spectrogram = self._model(waveform)
            scores = scores.numpy()

            # Get top predictions for each time frame
            self._analyze_scores(scores)

        except Exception as e:
            print(f"[SoundClassifier] Classification error: {e}")

    def _analyze_scores(self, scores):
        """Analyze YAMNet scores for threat sounds."""
        now = time.time()

        # Average scores across time frames
        mean_scores = np.mean(scores, axis=0)

        # Find threat sounds above threshold
        detections = []
        for yamnet_class, category in THREAT_SOUNDS.items():
            # Find the index for this class name
            if yamnet_class in self._class_names:
                idx = self._class_names.index(yamnet_class)
                confidence = float(mean_scores[idx])

                if confidence >= self.confidence_threshold:
                    detections.append((category, yamnet_class, confidence))

        # Notify classification callback with top results
        if self.on_classification and len(mean_scores) > 0:
            top_indices = np.argsort(mean_scores)[-5:][::-1]
            top_results = [
                (self._class_names[i], float(mean_scores[i]))
                for i in top_indices
            ]
            self.on_classification(top_results)

        # Process threat detections
        for category, class_name, confidence in detections:
            # Track consecutive detections
            if category in self._last_detection_time:
                time_gap = now - self._last_detection_time[category]
                if time_gap < 2.0:  # Within 2 seconds
                    self._consecutive_detections[category] = \
                        self._consecutive_detections.get(category, 0) + 1
                else:
                    self._consecutive_detections[category] = 1
            else:
                self._consecutive_detections[category] = 1

            self._last_detection_time[category] = now
            consecutive = self._consecutive_detections[category]

            # Only alert if we have enough consecutive detections
            if consecutive >= self.consecutive_required:
                level = SOUND_ALERT_LEVELS.get(category, 2)
                self._emit_alert(
                    category=category,
                    class_name=class_name,
                    level=level,
                    confidence=confidence,
                    consecutive=consecutive
                )
                # Reset to avoid spamming
                self._consecutive_detections[category] = 0

    def _emit_alert(self, category, class_name, level, confidence,
                    consecutive):
        """Emit a sound-based alert event."""
        alert = {
            "id": str(uuid.uuid4()),
            "type": "sound",
            "keyword": category,
            "level": level,
            "timestamp": datetime.now(timezone(timedelta(hours=5, minutes=30)))
                         .isoformat(),
            "confidence": confidence,
            "source": "yamnet",
            "metadata": {
                "sound_class": class_name,
                "category": category,
                "consecutive_detections": consecutive
            }
        }

        level_names = {1: "LOW", 2: "MEDIUM", 3: "HIGH"}
        level_icons = {1: "🟡", 2: "🟠", 3: "🔴"}

        print(f"\n{level_icons.get(level, '⚪')} "
              f"[SOUND ALERT L{level} — {level_names.get(level, '')}] "
              f"\"{class_name}\" → {category} "
              f"(confidence: {confidence:.0%}, "
              f"consecutive: {consecutive})")

        if self.on_alert:
            self.on_alert(alert)

    # ─── Mock Mode ──────────────────────────────────────────────────
    def mock_classify(self, sound_category, confidence=0.9):
        """
        Simulate a sound detection for testing without TensorFlow.

        Args:
            sound_category (str): One of the SOUND_ALERT_LEVELS keys
            confidence (float): Simulated confidence score
        """
        if sound_category not in SOUND_ALERT_LEVELS:
            print(f"[SoundClassifier] Unknown category: {sound_category}")
            print(f"  Valid categories: {list(SOUND_ALERT_LEVELS.keys())}")
            return

        level = SOUND_ALERT_LEVELS[sound_category]
        self._emit_alert(
            category=sound_category,
            class_name=f"Mock: {sound_category}",
            level=level,
            confidence=confidence,
            consecutive=self.consecutive_required
        )

    # ─── Control ────────────────────────────────────────────────────
    def pause(self):
        self._is_active = False
        print("[SoundClassifier] Paused.")

    def resume(self):
        self._is_active = True
        print("[SoundClassifier] Resumed.")

    def reset(self):
        self._consecutive_detections.clear()
        self._last_detection_time.clear()
        print("[SoundClassifier] Reset.")

    @property
    def is_active(self):
        return self._is_active

    @property
    def is_model_loaded(self):
        return self._model_loaded

    @staticmethod
    def get_threat_categories():
        """Get all trackable threat sound categories with alert levels."""
        return dict(SOUND_ALERT_LEVELS)


# ─── CLI Entry Point ───────────────────────────────────────────────────
if __name__ == "__main__":
    import json as json_module

    print("\n🔊 WATZS — Sound Classifier Test")
    print("=" * 50)

    classifier = SoundClassifier()

    def on_alert(alert):
        print(f"\n🚨 Alert: {json_module.dumps(alert, indent=2)}")

    classifier.on_alert = on_alert

    if not classifier.is_model_loaded:
        print("\n⚠️  TensorFlow not available — running mock tests...\n")
        print("Simulating threat sounds:\n")

        for category in ["scream", "gunshot", "glass_breaking", "crash"]:
            print(f"  Testing: {category}")
            classifier.mock_classify(category)
            print()

        print("✅ Mock tests complete.")
    else:
        print("\n✅ YAMNet model loaded. Ready for live classification.")
        print("   Run with AudioCapture for real-time detection.")

    print(f"\n📋 Supported threat categories:")
    for cat, level in sorted(SOUND_ALERT_LEVELS.items(), key=lambda x: x[1]):
        print(f"   L{level}: {cat}")
