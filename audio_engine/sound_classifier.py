"""
Sound Classifier Module
=======================
Classifies audio for abnormal/threat sounds using a custom-trained
classifier built on top of YAMNet embeddings.

Pipeline: Raw Audio → YAMNet (embeddings) → Custom Dense Head → 5-class prediction

Trained classes: scream, gunshot, glass_breaking, crash, normal

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


# ─── Model Configuration ────────────────────────────────────────────────
CLASSES = ["scream", "gunshot", "glass_breaking", "crash", "normal"]
NUM_CLASSES = len(CLASSES)
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "models", "watzs_sound_classifier.h5"
)

# Alert levels for each class
SOUND_ALERT_LEVELS = {
    "scream":         2,   # L2
    "gunshot":        2,   # L2
    "glass_breaking": 2,   # L2
    "crash":          2,   # L2
    "normal":         0,   # No alarm
}

DEFAULT_CONFIDENCE_THRESHOLD = 0.70
CONSECUTIVE_DETECTIONS_REQUIRED = 2  # Reduce false positives


class SoundClassifier:
    """
    Classifies audio chunks using YAMNet embeddings + a custom-trained
    dense classifier head for threat sound detection.

    The custom model was trained on ESC-50 + UrbanSound8K data,
    detecting: scream, gunshot, glass_breaking, crash, normal.

    Attributes:
        confidence_threshold (float): Minimum confidence to trigger (0.0–1.0)
        consecutive_required (int): Number of consecutive detections needed
        on_alert (callable): Callback for alert events
    """

    def __init__(self, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
                 consecutive_required=CONSECUTIVE_DETECTIONS_REQUIRED,
                 model_path=None):
        self.confidence_threshold = confidence_threshold
        self.consecutive_required = consecutive_required

        # Callbacks
        self.on_alert = None
        self.on_classification = None  # Called with all classifications

        # State
        self._is_active = True
        self._consecutive_detections = {}  # category → count
        self._last_detection_time = {}     # category → timestamp
        self._yamnet = None
        self._classifier = None
        self._model_loaded = False
        self._model_path = model_path or MODEL_PATH

        # Lazy-load models
        self._load_model()

    def _load_model(self):
        """Load YAMNet (for embeddings) + custom classifier head."""
        try:
            import tensorflow_hub as hub
            import tensorflow as tf

            # Load YAMNet for embedding extraction
            print("[SoundClassifier] Loading YAMNet for embeddings...")
            self._yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
            print("[SoundClassifier] ✅ YAMNet loaded.")

            # Load custom trained classifier
            resolved_path = os.path.abspath(self._model_path)
            if not os.path.exists(resolved_path):
                print(f"[SoundClassifier] WARNING: Model not found at {resolved_path}")
                print("  Running in MOCK mode.")
                self._model_loaded = False
                return

            print(f"[SoundClassifier] Loading custom classifier from {resolved_path}...")
            self._classifier = tf.keras.models.load_model(resolved_path)
            self._model_loaded = True
            print(f"[SoundClassifier] ✅ Custom model loaded — "
                  f"{NUM_CLASSES} classes: {CLASSES}")

        except ImportError:
            print("[SoundClassifier] WARNING: TensorFlow not installed!")
            print("  Install with: pip install tensorflow tensorflow-hub")
            print("  Running in MOCK mode — will not classify real audio.")
            self._model_loaded = False

        except Exception as e:
            print(f"[SoundClassifier] WARNING: Failed to load models: {e}")
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
        if not self._is_active or not self._model_loaded:
            return

        try:
            import tensorflow as tf

            # Convert bytes to float32 waveform normalized to [-1, 1]
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            waveform = samples / 32768.0

            # YAMNet expects at least 0.975s of audio at 16kHz
            min_samples = 15600  # ~0.975s at 16kHz
            if len(waveform) < min_samples:
                waveform = np.pad(waveform, (0, min_samples - len(waveform)))

            # Extract YAMNet embeddings (1024-dim per time frame)
            _, embeddings, _ = self._yamnet(waveform)
            embedding = embeddings.numpy().mean(axis=0, keepdims=True)

            # Run custom classifier on the embedding
            prediction = self._classifier.predict(embedding, verbose=0)[0]

            self._analyze_prediction(prediction)

        except Exception as e:
            print(f"[SoundClassifier] Classification error: {e}")

    def process_audio_float(self, audio_float, frame_count=None):
        """
        Classify a float32 audio chunk (already normalized to [-1, 1]).

        Use this when audio is captured as float32 (e.g., PyAudio paFloat32).

        Args:
            audio_float (np.ndarray): Float32 audio samples
            frame_count (int): Number of frames (unused)
        """
        if not self._is_active or not self._model_loaded:
            return

        try:
            waveform = np.array(audio_float, dtype=np.float32)

            min_samples = 15600
            if len(waveform) < min_samples:
                waveform = np.pad(waveform, (0, min_samples - len(waveform)))

            _, embeddings, _ = self._yamnet(waveform)
            embedding = embeddings.numpy().mean(axis=0, keepdims=True)

            prediction = self._classifier.predict(embedding, verbose=0)[0]
            self._analyze_prediction(prediction)

        except Exception as e:
            print(f"[SoundClassifier] Classification error: {e}")

    def _analyze_prediction(self, prediction):
        """Analyze custom model prediction for threat sounds."""
        now = time.time()

        idx = int(np.argmax(prediction))
        confidence = float(prediction[idx])
        label = IDX_TO_CLASS[idx]

        # Notify classification callback with all class scores
        if self.on_classification:
            results = [(CLASSES[i], float(prediction[i])) for i in range(NUM_CLASSES)]
            results.sort(key=lambda x: x[1], reverse=True)
            self.on_classification(results)

        # Ignore "normal" and low-confidence detections
        if label == "normal" or confidence < self.confidence_threshold:
            return

        # Track consecutive detections
        if label in self._last_detection_time:
            time_gap = now - self._last_detection_time[label]
            if time_gap < 2.0:  # Within 2 seconds
                self._consecutive_detections[label] = \
                    self._consecutive_detections.get(label, 0) + 1
            else:
                self._consecutive_detections[label] = 1
        else:
            self._consecutive_detections[label] = 1

        self._last_detection_time[label] = now
        consecutive = self._consecutive_detections[label]

        # Only alert if we have enough consecutive detections
        if consecutive >= self.consecutive_required:
            level = SOUND_ALERT_LEVELS.get(label, 2)
            self._emit_alert(
                category=label,
                class_name=label,
                level=level,
                confidence=confidence,
                consecutive=consecutive
            )
            # Reset to avoid spamming
            self._consecutive_detections[label] = 0

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
            "source": "watzs_custom",
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
        return {k: v for k, v in SOUND_ALERT_LEVELS.items() if k != "normal"}


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
        print("\n⚠️  Models not available — running mock tests...\n")
        print("Simulating threat sounds:\n")

        for category in ["scream", "gunshot", "glass_breaking", "crash"]:
            print(f"  Testing: {category}")
            classifier.mock_classify(category)
            print()

        print("✅ Mock tests complete.")
    else:
        print("\n✅ Custom model loaded. Ready for live classification.")
        print("   Run with AudioCapture for real-time detection.")

    print(f"\n📋 Supported threat categories:")
    for cat, level in sorted(
        SoundClassifier.get_threat_categories().items(), key=lambda x: x[1]
    ):
        print(f"   L{level}: {cat}")
