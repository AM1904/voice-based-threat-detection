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

DEFAULT_CONFIDENCE_THRESHOLD = 0.65  # Base threshold for scream/gunshot
CONSECUTIVE_DETECTIONS_REQUIRED = 1  # Default for instant events (gunshot, glass)
RMS_NOISE_GATE = 200  # Lowered from 500: speaker playback is quieter than real events
DIAGNOSTIC_LOGGING = False  # Set True for per-chunk diagnostic output

# Per-class confidence thresholds — tuned from v2 model confusion matrix:
#   normal->crash was 8.7% false positive, normal->glass_breaking was 6.0%
#   Raising thresholds for those classes eliminates the false positives
#   while keeping real detections (crash val avg 0.665, glass_breaking val avg 0.556)
CONFIDENCE_PER_CLASS = {
    "scream": 0.65,
    "gunshot": 0.65,
    "glass_breaking": 0.66,  # Tuned: real glass avg ~0.67, false positives ~0.60
    "crash": 0.80,           # Raised: v2 model confuses normal->crash at ~0.6
}

# Per-class consecutive detection overrides
# Instant events (gunshot, glass_breaking) = 1 detection is enough
# Sustained events (scream, crash) = need 2+ to confirm, reduces false positives
CONSECUTIVE_PER_CLASS = {
    "scream": 2,
    "gunshot": 1,
    "glass_breaking": 2,  # Raised from 1: double-check reduces false positives
    "crash": 3,            # Raised from 2: crash is the worst false positive class
}

# YAMNet class indices for speech (non-threat) - used as pre-filter
# If YAMNet thinks it's normal speech, skip the custom classifier to avoid
# false positives where speech gets misclassified as screaming.
YAMNET_SPEECH_INDICES = {0, 1, 2, 3, 5, 65}  # Speech, Child speech, Conversation, Narration, Speech synth, Babble
YAMNET_SPEECH_THRESHOLD = 0.3  # Min YAMNet speech score to trigger the filter


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
            print("[SoundClassifier] [OK] YAMNet loaded.")

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
            print(f"[SoundClassifier] [OK] Custom model loaded - "
                  f"{NUM_CLASSES} classes: {CLASSES}")

        except (ImportError, RuntimeError, Exception) as e:
            print(f"[SoundClassifier] WARNING: TensorFlow not available or failed to load: {e}")
            print("  Sound classification will be disabled (MOCK mode).")
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
            # ... rest of the method (rest of the code is unchanged because the try/except block is already here)

            # Convert bytes to float32 waveform normalized to [-1, 1]
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)

            # RMS noise gate: skip quiet audio to avoid false positives
            rms = np.sqrt(np.mean(samples ** 2))
            if rms < RMS_NOISE_GATE:
                if DIAGNOSTIC_LOGGING:
                    print(f"[SoundDiag] RMS {rms:.0f} < gate {RMS_NOISE_GATE} -> SKIP (too quiet)")
                return

            if DIAGNOSTIC_LOGGING:
                print(f"[SoundDiag] RMS {rms:.0f} (gate={RMS_NOISE_GATE}) -> PASS, classifying...")

            waveform = samples / 32768.0

            # YAMNet expects at least 0.975s of audio at 16kHz
            min_samples = 15600  # ~0.975s at 16kHz
            if len(waveform) < min_samples:
                waveform = np.pad(waveform, (0, min_samples - len(waveform)))

            # Extract YAMNet scores and embeddings
            scores, embeddings, _ = self._yamnet(waveform)

            # Speech pre-filter: if YAMNet's top class is normal speech,
            # skip the custom classifier to avoid false scream detections
            scores_np = scores.numpy()
            mean_scores = scores_np.mean(axis=0)
            top_yamnet_idx = int(mean_scores.argmax())
            top_yamnet_score = float(mean_scores.max())

            if DIAGNOSTIC_LOGGING:
                # Show top 3 YAMNet classes for debugging
                top3_idx = mean_scores.argsort()[-3:][::-1]
                top3_info = ", ".join(
                    f"cls{int(i)}={float(mean_scores[i]):.3f}"
                    for i in top3_idx
                )
                is_speech = top_yamnet_idx in YAMNET_SPEECH_INDICES
                print(f"[SoundDiag] YAMNet top3: [{top3_info}] | "
                      f"top_idx={top_yamnet_idx} is_speech={is_speech} "
                      f"score={top_yamnet_score:.3f} thresh={YAMNET_SPEECH_THRESHOLD}")

            if top_yamnet_idx in YAMNET_SPEECH_INDICES and top_yamnet_score >= YAMNET_SPEECH_THRESHOLD:
                if DIAGNOSTIC_LOGGING:
                    print(f"[SoundDiag] -> SKIP: speech pre-filter triggered "
                          f"(idx={top_yamnet_idx}, score={top_yamnet_score:.3f})")
                return

            embedding = embeddings.numpy().mean(axis=0, keepdims=True)

            # Run custom classifier on the embedding
            prediction = self._classifier.predict(embedding, verbose=0)[0]

            if DIAGNOSTIC_LOGGING:
                pred_info = ", ".join(
                    f"{CLASSES[i]}={float(prediction[i]):.3f}"
                    for i in range(NUM_CLASSES)
                )
                top_label = IDX_TO_CLASS[int(np.argmax(prediction))]
                top_conf = float(prediction[int(np.argmax(prediction))])
                class_thresh = CONFIDENCE_PER_CLASS.get(top_label, self.confidence_threshold)
                print(f"[SoundDiag] Classifier: [{pred_info}] -> "
                      f"{top_label} ({top_conf:.1%}) "
                      f"thresh={class_thresh}")

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

            # Extract YAMNet scores and embeddings
            scores, embeddings, _ = self._yamnet(waveform)

            # Speech pre-filter (same as process_audio)
            scores_np = scores.numpy()
            mean_scores = scores_np.mean(axis=0)
            top_yamnet_idx = int(mean_scores.argmax())
            top_yamnet_score = float(mean_scores.max())

            if DIAGNOSTIC_LOGGING:
                top3_idx = mean_scores.argsort()[-3:][::-1]
                top3_info = ", ".join(
                    f"cls{int(i)}={float(mean_scores[i]):.3f}"
                    for i in top3_idx
                )
                is_speech = top_yamnet_idx in YAMNET_SPEECH_INDICES
                print(f"[SoundDiag] (float) YAMNet top3: [{top3_info}] | "
                      f"top_idx={top_yamnet_idx} is_speech={is_speech} "
                      f"score={top_yamnet_score:.3f}")

            if top_yamnet_idx in YAMNET_SPEECH_INDICES and top_yamnet_score >= YAMNET_SPEECH_THRESHOLD:
                if DIAGNOSTIC_LOGGING:
                    print(f"[SoundDiag] (float) -> SKIP: speech pre-filter "
                          f"(idx={top_yamnet_idx}, score={top_yamnet_score:.3f})")
                return

            embedding = embeddings.numpy().mean(axis=0, keepdims=True)

            prediction = self._classifier.predict(embedding, verbose=0)[0]

            if DIAGNOSTIC_LOGGING:
                pred_info = ", ".join(
                    f"{CLASSES[i]}={float(prediction[i]):.3f}"
                    for i in range(NUM_CLASSES)
                )
                top_label = IDX_TO_CLASS[int(np.argmax(prediction))]
                top_conf = float(prediction[int(np.argmax(prediction))])
                print(f"[SoundDiag] (float) Classifier: [{pred_info}] -> "
                      f"{top_label} ({top_conf:.1%})")

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

        # Ignore "normal" and low-confidence detections (per-class thresholds)
        class_threshold = CONFIDENCE_PER_CLASS.get(label, self.confidence_threshold)
        if label == "normal" or confidence < class_threshold:
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

        # Only alert if we have enough consecutive detections (per-class)
        required = CONSECUTIVE_PER_CLASS.get(label, self.consecutive_required)
        if consecutive >= required:
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
        level_icons = {1: "[L1]", 2: "[L2]", 3: "[L3]"}

        print(f"\n{level_icons.get(level, '[??]')} "
              f"[SOUND ALERT L{level} - {level_names.get(level, '')}] "
              f"\"{class_name}\" -> {category} "
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

    print("\n[>] WATZS -- Sound Classifier Test")
    print("=" * 50)

    classifier = SoundClassifier()

    def on_alert(alert):
        print(f"\n[ALERT] Alert: {json_module.dumps(alert, indent=2)}")

    classifier.on_alert = on_alert

    if not classifier.is_model_loaded:
        print("\n[!] Models not available -- running mock tests...\n")
        print("Simulating threat sounds:\n")

        for category in ["scream", "gunshot", "glass_breaking", "crash"]:
            print(f"  Testing: {category}")
            classifier.mock_classify(category)
            print()

        print("[OK] Mock tests complete.")
    else:
        print("\n[OK] Custom model loaded. Ready for live classification.")
        print("   Run with AudioCapture for real-time detection.")

    print(f"\n[i] Supported threat categories:")
    for cat, level in sorted(
        SoundClassifier.get_threat_categories().items(), key=lambda x: x[1]
    ):
        print(f"   L{level}: {cat}")
