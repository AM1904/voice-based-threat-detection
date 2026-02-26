"""
Voice Code Tracker Module
=========================
Tracks occurrences of a secret emergency voice code phrase
within a sliding time window. Triggers a Level 3 alert when
the phrase is spoken the required number of times.

The secret phrase and thresholds are loaded from keywords.json.

Usage:
    from audio_engine.voice_code import VoiceCodeTracker

    def on_alert(alert):
        print(f"EMERGENCY: {alert}")

    tracker = VoiceCodeTracker()
    tracker.on_alert = on_alert

    # Feed transcription results from KeywordDetector:
    tracker.check_transcription("watzs emergency")
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone, timedelta


# ─── Default Configuration ──────────────────────────────────────────────
DEFAULT_KEYWORDS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "keywords.json"
)


class VoiceCodeTracker:
    """
    Tracks secret voice code phrase repetitions in a sliding time window.
    Fires a Level 3 alert when the threshold is reached.

    Attributes:
        phrase (str): The secret phrase to detect
        required_reps (int): Number of repetitions needed to trigger L3
        time_window (int): Sliding window in seconds
        on_alert (callable): Callback for L3 alert events
    """

    def __init__(self, keywords_path=None):
        self.keywords_path = keywords_path or DEFAULT_KEYWORDS_PATH

        # Callbacks
        self.on_alert = None

        # Load configuration
        self._load_config()

        # Tracking state
        self._timestamps = []  # Timestamps of phrase detections
        self._is_active = True
        self._total_detections = 0

        print(f"[VoiceCodeTracker] Initialized — "
              f"phrase=\"{self.phrase}\", "
              f"need {self.required_reps}× in {self.time_window}s")

    def _load_config(self):
        """Load secret code configuration from keywords.json."""
        with open(self.keywords_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        secret = config.get("secret_code", {})
        self.phrase = secret.get("phrase", "watzs emergency").lower()
        self.required_reps = secret.get("required_repetitions", 3)
        self.time_window = secret.get("time_window_seconds", 60)

    # ─── Core Detection ─────────────────────────────────────────────
    def check_transcription(self, text):
        """
        Check transcribed text for the secret voice code phrase.

        Call this with each final transcription result from the
        KeywordDetector or Vosk recognizer.

        Args:
            text (str): Transcribed speech text

        Returns:
            bool: True if L3 alert was triggered
        """
        if not self._is_active:
            return False

        text_lower = text.lower()

        # Count occurrences of the phrase in this transcription
        occurrences = text_lower.count(self.phrase)
        if occurrences == 0:
            return False

        now = time.time()
        self._total_detections += occurrences

        # Record timestamps for each occurrence
        for _ in range(occurrences):
            self._timestamps.append(now)

        # Clean expired timestamps
        self._timestamps = [
            t for t in self._timestamps
            if now - t <= self.time_window
        ]

        count = len(self._timestamps)
        print(f"[VoiceCodeTracker] Secret phrase detected! "
              f"({count}/{self.required_reps} in window)")

        # Check if threshold reached
        if count >= self.required_reps:
            self._trigger_alert(count)
            self._timestamps.clear()  # Reset after alert
            return True

        return False

    def _trigger_alert(self, count):
        """Emit a Level 3 emergency alert."""
        alert = {
            "id": str(uuid.uuid4()),
            "type": "voice_code",
            "keyword": self.phrase,
            "level": 3,
            "timestamp": datetime.now(timezone(timedelta(hours=5, minutes=30)))
                         .isoformat(),
            "confidence": 1.0,
            "source": "voice_code",
            "metadata": {
                "repeat_count": count,
                "time_window": self.time_window,
                "total_detections": self._total_detections
            }
        }

        print(f"\n🔴🔴🔴 [ALERT L3 — EMERGENCY] "
              f"Secret voice code activated! "
              f"(\"{self.phrase}\" ×{count} in {self.time_window}s)")

        if self.on_alert:
            self.on_alert(alert)

    # ─── Control ────────────────────────────────────────────────────
    def pause(self):
        """Pause tracking."""
        self._is_active = False
        print("[VoiceCodeTracker] Paused.")

    def resume(self):
        """Resume tracking."""
        self._is_active = True
        print("[VoiceCodeTracker] Resumed.")

    def reset(self):
        """Reset all counters and timestamps."""
        self._timestamps.clear()
        self._total_detections = 0
        print("[VoiceCodeTracker] Reset.")

    @property
    def current_count(self):
        """Get count of phrase detections in current window."""
        now = time.time()
        self._timestamps = [
            t for t in self._timestamps
            if now - t <= self.time_window
        ]
        return len(self._timestamps)

    @property
    def is_active(self):
        return self._is_active


# ─── CLI Entry Point ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔐 WATZS — Voice Code Tracker Test")
    print("=" * 50)

    tracker = VoiceCodeTracker()

    def on_alert(alert):
        print(f"\n🚨 ALERT FIRED: {json.dumps(alert, indent=2)}")

    tracker.on_alert = on_alert

    # Simulate phrase detection
    print(f"\nSimulating \"{tracker.phrase}\" spoken 3 times...\n")

    tracker.check_transcription(f"I need to say {tracker.phrase}")
    time.sleep(0.5)
    tracker.check_transcription(f"{tracker.phrase} please respond")
    time.sleep(0.5)
    tracker.check_transcription(f"repeat {tracker.phrase} now")

    print(f"\n✅ Test complete. Total detections: {tracker._total_detections}")
