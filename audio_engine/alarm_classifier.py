"""
Alarm Classifier Module
=======================
Applies escalation rules to incoming alert events before they
reach the server. Sits between detection modules and the server
bridge in the processing pipeline.

Escalation Rules (from keywords.json):
    • L1 + L2 within 30 seconds → emit additional L3 alert
    • Individual alert pass-through is always preserved

Usage:
    from audio_engine.alarm_classifier import AlarmClassifier

    classifier = AlarmClassifier()
    classifier.on_alert = lambda alert: send_to_server(alert)

    # Feed alerts from any detection module:
    keyword_detector.on_alert = classifier.process_alert
    sound_classifier.on_alert = classifier.process_alert
    voice_code_tracker.on_alert = classifier.process_alert
"""

import json
import os
import time
import uuid
import threading
from datetime import datetime, timezone, timedelta


# ─── Default Configuration ──────────────────────────────────────────────
DEFAULT_KEYWORDS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "keywords.json"
)

IST = timezone(timedelta(hours=5, minutes=30))


class AlarmClassifier:
    """
    Classifies and escalates alert events based on escalation rules.

    Maintains a sliding window of recent alerts and checks for
    combined-level escalation conditions.

    Attributes:
        escalation_window (int): Time window in seconds for L1+L2 → L3
        on_alert (callable): Callback for outgoing alerts (original + escalated)
    """

    def __init__(self, keywords_path=None):
        self.keywords_path = keywords_path or DEFAULT_KEYWORDS_PATH
        self._load_config()

        # Callback for outgoing alerts
        self.on_alert = None

        # Sliding window of recent alerts: list of (timestamp, alert_dict)
        self._recent_alerts = []
        self._lock = threading.Lock()

        # Track whether we already fired an escalation for a given window
        self._last_escalation_time = 0

        print(f"[AlarmClassifier] Initialized — "
              f"escalation window: {self.escalation_window}s")

    def _load_config(self):
        """Load escalation rules from keywords.json."""
        with open(self.keywords_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        rules = config.get("escalation_rules", {})
        combined = rules.get("combined_L1_L2_to_L3", {})
        self.escalation_window = combined.get("time_window_seconds", 30)

    # ─── Core Processing ────────────────────────────────────────────
    def process_alert(self, alert):
        """
        Process an incoming alert event.

        1. Always passes the original alert through to on_alert.
        2. Records the alert in the sliding window.
        3. Checks if combined escalation (L1+L2 → L3) should fire.

        Args:
            alert (dict): Alert event matching the shared schema
        """
        if not alert:
            return

        now = time.time()
        level = alert.get("level", 0)

        # 1. Always forward the original alert
        self._emit(alert)

        # 2. Record in sliding window (only L1 and L2 are relevant)
        if level in (1, 2):
            with self._lock:
                self._recent_alerts.append((now, alert))
                self._cleanup_window(now)

                # 3. Check for combined escalation
                self._check_combined_escalation(now, alert)

    def _cleanup_window(self, now):
        """Remove alerts older than the escalation window."""
        cutoff = now - self.escalation_window
        self._recent_alerts = [
            (t, a) for t, a in self._recent_alerts
            if t > cutoff
        ]

    def _check_combined_escalation(self, now, triggering_alert):
        """
        Check if both L1 and L2 alerts exist within the escalation window.
        If so, emit an additional L3 escalation alert.
        """
        has_l1 = any(a.get("level") == 1 for _, a in self._recent_alerts)
        has_l2 = any(a.get("level") == 2 for _, a in self._recent_alerts)

        if has_l1 and has_l2:
            # Don't fire multiple escalations within the same window
            if now - self._last_escalation_time < self.escalation_window:
                return

            self._last_escalation_time = now

            # Build escalation alert
            escalation = {
                "id": str(uuid.uuid4()),
                "type": triggering_alert.get("type", "keyword"),
                "keyword": triggering_alert.get("keyword", "escalation"),
                "level": 3,
                "timestamp": datetime.now(IST).isoformat(),
                "confidence": 0.95,
                "source": "alarm_classifier",
                "metadata": {
                    "escalated_from": triggering_alert.get("level"),
                    "reason": "L1+L2 combined within escalation window",
                    "window_seconds": self.escalation_window,
                }
            }

            print(f"\n🔴🔴🔴 [ESCALATION → L3] "
                  f"L1 + L2 detected within {self.escalation_window}s!")

            self._emit(escalation)

            # Clear the window after escalation to prevent repeats
            self._recent_alerts.clear()

    def _emit(self, alert):
        """Forward an alert to the registered callback."""
        if self.on_alert:
            self.on_alert(alert)

    # ─── Control ────────────────────────────────────────────────────
    def reset(self):
        """Clear the sliding window and escalation state."""
        with self._lock:
            self._recent_alerts.clear()
            self._last_escalation_time = 0
        print("[AlarmClassifier] Reset.")

    @property
    def pending_alerts_count(self):
        """Number of alerts currently in the sliding window."""
        with self._lock:
            self._cleanup_window(time.time())
            return len(self._recent_alerts)


# ─── CLI Entry Point ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n⚠️  WATZS — Alarm Classifier Test")
    print("=" * 50)

    classifier = AlarmClassifier()
    fired_alerts = []

    def on_alert(alert):
        fired_alerts.append(alert)
        level_icons = {1: "🟡", 2: "🟠", 3: "🔴"}
        icon = level_icons.get(alert["level"], "⚪")
        print(f"  {icon} Alert L{alert['level']}: "
              f"type={alert['type']}, keyword={alert.get('keyword')}")

    classifier.on_alert = on_alert

    # Simulate L1 keyword alert
    print("\n--- Sending L1 (keyword: 'gun') ---")
    classifier.process_alert({
        "id": str(uuid.uuid4()),
        "type": "keyword",
        "keyword": "gun",
        "level": 1,
        "confidence": 0.92,
        "source": "whisper",
    })

    # Simulate L2 sound alert within 30s
    print("\n--- Sending L2 (sound: 'scream') within 30s ---")
    classifier.process_alert({
        "id": str(uuid.uuid4()),
        "type": "sound",
        "keyword": "scream",
        "level": 2,
        "confidence": 0.85,
        "source": "watzs_custom",
    })

    print(f"\n📋 Total alerts emitted: {len(fired_alerts)}")
    print(f"   L3 escalations: {sum(1 for a in fired_alerts if a['level'] == 3)}")
    print(f"\n✅ Test complete.")
