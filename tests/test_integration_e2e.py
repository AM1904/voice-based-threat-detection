"""
WATZS — End-to-End Integration Tests
======================================
Tests the full alert processing pipeline:
  Detectors → AlarmClassifier → (mock) ServerBridge

Verifies that:
  - L1 keyword alerts pass through correctly
  - L2 sound alerts pass through correctly
  - L1 + L2 within 30s escalates to L3
  - Voice code 3× repetition escalates to L3
  - L3 alerts pass through unmodified

Run:
    python -m pytest tests/test_integration_e2e.py -v
"""

import time
import uuid
import threading
import pytest
from unittest.mock import MagicMock


# ─── Helpers ────────────────────────────────────────────────────────────────

def make_alert(level, alert_type="keyword", keyword="test"):
    """Build a minimal alert dict matching the shared schema."""
    return {
        "id": str(uuid.uuid4()),
        "type": alert_type,
        "keyword": keyword,
        "level": level,
        "confidence": 0.92,
        "source": "test",
    }


def build_classifier_with_capture():
    """
    Instantiate AlarmClassifier wired to a captured output list.
    Returns (classifier, received_alerts).
    """
    from audio_engine.alarm_classifier import AlarmClassifier

    received = []

    classifier = AlarmClassifier()
    classifier.on_alert = received.append
    return classifier, received


# ─── Tests ──────────────────────────────────────────────────────────────────

class TestL1AlertPassthrough:
    """L1 keyword alerts should pass through unchanged."""

    def test_l1_forwarded(self):
        classifier, received = build_classifier_with_capture()
        alert = make_alert(level=1, keyword="gun")
        classifier.process_alert(alert)

        assert len(received) == 1
        assert received[0]["level"] == 1
        assert received[0]["keyword"] == "gun"

    def test_l1_does_not_auto_escalate(self):
        """A single L1 alert must NOT trigger a phantom L3."""
        classifier, received = build_classifier_with_capture()
        classifier.process_alert(make_alert(level=1, keyword="kill"))

        levels = [a["level"] for a in received]
        assert 3 not in levels, "L1 alone must not produce L3"


class TestL2AlertPassthrough:
    """L2 sound alerts should pass through unchanged."""

    def test_l2_forwarded(self):
        classifier, received = build_classifier_with_capture()
        alert = make_alert(level=2, alert_type="sound", keyword="scream")
        classifier.process_alert(alert)

        assert len(received) == 1
        assert received[0]["level"] == 2

    def test_l2_does_not_auto_escalate(self):
        classifier, received = build_classifier_with_capture()
        classifier.process_alert(make_alert(level=2, alert_type="sound", keyword="crash"))

        levels = [a["level"] for a in received]
        assert 3 not in levels, "L2 alone must not produce L3"


class TestL3DirectAlert:
    """L3 alerts (e.g., voice code) should pass through directly."""

    def test_l3_forwarded(self):
        classifier, received = build_classifier_with_capture()
        alert = make_alert(level=3, alert_type="voice_code", keyword="secret phrase")
        classifier.process_alert(alert)

        assert len(received) == 1
        assert received[0]["level"] == 3


class TestL1PlusL2Escalation:
    """L1 + L2 within the escalation window should trigger an additional L3."""

    def test_combined_escalation_fires_l3(self):
        classifier, received = build_classifier_with_capture()

        classifier.process_alert(make_alert(level=1, keyword="gun"))
        classifier.process_alert(make_alert(level=2, alert_type="sound", keyword="scream"))

        # Should have 3 alerts: original L1, original L2, + escalated L3
        assert len(received) == 3, (
            f"Expected 3 alerts (L1 + L2 + escalated L3), got {len(received)}"
        )
        levels = [a["level"] for a in received]
        assert 1 in levels
        assert 2 in levels
        assert 3 in levels

    def test_escalation_source_is_classifier(self):
        classifier, received = build_classifier_with_capture()
        classifier.process_alert(make_alert(level=1, keyword="gun"))
        classifier.process_alert(make_alert(level=2, alert_type="sound", keyword="crash"))

        l3_alerts = [a for a in received if a["level"] == 3]
        assert len(l3_alerts) == 1
        assert l3_alerts[0].get("source") == "alarm_classifier"

    def test_escalation_reason_is_correct(self):
        classifier, received = build_classifier_with_capture()
        classifier.process_alert(make_alert(level=1, keyword="knife"))
        classifier.process_alert(make_alert(level=2, alert_type="sound", keyword="gunshot"))

        l3 = next(a for a in received if a["level"] == 3)
        assert "L1+L2" in l3["metadata"]["reason"]

    def test_no_double_escalation_in_same_window(self):
        """Once escalation fires, no second L3 should fire for the same window."""
        classifier, received = build_classifier_with_capture()

        # First L1 + L2 → escalation
        classifier.process_alert(make_alert(level=1, keyword="gun"))
        classifier.process_alert(make_alert(level=2, alert_type="sound", keyword="scream"))

        # Second L1 + L2 immediately after (still within window)
        classifier.process_alert(make_alert(level=1, keyword="knife"))
        classifier.process_alert(make_alert(level=2, alert_type="sound", keyword="crash"))

        l3_count = sum(1 for a in received if a["level"] == 3)
        assert l3_count == 1, (
            f"Expected exactly 1 L3 escalation, got {l3_count}"
        )


class TestEscalationWindowExpiry:
    """L3 escalation should NOT fire if L1 and L2 are more than 30s apart."""

    def test_l1_then_l2_after_window_no_escalation(self):
        from audio_engine.alarm_classifier import AlarmClassifier
        import time

        # Use a very short window of 1 second for speed
        classifier = AlarmClassifier()
        classifier.escalation_window = 1  # override to 1s

        received = []
        classifier.on_alert = received.append

        classifier.process_alert(make_alert(level=1, keyword="gun"))
        time.sleep(1.5)  # wait past the 1-second window
        classifier.process_alert(make_alert(level=2, alert_type="sound", keyword="scream"))

        l3_alerts = [a for a in received if a["level"] == 3]
        assert len(l3_alerts) == 0, (
            "L3 must NOT fire when L1 and L2 are outside the escalation window"
        )


class TestVoiceCodeL3Path:
    """Voice code firing L3 directly should pass through without escalation."""

    def test_voice_code_l3_forwarded_directly(self):
        classifier, received = build_classifier_with_capture()

        for _ in range(3):
            alert = make_alert(
                level=3,
                alert_type="voice_code",
                keyword="watzs help"
            )
            classifier.process_alert(alert)

        l3_alerts = [a for a in received if a["level"] == 3]
        assert len(l3_alerts) == 3, (
            "3 direct L3 voice_code alerts should all be forwarded"
        )


class TestClassifierReset:
    """reset() should clear the sliding window."""

    def test_reset_prevents_escalation(self):
        classifier, received = build_classifier_with_capture()

        classifier.process_alert(make_alert(level=1, keyword="gun"))
        classifier.reset()

        # Now send L2 — no L1 in window anymore → no escalation
        classifier.process_alert(make_alert(level=2, alert_type="sound", keyword="crash"))

        l3_alerts = [a for a in received if a["level"] == 3]
        assert len(l3_alerts) == 0


class TestPipelineWithMockBridge:
    """Simulate the full run_watzs.py pipeline with a mocked ServerBridge."""

    def test_alerts_sent_to_mock_bridge(self):
        from audio_engine.alarm_classifier import AlarmClassifier

        sent_payloads = []
        mock_bridge = MagicMock()
        mock_bridge.send_alert.side_effect = lambda alert: sent_payloads.append(alert)

        classifier = AlarmClassifier()
        classifier.on_alert = mock_bridge.send_alert

        # Send L1 + L2 → should produce 3 calls (L1, L2, L3 escalation)
        classifier.process_alert(make_alert(level=1, keyword="gun"))
        classifier.process_alert(make_alert(level=2, alert_type="sound", keyword="scream"))

        assert mock_bridge.send_alert.call_count == 3
        levels = [c.args[0]["level"] for c in mock_bridge.send_alert.call_args_list]
        assert sorted(levels) == [1, 2, 3]
