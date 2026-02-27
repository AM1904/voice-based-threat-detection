"""
Alarm Classifier Tests
======================
Tests for the AlarmClassifier module — alert passthrough,
combined L1+L2 → L3 escalation, and edge cases.

Run: python tests/test_alarm_classifier.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_engine.alarm_classifier import AlarmClassifier
import uuid


def _make_alert(level, alert_type="keyword", keyword="test"):
    """Helper to create a test alert dict."""
    return {
        "id": str(uuid.uuid4()),
        "type": alert_type,
        "keyword": keyword,
        "level": level,
        "confidence": 0.90,
        "source": "test",
    }


def test_l1_passthrough():
    """Test 1: L1 alert is forwarded unchanged."""
    print("=" * 60)
    print("TEST 1: L1 Passthrough")
    print("=" * 60)

    classifier = AlarmClassifier()
    emitted = []
    classifier.on_alert = lambda a: emitted.append(a)

    alert = _make_alert(level=1, keyword="gun")
    classifier.process_alert(alert)

    assert len(emitted) == 1, f"Expected 1 alert, got {len(emitted)}"
    assert emitted[0]["level"] == 1, "Should be L1"
    assert emitted[0]["keyword"] == "gun", "Keyword should be 'gun'"

    print("  L1 forwarded as-is ✓")
    print("✅ PASS")
    return True


def test_l2_passthrough():
    """Test 2: L2 alert is forwarded unchanged."""
    print("\n" + "=" * 60)
    print("TEST 2: L2 Passthrough")
    print("=" * 60)

    classifier = AlarmClassifier()
    emitted = []
    classifier.on_alert = lambda a: emitted.append(a)

    alert = _make_alert(level=2, alert_type="sound", keyword="scream")
    classifier.process_alert(alert)

    assert len(emitted) == 1, f"Expected 1 alert, got {len(emitted)}"
    assert emitted[0]["level"] == 2, "Should be L2"
    assert emitted[0]["type"] == "sound", "Type should be 'sound'"

    print("  L2 forwarded as-is ✓")
    print("✅ PASS")
    return True


def test_combined_escalation():
    """Test 3: L1 then L2 within window → produces additional L3."""
    print("\n" + "=" * 60)
    print("TEST 3: Combined L1+L2 → L3 Escalation")
    print("=" * 60)

    classifier = AlarmClassifier()
    emitted = []
    classifier.on_alert = lambda a: emitted.append(a)

    # Send L1
    classifier.process_alert(_make_alert(level=1, keyword="gun"))
    assert len(emitted) == 1, "L1 should pass through"

    # Send L2 within window — should trigger escalation
    classifier.process_alert(_make_alert(level=2, alert_type="sound", keyword="scream"))
    assert len(emitted) == 3, \
        f"Expected 3 alerts (L1 + L2 + escalated L3), got {len(emitted)}"

    # The third alert should be the escalation
    escalated = emitted[2]
    assert escalated["level"] == 3, "Escalated alert should be L3"
    assert escalated["source"] == "alarm_classifier", \
        "Escalated alert source should be 'alarm_classifier'"

    print(f"  L1 passed through ✓")
    print(f"  L2 passed through ✓")
    print(f"  L3 escalation fired ✓")
    print("✅ PASS")
    return True


def test_l3_passthrough_no_double():
    """Test 4: L3 from VoiceCodeTracker passes through, no double-escalation."""
    print("\n" + "=" * 60)
    print("TEST 4: L3 Passthrough (No Double Escalation)")
    print("=" * 60)

    classifier = AlarmClassifier()
    emitted = []
    classifier.on_alert = lambda a: emitted.append(a)

    alert = _make_alert(level=3, alert_type="voice_code", keyword="watzs emergency")
    classifier.process_alert(alert)

    assert len(emitted) == 1, f"Expected 1 alert, got {len(emitted)}"
    assert emitted[0]["level"] == 3, "Should remain L3"

    print("  L3 forwarded as-is, no double-escalation ✓")
    print("✅ PASS")
    return True


def test_reset():
    """Test 5: Reset clears the sliding window."""
    print("\n" + "=" * 60)
    print("TEST 5: Reset")
    print("=" * 60)

    classifier = AlarmClassifier()
    emitted = []
    classifier.on_alert = lambda a: emitted.append(a)

    # Send L1, then reset, then L2 — should NOT escalate
    classifier.process_alert(_make_alert(level=1, keyword="gun"))
    classifier.reset()
    classifier.process_alert(_make_alert(level=2, alert_type="sound", keyword="scream"))

    # Should have 2 alerts: L1 + L2 (no escalation since reset cleared the window)
    assert len(emitted) == 2, \
        f"Expected 2 alerts (L1 + L2, no escalation), got {len(emitted)}"

    print("  Reset clears window, no false escalation ✓")
    print("✅ PASS")
    return True


def test_reverse_order():
    """Test 6: L2 first, then L1 — should still escalate."""
    print("\n" + "=" * 60)
    print("TEST 6: Reverse Order (L2 then L1)")
    print("=" * 60)

    classifier = AlarmClassifier()
    emitted = []
    classifier.on_alert = lambda a: emitted.append(a)

    classifier.process_alert(_make_alert(level=2, alert_type="sound", keyword="scream"))
    classifier.process_alert(_make_alert(level=1, keyword="gun"))

    assert len(emitted) == 3, \
        f"Expected 3 alerts (L2 + L1 + L3 escalation), got {len(emitted)}"
    assert emitted[2]["level"] == 3, "Third alert should be L3 escalation"

    print("  L2 → L1 order also triggers escalation ✓")
    print("✅ PASS")
    return True


def main():
    print("\n⚠️  WATZS — Alarm Classifier Tests")
    print("━" * 60)

    results = []
    results.append(("L1 Passthrough", test_l1_passthrough()))
    results.append(("L2 Passthrough", test_l2_passthrough()))
    results.append(("Combined Escalation", test_combined_escalation()))
    results.append(("L3 Passthrough", test_l3_passthrough_no_double()))
    results.append(("Reset", test_reset()))
    results.append(("Reverse Order", test_reverse_order()))

    print("\n" + "━" * 60)
    print("📋 TEST SUMMARY")
    print("━" * 60)

    passed = sum(1 for _, r in results if r)
    for name, result in results:
        print(f"   {'✅ PASS' if result else '❌ FAIL'} — {name}")
    print(f"\n   Result: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All tests passed!")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
