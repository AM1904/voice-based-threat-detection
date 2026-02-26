"""
Voice Code Tracker Tests
========================
Tests for the VoiceCodeTracker module — phrase detection,
sliding window counting, and L3 alert triggering.

Run: python tests/test_voice_code.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_engine.voice_code import VoiceCodeTracker


def test_phrase_detection():
    """Test 1: Basic phrase detection."""
    print("=" * 60)
    print("TEST 1: Phrase Detection")
    print("=" * 60)

    tracker = VoiceCodeTracker()
    alerts_fired = []
    tracker.on_alert = lambda a: alerts_fired.append(a)

    # Single mention should not trigger
    result = tracker.check_transcription(f"say {tracker.phrase}")
    assert result is False, "Single mention should not trigger L3"
    assert len(alerts_fired) == 0, "No alert yet"
    assert tracker.current_count == 1, "Count should be 1"

    print(f"  Phrase: \"{tracker.phrase}\"")
    print(f"  After 1 mention: count = {tracker.current_count}, alert = False")
    print("✅ PASS — Basic phrase detection works!")
    return True


def test_l3_trigger():
    """Test 2: L3 trigger after required repetitions."""
    print("\n" + "=" * 60)
    print("TEST 2: L3 Trigger on Threshold")
    print("=" * 60)

    tracker = VoiceCodeTracker()
    alerts_fired = []
    tracker.on_alert = lambda a: alerts_fired.append(a)

    # Say phrase required_reps times
    for i in range(tracker.required_reps):
        result = tracker.check_transcription(tracker.phrase)

    assert len(alerts_fired) == 1, \
        f"Should have 1 L3 alert, got {len(alerts_fired)}"
    assert alerts_fired[0]["level"] == 3, "Should be L3"
    assert alerts_fired[0]["type"] == "voice_code", "Should be voice_code type"

    print(f"  Required reps: {tracker.required_reps}")
    print(f"  Alert fired: level={alerts_fired[0]['level']}")
    print("✅ PASS — L3 trigger works!")
    return True


def test_reset_after_alert():
    """Test 3: Counter resets after L3 alert."""
    print("\n" + "=" * 60)
    print("TEST 3: Counter Reset After Alert")
    print("=" * 60)

    tracker = VoiceCodeTracker()
    alerts_fired = []
    tracker.on_alert = lambda a: alerts_fired.append(a)

    # Trigger first alert
    for _ in range(tracker.required_reps):
        tracker.check_transcription(tracker.phrase)
    assert len(alerts_fired) == 1, "First alert should fire"

    # Count should be 0 after alert
    assert tracker.current_count == 0, \
        f"Count should be 0 after alert, got {tracker.current_count}"

    # Need full cycle again for second alert
    tracker.check_transcription(tracker.phrase)
    assert len(alerts_fired) == 1, "Shouldn't trigger on single mention"

    print("  Counter after alert: 0")
    print("  Needs full cycle for next alert: ✓")
    print("✅ PASS — Counter reset works!")
    return True


def test_no_match():
    """Test 4: Non-matching text does nothing."""
    print("\n" + "=" * 60)
    print("TEST 4: No Match")
    print("=" * 60)

    tracker = VoiceCodeTracker()
    alerts_fired = []
    tracker.on_alert = lambda a: alerts_fired.append(a)

    tracker.check_transcription("hello world")
    tracker.check_transcription("this is normal speech")
    tracker.check_transcription("nothing to see here")

    assert len(alerts_fired) == 0, "No alerts for normal speech"
    assert tracker.current_count == 0, "Count should still be 0"

    print("  Normal speech: no alerts ✓")
    print("✅ PASS — Non-matching text ignored!")
    return True


def test_pause_resume():
    """Test 5: Pause and resume functionality."""
    print("\n" + "=" * 60)
    print("TEST 5: Pause / Resume")
    print("=" * 60)

    tracker = VoiceCodeTracker()
    alerts_fired = []
    tracker.on_alert = lambda a: alerts_fired.append(a)

    tracker.pause()
    tracker.check_transcription(tracker.phrase)
    assert tracker.current_count == 0, "Should not count when paused"

    tracker.resume()
    tracker.check_transcription(tracker.phrase)
    assert tracker.current_count == 1, "Should count after resume"

    print("  While paused: ignored ✓")
    print("  After resume: counted ✓")
    print("✅ PASS — Pause/resume works!")
    return True


def main():
    print("\n🔐 WATZS — Voice Code Tracker Tests")
    print("━" * 60)

    results = []
    results.append(("Phrase Detection", test_phrase_detection()))
    results.append(("L3 Trigger", test_l3_trigger()))
    results.append(("Reset After Alert", test_reset_after_alert()))
    results.append(("No Match", test_no_match()))
    results.append(("Pause / Resume", test_pause_resume()))

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
