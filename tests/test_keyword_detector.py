"""
Keyword Detector Tests
======================
Tests for keyword matching and keyword escalation logic.

Run: python tests/test_keyword_detector.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_keyword_matching():
    """Test 1: Keyword matching against all levels."""
    print("=" * 60)
    print("TEST 1: Keyword Matching")
    print("=" * 60)

    from audio_engine.keyword_detector import KeywordDetector

    detector = KeywordDetector()
    alerts_fired = []
    detector.on_alert = lambda a: alerts_fired.append(a)

    # Test L1 keyword
    matches = detector._find_keywords("there is a gun here")
    assert any(kw == "gun" for kw, _ in matches), "Should find 'gun'"
    assert any(lvl == 1 for _, lvl in matches), "Should be L1"

    # Test L2 keyword
    matches = detector._find_keywords("we need help now")
    assert any(kw == "help" for kw, _ in matches), "Should find 'help'"

    # Test L3 keyword
    matches = detector._find_keywords("i will shoot you")
    assert any(kw == "i will shoot" for kw, _ in matches), "Should find 'i will shoot'"
    assert any(lvl == 3 for _, lvl in matches), "Should be L3"

    # Test multi-word keyword
    matches = detector._find_keywords("get on the floor now")
    assert any(kw == "get on the floor" for kw, _ in matches), \
        "Should find 'get on the floor'"

    # Test no match
    matches = detector._find_keywords("hello how are you doing today")
    assert len(matches) == 0, "Should find no keywords"

    print("  All keyword levels detected correctly.")
    print("PASS")
    return True


def test_keyword_escalation():
    """Test 2: Keyword repetition escalation to L3."""
    print("\n" + "=" * 60)
    print("TEST 2: Keyword Repetition Escalation")
    print("=" * 60)

    from audio_engine.keyword_detector import KeywordDetector

    detector = KeywordDetector()
    alerts_fired = []
    detector.on_alert = lambda a: alerts_fired.append(a)

    # Say an L1 keyword 3 times (threshold > 2 = escalation)
    detector._handle_transcription("i see a gun", is_final=True)
    detector._handle_transcription("he has a gun", is_final=True)
    detector._handle_transcription("gun right there", is_final=True)

    l3_alerts = [a for a in alerts_fired if a["level"] == 3]
    assert len(l3_alerts) >= 1, \
        f"Should have at least 1 L3 escalation, got {len(l3_alerts)}"
    assert l3_alerts[0]["metadata"].get("escalated_from") == 1, \
        "Should show escalated from L1"

    print(f"  Alerts fired: {len(alerts_fired)}")
    print(f"  L3 escalations: {len(l3_alerts)}")
    print("PASS")
    return True


def test_no_false_positives():
    """Test 3: Normal sentences do NOT trigger alerts."""
    print("\n" + "=" * 60)
    print("TEST 3: No False Positives")
    print("=" * 60)

    from audio_engine.keyword_detector import KeywordDetector

    detector = KeywordDetector()
    alerts_fired = []
    detector.on_alert = lambda a: alerts_fired.append(a)

    detector._handle_transcription("the weather is nice today", is_final=True)
    detector._handle_transcription("I went to the store", is_final=True)
    detector._handle_transcription("the the the the", is_final=True)

    assert len(alerts_fired) == 0, \
        f"Normal speech should not trigger alerts, got {len(alerts_fired)}"

    print("  Normal sentences: no alerts")
    print("  Repeated common words: no alerts")
    print("PASS")
    return True


def test_partial_ignored():
    """Test 4: Only final transcriptions trigger alerts."""
    print("\n" + "=" * 60)
    print("TEST 4: Partial Transcriptions Ignored")
    print("=" * 60)

    from audio_engine.keyword_detector import KeywordDetector

    detector = KeywordDetector()
    alerts_fired = []
    detector.on_alert = lambda a: alerts_fired.append(a)
    detector.reset_counts()

    detector._handle_transcription("there is a gun", is_final=False)
    assert len(alerts_fired) == 0, "Partial should not trigger"

    detector._handle_transcription("there is a gun", is_final=True)
    assert len(alerts_fired) >= 1, "Final should trigger"

    print("PASS")
    return True


def main():
    print("\nWATZS - Keyword Detector Tests")
    print("-" * 60)

    results = []
    results.append(("Keyword Matching", test_keyword_matching()))
    results.append(("Keyword Escalation", test_keyword_escalation()))
    results.append(("No False Positives", test_no_false_positives()))
    results.append(("Partial Ignored", test_partial_ignored()))

    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)

    passed = sum(1 for _, r in results if r)
    for name, result in results:
        print(f"   {'PASS' if result else 'FAIL'} - {name}")
    print(f"\n   Result: {passed}/{len(results)} tests passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
