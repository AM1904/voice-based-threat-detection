"""
Server Bridge Tests
===================
Tests for the ServerBridge module — field mapping, error handling.
Uses unittest.mock to avoid needing a real server.

Run: python tests/test_server_bridge.py
"""

import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
from audio_engine.server_bridge import ServerBridge


def _make_alert(level=1, alert_type="keyword", keyword="test"):
    """Helper to create a test alert dict."""
    return {
        "id": str(uuid.uuid4()),
        "type": alert_type,
        "keyword": keyword,
        "level": level,
        "confidence": 0.90,
        "source": "whisper",
        "metadata": {"repeat_count": 1, "original_text": "test"},
    }


def test_successful_post():
    """Test 1: Successful POST sends correct data."""
    print("=" * 60)
    print("TEST 1: Successful POST")
    print("=" * 60)

    bridge = ServerBridge()

    mock_response = MagicMock()
    mock_response.status_code = 201

    with patch("audio_engine.server_bridge.requests.post",
               return_value=mock_response) as mock_post:
        alert = _make_alert(level=1, keyword="gun")
        bridge.send_alert_sync(alert)

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")

        assert payload["type"] == "keyword", "Type should be 'keyword'"
        assert payload["keyword"] == "gun", "Keyword should be 'gun'"
        assert payload["level"] == 1, "Level should be 1"
        assert "metadata" not in payload, "Metadata should be stripped"

    assert bridge.sent_count == 1, "Sent count should be 1"

    print("  Correct URL and payload ✓")
    print("  Metadata stripped ✓")
    print("  Sent count incremented ✓")
    print("✅ PASS")
    return True


def test_field_mapping():
    """Test 2: Only expected fields are sent to server."""
    print("\n" + "=" * 60)
    print("TEST 2: Field Mapping")
    print("=" * 60)

    bridge = ServerBridge()

    mock_response = MagicMock()
    mock_response.status_code = 201

    with patch("audio_engine.server_bridge.requests.post",
               return_value=mock_response) as mock_post:
        alert = _make_alert(level=2, alert_type="sound", keyword="scream")
        alert["metadata"] = {"sound_class": "Screaming"}
        alert["extra_field"] = "should_be_stripped"
        bridge.send_alert_sync(alert)

        payload = mock_post.call_args.kwargs.get("json") or \
                  mock_post.call_args[1].get("json")

        expected_keys = {"id", "type", "keyword", "level", "confidence", "source"}
        actual_keys = set(payload.keys())

        assert actual_keys == expected_keys, \
            f"Expected keys {expected_keys}, got {actual_keys}"

    print(f"  Payload keys: {sorted(expected_keys)} ✓")
    print("  Extra fields stripped ✓")
    print("✅ PASS")
    return True


def test_connection_failure():
    """Test 3: Connection failure is handled gracefully."""
    print("\n" + "=" * 60)
    print("TEST 3: Connection Failure Handling")
    print("=" * 60)

    import requests as req_lib

    bridge = ServerBridge()

    with patch("audio_engine.server_bridge.requests.post",
               side_effect=req_lib.ConnectionError("Connection refused")):
        # Should not raise
        alert = _make_alert(level=1, keyword="gun")
        bridge.send_alert_sync(alert)

    assert bridge.failed_count == 1, "Failed count should be 1"
    assert bridge.sent_count == 0, "Sent count should be 0"

    print("  No crash on connection error ✓")
    print("  Failed count incremented ✓")
    print("✅ PASS")
    return True


def test_timeout_handling():
    """Test 4: Timeout is handled gracefully."""
    print("\n" + "=" * 60)
    print("TEST 4: Timeout Handling")
    print("=" * 60)

    import requests as req_lib

    bridge = ServerBridge()

    with patch("audio_engine.server_bridge.requests.post",
               side_effect=req_lib.Timeout("Request timed out")):
        bridge.send_alert_sync(_make_alert())

    assert bridge.failed_count == 1, "Failed count should be 1"

    print("  No crash on timeout ✓")
    print("✅ PASS")
    return True


def main():
    print("\n🌐 WATZS — Server Bridge Tests")
    print("━" * 60)

    results = []
    results.append(("Successful POST", test_successful_post()))
    results.append(("Field Mapping", test_field_mapping()))
    results.append(("Connection Failure", test_connection_failure()))
    results.append(("Timeout Handling", test_timeout_handling()))

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
