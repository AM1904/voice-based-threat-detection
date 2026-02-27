"""
Server Bridge Module
====================
HTTP bridge between the audio engine and the Flask backend.
Sends alert events to the server's POST /alert endpoint.

Usage:
    from audio_engine.server_bridge import ServerBridge

    bridge = ServerBridge()
    # or: bridge = ServerBridge(server_url="http://192.168.1.5:5000")

    # Wire to alarm classifier:
    alarm_classifier.on_alert = bridge.send_alert
"""

import requests
import threading


# ─── Default Configuration ──────────────────────────────────────────────
DEFAULT_SERVER_URL = "http://localhost:5000"
ALERT_ENDPOINT = "/alert"
REQUEST_TIMEOUT = 5  # seconds


class ServerBridge:
    """
    Sends alert events to the Flask backend via HTTP POST.

    Strips fields not expected by the server (e.g., 'metadata')
    and handles connection failures gracefully.

    Attributes:
        server_url (str): Base URL of the Flask server
        timeout (int): HTTP request timeout in seconds
    """

    # Fields that the server's POST /alert expects
    EXPECTED_FIELDS = {"id", "type", "keyword", "level", "confidence", "source"}

    def __init__(self, server_url=None, timeout=REQUEST_TIMEOUT):
        self.server_url = (server_url or DEFAULT_SERVER_URL).rstrip("/")
        self.alert_url = f"{self.server_url}{ALERT_ENDPOINT}"
        self.timeout = timeout

        self._lock = threading.Lock()
        self._sent_count = 0
        self._failed_count = 0

        print(f"[ServerBridge] Initialized — target: {self.alert_url}")

    def send_alert(self, alert):
        """
        Send an alert event to the server.

        Runs in a background thread so it doesn't block
        the audio processing pipeline.

        Args:
            alert (dict): Alert event dict from AlarmClassifier
        """
        thread = threading.Thread(
            target=self._post_alert,
            args=(alert,),
            daemon=True
        )
        thread.start()

    def _post_alert(self, alert):
        """POST the alert to the server (runs in background thread)."""
        # Build payload with only the fields the server expects
        payload = {
            key: alert[key]
            for key in self.EXPECTED_FIELDS
            if key in alert
        }

        try:
            response = requests.post(
                self.alert_url,
                json=payload,
                timeout=self.timeout,
            )

            with self._lock:
                if response.status_code == 201:
                    self._sent_count += 1
                    print(f"[ServerBridge] ✅ Alert sent — "
                          f"L{payload.get('level')} "
                          f"({payload.get('type')}: "
                          f"{payload.get('keyword', 'N/A')})")
                else:
                    self._failed_count += 1
                    print(f"[ServerBridge] ⚠️  Server returned "
                          f"{response.status_code}: {response.text[:100]}")

        except requests.ConnectionError:
            with self._lock:
                self._failed_count += 1
            print(f"[ServerBridge] ❌ Connection failed — "
                  f"is the server running at {self.server_url}?")

        except requests.Timeout:
            with self._lock:
                self._failed_count += 1
            print(f"[ServerBridge] ❌ Request timed out "
                  f"(>{self.timeout}s)")

        except Exception as e:
            with self._lock:
                self._failed_count += 1
            print(f"[ServerBridge] ❌ Unexpected error: {e}")

    def send_alert_sync(self, alert):
        """
        Synchronous version of send_alert (for testing).

        Args:
            alert (dict): Alert event dict
        """
        self._post_alert(alert)

    # ─── Status ─────────────────────────────────────────────────────
    @property
    def sent_count(self):
        """Number of alerts successfully sent."""
        with self._lock:
            return self._sent_count

    @property
    def failed_count(self):
        """Number of alerts that failed to send."""
        with self._lock:
            return self._failed_count

    def __repr__(self):
        return (f"ServerBridge(url={self.server_url}, "
                f"sent={self.sent_count}, failed={self.failed_count})")


# ─── CLI Entry Point ───────────────────────────────────────────────────
if __name__ == "__main__":
    import uuid
    from datetime import datetime, timezone, timedelta

    print("\n🌐 WATZS — Server Bridge Test")
    print("=" * 50)
    print(f"Target: {DEFAULT_SERVER_URL}{ALERT_ENDPOINT}")
    print("\nMake sure the server is running: python run_server.py\n")

    bridge = ServerBridge()

    test_alert = {
        "id": str(uuid.uuid4()),
        "type": "keyword",
        "keyword": "test_alert",
        "level": 1,
        "timestamp": datetime.now(
            timezone(timedelta(hours=5, minutes=30))
        ).isoformat(),
        "confidence": 0.99,
        "source": "test",
    }

    print("Sending test alert (sync)...")
    bridge.send_alert_sync(test_alert)

    print(f"\n📋 Sent: {bridge.sent_count}, Failed: {bridge.failed_count}")
    print("✅ Test complete.")
