"""
WATZS — Main Orchestrator
===========================
Starts the full WATZS voice-based threat detection system.

Wires together:
    AudioCapture → KeywordDetector + SoundClassifier
    KeywordDetector → VoiceCodeTracker (for secret phrase)
    All detectors → AlarmClassifier → ServerBridge → Flask API

Usage:
    1. Start the server first:   python run_server.py
    2. Then start the engine:    python run_watzs.py

Press Ctrl+C to stop.
"""

import signal
import sys
import time

from audio_engine.capture import AudioCapture
from audio_engine.keyword_detector import KeywordDetector
from audio_engine.sound_classifier import SoundClassifier
from audio_engine.voice_code import VoiceCodeTracker
from audio_engine.alarm_classifier import AlarmClassifier
from audio_engine.server_bridge import ServerBridge


def main():
    print()
    print("=" * 58)
    print("  🎙️  WATZS — Voice-Based Threat Detection System")
    print("=" * 58)
    print()

    # ─── Initialize Components ───────────────────────────────────
    print("📦 Initializing components...\n")

    capture = AudioCapture()
    keyword_detector = KeywordDetector()
    sound_classifier = SoundClassifier()
    voice_code_tracker = VoiceCodeTracker()
    alarm_classifier = AlarmClassifier()
    server_bridge = ServerBridge()

    # ─── Wire Callbacks ──────────────────────────────────────────
    print("\n🔗 Wiring pipeline...\n")

    # Audio → Detectors
    capture.add_listener(keyword_detector.process_audio)
    capture.add_listener(sound_classifier.process_audio)

    # KeywordDetector transcription → VoiceCodeTracker
    def on_transcription(text, is_final):
        if is_final and text.strip():
            voice_code_tracker.check_transcription(text)

    keyword_detector.on_transcription = on_transcription

    # All detectors → AlarmClassifier
    keyword_detector.on_alert = alarm_classifier.process_alert
    sound_classifier.on_alert = alarm_classifier.process_alert
    voice_code_tracker.on_alert = alarm_classifier.process_alert

    # AlarmClassifier → ServerBridge
    alarm_classifier.on_alert = server_bridge.send_alert

    # ─── Startup Summary ─────────────────────────────────────────
    print("─" * 58)
    print("  Pipeline:")
    print("    🎤 Mic → KeywordDetector (Whisper STT)")
    print("    🎤 Mic → SoundClassifier (YAMNet custom model)")
    print("    📝 Transcription → VoiceCodeTracker")
    print("    ⚠️  All alerts → AlarmClassifier → ServerBridge")
    print(f"    🌐 Target: {server_bridge.alert_url}")
    print("─" * 58)

    # Print keyword info
    print(f"\n📋 Loaded keywords:")
    for level_key in ["L1", "L2", "L3"]:
        level_data = keyword_detector.keywords_config["levels"].get(level_key, {})
        keywords = level_data.get("keywords", [])
        if keywords:
            print(f"   {level_key}: {', '.join(keywords)}")

    print(f"\n🔐 Secret code: \"{voice_code_tracker.phrase}\" "
          f"(×{voice_code_tracker.required_reps} "
          f"in {voice_code_tracker.time_window}s)")

    print(f"\n🔊 Sound classifier: "
          f"{'✅ Model loaded' if sound_classifier.is_model_loaded else '⚠️  Model not loaded (mock mode)'}")

    # ─── Graceful Shutdown ───────────────────────────────────────
    def signal_handler(sig, frame):
        print("\n\n⏹️  Shutting down WATZS...")
        capture.stop()
        print(f"\n📊 Session stats:")
        print(f"   Alerts sent:   {server_bridge.sent_count}")
        print(f"   Alerts failed: {server_bridge.failed_count}")
        print(f"   Pending in window: {alarm_classifier.pending_alerts_count}")
        print("\nGoodbye! 👋")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # ─── Start Listening ─────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  🟢 WATZS is ACTIVE — Listening for threats...")
    print("  Press Ctrl+C to stop.")
    print("=" * 58 + "\n")

    capture.start(blocking=True)


if __name__ == "__main__":
    main()
