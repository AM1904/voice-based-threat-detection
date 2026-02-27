"""
WATZS Audio Engine
==================
Voice-based threat and emergency detection system.
Handles audio capture, keyword detection, sound classification,
secret voice code tracking, alarm classification, and server bridging.
"""

from audio_engine.capture import AudioCapture
from audio_engine.keyword_detector import KeywordDetector
from audio_engine.voice_code import VoiceCodeTracker
from audio_engine.sound_classifier import SoundClassifier
from audio_engine.alarm_classifier import AlarmClassifier
from audio_engine.server_bridge import ServerBridge

__all__ = [
    "AudioCapture",
    "KeywordDetector",
    "SoundClassifier",
    "VoiceCodeTracker",
    "AlarmClassifier",
    "ServerBridge",
]
