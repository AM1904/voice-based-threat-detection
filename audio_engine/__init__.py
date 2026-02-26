"""
WATZS Audio Engine
==================
Voice-based threat and emergency detection system.
Handles audio capture, keyword detection, sound classification,
secret voice code tracking, and alert classification.
"""

from audio_engine.capture import AudioCapture
from audio_engine.keyword_detector import KeywordDetector
from audio_engine.voice_code import VoiceCodeTracker
from audio_engine.sound_classifier import SoundClassifier

__all__ = [
    "AudioCapture",
    "KeywordDetector",
    "SoundClassifier",
    "VoiceCodeTracker",
]
