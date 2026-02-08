"""
Audio Capture Module - Phase 1.1
Voice-Based Threat & Emergency Detection System

This module provides real-time audio capture capabilities for
continuous monitoring and threat detection applications.
"""

from .audio_config import AudioConfig
from .device_manager import DeviceManager
from .ring_buffer import RingBuffer
from .audio_stream import AudioStream
from .noise_handler import NoiseHandler
from .audio_capture import AudioCaptureModule

__version__ = "1.0.0"
__all__ = [
    "AudioConfig",
    "DeviceManager", 
    "RingBuffer",
    "AudioStream",
    "NoiseHandler",
    "AudioCaptureModule",
]
