"""
Speech-to-Text Module - Phase 1.2
Voice-Based Threat & Emergency Detection System

This module provides real-time speech recognition capabilities,
converting continuous audio streams into structured text output.
"""

from .stt_config import STTConfig, WhisperModelSize
from .transcription import TranscriptionResult, TranscriptionSegment
from .whisper_engine import WhisperEngine
from .speech_processor import SpeechProcessor
from .stt_pipeline import STTPipeline
from .transcription_logger import TranscriptionLogger

__version__ = "1.0.0"
__all__ = [
    "STTConfig",
    "WhisperModelSize",
    "TranscriptionResult",
    "TranscriptionSegment",
    "WhisperEngine",
    "SpeechProcessor",
    "STTPipeline",
    "TranscriptionLogger",
]
