"""
STT Configuration Module

Defines all configuration parameters for speech-to-text processing
including model selection, processing settings, and output formatting.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum
from pathlib import Path


class WhisperModelSize(Enum):
    """Available Whisper model sizes"""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large-v3"
    
    @property
    def description(self) -> str:
        descriptions = {
            "tiny": "Fastest, ~1GB VRAM, lower accuracy",
            "base": "Fast, ~1GB VRAM, good for real-time",
            "small": "Balanced, ~2GB VRAM, recommended",
            "medium": "Accurate, ~5GB VRAM, slower",
            "large-v3": "Most accurate, ~10GB VRAM, slowest",
        }
        return descriptions.get(self.value, "Unknown")


class ComputeType(Enum):
    """Compute precision types for inference"""
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT8 = "int8"
    AUTO = "auto"


@dataclass
class STTConfig:
    """Configuration for Speech-to-Text processing."""
    
    model_size: WhisperModelSize = WhisperModelSize.BASE
    compute_type: ComputeType = ComputeType.AUTO
    device: Literal["cuda", "cpu", "auto"] = "auto"
    language: Optional[str] = "en"
    sample_rate: int = 16000
    chunk_duration: float = 2.0
    min_silence_duration: float = 0.5
    vad_threshold: float = 0.5
    speech_pad_ms: int = 400
    beam_size: int = 5
    best_of: int = 1
    temperature: float = 0.0
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = True
    initial_prompt: Optional[str] = None
    model_path: Optional[Path] = None
    enable_timestamps: bool = True
    enable_word_timestamps: bool = False
    output_format: Literal["json", "text", "srt"] = "json"
    num_workers: int = 1
    max_queue_size: int = 10
    log_transcriptions: bool = True
    log_path: Optional[Path] = None
    
    def validate(self) -> list[str]:
        errors = []
        if self.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            errors.append(f"Unusual sample rate: {self.sample_rate}")
        if self.chunk_duration < 0.5:
            errors.append("Chunk duration too short (min 0.5s)")
        if self.chunk_duration > 30:
            errors.append("Chunk duration too long (max 30s)")
        return errors
    
    @classmethod
    def for_realtime(cls) -> "STTConfig":
        return cls(
            model_size=WhisperModelSize.BASE,
            compute_type=ComputeType.AUTO,
            chunk_duration=1.5,
            beam_size=3,
        )
    
    @classmethod
    def for_accuracy(cls) -> "STTConfig":
        return cls(
            model_size=WhisperModelSize.SMALL,
            compute_type=ComputeType.FLOAT16,
            chunk_duration=3.0,
            beam_size=5,
            enable_word_timestamps=True,
        )
