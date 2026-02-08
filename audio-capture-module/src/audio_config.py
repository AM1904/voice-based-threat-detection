"""
Audio Configuration Module

Defines all configuration parameters for audio capture including
sample rates, buffer sizes, and processing settings.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class AudioFormat(Enum):
    """Supported audio formats"""
    INT16 = "int16"
    INT32 = "int32"
    FLOAT32 = "float32"


@dataclass
class AudioConfig:
    """
    Configuration parameters for audio capture.
    
    Attributes:
        sample_rate: Audio sampling rate in Hz (default: 16000 for speech)
        channels: Number of audio channels (1=mono, 2=stereo)
        frame_size: Samples per frame (1024-4096 recommended)
        buffer_duration_seconds: Duration of ring buffer in seconds
        audio_format: Data type for audio samples
        device_index: Specific device index (None = default device)
        gain: Input gain multiplier (1.0 = no change)
        silence_threshold: RMS threshold for silence detection
        enable_noise_reduction: Toggle basic noise handling
    """
    
    # Core audio settings
    sample_rate: int = 16000  # 16kHz is standard for speech recognition
    channels: int = 1  # Mono for speech processing
    frame_size: int = 1024  # ~64ms at 16kHz
    
    # Buffer settings
    buffer_duration_seconds: float = 5.0  # 5 seconds of audio buffer
    
    # Format settings
    audio_format: AudioFormat = field(default=AudioFormat.FLOAT32)
    
    # Device settings
    device_index: Optional[int] = None  # None = system default
    device_name: Optional[str] = None  # Alternative: specify by name
    
    # Processing settings
    gain: float = 1.0
    silence_threshold: float = 0.01  # RMS threshold
    enable_noise_reduction: bool = True
    
    # Validation settings
    max_latency_ms: int = 100  # Maximum acceptable latency
    
    @property
    def buffer_size_frames(self) -> int:
        """Calculate number of frames that fit in the buffer"""
        frames_per_second = self.sample_rate / self.frame_size
        return int(frames_per_second * self.buffer_duration_seconds)
    
    @property
    def frame_duration_ms(self) -> float:
        """Duration of a single frame in milliseconds"""
        return (self.frame_size / self.sample_rate) * 1000
    
    @property
    def dtype_numpy(self) -> str:
        """Get numpy dtype string for the audio format"""
        mapping = {
            AudioFormat.INT16: "int16",
            AudioFormat.INT32: "int32",
            AudioFormat.FLOAT32: "float32",
        }
        return mapping[self.audio_format]
    
    def validate(self) -> list[str]:
        """
        Validate configuration parameters.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if self.sample_rate < 8000 or self.sample_rate > 48000:
            errors.append(f"Sample rate {self.sample_rate} outside valid range (8000-48000)")
        
        if self.channels not in [1, 2]:
            errors.append(f"Channels must be 1 or 2, got {self.channels}")
        
        if self.frame_size < 256 or self.frame_size > 8192:
            errors.append(f"Frame size {self.frame_size} outside valid range (256-8192)")
        
        if self.buffer_duration_seconds < 1.0:
            errors.append("Buffer duration must be at least 1 second")
        
        if self.gain <= 0:
            errors.append("Gain must be positive")
            
        if self.silence_threshold < 0 or self.silence_threshold > 1:
            errors.append("Silence threshold must be between 0 and 1")
        
        return errors


# Preset configurations for common use cases
PRESET_SPEECH_RECOGNITION = AudioConfig(
    sample_rate=16000,
    channels=1,
    frame_size=1024,
    audio_format=AudioFormat.FLOAT32,
)

PRESET_HIGH_QUALITY = AudioConfig(
    sample_rate=44100,
    channels=1,
    frame_size=2048,
    audio_format=AudioFormat.FLOAT32,
)

PRESET_LOW_LATENCY = AudioConfig(
    sample_rate=16000,
    channels=1,
    frame_size=512,
    audio_format=AudioFormat.FLOAT32,
    max_latency_ms=50,
)
