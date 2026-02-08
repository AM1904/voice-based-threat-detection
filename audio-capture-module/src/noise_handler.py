"""
Noise Handler Module

Provides basic noise reduction and audio preprocessing capabilities.
Includes gain normalization, silence detection, and simple filtering.
"""

import numpy as np
from typing import Optional, Callable
import logging
from dataclasses import dataclass

from .audio_config import AudioConfig

logger = logging.getLogger(__name__)


@dataclass
class NoiseStats:
    """Statistics about noise handling"""
    frames_processed: int
    silent_frames: int
    avg_rms: float
    peak_rms: float
    gain_adjustments: int


class NoiseHandler:
    """
    Basic noise handling for audio preprocessing.
    
    Features:
    - Gain normalization
    - Silence detection
    - Simple noise gate
    - RMS level monitoring
    
    Note: Advanced noise reduction (spectral subtraction, etc.)
    is deferred to later phases.
    """
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize the noise handler.
        
        Args:
            config: Audio configuration
        """
        self.config = config or AudioConfig()
        
        # State tracking
        self._frames_processed = 0
        self._silent_frames = 0
        self._total_rms = 0.0
        self._peak_rms = 0.0
        self._gain_adjustments = 0
        
        # Adaptive gain state
        self._current_gain = self.config.gain
        self._rms_history: list[float] = []
        self._history_size = 50  # frames for averaging
        
        # Noise gate state
        self._noise_floor = 0.0
        self._noise_calibrated = False
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process an audio frame with noise handling.
        
        Args:
            frame: Raw audio frame
            
        Returns:
            Processed audio frame
        """
        self._frames_processed += 1
        
        # Calculate RMS for this frame
        rms = self._calculate_rms(frame)
        self._update_rms_stats(rms)
        
        # Apply processing chain
        processed = frame.copy()
        
        if self.config.enable_noise_reduction:
            # Apply gain
            processed = self._apply_gain(processed)
            
            # Apply noise gate
            processed = self._apply_noise_gate(processed, rms)
        
        return processed
    
    def _calculate_rms(self, frame: np.ndarray) -> float:
        """Calculate Root Mean Square of frame"""
        return float(np.sqrt(np.mean(frame ** 2)))
    
    def _update_rms_stats(self, rms: float) -> None:
        """Update RMS statistics"""
        self._total_rms += rms
        self._peak_rms = max(self._peak_rms, rms)
        
        # Update history for adaptive processing
        self._rms_history.append(rms)
        if len(self._rms_history) > self._history_size:
            self._rms_history.pop(0)
    
    def _apply_gain(self, frame: np.ndarray) -> np.ndarray:
        """Apply gain to frame"""
        return frame * self._current_gain
    
    def _apply_noise_gate(self, frame: np.ndarray, rms: float) -> np.ndarray:
        """
        Apply a simple noise gate.
        
        Frames below the silence threshold are attenuated.
        """
        if rms < self.config.silence_threshold:
            self._silent_frames += 1
            # Apply soft attenuation instead of hard cut
            attenuation = rms / self.config.silence_threshold
            return frame * attenuation
        return frame
    
    def calibrate_noise_floor(self, frames: list[np.ndarray], percentile: float = 10.0) -> float:
        """
        Calibrate the noise floor from a collection of frames.
        
        Args:
            frames: List of audio frames to analyze
            percentile: Percentile of RMS values to use as noise floor
            
        Returns:
            Calculated noise floor level
        """
        if not frames:
            logger.warning("No frames provided for noise calibration")
            return 0.0
        
        rms_values = [self._calculate_rms(f) for f in frames]
        self._noise_floor = float(np.percentile(rms_values, percentile))
        self._noise_calibrated = True
        
        logger.info(f"Noise floor calibrated: {self._noise_floor:.6f}")
        return self._noise_floor
    
    def normalize_gain(self, target_rms: float = 0.1) -> float:
        """
        Calculate gain adjustment to reach target RMS level.
        
        Args:
            target_rms: Target RMS level (0.0-1.0)
            
        Returns:
            New gain value
        """
        if not self._rms_history:
            return self._current_gain
        
        current_avg_rms = np.mean(self._rms_history)
        if current_avg_rms > 0.001:  # Avoid division by very small numbers
            new_gain = target_rms / current_avg_rms
            # Limit gain range to prevent extreme values
            new_gain = np.clip(new_gain, 0.1, 10.0)
            
            if abs(new_gain - self._current_gain) > 0.1:
                self._current_gain = float(new_gain)
                self._gain_adjustments += 1
                logger.debug(f"Gain adjusted to: {self._current_gain:.2f}")
        
        return self._current_gain
    
    def is_silent(self, frame: np.ndarray) -> bool:
        """
        Check if a frame is silent.
        
        Args:
            frame: Audio frame to check
            
        Returns:
            True if frame is below silence threshold
        """
        rms = self._calculate_rms(frame)
        return rms < self.config.silence_threshold
    
    def get_level(self, frame: np.ndarray) -> dict:
        """
        Get audio level information for a frame.
        
        Args:
            frame: Audio frame
            
        Returns:
            Dict with level information
        """
        rms = self._calculate_rms(frame)
        peak = float(np.max(np.abs(frame)))
        
        # Convert to dB (with floor to avoid -inf)
        rms_db = 20 * np.log10(max(rms, 1e-10))
        peak_db = 20 * np.log10(max(peak, 1e-10))
        
        return {
            "rms": rms,
            "peak": peak,
            "rms_db": rms_db,
            "peak_db": peak_db,
            "is_silent": rms < self.config.silence_threshold,
            "is_clipping": peak > 0.99,
        }
    
    def get_stats(self) -> NoiseStats:
        """Get noise handling statistics"""
        avg_rms = self._total_rms / max(self._frames_processed, 1)
        return NoiseStats(
            frames_processed=self._frames_processed,
            silent_frames=self._silent_frames,
            avg_rms=avg_rms,
            peak_rms=self._peak_rms,
            gain_adjustments=self._gain_adjustments,
        )
    
    def reset_stats(self) -> None:
        """Reset statistics"""
        self._frames_processed = 0
        self._silent_frames = 0
        self._total_rms = 0.0
        self._peak_rms = 0.0
        self._gain_adjustments = 0
        self._rms_history.clear()


class VoiceActivityDetector:
    """
    Simple Voice Activity Detection (VAD).
    
    Detects the presence of human speech in audio frames
    using energy-based detection.
    """
    
    def __init__(
        self,
        threshold: float = 0.02,
        min_speech_frames: int = 3,
        min_silence_frames: int = 10
    ):
        """
        Initialize VAD.
        
        Args:
            threshold: RMS threshold for speech detection
            min_speech_frames: Minimum consecutive frames for speech start
            min_silence_frames: Minimum consecutive frames for speech end
        """
        self.threshold = threshold
        self.min_speech_frames = min_speech_frames
        self.min_silence_frames = min_silence_frames
        
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._is_speaking = False
        
        self._on_speech_start: Optional[Callable] = None
        self._on_speech_end: Optional[Callable] = None
    
    def process(self, frame: np.ndarray) -> bool:
        """
        Process a frame and detect voice activity.
        
        Args:
            frame: Audio frame
            
        Returns:
            True if speech is detected
        """
        rms = float(np.sqrt(np.mean(frame ** 2)))
        
        if rms > self.threshold:
            self._speech_frame_count += 1
            self._silence_frame_count = 0
            
            if not self._is_speaking and self._speech_frame_count >= self.min_speech_frames:
                self._is_speaking = True
                if self._on_speech_start:
                    self._on_speech_start()
        else:
            self._silence_frame_count += 1
            self._speech_frame_count = 0
            
            if self._is_speaking and self._silence_frame_count >= self.min_silence_frames:
                self._is_speaking = False
                if self._on_speech_end:
                    self._on_speech_end()
        
        return self._is_speaking
    
    @property
    def is_speaking(self) -> bool:
        return self._is_speaking
    
    def on_speech_start(self, callback: Callable) -> None:
        """Register callback for speech start"""
        self._on_speech_start = callback
    
    def on_speech_end(self, callback: Callable) -> None:
        """Register callback for speech end"""
        self._on_speech_end = callback
