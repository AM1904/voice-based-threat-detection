"""
Audio Capture Module - Main Interface

Unified interface for the audio capture system.
Provides a high-level API for real-time audio capture with
all components integrated: device management, streaming,
buffering, and noise handling.
"""

import time
import logging
import threading
from typing import Optional, Callable, Generator
from dataclasses import dataclass
import numpy as np

from .audio_config import AudioConfig
from .device_manager import DeviceManager, AudioDevice
from .ring_buffer import RingBuffer
from .audio_stream import AudioStream, StreamState
from .noise_handler import NoiseHandler, VoiceActivityDetector

logger = logging.getLogger(__name__)


@dataclass
class CaptureStats:
    """Comprehensive capture statistics"""
    # Stream stats
    stream_state: str
    duration_seconds: float
    frames_captured: int
    frames_per_second: float
    avg_latency_ms: float
    
    # Buffer stats
    buffer_fill_percent: float
    buffer_overruns: int
    buffer_underruns: int
    
    # Audio stats
    avg_rms: float
    peak_rms: float
    silent_frames: int
    
    # Health
    xrun_count: int
    callback_errors: int
    
    def is_healthy(self) -> bool:
        """Check if capture is running healthily"""
        return (
            self.stream_state == "running" and
            self.callback_errors < 10 and
            self.buffer_fill_percent < 90
        )
    
    def __str__(self) -> str:
        return (
            f"CaptureStats:\n"
            f"  State: {self.stream_state}\n"
            f"  Duration: {self.duration_seconds:.1f}s\n"
            f"  Frames: {self.frames_captured} ({self.frames_per_second:.1f}/s)\n"
            f"  Latency: {self.avg_latency_ms:.2f}ms\n"
            f"  Buffer: {self.buffer_fill_percent:.1f}% full\n"
            f"  RMS: avg={self.avg_rms:.4f}, peak={self.peak_rms:.4f}\n"
            f"  Silent frames: {self.silent_frames}\n"
            f"  Issues: xruns={self.xrun_count}, errors={self.callback_errors}"
        )


class AudioCaptureModule:
    """
    High-level audio capture interface.
    
    This is the main entry point for the audio capture system.
    It integrates all components and provides a simple API for:
    - Starting/stopping capture
    - Reading audio frames
    - Monitoring capture health
    - Registering processing callbacks
    
    Usage:
        # Simple usage
        capture = AudioCaptureModule()
        capture.start()
        
        for frame in capture.frames():
            process(frame)
        
        # With context manager
        with AudioCaptureModule() as capture:
            for frame in capture.frames(max_frames=1000):
                process(frame)
    
    Integration with Speech-to-Text:
        The output frames are ready for direct use with speech
        recognition engines. They are:
        - Properly sampled (default 16kHz)
        - Single channel (mono)
        - Float32 normalized (-1.0 to 1.0)
    """
    
    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        device_index: Optional[int] = None,
        device_name: Optional[str] = None,
        enable_noise_handling: bool = True,
        enable_vad: bool = False,
    ):
        """
        Initialize the audio capture module.
        
        Args:
            config: Audio configuration (uses defaults if not provided)
            device_index: Specific device index to use
            device_name: Device name to search for (partial match)
            enable_noise_handling: Enable basic noise preprocessing
            enable_vad: Enable voice activity detection
        """
        # Configuration
        self.config = config or AudioConfig()
        if device_index is not None:
            self.config.device_index = device_index
        if device_name:
            self.config.device_name = device_name
        
        # Validate configuration
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {errors}")
        
        # Components
        self.device_manager = DeviceManager(self.config)
        self.buffer = RingBuffer(
            capacity=self.config.buffer_size_frames,
            frame_size=self.config.frame_size,
            channels=self.config.channels,
            dtype=self.config.dtype_numpy,
        )
        
        self._stream: Optional[AudioStream] = None
        self._device: Optional[AudioDevice] = None
        
        # Processing
        self._enable_noise_handling = enable_noise_handling
        self._enable_vad = enable_vad
        
        if enable_noise_handling:
            self.noise_handler = NoiseHandler(self.config)
        else:
            self.noise_handler = None
        
        if enable_vad:
            self.vad = VoiceActivityDetector()
        else:
            self.vad = None
        
        # Callbacks for processed frames
        self._frame_callbacks: list[Callable[[np.ndarray], None]] = []
        
        # Logging setup
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Configure logging for the module"""
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    
    def list_devices(self) -> list[AudioDevice]:
        """
        List all available audio input devices.
        
        Returns:
            List of AudioDevice objects
        """
        return self.device_manager.list_devices()
    
    def print_devices(self) -> None:
        """Print available devices to console"""
        self.device_manager.print_devices()
    
    def select_device(
        self,
        index: Optional[int] = None,
        name: Optional[str] = None
    ) -> AudioDevice:
        """
        Select an audio device for capture.
        
        Args:
            index: Device index
            name: Device name (partial match)
            
        Returns:
            Selected AudioDevice
        """
        if index is not None:
            device = self.device_manager.get_device_by_index(index)
        elif name:
            device = self.device_manager.get_device_by_name(name)
        else:
            device = None
        
        self._device = self.device_manager.select_device(device)
        return self._device
    
    def test_device(self, duration: float = 1.0) -> bool:
        """
        Test the selected or default device.
        
        Args:
            duration: Test duration in seconds
            
        Returns:
            True if test passed
        """
        if not self._device:
            self.select_device()
        return self.device_manager.test_device(self._device, duration)
    
    def start(self) -> bool:
        """
        Start audio capture.
        
        Returns:
            True if capture started successfully
        """
        if self._stream and self._stream.is_running:
            logger.warning("Capture already running")
            return True
        
        # Select device if not already done
        if not self._device:
            self.select_device()
        
        # Create and start stream
        self._stream = AudioStream(
            config=self.config,
            buffer=self.buffer,
            device=self._device,
        )
        
        success = self._stream.start()
        if success:
            logger.info("Audio capture started")
        
        return success
    
    def stop(self) -> None:
        """Stop audio capture"""
        if self._stream:
            self._stream.stop()
            logger.info("Audio capture stopped")
    
    def pause(self) -> None:
        """Pause audio capture"""
        if self._stream:
            self._stream.pause()
    
    def resume(self) -> None:
        """Resume paused capture"""
        if self._stream:
            self._stream.resume()
    
    @property
    def is_running(self) -> bool:
        """Check if capture is currently running"""
        return self._stream is not None and self._stream.is_running
    
    @property
    def state(self) -> str:
        """Get current capture state"""
        if self._stream:
            return self._stream.state.value
        return "stopped"
    
    def read_frame(
        self,
        blocking: bool = True,
        timeout: float = 1.0,
        process: bool = True
    ) -> Optional[np.ndarray]:
        """
        Read a single audio frame.
        
        Args:
            blocking: Wait for frame if buffer is empty
            timeout: Max wait time in seconds
            process: Apply noise handling if enabled
            
        Returns:
            Audio frame or None
        """
        frame = self.buffer.read(blocking=blocking, timeout=timeout)
        
        if frame is not None and process and self.noise_handler:
            frame = self.noise_handler.process_frame(frame)
            
        if frame is not None and self.vad:
            self.vad.process(frame)
        
        return frame
    
    def frames(
        self,
        max_frames: Optional[int] = None,
        timeout: float = 1.0,
        process: bool = True
    ) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields audio frames.
        
        Args:
            max_frames: Maximum frames to yield (None = infinite)
            timeout: Timeout for each frame read
            process: Apply noise handling
            
        Yields:
            Audio frames
            
        Example:
            for frame in capture.frames(max_frames=100):
                process(frame)
        """
        count = 0
        while self.is_running:
            if max_frames and count >= max_frames:
                break
            
            frame = self.read_frame(blocking=True, timeout=timeout, process=process)
            if frame is not None:
                count += 1
                yield frame
    
    def on_frame(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        Register a callback for processed frames.
        
        Args:
            callback: Function to call with each frame
        """
        self._frame_callbacks.append(callback)
    
    def get_stats(self) -> CaptureStats:
        """Get comprehensive capture statistics"""
        stream_stats = self._stream.get_stats() if self._stream else None
        buffer_stats = self.buffer.get_stats()
        noise_stats = self.noise_handler.get_stats() if self.noise_handler else None
        
        return CaptureStats(
            stream_state=self.state,
            duration_seconds=stream_stats.duration_seconds if stream_stats else 0,
            frames_captured=stream_stats.frames_captured if stream_stats else 0,
            frames_per_second=stream_stats.frames_per_second if stream_stats else 0,
            avg_latency_ms=stream_stats.avg_latency_ms if stream_stats else 0,
            buffer_fill_percent=buffer_stats.fill_percentage,
            buffer_overruns=buffer_stats.overruns,
            buffer_underruns=buffer_stats.underruns,
            avg_rms=noise_stats.avg_rms if noise_stats else 0,
            peak_rms=noise_stats.peak_rms if noise_stats else 0,
            silent_frames=noise_stats.silent_frames if noise_stats else 0,
            xrun_count=stream_stats.xrun_count if stream_stats else 0,
            callback_errors=stream_stats.callback_errors if stream_stats else 0,
        )
    
    def get_audio_level(self) -> Optional[dict]:
        """
        Get current audio level information.
        
        Returns:
            Dict with RMS, peak, and dB levels
        """
        frame = self.buffer.peek()
        if frame is not None and self.noise_handler:
            return self.noise_handler.get_level(frame)
        return None
    
    def calibrate_noise(self, duration: float = 2.0) -> float:
        """
        Calibrate noise floor from ambient sound.
        
        Args:
            duration: Duration to sample in seconds
            
        Returns:
            Calibrated noise floor level
        """
        if not self.noise_handler:
            raise RuntimeError("Noise handling is disabled")
        
        if not self.is_running:
            raise RuntimeError("Capture must be running to calibrate")
        
        frames_needed = int(duration * (self.config.sample_rate / self.config.frame_size))
        frames = []
        
        logger.info(f"Calibrating noise floor for {duration}s...")
        for _ in range(frames_needed):
            frame = self.read_frame(process=False)
            if frame is not None:
                frames.append(frame)
        
        noise_floor = self.noise_handler.calibrate_noise_floor(frames)
        logger.info(f"Noise floor calibrated: {noise_floor:.6f}")
        return noise_floor
    
    def run_continuous(
        self,
        duration: Optional[float] = None,
        callback: Optional[Callable[[np.ndarray], None]] = None,
        status_interval: float = 10.0
    ) -> None:
        """
        Run continuous capture with optional callback.
        
        This is a blocking call that runs the capture loop.
        
        Args:
            duration: Max duration in seconds (None = run forever)
            callback: Function to call for each frame
            status_interval: Seconds between status logs
        """
        if not self.is_running:
            if not self.start():
                raise RuntimeError("Failed to start capture")
        
        start_time = time.time()
        last_status = start_time
        
        logger.info(f"Running continuous capture...")
        
        try:
            for frame in self.frames():
                # Check duration limit
                if duration and (time.time() - start_time) >= duration:
                    logger.info("Duration limit reached")
                    break
                
                # Call user callback
                if callback:
                    callback(frame)
                
                # Log status periodically
                if time.time() - last_status >= status_interval:
                    stats = self.get_stats()
                    logger.info(
                        f"Status: {stats.frames_captured} frames, "
                        f"{stats.avg_latency_ms:.2f}ms latency, "
                        f"{stats.buffer_fill_percent:.1f}% buffer"
                    )
                    last_status = time.time()
                    
        except KeyboardInterrupt:
            logger.info("Capture interrupted by user")
        finally:
            self.stop()
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
        return False


def create_capture_module(
    sample_rate: int = 16000,
    frame_size: int = 1024,
    buffer_seconds: float = 5.0,
    device_name: Optional[str] = None,
) -> AudioCaptureModule:
    """
    Factory function to create an AudioCaptureModule with common settings.
    
    Args:
        sample_rate: Sampling rate in Hz
        frame_size: Samples per frame
        buffer_seconds: Buffer duration
        device_name: Optional device name to search for
        
    Returns:
        Configured AudioCaptureModule
    """
    config = AudioConfig(
        sample_rate=sample_rate,
        channels=1,
        frame_size=frame_size,
        buffer_duration_seconds=buffer_seconds,
    )
    
    return AudioCaptureModule(
        config=config,
        device_name=device_name,
        enable_noise_handling=True,
    )
