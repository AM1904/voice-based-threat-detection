"""
Audio Stream Module

Manages continuous real-time audio streaming from input devices.
Provides the core streaming functionality with callback-based processing.
"""

import threading
import time
import logging
from typing import Optional, Callable
from enum import Enum
from dataclasses import dataclass
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    raise ImportError("sounddevice is required. Install with: pip install sounddevice")

from .audio_config import AudioConfig
from .ring_buffer import RingBuffer
from .device_manager import DeviceManager, AudioDevice

logger = logging.getLogger(__name__)


class StreamState(Enum):
    """Audio stream states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class StreamStats:
    """Statistics about the audio stream"""
    state: StreamState
    frames_captured: int
    duration_seconds: float
    avg_latency_ms: float
    callback_errors: int
    xrun_count: int  # Buffer under/overruns
    
    @property
    def frames_per_second(self) -> float:
        if self.duration_seconds > 0:
            return self.frames_captured / self.duration_seconds
        return 0


class AudioStream:
    """
    Continuous audio streaming from an input device.
    
    Features:
    - Callback-based real-time capture
    - Automatic reconnection on errors
    - Frame-based buffering
    - Latency monitoring
    - Event hooks for stream lifecycle
    
    Usage:
        stream = AudioStream(config)
        stream.start()
        
        while stream.is_running:
            frame = stream.read_frame()
            if frame is not None:
                process(frame)
        
        stream.stop()
    """
    
    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        buffer: Optional[RingBuffer] = None,
        device: Optional[AudioDevice] = None
    ):
        """
        Initialize the audio stream.
        
        Args:
            config: Audio configuration
            buffer: Ring buffer for frames (created if not provided)
            device: Audio device to use (auto-selected if not provided)
        """
        self.config = config or AudioConfig()
        self.device = device
        
        # Create buffer if not provided
        if buffer:
            self.buffer = buffer
        else:
            self.buffer = RingBuffer(
                capacity=self.config.buffer_size_frames,
                frame_size=self.config.frame_size,
                channels=self.config.channels,
                dtype=self.config.dtype_numpy,
            )
        
        # Stream state
        self._state = StreamState.STOPPED
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
        
        # Statistics
        self._frames_captured = 0
        self._start_time: Optional[float] = None
        self._callback_errors = 0
        self._xrun_count = 0
        self._latency_samples: list[float] = []
        
        # Callbacks
        self._on_frame_callbacks: list[Callable[[np.ndarray], None]] = []
        self._on_error_callbacks: list[Callable[[Exception], None]] = []
        self._on_state_change_callbacks: list[Callable[[StreamState], None]] = []
        
        # Processing thread
        self._processing_thread: Optional[threading.Thread] = None
        self._should_stop = threading.Event()
        
    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags
    ) -> None:
        """
        Callback function called by sounddevice for each audio block.
        
        This runs in a separate audio thread - keep it fast!
        """
        callback_start = time.perf_counter()
        
        # Check for stream issues
        if status:
            if status.input_overflow:
                self._xrun_count += 1
                logger.warning("Audio input overflow (xrun)")
            if status.input_underflow:
                self._xrun_count += 1
                logger.warning("Audio input underflow (xrun)")
        
        try:
            # Make a copy of the audio data
            frame = indata.copy()
            
            # Flatten if mono
            if self.config.channels == 1 and len(frame.shape) > 1:
                frame = frame.flatten()
            
            # Write to ring buffer
            self.buffer.write(frame)
            self._frames_captured += 1
            
            # Track latency
            latency = (time.perf_counter() - callback_start) * 1000
            self._latency_samples.append(latency)
            if len(self._latency_samples) > 100:
                self._latency_samples.pop(0)
            
            # Call registered frame callbacks (in audio thread - be careful!)
            for callback in self._on_frame_callbacks:
                try:
                    callback(frame)
                except Exception as e:
                    self._callback_errors += 1
                    logger.error(f"Frame callback error: {e}")
                    
        except Exception as e:
            self._callback_errors += 1
            logger.error(f"Audio callback error: {e}")
    
    def start(self) -> bool:
        """
        Start the audio stream.
        
        Returns:
            True if stream started successfully
        """
        with self._lock:
            if self._state == StreamState.RUNNING:
                logger.warning("Stream already running")
                return True
            
            self._set_state(StreamState.STARTING)
            
            try:
                # Get device if not set
                if not self.device:
                    device_manager = DeviceManager(self.config)
                    self.device = device_manager.select_device()
                
                logger.info(f"Starting audio stream on device: {self.device.name}")
                
                # Create the input stream
                self._stream = sd.InputStream(
                    device=self.device.index,
                    samplerate=self.config.sample_rate,
                    channels=self.config.channels,
                    dtype=self.config.dtype_numpy,
                    blocksize=self.config.frame_size,
                    callback=self._audio_callback,
                )
                
                # Start capturing
                self._stream.start()
                self._start_time = time.time()
                self._should_stop.clear()
                
                self._set_state(StreamState.RUNNING)
                logger.info("Audio stream started successfully")
                return True
                
            except Exception as e:
                self._set_state(StreamState.ERROR)
                logger.error(f"Failed to start stream: {e}")
                self._notify_error(e)
                return False
    
    def stop(self) -> None:
        """Stop the audio stream"""
        with self._lock:
            if self._state in [StreamState.STOPPED, StreamState.ERROR]:
                return
            
            logger.info("Stopping audio stream...")
            self._should_stop.set()
            
            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logger.error(f"Error stopping stream: {e}")
                finally:
                    self._stream = None
            
            self._set_state(StreamState.STOPPED)
            logger.info("Audio stream stopped")
    
    def pause(self) -> None:
        """Pause the audio stream"""
        with self._lock:
            if self._state != StreamState.RUNNING:
                return
            
            if self._stream:
                self._stream.stop()
                self._set_state(StreamState.PAUSED)
                logger.info("Audio stream paused")
    
    def resume(self) -> None:
        """Resume a paused stream"""
        with self._lock:
            if self._state != StreamState.PAUSED:
                return
            
            if self._stream:
                self._stream.start()
                self._set_state(StreamState.RUNNING)
                logger.info("Audio stream resumed")
    
    def read_frame(self, blocking: bool = True, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Read a single frame from the buffer.
        
        Args:
            blocking: If True, wait for a frame
            timeout: Max time to wait in seconds
            
        Returns:
            Audio frame or None
        """
        return self.buffer.read(blocking=blocking, timeout=timeout)
    
    def read_frames(self, count: int) -> list[np.ndarray]:
        """
        Read multiple frames from the buffer.
        
        Args:
            count: Number of frames to read
            
        Returns:
            List of audio frames
        """
        return self.buffer.read_batch(count)
    
    @property
    def is_running(self) -> bool:
        """Check if stream is currently running"""
        return self._state == StreamState.RUNNING
    
    @property
    def state(self) -> StreamState:
        """Get current stream state"""
        return self._state
    
    def _set_state(self, state: StreamState) -> None:
        """Update state and notify callbacks"""
        old_state = self._state
        self._state = state
        
        if old_state != state:
            for callback in self._on_state_change_callbacks:
                try:
                    callback(state)
                except Exception as e:
                    logger.error(f"State change callback error: {e}")
    
    def _notify_error(self, error: Exception) -> None:
        """Notify error callbacks"""
        for callback in self._on_error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error callback error: {e}")
    
    def on_frame(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        Register a callback for each captured frame.
        
        WARNING: Callbacks run in the audio thread. Keep them fast!
        
        Args:
            callback: Function to call with each frame
        """
        self._on_frame_callbacks.append(callback)
    
    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Register error callback"""
        self._on_error_callbacks.append(callback)
    
    def on_state_change(self, callback: Callable[[StreamState], None]) -> None:
        """Register state change callback"""
        self._on_state_change_callbacks.append(callback)
    
    def get_stats(self) -> StreamStats:
        """Get stream statistics"""
        duration = 0.0
        if self._start_time:
            duration = time.time() - self._start_time
        
        avg_latency = 0.0
        if self._latency_samples:
            avg_latency = sum(self._latency_samples) / len(self._latency_samples)
        
        return StreamStats(
            state=self._state,
            frames_captured=self._frames_captured,
            duration_seconds=duration,
            avg_latency_ms=avg_latency,
            callback_errors=self._callback_errors,
            xrun_count=self._xrun_count,
        )
    
    def reset_stats(self) -> None:
        """Reset stream statistics"""
        self._frames_captured = 0
        self._callback_errors = 0
        self._xrun_count = 0
        self._latency_samples.clear()
        self._start_time = time.time() if self.is_running else None
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
        return False
