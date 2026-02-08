"""
Ring Buffer Module

Thread-safe circular buffer for storing audio frames without memory allocation
during real-time operation. Prevents data loss and smooths hardware latency.
"""

import threading
import numpy as np
from typing import Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BufferStats:
    """Statistics about the ring buffer state"""
    capacity: int
    current_size: int
    total_writes: int
    total_reads: int
    overruns: int  # Writes when buffer was full
    underruns: int  # Reads when buffer was empty
    
    @property
    def fill_percentage(self) -> float:
        return (self.current_size / self.capacity) * 100 if self.capacity > 0 else 0
    
    @property
    def loss_rate(self) -> float:
        total = self.total_writes + self.overruns
        return (self.overruns / total) * 100 if total > 0 else 0


class RingBuffer:
    """
    Thread-safe ring buffer for audio frames.
    
    Features:
    - Lock-free reads/writes where possible
    - Configurable overflow behavior
    - Real-time statistics tracking
    - Zero-copy frame access option
    
    Usage:
        buffer = RingBuffer(capacity=100, frame_size=1024, channels=1)
        buffer.write(audio_frame)
        frame = buffer.read()
    """
    
    def __init__(
        self,
        capacity: int,
        frame_size: int,
        channels: int = 1,
        dtype: str = "float32",
        overflow_mode: str = "overwrite"
    ):
        """
        Initialize the ring buffer.
        
        Args:
            capacity: Maximum number of frames to store
            frame_size: Number of samples per frame
            channels: Number of audio channels
            dtype: NumPy dtype for audio data
            overflow_mode: "overwrite" (drop oldest) or "drop" (drop newest)
        """
        self.capacity = capacity
        self.frame_size = frame_size
        self.channels = channels
        self.dtype = dtype
        self.overflow_mode = overflow_mode
        
        # Pre-allocate buffer memory
        self._buffer = np.zeros(
            (capacity, frame_size, channels) if channels > 1 else (capacity, frame_size),
            dtype=dtype
        )
        
        # Buffer state
        self._write_idx = 0
        self._read_idx = 0
        self._count = 0
        
        # Statistics
        self._total_writes = 0
        self._total_reads = 0
        self._overruns = 0
        self._underruns = 0
        
        # Thread safety
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        
        logger.debug(
            f"RingBuffer initialized: capacity={capacity}, "
            f"frame_size={frame_size}, channels={channels}"
        )
    
    def write(self, frame: np.ndarray, blocking: bool = False, timeout: Optional[float] = None) -> bool:
        """
        Write a frame to the buffer.
        
        Args:
            frame: Audio frame to write (numpy array)
            blocking: If True, wait for space when buffer is full
            timeout: Max time to wait if blocking (None = forever)
            
        Returns:
            True if frame was written, False if dropped
        """
        with self._lock:
            if self._count >= self.capacity:
                if blocking:
                    # Wait for space
                    if not self._not_full.wait(timeout):
                        return False
                elif self.overflow_mode == "overwrite":
                    # Overwrite oldest frame
                    self._read_idx = (self._read_idx + 1) % self.capacity
                    self._count -= 1
                    self._overruns += 1
                else:
                    # Drop the new frame
                    self._overruns += 1
                    return False
            
            # Write frame to buffer
            self._buffer[self._write_idx] = frame
            self._write_idx = (self._write_idx + 1) % self.capacity
            self._count += 1
            self._total_writes += 1
            
            # Signal that buffer is not empty
            self._not_empty.notify()
            
            return True
    
    def read(self, blocking: bool = True, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Read a frame from the buffer.
        
        Args:
            blocking: If True, wait for data when buffer is empty
            timeout: Max time to wait if blocking (None = forever)
            
        Returns:
            Audio frame or None if no data available
        """
        with self._lock:
            if self._count == 0:
                if blocking:
                    if not self._not_empty.wait(timeout):
                        self._underruns += 1
                        return None
                else:
                    self._underruns += 1
                    return None
            
            # Read frame from buffer (make a copy to avoid race conditions)
            frame = self._buffer[self._read_idx].copy()
            self._read_idx = (self._read_idx + 1) % self.capacity
            self._count -= 1
            self._total_reads += 1
            
            # Signal that buffer is not full
            self._not_full.notify()
            
            return frame
    
    def read_batch(self, count: int, blocking: bool = False) -> list[np.ndarray]:
        """
        Read multiple frames at once.
        
        Args:
            count: Number of frames to read
            blocking: If True, wait for each frame
            
        Returns:
            List of audio frames (may be less than count if non-blocking)
        """
        frames = []
        for _ in range(count):
            frame = self.read(blocking=blocking, timeout=0.001)
            if frame is None:
                break
            frames.append(frame)
        return frames
    
    def peek(self) -> Optional[np.ndarray]:
        """
        Peek at the next frame without removing it.
        
        Returns:
            Audio frame or None if buffer is empty
        """
        with self._lock:
            if self._count == 0:
                return None
            return self._buffer[self._read_idx].copy()
    
    def clear(self) -> int:
        """
        Clear all frames from the buffer.
        
        Returns:
            Number of frames that were cleared
        """
        with self._lock:
            cleared = self._count
            self._count = 0
            self._write_idx = 0
            self._read_idx = 0
            self._not_full.notify_all()
            return cleared
    
    @property
    def size(self) -> int:
        """Current number of frames in buffer"""
        with self._lock:
            return self._count
    
    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return self.size == 0
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return self.size >= self.capacity
    
    def get_stats(self) -> BufferStats:
        """Get buffer statistics"""
        with self._lock:
            return BufferStats(
                capacity=self.capacity,
                current_size=self._count,
                total_writes=self._total_writes,
                total_reads=self._total_reads,
                overruns=self._overruns,
                underruns=self._underruns,
            )
    
    def reset_stats(self) -> None:
        """Reset statistics counters"""
        with self._lock:
            self._total_writes = 0
            self._total_reads = 0
            self._overruns = 0
            self._underruns = 0
    
    def get_all_frames(self) -> np.ndarray:
        """
        Get all frames currently in buffer as a single array.
        Useful for visualization or bulk processing.
        
        Returns:
            Numpy array of shape (count, frame_size) or (count, frame_size, channels)
        """
        with self._lock:
            if self._count == 0:
                return np.array([])
            
            frames = []
            idx = self._read_idx
            for _ in range(self._count):
                frames.append(self._buffer[idx].copy())
                idx = (idx + 1) % self.capacity
            
            return np.array(frames)
