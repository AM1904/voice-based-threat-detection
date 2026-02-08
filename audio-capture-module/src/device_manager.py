"""
Device Manager Module

Handles detection, enumeration, and configuration of audio input devices.
Supports USB microphones, built-in microphones, and IP camera audio.
"""

import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    raise ImportError("sounddevice is required. Install with: pip install sounddevice")

from .audio_config import AudioConfig

logger = logging.getLogger(__name__)


@dataclass
class AudioDevice:
    """Represents an audio input device"""
    index: int
    name: str
    channels: int
    default_sample_rate: float
    is_default: bool
    host_api: str
    
    def __str__(self) -> str:
        default_marker = " [DEFAULT]" if self.is_default else ""
        return f"[{self.index}] {self.name} ({self.channels}ch, {int(self.default_sample_rate)}Hz){default_marker}"


class DeviceManager:
    """
    Manages audio input device detection and configuration.
    
    Features:
    - Enumerate all available input devices
    - Auto-detect default device
    - Validate device capabilities
    - Test device connectivity
    """
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize the device manager.
        
        Args:
            config: Audio configuration (optional, uses defaults if not provided)
        """
        self.config = config or AudioConfig()
        self._selected_device: Optional[AudioDevice] = None
        self._devices: list[AudioDevice] = []
        
    def list_devices(self, refresh: bool = True) -> list[AudioDevice]:
        """
        List all available audio input devices.
        
        Args:
            refresh: If True, refresh the device list from hardware
            
        Returns:
            List of AudioDevice objects
        """
        if refresh or not self._devices:
            self._devices = []
            
            try:
                devices = sd.query_devices()
                default_input = sd.default.device[0]  # Default input device index
                
                for i, device in enumerate(devices):
                    # Only include input devices
                    if device['max_input_channels'] > 0:
                        host_api = sd.query_hostapis(device['hostapi'])['name']
                        
                        audio_device = AudioDevice(
                            index=i,
                            name=device['name'],
                            channels=device['max_input_channels'],
                            default_sample_rate=device['default_samplerate'],
                            is_default=(i == default_input),
                            host_api=host_api,
                        )
                        self._devices.append(audio_device)
                        
                logger.info(f"Found {len(self._devices)} audio input devices")
                
            except Exception as e:
                logger.error(f"Failed to enumerate audio devices: {e}")
                raise
                
        return self._devices
    
    def get_default_device(self) -> Optional[AudioDevice]:
        """
        Get the system's default audio input device.
        
        Returns:
            The default AudioDevice or None if not found
        """
        devices = self.list_devices(refresh=False)
        for device in devices:
            if device.is_default:
                return device
        return devices[0] if devices else None
    
    def get_device_by_index(self, index: int) -> Optional[AudioDevice]:
        """
        Get a device by its index.
        
        Args:
            index: Device index
            
        Returns:
            AudioDevice or None if not found
        """
        devices = self.list_devices(refresh=False)
        for device in devices:
            if device.index == index:
                return device
        return None
    
    def get_device_by_name(self, name: str, partial_match: bool = True) -> Optional[AudioDevice]:
        """
        Get a device by its name.
        
        Args:
            name: Device name to search for
            partial_match: If True, match partial names (case-insensitive)
            
        Returns:
            AudioDevice or None if not found
        """
        devices = self.list_devices(refresh=False)
        name_lower = name.lower()
        
        for device in devices:
            if partial_match:
                if name_lower in device.name.lower():
                    return device
            else:
                if device.name.lower() == name_lower:
                    return device
        return None
    
    def select_device(self, device: Optional[AudioDevice] = None) -> AudioDevice:
        """
        Select an audio device for capture.
        
        Args:
            device: Device to select (None = auto-select based on config)
            
        Returns:
            The selected AudioDevice
            
        Raises:
            ValueError: If no suitable device is found
        """
        if device:
            self._selected_device = device
        elif self.config.device_index is not None:
            self._selected_device = self.get_device_by_index(self.config.device_index)
        elif self.config.device_name:
            self._selected_device = self.get_device_by_name(self.config.device_name)
        else:
            self._selected_device = self.get_default_device()
            
        if not self._selected_device:
            raise ValueError("No suitable audio input device found")
            
        logger.info(f"Selected device: {self._selected_device}")
        return self._selected_device
    
    @property
    def selected_device(self) -> Optional[AudioDevice]:
        """Get the currently selected device"""
        return self._selected_device
    
    def validate_device(self, device: Optional[AudioDevice] = None) -> tuple[bool, list[str]]:
        """
        Validate that a device meets the configuration requirements.
        
        Args:
            device: Device to validate (None = use selected device)
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        device = device or self._selected_device
        issues = []
        
        if not device:
            return False, ["No device selected"]
            
        # Check channel support
        if device.channels < self.config.channels:
            issues.append(
                f"Device has {device.channels} channels, config requires {self.config.channels}"
            )
            
        # Check sample rate support (we'll verify this during stream test)
        if device.default_sample_rate < self.config.sample_rate:
            issues.append(
                f"Device default rate {device.default_sample_rate}Hz is below "
                f"configured rate {self.config.sample_rate}Hz (may still work)"
            )
            
        return len(issues) == 0, issues
    
    def test_device(self, device: Optional[AudioDevice] = None, duration: float = 1.0) -> bool:
        """
        Test a device by attempting to capture audio.
        
        Args:
            device: Device to test (None = use selected device)
            duration: Test duration in seconds
            
        Returns:
            True if test passed, False otherwise
        """
        device = device or self._selected_device
        
        if not device:
            logger.error("No device to test")
            return False
            
        logger.info(f"Testing device: {device.name} for {duration}s...")
        
        try:
            # Attempt to record a short sample
            recording = sd.rec(
                frames=int(self.config.sample_rate * duration),
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype_numpy,
                device=device.index,
            )
            sd.wait()
            
            # Check if we got valid audio data
            if recording is None or len(recording) == 0:
                logger.error("No audio data received")
                return False
                
            # Check if audio is not all zeros (device actually working)
            rms = np.sqrt(np.mean(recording ** 2))
            if rms < 1e-10:
                logger.warning("Audio data appears to be silent (RMS very low)")
                # Still return True as device is connected
                
            logger.info(f"Device test passed. RMS level: {rms:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"Device test failed: {e}")
            return False
    
    def print_devices(self) -> None:
        """Print all available devices to console"""
        devices = self.list_devices()
        print("\n=== Available Audio Input Devices ===")
        if not devices:
            print("No input devices found!")
        else:
            for device in devices:
                print(f"  {device}")
        print("=" * 40)
