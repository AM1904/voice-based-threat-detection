"""
Audio Capture Module
====================
Captures live audio from the system microphone using PyAudio.
Provides a callback-based architecture for downstream consumers
(keyword detector, sound classifier, etc.)

Usage:
    from audio_engine.capture import AudioCapture

    def on_audio(data, frame_count):
        # Process audio data
        print(f"Received {frame_count} frames")

    capture = AudioCapture()
    capture.add_listener(on_audio)
    capture.start()
"""

import pyaudio
import numpy as np
import threading
import time
import json
import os


# ─── Audio Configuration ───────────────────────────────────────────────
RATE = 16000           # 16kHz — required by Vosk
CHANNELS = 1           # Mono
FORMAT = pyaudio.paInt16
CHUNK = 4000           # ~250ms of audio per chunk at 16kHz
DEVICE_INDEX = None    # None = default mic


class AudioCapture:
    """
    Captures audio from the system microphone and dispatches
    audio chunks to registered listener callbacks.

    Attributes:
        rate (int): Sample rate in Hz (default 16000 for Vosk)
        channels (int): Number of audio channels (1 = mono)
        chunk (int): Number of frames per buffer
        device_index (int|None): PyAudio device index, None for default
    """

    def __init__(self, rate=RATE, channels=CHANNELS, chunk=CHUNK,
                 device_index=DEVICE_INDEX):
        self.rate = rate
        self.channels = channels
        self.chunk = chunk
        self.device_index = device_index

        self._pyaudio = None
        self._stream = None
        self._running = False
        self._thread = None
        self._listeners = []

    # ─── Listener Management ────────────────────────────────────────
    def add_listener(self, callback):
        """
        Register a callback to receive audio data.

        Args:
            callback: Function(data: bytes, frame_count: int) -> None
        """
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback):
        """Remove a previously registered callback."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify_listeners(self, data, frame_count):
        """Dispatch audio data to all registered listeners."""
        for listener in self._listeners:
            try:
                listener(data, frame_count)
            except Exception as e:
                print(f"[AudioCapture] Listener error: {e}")

    # ─── Device Info ────────────────────────────────────────────────
    def list_devices(self):
        """List all available audio input devices."""
        pa = pyaudio.PyAudio()
        devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                devices.append({
                    "index": i,
                    "name": info["name"],
                    "channels": info["maxInputChannels"],
                    "sample_rate": int(info["defaultSampleRate"])
                })
        pa.terminate()
        return devices

    def get_default_device(self):
        """Get info about the default input device."""
        pa = pyaudio.PyAudio()
        try:
            info = pa.get_default_input_device_info()
            return {
                "index": info["index"],
                "name": info["name"],
                "channels": info["maxInputChannels"],
                "sample_rate": int(info["defaultSampleRate"])
            }
        except IOError:
            return None
        finally:
            pa.terminate()

    # ─── Stream Control ─────────────────────────────────────────────
    def start(self, blocking=True):
        """
        Start capturing audio from the microphone.

        Args:
            blocking (bool): If True, blocks until stop() is called.
                             If False, runs in a background thread.
        """
        if self._running:
            print("[AudioCapture] Already running.")
            return

        self._pyaudio = pyaudio.PyAudio()

        # Verify a mic is available
        try:
            if self.device_index is not None:
                self._pyaudio.get_device_info_by_index(self.device_index)
            else:
                self._pyaudio.get_default_input_device_info()
        except IOError:
            print("[AudioCapture] ERROR: No microphone found!")
            self._pyaudio.terminate()
            self._pyaudio = None
            return False

        # Open the audio stream
        try:
            self._stream = self._pyaudio.open(
                format=FORMAT,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk
            )
        except Exception as e:
            print(f"[AudioCapture] ERROR opening stream: {e}")
            self._pyaudio.terminate()
            self._pyaudio = None
            return False

        self._running = True
        print(f"[AudioCapture] Started — rate={self.rate}Hz, "
              f"chunk={self.chunk}, channels={self.channels}")

        if blocking:
            self._capture_loop()
        else:
            self._thread = threading.Thread(target=self._capture_loop,
                                            daemon=True)
            self._thread.start()

        return True

    def stop(self):
        """Stop capturing audio and release resources."""
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None

        print("[AudioCapture] Stopped.")

    def _capture_loop(self):
        """Main capture loop — reads audio chunks and dispatches."""
        while self._running:
            try:
                data = self._stream.read(self.chunk,
                                         exception_on_overflow=False)
                self._notify_listeners(data, self.chunk)
            except IOError as e:
                print(f"[AudioCapture] Stream read error: {e}")
                time.sleep(0.01)
            except Exception as e:
                print(f"[AudioCapture] Unexpected error: {e}")
                break

    # ─── Utility ────────────────────────────────────────────────────
    @staticmethod
    def audio_to_numpy(data):
        """Convert raw audio bytes to a numpy array of int16 samples."""
        return np.frombuffer(data, dtype=np.int16)

    @staticmethod
    def get_amplitude(data):
        """Get the peak amplitude from raw audio bytes."""
        samples = np.frombuffer(data, dtype=np.int16)
        return int(np.max(np.abs(samples))) if len(samples) > 0 else 0

    @staticmethod
    def get_rms(data):
        """Get the RMS (root mean square) level from raw audio bytes."""
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)
        if len(samples) == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples ** 2)))

    @property
    def is_running(self):
        """Check if capture is currently active."""
        return self._running

    def __enter__(self):
        self.start(blocking=False)
        return self

    def __exit__(self, *args):
        self.stop()


# ─── CLI Entry Point ───────────────────────────────────────────────────
if __name__ == "__main__":
    import signal

    def print_audio_stats(data, frame_count):
        amp = AudioCapture.get_amplitude(data)
        rms = AudioCapture.get_rms(data)
        bar = "█" * min(int(rms / 100), 50)
        print(f"\rAmplitude: {amp:6d} | RMS: {rms:8.1f} | {bar:<50}", end="")

    capture = AudioCapture()

    # List available devices
    print("\n📱 Available input devices:")
    for dev in capture.list_devices():
        print(f"  [{dev['index']}] {dev['name']} "
              f"({dev['channels']}ch, {dev['sample_rate']}Hz)")

    default = capture.get_default_device()
    if default:
        print(f"\n🎙️  Default: [{default['index']}] {default['name']}")
    print()

    capture.add_listener(print_audio_stats)

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\n⏹️  Stopping...")
        capture.stop()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("🎤 Recording... Press Ctrl+C to stop.\n")
    capture.start(blocking=True)
