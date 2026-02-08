"""
Waveform Visualization Tool

Real-time visualization of captured audio waveforms.
Useful for validating audio capture quality and detecting issues.

Usage:
    python -m tests.visualize
"""

import sys
import os
import time
import threading
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.widgets import Button
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")

from audio_capture import AudioCaptureModule
from audio_config import AudioConfig


class AudioVisualizer:
    """
    Real-time audio waveform visualization.
    
    Displays:
    - Live waveform
    - RMS level meter
    - Spectrogram (optional)
    """
    
    def __init__(self, capture: AudioCaptureModule, display_seconds: float = 2.0):
        """
        Initialize visualizer.
        
        Args:
            capture: AudioCaptureModule instance
            display_seconds: Seconds of audio to display
        """
        self.capture = capture
        self.display_seconds = display_seconds
        
        # Calculate buffer size
        samples_to_display = int(capture.config.sample_rate * display_seconds)
        self.waveform_buffer = np.zeros(samples_to_display)
        
        # For level meter
        self.level_history = np.zeros(100)
        
        # State
        self.running = False
        self._lock = threading.Lock()
        
    def update_buffer(self, frame: np.ndarray):
        """Update waveform buffer with new frame"""
        with self._lock:
            # Roll buffer and append new data
            self.waveform_buffer = np.roll(self.waveform_buffer, -len(frame))
            self.waveform_buffer[-len(frame):] = frame
            
            # Update level history
            rms = float(np.sqrt(np.mean(frame ** 2)))
            self.level_history = np.roll(self.level_history, -1)
            self.level_history[-1] = rms
    
    def run(self):
        """Run the visualization"""
        if not HAS_MATPLOTLIB:
            print("Matplotlib is required for visualization")
            return
        
        self.running = True
        
        # Setup figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        fig.suptitle('Audio Capture Visualization', fontsize=14)
        
        # Waveform plot
        ax_wave = axes[0]
        ax_wave.set_title('Waveform')
        ax_wave.set_ylim(-1, 1)
        ax_wave.set_xlim(0, len(self.waveform_buffer))
        ax_wave.set_ylabel('Amplitude')
        line_wave, = ax_wave.plot([], [], 'b-', linewidth=0.5)
        
        # Level meter
        ax_level = axes[1]
        ax_level.set_title('Audio Level (RMS)')
        ax_level.set_ylim(0, 0.5)
        ax_level.set_xlim(0, len(self.level_history))
        ax_level.set_ylabel('RMS')
        ax_level.set_xlabel('Time')
        ax_level.axhline(y=0.01, color='r', linestyle='--', label='Silence threshold')
        line_level, = ax_level.plot([], [], 'g-', linewidth=1)
        ax_level.legend()
        
        # Stats text
        stats_text = ax_wave.text(0.02, 0.95, '', transform=ax_wave.transAxes,
                                   fontsize=10, verticalalignment='top',
                                   fontfamily='monospace')
        
        plt.tight_layout()
        
        # Start capture thread
        capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        capture_thread.start()
        
        def init():
            line_wave.set_data([], [])
            line_level.set_data([], [])
            return line_wave, line_level, stats_text
        
        def animate(i):
            with self._lock:
                # Update waveform
                x_wave = np.arange(len(self.waveform_buffer))
                line_wave.set_data(x_wave, self.waveform_buffer)
                
                # Update level
                x_level = np.arange(len(self.level_history))
                line_level.set_data(x_level, self.level_history)
                
                # Update stats
                stats = self.capture.get_stats()
                current_rms = self.level_history[-1]
                stats_str = (
                    f"Frames: {stats.frames_captured:,}  "
                    f"FPS: {stats.frames_per_second:.1f}  "
                    f"Latency: {stats.avg_latency_ms:.1f}ms  "
                    f"RMS: {current_rms:.4f}  "
                    f"Buffer: {stats.buffer_fill_percent:.0f}%"
                )
                stats_text.set_text(stats_str)
            
            return line_wave, line_level, stats_text
        
        # Animation
        ani = animation.FuncAnimation(
            fig, animate, init_func=init,
            interval=50,  # 20 fps
            blit=True
        )
        
        def on_close(event):
            self.running = False
            self.capture.stop()
        
        fig.canvas.mpl_connect('close_event', on_close)
        
        plt.show()
    
    def _capture_loop(self):
        """Background thread for audio capture"""
        self.capture.start()
        
        while self.running:
            frame = self.capture.read_frame(timeout=0.1)
            if frame is not None:
                self.update_buffer(frame)
        
        self.capture.stop()


def main():
    print("Audio Capture Visualization")
    print("=" * 40)
    
    if not HAS_MATPLOTLIB:
        print("ERROR: matplotlib is required")
        print("Install with: pip install matplotlib")
        sys.exit(1)
    
    # Create capture
    capture = AudioCaptureModule(
        enable_noise_handling=False,  # Raw audio for visualization
    )
    
    # List devices
    capture.print_devices()
    
    # Create and run visualizer
    visualizer = AudioVisualizer(capture, display_seconds=2.0)
    
    print()
    print("Starting visualization...")
    print("Close the window to stop.")
    print()
    
    try:
        visualizer.run()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
