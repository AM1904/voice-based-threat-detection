"""
Quick Demo Script

Simple demonstration of the audio capture module.
Captures audio for a short duration and displays statistics.

Usage:
    python demo.py
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.audio_capture import AudioCaptureModule
from src.audio_config import AudioConfig


def main():
    print("=" * 50)
    print("Audio Capture Module - Quick Demo")
    print("=" * 50)
    print()
    
    # Create capture module
    capture = AudioCaptureModule(
        enable_noise_handling=True,
        enable_vad=True,
    )
    
    # List available devices
    print("Available audio devices:")
    capture.print_devices()
    print()
    
    # Test device
    print("Testing audio device...")
    if not capture.test_device():
        print("ERROR: Device test failed!")
        return
    print("Device test passed!")
    print()
    
    # Run capture for 10 seconds
    duration = 10
    print(f"Starting capture for {duration} seconds...")
    print("Speak into your microphone to test!")
    print("-" * 50)
    
    capture.start()
    start_time = time.time()
    frame_count = 0
    speaking_frames = 0
    
    try:
        while (time.time() - start_time) < duration:
            frame = capture.read_frame(timeout=0.5)
            
            if frame is not None:
                frame_count += 1
                
                # Check audio level
                level = capture.get_audio_level()
                if level and not level['is_silent']:
                    speaking_frames += 1
                
                # Print level bar every 10 frames
                if frame_count % 10 == 0 and level:
                    bar_len = int(min(level['rms'] * 100, 50))
                    bar = "█" * bar_len + "░" * (50 - bar_len)
                    status = "🗣️" if capture.vad and capture.vad.is_speaking else "  "
                    print(f"\r{status} [{bar}] {level['rms_db']:.1f}dB", end="", flush=True)
        
        print()
        print("-" * 50)
        
    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        capture.stop()
    
    # Print final stats
    stats = capture.get_stats()
    print()
    print("Capture Statistics:")
    print(f"  Duration: {stats.duration_seconds:.1f}s")
    print(f"  Frames captured: {stats.frames_captured}")
    print(f"  Frame rate: {stats.frames_per_second:.1f} fps")
    print(f"  Average latency: {stats.avg_latency_ms:.2f}ms")
    print(f"  Buffer fill: {stats.buffer_fill_percent:.1f}%")
    print(f"  Peak RMS: {stats.peak_rms:.4f}")
    print(f"  Speaking frames: {speaking_frames} ({speaking_frames/frame_count*100:.1f}%)")
    print()
    
    if stats.buffer_overruns > 0 or stats.xrun_count > 0:
        print(f"  ⚠️  Overruns: {stats.buffer_overruns}, XRuns: {stats.xrun_count}")
    else:
        print("  ✓ No buffer issues detected")
    
    print()
    print("Demo complete!")


if __name__ == "__main__":
    main()
