"""
Mic Capture Test
================
Verifies that the microphone is accessible and audio stream
reads correctly. Records a short clip and prints statistics.

Usage:
    python tests/test_capture.py
"""

import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_engine.capture import AudioCapture


def test_device_listing():
    """Test 1: Can we enumerate audio devices?"""
    print("=" * 60)
    print("TEST 1: Device Listing")
    print("=" * 60)

    capture = AudioCapture()
    devices = capture.list_devices()

    if not devices:
        print("❌ FAIL — No input devices found!")
        return False

    print(f"✅ PASS — Found {len(devices)} input device(s):")
    for dev in devices:
        print(f"   [{dev['index']}] {dev['name']} "
              f"({dev['channels']}ch, {dev['sample_rate']}Hz)")

    default = capture.get_default_device()
    if default:
        print(f"\n   Default: [{default['index']}] {default['name']}")
    else:
        print("\n   ⚠️  No default input device set")

    return True


def test_audio_capture():
    """Test 2: Can we capture audio from the mic?"""
    print("\n" + "=" * 60)
    print("TEST 2: Audio Capture (2 seconds)")
    print("=" * 60)

    captured_chunks = []
    total_frames = 0

    def collect_audio(data, frame_count):
        nonlocal total_frames
        captured_chunks.append(data)
        total_frames += frame_count

    capture = AudioCapture()
    capture.add_listener(collect_audio)

    # Start non-blocking capture
    result = capture.start(blocking=False)
    if not result:
        print("❌ FAIL — Could not start audio capture!")
        return False

    print("🎤 Recording for 2 seconds...")
    time.sleep(2.0)
    capture.stop()

    if not captured_chunks:
        print("❌ FAIL — No audio data captured!")
        return False

    # Analyze captured audio
    all_audio = b"".join(captured_chunks)
    samples = np.frombuffer(all_audio, dtype=np.int16)

    duration = len(samples) / capture.rate
    peak_amplitude = int(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    is_silent = peak_amplitude < 50  # Very low threshold

    print(f"\n📊 Capture Results:")
    print(f"   Chunks captured:  {len(captured_chunks)}")
    print(f"   Total frames:     {total_frames}")
    print(f"   Total samples:    {len(samples)}")
    print(f"   Duration:         {duration:.2f}s")
    print(f"   Peak amplitude:   {peak_amplitude}")
    print(f"   RMS level:        {rms:.1f}")
    print(f"   Data size:        {len(all_audio)} bytes")

    if is_silent:
        print("\n⚠️  WARNING — Audio appears to be silent.")
        print("   Check that your microphone is not muted.")
    else:
        print(f"\n✅ PASS — Audio captured successfully! "
              f"(peak={peak_amplitude}, rms={rms:.0f})")

    return True


def test_audio_utilities():
    """Test 3: Do the audio utility functions work?"""
    print("\n" + "=" * 60)
    print("TEST 3: Audio Utility Functions")
    print("=" * 60)

    # Create a test sine wave (440Hz, 1 second)
    t = np.linspace(0, 1.0, 16000, dtype=np.float64)
    sine_wave = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    test_data = sine_wave.tobytes()

    amp = AudioCapture.get_amplitude(test_data)
    rms = AudioCapture.get_rms(test_data)
    arr = AudioCapture.audio_to_numpy(test_data)

    print(f"   Sine wave (440Hz) amplitude: {amp}")
    print(f"   Sine wave (440Hz) RMS:       {rms:.1f}")
    print(f"   Numpy array shape:           {arr.shape}")
    print(f"   Numpy array dtype:           {arr.dtype}")

    if amp > 0 and rms > 0 and len(arr) == 16000:
        print("\n✅ PASS — Audio utilities working correctly!")
        return True
    else:
        print("\n❌ FAIL — Audio utility functions returned unexpected values!")
        return False


def main():
    print("\n🔊 WATZS — Microphone Capture Test Suite")
    print("━" * 60)

    results = []

    # Test 1: Device listing
    results.append(("Device Listing", test_device_listing()))

    # Test 2: Audio capture
    results.append(("Audio Capture", test_audio_capture()))

    # Test 3: Utility functions
    results.append(("Audio Utilities", test_audio_utilities()))

    # Summary
    print("\n" + "━" * 60)
    print("📋 TEST SUMMARY")
    print("━" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} — {name}")

    print(f"\n   Result: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Mic capture is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check your microphone setup.")

    print()
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
