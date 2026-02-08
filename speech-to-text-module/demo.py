"""
STT Pipeline Demo
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'audio-capture-module'))

from src.stt_pipeline import STTPipeline, create_stt_pipeline


def demo_with_audio_capture():
    print("=" * 60)
    print("Speech-to-Text Pipeline Demo")
    print("=" * 60)
    
    try:
        from src.audio_capture import AudioCaptureModule
        capture = AudioCaptureModule(enable_noise_handling=True, enable_vad=True)
        capture.print_devices()
        if not capture.test_device():
            print("ERROR: Audio device test failed!")
            return
    except ImportError as e:
        print(f"Audio capture not available: {e}")
        demo_standalone()
        return
    
    stt = create_stt_pipeline(model_size="base", language="en", device="auto", log_path="./logs")
    print(f"Model: {stt.model_info}")
    
    print("\nLoading Whisper model...")
    if not stt.load_model():
        print("ERROR: Failed to load model!")
        return
    
    capture.start()
    stt.start(capture)
    stt.on_result(lambda r: print(f"\n>>> [{r.timestamp}] ({r.confidence:.0%}) {r.text}"))
    
    print("\n" + "-" * 60)
    print("LISTENING... Press Ctrl+C to stop")
    print("-" * 60 + "\n")
    
    try:
        for _ in stt.results():
            pass
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stt.stop()
        capture.stop()
    
    stats = stt.get_stats()
    print(f"\nStats: {stats.total_transcriptions} transcriptions, {stats.avg_confidence:.0%} avg confidence")


def demo_standalone():
    import numpy as np
    print("Running standalone demo...")
    stt = create_stt_pipeline(model_size="base", device="auto")
    if not stt.load_model():
        print("ERROR: Failed to load model!")
        return
    
    audio = np.random.randn(16000 * 3).astype(np.float32) * 0.001
    result = stt.transcribe(audio)
    print(f"Result: {result}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standalone", action="store_true")
    args = parser.parse_args()
    
    if args.standalone:
        demo_standalone()
    else:
        demo_with_audio_capture()


if __name__ == "__main__":
    main()
