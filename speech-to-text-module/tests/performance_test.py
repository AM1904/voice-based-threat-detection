"""Performance Validation Test"""

import sys
import os
import time
import argparse
import logging
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stt_pipeline import create_stt_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_test(duration: int = 60):
    logger.info("STT Performance Validation")
    
    stt = create_stt_pipeline(model_size="base", device="auto")
    if not stt.load_model():
        logger.error("Failed to load model")
        return {"passed": False}
    
    # Synthetic test
    latencies = []
    for i in range(5):
        audio = np.random.randn(16000 * 2).astype(np.float32) * 0.01
        start = time.time()
        result = stt.transcribe(audio)
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        logger.info(f"Test {i+1}: {latency:.0f}ms")
    
    avg = sum(latencies) / len(latencies)
    logger.info(f"\nAverage latency: {avg:.0f}ms")
    passed = avg < 2000
    logger.info(f"RESULT: {'PASSED' if passed else 'FAILED'}")
    return {"passed": passed, "avg_latency_ms": avg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=60)
    args = parser.parse_args()
    result = run_test(args.duration)
    sys.exit(0 if result.get("passed") else 1)
