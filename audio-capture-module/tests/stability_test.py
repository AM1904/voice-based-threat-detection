"""
Stability Test Script

Runs the audio capture module continuously for an extended duration
to validate stream stability, memory usage, and identify potential issues.

Usage:
    python -m tests.stability_test --duration 1800  # 30 minutes
"""

import argparse
import time
import sys
import os
import psutil
import logging
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from audio_capture import AudioCaptureModule, create_capture_module
from audio_config import AudioConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), '..', 'logs', 'stability_test.log')
        )
    ]
)
logger = logging.getLogger(__name__)


class StabilityTester:
    """
    Runs extended stability tests on the audio capture module.
    
    Monitors:
    - Frame capture rate
    - Buffer overruns/underruns
    - Memory usage
    - Latency trends
    - Audio gaps
    """
    
    def __init__(self, duration_seconds: float = 1800):
        """
        Initialize the stability tester.
        
        Args:
            duration_seconds: Test duration (default: 30 minutes)
        """
        self.duration = duration_seconds
        self.capture: AudioCaptureModule = None
        
        # Metrics tracking
        self.start_time = None
        self.frames_processed = 0
        self.silent_frames = 0
        self.gap_count = 0
        self.last_frame_time = None
        
        # Memory tracking
        self.initial_memory = 0
        self.peak_memory = 0
        self.memory_samples = []
        
        # Timing
        self.status_interval = 30  # seconds
        self.last_status_time = 0
        
    def run(self) -> dict:
        """
        Run the stability test.
        
        Returns:
            Dict with test results
        """
        logger.info("=" * 60)
        logger.info("STABILITY TEST STARTING")
        logger.info(f"Duration: {self.duration / 60:.1f} minutes")
        logger.info("=" * 60)
        
        # Create capture module
        self.capture = create_capture_module(
            sample_rate=16000,
            frame_size=1024,
            buffer_seconds=5.0,
        )
        
        # List and select device
        self.capture.print_devices()
        
        # Test device first
        logger.info("Testing audio device...")
        if not self.capture.test_device():
            logger.error("Device test failed!")
            return {"success": False, "error": "Device test failed"}
        
        # Record initial memory
        process = psutil.Process()
        self.initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        self.peak_memory = self.initial_memory
        
        logger.info(f"Initial memory usage: {self.initial_memory:.1f} MB")
        
        # Start capture
        self.start_time = time.time()
        self.last_status_time = self.start_time
        
        try:
            self.capture.start()
            logger.info("Audio capture started successfully")
            
            # Main test loop
            while (time.time() - self.start_time) < self.duration:
                self._process_iteration()
                
            logger.info("Test duration completed")
            
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        except Exception as e:
            logger.error(f"Test error: {e}")
            return {"success": False, "error": str(e)}
        finally:
            self.capture.stop()
        
        # Generate report
        return self._generate_report()
    
    def _process_iteration(self):
        """Process one iteration of the test loop"""
        current_time = time.time()
        
        # Read frame
        frame = self.capture.read_frame(blocking=True, timeout=0.5)
        
        if frame is not None:
            self.frames_processed += 1
            
            # Check for gaps
            if self.last_frame_time:
                gap = current_time - self.last_frame_time
                expected = self.capture.config.frame_size / self.capture.config.sample_rate
                if gap > expected * 2:  # More than 2x expected interval
                    self.gap_count += 1
                    logger.warning(f"Audio gap detected: {gap * 1000:.1f}ms")
            
            self.last_frame_time = current_time
            
            # Check for silence
            import numpy as np
            rms = float(np.sqrt(np.mean(frame ** 2)))
            if rms < 0.01:
                self.silent_frames += 1
        
        # Periodic status update
        if current_time - self.last_status_time >= self.status_interval:
            self._log_status()
            self.last_status_time = current_time
    
    def _log_status(self):
        """Log current status"""
        elapsed = time.time() - (self.start_time or time.time())
        remaining = self.duration - elapsed
        
        # Memory check
        process = psutil.Process()
        current_memory = process.memory_info().rss / (1024 * 1024)
        self.peak_memory = max(self.peak_memory, current_memory)
        self.memory_samples.append(current_memory)
        
        # Get capture stats
        stats = self.capture.get_stats()
        
        logger.info("-" * 40)
        logger.info(f"Elapsed: {timedelta(seconds=int(elapsed))}")
        logger.info(f"Remaining: {timedelta(seconds=int(remaining))}")
        logger.info(f"Frames processed: {self.frames_processed}")
        logger.info(f"Frames/second: {stats.frames_per_second:.1f}")
        logger.info(f"Buffer fill: {stats.buffer_fill_percent:.1f}%")
        logger.info(f"Latency: {stats.avg_latency_ms:.2f}ms")
        logger.info(f"Memory: {current_memory:.1f} MB (peak: {self.peak_memory:.1f} MB)")
        logger.info(f"Gaps detected: {self.gap_count}")
        logger.info(f"Buffer overruns: {stats.buffer_overruns}")
        
        # Health check
        if not stats.is_healthy():
            logger.warning("HEALTH CHECK FAILED")
    
    def _generate_report(self) -> dict:
        """Generate final test report"""
        elapsed = time.time() - (self.start_time or time.time())
        stats = self.capture.get_stats()
        
        # Calculate metrics
        memory_growth = self.peak_memory - self.initial_memory
        avg_memory = sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0
        
        report = {
            "success": True,
            "duration_seconds": elapsed,
            "duration_formatted": str(timedelta(seconds=int(elapsed))),
            "frames_processed": self.frames_processed,
            "frames_per_second": self.frames_processed / elapsed if elapsed > 0 else 0,
            "silent_frames": self.silent_frames,
            "silent_percentage": (self.silent_frames / self.frames_processed * 100) if self.frames_processed > 0 else 0,
            "gap_count": self.gap_count,
            "buffer_overruns": stats.buffer_overruns,
            "buffer_underruns": stats.buffer_underruns,
            "xrun_count": stats.xrun_count,
            "callback_errors": stats.callback_errors,
            "avg_latency_ms": stats.avg_latency_ms,
            "initial_memory_mb": self.initial_memory,
            "peak_memory_mb": self.peak_memory,
            "avg_memory_mb": avg_memory,
            "memory_growth_mb": memory_growth,
        }
        
        # Determine pass/fail
        issues = []
        if report["gap_count"] > 10:
            issues.append(f"Too many audio gaps: {report['gap_count']}")
        if report["buffer_overruns"] > 100:
            issues.append(f"High buffer overruns: {report['buffer_overruns']}")
        if report["memory_growth_mb"] > 100:
            issues.append(f"Memory growth concerning: {report['memory_growth_mb']:.1f} MB")
        if report["callback_errors"] > 0:
            issues.append(f"Callback errors: {report['callback_errors']}")
        if report["avg_latency_ms"] > 100:
            issues.append(f"High latency: {report['avg_latency_ms']:.2f}ms")
        
        report["issues"] = issues
        report["passed"] = len(issues) == 0
        
        # Print report
        self._print_report(report)
        
        return report
    
    def _print_report(self, report: dict):
        """Print formatted test report"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("STABILITY TEST REPORT")
        logger.info("=" * 60)
        logger.info(f"Duration: {report['duration_formatted']}")
        logger.info(f"Total frames: {report['frames_processed']:,}")
        logger.info(f"Frame rate: {report['frames_per_second']:.1f} fps")
        logger.info("")
        logger.info("Audio Quality:")
        logger.info(f"  Silent frames: {report['silent_percentage']:.1f}%")
        logger.info(f"  Audio gaps: {report['gap_count']}")
        logger.info(f"  Avg latency: {report['avg_latency_ms']:.2f}ms")
        logger.info("")
        logger.info("Buffer Health:")
        logger.info(f"  Overruns: {report['buffer_overruns']}")
        logger.info(f"  Underruns: {report['buffer_underruns']}")
        logger.info(f"  XRuns: {report['xrun_count']}")
        logger.info("")
        logger.info("Memory:")
        logger.info(f"  Initial: {report['initial_memory_mb']:.1f} MB")
        logger.info(f"  Peak: {report['peak_memory_mb']:.1f} MB")
        logger.info(f"  Growth: {report['memory_growth_mb']:.1f} MB")
        logger.info("")
        
        if report["passed"]:
            logger.info("RESULT: PASSED ✓")
        else:
            logger.info("RESULT: FAILED ✗")
            logger.info("Issues found:")
            for issue in report["issues"]:
                logger.info(f"  - {issue}")
        
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Audio Capture Stability Test")
    parser.add_argument(
        "--duration",
        type=int,
        default=1800,
        help="Test duration in seconds (default: 1800 = 30 minutes)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test (5 minutes)"
    )
    
    args = parser.parse_args()
    
    duration = 300 if args.quick else args.duration
    
    tester = StabilityTester(duration_seconds=duration)
    result = tester.run()
    
    # Exit with appropriate code
    sys.exit(0 if result.get("passed", False) else 1)


if __name__ == "__main__":
    main()
