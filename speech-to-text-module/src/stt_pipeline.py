"""
STT Pipeline Module

Main pipeline integrating audio capture with speech recognition.
"""

import logging
import time
import threading
from typing import Optional, Callable, Generator
from pathlib import Path
from queue import Queue, Empty
import numpy as np

from .stt_config import STTConfig
from .transcription import TranscriptionResult, TranscriptionStats
from .whisper_engine import WhisperEngine
from .speech_processor import SpeechProcessor
from .transcription_logger import TranscriptionLogger

logger = logging.getLogger(__name__)


class STTPipeline:
    """Complete speech-to-text pipeline."""
    
    def __init__(self, config: Optional[STTConfig] = None, log_path: Optional[Path] = None, source_id: str = "mic_01"):
        self.config = config or STTConfig.for_realtime()
        self.source_id = source_id
        self.engine = WhisperEngine(self.config)
        self.processor = SpeechProcessor(self.config)
        self.logger = TranscriptionLogger(log_path=log_path or Path("./logs"))
        self._audio_buffer: list = []
        self._chunk_samples = int(self.config.sample_rate * self.config.chunk_duration)
        self._result_queue: Queue = Queue(maxsize=100)
        self._running = False
        self._processing_thread = None
        self._audio_capture = None
        self._stats = TranscriptionStats()
        self._start_time = None
        self._on_result = None
    
    def load_model(self) -> bool:
        return self.engine.load_model()
    
    def start(self, audio_capture=None) -> bool:
        if self._running:
            return True
        if not self.engine.is_loaded and not self.load_model():
            return False
        self.logger.start()
        self._running = True
        self._start_time = time.time()
        self._audio_capture = audio_capture
        if audio_capture:
            self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self._processing_thread.start()
        return True
    
    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)
        final = self.processor.force_finalize()
        if final:
            self._handle_result(final)
        self.logger.stop()
    
    def transcribe(self, audio: np.ndarray, source: Optional[str] = None) -> TranscriptionResult:
        result = self.engine.transcribe(audio, source or self.source_id)
        processed = self.processor.process(result)
        result = processed or result
        self._stats.update(result)
        if result and result.text:
            self.logger.log(result)
        return result
    
    def transcribe_chunk(self, audio_chunk: np.ndarray) -> Optional[TranscriptionResult]:
        self._audio_buffer.append(audio_chunk)
        total = sum(len(c) for c in self._audio_buffer)
        if total >= self._chunk_samples:
            audio = np.concatenate(self._audio_buffer)
            self._audio_buffer = []
            if len(audio) > self._chunk_samples:
                self._audio_buffer = [audio[self._chunk_samples:]]
                audio = audio[:self._chunk_samples]
            return self.transcribe(audio)
        return None
    
    def results(self, timeout: float = 1.0) -> Generator[TranscriptionResult, None, None]:
        while self._running:
            try:
                yield self._result_queue.get(timeout=timeout)
            except Empty:
                continue
    
    def _processing_loop(self) -> None:
        while self._running and self._audio_capture:
            frame = self._audio_capture.read_frame(blocking=True, timeout=0.5, process=True)
            if frame is None:
                continue
            result = self.transcribe_chunk(frame)
            if result and result.text:
                self._handle_result(result)
    
    def _handle_result(self, result: TranscriptionResult) -> None:
        try:
            self._result_queue.put_nowait(result)
        except:
            pass
        if self._on_result:
            self._on_result(result)
        logger.info(f"Transcribed: {result}")
    
    def on_result(self, callback: Callable[[TranscriptionResult], None]) -> None:
        self._on_result = callback
    
    def get_stats(self) -> TranscriptionStats:
        return self._stats
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def model_info(self) -> dict:
        return self.engine.model_info
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


def create_stt_pipeline(model_size: str = "base", language: str = "en", 
                        device: str = "auto", log_path: Optional[str] = None) -> STTPipeline:
    from .stt_config import WhisperModelSize
    size_map = {s.value: s for s in WhisperModelSize}
    config = STTConfig(model_size=size_map.get(model_size, WhisperModelSize.BASE), 
                       language=language, device=device)
    return STTPipeline(config=config, log_path=Path(log_path) if log_path else None)
