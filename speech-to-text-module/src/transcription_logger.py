"""
Transcription Logger Module

Handles logging of transcription outputs for validation and debugging.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
import threading
from queue import Queue

from .transcription import TranscriptionResult, TranscriptionStats

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Single log entry for a transcription"""
    timestamp: str
    text: str
    confidence: float
    processing_delay_ms: float
    audio_duration: float
    source: str
    language: str
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "text": self.text,
            "confidence": self.confidence,
            "processing_delay_ms": self.processing_delay_ms,
            "source": self.source,
        }
    
    def to_line(self) -> str:
        return f"[{self.timestamp}] ({self.confidence:.0%}) [{self.processing_delay_ms:.0f}ms] {self.text}"


class TranscriptionLogger:
    """Logs transcription outputs to files."""
    
    def __init__(self, log_path: Optional[Path] = None, enable_json: bool = True, enable_text: bool = True):
        self.log_path = Path(log_path) if log_path else Path("./logs")
        self.enable_json = enable_json
        self.enable_text = enable_text
        self.log_path.mkdir(parents=True, exist_ok=True)
        self._stats = TranscriptionStats()
        self._entries: List[LogEntry] = []
        self._buffer: Queue = Queue(maxsize=100)
        self._json_file = None
        self._text_file = None
        self._running = False
        self._writer_thread = None
        self._lock = threading.Lock()
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._json_first_entry = True
    
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._open_files()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
    
    def stop(self) -> None:
        self._running = False
        if self._writer_thread:
            self._writer_thread.join(timeout=5.0)
        self._flush_buffer()
        self._close_files()
    
    def log(self, result: TranscriptionResult) -> None:
        if not result.text.strip():
            return
        entry = LogEntry(
            timestamp=result.timestamp, text=result.text, confidence=result.confidence,
            processing_delay_ms=result.processing_time_ms, audio_duration=result.audio_duration,
            source=result.source, language=result.language,
        )
        with self._lock:
            self._stats.update(result)
            self._entries.append(entry)
        try:
            self._buffer.put_nowait(entry)
        except:
            self._write_entry(entry)
    
    def _open_files(self) -> None:
        if self.enable_json:
            self._json_file = open(self.log_path / f"transcriptions_{self._session_id}.json", "w", encoding="utf-8")
            self._json_file.write("[\n")
        if self.enable_text:
            self._text_file = open(self.log_path / f"transcriptions_{self._session_id}.txt", "w", encoding="utf-8")
            self._text_file.write(f"=== Transcription Log ===\nSession: {self._session_id}\n\n")
    
    def _close_files(self) -> None:
        if self._json_file:
            self._json_file.write("\n]")
            self._json_file.close()
        if self._text_file:
            self._text_file.close()
    
    def _writer_loop(self) -> None:
        while self._running:
            try:
                entry = self._buffer.get(timeout=0.5)
                self._write_entry(entry)
            except:
                continue
    
    def _write_entry(self, entry: LogEntry) -> None:
        if self._json_file:
            prefix = ",\n" if not self._json_first_entry else ""
            self._json_first_entry = False
            self._json_file.write(f"{prefix}  {json.dumps(entry.to_dict())}")
            self._json_file.flush()
        if self._text_file:
            self._text_file.write(entry.to_line() + "\n")
            self._text_file.flush()
    
    def _flush_buffer(self) -> None:
        while not self._buffer.empty():
            try:
                self._write_entry(self._buffer.get_nowait())
            except:
                break
    
    def get_stats(self) -> TranscriptionStats:
        with self._lock:
            return self._stats
    
    def get_session_summary(self) -> dict:
        return {"session_id": self._session_id, "stats": self.get_stats().to_dict()}
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
