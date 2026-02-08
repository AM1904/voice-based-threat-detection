"""
Transcription Data Models

Defines structured output formats for transcription results.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime
import json


@dataclass
class TranscriptionSegment:
    """Represents a single segment of transcribed speech."""
    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    words: Optional[List[dict]] = None
    language: str = "en"
    is_partial: bool = False
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "duration": round(self.duration, 3),
            "confidence": round(self.confidence, 3),
            "language": self.language,
            "is_partial": self.is_partial,
        }


@dataclass
class TranscriptionResult:
    """Complete transcription result with metadata."""
    timestamp: str
    text: str
    confidence: float
    source: str = "mic_01"
    segments: List[TranscriptionSegment] = field(default_factory=list)
    processing_time_ms: float = 0.0
    audio_duration: float = 0.0
    language: str = "en"
    is_final: bool = True
    
    @classmethod
    def create(cls, text: str, confidence: float, source: str = "mic_01",
               segments: Optional[List[TranscriptionSegment]] = None,
               processing_time_ms: float = 0.0, audio_duration: float = 0.0,
               language: str = "en", is_final: bool = True) -> "TranscriptionResult":
        return cls(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            text=text.strip(),
            confidence=confidence,
            source=source,
            segments=segments or [],
            processing_time_ms=processing_time_ms,
            audio_duration=audio_duration,
            language=language,
            is_final=is_final,
        )
    
    @classmethod
    def empty(cls, source: str = "mic_01") -> "TranscriptionResult":
        return cls(timestamp=datetime.now().strftime("%H:%M:%S"),
                   text="", confidence=0.0, source=source, is_final=True)
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "language": self.language,
            "is_final": self.is_final,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "audio_duration": round(self.audio_duration, 3),
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    def __str__(self) -> str:
        return f"[{self.timestamp}] ({self.confidence:.0%}) {self.text}"
    
    def __bool__(self) -> bool:
        return bool(self.text.strip())


@dataclass
class TranscriptionStats:
    """Statistics about transcription performance"""
    total_transcriptions: int = 0
    total_audio_seconds: float = 0.0
    total_processing_ms: float = 0.0
    avg_confidence: float = 0.0
    word_count: int = 0
    empty_results: int = 0
    
    @property
    def realtime_factor(self) -> float:
        if self.total_audio_seconds > 0:
            return (self.total_processing_ms / 1000) / self.total_audio_seconds
        return 0.0
    
    def update(self, result: TranscriptionResult) -> None:
        self.total_transcriptions += 1
        self.total_audio_seconds += result.audio_duration
        self.total_processing_ms += result.processing_time_ms
        self.word_count += len(result.text.split())
        if not result.text.strip():
            self.empty_results += 1
        n = self.total_transcriptions
        self.avg_confidence = (self.avg_confidence * (n - 1) + result.confidence) / n
    
    def to_dict(self) -> dict:
        return {
            "total_transcriptions": self.total_transcriptions,
            "total_audio_seconds": round(self.total_audio_seconds, 2),
            "avg_confidence": round(self.avg_confidence, 3),
            "word_count": self.word_count,
            "realtime_factor": round(self.realtime_factor, 3),
        }
