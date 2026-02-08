"""
Speech Processor Module

Handles natural speech patterns including pause detection,
partial phrase handling, and duplicate prevention.
"""

import logging
import time
import threading
from typing import Optional, Callable, List
from collections import deque
import difflib

from .stt_config import STTConfig
from .transcription import TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)


class SpeechProcessor:
    """Processes speech patterns for natural conversation handling."""
    
    def __init__(self, config: Optional[STTConfig] = None):
        self.config = config or STTConfig()
        self._segment_buffer: deque = deque(maxlen=50)
        self._text_history: deque = deque(maxlen=20)
        self._last_speech_time = 0.0
        self._is_speaking = False
        self._min_utterance_gap = self.config.min_silence_duration
        self._similarity_threshold = 0.85
        self._lock = threading.Lock()
        self._on_partial: Optional[Callable[[str], None]] = None
    
    def process(self, result: TranscriptionResult) -> Optional[TranscriptionResult]:
        with self._lock:
            if not result or not result.text.strip():
                return self._check_utterance_end()
            
            cleaned = self._clean_text(result.text)
            if self._is_duplicate(cleaned):
                return None
            
            self._last_speech_time = time.time()
            self._is_speaking = True
            self._segment_buffer.append(result)
            self._text_history.append(cleaned)
            
            if self._on_partial:
                self._on_partial(cleaned)
            
            if result.is_final and len(cleaned) >= 2:
                return self._create_processed_result(result, cleaned)
            return None
    
    def _check_utterance_end(self) -> Optional[TranscriptionResult]:
        if not self._is_speaking:
            return None
        if time.time() - self._last_speech_time >= self._min_utterance_gap:
            return self._finalize_utterance()
        return None
    
    def _finalize_utterance(self) -> Optional[TranscriptionResult]:
        if not self._segment_buffer:
            self._is_speaking = False
            return None
        merged = self._merge_segments()
        self._is_speaking = False
        self._segment_buffer.clear()
        return merged
    
    def _merge_segments(self) -> Optional[TranscriptionResult]:
        if not self._segment_buffer:
            return None
        segments = list(self._segment_buffer)
        texts = [self._clean_text(s.text) for s in segments if s.text.strip()]
        if not texts:
            return None
        confidences = [s.confidence for s in segments]
        return TranscriptionResult.create(
            text=" ".join(texts),
            confidence=sum(confidences)/len(confidences),
            source=segments[0].source,
            processing_time_ms=sum(s.processing_time_ms for s in segments),
            audio_duration=sum(s.audio_duration for s in segments),
        )
    
    def _create_processed_result(self, original: TranscriptionResult, cleaned: str) -> TranscriptionResult:
        return TranscriptionResult.create(
            text=cleaned,
            confidence=original.confidence,
            source=original.source,
            segments=original.segments,
            processing_time_ms=original.processing_time_ms,
            audio_duration=original.audio_duration,
            language=original.language,
        )
    
    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        cleaned = " ".join(text.strip().split())
        for artifact in ["[BLANK_AUDIO]", "[MUSIC]", "[NOISE]"]:
            cleaned = cleaned.replace(artifact, "")
        return cleaned.strip()
    
    def _is_duplicate(self, text: str) -> bool:
        if not text or not self._text_history:
            return False
        for prev in list(self._text_history)[-5:]:
            if difflib.SequenceMatcher(None, text.lower(), prev.lower()).ratio() > self._similarity_threshold:
                return True
        return False
    
    def on_partial(self, callback: Callable[[str], None]) -> None:
        self._on_partial = callback
    
    def force_finalize(self) -> Optional[TranscriptionResult]:
        with self._lock:
            return self._finalize_utterance() if self._segment_buffer else None
    
    @property
    def is_speaking(self) -> bool:
        return self._is_speaking
