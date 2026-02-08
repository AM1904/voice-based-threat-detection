"""
Whisper Engine Module

Wrapper around faster-whisper for real-time speech recognition.
"""

import logging
import time
from typing import Optional, Generator
import numpy as np

try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False
    WhisperModel = None

from .stt_config import STTConfig, WhisperModelSize, ComputeType
from .transcription import TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)


class WhisperEngine:
    """Whisper-based speech recognition engine."""
    
    def __init__(self, config: Optional[STTConfig] = None):
        if not HAS_FASTER_WHISPER:
            raise ImportError("faster-whisper required. Install: pip install faster-whisper")
        
        self.config = config or STTConfig()
        self._model = None
        self._model_loaded = False
        self._device = self._detect_device()
        self._compute_type = self._get_compute_type()
        logger.info(f"WhisperEngine: device={self._device}, compute={self._compute_type}")
    
    def _detect_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    
    def _get_compute_type(self) -> str:
        if self.config.compute_type == ComputeType.AUTO:
            return "float16" if self._device == "cuda" else "int8"
        return self.config.compute_type.value
    
    def load_model(self) -> bool:
        if self._model_loaded:
            return True
        try:
            model_name = self.config.model_size.value
            logger.info(f"Loading Whisper model: {model_name}")
            start = time.time()
            self._model = WhisperModel(
                model_name,
                device=self._device,
                compute_type=self._compute_type,
                num_workers=self.config.num_workers,
            )
            self._model_loaded = True
            logger.info(f"Model loaded in {time.time()-start:.2f}s")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def transcribe(self, audio: np.ndarray, source: str = "mic_01") -> TranscriptionResult:
        if not self._model_loaded and not self.load_model():
            return TranscriptionResult.empty(source)
        
        start = time.time()
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
        
        audio_duration = len(audio) / self.config.sample_rate
        
        try:
            segments_gen, info = self._model.transcribe(
                audio,
                language=self.config.language,
                beam_size=self.config.beam_size,
                temperature=self.config.temperature,
                vad_filter=True,
            )
            
            segments = []
            full_text_parts = []
            total_confidence = 0.0
            
            for segment in segments_gen:
                confidence = min(1.0, max(0.0, (segment.avg_logprob + 2.0) / 2.0))
                seg = TranscriptionSegment(
                    text=segment.text.strip(),
                    start_time=segment.start,
                    end_time=segment.end,
                    confidence=confidence,
                    language=info.language,
                )
                segments.append(seg)
                full_text_parts.append(segment.text.strip())
                total_confidence += confidence
            
            processing_time = (time.time() - start) * 1000
            avg_confidence = total_confidence / len(segments) if segments else 0.0
            
            return TranscriptionResult.create(
                text=" ".join(full_text_parts),
                confidence=avg_confidence,
                source=source,
                segments=segments,
                processing_time_ms=processing_time,
                audio_duration=audio_duration,
                language=info.language,
            )
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return TranscriptionResult.empty(source)
    
    @property
    def is_loaded(self) -> bool:
        return self._model_loaded
    
    @property
    def model_info(self) -> dict:
        return {
            "model_size": self.config.model_size.value,
            "device": self._device,
            "compute_type": self._compute_type,
            "loaded": self._model_loaded,
        }
