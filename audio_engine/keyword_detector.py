"""
Keyword Detector Module
=======================
Real-time keyword detection using OpenAI Whisper (via HuggingFace Transformers)
for high-accuracy speech-to-text transcription.

Supports both stock Whisper models and WATZS fine-tuned models.
Auto-detects GPU for accelerated inference.

Loads threat keywords from keywords.json and scans live transcription
for matches at all alert levels (L1/L2/L3).

Usage:
    from audio_engine.keyword_detector import KeywordDetector

    def on_alert(alert):
        print(f"ALERT L{alert['level']}: {alert['keyword']}")

    detector = KeywordDetector()
    detector.on_alert = on_alert
    detector.process_audio(audio_bytes, frame_count)
"""

import json
import os
import uuid
import threading
import numpy as np
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path


# ─── Default Configuration ──────────────────────────────────────────────
DEFAULT_KEYWORDS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "keywords.json"
)
DEFAULT_FINETUNED_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "watzs-whisper"
)
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes
RECOGNITION_CHUNK_SECONDS = 2
SILENCE_THRESHOLD = 300


def _load_whisper_pipeline(model_path=None, model_size="small", language="en"):
    """
    Load a Whisper pipeline using HuggingFace Transformers.

    Priority:
      1. Fine-tuned model at model_path (if exists)
      2. Stock Whisper model of given size

    Returns:
        transformers.Pipeline for automatic-speech-recognition
    """
    try:
        from transformers import pipeline
        import torch
    except ImportError:
        raise ImportError(
            "HuggingFace Transformers is required. "
            "Install with: pip install transformers torch"
        )

    # Determine which model to load
    finetuned_path = Path(model_path) if model_path else Path(DEFAULT_FINETUNED_MODEL_PATH)

    if finetuned_path.exists() and (finetuned_path / "config.json").exists():
        model_id = str(finetuned_path)
        source_label = f"fine-tuned ({finetuned_path.name})"
    else:
        model_id = f"openai/whisper-{model_size}"
        source_label = f"stock (whisper-{model_size})"

    # Auto-detect device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"[KeywordDetector] Loading Whisper — {source_label}")
    print(f"[KeywordDetector] Device: {device}")

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        torch_dtype=torch_dtype,
        chunk_length_s=30,
        return_timestamps=False,
    )

    print(f"[KeywordDetector] Whisper loaded successfully.")
    return pipe, source_label


class KeywordDetector:
    """
    Detects threat keywords in live audio using OpenAI Whisper.

    Supports both stock Whisper models and WATZS fine-tuned models.
    Auto-loads fine-tuned model from models/watzs-whisper/ if present.

    Attributes:
        on_alert (callable): Callback for alert events
        on_transcription (callable): Callback for raw transcriptions
    """

    def __init__(self, keywords_path=None, sample_rate=SAMPLE_RATE,
                 model_path=None, model_size="small", language="en"):
        """
        Initialize the keyword detector.

        Args:
            keywords_path: Path to keywords.json config file
            sample_rate: Audio sample rate (default: 16000)
            model_path: Path to fine-tuned model directory (auto-detected if None)
            model_size: Stock Whisper size fallback: 'tiny', 'small', 'medium'
            language: Language code (default: 'en')
        """
        self.sample_rate = sample_rate
        self.keywords_path = keywords_path or DEFAULT_KEYWORDS_PATH
        self.model_size = model_size
        self.language = language

        # Callbacks
        self.on_alert = None
        self.on_transcription = None

        # Load Whisper model via HuggingFace Transformers
        self._stt_pipeline, self._model_source = _load_whisper_pipeline(
            model_path=model_path,
            model_size=model_size,
            language=language,
        )

        # Load keywords configuration
        self._load_keywords()

        # Audio buffer
        self._audio_buffer = bytearray()
        self._buffer_frames = 0
        self._frames_per_chunk = int(
            self.sample_rate * RECOGNITION_CHUNK_SECONDS
        )

        # Keyword repetition tracking (for escalation)
        self._keyword_counts = defaultdict(int)

        # State
        self._is_active = True
        self._processing_lock = threading.Lock()

    # ─── Keyword Loading ────────────────────────────────────────────
    def _load_keywords(self):
        """Load and index keywords from keywords.json."""
        if not os.path.exists(self.keywords_path):
            raise FileNotFoundError(
                f"Keywords config not found: {self.keywords_path}"
            )

        with open(self.keywords_path, "r", encoding="utf-8") as f:
            self.keywords_config = json.load(f)

        self._keyword_lookup = {}
        levels = self.keywords_config.get("levels", {})
        for level_key, level_data in levels.items():
            level_num = int(level_key.replace("L", ""))
            for keyword in level_data.get("keywords", []):
                self._keyword_lookup[keyword.lower()] = level_num

        rules = self.keywords_config.get("escalation_rules", {})
        self._keyword_rep_threshold = rules.get(
            "keyword_repetition_to_L3", {}
        ).get("threshold", 2)

        self._sorted_keywords = sorted(
            self._keyword_lookup.keys(), key=len, reverse=True
        )

        print(f"[KeywordDetector] Loaded {len(self._keyword_lookup)} keywords "
              f"across {len(levels)} levels.")

    # ─── Audio Processing ───────────────────────────────────────────
    def process_audio(self, data, frame_count=None):
        """
        Process an audio chunk — buffers audio and runs Whisper
        when enough data is accumulated.

        Args:
            data (bytes): Raw audio data (16-bit PCM, mono, 16kHz)
            frame_count (int): Number of frames in this chunk
        """
        if not self._is_active:
            return

        self._audio_buffer.extend(data)
        actual_frames = len(data) // SAMPLE_WIDTH
        self._buffer_frames += actual_frames

        if self._buffer_frames >= self._frames_per_chunk:
            samples = np.frombuffer(
                bytes(self._audio_buffer), dtype=np.int16
            ).astype(np.float64)
            rms = float(np.sqrt(np.mean(samples ** 2)))

            if rms > SILENCE_THRESHOLD:
                audio_data = bytes(self._audio_buffer)
                thread = threading.Thread(
                    target=self._recognize_audio,
                    args=(audio_data,),
                    daemon=True
                )
                thread.start()

            self._audio_buffer = bytearray()
            self._buffer_frames = 0

    def _recognize_audio(self, audio_bytes):
        """Transcribe audio using Whisper via HuggingFace Transformers pipeline."""
        with self._processing_lock:
            try:
                # Convert bytes to float32 audio array
                samples = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_float = samples.astype(np.float32) / 32768.0

                # Run Whisper inference
                result = self._stt_pipeline(
                    audio_float,
                    generate_kwargs={
                        "language": self.language,
                    },
                )

                text = result.get("text", "").strip()
                if text:
                    self._handle_transcription(text, is_final=True)

            except Exception as e:
                print(f"[KeywordDetector] Whisper error: {e}")

    def _handle_transcription(self, text, is_final):
        """Process transcribed text — check for keyword matches."""
        if self.on_transcription:
            self.on_transcription(text, is_final)

        if not is_final:
            return

        text_lower = text.lower()
        matched_keywords = self._find_keywords(text_lower)

        for keyword, level in matched_keywords:
            self._keyword_counts[keyword] += 1
            count = self._keyword_counts[keyword]

            if count > self._keyword_rep_threshold and level < 3:
                self._emit_alert(
                    alert_type="keyword",
                    keyword=keyword,
                    level=3,
                    confidence=0.95,
                    source="whisper",
                    metadata={
                        "repeat_count": count,
                        "escalated_from": level,
                        "original_text": text,
                        "model_source": self._model_source,
                    }
                )
            else:
                self._emit_alert(
                    alert_type="keyword",
                    keyword=keyword,
                    level=level,
                    confidence=0.92,
                    source="whisper",
                    metadata={
                        "repeat_count": count,
                        "original_text": text,
                        "model_source": self._model_source,
                    }
                )

    def _find_keywords(self, text):
        """Find all keyword matches in text. Returns list of (keyword, level)."""
        matches = []
        remaining = text
        for keyword in self._sorted_keywords:
            if keyword in remaining:
                level = self._keyword_lookup[keyword]
                matches.append((keyword, level))
                remaining = remaining.replace(keyword, "", 1)
        return matches

    # ─── Alert Emission ─────────────────────────────────────────────
    def _emit_alert(self, alert_type, keyword, level, confidence,
                    source, metadata=None):
        alert = {
            "id": str(uuid.uuid4()),
            "type": alert_type,
            "keyword": keyword,
            "level": level,
            "timestamp": datetime.now(timezone(timedelta(hours=5, minutes=30)))
                         .isoformat(),
            "confidence": confidence,
            "source": source,
            "metadata": metadata or {}
        }

        level_names = {1: "LOW", 2: "MEDIUM", 3: "HIGH"}
        level_icons = {1: "🟡", 2: "🟠", 3: "🔴"}

        print(f"\n{level_icons.get(level, '⚪')} "
              f"[ALERT L{level} — {level_names.get(level, 'UNKNOWN')}] "
              f"Type: {alert_type} | "
              f"Keyword: \"{keyword}\" | "
              f"Confidence: {confidence:.0%}")

        if self.on_alert:
            self.on_alert(alert)

    # ─── Control ────────────────────────────────────────────────────
    def pause(self):
        self._is_active = False
        print("[KeywordDetector] Paused.")

    def resume(self):
        self._is_active = True
        print("[KeywordDetector] Resumed.")

    def reset_counts(self):
        self._keyword_counts.clear()
        print("[KeywordDetector] Counters reset.")

    @property
    def keyword_list(self):
        return dict(self._keyword_lookup)

    @property
    def is_active(self):
        return self._is_active


# ─── CLI Entry Point ───────────────────────────────────────────────────
if __name__ == "__main__":
    import signal
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from audio_engine.capture import AudioCapture

    print("\n🔍 WATZS — Live Keyword Detection (Whisper)")
    print("=" * 50)

    detector = KeywordDetector()

    print("\n📋 Loaded keywords:")
    for level_key in ["L1", "L2", "L3"]:
        level_data = detector.keywords_config["levels"].get(level_key, {})
        keywords = level_data.get("keywords", [])
        print(f"  {level_key}: {', '.join(keywords)}")
    print()

    def show_transcription(text, is_final):
        prefix = "📝" if is_final else "💬"
        print(f"{prefix} \"{text}\"")

    detector.on_transcription = show_transcription

    capture = AudioCapture()
    capture.add_listener(detector.process_audio)

    def signal_handler(sig, frame):
        print("\n\n⏹️  Stopping...")
        capture.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("🎤 Listening... Speak threat keywords to test detection.")
    print("   Press Ctrl+C to stop.\n")
    capture.start(blocking=True)
