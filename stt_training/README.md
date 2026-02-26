# WATZS — Custom STT Fine-Tuning Pipeline

Fine-tune OpenAI's Whisper-small on Indian English speech + threat keywords
for high-accuracy voice-based threat detection.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r ../requirements.txt
```

### 2. Prepare Data
```bash
# Download Common Voice Indian English subset
python prepare_data.py

# Record your own keyword utterances (50-100 per keyword recommended)
python record_keywords.py

# Augment data (noise, pitch shift, speed, reverb)
python augment_data.py

# Build the final HuggingFace dataset
python build_manifest.py
```

### 3. Fine-Tune Whisper
```bash
python fine_tune_whisper.py
# ~2-4 hrs (GPU) or ~8-12 hrs (CPU-only)
```

### 4. Evaluate
```bash
python evaluate.py
# Compares stock Whisper-small vs your fine-tuned model
```

### 5. Use in WATZS
The fine-tuned model is saved to `../models/watzs-whisper/`.
The `keyword_detector.py` will auto-load it when present.

## Directory Structure
```
stt_training/
├── data/
│   ├── raw/           # Downloaded Common Voice clips
│   ├── custom/        # Your recorded keyword utterances
│   ├── augmented/     # Augmented audio files
│   ├── train/         # Final training split
│   └── val/           # Final validation split
├── prepare_data.py
├── record_keywords.py
├── augment_data.py
├── build_manifest.py
├── fine_tune_whisper.py
└── evaluate.py
```
