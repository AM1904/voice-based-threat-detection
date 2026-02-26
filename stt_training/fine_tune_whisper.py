"""
WATZS Whisper Fine-Tuning Script
==================================
Fine-tunes OpenAI Whisper-small on your prepared dataset for
high-accuracy threat keyword recognition with Indian English accents.

Auto-detects CPU/GPU and adjusts training configuration accordingly.

Usage:
    python fine_tune_whisper.py                         # Full training
    python fine_tune_whisper.py --max-steps 100         # Quick test run
    python fine_tune_whisper.py --resume checkpoint-500 # Resume from checkpoint

Prerequisites:
    1. Run prepare_data.py, record_keywords.py, augment_data.py, build_manifest.py
    2. Dataset must exist at stt_training/dataset/
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_dependencies():
    """Verify all required libraries are installed."""
    missing = []
    for lib in ["transformers", "datasets", "torch", "evaluate", "jiwer"]:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)

    if missing:
        print("ERROR: Missing dependencies:")
        for lib in missing:
            print(f"  - {lib}")
        print("\nInstall with:")
        print("  pip install transformers datasets torch evaluate jiwer accelerate")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper-small for WATZS threat detection"
    )
    parser.add_argument(
        "--model", type=str, default="openai/whisper-small",
        help="Base model to fine-tune (default: openai/whisper-small)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=4000,
        help="Maximum training steps (default: 4000)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size (auto-detected if not set)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-5,
        help="Learning rate (default: 1e-5)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from checkpoint directory name (e.g. 'checkpoint-500')"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for fine-tuned model"
    )
    args = parser.parse_args()

    check_dependencies()

    import torch
    import evaluate
    import soundfile as sf
    import numpy as np
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union
    from datasets import Dataset, DatasetDict, Features, Value
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperFeatureExtractor,
        WhisperTokenizer,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )

    # ─── Configuration ─────────────────────────────────────────────
    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"

    print("\n🎯 WATZS — Whisper Fine-Tuning")
    print("=" * 60)
    print(f"  Base model:  {args.model}")
    print(f"  Device:      {device}")
    if use_gpu:
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory:  {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"  Max steps:   {args.max_steps}")

    # ─── Paths ──────────────────────────────────────────────────────
    dataset_path = Path(__file__).parent / "dataset"
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    model_output = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "models" / "watzs-whisper"
    )

    if not dataset_path.exists():
        print(f"\n  ❌ Dataset not found at {dataset_path}")
        print(f"  Run build_manifest.py first.")
        sys.exit(1)

    # ─── Load Dataset ───────────────────────────────────────────────
    # Load dataset and rebuild WITHOUT the Audio feature type to avoid
    # torchcodec/FFmpeg dependency. We load audio manually with soundfile.
    print(f"\n  Loading dataset from {dataset_path}...")
    raw_dataset = DatasetDict.load_from_disk(str(dataset_path))
    print(f"  Train:      {len(raw_dataset['train'])} samples")
    print(f"  Validation: {len(raw_dataset['validation'])} samples")

    # Rebuild dataset with plain string paths (no Audio feature)
    def rebuild_without_audio_feature(split_dataset):
        """Extract audio paths and sentences from the Arrow table directly.

        Bypasses the Audio feature decoder (avoids torchcodec/FFmpeg).
        The Audio feature stores data as a struct {path, bytes} in Arrow.
        """
        table = split_dataset.data
        audio_col = table.column("audio")

        paths = []
        for i in range(len(audio_col)):
            item = audio_col[i].as_py()
            if isinstance(item, dict):
                paths.append(item.get("path", ""))
            elif isinstance(item, str):
                paths.append(item)
            else:
                paths.append(str(item))

        sentences = [table.column("sentence")[i].as_py() for i in range(len(table))]

        return Dataset.from_dict({
            "audio_path": paths,
            "sentence": sentences,
        })

    # Extract paths directly from Arrow (no audio decoding triggered)
    print("  Rebuilding dataset (bypassing audio decoder)...")
    try:
        dataset = DatasetDict({
            "train": rebuild_without_audio_feature(raw_dataset["train"]),
            "validation": rebuild_without_audio_feature(raw_dataset["validation"]),
        })
    except Exception as e:
        print(f"  ⚠️  Arrow extraction failed: {e}")
        print("  Falling back to reading manifest CSVs directly...")
        import csv
        data_dir = Path(__file__).parent / "data"
        splits = {}
        for split_name, csv_dirs in [
            ("train", ["custom", "augmented"]),
            ("validation", ["val"]),
        ]:
            all_paths, all_sentences = [], []
            for subdir in csv_dirs:
                manifest = data_dir / subdir / "manifest.csv"
                if manifest.exists():
                    with open(manifest, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if Path(row["path"]).exists():
                                all_paths.append(row["path"])
                                all_sentences.append(row["sentence"])
            splits[split_name] = Dataset.from_dict({
                "audio_path": all_paths,
                "sentence": all_sentences,
            })
        dataset = DatasetDict(splits)

    print(f"  ✅ Dataset loaded (no torchcodec required)")

    # ─── Load Model & Processor ─────────────────────────────────────
    print(f"\n  Loading Whisper model '{args.model}'...")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model)
    tokenizer = WhisperTokenizer.from_pretrained(
        args.model, language="en", task="transcribe"
    )
    processor = WhisperProcessor.from_pretrained(
        args.model, language="en", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.model)

    # Disable cache for training (saves memory)
    model.config.use_cache = False

    # Set forced decoder IDs for English transcription
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    print(f"  ✅ Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

    # ─── Preprocess Dataset ─────────────────────────────────────────
    print(f"\n  Preprocessing dataset...")

    TARGET_SR = 16000

    # Build a lookup of all WAV files for resolving relative paths
    _data_dir = Path(__file__).parent / "data"
    _wav_lookup = {}
    for wav_file in _data_dir.rglob("*.wav"):
        _wav_lookup[wav_file.name] = str(wav_file)

    def prepare_dataset(batch):
        """Load audio with soundfile, convert to log-mel, tokenize transcript."""
        audio_path = batch["audio_path"]

        # Resolve path: if it doesn't exist as-is, look up by filename
        if not Path(audio_path).exists():
            basename = Path(audio_path).name
            if basename in _wav_lookup:
                audio_path = _wav_lookup[basename]
            else:
                raise FileNotFoundError(
                    f"Audio file not found: {audio_path} "
                    f"(also searched data/ for '{basename}')"
                )

        # Load audio directly with soundfile
        audio_array, sr = sf.read(audio_path, dtype="float32")

        # Convert to mono if stereo
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != TARGET_SR:
            try:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=TARGET_SR)
            except ImportError:
                # Simple resampling fallback
                ratio = TARGET_SR / sr
                new_len = int(len(audio_array) * ratio)
                audio_array = np.interp(
                    np.linspace(0, len(audio_array), new_len),
                    np.arange(len(audio_array)),
                    audio_array
                ).astype(np.float32)

        # Extract log-mel spectrogram features
        batch["input_features"] = feature_extractor(
            audio_array,
            sampling_rate=TARGET_SR,
        ).input_features[0]

        # Tokenize transcript
        batch["labels"] = tokenizer(batch["sentence"]).input_ids

        return batch

    # Process datasets
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names["train"],
        desc="Preprocessing",
    )

    print(f"  ✅ Preprocessing complete")

    # ─── Data Collator ──────────────────────────────────────────────
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        """
        Custom data collator that pads input features and labels.
        Replaces padding token ID with -100 so they're ignored by the loss.
        """
        processor: Any

        def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
        ) -> Dict[str, torch.Tensor]:
            # Split inputs and labels
            input_features = [
                {"input_features": f["input_features"]} for f in features
            ]
            label_features = [
                {"input_ids": f["labels"]} for f in features
            ]

            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )

            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt"
            )

            # Replace padding with -100 to ignore in loss
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            # Remove BOS token if it was appended
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # ─── Evaluation Metric ──────────────────────────────────────────
    from jiwer import wer as compute_wer

    def compute_metrics(pred):
        """Compute Word Error Rate on predictions."""
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token for decoding
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * compute_wer(label_str, pred_str)
        return {"wer": wer}

    # ─── Training Configuration ─────────────────────────────────────
    batch_size = args.batch_size
    if batch_size is None:
        if use_gpu:
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            batch_size = 8 if gpu_mem > 8 else 4 if gpu_mem > 4 else 2
        else:
            batch_size = 2

    grad_accum = max(1, 16 // batch_size)  # Effective batch size ~16

    print(f"\n  Training configuration:")
    print(f"    Batch size:         {batch_size}")
    print(f"    Gradient accum:     {grad_accum}")
    print(f"    Effective batch:    {batch_size * grad_accum}")
    print(f"    Learning rate:      {args.learning_rate}")
    print(f"    Max steps:          {args.max_steps}")
    print(f"    FP16 (mixed prec):  {use_gpu}")
    print(f"    Checkpoints:        {checkpoint_dir}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        max_steps=args.max_steps,
        fp16=use_gpu,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=3,
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["none"],  # Disable wandb/tensorboard by default
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_num_workers=2 if use_gpu else 0,
        remove_unused_columns=False,
    )

    # ─── Trainer ────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    # ─── Train ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  🚀 Starting training...")
    print(f"{'='*60}")

    resume_from = None
    if args.resume:
        # Explicit checkpoint specified
        resume_path = checkpoint_dir / args.resume
        if resume_path.exists():
            resume_from = str(resume_path)
        else:
            print(f"  ⚠️  Checkpoint not found: {resume_path}")
    else:
        # Auto-detect latest checkpoint
        checkpoints = sorted(
            [d for d in checkpoint_dir.glob("checkpoint-*") if d.is_dir()],
            key=lambda d: int(d.name.split("-")[-1]),
        )
        if checkpoints:
            resume_from = str(checkpoints[-1])

    if resume_from:
        print(f"  ↩️  Resuming from: {resume_from}")
    else:
        print(f"  Starting from scratch.")

    if use_gpu:
        estimated_time = f"~{args.max_steps * 2 / 3600:.1f} hours"
    else:
        estimated_time = f"~{args.max_steps * 8 / 3600:.1f} hours"
    print(f"  Estimated time: {estimated_time}")
    print()

    trainer.train(resume_from_checkpoint=resume_from)

    # ─── Save Model ─────────────────────────────────────────────────
    print(f"\n  Saving fine-tuned model to {model_output}...")
    model_output.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(model_output))
    processor.save_pretrained(str(model_output))
    tokenizer.save_pretrained(str(model_output))

    print(f"  ✅ Model saved to: {model_output}")

    # ─── Final Evaluation ───────────────────────────────────────────
    print(f"\n  Running final evaluation...")
    metrics = trainer.evaluate()
    print(f"\n  Final metrics:")
    print(f"    WER: {metrics.get('eval_wer', 'N/A'):.2f}%")

    print(f"\n{'='*60}")
    print(f"✅ Fine-tuning complete!")
    print(f"{'='*60}")
    print(f"  Model saved to: {model_output}")
    print(f"  Final WER: {metrics.get('eval_wer', 'N/A'):.2f}%")
    print(f"\nNext steps:")
    print(f"  1. Evaluate:    python evaluate.py")
    print(f"  2. Integrate:   Model will be auto-loaded by keyword_detector.py")


if __name__ == "__main__":
    main()
