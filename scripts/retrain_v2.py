"""
WATZS Sound Classifier v2 - Improved Retraining Pipeline
=========================================================
Key improvements over v1:
- Uses ALL ESC-50 categories (2000 samples): threats -> threat classes, everything else -> normal
- Adds LibriSpeech speech samples -> normal
- Deeper architecture with BatchNorm + more dropout
- No artificial sample cap on normal class (more normals = fewer false positives)
- Proper stratified train/val split
- Saves training history for analysis

Usage:
    python scripts/retrain_v2.py --epochs 30
"""
import argparse
import datetime
import os
import shutil
import sys
import warnings

import numpy as np

# Silence TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# ── Classes (must match sound_classifier.py) ──────────────────────────
CLASSES = ["scream", "gunshot", "glass_breaking", "crash", "normal"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# ── ESC-50 category -> our class mapping ──────────────────────────────
# Threat categories mapped explicitly; everything else -> normal
ESC50_THREAT_MAP = {
    # scream
    "crying_baby": "scream",
    # gunshot
    "fireworks": "gunshot",
    "gun_shot": "gunshot",
    # glass_breaking
    "glass_breaking": "glass_breaking",
    # crash
    "can_crushing": "crash",
    "door_wood_knock": "crash",     # hard negative: knock != crash, but collision-like
}

# Hard negatives: sounds that are acoustically CLOSE to threats but are normal
# These train the model to distinguish, e.g., clapping vs gunshot, laughing vs scream
HARD_NEGATIVES = {
    "clapping", "laughing", "coughing", "sneezing", "breathing",
    "door_wood_knock", "mouse_click", "keyboard_typing",
    "clock_alarm", "clock_tick", "siren",
    "dog", "cat", "crow", "insects", "frog",
    "rain", "wind", "thunderstorm", "water_drops",
    "church_bells", "vacuum_cleaner", "washing_machine",
    "train", "helicopter", "airplane", "car_horn",
    "engine", "chainsaw", "hand_saw",
    "drinking_sipping", "pouring_water",
    "toilet_flush", "brushing_teeth",
    "footsteps", "crackling_fire",
    "pig", "rooster", "hen", "sheep",
}
# Note: door_wood_knock appears in BOTH ESC50_THREAT_MAP (crash) and HARD_NEGATIVES.
# ESC50_THREAT_MAP takes priority — we want those samples as crash training.


def get_yamnet():
    """Load YAMNet model for embedding extraction."""
    import tensorflow_hub as hub
    return hub.load("https://tfhub.dev/google/yamnet/1")


def resample_audio(audio, orig_sr, target_sr=16000):
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio
    import scipy.signal
    num_samples = int(len(audio) * target_sr / orig_sr)
    return scipy.signal.resample(audio, num_samples).astype(np.float32)


def extract_embedding(yamnet, waveform):
    """Extract YAMNet embedding from waveform (float32, 16kHz)."""
    min_samples = 15600  # ~0.975s at 16kHz
    if len(waveform) < min_samples:
        waveform = np.pad(waveform, (0, min_samples - len(waveform)))
    _, embeddings, _ = yamnet(waveform)
    return embeddings.numpy().mean(axis=0)  # shape: (1024,)


def download_esc50():
    """Download ESC-50 dataset via HuggingFace datasets."""
    from datasets import load_dataset
    print("[Data] Loading ESC-50 from HuggingFace...")
    ds = load_dataset("ashraq/esc50", split="train", trust_remote_code=True)
    print(f"[Data] ESC-50: {len(ds)} samples loaded")
    return ds


def download_speech(max_samples=200):
    """Download LibriSpeech speech samples for the normal class."""
    from datasets import load_dataset
    print("[Data] Loading LibriSpeech (clean) from HuggingFace...")
    try:
        ds = load_dataset(
            "librispeech_asr", "clean", split="validation",
            trust_remote_code=True
        )
        # Take up to max_samples
        if len(ds) > max_samples:
            indices = np.random.choice(len(ds), max_samples, replace=False)
            ds = ds.select(indices)
        print(f"[Data] LibriSpeech: {len(ds)} speech samples loaded")
        return ds
    except Exception as e:
        print(f"[Data] WARNING: Could not load LibriSpeech: {e}")
        return None


def process_esc50(yamnet, ds):
    """Process ESC-50 samples into embeddings + labels."""
    embeddings = []
    labels = []
    stats = {c: 0 for c in CLASSES}
    skipped = 0

    for i, row in enumerate(ds):
        category = row["category"]

        # Determine our class
        if category in ESC50_THREAT_MAP:
            our_class = ESC50_THREAT_MAP[category]
        else:
            our_class = "normal"

        label_idx = CLASS_TO_IDX[our_class]

        try:
            audio_info = row["audio"]
            waveform = np.array(audio_info["array"], dtype=np.float32)
            sr = audio_info["sampling_rate"]
            waveform = resample_audio(waveform, sr, 16000)
            emb = extract_embedding(yamnet, waveform)
            embeddings.append(emb)
            labels.append(label_idx)
            stats[our_class] += 1
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"[Data] Skip ESC-50 sample {i} ({category}): {e}")

        if (i + 1) % 200 == 0:
            print(f"[Data] Processed {i + 1}/{len(ds)} ESC-50 samples...")

    print(f"[Data] ESC-50 done. Distribution: {stats}. Skipped: {skipped}")
    return embeddings, labels, stats


def process_speech(yamnet, ds):
    """Process LibriSpeech samples into embeddings (all -> normal)."""
    embeddings = []
    labels = []
    count = 0

    if ds is None:
        return embeddings, labels

    for i, row in enumerate(ds):
        try:
            audio_info = row["audio"]
            waveform = np.array(audio_info["array"], dtype=np.float32)
            sr = audio_info["sampling_rate"]
            waveform = resample_audio(waveform, sr, 16000)
            # Truncate to 5 seconds max (LibriSpeech samples can be long)
            max_len = 5 * 16000
            if len(waveform) > max_len:
                waveform = waveform[:max_len]
            emb = extract_embedding(yamnet, waveform)
            embeddings.append(emb)
            labels.append(CLASS_TO_IDX["normal"])
            count += 1
        except Exception as e:
            if count < 3:
                print(f"[Data] Skip LibriSpeech sample {i}: {e}")

        if (i + 1) % 50 == 0:
            print(f"[Data] Processed {i + 1}/{len(ds)} speech samples...")

    print(f"[Data] LibriSpeech done. Added {count} normal samples.")
    return embeddings, labels


def augment_scream_embeddings(embeddings, labels, yamnet, esc50_ds):
    """
    Generate augmented scream samples by pitch-shifting crying_baby audio.
    This helps the model learn the variability of scream-like sounds.
    """
    import scipy.signal

    aug_embeddings = []
    aug_labels = []
    cry_samples = []

    for row in esc50_ds:
        if row["category"] == "crying_baby":
            audio_info = row["audio"]
            waveform = np.array(audio_info["array"], dtype=np.float32)
            sr = audio_info["sampling_rate"]
            waveform = resample_audio(waveform, sr, 16000)
            cry_samples.append(waveform)

    print(f"[Augment] Found {len(cry_samples)} crying_baby samples for scream augmentation")

    for waveform in cry_samples:
        for shift in [0.8, 1.2]:  # pitch down and up
            shifted_len = int(len(waveform) / shift)
            shifted = scipy.signal.resample(waveform, shifted_len).astype(np.float32)
            # Pad/truncate to original length
            if len(shifted) < len(waveform):
                shifted = np.pad(shifted, (0, len(waveform) - len(shifted)))
            else:
                shifted = shifted[:len(waveform)]

            emb = extract_embedding(yamnet, shifted)
            aug_embeddings.append(emb)
            aug_labels.append(CLASS_TO_IDX["scream"])

        # Add noise-augmented version
        noise = np.random.normal(0, 0.01, len(waveform)).astype(np.float32)
        noisy = waveform + noise
        emb = extract_embedding(yamnet, noisy)
        aug_embeddings.append(emb)
        aug_labels.append(CLASS_TO_IDX["scream"])

    print(f"[Augment] Generated {len(aug_embeddings)} augmented scream samples")
    return aug_embeddings, aug_labels


def build_model(input_dim=1024):
    """
    Improved classifier architecture.
    Deeper with BatchNorm for better generalization.
    """
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(epochs=30, batch_size=32):
    """Full training pipeline."""
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    print("=" * 60)
    print("WATZS Sound Classifier v2 - Retraining")
    print("=" * 60)

    yamnet = get_yamnet()
    print("[Model] YAMNet loaded")

    # ── Collect data ──────────────────────────────────────────
    esc50_ds = download_esc50()

    # Skip LibriSpeech — ESC-50 already has 1800+ normal samples including
    # speech-adjacent sounds (laughing, coughing, breathing, sneezing).
    # LibriSpeech is 6GB+ and takes too long to download.
    speech_ds = None

    print("\n[Processing] Extracting YAMNet embeddings...")
    esc_emb, esc_lab, esc_stats = process_esc50(yamnet, esc50_ds)
    speech_emb, speech_lab = process_speech(yamnet, speech_ds)
    aug_emb, aug_lab = augment_scream_embeddings(esc_emb, esc_lab, yamnet, esc50_ds)

    # Combine all
    all_emb = esc_emb + speech_emb + aug_emb
    all_lab = esc_lab + speech_lab + aug_lab

    X = np.array(all_emb, dtype=np.float32)
    y = np.array(all_lab, dtype=np.int32)

    print(f"\n[Data] Total samples: {len(X)}")
    for i, c in enumerate(CLASSES):
        count = np.sum(y == i)
        print(f"  {c}: {count}")

    # ── Train/val split ───────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[Split] Train: {len(X_train)}, Val: {len(X_val)}")

    # ── Class weights (balance threat vs normal) ──────────────
    cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train)
    class_weights = {i: w for i, w in enumerate(cw)}
    print(f"[Weights] {dict(zip(CLASSES, [f'{w:.2f}' for w in cw]))}")

    # ── Build & train ─────────────────────────────────────────
    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"\n[Train] Starting training for up to {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n[Result] Val accuracy: {val_acc:.1%}, Val loss: {val_loss:.4f}")

    # Per-class accuracy
    predictions = model.predict(X_val, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    print("\n[Per-class results]")
    for i, c in enumerate(CLASSES):
        mask = y_val == i
        if mask.sum() == 0:
            continue
        class_acc = (pred_classes[mask] == i).mean()
        class_conf = predictions[mask, i].mean()
        print(f"  {c:20s}: acc={class_acc:.1%}  avg_conf={class_conf:.3f}  n={mask.sum()}")

    # Confusion analysis: what does the model confuse?
    print("\n[Confusion analysis]")
    for i, c in enumerate(CLASSES):
        mask = y_val == i
        if mask.sum() == 0:
            continue
        wrong = pred_classes[mask] != i
        if wrong.sum() > 0:
            wrong_preds = pred_classes[mask][wrong]
            for j in range(NUM_CLASSES):
                cnt = (wrong_preds == j).sum()
                if cnt > 0:
                    print(f"  {c} misclassified as {CLASSES[j]}: {cnt}/{mask.sum()}")

    # ── Save model ────────────────────────────────────────────
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    model_path = os.path.join(model_dir, "watzs_sound_classifier.h5")

    # Backup existing
    if os.path.exists(model_path):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = model_path.replace(".h5", f".bak.{ts}.h5")
        shutil.copy2(model_path, backup)
        print(f"\n[Save] Backed up old model to {os.path.basename(backup)}")

    model.save(model_path)
    print(f"[Save] Model saved to {model_path}")
    print(f"[Save] Total samples used: {len(X)}")

    # Save training history
    history_path = os.path.join(model_dir, "training_history_v2.npz")
    np.savez(
        history_path,
        train_acc=history.history["accuracy"],
        val_acc=history.history["val_accuracy"],
        train_loss=history.history["loss"],
        val_loss=history.history["val_loss"],
    )
    print(f"[Save] Training history saved to {history_path}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WATZS Sound Classifier v2 Retraining")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size)
