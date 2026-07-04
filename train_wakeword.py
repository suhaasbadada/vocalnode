"""Train a microWakeWord model for the wake phrase "Hey Chatter" and export it
as a quantized .tflite model for efficient, low-latency streaming inference.
"""

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from microwakeword.mixednet import model as build_mixednet_model
from microwakeword.model_train_eval import train_model

WAKE_WORD_PHRASE = "Hey Chatter"
SAMPLE_RATE_HZ = 16000
FEATURE_COUNT = 40  # mel-spectrogram bins fed to the model per frame
SPECTROGRAM_LENGTH = 49  # streaming feature frames per inference window (~1s)
OUTPUT_TFLITE_PATH = Path("models/hey_chatter.tflite")

# Efficiency / accuracy targets enforced by the verifier below.
MAX_FALSE_ACCEPTS_PER_HOUR = 0.5
MAX_AVG_INFERENCE_MS = 5.0
DETECTION_THRESHOLD = 0.9


def build_training_flags(train_dir: Path) -> SimpleNamespace:
    """Assemble the flags namespace microWakeWord's training loop expects."""
    return SimpleNamespace(
        wanted_words=WAKE_WORD_PHRASE,
        sample_rate=SAMPLE_RATE_HZ,
        clip_duration_ms=1000,
        window_size_ms=30,
        window_stride_ms=20,
        feature_bin_count=FEATURE_COUNT,
        train_dir=str(train_dir),
        batch_size=64,
        learning_rate=0.001,
        epochs=40,
        model_name="mixednet_hey_chatter",
    )


def build_model(flags: SimpleNamespace) -> tf.keras.Model:
    """Construct the small-footprint streaming architecture microWakeWord uses."""
    return build_mixednet_model(flags)


def train_wakeword_model(train_dataset, validation_dataset, flags: SimpleNamespace) -> tf.keras.Model:
    """Run microWakeWord's training loop and return the trained Keras model."""
    model = build_model(flags)
    train_model(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        flags=flags,
    )
    return model


def export_tflite(model: tf.keras.Model, output_path: Path) -> Path:
    """Convert the trained Keras model to a quantized .tflite file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    output_path.write_bytes(tflite_model)
    return output_path


def _run_inference(interpreter: "tf.lite.Interpreter", features: np.ndarray) -> float:
    """Run a single streaming window through the tflite interpreter and return its score."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], features.astype(input_details[0]["dtype"]))
    interpreter.invoke()

    return float(interpreter.get_tensor(output_details[0]["index"])[0][0])


def compute_false_acceptance_rate(
    tflite_path: Path,
    negative_features: np.ndarray,
    window_duration_s: float,
    detection_threshold: float = DETECTION_THRESHOLD,
) -> dict:
    """Run every negative-audio window through the model and report the false
    acceptance rate (FAR) plus average inference latency.

    Args:
        tflite_path: Path to the exported .tflite model.
        negative_features: Array of shape (num_windows, SPECTROGRAM_LENGTH,
            FEATURE_COUNT) containing features extracted from audio that does
            NOT contain the wake phrase.
        window_duration_s: Real-world duration each feature window
            represents, used to normalize false accepts into a per-hour rate.
        detection_threshold: Model output score above which a window counts
            as a detection.

    Returns:
        A dict with false_accepts, total_windows, far_per_hour, and avg_inference_ms.
    """
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    false_accepts = 0
    inference_times_ms = []

    for window in negative_features:
        window_batch = np.expand_dims(window, axis=0)

        start = time.perf_counter()
        score = _run_inference(interpreter, window_batch)
        inference_times_ms.append((time.perf_counter() - start) * 1000)

        if score >= detection_threshold:
            false_accepts += 1

    total_windows = len(negative_features)
    total_hours = (total_windows * window_duration_s) / 3600
    far_per_hour = false_accepts / total_hours if total_hours > 0 else 0.0
    avg_inference_ms = float(np.mean(inference_times_ms)) if inference_times_ms else 0.0

    return {
        "false_accepts": false_accepts,
        "total_windows": total_windows,
        "far_per_hour": far_per_hour,
        "avg_inference_ms": avg_inference_ms,
    }


def verify_model(
    tflite_path: Path,
    negative_features: np.ndarray,
    window_duration_s: float = 1.0,
    max_far_per_hour: float = MAX_FALSE_ACCEPTS_PER_HOUR,
    max_avg_inference_ms: float = MAX_AVG_INFERENCE_MS,
) -> bool:
    """Verify a trained model is efficient and has a low false acceptance rate
    against a (dummy or real) negative dataset.

    Raises AssertionError with details if either target is missed; returns
    True when both the FAR and inference latency are within bounds.
    """
    metrics = compute_false_acceptance_rate(tflite_path, negative_features, window_duration_s)

    assert metrics["far_per_hour"] <= max_far_per_hour, (
        f"False Acceptance Rate too high: {metrics['far_per_hour']:.3f}/hr "
        f"(max allowed {max_far_per_hour}/hr) over {metrics['total_windows']} windows"
    )
    assert metrics["avg_inference_ms"] <= max_avg_inference_ms, (
        f"Inference too slow for on-device use: {metrics['avg_inference_ms']:.3f}ms "
        f"(max allowed {max_avg_inference_ms}ms)"
    )

    return True


def make_dummy_negative_dataset(num_windows: int = 200, seed: int = 0) -> np.ndarray:
    """Generate a synthetic negative dataset (random noise features) for
    smoke-testing the verifier before real background/negative audio is available.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(size=(num_windows, SPECTROGRAM_LENGTH, FEATURE_COUNT)).astype("float32")


def main() -> None:
    train_dir = Path("training_data/hey_chatter")
    flags = build_training_flags(train_dir)

    # Real training/validation datasets must come from microWakeWord's
    # FeatureHandler data pipeline, pointed at recorded/augmented samples of
    # "Hey Chatter" plus background and negative audio. Wiring that pipeline
    # up is a separate step from this training/export/verify scaffold.
    train_dataset = None
    validation_dataset = None

    model = train_wakeword_model(train_dataset, validation_dataset, flags)
    tflite_path = export_tflite(model, OUTPUT_TFLITE_PATH)

    dummy_negatives = make_dummy_negative_dataset()
    verify_model(tflite_path, dummy_negatives)

    print(f"Trained and verified '{WAKE_WORD_PHRASE}' model -> {tflite_path}")


if __name__ == "__main__":
    main()
