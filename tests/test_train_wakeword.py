import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from train_wakeword import (
    FEATURE_COUNT,
    SPECTROGRAM_LENGTH,
    build_training_flags,
    compute_false_acceptance_rate,
    export_tflite,
    make_dummy_negative_dataset,
    verify_model,
)


def _build_tiny_test_model():
    inputs = tf.keras.Input(shape=(SPECTROGRAM_LENGTH, FEATURE_COUNT))
    x = tf.keras.layers.Flatten()(inputs)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)


def test_build_training_flags_sets_wake_phrase_and_sample_rate(tmp_path):
    flags = build_training_flags(tmp_path)

    assert flags.wanted_words == "Hey Chatter"
    assert flags.sample_rate == 16000
    assert flags.train_dir == str(tmp_path)


def test_export_tflite_writes_a_loadable_model(tmp_path):
    model = _build_tiny_test_model()
    output_path = tmp_path / "test_model.tflite"

    result_path = export_tflite(model, output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    interpreter = tf.lite.Interpreter(model_path=str(output_path))
    interpreter.allocate_tensors()
    assert len(interpreter.get_input_details()) == 1
    assert len(interpreter.get_output_details()) == 1


def test_make_dummy_negative_dataset_shape():
    dataset = make_dummy_negative_dataset(num_windows=10)

    assert dataset.shape == (10, SPECTROGRAM_LENGTH, FEATURE_COUNT)
    assert dataset.dtype == np.float32


def test_compute_false_acceptance_rate_counts_detections_above_threshold(tmp_path, monkeypatch):
    model = _build_tiny_test_model()
    tflite_path = export_tflite(model, tmp_path / "far_test.tflite")
    negatives = make_dummy_negative_dataset(num_windows=5)

    scores = iter([0.1, 0.95, 0.2, 0.99, 0.3])
    monkeypatch.setattr(
        "train_wakeword._run_inference",
        lambda interpreter, features: next(scores),
    )

    metrics = compute_false_acceptance_rate(
        tflite_path, negatives, window_duration_s=1.0, detection_threshold=0.5
    )

    assert metrics["total_windows"] == 5
    assert metrics["false_accepts"] == 2
    assert metrics["far_per_hour"] == pytest.approx(2 / (5 / 3600))


def test_verify_model_raises_when_far_exceeds_target(tmp_path, monkeypatch):
    model = _build_tiny_test_model()
    tflite_path = export_tflite(model, tmp_path / "verify_far.tflite")
    negatives = make_dummy_negative_dataset(num_windows=5)

    monkeypatch.setattr(
        "train_wakeword.compute_false_acceptance_rate",
        lambda *a, **k: {
            "false_accepts": 10,
            "total_windows": 5,
            "far_per_hour": 100.0,
            "avg_inference_ms": 1.0,
        },
    )

    with pytest.raises(AssertionError, match="False Acceptance Rate too high"):
        verify_model(tflite_path, negatives, max_far_per_hour=0.5)


def test_verify_model_raises_when_inference_too_slow(tmp_path, monkeypatch):
    model = _build_tiny_test_model()
    tflite_path = export_tflite(model, tmp_path / "verify_slow.tflite")
    negatives = make_dummy_negative_dataset(num_windows=5)

    monkeypatch.setattr(
        "train_wakeword.compute_false_acceptance_rate",
        lambda *a, **k: {
            "false_accepts": 0,
            "total_windows": 5,
            "far_per_hour": 0.0,
            "avg_inference_ms": 50.0,
        },
    )

    with pytest.raises(AssertionError, match="Inference too slow"):
        verify_model(tflite_path, negatives, max_avg_inference_ms=5.0)


def test_verify_model_passes_when_within_targets(tmp_path, monkeypatch):
    model = _build_tiny_test_model()
    tflite_path = export_tflite(model, tmp_path / "verify_pass.tflite")
    negatives = make_dummy_negative_dataset(num_windows=5)

    monkeypatch.setattr(
        "train_wakeword.compute_false_acceptance_rate",
        lambda *a, **k: {
            "false_accepts": 0,
            "total_windows": 5,
            "far_per_hour": 0.0,
            "avg_inference_ms": 1.0,
        },
    )

    assert verify_model(tflite_path, negatives) is True
