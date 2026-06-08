from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    print("TensorFlow não está instalado. Instale as dependências antes de regravar os modelos.")
    raise


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODEL_NAMES = ("simple_cnn.keras", "deep_cnn.keras")
GITHUB_LIMIT_BYTES = 100 * 1024 * 1024


def size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def save_without_optimizer(model: tf.keras.Model, output_path: Path) -> None:
    temp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    if temp_path.exists():
        temp_path.unlink()

    try:
        model.save(temp_path, include_optimizer=False)
    except TypeError:
        model.save(temp_path)

    if output_path.exists():
        output_path.unlink()
    temp_path.replace(output_path)


def resave_model(model_path: Path) -> tuple[float, float]:
    before = size_mb(model_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    save_without_optimizer(model, model_path)
    after = size_mb(model_path)
    return before, after


def resolve_best_model_name() -> str:
    comparison_path = OUTPUTS_DIR / "model_comparison.csv"
    if not comparison_path.exists():
        return "simple_cnn.keras"

    comparison = pd.read_csv(comparison_path)
    if comparison.empty:
        return "simple_cnn.keras"

    if "selected_best" in comparison.columns and comparison["selected_best"].astype(bool).any():
        row = comparison.sort_values(["selected_best", "test_accuracy", "f1_macro"], ascending=[False, False, False]).iloc[0]
    else:
        row = comparison.sort_values(["test_accuracy", "f1_macro"], ascending=[False, False]).iloc[0]

    model_name = str(row["model_name"]).strip()
    if model_name == "DeepCNN":
        return "deep_cnn.keras"
    return "simple_cnn.keras"


def create_inference_copy() -> Path:
    source_path = MODELS_DIR / resolve_best_model_name()
    target_path = MODELS_DIR / f"{source_path.stem}_inference{source_path.suffix}"
    model = tf.keras.models.load_model(source_path, compile=False)
    save_without_optimizer(model, target_path)
    return target_path


def main() -> int:
    print("[INÍCIO] Regravação dos modelos sem optimizer")
    missing = [name for name in MODEL_NAMES if not (MODELS_DIR / name).exists()]
    if missing:
        print(f"Modelos ausentes: {', '.join(missing)}")
        return 1

    status_ok = True
    for model_name in MODEL_NAMES:
        model_path = MODELS_DIR / model_name
        before_mb, after_mb = resave_model(model_path)
        within_limit = model_path.stat().st_size < GITHUB_LIMIT_BYTES
        status_ok = status_ok and within_limit
        print(
            f"{model_name}: {before_mb:.2f} MB -> {after_mb:.2f} MB | "
            f"abaixo de 100 MB: {'sim' if within_limit else 'não'}"
        )

    inference_path = create_inference_copy()
    inference_within_limit = inference_path.stat().st_size < GITHUB_LIMIT_BYTES
    print(
        f"{inference_path.name}: {size_mb(inference_path):.2f} MB | "
        f"abaixo de 100 MB: {'sim' if inference_within_limit else 'não'}"
    )

    print("[FIM] Regravação concluída")
    return 0 if (status_ok and inference_within_limit) else 2


if __name__ == "__main__":
    sys.exit(main())
