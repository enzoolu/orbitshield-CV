from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    print("TensorFlow nao instalado. Instale as dependencias antes de prever.")
    raise


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
SAMPLES_DIR = ROOT_DIR / "samples"
TRAIN_DIR = ROOT_DIR / "dataset" / "processed" / "train"
IMAGE_SIZE = (128, 128)


def local_model_candidates() -> list[Path]:
    return [
        MODELS_DIR / "simple_cnn_inference.keras",
        MODELS_DIR / "simple_cnn.keras",
        MODELS_DIR / "deep_cnn.keras",
        MODELS_DIR / "deep_cnn_inference.keras",
    ]


def resolve_best_model() -> Path:
    for candidate in local_model_candidates():
        if candidate.exists():
            return candidate

    comparison_path = OUTPUTS_DIR / "model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        if "selected_best" in comparison.columns and comparison["selected_best"].astype(bool).any():
            row = comparison.sort_values(["selected_best", "test_accuracy", "f1_macro"], ascending=[False, False, False]).iloc[0]
            model_path = Path(str(row["model_path"]))
            if model_path.exists():
                return model_path
        row = comparison.sort_values(["test_accuracy", "f1_macro"], ascending=[False, False]).iloc[0]
        model_path = Path(str(row["model_path"]))
        if model_path.exists():
            return model_path
    raise FileNotFoundError("Nenhum modelo treinado encontrado.")


def resolve_image_path(image_path: str | None) -> Path:
    if image_path:
        path = Path(image_path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    sample_images = sorted(SAMPLES_DIR.glob("*.jpg"))
    if not sample_images:
        raise FileNotFoundError("Nenhuma imagem de amostra encontrada em samples/.")
    return sample_images[0]


def load_class_names() -> list[str]:
    class_dirs = sorted([path.name for path in TRAIN_DIR.iterdir() if path.is_dir()])
    if not class_dirs:
        raise FileNotFoundError("Classes nao encontradas em dataset/processed/train.")
    return class_dirs


def preprocess_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32)
        array /= 255.0
    return np.expand_dims(array, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Predicao OrbitShield Vision")
    parser.add_argument("image", nargs="?", help="Caminho opcional para a imagem")
    args = parser.parse_args()

    model_path = resolve_best_model()
    image_path = resolve_image_path(args.image)
    class_names = load_class_names()

    model = tf.keras.models.load_model(model_path, compile=False)
    probabilities = model.predict(preprocess_image(image_path), verbose=0)[0]
    prediction_idx = int(np.argmax(probabilities))

    print(f"Imagem: {image_path}")
    print(f"Modelo: {model_path}")
    print(f"Classe prevista: {class_names[prediction_idx]}")
    print(f"Confianca: {probabilities[prediction_idx]:.2%}")
    print("Probabilidades por classe:")
    for class_name, probability in zip(class_names, probabilities):
        print(f"  - {class_name}: {probability:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
