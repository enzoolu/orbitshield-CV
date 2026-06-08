from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    print("TensorFlow nao instalado. Instale as dependencias antes de treinar.")
    raise


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "dataset" / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 45
SEED = 42


@dataclass
class ModelResult:
    model_name: str
    model_path: str
    test_loss: float
    test_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    meets_reference_88: bool
    selected_best: bool = False


def set_seed() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def build_generators():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=8,
        width_shift_range=0.06,
        height_shift_range=0.06,
        zoom_range=0.10,
        brightness_range=(0.90, 1.10),
        horizontal_flip=True,
        fill_mode="nearest",
    )
    eval_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)

    common_args = {
        "target_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "class_mode": "sparse",
        "color_mode": "rgb",
        "seed": SEED,
    }

    train_gen = train_datagen.flow_from_directory(
        str(PROCESSED_DIR / "train"),
        shuffle=True,
        **common_args,
    )
    val_gen = eval_datagen.flow_from_directory(
        str(PROCESSED_DIR / "val"),
        shuffle=False,
        **common_args,
    )
    test_gen = eval_datagen.flow_from_directory(
        str(PROCESSED_DIR / "test"),
        shuffle=False,
        **common_args,
    )

    class_names = [name for name, _ in sorted(train_gen.class_indices.items(), key=lambda item: item[1])]
    return train_gen, val_gen, test_gen, class_names


def build_simple_cnn(num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
            tf.keras.layers.Conv2D(48, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(96, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(160, activation="relu"),
            tf.keras.layers.Dropout(0.40),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="SimpleCNN",
    )
    return model


def build_deep_cnn(num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
            tf.keras.layers.Conv2D(24, (3, 3), padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.10),
            tf.keras.layers.Conv2D(48, (3, 3), padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Conv2D(96, (3, 3), padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(96, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="DeepCNN",
    )
    return model


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def save_model_summary(model: tf.keras.Model, output_path: Path) -> None:
    lines: List[str] = []
    model.summary(print_fn=lines.append)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_model_without_optimizer(model: tf.keras.Model, output_path: Path) -> None:
    temp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    if temp_path.exists():
        temp_path.unlink()

    try:
        model.save(temp_path, include_optimizer=False)
    except TypeError:
        # Some TensorFlow/Keras builds ignore `include_optimizer` for native `.keras`.
        # Saving a model loaded without recompiling still strips optimizer state.
        reloaded = tf.keras.models.load_model(output_path, compile=False) if output_path.exists() else model
        reloaded.save(temp_path)

    if output_path.exists():
        output_path.unlink()
    temp_path.replace(output_path)


def plot_training_curves(history: tf.keras.callbacks.History, model_slug: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Treino")
    axes[0].plot(history.history["val_accuracy"], label="Validacao")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Treino")
    axes[1].plot(history.history["val_loss"], label="Validacao")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / f"training_curves_{model_slug}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def collect_test_arrays(test_gen) -> Tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []
    test_gen.reset()
    steps = int(np.ceil(test_gen.samples / test_gen.batch_size))
    for _ in range(steps):
        batch_images, batch_labels = next(test_gen)
        images.append(batch_images)
        labels.append(batch_labels)
    test_gen.reset()
    return (
        np.concatenate(images, axis=0)[: test_gen.samples],
        np.concatenate(labels, axis=0)[: test_gen.samples].astype(int),
    )


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], model_slug: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusao - {model_slug}")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / f"confusion_matrix_{model_slug}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_model(
    model: tf.keras.Model,
    model_name: str,
    model_slug: str,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    class_names: List[str],
) -> ModelResult:
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH_SIZE)
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    probabilities = model.predict(test_ds, verbose=0)
    predictions = probabilities.argmax(axis=1)

    precision = precision_score(test_labels, predictions, average="macro", zero_division=0)
    recall = recall_score(test_labels, predictions, average="macro", zero_division=0)
    f1 = f1_score(test_labels, predictions, average="macro", zero_division=0)

    report_text = classification_report(test_labels, predictions, target_names=class_names, zero_division=0)
    (OUTPUTS_DIR / f"classification_report_{model_slug}.txt").write_text(report_text + "\n", encoding="utf-8")

    plot_confusion_matrix(test_labels, predictions, class_names, model_slug)

    return ModelResult(
        model_name=model_name,
        model_path=str((MODELS_DIR / f"{model_slug}.keras").resolve()),
        test_loss=float(test_loss),
        test_accuracy=float(test_accuracy),
        precision_macro=float(precision),
        recall_macro=float(recall),
        f1_macro=float(f1),
        meets_reference_88=bool(test_accuracy >= 0.88),
    )


def render_prediction_gallery(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_names: List[str],
    output_path: Path,
    mode: str,
    max_items: int = 9,
) -> None:
    if mode == "correct":
        indices = np.where(y_true == y_pred)[0]
        title = "Exemplos de Acertos"
    else:
        indices = np.where(y_true != y_pred)[0]
        title = "Exemplos de Erros"

    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    axes = axes.flatten()

    if len(indices) == 0:
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{title} - nenhum exemplo disponivel", fontsize=14)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    for ax, idx in zip(axes, indices[:max_items]):
        ax.imshow(np.clip(images[idx] * 255.0, 0, 255).astype("uint8"))
        ax.set_title(
            f"Real: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}\nConf: {probabilities[idx][y_pred[idx]]:.2%}",
            fontsize=9,
        )
        ax.axis("off")

    for ax in axes[len(indices[:max_items]) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_dataset_source_counts() -> List[Dict[str, str | int]]:
    source_counts_path = OUTPUTS_DIR / "dataset_source_totals.csv"
    if not source_counts_path.exists():
        return []
    source_counts_df = pd.read_csv(source_counts_path)
    return source_counts_df.to_dict(orient="records")


def save_training_summary(
    class_names: List[str],
    class_counts: Dict[str, Dict[str, int]],
    results: List[ModelResult],
    best_model: str,
) -> None:
    payload = {
        "class_names": class_names,
        "class_counts": class_counts,
        "dataset_source_counts": load_dataset_source_counts(),
        "best_model": best_model,
        "results": [asdict(item) for item in results],
    }
    (OUTPUTS_DIR / "training_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def count_processed_images() -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for split_dir in sorted(PROCESSED_DIR.iterdir()):
        if not split_dir.is_dir():
            continue
        counts[split_dir.name] = {}
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                counts[split_dir.name][class_dir.name] = len(list(class_dir.glob("*.jpg")))
    return counts


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ensure_dirs()
    set_seed()

    if not (PROCESSED_DIR / "train").exists():
        print("Dataset processado nao encontrado. Execute src/prepare_dataset.py primeiro.")
        return 1

    print("[INICIO] Treinamento das CNNs do OrbitShield Vision")
    train_gen, val_gen, test_gen, class_names = build_generators()
    class_counts = count_processed_images()
    test_images, test_labels = collect_test_arrays(test_gen)
    num_classes = len(class_names)

    model_specs = [
        ("SimpleCNN", "simple_cnn", build_simple_cnn(num_classes), 1.5e-4),
        ("DeepCNN", "deep_cnn", build_deep_cnn(num_classes), 8e-5),
    ]

    results: List[ModelResult] = []

    for model_name, model_slug, model, learning_rate in model_specs:
        print(f"\n[TREINO] {model_name}")
        train_gen.reset()
        val_gen.reset()
        model = compile_model(model, learning_rate)
        save_model_summary(model, OUTPUTS_DIR / f"model_summary_{model_slug}.txt")

        checkpoint_path = MODELS_DIR / f"{model_slug}.keras"
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=10,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1,
        )
        plot_training_curves(history, model_slug)

        trained_model = tf.keras.models.load_model(checkpoint_path)
        result = evaluate_model(trained_model, model_name, model_slug, test_images, test_labels, class_names)
        save_model_without_optimizer(trained_model, checkpoint_path)
        results.append(result)
        print(
            f"  test_accuracy={result.test_accuracy:.4f} | "
            f"test_loss={result.test_loss:.4f} | "
            f"f1_macro={result.f1_macro:.4f}"
        )

    results_sorted = sorted(results, key=lambda item: (item.test_accuracy, item.f1_macro), reverse=True)
    best_model_name = results_sorted[0].model_name
    for item in results:
        if item.model_name == best_model_name:
            item.selected_best = True

    comparison_df = pd.DataFrame([asdict(item) for item in results]).sort_values(
        by=["selected_best", "test_accuracy", "f1_macro"],
        ascending=[False, False, False],
    )
    comparison_df.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)

    best_slug = "simple_cnn" if best_model_name == "SimpleCNN" else "deep_cnn"
    best_model = tf.keras.models.load_model(MODELS_DIR / f"{best_slug}.keras")
    best_probabilities = best_model.predict(tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH_SIZE), verbose=0)
    best_predictions = best_probabilities.argmax(axis=1)

    render_prediction_gallery(
        test_images,
        test_labels,
        best_predictions,
        best_probabilities,
        class_names,
        OUTPUTS_DIR / "correct_predictions.png",
        mode="correct",
    )
    render_prediction_gallery(
        test_images,
        test_labels,
        best_predictions,
        best_probabilities,
        class_names,
        OUTPUTS_DIR / "wrong_predictions.png",
        mode="wrong",
    )

    save_training_summary(class_names, class_counts, results, best_model_name)

    print(f"\n[MELHOR MODELO] {best_model_name}")
    print(f"Comparacao salva em: {OUTPUTS_DIR / 'model_comparison.csv'}")
    print("[FIM] Treinamento concluido")
    return 0


if __name__ == "__main__":
    sys.exit(main())
