from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from dataset_utils import IMAGE_EXTENSIONS


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "dataset" / "raw"
SYNTHETIC_DIR = ROOT_DIR / "dataset" / "synthetic"
SYNTHETIC_MANIFEST = SYNTHETIC_DIR / "synthetic_manifest.csv"
PROCESSED_DIR = ROOT_DIR / "dataset" / "processed"
SAMPLES_DIR = ROOT_DIR / "samples"
DOCS_DIR = ROOT_DIR / "docs"
OUTPUTS_DIR = ROOT_DIR / "outputs"
IMAGE_SIZE = (128, 128)
RANDOM_SEED = 42
SOURCE_PRIORITY = {"real": 0, "synthetic": 1}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

Image.MAX_IMAGE_PIXELS = None


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def collect_real_images() -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    if not RAW_DIR.exists():
        return records

    for class_dir in sorted(RAW_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            records.append(
                {
                    "path": str(image_path.resolve()),
                    "class_name": class_dir.name,
                    "source_type": "real",
                    "output_name": f"real_{image_path.stem}.jpg",
                    "group_id": f"real::{class_dir.name}::{image_path.stem}",
                }
            )
    return records


def collect_synthetic_images() -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    if SYNTHETIC_MANIFEST.exists():
        manifest_df = pd.read_csv(SYNTHETIC_MANIFEST)
        for row in manifest_df.itertuples(index=False):
            image_path = Path(row.path)
            if not image_path.exists():
                continue
            records.append(
                {
                    "path": str(image_path.resolve()),
                    "class_name": row.class_name,
                    "source_type": "synthetic",
                    "output_name": f"synthetic_{Path(row.file_name).stem}.jpg",
                    "group_id": str(row.group_id),
                }
            )
        return records

    if not SYNTHETIC_DIR.exists():
        return records

    for class_dir in sorted(SYNTHETIC_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            records.append(
                {
                    "path": str(image_path.resolve()),
                    "class_name": class_dir.name,
                    "source_type": "synthetic",
                    "output_name": f"synthetic_{image_path.stem}.jpg",
                    "group_id": f"synthetic::{class_dir.name}::{image_path.stem}",
                }
            )
    return records


def build_dataset_index() -> pd.DataFrame:
    records = collect_real_images()
    records.extend(collect_synthetic_images())
    columns = ["path", "class_name", "source_type", "output_name", "group_id"]
    if not records:
        return pd.DataFrame(columns=columns)
    dataset_df = pd.DataFrame(records)
    return dataset_df.sort_values(["class_name", "source_type", "group_id", "path"]).reset_index(drop=True)


def split_subset_by_group(subset_df: pd.DataFrame, seed: int) -> Dict[str, pd.DataFrame]:
    group_ids = sorted(subset_df["group_id"].unique().tolist())
    if len(group_ids) < 3:
        return {"train": subset_df.copy(), "val": subset_df.iloc[0:0].copy(), "test": subset_df.iloc[0:0].copy()}

    rng = random.Random(seed)
    rng.shuffle(group_ids)

    total_groups = len(group_ids)
    val_count = int(round(total_groups * SPLIT_RATIOS["val"]))
    test_count = int(round(total_groups * SPLIT_RATIOS["test"]))
    train_count = max(1, total_groups - val_count - test_count)

    train_groups = group_ids[:train_count]
    val_groups = group_ids[train_count : train_count + val_count]
    test_groups = group_ids[train_count + val_count :]

    if not val_groups and test_groups:
        val_groups = [test_groups.pop(0)]
    if not test_groups and val_groups:
        test_groups = [val_groups.pop()]

    split_groups = {
        "train": set(train_groups),
        "val": set(val_groups),
        "test": set(test_groups),
    }
    return {
        split_name: subset_df[subset_df["group_id"].isin(group_set)].copy()
        for split_name, group_set in split_groups.items()
    }


def group_aware_split(dataset_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if dataset_df.empty:
        empty = dataset_df.copy()
        return {"train": empty, "val": empty, "test": empty}

    split_parts: Dict[str, List[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    grouped = dataset_df.groupby(["class_name", "source_type"], sort=True)

    for index, ((class_name, source_type), subset_df) in enumerate(grouped):
        subset_splits = split_subset_by_group(subset_df.reset_index(drop=True), RANDOM_SEED + index * 101)
        for split_name, split_df in subset_splits.items():
            if not split_df.empty:
                split_parts[split_name].append(split_df)

    final_splits: Dict[str, pd.DataFrame] = {}
    for split_name, parts in split_parts.items():
        if parts:
            final_splits[split_name] = pd.concat(parts, ignore_index=True).sort_values(
                ["class_name", "source_type", "group_id", "path"]
            ).reset_index(drop=True)
        else:
            final_splits[split_name] = dataset_df.iloc[0:0].copy()
    return final_splits


def process_and_save_image(source_path: Path, destination_path: Path) -> bool:
    try:
        with Image.open(source_path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination_path.with_suffix(".jpg"), format="JPEG", quality=95, optimize=True)
        return True
    except (UnidentifiedImageError, OSError):
        return False


def copy_samples(test_df: pd.DataFrame) -> None:
    reset_dir(SAMPLES_DIR)
    for class_name, group_df in test_df.groupby("class_name"):
        ordered = group_df.assign(source_rank=group_df["source_type"].map(SOURCE_PRIORITY)).sort_values(
            ["source_rank", "path"]
        )
        for index, row in enumerate(ordered.head(3).itertuples(index=False), start=1):
            destination = SAMPLES_DIR / f"{class_name}_{index:02d}.jpg"
            process_and_save_image(Path(row.path), destination)


def count_by_class_and_source(dataset_df: pd.DataFrame) -> pd.DataFrame:
    if dataset_df.empty:
        return pd.DataFrame(columns=["class_name", "source_type", "count"])
    counts = dataset_df.groupby(["class_name", "source_type"]).size().reset_index(name="count")
    return counts.sort_values(["class_name", "source_type"]).reset_index(drop=True)


def build_split_counts(split_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    class_names = sorted(
        {
            class_name
            for split_df in split_frames.values()
            for class_name in split_df["class_name"].unique().tolist()
        }
    )
    rows = []
    for split_name, split_df in split_frames.items():
        counts = {class_name: 0 for class_name in class_names}
        counts.update(split_df["class_name"].value_counts().to_dict())
        counts["split"] = split_name
        rows.append(counts)
    return pd.DataFrame(rows).fillna(0).set_index("split").astype(int).sort_index()


def build_source_split_counts(split_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split_name, split_df in split_frames.items():
        grouped = split_df.groupby(["class_name", "source_type"]).size()
        row = {"split": split_name}
        for (class_name, source_type), count in grouped.items():
            row[f"{class_name}_{source_type}"] = int(count)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).fillna(0).set_index("split").astype(int).sort_index()


def validate_split_integrity(split_frames: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    path_locations: Dict[str, str] = {}
    group_locations: Dict[str, str] = {}
    path_leaks: List[Dict[str, str]] = []
    group_leaks: List[Dict[str, str]] = []

    for split_name, split_df in split_frames.items():
        for row in split_df.itertuples(index=False):
            previous_path_split = path_locations.get(row.path)
            if previous_path_split and previous_path_split != split_name:
                path_leaks.append({"path": row.path, "split_a": previous_path_split, "split_b": split_name})
            else:
                path_locations[row.path] = split_name

            previous_group_split = group_locations.get(row.group_id)
            if previous_group_split and previous_group_split != split_name:
                group_leaks.append({"group_id": row.group_id, "split_a": previous_group_split, "split_b": split_name})
            else:
                group_locations[row.group_id] = split_name

    return {
        "path_leaks": path_leaks,
        "group_leaks": group_leaks,
        "path_leak_count": len(path_leaks),
        "group_leak_count": len(group_leaks),
    }


def write_split_report(
    dataset_df: pd.DataFrame,
    split_frames: Dict[str, pd.DataFrame],
    counts_df: pd.DataFrame,
    source_split_df: pd.DataFrame,
    failures: int,
    integrity: Dict[str, object],
) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    counts_df.to_csv(OUTPUTS_DIR / "dataset_split_summary.csv")
    source_split_df.to_csv(OUTPUTS_DIR / "dataset_source_split_summary.csv")
    dataset_df.to_csv(OUTPUTS_DIR / "dataset_manifest.csv", index=False)
    count_by_class_and_source(dataset_df).to_csv(OUTPUTS_DIR / "dataset_source_totals.csv", index=False)
    (OUTPUTS_DIR / "split_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    split_table = [
        "| Split | " + " | ".join(counts_df.columns.tolist()) + " |",
        "| --- | " + " | ".join(["---:"] * len(counts_df.columns)) + " |",
    ]
    for split_name, row in counts_df.iterrows():
        split_table.append(
            "| " + split_name + " | " + " | ".join(str(int(row[column])) for column in counts_df.columns) + " |"
        )

    source_total_counts = count_by_class_and_source(dataset_df)
    source_table = [
        "| Classe | Real | Sintetica | Total |",
        "| --- | ---: | ---: | ---: |",
    ]
    for class_name in sorted(dataset_df["class_name"].unique()):
        real_count = int(
            source_total_counts[
                (source_total_counts["class_name"] == class_name) & (source_total_counts["source_type"] == "real")
            ]["count"].sum()
        )
        synthetic_count = int(
            source_total_counts[
                (source_total_counts["class_name"] == class_name)
                & (source_total_counts["source_type"] == "synthetic")
            ]["count"].sum()
        )
        source_table.append(f"| {class_name} | {real_count} | {synthetic_count} | {real_count + synthetic_count} |")

    source_split_table = []
    if not source_split_df.empty:
        source_split_table.extend(
            [
                "| Split | " + " | ".join(source_split_df.columns.tolist()) + " |",
                "| --- | " + " | ".join(["---:"] * len(source_split_df.columns)) + " |",
            ]
        )
        for split_name, row in source_split_df.iterrows():
            source_split_table.append(
                "| "
                + split_name
                + " | "
                + " | ".join(str(int(row[column])) for column in source_split_df.columns)
                + " |"
            )

    lines = [
        "# Relatorio de Split do Dataset",
        "",
        "## Dataset Hibrido Final",
        "",
        *source_table,
        "",
        "## Distribuicao por Split",
        "",
        *split_table,
        "",
        "## Distribuicao por Origem e Split",
        "",
        *source_split_table,
        "",
        "## Integridade do Split",
        "",
        f"- Vazamentos de caminho entre splits: {integrity['path_leak_count']}",
        f"- Vazamentos de grupo entre splits: {integrity['group_leak_count']}",
        "- As imagens sinteticas foram divididas por `group_id` de familia para evitar que variantes muito parecidas caiam em splits diferentes.",
        "",
        "## Observacoes",
        "",
        "- O dataset processado combina imagens reais em `dataset/raw/` com imagens simuladas em `dataset/synthetic/`.",
        "- O split foi gerado por classe e origem, preservando grupos sinteticos inteiros para evitar vazamento.",
        "- A distribuicao por classe foi mantida balanceada entre treino, validacao e teste.",
        f"- Falhas de processamento de imagem durante o resize: {failures}",
    ]
    (DOCS_DIR / "dataset_split_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    random.seed(RANDOM_SEED)
    dataset_df = build_dataset_index()
    if dataset_df.empty:
        print("Nenhuma imagem encontrada em dataset/raw ou dataset/synthetic.")
        return 1

    for split_name in ("train", "val", "test"):
        reset_dir(PROCESSED_DIR / split_name)

    split_frames = group_aware_split(dataset_df)
    integrity = validate_split_integrity(split_frames)
    if integrity["path_leak_count"] or integrity["group_leak_count"]:
        print("Falha de integridade no split. Corrija os grupos antes de continuar.")
        return 1

    failures = 0
    print("[INICIO] Preparacao do dataset hibrido")

    for split_name, split_df in split_frames.items():
        for row in split_df.itertuples(index=False):
            destination = PROCESSED_DIR / split_name / row.class_name / row.output_name
            ok = process_and_save_image(Path(row.path), destination)
            if not ok:
                failures += 1

    counts_df = build_split_counts(split_frames)
    source_split_df = build_source_split_counts(split_frames)
    write_split_report(dataset_df, split_frames, counts_df, source_split_df, failures, integrity)
    copy_samples(split_frames["test"])

    total_by_class = dataset_df["class_name"].value_counts().to_dict()
    split_lookup: Dict[str, Dict[str, int]] = {
        split_name: split_df["class_name"].value_counts().to_dict() for split_name, split_df in split_frames.items()
    }
    source_counts = count_by_class_and_source(dataset_df)

    for class_name in sorted(total_by_class):
        real_count = int(
            source_counts[(source_counts["class_name"] == class_name) & (source_counts["source_type"] == "real")][
                "count"
            ].sum()
        )
        synthetic_count = int(
            source_counts[
                (source_counts["class_name"] == class_name) & (source_counts["source_type"] == "synthetic")
            ]["count"].sum()
        )
        print(
            f"  - {class_name}: total={total_by_class[class_name]} "
            f"(real={real_count}, sintetica={synthetic_count}) | "
            f"train={split_lookup['train'].get(class_name, 0)} | "
            f"val={split_lookup['val'].get(class_name, 0)} | "
            f"test={split_lookup['test'].get(class_name, 0)}"
        )

    print(f"  - integridade: path_leaks={integrity['path_leak_count']} | group_leaks={integrity['group_leak_count']}")
    print(f"[FIM] Dataset preparado. Falhas no processamento: {failures}")
    print(f"Samples salvos em: {SAMPLES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
