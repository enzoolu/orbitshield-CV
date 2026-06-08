from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError


Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MIN_DIMENSION = 128
LOW_VARIANCE_STD_THRESHOLD = 8.0
APPROX_DUPLICATE_HAMMING_THRESHOLD = 4
APPROX_DUPLICATE_COLOR_DISTANCE_THRESHOLD = 24.0


@dataclass
class ImageRecord:
    path: Path
    width: int
    height: int
    file_size: int
    sha256: str
    dhash: int
    grayscale_std: float
    entropy: float
    mean_rgb: Tuple[float, float, float]


@dataclass
class ApproxDuplicatePair:
    left: Path
    right: Path
    hamming_distance: int
    color_distance: float


@dataclass
class ClassAudit:
    class_name: str
    total_files: int = 0
    valid_records: List[ImageRecord] = field(default_factory=list)
    broken_files: List[Path] = field(default_factory=list)
    exact_duplicate_groups: List[List[Path]] = field(default_factory=list)
    approximate_duplicate_pairs: List[ApproxDuplicatePair] = field(default_factory=list)
    small_images: List[ImageRecord] = field(default_factory=list)
    low_variance_images: List[ImageRecord] = field(default_factory=list)


def iter_image_paths(directory: Path) -> Iterable[Path]:
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    grayscale = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = 0
    for value in diff.flatten():
        bits = (bits << 1) | int(value)
    return bits


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def color_distance(left: Tuple[float, float, float], right: Tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def load_image_record(path: Path) -> ImageRecord | None:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            image.load()
            stat = ImageStat.Stat(image)
            grayscale = image.convert("L")
            grayscale_stat = ImageStat.Stat(grayscale)
            return ImageRecord(
                path=path,
                width=image.width,
                height=image.height,
                file_size=path.stat().st_size,
                sha256=compute_sha256(path),
                dhash=compute_dhash(image),
                grayscale_std=float(grayscale_stat.stddev[0]),
                entropy=float(grayscale.entropy()),
                mean_rgb=(float(stat.mean[0]), float(stat.mean[1]), float(stat.mean[2])),
            )
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def scan_class_directory(
    class_dir: Path,
    min_dimension: int = MIN_DIMENSION,
    low_variance_threshold: float = LOW_VARIANCE_STD_THRESHOLD,
    approx_hamming_threshold: int = APPROX_DUPLICATE_HAMMING_THRESHOLD,
    approx_color_threshold: float = APPROX_DUPLICATE_COLOR_DISTANCE_THRESHOLD,
) -> ClassAudit:
    audit = ClassAudit(class_name=class_dir.name)
    sha_groups: Dict[str, List[Path]] = {}

    for image_path in iter_image_paths(class_dir):
        audit.total_files += 1
        record = load_image_record(image_path)
        if record is None:
            audit.broken_files.append(image_path)
            continue

        audit.valid_records.append(record)
        sha_groups.setdefault(record.sha256, []).append(record.path)

        if min(record.width, record.height) < min_dimension:
            audit.small_images.append(record)
        if record.grayscale_std < low_variance_threshold:
            audit.low_variance_images.append(record)

    audit.valid_records.sort(key=lambda item: item.path.name)
    audit.exact_duplicate_groups = [
        sorted(group)
        for _, group in sorted(sha_groups.items())
        if len(group) > 1
    ]

    for index, left in enumerate(audit.valid_records):
        for right in audit.valid_records[index + 1 :]:
            if left.sha256 == right.sha256:
                continue
            distance = hamming_distance(left.dhash, right.dhash)
            if distance > approx_hamming_threshold:
                continue
            mean_distance = color_distance(left.mean_rgb, right.mean_rgb)
            if mean_distance > approx_color_threshold:
                continue
            audit.approximate_duplicate_pairs.append(
                ApproxDuplicatePair(
                    left=left.path,
                    right=right.path,
                    hamming_distance=distance,
                    color_distance=mean_distance,
                )
            )

    return audit


def audit_dataset(raw_dir: Path) -> List[ClassAudit]:
    reports: List[ClassAudit] = []
    for class_dir in sorted(raw_dir.iterdir()):
        if class_dir.is_dir():
            reports.append(scan_class_directory(class_dir))
    return reports
