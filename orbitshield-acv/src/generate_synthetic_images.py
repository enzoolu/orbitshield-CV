from __future__ import annotations

import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter


ROOT_DIR = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = ROOT_DIR / "dataset" / "synthetic"
MANIFEST_PATH = SYNTHETIC_DIR / "synthetic_manifest.csv"
IMAGE_SIZE = (128, 128)
CANVAS_SIZE = (256, 256)
FAMILIES_PER_CLASS = 180
VARIANTS_PER_FAMILY = 5
IMAGES_PER_CLASS = FAMILIES_PER_CLASS * VARIANTS_PER_FAMILY
SEED = 42


@dataclass
class FamilySpec:
    class_name: str
    group_id: str
    family_index: int
    body_width: int
    body_height: int
    panel_width: int = 0
    panel_height: int = 0
    panel_gap: int = 0
    cone_height: int = 0
    fin_width: int = 0
    fin_height: int = 0
    wing_span: int = 0
    wing_depth: int = 0
    tail_height: int = 0
    nose_length: int = 0
    body_color: Tuple[int, int, int, int] = (235, 235, 240, 255)
    accent_color: Tuple[int, int, int, int] = (60, 120, 210, 255)
    angle_range: Tuple[float, float] = (-20.0, 20.0)
    scale_range: Tuple[float, float] = (0.84, 1.08)
    x_jitter: int = 24
    y_jitter: int = 24
    preferred_y_offset: int = 0


def ensure_dirs() -> None:
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    for class_name in ("satellite", "rocket", "space_shuttle"):
        (SYNTHETIC_DIR / class_name).mkdir(parents=True, exist_ok=True)


def reset_class_dir(class_name: str) -> Path:
    class_dir = SYNTHETIC_DIR / class_name
    if class_dir.exists():
        for image_path in class_dir.glob("*.png"):
            image_path.unlink()
    class_dir.mkdir(parents=True, exist_ok=True)
    return class_dir


def random_space_background(rng: random.Random) -> Image.Image:
    width, height = CANVAS_SIZE
    top = np.array(
        [
            rng.randint(2, 10),
            rng.randint(4, 14),
            rng.randint(12, 28),
        ],
        dtype=np.float32,
    )
    bottom = np.array(
        [
            rng.randint(0, 8),
            rng.randint(0, 10),
            rng.randint(3, 20),
        ],
        dtype=np.float32,
    )

    base = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        alpha = y / max(1, height - 1)
        row = (1.0 - alpha) * top + alpha * bottom
        base[y, :, :] = np.clip(row, 0, 255).astype(np.uint8)

    background = Image.fromarray(base)
    draw = ImageDraw.Draw(background)

    for _ in range(rng.randint(120, 220)):
        x = rng.randint(0, width - 1)
        y = rng.randint(0, height - 1)
        radius = rng.choice([0, 0, 1, 1, 2])
        intensity = rng.randint(170, 255)
        star_color = (intensity, intensity, rng.randint(200, 255))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=star_color)

    if rng.random() < 0.65:
        haze = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
        haze_draw = ImageDraw.Draw(haze)
        for _ in range(rng.randint(2, 4)):
            x0 = rng.randint(-40, width - 40)
            y0 = rng.randint(-30, height - 30)
            x1 = x0 + rng.randint(70, 180)
            y1 = y0 + rng.randint(50, 130)
            haze_color = (
                rng.randint(12, 36),
                rng.randint(18, 48),
                rng.randint(40, 90),
                rng.randint(18, 46),
            )
            haze_draw.ellipse((x0, y0, x1, y1), fill=haze_color)
        haze = haze.filter(ImageFilter.GaussianBlur(radius=rng.uniform(10.0, 20.0)))
        background = Image.alpha_composite(background.convert("RGBA"), haze).convert("RGB")

    noise_rng = np.random.default_rng(rng.randint(0, 10_000_000))
    noise = noise_rng.normal(0, rng.uniform(3.0, 9.0), size=(height, width, 3))
    noisy = np.clip(np.asarray(background, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def add_sensor_effects(image: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.75:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.8)))

    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.92, 1.16))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.95, 1.22))
    image = ImageEnhance.Sharpness(image).enhance(rng.uniform(0.92, 1.18))

    scanline = Image.new("L", image.size, 0)
    scanline_draw = ImageDraw.Draw(scanline)
    scan_alpha = rng.randint(8, 16)
    for y in range(0, image.height, rng.choice([3, 4])):
        scanline_draw.line((0, y, image.width, y), fill=scan_alpha, width=1)
    image = ImageChops.subtract(image, Image.merge("RGB", (scanline, scanline, scanline)))

    vignette = Image.new("L", image.size, 255)
    vignette_draw = ImageDraw.Draw(vignette)
    border = rng.randint(14, 22)
    vignette_draw.rectangle(
        (border, border, image.width - border, image.height - border),
        fill=185,
    )
    vignette = vignette.filter(ImageFilter.GaussianBlur(radius=rng.uniform(18.0, 28.0)))
    image = Image.composite(image, Image.new("RGB", image.size, (0, 0, 0)), ImageChops.invert(vignette))

    final_noise = np.random.default_rng(rng.randint(0, 10_000_000)).normal(
        0,
        rng.uniform(1.0, 4.0),
        size=(image.height, image.width, 3),
    )
    array = np.clip(np.asarray(image, dtype=np.float32) + final_noise, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def paste_transformed_layer(
    background: Image.Image,
    layer: Image.Image,
    spec: FamilySpec,
    rng: random.Random,
) -> Image.Image:
    scale = rng.uniform(*spec.scale_range)
    angle = rng.uniform(*spec.angle_range)
    transformed = layer.resize(
        (
            max(20, int(layer.width * scale)),
            max(20, int(layer.height * scale)),
        ),
        Image.Resampling.BICUBIC,
    )
    transformed = transformed.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)

    center_x = CANVAS_SIZE[0] // 2 + rng.randint(-spec.x_jitter, spec.x_jitter)
    center_y = CANVAS_SIZE[1] // 2 + spec.preferred_y_offset + rng.randint(-spec.y_jitter, spec.y_jitter)
    offset_x = max(0, min(CANVAS_SIZE[0] - transformed.width, center_x - transformed.width // 2))
    offset_y = max(0, min(CANVAS_SIZE[1] - transformed.height, center_y - transformed.height // 2))

    return Image.alpha_composite(background.convert("RGBA"), Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))).copy().convert("RGBA")


def composite_layer(background: Image.Image, layer: Image.Image, spec: FamilySpec, rng: random.Random) -> Image.Image:
    scale = rng.uniform(*spec.scale_range)
    angle = rng.uniform(*spec.angle_range)
    transformed = layer.resize(
        (
            max(20, int(layer.width * scale)),
            max(20, int(layer.height * scale)),
        ),
        Image.Resampling.BICUBIC,
    )
    transformed = transformed.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)

    center_x = CANVAS_SIZE[0] // 2 + rng.randint(-spec.x_jitter, spec.x_jitter)
    center_y = CANVAS_SIZE[1] // 2 + spec.preferred_y_offset + rng.randint(-spec.y_jitter, spec.y_jitter)
    offset_x = max(0, min(CANVAS_SIZE[0] - transformed.width, center_x - transformed.width // 2))
    offset_y = max(0, min(CANVAS_SIZE[1] - transformed.height, center_y - transformed.height // 2))

    object_canvas = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
    object_canvas.alpha_composite(transformed, (offset_x, offset_y))
    return Image.alpha_composite(background.convert("RGBA"), object_canvas).convert("RGB")


def build_satellite_family(rng: random.Random, family_index: int) -> FamilySpec:
    return FamilySpec(
        class_name="satellite",
        group_id=f"synthetic::satellite::family_{family_index:03d}",
        family_index=family_index,
        body_width=rng.randint(36, 52),
        body_height=rng.randint(40, 58),
        panel_width=rng.randint(22, 32),
        panel_height=rng.randint(60, 88),
        panel_gap=rng.randint(8, 14),
        body_color=(rng.randint(170, 220), rng.randint(172, 220), rng.randint(185, 235), 255),
        accent_color=(rng.randint(20, 55), rng.randint(90, 145), rng.randint(170, 230), 245),
        angle_range=(-55.0, 55.0),
        scale_range=(0.84, 1.08),
        x_jitter=34,
        y_jitter=30,
        preferred_y_offset=0,
    )


def build_rocket_family(rng: random.Random, family_index: int) -> FamilySpec:
    return FamilySpec(
        class_name="rocket",
        group_id=f"synthetic::rocket::family_{family_index:03d}",
        family_index=family_index,
        body_width=rng.randint(22, 28),
        body_height=rng.randint(108, 136),
        cone_height=rng.randint(18, 26),
        fin_width=rng.randint(12, 18),
        fin_height=rng.randint(18, 28),
        body_color=(rng.randint(228, 246), rng.randint(228, 246), rng.randint(232, 250), 255),
        accent_color=(rng.randint(150, 220), rng.randint(45, 92), rng.randint(35, 80), 255),
        angle_range=(-34.0, 34.0),
        scale_range=(0.88, 1.08),
        x_jitter=26,
        y_jitter=20,
        preferred_y_offset=10,
    )


def build_space_shuttle_family(rng: random.Random, family_index: int) -> FamilySpec:
    return FamilySpec(
        class_name="space_shuttle",
        group_id=f"synthetic::space_shuttle::family_{family_index:03d}",
        family_index=family_index,
        body_width=rng.randint(20, 28),
        body_height=rng.randint(104, 122),
        wing_span=rng.randint(58, 82),
        wing_depth=rng.randint(26, 38),
        tail_height=rng.randint(26, 36),
        nose_length=rng.randint(16, 24),
        body_color=(235, 238, 243, 255),
        accent_color=(70, 78, 98, 255),
        angle_range=(-18.0, 18.0),
        scale_range=(0.86, 1.08),
        x_jitter=24,
        y_jitter=18,
        preferred_y_offset=0,
    )


def draw_satellite(spec: FamilySpec, rng: random.Random) -> Image.Image:
    layer = Image.new("RGBA", (220, 220), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    cx = 110
    cy = 110
    body_box = (
        cx - spec.body_width // 2,
        cy - spec.body_height // 2,
        cx + spec.body_width // 2,
        cy + spec.body_height // 2,
    )
    draw.rounded_rectangle(
        body_box,
        radius=8,
        fill=spec.body_color,
        outline=(238, 242, 250, 255),
        width=2,
    )

    left_panel = (
        body_box[0] - spec.panel_gap - spec.panel_width,
        cy - spec.panel_height // 2,
        body_box[0] - spec.panel_gap,
        cy + spec.panel_height // 2,
    )
    right_panel = (
        body_box[2] + spec.panel_gap,
        cy - spec.panel_height // 2,
        body_box[2] + spec.panel_gap + spec.panel_width,
        cy + spec.panel_height // 2,
    )
    draw.rectangle(left_panel, fill=spec.accent_color, outline=(220, 235, 255, 255), width=2)
    draw.rectangle(right_panel, fill=spec.accent_color, outline=(220, 235, 255, 255), width=2)

    for panel in (left_panel, right_panel):
        for x in range(panel[0] + 3, panel[2], 6):
            draw.line((x, panel[1] + 2, x, panel[3] - 2), fill=(160, 210, 255, 120), width=1)
        for y in range(panel[1] + 4, panel[3], 10):
            draw.line((panel[0] + 2, y, panel[2] - 2, y), fill=(140, 190, 255, 90), width=1)

    antenna_len = rng.randint(18, 30)
    draw.line((cx, body_box[1], cx, body_box[1] - antenna_len), fill=(235, 235, 245, 255), width=2)
    draw.arc(
        (
            cx - 10,
            body_box[1] - antenna_len - 10,
            cx + 10,
            body_box[1] - antenna_len + 10,
        ),
        start=200,
        end=340,
        fill=(215, 225, 245, 255),
        width=2,
    )
    if rng.random() < 0.55:
        draw.line((body_box[2], cy, body_box[2] + 18, cy + 10), fill=(225, 225, 235, 255), width=2)

    thruster = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    thruster_draw = ImageDraw.Draw(thruster)
    thruster_draw.ellipse((cx - 7, body_box[3] - 2, cx + 7, body_box[3] + 18), fill=(255, 165, 60, 70))
    thruster = thruster.filter(ImageFilter.GaussianBlur(radius=6))
    return Image.alpha_composite(layer, thruster)


def draw_rocket(spec: FamilySpec, rng: random.Random) -> Image.Image:
    layer = Image.new("RGBA", (220, 220), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    cx = 110
    nose_y = 28
    body_top = nose_y + spec.cone_height
    body_bottom = body_top + spec.body_height
    body_box = (
        cx - spec.body_width // 2,
        body_top,
        cx + spec.body_width // 2,
        body_bottom,
    )

    cone = [
        (cx, nose_y),
        (cx - spec.body_width // 2, body_top),
        (cx + spec.body_width // 2, body_top),
    ]
    draw.polygon(cone, fill=(245, 245, 248, 255), outline=(228, 228, 236, 255))
    draw.rounded_rectangle(
        body_box,
        radius=spec.body_width // 3,
        fill=spec.body_color,
        outline=(228, 228, 236, 255),
        width=2,
    )

    band_y1 = body_top + rng.randint(16, 22)
    band_y2 = body_top + rng.randint(48, 60)
    draw.rectangle((body_box[0], band_y1, body_box[2], band_y1 + 7), fill=spec.accent_color)
    draw.rectangle((body_box[0], band_y2, body_box[2], band_y2 + 7), fill=spec.accent_color)

    left_fin = [
        (body_box[0], body_bottom - 12),
        (body_box[0] - spec.fin_width, body_bottom + spec.fin_height),
        (body_box[0] + 2, body_bottom + spec.fin_height // 2),
    ]
    right_fin = [
        (body_box[2], body_bottom - 12),
        (body_box[2] + spec.fin_width, body_bottom + spec.fin_height),
        (body_box[2] - 2, body_bottom + spec.fin_height // 2),
    ]
    draw.polygon(left_fin, fill=(178, 184, 196, 255))
    draw.polygon(right_fin, fill=(178, 184, 196, 255))

    flame = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    flame_draw = ImageDraw.Draw(flame)
    flame_tip = body_bottom + rng.randint(34, 48)
    flame_draw.polygon(
        [
            (cx, flame_tip),
            (cx - rng.randint(8, 16), body_bottom),
            (cx + rng.randint(8, 16), body_bottom),
        ],
        fill=(255, rng.randint(150, 220), rng.randint(40, 90), 225),
    )
    flame_draw.ellipse((cx - 20, body_bottom - 2, cx + 20, body_bottom + 46), fill=(255, 120, 20, 70))
    flame = flame.filter(ImageFilter.GaussianBlur(radius=10))
    return Image.alpha_composite(layer, flame)


def draw_space_shuttle(spec: FamilySpec, rng: random.Random) -> Image.Image:
    layer = Image.new("RGBA", (220, 220), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    cx = 110
    cy = 110
    fuselage = (
        cx - spec.body_height // 2,
        cy - spec.body_width // 2,
        cx + spec.body_height // 2,
        cy + spec.body_width // 2,
    )
    draw.rounded_rectangle(
        fuselage,
        radius=spec.body_width // 2,
        fill=spec.body_color,
        outline=(205, 210, 220, 255),
        width=2,
    )

    nose = [
        (fuselage[2] + spec.nose_length, cy),
        (fuselage[2] - 6, fuselage[1]),
        (fuselage[2] - 6, fuselage[3]),
    ]
    draw.polygon(nose, fill=(240, 242, 246, 255), outline=(205, 210, 220, 255))

    left_wing = [
        (cx - 6, cy - 3),
        (cx - spec.wing_span, cy - spec.wing_depth),
        (cx + 10, cy - 14),
    ]
    right_wing = [
        (cx - 6, cy + 3),
        (cx - spec.wing_span, cy + spec.wing_depth),
        (cx + 10, cy + 14),
    ]
    draw.polygon(left_wing, fill=(218, 221, 228, 255), outline=(188, 192, 200, 255))
    draw.polygon(right_wing, fill=(218, 221, 228, 255), outline=(188, 192, 200, 255))

    tail = [
        (fuselage[0] + 18, cy),
        (fuselage[0] - 10, cy - spec.tail_height),
        (fuselage[0] + 6, cy - 10),
        (fuselage[0] + 6, cy + 10),
        (fuselage[0] - 10, cy + spec.tail_height),
    ]
    draw.polygon(tail, fill=spec.accent_color)

    draw.line((fuselage[0] + 12, cy, fuselage[2] - 18, cy), fill=(95, 100, 110, 180), width=2)
    draw.rectangle((fuselage[2] - 24, fuselage[1] + 4, fuselage[2] - 8, fuselage[1] + 12), fill=(48, 78, 118, 220))
    draw.line((fuselage[2] - 8, fuselage[1], fuselage[2] + spec.nose_length, cy), fill=(72, 78, 88, 255), width=2)

    engine_glow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(engine_glow)
    glow_draw.ellipse((fuselage[0] - 20, cy - 13, fuselage[0] + 8, cy + 13), fill=(255, 150, 40, 60))
    engine_glow = engine_glow.filter(ImageFilter.GaussianBlur(radius=8))
    return Image.alpha_composite(layer, engine_glow)


def build_family_spec(class_name: str, family_rng: random.Random, family_index: int) -> FamilySpec:
    if class_name == "satellite":
        return build_satellite_family(family_rng, family_index)
    if class_name == "rocket":
        return build_rocket_family(family_rng, family_index)
    return build_space_shuttle_family(family_rng, family_index)


def draw_family_image(spec: FamilySpec, rng: random.Random) -> Image.Image:
    if spec.class_name == "satellite":
        layer = draw_satellite(spec, rng)
    elif spec.class_name == "rocket":
        layer = draw_rocket(spec, rng)
    else:
        layer = draw_space_shuttle(spec, rng)

    background = random_space_background(rng)
    composite = composite_layer(background, layer, spec, rng)
    composite = add_sensor_effects(composite, rng)
    return composite.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)


def save_class_images(class_name: str, class_offset: int) -> List[Dict[str, str | int]]:
    class_dir = reset_class_dir(class_name)
    records: List[Dict[str, str | int]] = []

    for family_index in range(1, FAMILIES_PER_CLASS + 1):
        family_rng = random.Random(SEED + class_offset + family_index * 97)
        spec = build_family_spec(class_name, family_rng, family_index)

        for variant_index in range(1, VARIANTS_PER_FAMILY + 1):
            variant_rng = random.Random(SEED + class_offset + family_index * 1_000 + variant_index * 131)
            image = draw_family_image(spec, variant_rng)
            file_name = f"{class_name}_family_{family_index:03d}_var_{variant_index:02d}.png"
            output_path = class_dir / file_name
            image.save(output_path, format="PNG", optimize=True)
            records.append(
                {
                    "class_name": class_name,
                    "group_id": spec.group_id,
                    "family_index": family_index,
                    "variant_index": variant_index,
                    "file_name": file_name,
                    "path": str(output_path.resolve()),
                }
            )

    print(f"  - {class_name}: {IMAGES_PER_CLASS} imagens sinteticas geradas em {class_dir}")
    return records


def write_manifest(records: List[Dict[str, str | int]]) -> None:
    fieldnames = ["class_name", "group_id", "family_index", "variant_index", "file_name", "path"]
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    ensure_dirs()
    all_records: List[Dict[str, str | int]] = []
    print("[INICIO] Geracao de imagens sinteticas OrbitShield Vision")

    for offset, class_name in enumerate(("satellite", "rocket", "space_shuttle")):
        all_records.extend(save_class_images(class_name, offset * 100_000))

    write_manifest(all_records)
    print(f"Manifest salvo em: {MANIFEST_PATH}")
    print("[FIM] Geracao concluida")
    return 0


if __name__ == "__main__":
    sys.exit(main())
