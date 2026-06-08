from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List

import pandas as pd
from PIL import Image, ImageOps

from dataset_utils import ClassAudit, audit_dataset


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "dataset" / "raw"
OUTPUTS_DIR = ROOT_DIR / "outputs"
DOCS_DIR = ROOT_DIR / "docs"
CONTACT_SHEET_THUMB = (100, 100)
CONTACT_SHEET_COLUMNS = 6


def ensure_dirs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def clear_old_contact_sheets() -> None:
    for contact_sheet in OUTPUTS_DIR.glob("contact_sheet_*.png"):
        contact_sheet.unlink(missing_ok=True)


def create_contact_sheet(audit: ClassAudit, output_path: Path) -> None:
    records = audit.valid_records
    if not records:
        image = Image.new("RGB", CONTACT_SHEET_THUMB, color=(245, 245, 245))
        image.save(output_path)
        return

    rows = math.ceil(len(records) / CONTACT_SHEET_COLUMNS)
    sheet = Image.new(
        "RGB",
        (CONTACT_SHEET_COLUMNS * CONTACT_SHEET_THUMB[0], rows * CONTACT_SHEET_THUMB[1]),
        color=(18, 18, 18),
    )

    for index, record in enumerate(records):
        row = index // CONTACT_SHEET_COLUMNS
        col = index % CONTACT_SHEET_COLUMNS
        with Image.open(record.path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            image.thumbnail(CONTACT_SHEET_THUMB, Image.Resampling.LANCZOS)
            tile = Image.new("RGB", CONTACT_SHEET_THUMB, color=(24, 24, 24))
            offset_x = (CONTACT_SHEET_THUMB[0] - image.width) // 2
            offset_y = (CONTACT_SHEET_THUMB[1] - image.height) // 2
            tile.paste(image, (offset_x, offset_y))
        sheet.paste(tile, (col * CONTACT_SHEET_THUMB[0], row * CONTACT_SHEET_THUMB[1]))

    sheet.save(output_path)


def write_markdown_report(reports: List[ClassAudit]) -> None:
    report_path = DOCS_DIR / "dataset_audit_report.md"
    lines = [
        "# Relatorio de Auditoria do Dataset",
        "",
        "## Resumo por Classe",
        "",
        "| Classe | Arquivos encontrados | Imagens validas | Quebradas | Grupos de duplicata exata | Pares de duplicata aproximada | Muito pequenas | Baixa variancia |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    rows = []
    for audit in reports:
        exact_duplicate_files = sum(max(0, len(group) - 1) for group in audit.exact_duplicate_groups)
        lines.append(
            f"| {audit.class_name} | {audit.total_files} | {len(audit.valid_records)} | "
            f"{len(audit.broken_files)} | {len(audit.exact_duplicate_groups)} ({exact_duplicate_files} excedentes) | "
            f"{len(audit.approximate_duplicate_pairs)} | {len(audit.small_images)} | {len(audit.low_variance_images)} |"
        )
        rows.append(
            {
                "class_name": audit.class_name,
                "files_found": audit.total_files,
                "valid_images": len(audit.valid_records),
                "broken_images": len(audit.broken_files),
                "exact_duplicate_groups": len(audit.exact_duplicate_groups),
                "approx_duplicate_pairs": len(audit.approximate_duplicate_pairs),
                "small_images": len(audit.small_images),
                "low_variance_images": len(audit.low_variance_images),
            }
        )

    lines.extend(
        [
            "",
            "## Regras Aplicadas",
            "",
            "- Duplicata exata: hash SHA-256 identico.",
            "- Duplicata aproximada: perceptual hash (dHash) com distancia de Hamming <= 4 e media de cor semelhante.",
            "- Muito pequena: menor lado < 128 px.",
            "- Baixa variancia: desvio padrao do canal em escala de cinza < 8.0.",
            "",
            "## Observacoes por Classe",
            "",
        ]
    )

    for audit in reports:
        lines.append(f"### {audit.class_name}")
        lines.append("")
        lines.append(
            f"- Contact sheet: `outputs/contact_sheet_{audit.class_name}.png`"
        )
        if audit.broken_files:
            broken_names = ", ".join(path.name for path in audit.broken_files[:10])
            lines.append(f"- Quebradas: {len(audit.broken_files)}. Exemplos: {broken_names}")
        else:
            lines.append("- Quebradas: 0")

        if audit.exact_duplicate_groups:
            example_group = ", ".join(path.name for path in audit.exact_duplicate_groups[0][:4])
            lines.append(
                f"- Duplicatas exatas: {len(audit.exact_duplicate_groups)} grupos. Exemplo: {example_group}"
            )
        else:
            lines.append("- Duplicatas exatas: 0 grupos")

        if audit.approximate_duplicate_pairs:
            pair = audit.approximate_duplicate_pairs[0]
            lines.append(
                f"- Duplicatas aproximadas: {len(audit.approximate_duplicate_pairs)} pares. "
                f"Exemplo: `{pair.left.name}` vs `{pair.right.name}` (hamming={pair.hamming_distance}, cor={pair.color_distance:.2f})"
            )
        else:
            lines.append("- Duplicatas aproximadas: 0 pares")

        lines.append(
            f"- Muito pequenas: {len(audit.small_images)} | Baixa variancia: {len(audit.low_variance_images)}"
        )
        lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUTPUTS_DIR / "dataset_audit_summary.csv", index=False)


def main() -> int:
    ensure_dirs()
    if not RAW_DIR.exists():
        print("Dataset raw nao encontrado.")
        return 1

    print("[INICIO] Auditoria do dataset raw")
    clear_old_contact_sheets()
    reports = audit_dataset(RAW_DIR)
    for audit in reports:
        contact_sheet_path = OUTPUTS_DIR / f"contact_sheet_{audit.class_name}.png"
        create_contact_sheet(audit, contact_sheet_path)
        print(
            f"  - {audit.class_name}: validas={len(audit.valid_records)} | quebradas={len(audit.broken_files)} | "
            f"dup_exatas={len(audit.exact_duplicate_groups)} | dup_aprox={len(audit.approximate_duplicate_pairs)} | "
            f"pequenas={len(audit.small_images)} | baixa_variancia={len(audit.low_variance_images)}"
        )

    write_markdown_report(reports)
    print(f"[FIM] Relatorio salvo em {DOCS_DIR / 'dataset_audit_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
