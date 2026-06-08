from __future__ import annotations

import csv
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

from dataset_utils import ClassAudit, audit_dataset


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "dataset" / "raw"
BACKUP_ROOT = ROOT_DIR / "dataset" / "backup"
DOCS_DIR = ROOT_DIR / "docs"
OUTPUTS_DIR = ROOT_DIR / "outputs"


def ensure_dirs() -> None:
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def move_to_backup(source_path: Path, backup_root: Path) -> Path:
    relative_path = source_path.relative_to(RAW_DIR)
    destination = backup_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    counter = 1
    while destination.exists():
        destination = destination.with_name(f"{destination.stem}_{counter}{destination.suffix}")
        counter += 1
    shutil.move(str(source_path), str(destination))
    return destination


def select_duplicate_removals(audit: ClassAudit) -> Set[Path]:
    removals: Set[Path] = set()
    for group in audit.exact_duplicate_groups:
        keep = sorted(group)[0]
        for candidate in group:
            if candidate != keep:
                removals.add(candidate)
    return removals


def select_near_duplicate_removals(audit: ClassAudit) -> Set[Path]:
    removals: Set[Path] = set()
    for pair in audit.approximate_duplicate_pairs:
        if pair.hamming_distance == 0 and pair.color_distance <= 1.0:
            removals.add(max(pair.left, pair.right))
    return removals


def clean_dataset() -> tuple[List[Dict[str, str]], List[ClassAudit], List[ClassAudit], Path]:
    before_reports = audit_dataset(RAW_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = BACKUP_ROOT / f"cleaned_{timestamp}"
    actions: List[Dict[str, str]] = []

    for audit in before_reports:
        duplicate_removals = select_duplicate_removals(audit)
        near_duplicate_removals = select_near_duplicate_removals(audit)
        broken_paths = set(audit.broken_files)
        small_paths = {record.path for record in audit.small_images}
        low_variance_paths = {record.path for record in audit.low_variance_images}

        reason_order = [
            ("broken_image", broken_paths),
            ("exact_duplicate", duplicate_removals),
            ("near_duplicate_visual", near_duplicate_removals),
            ("too_small", small_paths),
            ("low_variance", low_variance_paths),
        ]

        already_moved: Set[Path] = set()
        for reason, candidates in reason_order:
            for candidate in sorted(candidates):
                if candidate in already_moved or not candidate.exists():
                    continue
                backup_path = move_to_backup(candidate, backup_root)
                already_moved.add(candidate)
                actions.append(
                    {
                        "class_name": audit.class_name,
                        "reason": reason,
                        "source_path": str(candidate),
                        "backup_path": str(backup_path),
                    }
                )

    after_reports = audit_dataset(RAW_DIR)
    return actions, before_reports, after_reports, backup_root


def write_outputs(
    actions: List[Dict[str, str]],
    before_reports: List[ClassAudit],
    after_reports: List[ClassAudit],
    backup_root: Path,
) -> None:
    actions_path = OUTPUTS_DIR / "clean_dataset_actions.csv"
    with actions_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=["class_name", "reason", "source_path", "backup_path"])
        writer.writeheader()
        writer.writerows(actions)

    before_lookup = {audit.class_name: audit for audit in before_reports}
    after_lookup = {audit.class_name: audit for audit in after_reports}
    reason_counter = Counter(action["reason"] for action in actions)
    per_class_reasons: Dict[str, Counter[str]] = defaultdict(Counter)
    for action in actions:
        per_class_reasons[action["class_name"]][action["reason"]] += 1

    lines = [
        "# Relatorio de Limpeza do Dataset",
        "",
        f"- Backup das imagens removidas: `{backup_root}`",
        f"- Registro detalhado das acoes: `outputs/{actions_path.name}`",
        "",
        "## Resumo Geral",
        "",
        f"- Total de arquivos movidos: {len(actions)}",
        f"- Removidos por quebra: {reason_counter['broken_image']}",
        f"- Removidos por duplicata exata: {reason_counter['exact_duplicate']}",
        f"- Removidos por duplicata visual quase exata: {reason_counter['near_duplicate_visual']}",
        f"- Removidos por tamanho insuficiente: {reason_counter['too_small']}",
        f"- Removidos por baixa variancia: {reason_counter['low_variance']}",
        "",
        "## Resultado por Classe",
        "",
        "| Classe | Antes | Depois | Removidos | Duplicatas exatas removidas | Duplicatas visuais removidas | Pequenas removidas | Baixa variancia removida |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for class_name in sorted(before_lookup):
        before = before_lookup[class_name]
        after = after_lookup[class_name]
        lines.append(
            f"| {class_name} | {before.total_files} | {after.total_files} | {before.total_files - after.total_files} | "
            f"{per_class_reasons[class_name]['exact_duplicate']} | "
            f"{per_class_reasons[class_name]['near_duplicate_visual']} | "
            f"{per_class_reasons[class_name]['too_small']} | "
            f"{per_class_reasons[class_name]['low_variance']} |"
        )

    lines.extend(
        [
            "",
            "## Observacoes",
            "",
            "- A limpeza foi nao destrutiva: arquivos removidos foram movidos para backup.",
            "- Duplicatas aproximadas foram removidas apenas quando a similaridade visual era praticamente total (hamming 0 e distancia de cor <= 1.0).",
            "- Esta rodada manteve o dataset real-only e aplicou apenas filtros objetivos de qualidade estrutural.",
        ]
    )

    (DOCS_DIR / "dataset_cleaning_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()
    if not RAW_DIR.exists():
        print("Dataset raw nao encontrado.")
        return 1

    print("[INICIO] Limpeza do dataset raw")
    actions, before_reports, after_reports, backup_root = clean_dataset()
    write_outputs(actions, before_reports, after_reports, backup_root)

    for audit in after_reports:
        print(f"  - {audit.class_name}: restantes={audit.total_files}")

    print(f"[FIM] Limpeza concluida. Backup em {backup_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
