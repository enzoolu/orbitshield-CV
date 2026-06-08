from __future__ import annotations

import io
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import requests
from PIL import Image, ImageOps, UnidentifiedImageError


SEARCH_URL = "https://images-api.nasa.gov/search"
ASSET_URL = "https://images-api.nasa.gov/asset/{nasa_id}"
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "dataset" / "raw"
DOCS_DIR = ROOT_DIR / "docs"
MAX_IMAGES_PER_CLASS = 300
MIN_REFERENCE_IMAGES = 150
REQUEST_TIMEOUT = 30
USER_AGENT = "OrbitShield-ACV/1.0"


CLASS_CONFIG = {
    "satellite": {
        "queries": [
            "communications satellite",
            "spacecraft satellite",
            "satellite",
        ],
        "keywords": ["satellite", "spacecraft", "orbital", "communications"],
    },
    "rocket": {
        "queries": [
            "launch vehicle",
            "rocket launch",
            "rocket engine",
        ],
        "keywords": ["rocket", "launch vehicle", "booster", "engine", "launch"],
    },
    "space_shuttle": {
        "queries": [
            "space shuttle",
            "space shuttle orbiter",
            "orbiter spacecraft",
            "space shuttle in orbit",
        ],
        "keywords": [
            "space shuttle",
            "shuttle orbiter",
            "orbiter vehicle",
            "columbia",
            "challenger",
            "discovery",
            "atlantis",
            "endeavour",
            "enterprise",
        ],
    },
}


@dataclass
class ClassReport:
    class_name: str
    saved: int = 0
    duplicates: int = 0
    invalid_images: int = 0
    download_failures: int = 0
    metadata_filtered: int = 0
    asset_failures: int = 0
    search_items_seen: int = 0
    queries_attempted: List[str] = field(default_factory=list)


def archive_obsolete_class_dirs() -> List[str]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    active_classes = set(CLASS_CONFIG)
    obsolete_dirs = [path for path in RAW_DIR.iterdir() if path.is_dir() and path.name not in active_classes]
    if not obsolete_dirs:
        return []

    backup_root = ROOT_DIR / "dataset" / "backup" / f"replaced_classes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_root.mkdir(parents=True, exist_ok=True)
    archived: List[str] = []

    for obsolete_dir in sorted(obsolete_dirs):
        destination = backup_root / obsolete_dir.name
        counter = 1
        while destination.exists():
            destination = backup_root / f"{obsolete_dir.name}_{counter}"
            counter += 1
        shutil.move(str(obsolete_dir), str(destination))
        archived.append(f"{obsolete_dir.name} -> {destination}")
        print(f"[SYNC] Classe obsoleta movida para backup: {obsolete_dir.name}")

    return archived


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "nasa_image"


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def search_items(session: requests.Session, query: str, max_pages: int = 12) -> Iterable[dict]:
    for page in range(1, max_pages + 1):
        response = session.get(
            SEARCH_URL,
            params={"q": query, "media_type": "image", "page": page},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        items = payload.get("collection", {}).get("items", [])
        if not items:
            break
        for item in items:
            yield item


def is_relevant(item: dict, keywords: List[str]) -> bool:
    data = item.get("data", [{}])[0]
    title = data.get("title", "")
    description = data.get("description", "")
    item_keywords = " ".join(data.get("keywords", []) or [])
    metadata = f"{title} {description} {item_keywords}".lower()
    return any(keyword.lower() in metadata for keyword in keywords)


def fetch_asset_urls(session: requests.Session, nasa_id: str) -> List[str]:
    response = session.get(ASSET_URL.format(nasa_id=nasa_id), timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    items = payload.get("collection", {}).get("items", [])
    urls = []
    for item in items:
        href = item.get("href")
        if not href:
            continue
        lowered = href.lower()
        if lowered.endswith((".jpg", ".jpeg", ".png")):
            urls.append(href)
    return sorted(
        urls,
        key=lambda url: (
            "~large" not in url.lower(),
            "~medium" not in url.lower(),
            "~orig" not in url.lower() and "orig" not in url.lower(),
            "~small" in url.lower() or "~thumb" in url.lower(),
        ),
    )


def validate_and_convert_image(content: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(content)) as image:
            image = ImageOps.exif_transpose(image)
            rgb_image = image.convert("RGB")
            rgb_image.load()
            return rgb_image
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("invalid image") from exc


def save_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=95, optimize=True)


def download_one_image(session: requests.Session, url: str) -> bytes:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.content


def collect_for_class(session: requests.Session, class_name: str, config: Dict[str, List[str]]) -> ClassReport:
    report = ClassReport(class_name=class_name)
    report.queries_attempted = list(config["queries"])
    output_dir = RAW_DIR / class_name
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_nasa_ids = {path.stem for path in output_dir.glob("*.jpg")}
    report.saved = len(seen_nasa_ids)

    print(f"\n[COLETA] Classe: {class_name}")
    if report.saved >= MAX_IMAGES_PER_CLASS:
        print(f"  - limite ja atingido com {report.saved} imagens validas. Pulando.")
        return report

    for query in config["queries"]:
        if report.saved >= MAX_IMAGES_PER_CLASS:
            break

        print(f"  - Buscando por: {query!r}")

        try:
            iterator = search_items(session, query)
            for item in iterator:
                if report.saved >= MAX_IMAGES_PER_CLASS:
                    break

                report.search_items_seen += 1
                data = item.get("data", [{}])[0]
                nasa_id = data.get("nasa_id")
                if not nasa_id:
                    continue
                nasa_slug = slugify(nasa_id)
                if nasa_slug in seen_nasa_ids:
                    report.duplicates += 1
                    continue
                if not is_relevant(item, config["keywords"]):
                    report.metadata_filtered += 1
                    continue

                try:
                    asset_urls = fetch_asset_urls(session, nasa_id)
                except requests.RequestException:
                    report.asset_failures += 1
                    continue
                if not asset_urls:
                    report.asset_failures += 1
                    continue

                saved = False
                for asset_url in asset_urls:
                    try:
                        content = download_one_image(session, asset_url)
                        image = validate_and_convert_image(content)
                    except requests.RequestException:
                        report.download_failures += 1
                        continue
                    except ValueError:
                        report.invalid_images += 1
                        continue

                    file_name = f"{nasa_slug}.jpg"
                    save_image(image, output_dir / file_name)
                    seen_nasa_ids.add(nasa_slug)
                    report.saved += 1
                    saved = True
                    print(f"    salvou {file_name} ({report.saved}/{MAX_IMAGES_PER_CLASS})")
                    break

                if not saved:
                    report.download_failures += 1

                time.sleep(0.02)
        except requests.RequestException as exc:
            print(f"    falha na consulta '{query}': {exc}")

    print(
        "  resumo: "
        f"salvas={report.saved}, duplicadas={report.duplicates}, "
        f"invalidas={report.invalid_images}, falhas_download={report.download_failures}, "
        f"falhas_asset={report.asset_failures}, filtradas={report.metadata_filtered}"
    )
    return report


def write_markdown_report(reports: List[ClassReport], archived_dirs: List[str]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DOCS_DIR / "dataset_collection_report.md"

    total_saved = sum(item.saved for item in reports)
    weak_classes = [item.class_name for item in reports if item.saved < MIN_REFERENCE_IMAGES]

    lines = [
        "# Relatorio de Coleta do Dataset",
        "",
        "## Resumo",
        "",
        f"- Total de imagens validas baixadas: {total_saved}",
        f"- Limite configurado por classe: {MAX_IMAGES_PER_CLASS}",
        f"- Referencia minima por classe: {MIN_REFERENCE_IMAGES}",
        "- Esta rodada substituiu a classe `space_station` por `space_shuttle` para reduzir ambiguidade visual no dataset real-only.",
        "",
        "## Quantidade por Classe",
        "",
        "| Classe | Imagens validas | Duplicatas | Falhas de download | Imagens invalidas | Falhas no endpoint asset | Filtradas por metadata | Itens vistos |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for item in reports:
        lines.append(
            f"| {item.class_name} | {item.saved} | {item.duplicates} | "
            f"{item.download_failures} | {item.invalid_images} | {item.asset_failures} | "
            f"{item.metadata_filtered} | {item.search_items_seen} |"
        )

    lines.extend(
        [
            "",
            "## Consultas Utilizadas",
            "",
        ]
    )
    for item in reports:
        lines.append(f"- `{item.class_name}`: {', '.join(item.queries_attempted)}")

    if archived_dirs:
        lines.extend(
            [
                "",
                "## Classes Arquivadas",
                "",
            ]
        )
        for archived_dir in archived_dirs:
            lines.append(f"- `{archived_dir}`")

    lines.extend(
        [
            "",
            "## Limitacoes da Busca",
            "",
            "- A NASA Image and Video Library possui metadados heterogeneos; parte dos resultados usa descricoes amplas ou incompletas.",
            "- Algumas buscas retornam ativos historicos, ilustracoes ou enquadramentos pouco uteis para classificacao direta.",
            "- O endpoint `asset` nem sempre oferece um arquivo de imagem valido mesmo quando o resultado aparece no endpoint de busca.",
            "",
            "## Recomendacao Futura",
            "",
            "- Manter a estrategia real-only enquanto a troca para `space_shuttle` continuar elevando a separabilidade das classes.",
            "- Considerar dataset hibrido apenas se a nova composicao com `space_shuttle` continuar insuficiente para a meta de desempenho.",
        ]
    )

    if weak_classes:
        lines.extend(
            [
                "",
                "## Plano B Obrigatorio",
                "",
                f"- Classes abaixo de {MIN_REFERENCE_IMAGES} imagens validas: {', '.join(weak_classes)}.",
                "- Recomendacao: complementar a base com dataset hibrido em uma proxima iteracao, preservando a origem real ja coletada.",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[RELATORIO] salvo em {report_path}")


def main() -> int:
    print("[INICIO] Coleta de imagens NASA para OrbitShield Vision")
    session = make_session()
    reports: List[ClassReport] = []
    archived_dirs = archive_obsolete_class_dirs()

    for class_name, config in CLASS_CONFIG.items():
        reports.append(collect_for_class(session, class_name, config))

    write_markdown_report(reports, archived_dirs)
    print("[FIM] Coleta concluida")
    return 0


if __name__ == "__main__":
    sys.exit(main())
