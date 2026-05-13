"""Saved article files with simple header metadata."""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from src import config


SAMPLE_DIR = config.SAMPLE_DIR
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Article:
    filename: str
    body: str
    outlet: str = "Unknown"
    lean: str = "unknown"
    date: str = ""
    url: str = ""
    title: str = ""
    metadata_extra: Dict[str, str] = field(default_factory=dict)

    @property
    def display_label(self):
        return f"{self.outlet} ({self.lean})"

    @property
    def word_count(self):
        return len(self.body.split())


def parse_article_file(path: Path) -> Article:
    raw = path.read_text(encoding="utf-8")
    if "---" in raw:
        header, body = raw.split("---", 1)
    else:
        header, body = "", raw

    metadata: Dict[str, str] = {}
    for line in header.splitlines():
        line = line.strip()
        if line.startswith("#") and ":" in line:
            key, _, value = line.lstrip("#").strip().partition(":")
            metadata[key.strip().lower()] = value.strip()

    return Article(
        filename=path.name,
        body=body.strip(),
        outlet=metadata.get("outlet", "Unknown"),
        lean=metadata.get("lean", "unknown"),
        date=metadata.get("date", ""),
        url=metadata.get("url", ""),
        title=metadata.get("title", ""),
        metadata_extra={k: v for k, v in metadata.items()
                        if k not in {"outlet", "lean", "date", "url", "title"}},
    )


def list_articles(directory: Path = SAMPLE_DIR) -> List[Article]:
    if not directory.exists():
        return []
    txt_files = sorted(directory.glob("*.txt"))
    articles = []
    for f in txt_files:
        if f.name.startswith("SAMPLE_TEMPLATE"):
            continue
        try:
            articles.append(parse_article_file(f))
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
    return articles


def save_article(
    body: str,
    outlet: str,
    lean: str,
    title: str = "",
    url: str = "",
    date: str = "",
    directory: Path = SAMPLE_DIR,
) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", f"{lean}_{outlet}_{title}".lower()).strip("_")
    if len(slug) > 60:
        slug = slug[:60]
    if not slug:
        slug = f"{lean}_{outlet}".lower()
    path = directory / f"{slug}.txt"
    header_lines = [
        f"# outlet: {outlet}",
        f"# lean: {lean}",
        f"# title: {title}",
        f"# url: {url}",
        f"# date: {date}",
        "---",
    ]
    path.write_text("\n".join(header_lines) + "\n" + body.strip() + "\n", encoding="utf-8")
    return path


LEAN_COLORS = {
    "left": "#3B82F6",
    "center": "#6B7280",
    "right": "#EF4444",
    "unknown": "#9CA3AF",
}
