"""Text helpers: cleaning, sentence splitting, slugs, file hashes."""
import re
import hashlib
from pathlib import Path
from typing import List


def clean_text(text: str) -> str:
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _regex_sentence_split(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def split_sentences(text: str) -> List[str]:
    try:
        from nltk.tokenize import sent_tokenize
        try:
            return sent_tokenize(text)
        except LookupError:
            try:
                import nltk
                nltk.download("punkt_tab", quiet=True)
                nltk.download("punkt", quiet=True)
                return sent_tokenize(text)
            except Exception:
                return _regex_sentence_split(text)
    except ImportError:
        return _regex_sentence_split(text)


def slugify_case_name(filename: str) -> str:
    name = Path(filename).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    if len(slug) < 3:
        slug = f"case_{slug}"
    return slug[:60]


def file_hash(path: Path) -> str:
    sha1 = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha1.update(chunk)
    return sha1.hexdigest()[:12]
