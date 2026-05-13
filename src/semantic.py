"""Word-embedding axis projection for tone analysis."""
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src import config


_GLOVE_CACHE: Optional[Dict[str, np.ndarray]] = None


def load_glove(path: Path = config.GLOVE_PATH, expected_dim: int = 300) -> Dict[str, np.ndarray]:
    global _GLOVE_CACHE
    if _GLOVE_CACHE is not None:
        return _GLOVE_CACHE

    if not path.exists():
        raise FileNotFoundError(f"GloVe vectors not found at {path}")

    vectors: Dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            if len(vec) != expected_dim:
                continue
            vectors[word] = vec

    _GLOVE_CACHE = vectors
    return vectors


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


def document_vector(text, glove, min_word_length=3):
    tokens = tokenize(text)
    vecs = [glove[t] for t in tokens if len(t) >= min_word_length and t in glove]
    if not vecs:
        return None
    return np.mean(np.stack(vecs), axis=0)


def axis_vector(negative_pole, positive_pole, glove):
    def _centroid(words):
        present = [glove[w] for w in words if w in glove]
        if not present:
            raise ValueError(f"None of {words} in vocab.")
        return np.mean(np.stack(present), axis=0)

    axis = _centroid(positive_pole) - _centroid(negative_pole)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("Zero-norm axis.")
    return axis / norm


def project_onto_axis(doc_vec, axis_vec):
    norm = np.linalg.norm(doc_vec)
    if norm == 0:
        return 0.0
    return float(np.dot(doc_vec / norm, axis_vec))


def analyze_tone(documents, glove, axes=None):
    axes = axes or config.SEMANTIC_AXES
    axis_vectors = {
        name: axis_vector(spec["negative_pole"], spec["positive_pole"], glove)
        for name, spec in axes.items()
    }
    doc_vectors = {label: document_vector(text, glove) for label, text in documents.items()}
    results = {axis: {} for axis in axes}
    for axis_name, axis_vec in axis_vectors.items():
        for doc_label, doc_vec in doc_vectors.items():
            results[axis_name][doc_label] = (
                project_onto_axis(doc_vec, axis_vec) if doc_vec is not None else 0.0
            )
    return results


def radar_chart(scores, title="Semantic Tone Profile"):
    import plotly.graph_objects as go

    axis_names = list(scores.keys())
    doc_labels: List[str] = []
    for ax_scores in scores.values():
        for label in ax_scores:
            if label not in doc_labels:
                doc_labels.append(label)

    fig = go.Figure()
    for label in doc_labels:
        values = [scores[ax].get(label, 0.0) for ax in axis_names]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=axis_names + [axis_names[0]],
            fill="toself",
            name=label,
            line=dict(width=2),
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-0.5, 0.5], tickformat=".2f")),
        title=title,
        showlegend=True,
        height=500,
    )
    return fig
