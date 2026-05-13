"""Emotionality score: cos(doc, emotion_centroid) - cos(doc, reason_centroid)."""
from typing import Dict, List, Optional

import numpy as np

from src.semantic import document_vector


EMOTION_ANCHORS: List[str] = [
    "anger", "fear", "joy", "sadness", "love", "hate", "guilt", "shame",
    "pride", "anxiety", "disgust", "passion", "grief", "rage", "pity",
    "horror", "delight", "sorrow", "desire", "panic", "terror", "envy",
    "outrage", "compassion", "agony", "ecstasy", "hope", "despair",
    "happiness", "misery", "frustration", "excitement", "regret",
]

REASON_ANCHORS: List[str] = [
    "analysis", "evidence", "reason", "logic", "argument", "rational",
    "statistic", "method", "fact", "research", "data", "calculation",
    "hypothesis", "theory", "principle", "concept", "knowledge", "study",
    "examine", "evaluate", "investigate", "assess", "measure", "determine",
    "conclude", "deduce", "infer", "observe", "verify", "prove",
    "doctrine", "framework",
]


def _centroid(words, glove):
    present = [glove[w] for w in words if w in glove]
    if not present:
        raise ValueError("No anchor words found in GloVe vocab.")
    return np.mean(np.stack(present), axis=0)


def _cosine(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def emotionality_score(
    text: str,
    glove: Dict[str, np.ndarray],
    emotion_anchors: Optional[List[str]] = None,
    reason_anchors: Optional[List[str]] = None,
) -> Dict[str, float]:
    e_anchors = emotion_anchors or EMOTION_ANCHORS
    r_anchors = reason_anchors or REASON_ANCHORS
    A_emotion = _centroid(e_anchors, glove)
    A_reason = _centroid(r_anchors, glove)
    v_doc = document_vector(text, glove)
    if v_doc is None:
        return {"emotionality": 0.0, "emotion_similarity": 0.0, "reason_similarity": 0.0}
    sim_e = _cosine(v_doc, A_emotion)
    sim_r = _cosine(v_doc, A_reason)
    return {
        "emotionality": round(sim_e - sim_r, 4),
        "emotion_similarity": round(sim_e, 4),
        "reason_similarity": round(sim_r, 4),
    }
