"""Check whether a pasted news article matches the loaded court opinion."""
import re
from typing import Dict, List

import numpy as np
from langchain_chroma import Chroma

from src import config
from src.rag import get_embeddings


RELEVANCE_THRESHOLD = config.RELEVANCE_THRESHOLD


def extract_likely_case_terms(opinion_text: str, top_n: int = 8) -> List[str]:
    candidates = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}\b", opinion_text)
    counts: Dict[str, int] = {}
    blocklist = {
        "United States", "Supreme Court", "The Court", "Congress",
        "Constitution", "Court", "United", "States", "Justice",
    }
    for c in candidates:
        if 4 <= len(c) <= 60 and c not in blocklist:
            counts[c] = counts.get(c, 0) + 1
    sorted_terms = sorted(counts.items(), key=lambda x: -x[1])
    return [t for t, count in sorted_terms[:top_n] if count >= 2]


def compute_centroid(vectorstore: Chroma) -> np.ndarray:
    collection = vectorstore._collection
    result = collection.get(include=["embeddings"])
    embeddings = np.array(result["embeddings"])
    if len(embeddings) == 0:
        raise ValueError("Vector store is empty")
    return np.mean(embeddings, axis=0)


def check_relevance(
    news_text: str,
    vectorstore: Chroma,
    opinion_full_text: str,
    threshold: float = RELEVANCE_THRESHOLD,
) -> Dict:
    if not news_text.strip():
        return {
            "is_relevant": False, "similarity": 0.0, "term_overlap": 0.0,
            "verdict": "Empty input", "suggestion": "Paste a news article first.",
            "key_terms": [],
        }

    embedder = get_embeddings()
    news_vec = np.array(embedder.embed_query(news_text))
    opinion_vec = compute_centroid(vectorstore)
    similarity = float(np.dot(news_vec, opinion_vec) / (
        np.linalg.norm(news_vec) * np.linalg.norm(opinion_vec) + 1e-9
    ))

    key_terms = extract_likely_case_terms(opinion_full_text)
    if key_terms:
        news_lower = news_text.lower()
        terms_found = sum(1 for term in key_terms if term.lower() in news_lower)
        term_overlap = terms_found / len(key_terms)
    else:
        term_overlap = 0.0

    is_relevant = similarity >= threshold or term_overlap >= 0.5

    if similarity >= 0.75 and term_overlap >= 0.4:
        verdict = "Strongly related to this opinion"
        suggestion = "Proceed with confidence."
    elif similarity >= threshold:
        verdict = "Likely about this case"
        suggestion = "Analysis should be meaningful."
    elif term_overlap >= 0.5:
        verdict = "Mentions the case but may not be the primary topic"
        suggestion = "Results may be noisier than usual."
    elif similarity < 0.40:
        verdict = "Likely about a different case or topic"
        suggestion = (
            "This article does not seem to be about the uploaded opinion. "
            "Check that you pasted the right text, or upload a different opinion PDF."
        )
    else:
        verdict = "Marginal relevance"
        suggestion = "Some overlap detected but weak. Treat scores with caution."

    return {
        "is_relevant": is_relevant,
        "similarity": round(similarity, 3),
        "term_overlap": round(term_overlap, 3),
        "verdict": verdict,
        "suggestion": suggestion,
        "key_terms": key_terms,
    }
