"""Greedy oracle extractive summaries plus ROUGE scoring."""
import time
from typing import List, Dict, Any, Tuple

from rouge_score import rouge_scorer

from src.utils import split_sentences, clean_text


_SCORER = rouge_scorer.RougeScorer(
    rouge_types=["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)

MAX_CANDIDATES = 200
MAX_RUNTIME_SECONDS = 90


def rouge_scores(reference: str, hypothesis: str) -> Dict[str, Dict[str, float]]:
    raw = _SCORER.score(reference, hypothesis)
    return {
        m: {
            "precision": round(s.precision, 4),
            "recall": round(s.recall, 4),
            "f1": round(s.fmeasure, 4),
        }
        for m, s in raw.items()
    }


def greedy_oracle_summary(
    source_text: str,
    target_text: str,
    max_sentences: int = 25,
    min_gain: float = 1e-4,
) -> Tuple[str, List[int]]:
    start_time = time.time()
    source_sentences = split_sentences(clean_text(source_text))
    target_clean = clean_text(target_text)

    # Pre-filter: score every sentence against the target, keep top-K.
    scored: List[Tuple[float, int, str]] = []
    for idx, sent in enumerate(source_sentences):
        if len(sent.split()) < 4:
            continue
        if time.time() - start_time > MAX_RUNTIME_SECONDS:
            break
        r1 = _SCORER.score(target_clean, sent)["rouge1"].fmeasure
        if r1 > 0:
            scored.append((r1, idx, sent))

    scored.sort(key=lambda x: -x[0])
    candidates: List[Tuple[int, str]] = [
        (idx, sent) for _, idx, sent in scored[:MAX_CANDIDATES]
    ]

    chosen_indices: List[int] = []
    chosen_sentences: List[str] = []
    best_score = 0.0

    for _ in range(max_sentences):
        if time.time() - start_time > MAX_RUNTIME_SECONDS:
            break

        best_gain = -1.0
        best_candidate = None
        for cand_idx, cand_sent in candidates:
            if cand_idx in chosen_indices:
                continue
            trial = " ".join(chosen_sentences + [cand_sent])
            trial_score = _SCORER.score(target_clean, trial)["rougeL"].fmeasure
            gain = trial_score - best_score
            if gain > best_gain:
                best_gain = gain
                best_candidate = (cand_idx, cand_sent, trial_score)

        if best_candidate is None or best_gain < min_gain:
            break
        cand_idx, cand_sent, new_score = best_candidate
        chosen_indices.append(cand_idx)
        chosen_sentences.append(cand_sent)
        best_score = new_score

    return " ".join(chosen_sentences), chosen_indices


def analyze_news_article(
    opinion_text: str,
    news_text: str,
    max_oracle_sentences: int = 25,
) -> Dict[str, Any]:
    oracle_text, oracle_indices = greedy_oracle_summary(
        source_text=opinion_text,
        target_text=news_text,
        max_sentences=max_oracle_sentences,
    )
    return {
        "oracle_summary": oracle_text,
        "num_oracle_sentences": len(oracle_indices),
        "rouge_news_vs_opinion": rouge_scores(opinion_text, news_text),
        "rouge_news_vs_oracle": rouge_scores(oracle_text, news_text),
        "rouge_oracle_vs_opinion": rouge_scores(opinion_text, oracle_text),
    }
