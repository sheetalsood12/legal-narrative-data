"""Zero-shot stance detection using BART-MNLI. Optional; degrades gracefully."""
from typing import Dict, List, Optional

from src import config


_PIPELINE = None
_LOAD_FAILED = False


def _try_load_pipeline():
    global _PIPELINE, _LOAD_FAILED
    if _PIPELINE is not None:
        return _PIPELINE
    if _LOAD_FAILED:
        return None
    try:
        from transformers import pipeline
        _PIPELINE = pipeline(
            task="zero-shot-classification",
            model=config.STANCE_MODEL,
            device=-1,
        )
        return _PIPELINE
    except Exception as e:
        print(f"Stance model failed to load: {e}")
        _LOAD_FAILED = True
        return None


def is_stance_available() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def analyze_stance(
    text: str,
    labels: Optional[List[str]] = None,
    max_chars: int = 4000,
) -> Optional[Dict]:
    if not text or not text.strip():
        return None

    pipe = _try_load_pipeline()
    if pipe is None:
        return None

    text_input = text[:max_chars]
    label_set = labels or config.STANCE_LABELS

    try:
        result = pipe(text_input, candidate_labels=label_set, multi_label=False)
        scores = {label: round(float(score), 4)
                  for label, score in zip(result["labels"], result["scores"])}
        predicted_label = result["labels"][0]
        confidence = round(float(result["scores"][0]), 4)
        return {
            "scores": scores,
            "predicted_label": predicted_label,
            "confidence": confidence,
        }
    except Exception as e:
        print(f"Stance inference failed: {e}")
        return None


def stance_to_scalar(stance_result: Dict) -> Optional[float]:
    if not stance_result or "scores" not in stance_result:
        return None
    scores = stance_result["scores"]
    supportive = 0.0
    critical = 0.0
    for label, score in scores.items():
        if "support" in label.lower() or "favor" in label.lower():
            supportive = score
        elif "critic" in label.lower() or "against" in label.lower():
            critical = score
    return round(supportive - critical, 4)
