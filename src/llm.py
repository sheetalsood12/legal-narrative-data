"""Ollama integration for optional retrieval-augmented synthesis and LLM-as-judge scoring."""
import json
from typing import Dict, List, Any, Optional

import requests

from src import config


SYSTEM_PROMPT_RAG = """You are a legal research assistant. Given retrieved passages from a court opinion, answer the user's question using ONLY information present in the passages.

RULES:
- Quote directly from the passages whenever possible.
- If passages don't contain the answer, say so explicitly. Do not speculate.
- Cite passage numbers like [Passage 1], [Passage 2] in your answer.
- Be concise. No more than 4 sentences unless the question demands more.
"""


JUDGE_RUBRICS = {
    "majority_vs_dissent_emphasis": {
        "description": "Does the article emphasize the majority opinion's reasoning or the dissent's reasoning?",
        "scale": "1=heavily majority-focused, 3=balanced, 5=heavily dissent-focused",
    },
    "emotional_vs_analytical_framing": {
        "description": "Does the article frame the ruling in emotional terms or analytical/legal terms?",
        "scale": "1=heavily emotional, 3=balanced, 5=heavily analytical",
    },
    "consequences_vs_reasoning": {
        "description": "Does the article focus on the consequences of the ruling or the legal reasoning?",
        "scale": "1=heavily consequences-focused, 3=balanced, 5=heavily reasoning-focused",
    },
}


def is_ollama_available(host: str = config.OLLAMA_HOST) -> bool:
    try:
        r = requests.get(f"{host}/api/tags", timeout=2)
        return r.status_code == 200
    except requests.RequestException:
        return False


def list_available_models(host: str = config.OLLAMA_HOST) -> List[str]:
    try:
        r = requests.get(f"{host}/api/tags", timeout=2)
        return [m["name"] for m in r.json().get("models", [])]
    except requests.RequestException:
        return []


def synthesize_answer(
    question: str,
    passages: List[Dict[str, Any]],
    model: str = config.OLLAMA_MODEL,
    host: str = config.OLLAMA_HOST,
) -> str:
    if not is_ollama_available(host):
        return (
            "Ollama is not running. To enable LLM synthesis:\n"
            "1. Install Ollama from https://ollama.com/download\n"
            "2. Run `ollama pull llama3.1:8b` in your terminal\n"
            "3. Refresh this page"
        )

    context = "\n\n".join(
        f"[Passage {i+1} | page {p['page']}]\n{p['text']}"
        for i, p in enumerate(passages)
    )
    user_message = (
        f"Retrieved passages from the court opinion:\n\n{context}\n\n"
        f"Question: {question}\n\nAnswer using ONLY the passages above. "
        f"Cite passage numbers."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_RAG},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 4096},
    }
    try:
        r = requests.post(f"{host}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except requests.RequestException as e:
        return f"LLM call failed: {e}"


JUDGE_SYSTEM_PROMPT = """You are an objective news-coverage analyst. You score news articles about court rulings on specific rubrics.

For each rubric, return ONLY a JSON object with this exact structure:
{
  "score": <integer 1-5>,
  "reasoning": "<one-sentence explanation>"
}

Be deterministic and consistent. Use the integer scale exactly as defined.
Do not include any text outside the JSON object.
"""


def llm_as_judge(
    article_text: str,
    rubric_key: str,
    case_name: str = "the ruling",
    model: str = config.OLLAMA_MODEL,
    host: str = config.OLLAMA_HOST,
    max_chars: int = 6000,
) -> Optional[Dict]:
    if not is_ollama_available(host):
        return None
    if rubric_key not in JUDGE_RUBRICS:
        return None

    rubric = JUDGE_RUBRICS[rubric_key]
    text_input = article_text[:max_chars]

    user_message = (
        f"RUBRIC: {rubric['description']}\n"
        f"SCALE: {rubric['scale']}\n\n"
        f"CASE: {case_name}\n\n"
        f"NEWS ARTICLE TEXT:\n{text_input}\n\n"
        f"Return ONLY the JSON object with `score` (integer 1-5) and "
        f"`reasoning` (one sentence). No other text."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 8192},
        "format": "json",
    }
    try:
        r = requests.post(f"{host}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        content = r.json()["message"]["content"]
        parsed = json.loads(content.strip())
        score = parsed.get("score")
        if isinstance(score, str):
            try:
                score = int(score)
            except ValueError:
                return None
        if not isinstance(score, int) or not (1 <= score <= 5):
            return None
        return {
            "rubric": rubric_key,
            "score": score,
            "reasoning": str(parsed.get("reasoning", ""))[:300],
        }
    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        print(f"Judge call failed: {e}")
        return None


def llm_judge_all_rubrics(
    article_text: str,
    case_name: str = "the ruling",
    model: str = config.OLLAMA_MODEL,
    host: str = config.OLLAMA_HOST,
) -> Dict[str, Dict]:
    out = {}
    for rubric_key in JUDGE_RUBRICS:
        result = llm_as_judge(article_text, rubric_key, case_name, model, host)
        if result is not None:
            out[rubric_key] = result
    return out
