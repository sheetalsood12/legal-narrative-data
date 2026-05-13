"""Project-wide constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHROMA_ROOT = DATA_DIR / "chroma_db"
SAMPLE_DIR = PROJECT_ROOT / "sample_articles"

for d in (RAW_DIR, CHROMA_ROOT, SAMPLE_DIR):
    d.mkdir(parents=True, exist_ok=True)


EMBEDDING_MODEL = "nlpaueb/legal-bert-base-uncased"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

DEFAULT_TOP_K = 5

ORACLE_MAX_SENTENCES = 25
ORACLE_MIN_GAIN = 1e-4


SEMANTIC_AXES = {
    "Reason ↔ Emotion": {
        "negative_pole": ["statute", "precedent", "analysis", "holding",
                          "reasoning", "doctrine", "jurisprudence", "argument",
                          "principle", "rationale"],
        "positive_pole": ["outrage", "fear", "devastating", "heartbreak",
                          "anger", "grief", "shocking", "horrified",
                          "tragic", "furious"],
    },
    "Abstract ↔ Concrete": {
        "negative_pole": ["concept", "principle", "theory", "ideology",
                          "philosophy", "framework", "abstraction", "notion"],
        "positive_pole": ["woman", "doctor", "clinic", "hospital",
                          "patient", "mother", "child", "pregnancy"],
    },
    "Passive ↔ Active": {
        "negative_pole": ["considered", "noted", "observed", "stated",
                          "described", "indicated", "mentioned", "addressed"],
        "positive_pole": ["overruled", "struck", "held", "ruled",
                          "decided", "rejected", "reversed", "abolished"],
    },
}


GLOVE_PATH = DATA_DIR / "glove.6B.300d.txt"

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"

STANCE_MODEL = "facebook/bart-large-mnli"
STANCE_LABELS = [
    "supportive of the ruling",
    "critical of the ruling",
    "neutral coverage of the ruling",
]

RELEVANCE_THRESHOLD = 0.60
