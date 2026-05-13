"""RAG pipeline: LEGAL-BERT embeddings, sentence-aware chunking, ChromaDB store."""
from pathlib import Path
from typing import Dict, List, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src import config
from src.utils import slugify_case_name, file_hash, split_sentences


# LEGAL-BERT max input is 512 tokens; 400 leaves headroom for special tokens.
MAX_TOKENS_PER_CHUNK = 400
SENTENCE_OVERLAP = 2
MIN_TOKENS_PER_CHUNK = 30


_EMBEDDINGS_CACHE: Dict[str, HuggingFaceEmbeddings] = {}
_TOKENIZER_CACHE: Any = None


def get_embeddings(model_name: str = config.EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    if model_name not in _EMBEDDINGS_CACHE:
        _EMBEDDINGS_CACHE[model_name] = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _EMBEDDINGS_CACHE[model_name]


def _get_tokenizer():
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is None:
        from transformers import AutoTokenizer
        _TOKENIZER_CACHE = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
    return _TOKENIZER_CACHE


def _count_tokens(text: str) -> int:
    tok = _get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))


def _pack_sentences_into_chunks(
    sentences: List[str],
    page_number: int,
    page_label: str,
) -> List[Document]:
    chunks: List[Document] = []
    current_sentences: List[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _count_tokens(sent)

        # Sentence longer than budget on its own: emit it as its own chunk.
        if sent_tokens > MAX_TOKENS_PER_CHUNK:
            if current_sentences:
                chunks.append(_emit_chunk(current_sentences, page_number, page_label))
                current_sentences, current_tokens = [], 0
            chunks.append(_emit_chunk([sent], page_number, page_label))
            continue

        if current_tokens + sent_tokens > MAX_TOKENS_PER_CHUNK and current_sentences:
            chunks.append(_emit_chunk(current_sentences, page_number, page_label))
            carry = current_sentences[-SENTENCE_OVERLAP:] if SENTENCE_OVERLAP > 0 else []
            current_sentences = list(carry)
            current_tokens = sum(_count_tokens(s) for s in current_sentences)

        current_sentences.append(sent)
        current_tokens += sent_tokens

    if current_sentences:
        chunks.append(_emit_chunk(current_sentences, page_number, page_label))

    chunks = [c for c in chunks if _count_tokens(c.page_content) >= MIN_TOKENS_PER_CHUNK]
    return chunks


def _emit_chunk(sentences: List[str], page_number: int, page_label: str) -> Document:
    text = " ".join(sentences).strip()
    return Document(
        page_content=text,
        metadata={
            "page": page_number,
            "page_label": page_label,
            "source": "court_opinion",
            "n_sentences": len(sentences),
        },
    )


def load_and_split(pdf_path: Path) -> List[Document]:
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    all_chunks: List[Document] = []
    for page_doc in pages:
        text = page_doc.page_content
        if not text or not text.strip():
            continue

        page_number = page_doc.metadata.get("page", -1)
        page_label = str(page_doc.metadata.get("page_label", page_number + 1))

        sentences = split_sentences(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            continue

        page_chunks = _pack_sentences_into_chunks(sentences, page_number, page_label)
        all_chunks.extend(page_chunks)

    for i, c in enumerate(all_chunks):
        c.metadata["chunk_id"] = i

    return all_chunks


def build_or_load_vectorstore(pdf_path: Path, force_rebuild: bool = False) -> Chroma:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    slug = slugify_case_name(pdf_path.name)
    fhash = file_hash(pdf_path)
    collection_name = f"{slug}_{fhash}_v2"
    persist_dir = config.CHROMA_ROOT / collection_name
    embeddings = get_embeddings()

    if persist_dir.exists() and not force_rebuild:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )
        if vectorstore._collection.count() > 0:
            return vectorstore

    chunks = load_and_split(pdf_path)
    persist_dir.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )
    return vectorstore


class Retriever:
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path
        self.vectorstore = build_or_load_vectorstore(pdf_path)

    def query(self, question: str, top_k: int = config.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        results = self.vectorstore.similarity_search_with_score(question, k=top_k)
        formatted = []
        for doc, distance in results:
            cos_sim = max(0.0, 1.0 - (distance ** 2) / 2.0)
            page_num = doc.metadata.get("page_label") or doc.metadata.get("page", "?")
            formatted.append({
                "page": page_num,
                "similarity": round(cos_sim, 3),
                "distance": round(distance, 3),
                "text": doc.page_content,
                "chunk_id": doc.metadata.get("chunk_id"),
            })
        return formatted

    def get_full_opinion_text(self) -> str:
        loader = PyPDFLoader(str(self.pdf_path))
        pages = loader.load()
        return "\n\n".join(p.page_content for p in pages)
