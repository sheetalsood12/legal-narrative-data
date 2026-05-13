"""
scripts/ingest.py
=================
CLI to pre-build the vector store for an opinion PDF.

Usage:
    python -m scripts.ingest data/raw/dobbs_opinion.pdf
    python -m scripts.ingest data/raw/dobbs_opinion.pdf --smoke-test
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag import Retriever


def main():
    parser = argparse.ArgumentParser(description="Build vector store for an opinion.")
    parser.add_argument("pdf_path", type=Path, help="Path to the opinion PDF.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force re-embedding even if a cached vector store exists.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a sample query after ingestion.")
    args = parser.parse_args()

    if not args.pdf_path.exists():
        sys.exit(f"PDF not found: {args.pdf_path}")

    print(f"📄 Ingesting: {args.pdf_path}")
    t0 = time.time()
    retriever = Retriever(args.pdf_path)
    print(f"✅ Vector store ready in {time.time() - t0:.1f}s.")

    if args.smoke_test:
        print("\n--- Smoke test ---")
        question = "What did the court hold?"
        print(f"Q: {question}")
        for i, r in enumerate(retriever.query(question, top_k=3), 1):
            print(f"\n[{i}] page {r['page']} | similarity {r['similarity']}")
            print(f"    {r['text'][:300]}…")


if __name__ == "__main__":
    main()
