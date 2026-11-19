"""Reproducible RAG Knowledge Base Ingestion Script for Adaptive Energy Brain.

Usage:
    python scripts/ingest_rag_docs.py [--rebuild] [--persist-dir PATH]
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assistant.rag import ProjectRAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_rag")


def main():
    parser = argparse.ArgumentParser(description="Ingest verified project documentation into ChromaDB RAG index.")
    parser.add_argument("--rebuild", action="store_true", help="Delete existing index and re-ingest all documents.")
    parser.add_argument("--persist-dir", type=str, default=None, help="Custom persistence directory for ChromaDB.")
    args = parser.parse_args()

    persist_path = Path(args.persist_dir) if args.persist_dir else None
    pipeline = ProjectRAGPipeline(persist_dir=persist_path)

    logger.info(f"Target ChromaDB directory: {pipeline.persist_dir}")
    logger.info(f"Rebuild requested: {args.rebuild}")

    result = pipeline.ingest_documents(rebuild=args.rebuild)
    print("\n--- Ingestion Summary ---")
    print(f"Status: {result.get('status')}")
    print(f"Total Chunks: {result.get('chunk_count')}")
    print(f"Ingested Files ({len(result.get('ingested_files', []))}):")
    for f in result.get("ingested_files", []):
        print(f"  + {f}")
    if result.get("missing_files"):
        print(f"Missing Files ({len(result.get('missing_files', []))}):")
        for f in result.get("missing_files", []):
            print(f"  - {f}")
    print("-------------------------\n")


if __name__ == "__main__":
    main()
