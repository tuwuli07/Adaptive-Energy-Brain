"""ChromaDB-based Retrieval-Augmented Generation (RAG) for Adaptive Energy Brain.

Indexes verified architecture documents, completion reports, mathematical formulations,
and benchmark summaries to provide grounded contextual retrieval for the AI Assistant.
"""

from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.models.Collection import Collection

logger = logging.getLogger("rag")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PERSIST_DIR = ROOT / "data" / "rag_index"
COLLECTION_NAME = "adaptive_energy_brain_kb"

# Authoritative documents to index for verified RAG knowledge
CORE_DOCS = [
    "README.md",
    "docs/baseline_architecture.md",
    "docs/simulation_architecture.md",
    "docs/forecasting_requirements.md",
    "docs/part4_completion_report.md",
    "docs/part5_completion_report.md",
    "docs/part6_completion_report.md",
    "docs/part7_model_based_control_architecture.md",
    "docs/part7_completion_report.md",
    "docs/part8_reconciliation_report.md",
    "docs/part10_completion_report.md",
    "docs/part11_completion_report.md",
    "docs/part12_completion_report.md",
]


@dataclass
class RetrievedChunk:
    """Represents a grounded passage retrieved from the knowledge base."""
    chunk_id: str
    text: str
    source_file: str
    section_title: str
    distance: float
    relevance_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_file": self.source_file,
            "section_title": self.section_title,
            "distance": round(self.distance, 4),
            "relevance_score": round(self.relevance_score, 4),
        }


class ProjectRAGPipeline:
    """Manages persistent ChromaDB vector storage and semantic retrieval."""

    def __init__(self, persist_dir: Optional[Path] = None):
        env_dir = os.environ.get("RAG_PERSIST_DIR")
        self.persist_dir = Path(env_dir) if env_dir else (persist_dir or DEFAULT_PERSIST_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[Collection] = None

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._client

    def get_collection(self) -> Collection:
        """Get or create the ChromaDB collection."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Adaptive Energy Brain verified research and documentation"}
            )
        return self._collection

    @staticmethod
    def chunk_markdown(content: str, source_path: str, chunk_size: int = 900, overlap: int = 150) -> List[Dict[str, Any]]:
        """Split markdown content into contextual sections and chunks."""
        sections = []
        current_heading = "Introduction"
        current_lines = []

        for line in content.splitlines():
            heading_match = re.match(r"^(#{1,3})\s+(.+)$", line)
            if heading_match:
                if current_lines:
                    text_block = "\n".join(current_lines).strip()
                    if text_block:
                        sections.append((current_heading, text_block))
                    current_lines = []
                current_heading = heading_match.group(2).strip()
            else:
                current_lines.append(line)

        if current_lines:
            text_block = "\n".join(current_lines).strip()
            if text_block:
                sections.append((current_heading, text_block))

        chunks = []
        step = max(100, chunk_size - overlap)

        for heading, section_text in sections:
            if len(section_text) <= chunk_size:
                text_slice = section_text
                c_hash = hashlib.sha256(text_slice.encode("utf-8")).hexdigest()[:8]
                chunk_id = f"{Path(source_path).stem}_{heading.lower()[:15].replace(' ', '_')}_{c_hash}"
                chunks.append({
                    "id": chunk_id,
                    "text": f"[{heading}]\n{text_slice}",
                    "source": source_path,
                    "section": heading,
                    "hash": c_hash,
                })
            else:
                for start_idx in range(0, len(section_text), step):
                    text_slice = section_text[start_idx : start_idx + chunk_size]
                    if len(text_slice.strip()) < 80:
                        continue
                    c_hash = hashlib.sha256(text_slice.encode("utf-8")).hexdigest()[:8]
                    chunk_id = f"{Path(source_path).stem}_{heading.lower()[:12].replace(' ', '_')}_{start_idx}_{c_hash}"
                    chunks.append({
                        "id": chunk_id,
                        "text": f"[{heading}]\n{text_slice}",
                        "source": source_path,
                        "section": heading,
                        "hash": c_hash,
                    })

        return chunks

    def ingest_documents(self, doc_rel_paths: Optional[List[str]] = None, rebuild: bool = False) -> Dict[str, Any]:
        """Ingest documentation files into ChromaDB with deterministic IDs and metadata."""
        collection = self.get_collection()
        doc_paths = doc_rel_paths or CORE_DOCS

        existing_count = collection.count()
        if not rebuild and existing_count > 0:
            logger.info(f"ChromaDB collection '{COLLECTION_NAME}' already populated with {existing_count} chunks. Skipping re-ingestion.")
            return {"status": "skipped", "reason": "already_populated", "chunk_count": existing_count}

        if rebuild and existing_count > 0:
            logger.info(f"Rebuild requested: purging {existing_count} existing chunks.")
            client = self._get_client()
            client.delete_collection(COLLECTION_NAME)
            self._collection = None
            collection = self.get_collection()

        all_chunks = []
        ingested_files = []
        missing_files = []

        for rel_path in doc_paths:
            full_path = ROOT / rel_path
            if not full_path.exists():
                logger.warning(f"RAG document missing: {rel_path}")
                missing_files.append(rel_path)
                continue

            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                file_chunks = self.chunk_markdown(content, rel_path)
                all_chunks.extend(file_chunks)
                ingested_files.append(rel_path)
            except Exception as e:
                logger.error(f"Error reading {rel_path} for RAG: {e}")

        if not all_chunks:
            return {"status": "empty", "ingested_files": [], "missing_files": missing_files, "chunk_count": 0}

        # Batch insert into ChromaDB
        batch_size = 50
        total_added = 0
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            collection.upsert(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[{"source": c["source"], "section": c["section"], "content_hash": c["hash"]} for c in batch],
            )
            total_added += len(batch)

        logger.info(f"Successfully indexed {total_added} chunks across {len(ingested_files)} files into ChromaDB.")
        return {
            "status": "success",
            "ingested_files": ingested_files,
            "missing_files": missing_files,
            "chunk_count": total_added,
        }

    def retrieve(self, query: str, top_k: int = 4, source_filter: Optional[str] = None) -> List[RetrievedChunk]:
        """Perform semantic retrieval against the ChromaDB knowledge base."""
        if not query or not query.strip():
            return []

        collection = self.get_collection()
        if collection.count() == 0:
            logger.warning("RAG retrieval attempted on empty collection. Attempting on-demand ingestion.")
            self.ingest_documents(rebuild=False)
            if collection.count() == 0:
                return []

        where_filter = {"source": source_filter} if source_filter else None
        
        try:
            results = collection.query(
                query_texts=[query],
                n_results=min(top_k, max(1, collection.count())),
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.error(f"ChromaDB query error: {e}")
            return []

        retrieved: List[RetrievedChunk] = []
        if not results or not results.get("documents"):
            return retrieved

        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
        ids = results["ids"][0] if results.get("ids") else [f"doc_{i}" for i in range(len(docs))]

        for cid, doc, meta, dist in zip(ids, docs, metas, distances):
            # Chroma L2 distance conversion to relevance score in [0.0, 1.0]
            relevance = 1.0 / (1.0 + float(dist)) if dist is not None else 0.5
            retrieved.append(RetrievedChunk(
                chunk_id=cid,
                text=doc,
                source_file=meta.get("source", "unknown"),
                section_title=meta.get("section", "General"),
                distance=float(dist) if dist is not None else 0.0,
                relevance_score=relevance,
            ))

        return retrieved


# Global RAG pipeline instance
RAG_PIPELINE = ProjectRAGPipeline()
