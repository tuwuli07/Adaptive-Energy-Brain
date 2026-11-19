"""Unit tests for ChromaDB RAG Knowledge Base and semantic retrieval."""

import tempfile
from pathlib import Path
import pytest

from src.assistant.rag import ProjectRAGPipeline, RetrievedChunk, COLLECTION_NAME


@pytest.fixture
def temp_rag_pipeline(tmp_path):
    """Fixture providing an isolated RAG pipeline with temporary ChromaDB storage."""
    return ProjectRAGPipeline(persist_dir=tmp_path / "rag_test_db")


def test_markdown_chunking_deterministic():
    content = """# Architecture Overview
This is the introduction to the microgrid.

## Battery Storage System
Battery capacity is 500 kWh with symmetric 150 kW inverter limits.
Degradation is linear throughput cost of $0.02/kWh.

## Model Predictive Control
GNN-Transformer forecast feeds the 24-hour LP rolling horizon.
"""
    chunks = ProjectRAGPipeline.chunk_markdown(content, "docs/test_architecture.md", chunk_size=300, overlap=50)
    assert len(chunks) >= 3

    # Verify deterministic IDs and hashes
    c1 = chunks[0]
    assert "id" in c1
    assert "source" in c1
    assert "hash" in c1
    assert c1["source"] == "docs/test_architecture.md"

    # Running chunking again produces identical IDs
    chunks2 = ProjectRAGPipeline.chunk_markdown(content, "docs/test_architecture.md", chunk_size=300, overlap=50)
    assert [c["id"] for c in chunks] == [c["id"] for c in chunks2]


def test_rag_ingestion_and_retrieval(temp_rag_pipeline, tmp_path):
    pipeline = temp_rag_pipeline
    
    # Create sample docs in tmp_path
    doc_file = tmp_path / "sample_doc.md"
    doc_file.write_text(
        "## Battery Inverter Specifications\n"
        "The battery inverter power limit is strictly 150.0 kW symmetric charging and discharging.\n"
        "Nominal storage capacity is 500 kWh.\n",
        encoding="utf-8"
    )

    # Ingest document
    result = pipeline.ingest_documents(doc_rel_paths=[str(doc_file.resolve())], rebuild=True)
    assert result["status"] == "success"
    assert result["chunk_count"] > 0

    # Query document
    results = pipeline.retrieve("What is the battery inverter power limit?", top_k=1)
    assert len(results) == 1
    top = results[0]
    assert isinstance(top, RetrievedChunk)
    assert "150" in top.text
    assert top.relevance_score > 0.0


def test_rag_empty_and_whitespace_query(temp_rag_pipeline):
    pipeline = temp_rag_pipeline
    assert pipeline.retrieve("") == []
    assert pipeline.retrieve("   ") == []


def test_rag_duplicate_prevention(temp_rag_pipeline, tmp_path):
    pipeline = temp_rag_pipeline
    doc_file = tmp_path / "doc.md"
    doc_file.write_text("## Test Section\nTest content for duplication checks.\n", encoding="utf-8")

    # Ingest once
    pipeline.ingest_documents(doc_rel_paths=[str(doc_file.resolve())], rebuild=True)
    count1 = pipeline.get_collection().count()

    # Ingest without rebuild should skip
    res2 = pipeline.ingest_documents(doc_rel_paths=[str(doc_file.resolve())], rebuild=False)
    assert res2["status"] == "skipped"
    assert pipeline.get_collection().count() == count1
