'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-05-20 18:28:55
LastEditTime: 2026-05-20 19:10:07
FilePath: /Agent_MedRag/src/agent_medrag/indexing/__init__.py
Description:
Indexing module: text chunking and embedding components that prepare documents for vector-store retrieval. Re-exports chunk_document, chunk_documents, the TextEmbeddingModel Protocol, and the SentenceTransformerEmbedding implementation for convenient single-point imports.
'''
from agent_medrag.indexing.chunker import chunk_document, chunk_documents
from agent_medrag.indexing.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbedding,
    TextEmbeddingModel,
)

__all__ = [
    "chunk_document",
    "chunk_documents",
    "DEFAULT_EMBEDDING_MODEL",
    "SentenceTransformerEmbedding",
    "TextEmbeddingModel",
]


__version__ = '0.1.0'