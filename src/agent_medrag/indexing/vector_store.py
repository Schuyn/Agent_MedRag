'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-05-24 19:34:44
LastEditTime: 2026-05-24 20:03:26
FilePath: /Agent_MedRag/src/agent_medrag/indexing/vector_store.py
Description: 
This module uses Chroma as the project's vector store for the following reasons:
1. Runs locally without requiring a separate database service.
2. Can persist index data under `indexes/chroma`.
3. Simple and well-documented Python API.
4. Common choice in RAG demos, easy to explain in the README.
5. Supports storing metadata alongside vectors (useful for citations).

Responsibilities:
1. Create or open a Chroma collection.
2. add_chunks(chunks, embeddings)
3. query(query_embedding, top_k)
4. Persist the collection to `indexes/chroma`.

Chroma record schema (one record per medical chunk):
- id         = chunk.chunk_id
- document   = chunk.text
- embedding  = embedding model output (384-dimensional vector)
- metadata   = {doc_id, title, source, pub_date, chunk_index, ...}

Example:
collection.add(
    ids=["pubmed_000123_chunk_000"],
    documents=["This study investigates ..."],
    embeddings=[[0.012, -0.031, 0.044, ...]],
    metadatas=[
        {
            "doc_id": "pubmed_000123",
            "title": "Example article title",
            "source": "pubmed",
            "pub_date": "2025-04-03",
            "chunk_index": 0,
        }
    ],
)
'''
from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from agent_medrag.schemas import MedicalChunk

DEFAULT_COLLECTION_NAME="pubmed_chunks"
DEFAULT_PERSIST_DIRECTORY="indexes/chroma"

class ChromaVectorStore:
    def __init__(
        self,
        persist_directory:str | Path=DEFAULT_PERSIST_DIRECTORY,
        collection_name:str | Path=DEFAULT_COLLECTION_NAME,
    )->None:
        self.persist_directory=Path(persist_directory)
        self.persist_directory.mkdir(parents=True,exist_ok=True)
        
        self.client=chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        
        self.collection=self.client.get_or_create_collection(
            name=collection_name
        )
        
    def add_chunks(
        self,
        chunks:list[MedicalChunk],
        embeddings:list[list[float]],
    )->None:
        if not chunks:
            return
        
        if len(chunks)!=len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )
            
        ids=[chunk.chunk_id for chunk in chunks]
        documents=[chunk.text for chunk in chunks]
        metadatas=[self._build_metadata(chunk) for chunk in chunks]
        
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        
    def count(self)->int:
        return self.collection.count()
    
    def query(
        self,
        query_embedding:list[float],
        top_k:int=5,
    )->dict[str,Any]:
        if top_k<=0:
            raise ValueError("top_k must be positive.")
        
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents","metadatas","distances"],
        )
        
    @staticmethod
    def _build_metadata(chunk:MedicalChunk)->dict[str,Any]:
        metadata={
            "doc_id":chunk.doc_id,  # There is no place for "doc_id" in chroma record, this is different from our schema, so we need to store it in metadata
            **chunk.metadata,
        }
        
        return {
            key:value
            for key,value in metadata.items()
            if value is not None    # Filter out None because we can not store null value in vc
        }