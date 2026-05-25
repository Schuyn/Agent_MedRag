'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-05-24 19:35:06
LastEditTime: 2026-05-25 17:12:42
FilePath: /Agent_MedRag/src/agent_medrag/indexing/index_builder.py
Description: 
Build a persistent vector index from chunked medical documents.

Pipeline stage:
    chunks.jsonl -> embeddings -> Chroma vector store
'''
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_medrag.indexing.embeddings import TextEmbeddingModel
from agent_medrag.indexing.vector_store import ChromaVectorStore
from agent_medrag.schemas import MedicalChunk

def load_chunks_jsonl(
    input_path: str | Path,
)->list[MedicalChunk]:
    input_path=Path(input_path)
    chunks:list[MedicalChunk]=[]
    
    with input_path.open("r",encoding="utf-8") as file:
        for line_number,line in enumerate(file,start=1):    # We want id start from 1
            line=line.strip()
            
            if not line:
                continue
            
            try:
                raw_chunk:dict[str,Any]=json.loads(line)
                
                chunks.append(
                    MedicalChunk(
                        chunk_id=raw_chunk["chunk_id"],
                        doc_id=raw_chunk["doc_id"],
                        text=raw_chunk["text"],
                        metadata=raw_chunk.get("metadata",{}),
                    )
                )
            except (json.JSONDecodeError,KeyError,TypeError) as exc:
                raise ValueError(
                    f"Invalid chunk record at line {line_number} in {input_path}"
                )from exc
                
    return chunks

def build_index(
    chunks:list[MedicalChunk],
    embedding_model:TextEmbeddingModel,
    vector_storage:ChromaVectorStore,
    batch_size:int=64,
)->int:  # Returns the number of indexed chunks
    """
    Build the vector index by embedding chunks in batches and adding them to
    the provided vector store.

    Processing is done in batches to avoid high memory usage and to provide
    progress visibility. Do not embed and write all chunks at once.

    Args:
        chunks: list of MedicalChunk objects to index.
        embedding_model: a TextEmbeddingModel instance for computing embeddings.
        vector_storage: a ChromaVectorStore instance where vectors are stored.
        batch_size: number of chunks to process per batch (must be > 0).

    Returns:
        int: total number of chunks indexed.
    """
    if batch_size<=0:
        raise ValueError("batch_size must be positive.")
    
    if not chunks:
        return 0
    
    indexed_count=0
    
    for start in range(0,len(chunks),batch_size):
        end=start+batch_size
        batch_chunks=chunks[start:end]  # Python list slicing ensures there will be no out of range
        batch_texts=[chunk.text for chunk in batch_chunks]
        
        batch_embeddings=embedding_model.embed_texts(batch_texts)
        
        vector_storage.add_chunks(
            chunks=batch_chunks,
            embeddings=batch_embeddings,
        )
        
        indexed_count+=len(batch_chunks)
        
        print(f"Indexed {indexed_count}/{len(chunks)} chunks")
        
    return indexed_count