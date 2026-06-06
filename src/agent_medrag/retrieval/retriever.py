'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-06-03 15:33:02
LastEditTime: 2026-06-03 15:43:33
FilePath: /Agent_MedRag/src/agent_medrag/retrieval/retriever.py
Description: 

'''
from __future__ import annotations

from typing import Any

from agent_medrag.indexing.embeddings import TextEmbeddingModel
from agent_medrag.indexing.vector_store import ChromaVectorStore
from agent_medrag.schemas import RetrievedEvidence

class LocalVectorRetriever:
    def __init__(
        self,
        embedding_model:TextEmbeddingModel,
        vector_store:ChromaVectorStore,
        top_k:int=5,
    )->None:
        if top_k<=0:
            raise ValueError("top_k must be positive.")

        self.embedding_model=embedding_model
        self.vector_store=vector_store
        self.top_k=top_k
        
    def retrieve(self,query:str,top_k:int | None=None)->list[RetrievedEvidence]:
        query=query.strip()
        if not query:
            raise ValueError("query must not be empty.")
        
        effective_top_k=top_k or self.top_k
        if effective_top_k<=0:
            raise ValueError("top_k must be positive.")
        
        query_embeddings=self.embedding_model.embed_texts([query])
        if not query_embeddings:
            return []
        query_embedding=query_embeddings[0]
        
        raw_results=self.vector_store.query(
            query_embedding=query_embedding,
            top_k=effective_top_k,
        )
        
        return _chroma_results_to_device(raw_results)
    
def _chroma_results_to_device(results:dict[str,Any])->list[RetrievedEvidence]:
    ids=_first_result_list(results.get('ids'))
    documents=_first_result_list(results.get('documents'))
    metadatas=_first_result_list(results.get('metadatas'))
    distances=_first_result_list(results.get('distances'))
    
    evidence:list[RetrievedEvidence]=[]
    
    for i,chunk_id in enumerate(ids):
        metadata=metadatas[i] or {}
        distance=distances[i] if i<len(distances) else None
        
        evidence.append(
            RetrievedEvidence(
                chunk_id=str(chunk_id),
                doc_id=str(metadata.get('doc_id','')),   # If doc_id not exist, return '' instead of None, which is not acceptable
                title=str(metadata.get('title','')),
                text=str(documents[i] or ''),
                retrieval_score=float(distance) if distance is not None else None,
                pub_date=metadata.get('pub_date'),
                source=str(metadata.get('source','local')),
                metadata=dict(metadata),
            )
        )
    
    return evidence
    
def _first_result_list(value:Any)->list[Any]:
    if not value:
        return []
    
    if isinstance(value,list) and isinstance(value[0],list):
        return value[0]
    
    if isinstance(value,list):
        return value
    
    return []