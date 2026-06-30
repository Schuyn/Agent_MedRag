'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-06-25 21:47:11
LastEditTime: 2026-06-25 23:08:11
FilePath: /Agent_MedRag/src/agent_medrag/reranking/bge_reranker.py
Description: 

'''
from __future__ import annotations

from dataclasses import replace
from typing import Any

from agent_medrag.schemas import RetrievedEvidence

DEFAULT_RERANKER_MODEL='BAAI/bge-reranker-base'

class BGEReranker:
    def __init__(
        self,
        model_name:str=DEFAULT_RERANKER_MODEL,
        device:str | None='cuda',
        batch_size:int=16,
        show_progress_bar:bool=True,
    )->None:
        if not model_name:
            raise ValueError("model_name must not be empty.")

        if batch_size<=0:
            raise ValueError("batch_size must be positive.")
        
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for BGEReranker. "
                "Install it with: pip install sentence-transformers"
            ) from exc
            
        self.model_name=model_name
        self.device=_normalize_device(device),
        self.batch_size=batch_size,
        self.show_progress_bar=show_progress_bar,
        self.model=CrossEncoder(model_name,device=self.device)
        
    def rerank(
        self,
        query:str,
        evidence:list[RetrievedEvidence],
        top_k:int,
    )->list[RetrievedEvidence]:
        query=query.strip()
        
        if not query:
            raise ValueError("query must not be empty.")
        
        if top_k<=0:
            raise ValueError("top_k must be positive.")
            
        if not evidence:
            return []
        
        pairs=[
            (query,_format_evidence_text(item))
             for item in evidence
        ]
        
        raw_scores=self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
        )
        
        ranked:list[RetrievedEvidence]=[]
        
        for original_rank,item_and_score in enumerate(zip(evidence,raw_scores),start=1):
            item,raw_score=item_and_score
            rerank_score=_score_to_float(raw_score)
            
            metadata=dict(item.metadata)
            metadata['retrieval_rank']=original_rank
            metadata['reranker_model']=self.model_name
            
            ranked.append(
                replace(
                    item,
                    rerank_score=rerank_score,
                    metadata=metadata,
                )
            )
            
        ranked.sort(
            key=lambda item:item.rerank_score if item.rerank_score is not None else float('-inf')
        )    
        