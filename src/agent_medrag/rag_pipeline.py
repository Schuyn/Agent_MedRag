'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-06-19 18:37:20
LastEditTime: 2026-06-19 19:14:20
FilePath: /Agent_MedRag/src/agent_medrag/rag_pipeline.py
Description:
Medical RAG (Retrieval-Augmented Generation) pipeline orchestrator. Provides
RAGPipeline class that integrates retrieval and answer generation components
for unified question-answering interface. Exposes build_rag_pipeline() factory
function to construct the complete pipeline from YAML configuration.
'''
from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_medrag.config import get_config_value,load_config,resolve_device
from agent_medrag.generation import AnswerGenerator,DeepSeekProvider
from agent_medrag.indexing.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbedding,
)
from agent_medrag.indexing.vector_store import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIRECTORY,
    ChromaVectorStore,
)
from agent_medrag.retrieval import LocalVectorRetriever
from agent_medrag.schemas import MedicalAnswer

class RAGPipeline:
    def __init__(
        self,
        retriever:LocalVectorRetriever,
        answer_generator:AnswerGenerator
    ):
        self.retriever=retriever
        self.answer_generator=answer_generator
        
    def ask(
        self,
        question:str,
        top_k:int | None=None,
    )->MedicalAnswer:
        evidence=self.retriever.retrieve(question,top_k=top_k)
        return self.answer_generator.generate(question,evidence)


def build_rag_pipeline(
    config_path:str | Path='configs/default.yaml',
)->RAGPipeline:
    config=load_config(config_path)
    
    embedding_model=SentenceTransformerEmbedding(
        model_name=get_config_value(
            config,
            'embedding',
            'model_name',
            DEFAULT_EMBEDDING_MODEL,
        ),
        normalize_embeddings=get_config_value(
            config,
            'embedding',
            'normalize_embeddings',
            True,
        ),
        device=resolve_device(
            get_config_value(config,'embedding','device',None)
        ),
    )
    
    vector_store=ChromaVectorStore(
        persist_directory=get_config_value(
            config,
            'vector_store',
            'persist_dir',
            DEFAULT_PERSIST_DIRECTORY,
        ),
        collection_name=get_config_value(
            config,
            'vector_store',
            'collection_name',
            DEFAULT_COLLECTION_NAME,
        ),
    )
    
    retriever=LocalVectorRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=get_config_value(config,'retrieval','top_k',5),
    )
    
    llm_provider=_build_llm_provider(config)
    answer_generator=AnswerGenerator(llm_provider)
    
    return RAGPipeline(
        retriever,
        answer_generator,
    )

def _build_llm_provider(config:dict[str,Any])->DeepSeekProvider:
    provider=get_config_value(config,'llm','provider','deepseek')
    
    if provider!='deepseek':
        raise ValueError(
            f"Unsupported llm.provider: {provider}. "
            "Stage 3 currently supports only 'deepseek'."
        )
    
    return DeepSeekProvider(
        model=get_config_value(
            config,
            'llm',
            'model',
            'deepseek-v4-pro',
        ),
        temperature=get_config_value(
            config,
            'llm',
            'temperature',
            0.1,
        ),
        max_output_tokens=get_config_value(
            config,
            'llm',
            'max_output_tokens',
            800,
        ),
        thinking_enabled=get_config_value(
            config,
            'llm',
            'thinking_enabled',
            False,
        ),
        reasoning_effort=get_config_value(
            config,
            'llm',
            'reasoning_effort',
            'high',
        ),
    )