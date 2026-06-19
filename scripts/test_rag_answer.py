import json

from agent_medrag.config import get_config_value,load_config,resolve_device
from agent_medrag.generation import AnswerGenerator,DeepSeekProvider
from agent_medrag.indexing.embeddings import SentenceTransformerEmbedding
from agent_medrag.indexing.vector_store import ChromaVectorStore
from agent_medrag.retrieval import LocalVectorRetriever

def main()->None:
    config=load_config('configs/default.yaml')
    
    question=(
        "How does sonodynamic therapy differ from conventional antibiotics?"
    )

    embedding_model=SentenceTransformerEmbedding(
        model_name=get_config_value(
            config,
            'embedding',
            'model_name',
            'BAAI/bge-large-en-v1.5',
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
            'indexes/chroma',
        ),
        collection_name=get_config_value(
            config,
            'vector_store',
            'collection_name',
            'pubmed_articles',
        ),
    )
    
    retriever=LocalVectorRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=get_config_value(config,'retrieval','top_k',5),
    )
    
    evidence=retriever.retrieve(question,top_k=3)
    
    print(f"Retrieved {len(evidence)} evidence chunks")
    
    for index,item in enumerate(evidence,start=1):
        print(f'E{index}: {item.chunk_id} |  {item.title}')
        
    llm_provider=DeepSeekProvider(
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
    
    generator=AnswerGenerator(llm_provider)
    answer=generator.generate(question,evidence)
    
    print(json.dumps(answer.to_json_dict(),indent='\t',ensure_ascii=False))
    assert answer.answer
    assert answer.confidence in {'low','medium','high'}
    assert answer.safety_note
    assert answer.retrieval_summary.retrieved_count==len(evidence)
    
    if evidence:
        assert answer.citations,(
            "Expected at least one citation when evidence was retrieved."
        )
        assert answer.retrieval_summary.used_evidence_count==len(
            answer.citations
        )
    
    print("rag answer pipeline works")

if __name__=='__main__':
    main()