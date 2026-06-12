'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-06-12 13:03:20
LastEditTime: 2026-06-12 13:17:02
FilePath: /Agent_MedRag/scripts/test_retriever.py
Description: 
Individually test retriever.
'''
from agent_medrag.indexing.embeddings import SentenceTransformerEmbedding
from agent_medrag.indexing.vector_store import ChromaVectorStore
from agent_medrag.retrieval import LocalVectorRetriever

def main()->None:
    embedding_model=SentenceTransformerEmbedding(
        model_name='BAAI/bge-large-en-v1.5',
        normalize_embeddings=True,
        device='cuda',
    )
    
    vector_store=ChromaVectorStore(
        persist_directory='indexes/chroma',
        collection_name='pubmed_articles',
    )
    
    retiever=LocalVectorRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=3,
    )
    
    evidence=retiever.retrieve(
        "How does sonodynamic therapy treat bacterial infections?"
    )
    
    print(f"Retrieved {len(evidence)} chunks")
    
    for rank,item in enumerate(evidence,start=1):
        print(f"\n--- Result {rank} ---")
        print(f"chunk_id: {item.chunk_id}")
        print(f"doc_id: {item.doc_id}")
        print(f"title: {item.title}")
        print(f"distance: {item.retrieval_score}")
        print(f"text: {item.text[:300]}")
        
if __name__=='__main__':
    main()