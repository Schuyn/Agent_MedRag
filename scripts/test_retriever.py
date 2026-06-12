'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-06-12 13:03:20
LastEditTime: 2026-06-12 13:17:02
FilePath: /Agent_MedRag/scripts/test_retriever.py
Description: 
Individually test retriever.

Test Result:

(MedRag) F:\GitHub\Agent_MedRag>C:/Users/Victo/.conda/envs/MedRag/python.exe f:/GitHub/Agent_MedRag/scripts/test_retriever.py
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 6748.58it/s]
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.56it/s]
Retrieved 3 chunks

--- Result 1 ---
chunk_id: pubmed_000008_chunk_000
doc_id: pubmed_000008
title: Antibacterial sonodynamic nanomedicine: mechanism, category, and applications.
distance: 0.2826382517814636
text: Sonodynamic therapy (SDT) has emerged as a cutting-edge strategy for combating multidrug-resistant bacterial infections. Unlike conventional antibiotics, SDT leverages the generation of reactive oxygen species during the treatment process to inflict multifaceted damage on bacterial cells, thereby si

--- Result 2 ---
chunk_id: pubmed_000008_chunk_002
doc_id: pubmed_000008
title: Antibacterial sonodynamic nanomedicine: mechanism, category, and applications.
distance: 0.4596604108810425
text: reactive oxygen species. Furthermore, it provides a comprehensive analysis of various sonosensitisers used in SDT, emphasising their potential to enhance therapeutic outcomes in areas such as infected wound healing, bone regeneration, and the mitigation of deep tissue inflammation. While SDT shows g

--- Result 3 ---
chunk_id: pubmed_000750_chunk_000
doc_id: pubmed_000750
title: The action of different irrigant activation methods on engineered endodontic biofilm: an <i>in vitro</i> study.
distance: 0.5811762809753418
text: Endodontic infections are biofilm-mediated, demanding effective biofilm eradication from the root canal. Root canal complexities, coupled with bacterial biofilm resistance, pose challenges to thorough disinfection. Irrigation, particularly with sodium hypochlorite, is crucial in endodontics. Activat
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