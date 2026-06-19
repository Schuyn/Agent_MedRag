import json

from agent_medrag.rag_pipeline import build_rag_pipeline

def main()->None:
    question=(
        "How does sonodynamic therapy differ from conventional antibiotics?"
    )

    pipeline=build_rag_pipeline('configs/default.yaml')
    answer=pipeline.ask(question,top_k=3)
    
    print(json.dumps(answer.to_json_dict(),indent='\t',ensure_ascii=False))
    assert answer.answer
    assert answer.confidence in {'low','medium','high'}
    assert answer.safety_note
    assert answer.retrieval_summary.retrieved_count==3
    assert answer.citations
    
    print("rag answer pipeline works")

if __name__=='__main__':
    main()
    
'''
Result:
(MedRag) F:\GitHub\Agent_MedRag>C:/Users/Victo/.conda/envs/MedRag/python.exe f:/GitHub/Agent_MedRag/scripts/test_rag_answer.py
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:00<00:00, 11987.40it/s]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.96it/s]
Retrieved 3 evidence chunks
E1: pubmed_000008_chunk_000 |  Antibacterial sonodynamic nanomedicine: mechanism, category, and applications.
E2: pubmed_000008_chunk_002 |  Antibacterial sonodynamic nanomedicine: mechanism, category, and applications.
E3: pubmed_000750_chunk_000 |  The action of different irrigant activation methods on engineered endodontic biofilm: an <i>in vitro</i> study.
{
        "question": "How does sonodynamic therapy differ from conventional antibiotics?",
        "answer": "Sonodynamic therapy (SDT) differs from conventional antibiotics primarily in its mechanism of action. While conventional antibiotics typically target specific bacterial structures or metabolic pathways, SDT generates reactive oxygen species that inflict multifaceted damage on bacterial cells, which significantly reduces the likelihood of developing drug resistance [E1]. Additionally, SDT offers enhanced tissue penetration compared to other physical sterilization methods like ultraviolet irradiation [E1].",
        "confidence": "medium",
        "citations": [
                {
                        "title": "Antibacterial sonodynamic nanomedicine: mechanism, category, and applications.",
                        "pub_date": "2025-03-25",
                        "evidence": "Sonodynamic therapy (SDT) has emerged as a cutting-edge strategy for combating multidrug-resistant bacterial infections. Unlike conventional antibiotics, SDT leverages the generation of reactive oxygen species during the treatment process to inflict multifaceted damage on bacterial cells, thereby significantly reducing the likelihood of developing drug resistance. Compared to other physical sterilisation methods, such as ultraviolet irradiation, SDT offers enhanced tissue penetration, making it particularly",
                        "doc_id": "pubmed_000008",
                        "chunk_id": "pubmed_000008_chunk_000",
                        "url": null
                }
        ],
        "safety_note": "This answer is for medical literature exploration and is not medical advice.",
        "retrieval_summary": {
                "query": "How does sonodynamic therapy differ from conventional antibiotics?",
                "top_k": 3,
                "retrieved_count": 3,
                "used_evidence_count": 1
        },
        "limitations": "The provided evidence does not detail the specific mechanisms of conventional antibiotics for a direct comparison, and the information is derived from a single source focused on SDT. Evidence E2 and E3 do not directly address the comparison between SDT and conventional antibiotics.",
        "metadata": {
                "used_evidence_ids": [
                        "E1"
                ],
                "raw_llm_output": "{\n  \"answer\": \"Sonodynamic therapy (SDT) differs from conventional antibiotics primarily in its mechanism of action. While conventional antibiotics typically target specific bacterial structures or metabolic pathways, SDT generates reactive oxygen species that inflict multifaceted damage on bacterial cells, which significantly reduces the likelihood of developing drug resistance [E1]. Additionally, SDT offers enhanced tissue penetration compared to other physical sterilization methods like ultraviolet irradiation [E1].\",\n  \"confidence\": \"medium\",\n  \"used_evidence_ids\": [\"E1\"],\n  \"limitations\": \"The provided evidence does not detail the specific mechanisms of conventional antibiotics for a direct comparison, and the information is derived from a single source focused on SDT. Evidence E2 and E3 do not directly address the comparison between SDT and conventional antibiotics.\",\n  \"safety_note\": \"This answer is for medical literature exploration and is not medical advice.\"\n}"
        }
}
rag answer pipeline works
'''
