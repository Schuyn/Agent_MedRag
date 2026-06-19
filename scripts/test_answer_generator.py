from agent_medrag.generation.answer_generator import AnswerGenerator
from agent_medrag.schemas import RetrievedEvidence

class FakeLLMProvider:
    def generate(self,prompt:str)->str:
        return """
        {
          "answer": "Sonodynamic therapy can damage bacteria by generating reactive oxygen species [E1].",
          "confidence": "high",
          "used_evidence_ids": ["E1"],
          "limitations": null,
          "safety_note": "This answer is for medical literature exploration and is not medical advice."
        }
        """
        
def main()->None:
    evidence=[
        RetrievedEvidence(
            chunk_id="pubmed_000008_chunk_000",
            doc_id="pubmed_000008",
            title="Antibacterial sonodynamic nanomedicine",
            text="Sonodynamic therapy generates reactive oxygen species.",
            retrieval_score=0.2826,
            pub_date="2025",
        ),
        RetrievedEvidence(
            chunk_id="pubmed_000008_chunk_002",
            doc_id="pubmed_000008",
            title="Antibacterial sonodynamic nanomedicine",
            text="SDT may support infected wound healing.",
            retrieval_score=0.4596,
            pub_date="2025",
        ),
    ]
    
    generator=AnswerGenerator(FakeLLMProvider())
    answer=generator.generate(
        "How does sonodynamic therapy treat bacterial infections?",
        evidence,
    )
    
    print(answer.to_json_dict())
    
    assert answer.confidence=='high'
    assert len(answer.citations)==1
    assert answer.citations[0].chunk_id=='pubmed_000008_chunk_000'
    assert answer.metadata['used_evidence_ids']==['E1']
    assert answer.retrieval_summary.retrieved_count==2
    assert answer.retrieval_summary.used_evidence_count==1
    
    print("answer generator works")

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
