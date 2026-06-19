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
    