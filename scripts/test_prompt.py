from agent_medrag.generation.prompts import build_answer_prompt
from agent_medrag.schemas import RetrievedEvidence


def main()->None:
    evidence=[
        RetrievedEvidence(
            chunk_id='pubmed_000008_chunk_000',
            doc_id='punmed_000008',
            title="Antibacterial sonodynamic nanomedicine",
            text="Sonodynamic therapy generates reactive oxygen species.",
            retrieval_score=0.2826,
            pub_date="2025",
        ),
        RetrievedEvidence(
            chunk_id="pubmed_000008_chunk_002",
            doc_id="pubmed_000008",
            title="Antibacterial sonodynamic nanomedicine",
            text="SDT may support infected wound healing and deep tissue inflammation treatment.",
            retrieval_score=0.4596,
            pub_date="2025",
        ),
    ]

    prompt=build_answer_prompt(
        "How does sonodynamic therapy treat bacterial infections?",
        evidence,
    )
    
    print(prompt)
    
if __name__=='__main__':
    main()