from agent_medrag.generation.prompts import build_answer_prompt
from agent_medrag.schemas import RetrievedEvidence

evidence=[
    RetrievedEvidence(
        chunk_id='pubmed_000008_chunk_000',
        doc_id='punmed_000008',
        title="Antibacterial sonodynamic nanomedicine",
        text="Sonodynamic therapy generates reactive oxygen species.",
        retrieval_score=0.2826,
        pub_date="2025",
    )
]

print(build_answer_prompt('How does SDT kill bacteria?',evidence))