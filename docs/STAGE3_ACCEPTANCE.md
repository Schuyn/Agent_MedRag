# Stage 3 Acceptance Record

Stage 3 scope: baseline RAG ask path from a persisted local Chroma index to a structured, citation-grounded answer.

## Accepted Capabilities

- `agent-medrag ask` is wired into the project CLI.
- `rag_pipeline.py` builds the fixed baseline chain from `configs/default.yaml`.
- The pipeline loads the configured embedding model and local Chroma collection.
- The retriever returns structured evidence with `chunk_id`, `doc_id`, title, text, score, and publication date.
- The DeepSeek provider generates a JSON answer from retrieved evidence.
- The answer generator maps used evidence IDs such as `E1` back to structured citations.
- The final response includes `answer`, `confidence`, `citations`, `limitations`, `safety_note`, `retrieval_summary`, and metadata.

## Validation Command

```powershell
agent-medrag ask "How does sonodynamic therapy differ from conventional antibiotics?" --config configs/default.yaml --top-k 3
```

## Validation Result

The command completed successfully and returned a structured JSON response.

Key fields from the accepted run:

```json
{
  "question": "How does sonodynamic therapy differ from conventional antibiotics?",
  "confidence": "medium",
  "retrieval_summary": {
    "query": "How does sonodynamic therapy differ from conventional antibiotics?",
    "top_k": 3,
    "retrieved_count": 3,
    "used_evidence_count": 1
  },
  "metadata": {
    "used_evidence_ids": ["E1"]
  }
}
```

Accepted citation trace:

```json
{
  "title": "Antibacterial sonodynamic nanomedicine: mechanism, category, and applications.",
  "pub_date": "2025-03-25",
  "doc_id": "pubmed_000008",
  "chunk_id": "pubmed_000008_chunk_000",
  "url": null
}
```

Accepted safety note:

```text
This answer is for medical literature exploration and is not medical advice.
```

## Acceptance Criteria

- Pass: CLI command dispatches through `agent-medrag ask`.
- Pass: `--config configs/default.yaml` is accepted.
- Pass: `--top-k 3` is accepted and reflected in `retrieval_summary.top_k`.
- Pass: The system retrieves three evidence chunks from the local Chroma index.
- Pass: The answer cites retrieved evidence using `[E1]`.
- Pass: Citation metadata includes `doc_id` and `chunk_id`.
- Pass: The answer includes limitations.
- Pass: The answer includes the medical safety note.

## Known Limitations After Stage 3

- Reranking is not yet part of the accepted ask path.
- Advanced medical safety classification is not yet implemented.
- The DeepSeek API key is required through `DEEPSEEK_API_KEY`; no key should be committed to the repository.
- `metadata.raw_llm_output` is useful for debugging but may be too verbose for portfolio-facing output.
- HuggingFace may warn about unauthenticated downloads if `HF_TOKEN` is not set; this does not block the accepted local ask path once model assets are available.

## Conclusion

Stage 3 is accepted as a working baseline RAG ask implementation. The next project stage should focus on reranking, citation quality, and stronger safety handling rather than more baseline pipeline wiring.
