# AgentMedRAG Portfolio Development Plan

## Summary

This plan defines a 3-4 week implementation path for turning AgentMedRAG into a portfolio-ready project for AI/ML Engineer roles.

The project should emphasize:

- A runnable medical RAG pipeline.
- Clean Python package structure.
- Citation-grounded answers.
- Reranking and reproducible evaluation.
- Medical safety boundaries.
- A strong README and GitHub presentation.

The first portfolio version should focus on GitHub and CLI usage. Web UI, FastAPI, PubMed live search, LangGraph, multi-turn memory, and ColBERT production integration are intentionally out of scope for the first version.

## Phase 1: Project Skeleton and Ingestion

Estimated time: 2-3 days

Goal: establish a maintainable Python package and run the first data pipeline.

Implementation:

- Define core schemas:
  - `MedicalDocument`
  - `DocumentChunk`
  - `RetrievedEvidence`
  - `Citation`
  - `MedicalAnswer`
- Read `data/raw/pubmed_article.json`.
- Clean and normalize PubMed article records.
- Filter records with empty titles or abstracts.
- Write normalized records to `data/processed/documents.jsonl`.
- Add the first CLI command:

```bash
medrag ingest --input data/raw/pubmed_article.json --output data/processed/documents.jsonl
```

Acceptance criteria:

- `documents.jsonl` is generated successfully.
- Every output record has a stable `doc_id`, `title`, `text`, and `metadata`.
- Invalid or empty raw records are skipped deterministically.

## Phase 2: Chunking and Vector Index

Estimated time: 4-5 days

Goal: convert normalized documents into a searchable local corpus.

Implementation:

- Implement document chunking.
- Use BGE embeddings, with either `BAAI/bge-large-en-v1.5` or a lighter BGE model as the default.
- Use one vector store for v1, preferably Chroma.
- Persist the index under `indexes/`.
- Add the index build command:

```bash
medrag build-index --config configs/default.yaml
```

Acceptance criteria:

- Chunks can be generated from `documents.jsonl`.
- A local vector index can be built and persisted.
- README explains how to build the index.

## Phase 3: Baseline RAG Ask

Estimated time: 5-6 days

Goal: implement a fixed RAG chain before adding more agentic behavior.

Implementation:

- Implement retriever.
- Implement LLM provider abstraction.
- Use OpenAI as the first default provider to reduce local environment complexity.
- Implement answer generation.
- Return structured output with:
  - `answer`
  - `citations`
  - `limitations`
  - `safety_note`
- Add the ask command:

```bash
medrag ask "How does sonodynamic therapy differ from conventional antibiotics?"
```

Acceptance criteria:

- The system retrieves evidence from the local index.
- The generated answer is grounded in retrieved evidence.
- The answer includes citation information.
- If evidence is missing, the system clearly states that evidence is insufficient.

## Phase 4: Reranker, Citation Quality, and Safety

Estimated time: 5-6 days

Goal: make the project read as a medical literature RAG system rather than a generic RAG demo.

Implementation:

- Add a BGE reranker wrapper.
- Support configurable `rerank_top_k`.
- Include the following fields in each citation:
  - `title`
  - `pub_date`
  - `evidence`
  - `doc_id`
  - `chunk_id`
- Add medical safety guardrails.
- Restrict answers for high-risk questions involving diagnosis, dosage, personalized treatment, emergency symptoms, or medication changes.

Acceptance criteria:

- `medrag ask` runs retrieve -> rerank -> answer by default.
- Every answer includes a medical safety note.
- High-risk medical questions do not receive personalized medical advice.
- Citations are traceable to specific chunks.

## Phase 5: Evaluation and Experiment Report

Estimated time: 5-7 days

Goal: make the project valuable as an AI/ML engineering portfolio project, not only a demo.

Implementation:

- Standardize the QA evaluation input format.
- Add the evaluation command:

```bash
medrag evaluate --qa data/qa/generated_qa_pairs100.json
```

- Write evaluation outputs to `reports/<run_id>/`:
  - `metrics.json`
  - `per_question_results.jsonl`
  - `summary.md`
- Compare at least two configurations:
  - without reranker
  - with BGE reranker

Acceptance criteria:

- Evaluation runs are reproducible.
- README includes a compact results table.
- The report explains how reranking affects retrieval or citation quality.

## Phase 6: Portfolio Polish

Estimated time: 3-5 days

Goal: make the repository understandable and credible for recruiters and engineering interviewers.

Implementation:

- Rewrite README.
- Add an architecture diagram.
- Add Quickstart instructions.
- Add example question and answer output.
- Add a medical safety disclaimer.
- Add evaluation summary.
- Add `.env.example`.
- Add basic unit tests and a smoke test.

README should highlight:

- Project positioning: a medical literature assistant, not medical advice.
- Tech stack: PubMed, RAG, BGE embeddings, Chroma, reranker, OpenAI or Mistral, evaluation.
- System architecture.
- CLI usage.
- Example output.
- Evaluation results.
- Future improvements.

Acceptance criteria:

- A new user can run ingest, build-index, and ask by following the README.
- The GitHub project can be understood in about one minute.
- Resume bullets can map naturally to the implemented system.

## Final Portfolio Version

The 3-4 week version should include at least:

```bash
medrag ingest
medrag build-index
medrag ask
medrag evaluate
```

Core capabilities:

- Local PubMed abstract corpus.
- Chroma vector index.
- BGE embeddings.
- BGE reranker.
- Structured citation answer.
- Medical safety note.
- Evaluation report.
- High-quality README.

Deferred from v1:

- Web UI.
- FastAPI service.
- PubMed live search.
- LangGraph complex agent.
- Multi-turn memory.
- ColBERT production integration.
- Multiple vector database switching.

## Assumptions

- Target role: AI/ML Engineer.
- Primary showcase surface: GitHub plus CLI.
- Target timeline: 3-4 weeks.
- Existing notebooks are background references only, not trusted implementation sources.
- `data/raw/pubmed_article.json` already exists, so v1 does not rewrite the PubMed scraping script.
- The first version prioritizes completeness, reproducibility, and clarity over feature breadth.
