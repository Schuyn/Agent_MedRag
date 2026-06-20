# AgentMedRAG

A medical literature question-answering agent built on PubMed abstracts with retrieval-augmented generation, reranking, citation verification, and reproducible RAG evaluation.

**Intended for medical literature exploration and educational use, not for diagnosis or personalized medical advice.**

## Project Status

**Stage 1 (complete):** Project structure, ingestion pipeline, document normalization, chunking, and CLI foundation.

**Stage 2 (complete):** Config-driven chunking and Chroma index construction are implemented and validated locally.

Validation result:

```text
Loaded 6459 chunks from data/processed/chunks.jsonl
Indexed 6459 chunks
Vector store contains 6459 records
Index written to indexes/chroma
```

**Stage 3 (complete):** Baseline RAG retrieval, DeepSeek-backed answer generation, and the `agent-medrag ask` command are implemented and validated.

| Module | Status |
|---|---|
| `ingestion/` - JSON loader, document normalizer, PubMed client stub | Done |
| `indexing/` - Chunker, BGE embeddings, Chroma store, index builder | Done |
| `config.py` + `configs/default.yaml` - Centralized Stage 2 configuration | Done |
| `cli.py` - `ingest`, `chunk`, `build-index`, and `ask` subcommands | Done |
| `rag_pipeline.py` - Config-driven retrieve -> answer orchestration | Done |
| Retrieval from local Chroma index | Done |
| DeepSeek LLM provider | Done |
| Evidence-grounded answer generation with citations | Done |
| Baseline medical safety note | Done |
| Reranking (BGE) | Planned |
| Advanced medical safety guardrails | Planned |
| RAGAS evaluation pipeline | Planned |
| API / UI | Planned |

See [docs/PROJECT_REFACTOR_PLAN.md](docs/PROJECT_REFACTOR_PLAN.md) for the full architecture and migration plan.

## Project Structure

```text
Agent_MedRag/
  pyproject.toml
  configs/
    default.yaml          # Main configuration
    eval.yaml             # Evaluation configuration
    openai.yaml           # OpenAI provider configuration
  data/
    raw/                  # Raw PubMed JSON input
    processed/            # Normalized documents.jsonl, chunks.jsonl
  src/agent_medrag/
    __init__.py
    cli.py                # CLI entry point
    config.py             # Configuration loader
    rag_pipeline.py       # Baseline RAG pipeline orchestration
    schemas.py            # Data models
    ingestion/
      __init__.py
      json_loader.py      # Parse raw PubMed JSON
      document_normalizer.py  # Normalize to MedicalDocument
      pubmed_client.py    # PubMed Entrez API downloader (stub)
    indexing/
      __init__.py
      chunker.py          # Sliding-window text chunking
      embeddings.py       # Sentence-transformers embedding backend
      vector_store.py     # Persistent Chroma collection wrapper
      index_builder.py    # Batch index construction from chunks
    retrieval/
      retriever.py        # Local Chroma retriever
    generation/
      llm_provider.py     # DeepSeek provider abstraction
      prompts.py          # Evidence-grounded answer prompt
      answer_generator.py # Structured answer generation
  scripts/
    ingested_pubmed.py    # Standalone ingestion script (stub)
    test_rag_answer.py    # Stage 3 RAG pipeline smoke test
  notebooks/
    legacy/               # Original course project notebooks
    demos/                # Runnable demo notebooks
  docs/
    PROJECT_REFACTOR_PLAN.md
    PORTFOLIO_DEVELOPMENT_PLAN.md
```

## Quickstart

### Install

```bash
pip install -e .
```

### Ingest PubMed Data

```bash
agent-medrag ingest --config configs/default.yaml
```

This reads `data.raw_pubmed_path`, normalizes PubMed records into `MedicalDocument` objects, and writes `data.processed_documents_path`.

### Chunk Normalized Documents

```bash
agent-medrag chunk --config configs/default.yaml
```

This reads `data.processed_documents_path`, writes `data.chunks_path`, and uses `chunking.chunk_size` plus `chunking.chunk_overlap`.

### Build The Vector Index

```bash
agent-medrag build-index --config configs/default.yaml --rebuild
```

This reads `data.chunks_path`, embeds chunks with `embedding.model_name`, and persists the configured Chroma collection under `vector_store.persist_dir`.

Use `--rebuild` when chunking settings or the embedding model changes. It resets only the selected Chroma collection before indexing, so stale chunks are not retained.

CLI flags can override config values when needed:

```bash
agent-medrag build-index --config configs/default.yaml --model BAAI/bge-large-en-v1.5 --device cuda --rebuild
```

Generated Chroma indexes under `indexes/` are local build artifacts and are not committed to Git. Rebuild them with `agent-medrag build-index --config configs/default.yaml --rebuild`.

### Ask A Question

Set `DEEPSEEK_API_KEY` in your shell or environment before running the ask command.

```bash
agent-medrag ask "How does sonodynamic therapy differ from conventional antibiotics?" --config configs/default.yaml --top-k 3
```

This loads the configured Chroma index, retrieves evidence chunks, calls the configured DeepSeek model, and prints a structured `MedicalAnswer` JSON object containing `answer`, `confidence`, `citations`, `limitations`, `safety_note`, `retrieval_summary`, and metadata.

### Configuration

Copy and edit the relevant config file:

```bash
cp configs/default.yaml configs/local.yaml
```

The current baseline uses these sections from `configs/default.yaml`:

| Section | Purpose |
|---|---|
| `data` | Paths to raw documents, normalized documents, and chunks |
| `chunking` | Chunk size, overlap, and minimum chunk length |
| `embedding` | HuggingFace model, device, batch size, and normalization |
| `vector_store` | Chroma persist directory and collection name |
| `retrieval` | Default top-k retrieval setting |
| `llm` | DeepSeek provider, model, temperature, token limit, and reasoning settings |

## Stage 3 Acceptance

Stage 3 turns the local Chroma index into a usable RAG question-answering path.

Validated command:

```bash
agent-medrag ask "How does sonodynamic therapy differ from conventional antibiotics?" --config configs/default.yaml --top-k 3
```

Validation result:

```text
retrieval_summary.retrieved_count = 3
retrieval_summary.used_evidence_count = 1
citations[0].doc_id = pubmed_000008
citations[0].chunk_id = pubmed_000008_chunk_000
safety_note = This answer is for medical literature exploration and is not medical advice.
```

Stage 3 acceptance criteria:

- The system retrieves evidence from the local Chroma index.
- Retrieved chunks include `chunk_id`, `doc_id`, title, text, and retrieval score.
- The `ask` command returns a grounded answer with citations.
- The answer includes limitations and a medical safety note.

See [docs/STAGE3_ACCEPTANCE.md](docs/STAGE3_ACCEPTANCE.md) for the full acceptance record.

## Next Stage: Reranking, Citation Quality, and Safety

Stage 4 will improve answer quality and safety beyond the fixed baseline chain:

- Add reranking to improve evidence ordering.
- Tighten citation quality checks.
- Add stronger safety handling for personalized medical advice, dosage, medication changes, and emergency symptoms.

### Environment

```bash
cp .env.example .env
# Fill in API keys:
#   DEEPSEEK_API_KEY=...
#   HF_TOKEN=...
#   ENTREZ_EMAIL=...
```

## Data Schema

### Input (Raw PubMed JSON)

```json
{
  "article_title": "...",
  "article_abstract": "...",
  "pub_date": { "year": "2025", "month": "04", "day": "03" }
}
```

### Output (Normalized Document)

```json
{
  "doc_id": "pubmed_000001",
  "title": "...",
  "text": "...",
  "metadata": {
    "source": "pubmed",
    "pub_date": "2025-04-03",
    "raw_index": 1
  }
}
```

## Medical Safety

This system is a **medical literature assistant**. It does not:

- Diagnose conditions
- Recommend personal treatments or drug dosages
- Replace a qualified clinician

All answers include a safety disclaimer. For details, see the safety design in [docs/PROJECT_REFACTOR_PLAN.md](docs/PROJECT_REFACTOR_PLAN.md).

## License

MIT - see [LICENSE](LICENSE).
