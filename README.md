# AgentMedRAG

A medical literature question-answering agent built on PubMed abstracts with retrieval-augmented generation, reranking, citation verification, and reproducible RAG evaluation.

**Intended for medical literature exploration and educational use — not for diagnosis or personalized medical advice.**

## Project Status

**Stage 1 (complete):** Project structure, ingestion pipeline, document normalization, chunking, and CLI foundation.

**Stage 2 (in progress):** BGE embeddings and persistent Chroma index construction are implemented; reproducible validation is still pending.

| Module | Status |
|---|---|
| `ingestion/` — JSON loader, document normalizer, PubMed client stub | Done |
| `indexing/` — Chunker, BGE embeddings, Chroma store, index builder | Implemented |
| `schemas.py` — `RawPubMedArticle`, `MedicalDocument`, `MedicalChunk` | Done |
| `config.py` + `configs/default.yaml` — Centralized configuration | Done |
| `cli.py` — `ingest`, `chunk`, and `build-index` subcommands | Implemented |
| Retrieval / embedding / vector store | Index build implemented; retrieval planned |
| Reranking (BGE) | Planned |
| LLM generation with citations | Planned |
| Medical safety guardrails | Planned |
| RAGAS evaluation pipeline | Planned |
| API / UI | Planned |

See [docs/PROJECT_REFACTOR_PLAN.md](docs/PROJECT_REFACTOR_PLAN.md) for the full architecture and migration plan.

## Project Structure

```
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
  scripts/
    ingested_pubmed.py    # Standalone ingestion script (stub)
  notebooks/
    legacy/               # Original course project notebooks
    demos/                 # Runnable demo notebooks
  docs/
    PROJECT_REFACTOR_PLAN.md
    PORTFOLIO_DEVELOPMENT_PLAN.md
```

## Quickstart

### Install

```bash
pip install -e .
```

### Ingest PubMed data

```bash
agent-medrag ingest --input data/raw/pubmed_article.json --output data/processed/documents.jsonl
```

This loads raw PubMed JSON, normalizes each article into a `MedicalDocument`, and writes the result as JSONL.

### Chunk normalized documents

```bash
agent-medrag chunk --input data/processed/documents.jsonl --output data/processed/chunks.jsonl --chunk-size 512 --chunk-overlap 80
```

This creates retrieval-sized chunks while preserving their parent document metadata.

### Build the vector index

```bash
agent-medrag build-index --input data/processed/chunks.jsonl --index indexes/chroma --collection pubmed_articles --model BAAI/bge-small-en-v1.5
```

This embeds each chunk and persists a Chroma collection under `indexes/chroma`.

After changing chunking settings or the embedding model, rebuild the target collection so stale chunks are not retained:

```bash
agent-medrag build-index --input data/processed/chunks.jsonl --index indexes/chroma --collection pubmed_articles --model BAAI/bge-small-en-v1.5 --rebuild
```

`--rebuild` removes and recreates only the selected collection before indexing. It does not remove other index directories or experiment collections.

### Configuration

Copy and edit the relevant config file:

```bash
cp configs/default.yaml configs/local.yaml
```

Key settings in `configs/default.yaml`:

| Section | Purpose |
|---|---|
| `data` | Paths to raw/processed/chunks files |
| `chunking` | Chunk size, overlap, minimum chars |
| `embedding` | HuggingFace model selection (BGE) |
| `vector_store` | Chroma persist directory and collection |
| `retrieval` | Top-k, score threshold |
| `reranking` | BGE reranker model and top-k |
| `llm` | Provider (OpenAI), model, temperature |
| `safety` | Medical disclaimer and guardrail toggles |

### Environment

```bash
cp .env.example .env
# Fill in API keys:
#   OPENAI_API_KEY=...
#   HF_TOKEN=...
#   ENTREZ_EMAIL=...
```

## Data Schema

### Input (raw PubMed JSON)

```json
{
  "article_title": "...",
  "article_abstract": "...",
  "pub_date": { "year": "2025", "month": "04", "day": "03" }
}
```

### Output (normalized document)

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

All answers include a safety disclaimer. For details, see the safety design in [docs/PROJECT_REFACTOR_PLAN.md](docs/PROJECT_REFACTOR_PLAN.md#11-医学安全设计).

## License

MIT — see [LICENSE](LICENSE).
