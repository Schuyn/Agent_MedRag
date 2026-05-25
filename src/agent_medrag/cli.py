'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-05-15 15:40:31
LastEditTime: 2026-05-25 18:25:45
FilePath: /Agent_MedRag/src/agent_medrag/cli.py
Description:
CLI entry point for Agent_MedRag. Exposes subcommands for the full pipeline —
ingest (normalize raw PubMed JSON to documents.jsonl), chunk (split documents
into retrieval-ready chunks), ask (single-question RAG query), evaluate (batch
RAGAS benchmark), and serve (FastAPI/Streamlit UI). Each subcommand maps to
a dedicated handler below.
'''
from __future__ import annotations

import argparse
from pathlib import Path
import json

from agent_medrag.ingestion.document_normalizer import normalize_pubmed_records
from agent_medrag.ingestion.json_loader import load_pubmed_json
from agent_medrag.indexing.chunker import chunk_documents
from agent_medrag.indexing.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbedding,
)
from agent_medrag.indexing.vector_store import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIRECTORY,
    ChromaVectorStore,
)
from agent_medrag.indexing.index_builder import build_index,load_chunks_jsonl
from agent_medrag.schemas import MedicalDocument,MedicalChunk

def write_documents_jsonl(
    documents:list[MedicalDocument],
    output_path:str | Path,
)->None:
    """Serialize each MedicalDocument as a single-line JSON record (JSONL, UTF-8)."""
    output_path=Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)

    with output_path.open('w',encoding='utf-8') as file:
        for document in documents:
            json_line=json.dumps(document.to_json_dict(),ensure_ascii=False)
            file.write(json_line+'\n')

def run_ingest(
    input_path:str | Path,
    output_path:str | Path,
)->None:
    """Load raw PubMed JSON, normalize into MedicalDocument instances, and write JSONL output."""
    raw_records=load_pubmed_json(input_path)
    documents=normalize_pubmed_records(raw_records)

    write_documents_jsonl(documents,output_path)

    print(f'Loaded {len(raw_records)} raw records from {input_path}')
    print(f"Ingestion complete. Processed {len(documents)} documents. Output written to {output_path}")
         
def load_documents_jsonl(path:str | Path)->list[MedicalDocument]:
    """Read a documents.jsonl file back into MedicalDocument instances."""
    documents:list[MedicalDocument]=[]
    
    with Path(path).open('r',encoding='utf-8') as file:
        for line in file:
            if not line.strip():
                continue  # Skip empty lines
                
            item=json.loads(line)
            documents.append(
                MedicalDocument(
                    doc_id=item['doc_id'],
                    title=item['title'],
                    text=item['text'],
                    metadata=item.get('metadata',{})
                )
            )
            
    return documents    

def write_chunks_jsonl(chunks:list[MedicalChunk],output_path:str | Path)->None:
    """Serialize each MedicalChunk as a single-line JSON record (JSONL, UTF-8)."""
    output_path=Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    
    with output_path.open('w',encoding='utf-8') as file:
        for chunk in chunks:
            file.write(json.dumps(chunk.to_json_dict(),ensure_ascii=False)+'\n')
            
            
def run_chunk(
    input_path:str | Path,
    output_path:str | Path,
    chunk_size:int,
    chunk_overlap:int,
)->None:
    """Load normalized documents, split into overlapping chunks, and write chunks.jsonl."""
    documents=load_documents_jsonl(input_path)
    chunks=chunk_documents(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    write_chunks_jsonl(chunks,output_path)
    
    print(f"Loaded {len(documents)} documents from {input_path}")
    print(f"Created {len(chunks)} chunks")
    print(f"Chunks written to {output_path}")   

def run_build_index(
    input_path:str | Path,
    persist_directory:str | Path,
    collection_name:str,
    model_name:str,
    batch_size:int,
    device:str | None,
    rebuild:bool=False,
)->None:
    chunks=load_chunks_jsonl(input_path)

    embedding_model=SentenceTransformerEmbedding(
        model_name=model_name,
        normalize_embeddings=True,
        device=device,
    )

    vector_store=ChromaVectorStore(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    if rebuild:
        vector_store.reset_collection()
        print(f"Rebuilding collection '{collection_name}' in {persist_directory}")

    indexed_count=build_index(
        chunks=chunks,
        embedding_model=embedding_model,
        vector_storage=vector_store,
        batch_size=batch_size,
    )

    print(f"Loaded {len(chunks)} chunks from {input_path}")
    print(f"Indexed {indexed_count} chunks")
    print(f"Vector store contains {vector_store.count()} records")
    print(f"Index written to {persist_directory}")

def build_parser()->argparse.ArgumentParser:
    """Create and return a configured ArgumentParser for the CLI.

    Using a command-line argument parser (e.g. argparse or click) is preferred
    over interactive prompts input() for tools intended to be used in
    automation and production environments. Benefits include:

    - Non-interactive operation: can run in CI, cron jobs, containers, and
        other automated pipelines without human interaction.
    - Composability: supports flags, options and subcommands; integrates well
        with shell pipelines and scripting.
    - Self-documenting help: automatically generates `--help` output for users.
    - Type validation and defaults: argument types (int/float/path) and default
        values are enforced by the parser.
    - Testability: CLI behavior can be tested by supplying argv; avoids
        mocking stdin used by `input()`.
    - Maintainability: adding new options is backward-compatible and clearer
        than changing an interactive flow.

    Returns:
            argparse.ArgumentParser: parser configured with the available subcommands.
    """
    parser=argparse.ArgumentParser(
        prog='medrag',
        description=(
            "Agent_MedRag: A medical retrieval-augmented generation agent. "
            "This CLI supports PubMed ingestion and document chunking."
        ),
    )
    
    subparsers=parser.add_subparsers(dest='command',required=True)
    
    ingest_parser=subparsers.add_parser(
        'ingest',
        help='Normalize raw PubMed JSON into documents.jsonl.'
    )
    
    ingest_parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the input JSON file containing raw PubMed records.'
    )
    
    ingest_parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to the output JSONL file where normalized documents will be written.'
    )
    
    chunk_parser=subparsers.add_parser(
        'chunk',
        help="Split normalized documents.jsonl into chunks.jsonl.",
    )
    
    chunk_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input documents.jsonl file.",
    )
    
    chunk_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the out chunks.jsonl file.",
    )
    
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum number of characters per chunk.",
    )
    
    chunk_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Number of overlapping characters between adjacent chunks.",
    )
    
    build_index_parser=subparsers.add_parser(
        "build-index",
        help="Embed chunks and store them in a persistent Chroma index.",
    )

    build_index_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input chunks.jsonl file.",
    )

    build_index_parser.add_argument(
        "--index",
        type=str,
        default=DEFAULT_PERSIST_DIRECTORY,
        help="Directory where the persistent Chroma index will be stored.",
    )

    build_index_parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help="Name of the Chroma collection.",
    )

    build_index_parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Delete and recreate the target collection before indexing. "
            "Use this after changing chunks or the embedding model."
        ),
    )
    
    build_index_parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence-transformers embedding model name.",
    )

    build_index_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of chunks embedded and written per batch.",
    )

    build_index_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Embedding device, for example cpu, cuda, or mps."
    )

    return parser
    # Return configured argparse parser. Subcommands: ingest, chunk.

def main()->None:
    """Parse CLI arguments and dispatch to the matching subcommand handler."""
    parser=build_parser()
    args=parser.parse_args()

    if args.command=='ingest':
        run_ingest(args.input,args.output)

    elif args.command=='chunk':
        run_chunk(
            input_path=args.input,
            output_path=args.output,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

    elif args.command=='build-index':
        run_build_index(
            input_path=args.input,
            persist_directory=args.index,
            collection_name=args.collection,
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device,
            rebuild=args.rebuild,
        )

    else:
        print(f"Unknown command: {args.command}")
        
if __name__=='__main__':
    main()
