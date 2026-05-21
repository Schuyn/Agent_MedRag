'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-05-15 15:40:31
LastEditTime: 2026-05-20 19:17:57
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

def build_parser()->argparse.ArgumentParser:
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
    else:
        print(f"Unknown command: {args.command}")
        
if __name__=='__main__':
    main()