'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-05-15 15:40:31
LastEditTime: 2026-05-16 16:19:47
FilePath: /Agent_MedRag/src/agent_medrag/cli.py
Description: 
Main script of my agent.
'''
from __future__ import annotations

import argparse
from pathlib import Path
import json

from agent_medrag.ingestion.document_normalizer import normalize_pubmed_records
from  agent_medrag.ingestion.json_loader import load_pubmed_json
from agent_medrag.schemas import MedicalDocument

def write_documents_jsonl(
    documents:list[MedicalDocument],
    output_path:str | Path,
)->None:
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
    raw_records=load_pubmed_json(input_path)
    documents=normalize_pubmed_records(raw_records)
    
    write_documents_jsonl(documents,output_path)
    
    print(f'Loaded {len(raw_records)} raw records from {input_path}')
    print(f"Ingestion complete. Processed {len(documents)} documents. Output written to {output_path}")
    
def build_parser()->argparse.ArgumentParser:
    parser=argparse.ArgumentParser(
        prog='medrag',
        description="Agent_MedRag: A medical retrieval-augmented generation agent. This CLI allows you to ingest raw PubMed JSON data and output normalized documents in JSONL format."
    )
    
    subparsers=parser.add_subparsers(dest='command',required=True)
    
    ingest_parser=subparsers.add_parser(
        'ingest',
        help='"Normalize raw PubMed JSON into documents.jsonl.'
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
    
    return parser

def main()->None:
    parser=build_parser()
    args=parser.parse_args()
    
    if args.command=='ingest':
        run_ingest(args.input,args.output)
    else:
        print(f"Unknown command: {args.command}")
        
if __name__=='__main__':
    main()