'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-05-16 14:54:00
LastEditTime: 2026-05-16 15:06:48
FilePath: /Agent_MedRag/src/agent_medrag/ingestion/json_loader.py
Description:
Load and validate the raw PubMed JSON container. Only checks that the input is a
well-formed JSON array of objects — no normalization, filtering, or enrichment.
Those concerns belong to document_normalizer.py.
'''
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

def load_pubmed_json(path: str|Path)->list[dict[str,Any]]:
    '''
    Load raw PubMed article records from a JSON file.

    Expected input format:
        [
            {
                "article_title": "...",
                "article_abstract": "...",
                "pub_date": {
                    "year": "2025",
                    "month": "04",
                    "day": "03"
                }
            },
            ...
        ]
    
    Returns:
        A list of raw article dictionaries.
        
    Raises:
        FileNotFoundError: if the input file does not exist.
        ValueError: if the JSON is invalid or has an unsupported shape.
    '''
    input_path=Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON file does not exist: {input_path}")

    try:
        with input_path.open('r',encoding='utf-8') as file:
            data=json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {input_path}") from exc

    # Top-level must be a JSON array of article objects.
    if not isinstance(data,list):
        raise ValueError(
            f"Expected top-level JSON array in {input_path}, "
            f"but got {type(data).__name__}"
        )

    for i,record in enumerate(data):
        if not isinstance(record,dict):
            raise ValueError(
                f"Expected every PubMed record to be an object, "
                f"but record {i} is {type(record).__name__}"
            )

    # Return validated raw records — normalization happens downstream.
    return data