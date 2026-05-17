'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-05-16 14:54:00
LastEditTime: 2026-05-16 15:06:48
FilePath: /Agent_MedRag/src/agent_medrag/ingestion/json_loader.py
Description: 
Load raw data. Only read and validate the input JSON container shape, do not do any other prepocessing operations.
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

    # Resolve the input path first.
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON file does not exist: {input_path}")

    try:
        # Read the file as raw JSON.
        with input_path.open('r',encoding='utf-8') as file:
            data=json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {input_path}") from exc

    # Expect a top-level array of records.
    if not isinstance(data,list):
        raise ValueError(
            f"Expected top-level JSON array in {input_path}, "
            f"but got {type(data).__name__}"
        )

    for i,record in enumerate(data):
        # Each item should be a JSON object.
        if not isinstance(record,dict):
            raise ValueError(
                f"Expected every PubMed record to be an object, "
                f"but record {i} is {type(record).__name__}"
            )

    # Return the validated raw records unchanged.
    return data