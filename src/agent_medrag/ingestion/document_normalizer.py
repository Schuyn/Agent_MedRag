'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-05-15 15:46:41
LastEditTime: 2026-05-16 21:26:32
FilePath: /Agent_MedRag/src/agent_medrag/ingestion/document_normalizer.py
Description: 
Normalize and filter raw PubMed article records into a unified
`MedicalDocument` format consumed by the ingestion pipeline.

Responsibilities:
- Normalize publication dates to an ISO-like YYYY-MM-DD string when
    possible.
- Filter out low-value records such as corrigenda, errata, retractions,
    and other non-content notices.
- Trim and validate title/abstract text, generate a stable `doc_id`
    (format `pubmed_{index:06d}`), and attach metadata (`source`,
    `pub_date`, `raw_index`) for traceability.

Provides both a single-record normalizer and a bulk `normalize_pubmed_records`
helper that returns a list of `MedicalDocument` instances.
'''
from __future__ import annotations
from typing import Any

from agent_medrag.schemas import MedicalDocument

LOW_VALUE_TITLE_PREFIXES = (
    "corrigendum",
    "erratum",
    "retraction",
    "retracted",
    "withdrawn",
)

def is_low_value_record(title:str,abstract:str)->bool:
    normalized_title=title.strip().lower().rstrip('.')
    normalized_abstract=abstract.strip().lower()
    
    if normalized_title in LOW_VALUE_TITLE_PREFIXES:
        return True
    
    if normalized_abstract.startswith("[this corrects the article doi:"):
        return True
    
    if 'retracted' in normalized_title or 'retracted' in normalized_abstract:
        return True
    
    return False

def format_pub_date(pub_date:dict[str,Any] | None)->str | None:
    # If pub_date is not a dict, we cannot format it.
    if not isinstance(pub_date,dict):
        return None

    # Extract possible year/month/day components (may be missing).
    year=pub_date.get('year')
    month=pub_date.get('month')
    day=pub_date.get('day')

    # Year is required for a valid date; otherwise return None.
    if not year:
        return None

    # Normalize numeric parts and provide defaults for month/day when missing.
    year_text=str(year).zfill(4)
    month_text=str(month).zfill(2) if month else '01'
    day_text=str(day).zfill(2) if day else '01'

    # Return ISO-like date string YYYY-MM-DD.
    return f'{year_text}-{month_text}-{day_text}'
    
def normalize_pubmed_record(
    record:dict[str,Any],
    raw_index:int,
)->MedicalDocument | None:
    # Pull fields from the raw record, using empty strings as fallbacks.
    title=record.get('article_title','')
    abstract=record.get('article_abstract','')
    pub_date=record.get('pub_date')

    # Ensure title/abstract are strings and strip surrounding whitespace.
    title=title.strip() if isinstance(title,str) else ''
    abstract=abstract.strip() if isinstance(abstract,str) else ''

    # Skip records missing essential text fields.
    if not title or not abstract:
        return None
    
    if is_low_value_record(title,abstract):
        return None

    # Build and return a MedicalDocument dataclass instance.
    return MedicalDocument(
        doc_id=f'pubmed_{raw_index:06d}',
        title=title,
        text=abstract,
        metadata={
            # Source identifier to allow multi-source ingestion later.
            'source':'pubmed',  
            # Normalized publication date (or None).
            'pub_date':format_pub_date(pub_date),
            # Original index in the raw list for traceability.
            'raw_index':raw_index, 
        },
    )
    
def normalize_pubmed_records(
    records:list[dict[str,Any]],
)->list[MedicalDocument]:
    # Normalize a list of raw records into MedicalDocument instances.
    documents:list[MedicalDocument]=[]

    for raw_index,record in enumerate(records):
        # Reuse the single-record normalizer above.
        document=normalize_pubmed_record(record,raw_index)
        if document is not None:
            documents.append(document)

    return documents