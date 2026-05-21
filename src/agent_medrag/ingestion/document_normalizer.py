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
    """Flag corrigenda, errata, retractions, and other non-content PubMed notices."""
    normalized_title=title.strip().lower().rstrip('.')
    normalized_abstract=abstract.strip().lower()

    if normalized_title in LOW_VALUE_TITLE_PREFIXES:
        return True

    # Catch auto-generated correction notices (e.g. "[This corrects the article DOI: ...]").
    if normalized_abstract.startswith("[this corrects the article doi:"):
        return True

    if 'retracted' in normalized_title or 'retracted' in normalized_abstract:
        return True

    return False

def format_pub_date(pub_date:dict[str,Any] | None)->str | None:
    """Normalize a PubMed pub_date dict into an ISO-like 'YYYY-MM-DD' string."""
    if not isinstance(pub_date,dict):
        return None

    year=pub_date.get('year')
    month=pub_date.get('month')
    day=pub_date.get('day')

    # Year is required; month/day default to '01' when absent.
    if not year:
        return None

    year_text=str(year).zfill(4)
    month_text=str(month).zfill(2) if month else '01'
    day_text=str(day).zfill(2) if day else '01'

    return f'{year_text}-{month_text}-{day_text}'
    
def normalize_pubmed_record(
    record:dict[str,Any],
    raw_index:int,
)->MedicalDocument | None:
    """Convert one raw PubMed dict into a MedicalDocument, or return None if filtered out."""
    title=record.get('article_title','')
    abstract=record.get('article_abstract','')
    pub_date=record.get('pub_date')

    title=title.strip() if isinstance(title,str) else ''
    abstract=abstract.strip() if isinstance(abstract,str) else ''

    # Drop records with missing text fields.
    if not title or not abstract:
        return None

    if is_low_value_record(title,abstract):
        return None

    return MedicalDocument(
        doc_id=f'pubmed_{raw_index:06d}',
        title=title,
        text=abstract,
        metadata={
            'source':'pubmed',
            # source allows multi-source ingestion (e.g. ArXiv, clinical trials) later.
            'pub_date':format_pub_date(pub_date),
            # raw_index preserves position in the original JSON array for traceability.
            'raw_index':raw_index,
        },
    )
    
def normalize_pubmed_records(
    records:list[dict[str,Any]],
)->list[MedicalDocument]:
    """Normalize a batch of raw PubMed records, silently dropping any that fail filtering."""
    documents:list[MedicalDocument]=[]

    for raw_index,record in enumerate(records):
        # Reuse the single-record normalizer above.
        document=normalize_pubmed_record(record,raw_index)
        if document is not None:
            documents.append(document)

    return documents