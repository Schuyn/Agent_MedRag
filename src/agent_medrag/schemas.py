'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-05-15 15:46:07
LastEditTime: 2026-05-20 18:31:08
FilePath: /Agent_MedRag/src/agent_medrag/schemas.py
Description:
Immutable dataclass schemas that define the shape of data at each pipeline stage —
from raw PubMed JSON records through normalized documents to retrieval-ready chunks.

Input  (RawPubMedArticle):
{
  "article_title": "...",
  "article_abstract": "...",
  "pub_date": { "year": "2025", "month": "04", "day": "03" }
}

Output (MedicalDocument):
{
  "doc_id": "pubmed_000001",
  "title": "...",
  "text": "...",
  "metadata": { "source": "pubmed", "pub_date": "2025-04-03", "raw_index": 1 }
}
'''
from __future__ import annotations
# PEP 563: all annotations are strings at runtime. This lets us use
# forward-referenced types inside dataclass field defaults without
# ImportError, and avoids evaluation overhead for complex type hints.

from dataclasses import asdict, dataclass, field
# Frozen dataclasses keep pipeline data hashable and guard against
# accidental mutation. asdict() provides the serialization path.

from typing import Any
# Metadata dicts carry heterogeneous values (str, int, None) so
# dict[str, Any] is the least-surprising container type.

# -- Input schema: raw PubMed article as loaded from JSON ------------------
@dataclass(frozen=True)
class RawPubMedArticle:
  article_title:str   # verbatim title from the PubMed record
  article_abstract:str  # verbatim abstract from the PubMed record
  # Publication date as a string-keyed mapping (e.g. {'year':'2025','month':'04','day':'03'}).
  # None when the source record omits pub_date entirely. dict[str, Any] tolerates
  # values already parsed as int or missing month/day keys.
  pub_date: dict[str, Any] | None = None

# -- Output schema: normalized document after ingestion pipeline ----------
@dataclass(frozen=True)
class MedicalDocument:
  doc_id:str     # stable identifier, format: pubmed_{raw_index:06d}
  title:str      # cleaned article title
  text:str       # article abstract (or full text once the pipeline is extended past abstracts)
  # Supplementary fields not consumed by the answer generator — source,
  # pub_date, and raw_index are carried through for traceability and citation.
  metadata:dict[str,Any]=field(default_factory=dict)

  def to_json_dict(self):
      return asdict(self)

# -- Chunk schema: retrieval-ready fragments produced by the indexer -------
@dataclass(frozen=True)
class MedicalChunk:
  chunk_id:str    # compound key: {doc_id}_chunk_{seq:03d}
  doc_id:str      # back-pointer to the parent MedicalDocument
  text:str        # sliced window of the parent document's text field
  # Inherits parent document metadata plus chunk-specific bookkeeping
  # (chunk_index, chunk_start, chunk_end, title).
  metadata:dict[str,Any]=field(default_factory=dict)

  def to_json_dict(self):
      return asdict(self)