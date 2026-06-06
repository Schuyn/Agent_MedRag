'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-05-15 15:46:07
LastEditTime: 2026-06-03 15:30:22
FilePath: /Agent_MedRag/src/agent_medrag/schemas.py
Description:
Immutable dataclass schemas used throughout the ingestion, indexing, retrieval,
and answer-generation pipeline. These types model the full document workflow:

- RawPubMedArticle: verbatim JSON records as loaded from source (PubMed).
- MedicalDocument: normalized, traceable documents produced by the ingestion
  pipeline; carries provenance metadata (source, pub_date, raw_index) for
  citation and debugging.
- MedicalChunk: retrieval-ready text slices created for embedding, indexing,
  and efficient retrieval; includes chunk bookkeeping (index, start/end).
- RetrievedEvidence / Citation: structures describing retrieved fragments,
  their scores, and citation metadata surfaced in final answers.
- MedicalAnswer: the structured final response (answer text, confidence,
  citations, retrieval summary, safety note, and optional limitations).

All dataclasses are frozen to ensure immutability, hashability, and
deterministic behavior in the pipeline. Helper methods (e.g. to_json_dict)
provide simple serialization for logging, storage, and evaluation.
'''
from __future__ import annotations
# PEP 563: all annotations are strings at runtime. This lets us use
# forward-referenced types inside dataclass field defaults without
# ImportError, and avoids evaluation overhead for complex type hints.

from dataclasses import asdict,dataclass,field
# Frozen dataclasses keep pipeline data hashable and guard against
# accidental mutation. asdict() provides the serialization path.

from typing import Any,Literal
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
  
# -- Retrieval schema: evidence returned by the retriever --------
@dataclass(frozen=True)
class RetrievedEvidence:
  chunk_id:str
  doc_id:str
  title:str
  text:str
  retrieval_score:float | None=None
  rerank_score:float | None=None
  pub_date:str | None=None
  source:str='local'
  metadata:dict[str,Any]=field(default_factory=dict)
  
  def to_json_dict(self):
    return asdict(self)
  
# -- Citation schema: source evidence exposed in the final answer -----
@dataclass(frozen=True)
class Citation:
  title:str
  pub_date:str | None
  evidence:str
  doc_id:str
  chunk_id:str
  url:str | None=None
  
  def to_json_dict(self):
    return asdict(self)
  
# -- Retrieval summary: compact debug/trace info for an ask run ------
@dataclass(frozen=True)
class RetrievalSummary:
  query:str
  top_k:int
  retrieved_count:int
  used_evidence_count:int
  
  def to_json_dict(self):
    return asdict(self)
  
# -- Final answer schema: structured output from medrag task
@dataclass(frozen=True)
class MedicalAnswer:
  question:str
  answer:str
  confidence:Literal['low','medium','high']
  citations:list[Citation]
  safety_note:str
  retrieval_summary: RetrievalSummary
  limitations:str | None=None
  metadata:dict[str,Any]=field(default_factory=dict)
  
  def to_json_dict(self):
    return asdict(self)

  