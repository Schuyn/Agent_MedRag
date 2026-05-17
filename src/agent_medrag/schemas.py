'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-05-15 15:46:07
LastEditTime: 2026-05-15 16:04:34
FilePath: /Agent_MedRag/src/agent_medrag/schemas.py
Description: 
Define the apperance of data

From:
{
  "article_title": "...",
  "article_abstract": "...",
  "pub_date": {
    "year": "2025",
    "month": "04",
    "day": "03"
  }
}

To:
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
'''
from __future__ import annotations
# Defer evaluation of type annotations to allow forward references
# (useful for dataclasses referring to types defined later).

from dataclasses import asdict, dataclass, field  # dataclass decorator, field() helper, asdict() serializer

# 'Any' to annotate values that can be of any type (e.g., flexible metadata fields).
from typing import Any

# Input raw data's schema
@dataclass(frozen=True)
class RawPubMedArticle:
  article_title: str  # article title text
  article_abstract: str  # article abstract text
  # publication date as a mapping (e.g. {'year':'2025','month':'04','day':'03'})
  # or None if date is missing. Use Any for flexible value types.
  pub_date: dict[str, Any] | None = None
  
# Output processed data's schema
@dataclass(frozen=True)
class MedicalDocument:
    doc_id:str
    title:str
    text:str    # Transform abtract to text, if future need to extend to full article, can modify this
    metadata:dict[str,Any]=field(default_factory=dict)  # Contains source, date and raw index, do not directly involved in answering question.
    
    def to_json_dict(self):
        return asdict(self)