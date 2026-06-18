'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-06-17 12:12:57
LastEditTime: 2026-06-17 19:59:26
FilePath: /Agent_MedRag/src/agent_medrag/generation/answer_generator.py
Description: 
职责是把 LLM 返回的 JSON 转成 MedicalAnswer，并把 E1/E2 映射回真实 Citation。
'''
from __future__ import annotations

import json
from typing import Any

from agent_medrag.generation.llm_provider import LLMProvider
from agent_medrag.generation.prompts import build_answer_prompt
from agent_medrag.schemas import Citation,MedicalAnswer,RetrievalSummary,RetrievedEvidence

class AnswerGenerator:
    def __init__(self,llm_provider:LLMProvider)->None:
        self.llm_provider=llm_provider
        
    def generate(
        self,
        question:str,
        evidence:list[RetrievedEvidence],
    )->MedicalAnswer:
        if not evidence:
            return _insufficient_evidence_answer(question)
        
        prompt=build_answer_prompt(question,evidence)
        raw_output=self.llm_provider.generate(prompt)
        parsed=_parse_llm_json(raw_output)
        
        answer=str(parsed.get('answer','')).strip()
        confidence=_normalize_confidence(parsed.get('confidence'))
        limitations=_normalize_optional_text(parsed.get('limitations'))
        safety_note=str(
            parsed.get(
                'safety_note',
                "This answer is for medical literature exploration and is not medical advice.",
            )
        ).strip()
        
        used_ids=_normalize_used_evidence_ids(
            parsed.get('used_evidence_ids',[]),
            evidence,
        )
        citations=_build_citations(used_ids,evidence)
        
        if not answer:
            answer = "The model did not return a usable answer."
            confidence = "low"
        
        return MedicalAnswer(
            question=question,
            answer=answer,
            confidence=confidence,
            citations=citations,
            limitations=limitations,
            safety_note=safety_note,
            retrieval_summary=RetrievalSummary(
                query=question,
                top_k=len(evidence),
                retrieved_count=len(evidence),
                used_evidence_count=len(citations),
            ),
            metadata={
                'used_evidence_ids':used_ids,
                'raw_llm_output':raw_output,
            },
        )
        
def _parse_llm_json(raw_output:str)->dict[str,Any]:
    cleaned=raw_output.strip()
    
    if cleaned.startswith('```'):
        cleaned=cleaned.removeprefix('```json').removeprefix('```').strip()
        cleaned=cleaned.removesuffix('```').strip()
        
    start=cleaned.find('{')
    end=cleaned.rfind('}')
    if start!=-1 and end!=-1:
        cleaned=cleaned[start:end+1]
    
    parsed=json.loads(cleaned)
    if not isinstance(parsed,dict):
        raise ValueError("LLM output must be a JSON object.")
    
    return parsed
    
def _normalize_confidence(value:Any)->str:
    confidence=str(value or 'low').strip().lower()
    if confidence not in {'low','medium','high'}:
        return 'low'
    
    return confidence
    
def _normalize_optional_text(value:Any)->str | None:
    if value is None:
        return None
    
    text=str(value).strip()
    if not text or text.lower()=='null':
        return None
    
    return text
        
def _normalize_used_evidence_ids(
    used_ids:Any,
    evidence:list[RetrievedEvidence],
)->list[str]:
    allowed_ids={f'E{index}' for index in range(1,len(evidence)+1)}
    
    if not isinstance(used_ids,list):
        return []
    
    normalized:list[str]=[]
    
    for item in used_ids:
        evidence_id=str(item).strip().strip('[]')
        
        if evidence_id in allowed_ids and evidence_id not in normalized:
            normalized.append(evidence_id)
    
    return normalized

def _build_citations(
    used_ids:list[str],
    evidence:list[RetrievedEvidence],
)->list[Citation]:
    citations:list[Citation]=[]
    
    for evidence_id in used_ids:
        evidence_index=int(evidence_id.removeprefix('E'))-1
        item=evidence[evidence_index]

        citations.append(
            Citation(
                title=item.title,
                pub_date=item.pub_date,
                evidence=item.text,
                doc_id=item.doc_id,
                chunk_id=item.chunk_id,
                url=None,
            )
        )
        
    return citations
    
def _insufficient_evidence_answer(question:str)->MedicalAnswer:
    return MedicalAnswer(
        question=question,
        answer="I do not have enough retrieved evidence to answer this question.",
        confidence='low',
        citations=[],
        limitations="No relevant evidence was retrieved from the local corpus.",
        safety_note="This answer is for medical literature exploration and is not medical advice.",
        retrieval_summary=RetrievalSummary(
            query=question,
            top_k=0,
            retrieved_count=0,
            used_evidence_count=0,
        ),
        metadata={},
    )