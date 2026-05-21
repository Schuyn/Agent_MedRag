'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-05-20 18:28:50
LastEditTime: 2026-05-20 18:41:39
FilePath: /Agent_MedRag/src/agent_medrag/indexing/chunker.py
Description: 
This module defines the chunking logic for processing medical documents. It takes in raw medical documents, processes them (e.g., splitting abstracts into smaller chunks), and prepares them for indexing and retrieval. 
'''
from __future__ import annotations  # Defer evaluation of type annotations to allow forward references

from agent_medrag.schemas import MedicalDocument, MedicalChunk  # Import the MedicalDocument and MedicalChunk schemas defined in schemas.py

def chunk_document(
    document:MedicalDocument,
    chunk_size:int=1000,  # Default chunk size (number of characters)
    chunk_overlap:int=150  # Default overlap size (number of characters)
)->list[MedicalChunk]:
    # Validate input parameters
    if chunk_size<=0:
        raise ValueError("chunk_size must be a positive integer.")
    
    if chunk_overlap<0 or chunk_overlap>=chunk_size:
        raise ValueError("chunk_overlap must be a non-negative integer less than chunk_size.")
        
    text=document.text.strip()
    if not text:
        return []
    
    chunks:list[MedicalChunk]=[]
    start=0
    chunk_index=0
    
    while start<len(text):
        end=min(start+chunk_size,len(text)) # In case of out of range
        chunk_text=text[start:end].strip()
        
        if chunk_text:
            chunks.append(
                MedicalChunk(
                    chunk_id=f"{document.doc_id}_chunk_{chunk_index:03d}",
                    doc_id=document.doc_id,
                    text=chunk_text,
                    metadata={
                        **document.metadata,  # Inherit metadata from the original document
                        "title": document.title,
                        "chunk_index": chunk_index,
                        "chunk_start":start,
                        "chunk_end":end,
                    },
                )
            )
            chunk_index+=1
            
        if end==len(text):
            break
        
        start=end-chunk_overlap  # Move back by chunk_overlap for the next chunk
        
    return chunks

def chunk_documents(
    documents:list[MedicalDocument],
    chunk_size:int=1000,
    chunk_overlap:int=150,
)->list[MedicalChunk]:
    
    chunks:list[MedicalChunk]=[]
    
    for document in documents:
        chunks.extend(
            chunk_document(
                document=document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
        
    return chunks