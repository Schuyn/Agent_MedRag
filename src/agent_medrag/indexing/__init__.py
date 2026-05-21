'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-05-20 18:28:55
LastEditTime: 2026-05-20 19:10:07
FilePath: /Agent_MedRag/src/agent_medrag/indexing/__init__.py
Description: 
Just an empty __init__.py file to make the indexing module a package. The actual indexing logic will be implemented in other files within this module.
'''
from agent_medrag.indexing.chunker import chunk_document, chunk_documents

__all__ = [
    "chunk_document",
    "chunk_documents",
]


__version__ = '0.1.0'