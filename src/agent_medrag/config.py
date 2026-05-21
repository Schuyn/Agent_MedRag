'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-05-16 18:06:40
LastEditTime: 2026-05-16 18:06:44
FilePath: /Agent_MedRag/src/agent_medrag/config.py
Description:
Centralized YAML configuration loader. Reads configs/ files and provides typed access to data paths, chunking parameters, embedding settings, vector store options, retrieval/reranking knobs, LLM provider settings, safety flags, and logging levels. All tunable parameters flow through here so no hardcoded values leak into the rest of the codebase.
'''
