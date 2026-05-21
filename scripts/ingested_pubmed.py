'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-05-20 18:26:42
LastEditTime: 2026-05-20 18:26:50
FilePath: /Agent_MedRag/scripts/ingested_pubmed.py
Description:
Standalone ingestion runner (stub). Thin wrapper around agent_medrag.ingestion
that downloads PubMed articles and runs the normalize→write pipeline end-to-end,
producing documents.jsonl in one shot without the CLI.

'''
