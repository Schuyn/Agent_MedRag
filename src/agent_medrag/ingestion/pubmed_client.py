'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-05-20 18:23:23
LastEditTime: 2026-05-20 18:23:24
FilePath: /Agent_MedRag/src/agent_medrag/ingestion/pubmed_client.py
Description:
PubMed Entrez API client (stub). Will download article abstracts via BioPython's
Entrez for a user-specified date range and maximum article count, writing the
results in the same JSON format expected by json_loader.py. Respects NCBI rate
limits and requires ENTREZ_EMAIL in the environment.

'''
