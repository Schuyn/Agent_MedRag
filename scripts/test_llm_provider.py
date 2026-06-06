'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-06-05 15:03:32
LastEditTime: 2026-06-05 20:17:28
FilePath: /Agent_MedRag/scripts/test_llm_provider.py
Description: 
Test scipt for llmprovider
'''
from agent_medrag.generation.llm_provider import DeepSeekProvider

def main() -> None:
    provider = DeepSeekProvider(max_output_tokens=30)
    print(provider.generate(
        "Reply with exactly these two words: provider works"
    ))


if __name__ == "__main__":
    main()