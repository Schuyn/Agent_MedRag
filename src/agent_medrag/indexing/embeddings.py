'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2026-05-20 20:23:23
LastEditTime: 2026-05-21 18:09:53
FilePath: /Agent_MedRag/src/agent_medrag/indexing/embeddings.py
Description: 
Turn text into vector embeddings for retrieval.
'''
from __future__ import annotations
from typing import Protocol

# Default for rapid local iteration; swap to bge-large-en-v1.5 or text-embedding-3-small
# (OpenAI) via config once the retrieval pipeline is stable end-to-end.
DEFAULT_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"


class TextEmbeddingModel(Protocol):
    """Structural contract for embedding backends.

    Any class satisfying this protocol can be used by the retrieval pipeline —
    just provide ``embed_texts(texts) -> list[list[float]]``.
    """
    def embed_texts(self,texts:list[str])->list[list[float]]:
        """Convert a list of texts into their corresponding vector embeddings."""
        ...


class SentenceTransformerEmbedding:
    """Production embedding backend wrapping sentence-transformers.

    - Lazily imports SentenceTransformer so the module is importable even
      when the optional dependency is not installed (friendly error at init time).
    - Batches texts through ``model.encode`` and returns plain Python lists
      of floats (not numpy arrays) for serialization safety.
    """
    def __init__(
        self,
        model_name:str=DEFAULT_EMBEDDING_MODEL,
        normalize_embeddings:bool=True,
        device:str | None=None,
    )->None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedding. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.model_name=model_name
        self.normalize_embeddings=normalize_embeddings
        self.model=SentenceTransformer(model_name,device=device)

    def embed_texts(self,texts:list[str])->list[list[float]]:
        """Encode a batch of texts into unit-normed (cosine-friendly) vectors."""
        cleaned_texts=[text.strip() for text in texts]
        # Strip guards against whitespace-only strings producing degenerate embeddings.

        if not cleaned_texts:
            return []

        # Request raw numpy output; convert_to_numpy=True avoids the extra
        # allocation of converting from torch → numpy inside the library.
        embeddings=self.model.encode(
            cleaned_texts,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        # Return plain Python lists — callers shouldn't need to import numpy.
        return embeddings.tolist()

    def embed_text(self,text:str)->list[float]:
        """Convenience wrapper: encode a single text string to a single vector."""
        embeddings=self.embed_texts([text])

        if not embeddings:
            return []

        return embeddings[0]
        