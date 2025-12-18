"""
FAISS HNSW index creation and persistence utilities.

Provides thin wrappers around FAISS IndexHNSWFlat for creating and
saving HNSW indexes with specified parameters.
"""

from typing import Optional
import os

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore

# Create a new HNSW index
def create_index(dimension: int, hnsw_m: int, ef_construction: int, ef_search: int):
    index = faiss.IndexHNSWFlat(dimension, hnsw_m)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    return index

# Persist the index to a file
def persist_index(index, path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)  # type: ignore
    return os.path.getsize(path)


