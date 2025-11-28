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


def create_index(dimension: int, hnsw_m: int, ef_construction: int, ef_search: int):
    """
    Create a FAISS HNSW index with specified parameters.
    
    Args:
        dimension: Vector dimensionality
        hnsw_m: Graph connectivity (number of bidirectional links per node)
        ef_construction: Build-time search depth
        ef_search: Query-time search depth
        
    Returns:
        Configured FAISS IndexHNSWFlat instance
    """
    index = faiss.IndexHNSWFlat(dimension, hnsw_m)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    return index


def persist_index(index, path: Optional[str]) -> Optional[int]:
    """
    Save FAISS index to disk and return file size.
    
    Args:
        index: FAISS index instance
        path: Destination file path (creates directories if needed)
        
    Returns:
        File size in bytes, or None if no path provided
    """
    if not path:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)  # type: ignore
    return os.path.getsize(path)


