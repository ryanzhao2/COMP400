import os
from typing import Any
import numpy as np

try:
    import faiss  # type: ignore
except Exception as import_error:
    raise SystemExit(
        "FAISS import failed. Ensure the virtual environment is active and faiss-cpu is installed.\n"
        "Activate: .\\.venv\\Scripts\\Activate.ps1"
    ) from import_error


def build_sample_index(
    dimension: int = 3,
    num_vectors: int = 5,
    seed: int = 42,
    hnsw_m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 50,
) -> Any:
    """Create a FAISS HNSW (L2) index with random vectors.

    hnsw_m controls graph connectivity; ef_construction/ef_search trade speed vs recall.
    """
    rng = np.random.default_rng(seed)
    vectors = rng.random((num_vectors, dimension), dtype=np.float32)
    index = faiss.IndexHNSWFlat(dimension, hnsw_m)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(vectors)  # type: ignore[arg-type]
    return index


def save_index(index: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    faiss.write_index(index, path)


def load_index(path: str) -> Any:
    return faiss.read_index(path)


def demo() -> None:
    dimension = 3
    index_path = "data/faiss_demo.index"
    ef_search = 64

    # 1) Build and save the index
    index = build_sample_index(dimension=dimension, ef_search=ef_search)
    save_index(index, index_path)

    # 2) Load the index back
    loaded_index = load_index(index_path)
    # Ensure query-time efSearch is set on loaded index
    if hasattr(loaded_index, "hnsw"):
        loaded_index.hnsw.efSearch = ef_search  # type: ignore[attr-defined]

    # 3) Query with a sample vector
    query = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    distances, ids = loaded_index.search(query, k=3)  # type: ignore[misc]

    print("Query:", query.tolist())
    print("Nearest IDs:", ids.tolist())
    print("Distances:", distances.tolist())
    print(f"Index saved to: {index_path}")


if __name__ == "__main__":
    demo()
