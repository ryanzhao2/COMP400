"""
Basic FAISS HNSW Example
Demonstrates creating, saving, and querying a FAISS HNSW index.
"""

import os
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError("FAISS not installed. Run: pip install faiss-cpu")


def create_hnsw_index(dimension: int = 128, num_vectors: int = 1000) -> faiss.IndexHNSWFlat:
    """Create a FAISS HNSW index with sample data."""
    
    # Create HNSW index
    hnsw_m = 32  # Graph connectivity
    index = faiss.IndexHNSWFlat(dimension, hnsw_m)
    
    # Set construction parameters
    index.hnsw.efConstruction = 200  # Build quality
    index.hnsw.efSearch = 50         # Search quality
    
    # Generate random vectors
    vectors = np.random.random((num_vectors, dimension)).astype(np.float32)
    
    # Add vectors to index
    index.add(vectors)
    
    print(f"✅ Created HNSW index with {num_vectors} vectors, dimension {dimension}")
    print(f"   HNSW M: {hnsw_m}")
    print(f"   EF Construction: {index.hnsw.efConstruction}")
    print(f"   EF Search: {index.hnsw.efSearch}")
    
    return index


def query_index(index: faiss.IndexHNSWFlat, num_queries: int = 5, k: int = 10):
    """Query the index with random vectors."""
    
    dimension = index.d
    queries = np.random.random((num_queries, dimension)).astype(np.float32)
    
    # Search for nearest neighbors
    distances, indices = index.search(queries, k)
    
    print(f"\n🔍 Query Results:")
    for i in range(num_queries):
        print(f"   Query {i+1}: Found {len(indices[i])} neighbors")
        print(f"   Distances: {distances[i][:3]:.3f}...")
        print(f"   Indices: {indices[i][:3]}...")


def save_and_load_demo():
    """Demonstrate saving and loading FAISS index."""
    
    # Create index
    index = create_hnsw_index()
    
    # Save index
    index_path = "hnsw_index.faiss"
    faiss.write_index(index, index_path)
    print(f"💾 Index saved to: {index_path}")
    
    # Load index
    loaded_index = faiss.read_index(index_path)
    print(f"📂 Index loaded from: {index_path}")
    
    # Query both indexes
    print("\n🔍 Querying original index:")
    query_index(index)
    
    print("\n🔍 Querying loaded index:")
    query_index(loaded_index)
    
    # Clean up
    if os.path.exists(index_path):
        os.remove(index_path)
        print(f"🗑️  Cleaned up: {index_path}")


def main():
    """Run the FAISS HNSW demo."""
    print("🚀 FAISS HNSW Demo")
    print("=" * 40)
    
    # Basic index creation and querying
    index = create_hnsw_index()
    query_index(index)
    
    print("\n" + "=" * 40)
    print("💾 Save/Load Demo")
    print("=" * 40)
    
    # Save and load demonstration
    save_and_load_demo()
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main()
