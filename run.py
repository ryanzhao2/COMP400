"""
CLI entrypoint for HNSW parameter optimization.

Runs the VectorDatabaseAgent to automatically tune FAISS HNSW parameters,
logging all experiments for analysis and visualization.

Usage examples:
  # Basic run with synthetic data
  python run.py --iterations 20 --database-type knowledge_reasoning
  
  # Run with custom dataset
  python run.py --iterations 20 \
    --dataset-vectors data/gaussian_n100000_d128_k8_std0.1.npy \
    --dataset-queries data/gaussian_n100000_d128_k8_std0.1_queries.npy \
    --database-type memory_reaction
"""

import os
import argparse

from vdb.agent import VectorDatabaseAgent


def main() -> None:
    """Parse CLI arguments and run the optimization loop."""
    from vdb.config import DATABASE_TYPES
    
    parser = argparse.ArgumentParser(description="Run HNSW optimization experiments and log results.")
    parser.add_argument("--iterations", type=int, default=5, help="Max number of experiments to run")
    parser.add_argument("--log", default=os.getenv("EXPERIMENT_LOG", "experiments.jsonl"), help="Path to experiments JSONL log")
    parser.add_argument("--dataset-vectors", dest="dataset_vectors", default=None, help="Path to dataset vectors .npy file")
    parser.add_argument("--dataset-queries", dest="dataset_queries", default=None, help="Path to dataset queries .npy file")
    parser.add_argument("--database-type", dest="database_type", default="knowledge_reasoning", 
                       choices=list(DATABASE_TYPES.keys()),
                       help="Database type: 'knowledge_reasoning' (high recall, slower) or 'memory_reaction' (low latency, fast)")
    args = parser.parse_args()

    agent = VectorDatabaseAgent(
        log_path=args.log,
        dataset_vectors_path=args.dataset_vectors,
        dataset_queries_path=args.dataset_queries,
        database_type=args.database_type,
    )

    results = agent.optimize(max_iterations=args.iterations)

    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Experiments run: {results['iteration_count']}")

    if results.get("best_config"):
        print(f"\nBest configuration:")
        print(f"  HNSW M: {results['best_config']['params']['hnsw_m']}")
        print(f"  EF Construction: {results['best_config']['params']['ef_construction']}")
        print(f"  EF Search: {results['best_config']['params']['ef_search']}")
        print(f"\nPerformance:")
        print(f"  Recall: {results['best_config']['metrics']['recall']:.3f}")
        print(f"  Latency: {results['best_config']['metrics']['latency_ms']:.1f} ms")
        print(f"  Memory: {results['best_config']['metrics']['memory_gb']:.2f} GB")


if __name__ == "__main__":
    main()


