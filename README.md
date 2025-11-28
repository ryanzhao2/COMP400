# FAISS HNSW Agentic Optimization

Intelligent agent system for automatic FAISS HNSW parameter tuning using LangGraph and LLM-guided optimization.

## Overview

This research project demonstrates autonomous AI agents that optimize vector database parameters by balancing recall, latency, and memory through iterative experimentation. The agent uses LangGraph workflows and optional LLM guidance to find optimal HNSW configurations for different use cases.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run optimization (synthetic data)
python run.py --iterations 20 --database-type knowledge_reasoning

# Run with custom dataset
python run.py --iterations 20 \
  --dataset-vectors data/gaussian_n100000_d128_k8_std0.1.npy \
  --dataset-queries data/gaussian_n100000_d128_k8_std0.1_queries.npy
```

### Python API

```python
from vdb.agent import VectorDatabaseAgent

agent = VectorDatabaseAgent(database_type="knowledge_reasoning")
results = agent.optimize(max_iterations=20)
print(f"Best params: {results['best_config']['params']}")
```

## Database Types

Two optimization profiles for different use cases:

| Type | Priority | Use Cases | Targets |
|------|----------|-----------|---------|
| **knowledge_reasoning** | High recall | RAG, semantic search, Q&A | ≥90% recall, <200ms |
| **memory_reaction** | Low latency | Chatbots, real-time systems | ≥70% recall, <5ms |

## HNSW Parameters

| Parameter | Description | Impact |
|-----------|-------------|--------|
| **hnsw_m** | Graph connectivity (links per node) | Higher = better recall, more memory |
| **ef_construction** | Build-time search depth | Higher = better quality, slower build |
| **ef_search** | Query-time search depth | Higher = better recall, slower queries |

## Visualization

Generate plots from experiments:

```bash
# Plot all experiments by database type
python plot_experiments.py --log experiments/

# Filter by specific type
python plot_experiments.py --log experiments/knowledge_reasoning/

# Fixed parameter analysis
python plot_experiments.py --log experiments/ --hnsw_m 32 --ef_construction 400
```

Generates: `recall_vs_latency.png`, `pareto_frontier.png`, `efsearch_vs_recall.png`, `true_vs_estimated_recall.png`


```

## Generating Datasets

```bash
# Default sizes (10K, 20K, 100K)
python generate_datasets.py --n 100000 --d 128

# Custom configurations
python generate_datasets.py --n 500000 --d 256 --queries 500
python generate_datasets.py --n 50000 --clusters 16 --std 0.05  # Tight clusters
```

Distributions: Gaussian clusters, uniform sphere, power-law


The LLM receives trial history, performance targets, and constraints to suggest intelligent parameter choices. Falls back to heuristic search if unavailable.

## Performance

Typical results:
- **Knowledge+Reasoning**: 0.92-0.95 recall, 50-150ms latency (15-20 iterations)
- **Memory+Reaction**: 0.72-0.80 recall, 2-4ms latency (10-15 iterations)

## Research Contributions

- **Agentic AI for hyperparameter optimization** using LLM guidance
- **Two-phase optimization** (recall → latency) for better Pareto solutions
- **Cross-run learning** from historical experiment logs
- **Automatic organization** of experiments by database type and size

## Troubleshooting

**FAISS installation**: Use `pip install faiss-cpu` (or `faiss-gpu` with CUDA)

**Memory issues**: Set `FAISS_USE_GPU_EXACT=0` or reduce dataset size. Agent uses heuristic recall for datasets >100MB.

**LLM errors**: Agent automatically falls back to heuristic parameter generation

## License

This project is for research and educational purposes.
