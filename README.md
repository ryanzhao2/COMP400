# FAISS HNSW Parameter Optimization

Automated tuning of FAISS HNSW parameters using LangGraph and optional LLM guidance.

This project uses an agent-based approach to find good HNSW parameter settings by running experiments and learning from results. It supports two different optimization profiles depending on whether you need high recall or low latency.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Basic run with synthetic data:
```bash
python run.py --iterations 20 --database-type knowledge_reasoning
```

With your own dataset:
```bash
python run.py --iterations 20 --dataset-vectors data/gaussian_n100000_d128_k8_std0.1.npy --dataset-queries data/gaussian_n100000_d128_k8_std0.1_queries.npy
```

Or use the Python API:
```python
from vdb.agent import VectorDatabaseAgent

agent = VectorDatabaseAgent(database_type="knowledge_reasoning")
results = agent.optimize(max_iterations=20)
```

## Database Types

Two optimization profiles:

- **knowledge_reasoning**: Prioritizes high recall (targets ~90%+). Good for RAG, semantic search, Q&A systems.
- **memory_reaction**: Prioritizes low latency (targets <5ms). Good for chatbots and real-time systems, accepts lower recall (~70%).

## HNSW Parameters

- **hnsw_m**: Number of connections per node. Higher values improve recall but use more memory.
- **ef_construction**: Search depth during index building. Higher values build better indexes but take longer.
- **ef_search**: Search depth during queries. Higher values improve recall but slow down queries.

## Plotting Results

Generate plots from experiment logs:

```bash
python plot_experiments.py --log experiments/
python plot_experiments.py --log experiments/knowledge_reasoning/
python plot_experiments.py --log experiments/ --hnsw_m 32 --ef_construction 400
```

Outputs recall vs latency, Pareto frontier, ef_search analysis, and true vs estimated recall plots.

## Generating Datasets

```bash
python generate_datasets.py --n 100000 --d 128
python generate_datasets.py --n 500000 --d 256 --queries 500
```

Supports Gaussian clusters, uniform sphere, and power-law distributions.

The agent uses LLM guidance when available (reads past experiment logs for context), otherwise falls back to heuristic parameter selection.

## Results

Typical performance:
- Knowledge reasoning: ~0.90-0.95 recall, 50-150ms latency
- Memory reaction: ~0.70-0.75 recall, 2-4ms latency

The agent usually finds acceptable results within 5 iterations. It uses a two-phase approach: first optimizing for recall, then switching to latency optimization once targets are met.

## Notes

- Install FAISS with `pip install faiss-cpu` (or `faiss-gpu` if you have CUDA)
- For large datasets (>100MB), the agent uses heuristic recall estimation instead of brute-force computation
- If LLM calls fail, the agent falls back to heuristic parameter generation automatically

Research project for educational purposes.
