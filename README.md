# FAISS HNSW Tuning Agent

Automated tuning of FAISS HNSW parameters using an agentic workflow. This system optimizes index configurations (`M`, `efConstruction`, `efSearch`) for specific recall and latency targets using iterative experimentation.

## Setup

```bash
pip install -r requirements.txt
# Optional: Set GEMINI_API_KEY in .env for LLM guidance
```

## Usage

Run the optimization agent:

```bash
# High recall focus (Target: >0.90)
python run.py --database-type knowledge_reasoning

# Low latency focus (Target: ~0.70 recall, min latency)
python run.py --database-type memory_reaction
```

Run with custom dataset:
```bash
python run.py --dataset-vectors data/vectors.npy --dataset-queries data/queries.npy
```

## Workload Profiles

| Profile | Goal | Target Metrics |
|---------|------|----------------|
| **Knowledge + Reasoning** | High Recall | Recall ≥ 0.90 |
| **Memory + Reaction** | Low Latency | Latency < 1ms, Recall ≈ 0.70 |

## Project Structure

- `vdb/agent.py`: Main agent logic (LangGraph workflow).
- `vdb/hnsw.py`: FAISS index wrappers.
- `run.py`: Entry point for experiments.
- `plot_experiments.py`: Visualization tools for experiment logs.

## Output

Experiments are logged to `experiments/{profile}/experiments_n{size}.jsonl`.

Generate plots:
```bash
python plot_experiments.py --log experiments/
```
