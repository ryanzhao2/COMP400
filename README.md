# Vector Database Agentic AI

LangGraph-based framework for automatically tuning FAISS HNSW parameters using intelligent agents.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic example
python main.py

# Run agent optimization
python vector_db_agent.py
```

## Usage

```python
from vector_db_agent import VectorDatabaseAgent

# Create and run agent
agent = VectorDatabaseAgent()
results = agent.optimize(max_iterations=10)

print(f"Best config: {results['best_config']}")
```

## With OpenAI (Optional)

```python
import os
os.environ["OPENAI_API_KEY"] = "your-key"

agent = VectorDatabaseAgent()  # Will use LLM for parameter generation
```

## Files

- `vector_db_agent.py` - Main agent implementation
- `main.py` - Basic FAISS HNSW example
- `requirements.txt` - Dependencies

## HNSW Parameters

- **hnsw_m**: Graph connectivity (4-64)
- **ef_construction**: Build quality (50-1000) 
- **ef_search**: Search quality (10-500)

For research in automated hyperparameter optimization and agentic AI systems.
