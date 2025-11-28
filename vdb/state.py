"""
LangGraph state definition for optimization workflow.

Defines OptimizationState TypedDict that flows through all workflow nodes,
carrying dataset info, current/best configurations, trial history, and
internal cached data.
"""

from typing import TypedDict, Dict, Any, Optional, List


class OptimizationState(TypedDict):
    """
    Complete state for HNSW parameter optimization workflow.
    
    Passed between LangGraph nodes, tracking:
    - Dataset characteristics (size, dimension, type)
    - Current experiment parameters and metrics
    - Historical experiments and best configuration
    - Optimization progress (iteration count, phase, status)
    - Cached data for efficiency (_vectors, _queries)
    """
    dataset_size: int
    dimension: int
    database_type: str  # "knowledge_reasoning" or "memory_reaction"
    current_params: Dict[str, int]
    current_metrics: Dict[str, float]
    experiment_history: List[Dict[str, Any]]
    best_config: Dict[str, Any]
    iteration_count: int
    status: str
    error_message: Optional[str]
    # Internal cached data (not persisted):
    _vectors: Optional[Any]
    _queries: Optional[Any]
    _index: Optional[Any]  # Cached FAISS index from build_index_node
    _last_build_ms: Optional[float]
    phase: str  # "recall" or "latency"
    tried_params: List[Dict[str, int]]
    exploration_count: int


