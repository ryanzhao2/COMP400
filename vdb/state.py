from typing import TypedDict, Dict, Any, Optional, List


class OptimizationState(TypedDict):
    """State for the optimization workflow."""
    dataset_size: int
    dimension: int
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
    _last_build_ms: Optional[float]
    phase: str  # "recall" or "latency"
    tried_params: List[Dict[str, int]]
    exploration_count: int


