from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DistanceStats:
    """Summary statistics of vector distances in the graph/index."""
    mean: float
    std: float
    minimum: float
    maximum: float
    p25: float
    p50: float
    p75: float


@dataclass
class TrialRecord:
    """Single trial observed in the graph with params, metrics, and distance stats."""
    params: Dict[str, int]
    metrics: Dict[str, float]
    distance_stats: DistanceStats


@dataclass
class GraphData:
    """External graph data to inform LLM recommendations."""
    dataset_size: int
    dimension: int
    global_distance_stats: DistanceStats
    trials: List[TrialRecord]


