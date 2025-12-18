from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DistanceStats:
    # Statistical summary of pairwise vector distances
    mean: float
    std: float
    minimum: float
    maximum: float
    p25: float
    p50: float
    p75: float


@dataclass
class TrialRecord:
    # Record of a single HNSW parameter experiment
    params: Dict[str, int]
    metrics: Dict[str, float]
    distance_stats: DistanceStats


@dataclass
class GraphData:
    # Graph data for HNSW parameter optimization
    dataset_size: int
    dimension: int
    global_distance_stats: DistanceStats
    trials: List[TrialRecord]


