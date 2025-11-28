"""
Data models for optimization state and results.

Defines dataclasses for:
- DistanceStats: Statistical summaries of vector distances
- TrialRecord: Single experiment with parameters and metrics
- GraphData: Collection of trials for LLM-based recommendations
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DistanceStats:
    """
    Statistical summary of pairwise vector distances.
    
    Used to characterize dataset geometry, which influences optimal HNSW parameters.
    Tight clusters (low std, low p50) may benefit from different settings than
    dispersed distributions.
    """
    mean: float
    std: float
    minimum: float
    maximum: float
    p25: float
    p50: float
    p75: float


@dataclass
class TrialRecord:
    """
    Record of a single HNSW parameter experiment.
    
    Contains the configuration tested, resulting performance metrics,
    and distance statistics of the dataset used.
    """
    params: Dict[str, int]
    metrics: Dict[str, float]
    distance_stats: DistanceStats


@dataclass
class GraphData:
    """
    Historical experiment data for LLM-based recommendations.
    
    Packages multiple trials with dataset characteristics to provide
    the LLM with rich context for parameter suggestions.
    """
    dataset_size: int
    dimension: int
    global_distance_stats: DistanceStats
    trials: List[TrialRecord]


