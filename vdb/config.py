from dataclasses import dataclass


@dataclass
class ParameterConstraints:
    """Constraints for HNSW parameters"""
    hnsw_m_min: int = 4
    hnsw_m_max: int = 16
    ef_construction_min: int = 50
    ef_construction_max: int = 150
    ef_search_min: int = 10
    ef_search_max: int = 100


@dataclass
class PerformanceThresholds:
    """Performance guardrails"""
    min_recall: float = 0.9
    max_latency_ms: float = 10.0
    max_memory_gb: float = 1.0
    max_experiments: int = 20


