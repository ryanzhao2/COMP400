from dataclasses import dataclass
from typing import Dict, Any


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


# Database type configurations
DATABASE_TYPES: Dict[str, Dict[str, Any]] = {
    "knowledge_reasoning": {
        "name": "Knowledge + Reasoning DB",
        "description": "Retrieves broad context. Slow but extremely thorough. Supports deep reasoning tasks. Use when missing info = bad.",
        "dataset_size": 100000,  # Default dataset size for synthetic data generation
        "dimension": 128,  # Default vector dimension
        "thresholds": PerformanceThresholds(
            min_recall=0.90,  # Target: at least 0.90 recall, then optimize for latency
            max_latency_ms=200.0,  # Can tolerate higher latency for thoroughness
            max_memory_gb=5.0,  # Can use more memory for better recall
        ),
        "constraints": ParameterConstraints(
            hnsw_m_min=16,  # Higher connectivity for better recall
            hnsw_m_max=128,  # Reduced from 128 for faster builds
            ef_construction_min=100,  # Reduced from 200 for faster builds
            ef_construction_max=3000,  # Reduced from 400 for faster builds
            ef_search_min=50,  # Reduced from 100 for faster builds
            ef_search_max=3000,  # Reduced from 500 for faster builds
        ),
    },
    "memory_reaction": {
        "name": "Memory + Reaction DB",
        "description": "Retrieves the strongest match fast. Very low latency. Supports decision-making or fast assistant behaviors. Use when speed = critical.",
        "dataset_size": 100000,  # Default dataset size for synthetic data generation (smaller for faster experiments)
        "dimension": 128,  # Default vector dimension
        "thresholds": PerformanceThresholds(
            min_recall=0.70,  # Target: at least 0.70 recall (don't care much about recall), then optimize for latency
            max_latency_ms=5.0,  # Very strict latency requirement - minimize latency
            max_memory_gb=1.0,  # Keep memory usage low
        ),
        "constraints": ParameterConstraints(
            hnsw_m_min=4,  # Lower connectivity for speed
            hnsw_m_max=128,
            ef_construction_min=100,  # Lower construction for speed
            ef_construction_max=3000,
            ef_search_min=10,  # Lower search for speed
            ef_search_max=3000,
        ),
    },
}


def get_database_config(db_type: str) -> Dict[str, Any]:
    """Get database type-specific configuration preset."""
    return DATABASE_TYPES.get(db_type, DATABASE_TYPES["knowledge_reasoning"])


