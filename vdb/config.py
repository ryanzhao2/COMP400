from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ParameterConstraints:
    # Constraints for HNSW parameters during optimization
    hnsw_m_min: int = 4
    hnsw_m_max: int = 16
    ef_construction_min: int = 50
    ef_construction_max: int = 150
    ef_search_min: int = 10
    ef_search_max: int = 100


@dataclass
class PerformanceThresholds:
    # Performance targets and limits for optimization

    min_recall: float = 0.9
    max_latency_ms: float = 10.0
    max_memory_gb: float = 1.0
    max_experiments: int = 20


# Database type configurations for different use cases
DATABASE_TYPES: Dict[str, Dict[str, Any]] = {
    "knowledge_reasoning": {
        "name": "Knowledge + Reasoning DB",
        "description": "Optimized for high recall and thoroughness. Ideal for RAG systems, semantic search, and tasks where missing relevant information is costly. Prioritizes accuracy over speed.",
        "dataset_size": 100000,
        "dimension": 128,
        "thresholds": PerformanceThresholds(
            min_recall=0.90,  # High recall requirement
            max_latency_ms=200.0,  # Can tolerate higher latency
            max_memory_gb=5.0,  # Can use more memory for better results
        ),
        "constraints": ParameterConstraints(
            hnsw_m_min=16,  # Higher connectivity improves recall
            hnsw_m_max=128,
            ef_construction_min=200, 
            ef_construction_max=1000,
            # Balanced search depth: good recall without excessive query time
            ef_search_min=200,
            ef_search_max=1000,
        ),
    },
    "memory_reaction": {
        "name": "Memory + Reaction DB",
        "description": "Optimized for low latency and fast response. Ideal for real-time systems, chatbots, and applications where speed is critical. Prioritizes responsiveness over exhaustive search.",
        "dataset_size": 100000,
        "dimension": 128,
        "thresholds": PerformanceThresholds(
            min_recall=0.70,  # Lower recall acceptable
            max_latency_ms=5.0,  # Strict latency requirement
            max_memory_gb=1.0,  # Keep memory footprint small
        ),
        "constraints": ParameterConstraints(
            hnsw_m_min=4,  # Lower connectivity for speed
            hnsw_m_max=128,
            ef_construction_min=50,
            ef_construction_max=500,
            ef_search_min=50,
            ef_search_max=500,
        ),
    },
}


def get_database_config(db_type: str) -> Dict[str, Any]:
    # Get database type-specific configuration preset
    return DATABASE_TYPES.get(db_type, DATABASE_TYPES["knowledge_reasoning"])


