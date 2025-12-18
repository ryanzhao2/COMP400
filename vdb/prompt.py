from typing import Any, Dict, List, Optional
import json


def get_recent_trials(state: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    """Extract last n trials from current optimization state."""
    trials_src = state.get("experiment_history", [])
    if not trials_src or n <= 0:
        return []
    trials = trials_src[-n:][::-1]
    compact: List[Dict[str, Any]] = []
    for t in trials:
        params = t.get("params", {})
        metrics = t.get("metrics", {})
        compact.append({
            "params": {
                "hnsw_m": int(params.get("hnsw_m", 0)),
                "ef_construction": int(params.get("ef_construction", 0)),
                "ef_search": int(params.get("ef_search", 0)),
            },
            "metrics": {
                "recall": float(metrics.get("recall", 0.0)),
                "true_recall": (float(metrics.get("true_recall")) if metrics.get("true_recall") is not None else None),
                "latency_ms": float(metrics.get("latency_ms", 0.0)),
                "memory_gb": float(metrics.get("memory_gb", 0.0)),
            }
        })
    return compact


def build_tuning_prompt(agent: Any, state: Dict[str, Any], recent_trials: List[Dict[str, Any]], past_log_trials: List[Dict[str, Any]]) -> str:
    """
    Construct the complete prompt for LLM-guided parameter tuning.
    
    Includes optimization goals, constraints, targets, trial history, and
    phase-specific instructions for the LLM.
    """
    database_type = state.get("database_type") or getattr(agent, "database_type", "knowledge_reasoning")
    phase = state.get("phase", "recall")
    db_name = "Knowledge + Reasoning DB" if database_type == "knowledge_reasoning" else "Memory + Reaction DB"
    
    # Define optimization goals based on database type and phase
    if database_type == "knowledge_reasoning":
        if phase == "recall":
            goal = "First, achieve at least 0.90 recall. Once recall >= 0.90, we will switch to optimizing latency."
        else:
            goal = "Recall target (0.90) is met. Now find the configuration with MINIMUM 0.90 recall (must be >= 0.90) and the LOWEST latency. The best configuration is the one with recall >= 0.90 that has the lowest latency. If two configs have the same latency and both meet 0.90, prefer the one with lower memory. To reduce latency, consider lowering ef_search (primary driver of query latency), and potentially reducing hnsw_m and ef_construction if they are higher than necessary for the current recall level."
    else:  # memory_reaction
        if phase == "recall":
            goal = "First, achieve at least 0.70 recall. Once recall >= 0.70, we will switch to optimizing latency."
        else:
            goal = "Recall target (0.70) is met. Now find the configuration CLOSEST to 0.70 recall (not necessarily above it) with the LOWEST latency. The best configuration minimizes both: (1) distance from 0.70 recall, and (2) latency. If two configs are equally close to 0.70, choose the one with lower latency. To reduce latency, consider lowering ef_search (primary driver of query latency), and potentially reducing hnsw_m and ef_construction if they are higher than necessary."
    
    return (
        f"You are an expert on FAISS HNSW parameter tuning for {db_name}.\n\n"
        f"OPTIMIZATION GOAL: {goal}\n\n"
        "HNSW Parameters (what they control):\n"
        f"- hnsw_m ({agent.constraints.hnsw_m_min}-{agent.constraints.hnsw_m_max}): Graph connectivity - number of bidirectional links per node. Higher M = better recall but more memory and slower build time. Affects both recall and latency.\n"
        f"- ef_construction ({agent.constraints.ef_construction_min}-{agent.constraints.ef_construction_max}): Build-time search depth - how many candidates to explore when inserting vectors. Higher ef_construction = better recall but slower index construction. Only affects build time, not query latency.\n"
        f"- ef_search ({agent.constraints.ef_search_min}-{agent.constraints.ef_search_max}): Query-time search depth - how many candidates to explore during search. Higher ef_search = better recall but MUCH higher query latency. This is the PRIMARY driver of query latency.\n\n"
        "Performance Targets:\n"
        f"- min_recall: {agent.thresholds.min_recall} (HARD CONSTRAINT: recall must NOT dip below this)\n"
        f"- max_latency_ms: {agent.thresholds.max_latency_ms} (target to minimize)\n"
        f"- max_memory_gb: {agent.thresholds.max_memory_gb}\n\n"
        + (f"In latency phase, recall must stay >= {agent.thresholds.min_recall}. Prioritize lower latency while maintaining this constraint.\n\n" if phase == "latency" else "")
        + "Context:\n"
        f"- Current phase: {phase}\n"
        f"- dataset_size: {state.get('dataset_size')}\n"
        f"- dimension: {state.get('dimension')}\n"
        f"- last_params: {json.dumps(state.get('current_params', {}))}\n"
        f"- last_metrics: {json.dumps(state.get('current_metrics', {}))}\n"
        f"- best_config: {json.dumps(state.get('best_config', {}))}\n"
        f"- recent_trials (most recent first): {json.dumps(recent_trials)}\n"
        + (f"- past_runs_similar (from log): {json.dumps(past_log_trials)}\n\n" if past_log_trials else "\n")
        + "Return ONLY compact JSON with keys hnsw_m, ef_construction, ef_search, rationale."
    )
