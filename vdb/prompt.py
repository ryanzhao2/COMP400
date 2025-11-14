from typing import Any, Dict, List, Optional
import os
import json


def get_recent_trials(state: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
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


def load_past_log_trials(log_path: Optional[str], dim: Optional[int], size: Optional[int], limit: int = 5) -> List[Dict[str, Any]]:
    """
    Load past trials from log files, prioritizing similar dataset sizes.
    
    Strategy:
    1. Try exact size match first (if size-specific file exists)
    2. If not enough, search other size-specific files, sorted by closest dataset_size
    3. Filter by dimension (exact match required)
    """
    if not log_path:
        return []
    
    base_dir = os.path.dirname(log_path) or "."
    rows: List[Dict[str, Any]] = []
    
    # Collect candidate files: exact match first, then closest sizes
    candidate_files: List[tuple[str, int]] = []  # (path, dataset_size)
    
    if size is not None:
        # Try exact size-specific file first
        exact_path = os.path.join(base_dir, f"experiments_n{size}.jsonl")
        if os.path.isfile(exact_path):
            candidate_files.append((exact_path, size))
        
        # Find other size-specific files and sort by distance from target size
        try:
            for filename in os.listdir(base_dir):
                if filename.startswith("experiments_n") and filename.endswith(".jsonl"):
                    try:
                        # Extract size from filename: experiments_n100000.jsonl -> 100000
                        size_str = filename.replace("experiments_n", "").replace(".jsonl", "")
                        file_size = int(size_str)
                        file_path = os.path.join(base_dir, filename)
                        if file_path != exact_path and os.path.isfile(file_path):
                            candidate_files.append((file_path, file_size))
                    except (ValueError, AttributeError):
                        continue
            # Sort by absolute distance from target size
            candidate_files.sort(key=lambda x: abs(x[1] - size))
        except Exception:
            pass
    
    # Fallback to original log_path if no size-specific files found
    if not candidate_files and os.path.isfile(log_path):
        candidate_files.append((log_path, size or 0))
    
    # Load from candidate files until we have enough results
    for file_path, _ in candidate_files:
        if len(rows) >= limit:
            break
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if len(rows) >= limit:
                        break
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except Exception:
                        continue
                    ds = obj.get("dataset", {})
                    # Require exact dimension match if specified
                    if dim and ds.get("dimension") != dim:
                        continue
                    # Prefer exact size match, but accept similar sizes if we're searching other files
                    file_size = ds.get("dataset_size")
                    if size is not None and file_size != size:
                        # Only include if we're searching other files (not the exact match file)
                        # This allows us to get similar sizes when exact match doesn't have enough
                        pass  # Accept it for now, we'll prioritize exact matches in final sort
                    
                    params = obj.get("params", {})
                    metrics = obj.get("metrics", {})
                    rows.append({
                        "dataset_size": file_size,  # Keep for sorting
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
        except Exception:
            continue
    
    # Sort by dataset_size proximity (exact matches first, then closest)
    if size is not None:
        rows.sort(key=lambda x: abs(x.get("dataset_size", float("inf")) - size))
    
    # Remove dataset_size from final output and return most recent first
    result = []
    for row in rows[:limit]:
        row_copy = {k: v for k, v in row.items() if k != "dataset_size"}
        result.append(row_copy)
    
    return result[::-1]  # Most recent first


def build_tuning_prompt(agent: Any, state: Dict[str, Any], recent_trials: List[Dict[str, Any]], past_log_trials: List[Dict[str, Any]]) -> str:
    return (
        "You are an expert on FAISS HNSW parameter tuning. Propose parameters to maximize recall while respecting latency and memory targets.\n\n"
        "Constraints:\n"
        f"- hnsw_m: {agent.constraints.hnsw_m_min}-{agent.constraints.hnsw_m_max}\n"
        f"- ef_construction: {agent.constraints.ef_construction_min}-{agent.constraints.ef_construction_max}\n"
        f"- ef_search: {agent.constraints.ef_search_min}-{agent.constraints.ef_search_max}\n\n"
        "Targets:\n"
        f"- min_recall: {agent.thresholds.min_recall}\n"
        f"- max_latency_ms: {agent.thresholds.max_latency_ms}\n"
        f"- max_memory_gb: {agent.thresholds.max_memory_gb}\n\n"
        "Context:\n"
        f"- dataset_size: {state.get('dataset_size')}\n"
        f"- dimension: {state.get('dimension')}\n"
        f"- last_params: {json.dumps(state.get('current_params', {}))}\n"
        f"- last_metrics: {json.dumps(state.get('current_metrics', {}))}\n"
        f"- best_config: {json.dumps(state.get('best_config', {}))}\n"
        f"- recent_trials (most recent first): {json.dumps(recent_trials)}\n"
        + (f"- past_runs_similar (from log): {json.dumps(past_log_trials)}\n\n" if past_log_trials else "\n")
        + "Return ONLY compact JSON with keys hnsw_m, ef_construction, ef_search, rationale."
    )


