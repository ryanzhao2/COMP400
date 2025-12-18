"""
LangGraph workflow nodes for HNSW optimization.

Each node performs a specific step in the optimization process:
- analyze_dataset: Load/generate dataset and queries
- generate_parameters: Propose next parameter configuration
- build_index: Build HNSW index with current parameters
- evaluate_performance: Measure latency and estimate recall
- evaluate_exact_recall: Compute true recall via brute-force search
- update_best_config: Track best configuration found so far
- decide_next_action: Determine whether to continue or terminate
"""

from typing import Any, Dict, List, Optional
import os
import time
import json
import numpy as np
from .hnsw import create_index as _create_index, persist_index as _persist_index
from .utils import format_bytes as _format_bytes, calculate_estimated_recall
from .models import DistanceStats, TrialRecord
from vdb.datasets import generate_gaussian_clusters, generate_queries

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore


def analyze_dataset_node(agent: Any, state: Any) -> Any:
    """
    Load or generate dataset vectors and queries.
    
    Initializes the optimization state with dataset metadata and prepares
    vectors for index building and evaluation.
    """
    print("Analyzing dataset...")
    vectors = None
    queries = None
    try:
        if agent.dataset_vectors_path and os.path.isfile(agent.dataset_vectors_path):
            vectors = np.load(agent.dataset_vectors_path).astype(np.float32)
            if agent.dataset_queries_path and os.path.isfile(agent.dataset_queries_path):
                queries = np.load(agent.dataset_queries_path).astype(np.float32)
    except Exception as e:
        print(f"Warning: Failed to load dataset from disk: {e}. Falling back to synthetic.")

    if vectors is None:
        # Get dataset size from config (database type specific) or environment variable
        database_type = state.get("database_type") or getattr(agent, "database_type", "knowledge_reasoning")
        from vdb.config import get_database_config
        db_config = get_database_config(database_type)
        
        # Priority: environment variable > config > default
        env_dataset_size = os.getenv("DATASET_SIZE")
        if env_dataset_size:
            try:
                dataset_size = int(env_dataset_size)
            except Exception:
                dataset_size = db_config.get("dataset_size", 100000)
        else:
            dataset_size = db_config.get("dataset_size", 100000)
        
        dimension = db_config.get("dimension", 128)
        # Use vdb.datasets generator instead of raw numpy random
        # Assuming 8 clusters and std 0.1 as defaults, or randomizing slightly
        vectors = generate_gaussian_clusters(dataset_size, dimension, num_clusters=8, cluster_std=0.1)
        queries = generate_queries(vectors, num_queries=min(10000, dataset_size))
    else:
        dataset_size, dimension = int(vectors.shape[0]), int(vectors.shape[1])
        if queries is None:
            num_q = min(10000, dataset_size)
            queries = generate_queries(vectors, num_queries=num_q)

    state["dataset_size"] = dataset_size
    state["dimension"] = dimension
    state["database_type"] = state.get("database_type") or getattr(agent, "database_type", "knowledge_reasoning")
    state["_vectors"] = vectors
    state["_queries"] = queries
    state["status"] = "experimenting"
    state["phase"] = state.get("phase") or "recall"
    state["tried_params"] = state.get("tried_params", [])
    state["exploration_count"] = int(state.get("exploration_count", 0))
    print(f"Dataset: {dataset_size} vectors, {dimension}D")
    return state


def _get_exploration_grid(agent: Any) -> List[Dict[str, int]]:
    """
    Generate a grid of parameter combinations for initial exploration.
    
    Creates diverse parameter settings to explore the search space
    before switching to LLM-guided or focused optimization.
    """
    c = agent.constraints
    m_candidates = [c.hnsw_m_min, 16, 32, 48, 64, c.hnsw_m_max]
    m_candidates = sorted({max(c.hnsw_m_min, min(c.hnsw_m_max, v)) for v in m_candidates})
    c_candidates = [c.ef_construction_min, 200, 400, 800, 1200, c.ef_construction_max]
    c_candidates = sorted({max(c.ef_construction_min, min(c.ef_construction_max, v)) for v in c_candidates})
    s_candidates = [c.ef_search_min, 20, 30, 40, 50, 75, 100, 200, 400, 600, 800, c.ef_search_max]
    s_candidates = sorted({max(c.ef_search_min, min(c.ef_search_max, v)) for v in s_candidates})
    combos: List[Dict[str, int]] = []
    for i, s in enumerate(s_candidates):
        m = m_candidates[min(i % len(m_candidates), len(m_candidates) - 1)]
        ec = c_candidates[min((i // 2) % len(c_candidates), len(c_candidates) - 1)]
        combos.append({"hnsw_m": m, "ef_construction": ec, "ef_search": s})
    if len(combos) < agent.initial_exploration_trials:
        for m in m_candidates:
            for ec in c_candidates:
                for s in s_candidates:
                    combos.append({"hnsw_m": m, "ef_construction": ec, "ef_search": s})
                    if len(combos) >= agent.initial_exploration_trials * 2:
                        break
                if len(combos) >= agent.initial_exploration_trials * 2:
                    break
            if len(combos) >= agent.initial_exploration_trials * 2:
                break
    seen = set()
    unique: List[Dict[str, int]] = []
    for d in combos:
        key = (d["hnsw_m"], d["ef_construction"], d["ef_search"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)
    return unique


def generate_parameters_node(agent: Any, state: Any) -> Any:
    """
    Propose next set of HNSW parameters to evaluate.
    
    Uses exploration grid for initial trials, then switches to
    LLM-guided or heuristic parameter generation.
    """
    exploring = state.get("exploration_count", 0) < getattr(agent, "initial_exploration_trials", 0)
    llm_works = getattr(agent, "llm_works", False)
    mode = "exploration" if exploring else ("LLM" if llm_works else "heuristic")
    print(f"Generating parameters ({mode})...")
    proposed = agent.param_proposer.propose(state) if not exploring else {}
    phase = state.get("phase", "recall")
    constraints = agent.constraints
    tried = state.get("tried_params", [])
    tried_set = {(p.get("hnsw_m"), p.get("ef_construction"), p.get("ef_search")) for p in tried}
    if exploring:
        grid = _get_exploration_grid(agent)
        idx = min(len(tried), len(grid) - 1)
        hnsw_m = grid[idx]["hnsw_m"]
        ef_c = grid[idx]["ef_construction"]
        ef_s = grid[idx]["ef_search"]
    else:
        hnsw_m = int(proposed.get("hnsw_m", 16))
        ef_c = int(proposed.get("ef_construction", 200))
        ef_s = int(proposed.get("ef_search", 50))
    # Apply constraints (LLM controls parameter selection - no forced reductions)
    hnsw_m = max(constraints.hnsw_m_min, min(constraints.hnsw_m_max, hnsw_m))
    ef_c = max(constraints.ef_construction_min, min(constraints.ef_construction_max, ef_c))
    ef_s = max(constraints.ef_search_min, min(constraints.ef_search_max, ef_s))
    candidate = {"hnsw_m": hnsw_m, "ef_construction": ef_c, "ef_search": ef_s}
    if (hnsw_m, ef_c, ef_s) in tried_set:
        import random
        for _ in range(5):
            jitter_m = hnsw_m + random.choice([-8, -4, 0, 4, 8])
            jitter_c = ef_c + random.choice([-200, -100, 0, 100, 200])
            jitter_s = ef_s + random.choice([-100, -50, 0, 50, 100])
            jitter_m = max(constraints.hnsw_m_min, min(constraints.hnsw_m_max, jitter_m))
            jitter_c = max(constraints.ef_construction_min, min(constraints.ef_construction_max, jitter_c))
            jitter_s = max(constraints.ef_search_min, min(constraints.ef_search_max, jitter_s))
            if (jitter_m, jitter_c, jitter_s) not in tried_set:
                candidate = {"hnsw_m": jitter_m, "ef_construction": jitter_c, "ef_search": jitter_s}
                break
    state["current_params"] = candidate
    tried.append(candidate.copy())
    state["tried_params"] = tried
    print(f"Generated params ({'explore' if exploring else phase} phase): {state['current_params']}")
    return state


def build_index_node(agent: Any, state: Any) -> Any:
    """
    Build HNSW index with current parameters.
    
    Creates a FAISS HNSW index and measures build time.
    Stores the index in state for reuse in evaluate_performance_node.
    """
    print("Building index...")
    try:
        params = state["current_params"]
        dimension = state["dimension"]
        dataset_size = state["dataset_size"]
        build_start = time.time()
        index = _create_index(dimension, params["hnsw_m"], params["ef_construction"], params["ef_search"])
        vectors = state.get("_vectors")
        if vectors is None or int(vectors.shape[0]) != dataset_size:
            # Should have been handled in analyze_dataset, but as safeguard:
            vectors = generate_gaussian_clusters(dataset_size, dimension)
            state["_vectors"] = vectors
        index.add(vectors)  # type: ignore
        build_ms = (time.time() - build_start) * 1000.0
        state["_last_build_ms"] = float(build_ms)
        state["_index"] = index  # Store index in state for reuse
        print(f"Index build time: {build_ms:.1f} ms")
        print(f"Index built with {dataset_size} vectors")
    except Exception as e:
        print(f"Error building index: {e}")
        state["error_message"] = f"Index building failed: {str(e)}"
        state["status"] = "error"
    return state


def evaluate_performance_node(agent: Any, state: Any) -> Any:
    """
    Measure performance metrics using the index built in build_index_node.
    
    Reuses the index from state if available, otherwise builds a new one.
    Evaluates:
    - Build time (from build_index_node or measured here)
    - Query latency (averaged over query set)
    - Memory usage (vectors + graph estimate)
    - Estimated recall (heuristic)
    """
    print("Evaluating performance...")
    try:
        params = state["current_params"]
        dimension = state["dimension"]
        dataset_size = state["dataset_size"]
        
        # Reuse index from build_index_node if available
        index = state.get("_index")
        if index is not None:
            build_ms = state.get("_last_build_ms", 0.0)
            print(f"Reusing index from build_index_node (saved {build_ms:.1f} ms)")
        else:
            # Fallback: build index here if not already built
            build_start = time.time()
            index = _create_index(dimension, params["hnsw_m"], params["ef_construction"], params["ef_search"])
            vectors = state.get("_vectors")
            if vectors is None or int(vectors.shape[0]) != dataset_size:
                vectors = generate_gaussian_clusters(dataset_size, dimension)
                state["_vectors"] = vectors
            index.add(vectors)  # type: ignore
            build_ms = (time.time() - build_start) * 1000.0
            state["_index"] = index
        index_file_path = None
        index_file_bytes = None
        # Persisting very large indexes (e.g., 10M vectors) can be slow. Only write to disk
        # for smaller datasets and skip for large ones to speed up experiments.
        try:
            dataset_size_int = int(dataset_size)
        except Exception:
            dataset_size_int = dataset_size
        if dataset_size_int <= 1_000_000:
            try:
                index_file_path = os.getenv("INDEX_FILE_PATH", os.path.join("data", "hnsw.index"))
                index_file_bytes = _persist_index(index, index_file_path)
            except Exception:
                index_file_path = None
                index_file_bytes = None
        queries = state.get("_queries")
        if queries is None or int(queries.shape[1]) != dimension:
            num_queries = min(10000, dataset_size)
            # Use generator
            vectors = state.get("_vectors")
            queries = generate_queries(vectors, num_queries=num_queries)
            state["_queries"] = queries
        num_queries = int(queries.shape[0])
        # Set ef_search before searching (required when reusing index)
        index.hnsw.efSearch = params["ef_search"]  # type: ignore
        start_time = time.time()
        index.search(queries, k=10)  # type: ignore
        search_time = (time.time() - start_time) * 1000
        avg_latency = search_time / num_queries
        
        # Use helper for estimated recall
        estimated_recall = calculate_estimated_recall(params["ef_search"], agent.constraints.ef_search_max)
        
        vector_bytes = int(index.ntotal) * int(dimension) * 4
        graph_bytes_est = int(index.ntotal) * int(params["hnsw_m"]) * 4
        total_bytes_est = vector_bytes + graph_bytes_est
        memory_usage_gb = vector_bytes / (1024**3)
        memory_usage_total_gb = total_bytes_est / (1024**3)
        metrics = {
            "recall": estimated_recall,
            "latency_ms": avg_latency,
            "memory_gb": memory_usage_gb,
            "memory_total_gb_est": memory_usage_total_gb,
            "index_size": index.ntotal,
            "build_ms": float(build_ms),
            "index_file_path": index_file_path,
            "vector_db_disk_size": (_format_bytes(int(index_file_bytes)) if index_file_bytes is not None else None),
            "vector_size": _format_bytes(int(vector_bytes)),
            "graph_size_est": _format_bytes(int(graph_bytes_est)),
            "total_size_est": _format_bytes(int(total_bytes_est))
        }
        state["current_metrics"] = metrics
        print(
            "Metrics: "
            f"Recall={metrics['recall']:.3f}, "
            f"Latency={metrics['latency_ms']:.4f}ms, "
            f"Memory={metrics['memory_gb']:.4f}GB"
        )
    except Exception as e:
        print(f"Error evaluating performance: {e}")
        state["error_message"] = f"Performance evaluation failed: {str(e)}"
        state["status"] = "error"
    return state


def evaluate_exact_recall_node(agent: Any, state: Any) -> Any:
    """
    Compute true recall@k using brute-force exact search.
    
    For small datasets (<1GB), performs exhaustive search to measure
    actual recall. For larger datasets, uses heuristic to avoid expensive computation.
    """
    print("Computing exact recall@k...")
    try:
        params = state["current_params"]
        dimension = state["dimension"]
        dataset_size = state["dataset_size"]
        k = 10
        
        # Calculate database size in MB
        vectors = state.get("_vectors")
        if vectors is None or int(vectors.shape[0]) != dataset_size:
            vectors = generate_gaussian_clusters(dataset_size, dimension)
            state["_vectors"] = vectors
        
        # Calculate total size: vectors + graph estimate
        vector_bytes = int(dataset_size) * int(dimension) * 4
        graph_bytes_est = int(dataset_size) * int(params["hnsw_m"]) * 4
        total_bytes_est = vector_bytes + graph_bytes_est
        total_size_mb = total_bytes_est / (1024 * 1024)  # Convert to MB
        
        # For large databases (>1GB), use heuristic instead of brute force
        if total_size_mb > 1024:
            print(f"Database size ({total_size_mb:.1f} MB) > 100 MB. Using heuristic instead of brute force.")
            # Use helper
            estimated_recall = calculate_estimated_recall(params["ef_search"], agent.constraints.ef_search_max)
            metrics = state.get("current_metrics", {})
            metrics["true_recall"] = estimated_recall
            metrics["k"] = k
            metrics["recall_method"] = "heuristic"
            state["current_metrics"] = metrics
            print(f"Estimated Recall@{k} (heuristic): {estimated_recall:.3f}")
            return state
        
        # For smaller databases, use brute force exact search
        # Reuse index from build_index_node if available (saves rebuild time)
        ann_index = state.get("_index")
        if ann_index is None:
            # Build new index if not already in state
            ann_index = _create_index(dimension, params["hnsw_m"], params["ef_construction"], params["ef_search"])
            ann_index.add(vectors)  # type: ignore
        # If index was reused from state, vectors are already added in build_index_node
        queries = state.get("_queries")
        if queries is None or int(queries.shape[1]) != dimension:
            num_queries = min(10000, dataset_size)
            queries = generate_queries(vectors, num_queries=num_queries)
            state["_queries"] = queries
        num_queries = int(queries.shape[0])
        ann_index.hnsw.efSearch = params["ef_search"]  # type: ignore
        _, ann_idx = ann_index.search(queries, k)  # type: ignore
        gt_dist = None
        gt_idx = None
        use_gpu = False
        try:
            use_gpu = bool(agent.use_gpu_exact) and hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0  # type: ignore
        except Exception:
            use_gpu = False
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()  # type: ignore
                flat_cpu = faiss.IndexFlatL2(dimension)
                exact_index = faiss.index_cpu_to_gpu(res, 0, flat_cpu)  # type: ignore
                exact_index.add(vectors)  # type: ignore
                gt_dist, gt_idx = exact_index.search(queries, k)  # type: ignore
            except Exception:
                exact_index = faiss.IndexFlatL2(dimension)
                exact_index.add(vectors)  # type: ignore
                gt_dist, gt_idx = exact_index.search(queries, k)  # type: ignore
        else:
            exact_index = faiss.IndexFlatL2(dimension)
            exact_index.add(vectors)  # type: ignore
            gt_dist, gt_idx = exact_index.search(queries, k)  # type: ignore
        recalls = []
        for i in range(num_queries):
            ann_set = set(ann_idx[i].tolist())
            gt_set = set(gt_idx[i].tolist())
            inter = len(ann_set & gt_set)
            recalls.append(inter / k)
        true_recall = float(np.mean(recalls))
        metrics = state.get("current_metrics", {})
        metrics["true_recall"] = true_recall
        metrics["recall"] = true_recall
        metrics["k"] = k
        metrics["recall_method"] = "exact"
        state["current_metrics"] = metrics
        print(f"True Recall@{k}: {true_recall:.3f}")
    except Exception as e:
        print(f"Error computing exact recall: {e}")
    return state


def update_best_config_node(agent: Any, state: Any) -> Any:
    """
    Update best configuration and log experiment results.
    
    Compares current metrics against best known configuration and
    records all trials to JSONL log for analysis.
    """
    print("Updating best configuration...")
    current_metrics = state["current_metrics"]
    current_params = state["current_params"]
    phase = state.get("phase", "recall")
    database_type = state.get("database_type") or getattr(agent, "database_type", "knowledge_reasoning")
    if not state["best_config"] or agent._is_better_config(current_metrics, state["best_config"].get("metrics", {}), phase=phase, database_type=database_type):
        state["best_config"] = {
            "params": current_params.copy(),
            "metrics": current_metrics.copy(),
            "iteration": state["iteration_count"]
        }
        print("New best configuration found!")
    experiment = {
        "iteration": state["iteration_count"],
        "params": current_params.copy(),
        "metrics": current_metrics.copy(),
        "timestamp": time.time(),
        "datetime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    state["experiment_history"].append(experiment)
    state["exploration_count"] = int(state.get("exploration_count", 0)) + 1
    if agent.archivist is not None:
        try:
            ds = {
                "dataset_size": state.get("dataset_size"),
                "dimension": state.get("dimension"),
                "database_type": state.get("database_type") or getattr(agent, "database_type", "knowledge_reasoning"),
            }
            agent.archivist.log({"database_type": state.get("database_type") or getattr(agent, "database_type", "knowledge_reasoning"), "dataset": ds, **experiment})
        except Exception as e:
            print(f"Warning: Logging failed: {e}")
    state["iteration_count"] += 1
    return state


def decide_next_action_node(agent: Any, state: Any) -> Any:
    """
    Decide whether to continue optimization or terminate.
    
    Checks:
    - Maximum experiments reached
    - Phase transitions (recall -> latency)
    - Performance targets met
    """
    print("Deciding next action...")
    if state["iteration_count"] >= agent.thresholds.max_experiments:
        print("Maximum experiments reached")
        state["status"] = "done"
        return state
    exploration_count = int(state.get("exploration_count", 0))
    if exploration_count < getattr(agent, "initial_exploration_trials", 0):
        state["status"] = "experimenting"
        return state
    current_metrics = state["current_metrics"]
    phase = state.get("phase", "recall")
    if phase == "recall" and current_metrics.get("recall", 0.0) >= agent.thresholds.min_recall:
        print("Recall target met. Switching to latency optimization phase.")
        state["phase"] = "latency"
        state["status"] = "experimenting"
        return state
    if phase == "latency" and current_metrics.get("recall", 0.0) < agent.thresholds.min_recall:
        print("Warning: Recall dropped below target. Returning to recall phase.")
        state["phase"] = "recall"
        state["status"] = "experimenting"
        return state
    state["status"] = "experimenting"
    return state
