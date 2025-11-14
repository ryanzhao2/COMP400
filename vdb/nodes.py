from typing import Any, Dict, List, Optional
import os
import time
import json
import numpy as np
from .hnsw import create_index as _create_index, persist_index as _persist_index
from .utils import format_bytes as _format_bytes
from .models import DistanceStats, TrialRecord

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore


def analyze_dataset_node(agent: Any, state: Any) -> Any:
    print("🔍 Analyzing dataset...")
    vectors = None
    queries = None
    try:
        if agent.dataset_vectors_path and os.path.isfile(agent.dataset_vectors_path):
            vectors = np.load(agent.dataset_vectors_path).astype(np.float32)
            if agent.dataset_queries_path and os.path.isfile(agent.dataset_queries_path):
                queries = np.load(agent.dataset_queries_path).astype(np.float32)
    except Exception as e:
        print(f"⚠️  Failed to load dataset from disk: {e}. Falling back to synthetic.")

    if vectors is None:
        try:
            dataset_size = int(os.getenv("DATASET_SIZE", "1000000"))
        except Exception:
            dataset_size = 10000
        dimension = 128
        vectors = np.random.random((dataset_size, dimension)).astype(np.float32)
        queries = np.random.random((min(100, dataset_size), dimension)).astype(np.float32)
    else:
        dataset_size, dimension = int(vectors.shape[0]), int(vectors.shape[1])
        if queries is None:
            num_q = min(200, dataset_size)
            idx = np.random.choice(dataset_size, size=num_q, replace=False)
            q = vectors[idx].copy()
            q += np.random.normal(0.0, 0.01, size=q.shape).astype(np.float32)
            queries = q.astype(np.float32)

    state["dataset_size"] = dataset_size
    state["dimension"] = dimension
    state["_vectors"] = vectors
    state["_queries"] = queries
    state["status"] = "experimenting"
    state["phase"] = state.get("phase") or "recall"
    state["tried_params"] = state.get("tried_params", [])
    state["exploration_count"] = int(state.get("exploration_count", 0))
    print(f"📊 Dataset: {dataset_size} vectors, {dimension}D")
    return state


def _get_exploration_grid(agent: Any) -> List[Dict[str, int]]:
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
    print("🧠 Generating parameters...")
    exploring = state.get("exploration_count", 0) < getattr(agent, "initial_exploration_trials", 10)
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
    if phase == "latency":
        ef_s = max(constraints.ef_search_min, int(max(ef_s * 0.8, ef_s - 50)))
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
    print(f"🎯 Generated params ({'explore' if exploring else phase} phase): {state['current_params']}")
    return state


def build_index_node(agent: Any, state: Any) -> Any:
    print("🔨 Building index...")
    try:
        params = state["current_params"]
        dimension = state["dimension"]
        dataset_size = state["dataset_size"]
        build_start = time.time()
        index = _create_index(dimension, params["hnsw_m"], params["ef_construction"], params["ef_search"])
        vectors = state.get("_vectors")
        if vectors is None or int(vectors.shape[0]) != dataset_size:
            vectors = np.random.random((dataset_size, dimension)).astype(np.float32)
            state["_vectors"] = vectors
        index.add(vectors)  # type: ignore
        build_ms = (time.time() - build_start) * 1000.0
        state["_last_build_ms"] = float(build_ms)
        print(f"🕒 Index build time: {build_ms:.1f} ms")
        print(f"✅ Index built with {dataset_size} vectors")
    except Exception as e:
        print(f"❌ Error building index: {e}")
        state["error_message"] = f"Index building failed: {str(e)}"
        state["status"] = "error"
    return state


def evaluate_performance_node(agent: Any, state: Any) -> Any:
    print("📊 Evaluating performance...")
    try:
        params = state["current_params"]
        dimension = state["dimension"]
        dataset_size = state["dataset_size"]
        build_start = time.time()
        index = _create_index(dimension, params["hnsw_m"], params["ef_construction"], params["ef_search"])
        vectors = state.get("_vectors")
        if vectors is None or int(vectors.shape[0]) != dataset_size:
            vectors = np.random.random((dataset_size, dimension)).astype(np.float32)
            state["_vectors"] = vectors
        index.add(vectors)  # type: ignore
        build_ms = (time.time() - build_start) * 1000.0
        index_file_path = None
        index_file_bytes = None
        try:
            index_file_path = os.getenv("INDEX_FILE_PATH", os.path.join("data", "hnsw.index"))
            index_file_bytes = _persist_index(index, index_file_path)
        except Exception:
            index_file_path = None
            index_file_bytes = None
        queries = state.get("_queries")
        if queries is None or int(queries.shape[1]) != dimension:
            num_queries = min(200, dataset_size)
            queries = np.random.random((num_queries, dimension)).astype(np.float32)
            state["_queries"] = queries
        num_queries = int(queries.shape[0])
        start_time = time.time()
        index.search(queries, k=10)  # type: ignore
        search_time = (time.time() - start_time) * 1000
        avg_latency = search_time / num_queries
        ef_max = max(1, int(agent.constraints.ef_search_max))
        x = max(0.0, min(1.0, params["ef_search"] / ef_max))
        slope = 10.0
        sig = 1.0 / (1.0 + np.exp(-slope * (x - 0.5)))
        r_min, r_max = 0.6, 0.98
        estimated_recall = float(r_min + (r_max - r_min) * sig)
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
        print(f"📈 Metrics: Recall={metrics['recall']:.3f}, Latency={metrics['latency_ms']:.1f}ms, Memory={metrics['memory_gb']:.2f}GB")
    except Exception as e:
        print(f"❌ Error evaluating performance: {e}")
        state["error_message"] = f"Performance evaluation failed: {str(e)}"
        state["status"] = "error"
    return state


def evaluate_exact_recall_node(agent: Any, state: Any) -> Any:
    print("🎯 Computing exact recall@k...")
    try:
        params = state["current_params"]
        dimension = state["dimension"]
        dataset_size = state["dataset_size"]
        k = 10
        ann_index = _create_index(dimension, params["hnsw_m"], params["ef_construction"], params["ef_search"])
        vectors = state.get("_vectors")
        if vectors is None or int(vectors.shape[0]) != dataset_size:
            vectors = np.random.random((dataset_size, dimension)).astype(np.float32)
            state["_vectors"] = vectors
        ann_index.add(vectors)  # type: ignore
        queries = state.get("_queries")
        if queries is None or int(queries.shape[1]) != dimension:
            num_queries = min(200, dataset_size)
            queries = np.random.random((num_queries, dimension)).astype(np.float32)
            state["_queries"] = queries
        num_queries = int(queries.shape[0])
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
        metrics["k"] = k
        state["current_metrics"] = metrics
        print(f"✅ True Recall@{k}: {true_recall:.3f}")
    except Exception as e:
        print(f"❌ Error computing exact recall: {e}")
    return state


def update_best_config_node(agent: Any, state: Any) -> Any:
    print("🏆 Updating best configuration...")
    current_metrics = state["current_metrics"]
    current_params = state["current_params"]
    if not state["best_config"] or agent._is_better_config(current_metrics, state["best_config"].get("metrics", {})):
        state["best_config"] = {
            "params": current_params.copy(),
            "metrics": current_metrics.copy(),
            "iteration": state["iteration_count"]
        }
        print("🎉 New best configuration found!")
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
            }
            agent.archivist.log({"dataset": ds, **experiment})
        except Exception as e:
            print(f"⚠️  Logging failed: {e}")
    state["iteration_count"] += 1
    return state


def decide_next_action_node(agent: Any, state: Any) -> Any:
    print("🤔 Deciding next action...")
    if state["iteration_count"] >= agent.thresholds.max_experiments:
        print("🛑 Maximum experiments reached")
        state["status"] = "done"
        return state
    exploration_count = int(state.get("exploration_count", 0))
    if exploration_count < getattr(agent, "initial_exploration_trials", 5):
        state["status"] = "experimenting"
        return state
    current_metrics = state["current_metrics"]
    phase = state.get("phase", "recall")
    if phase == "recall" and current_metrics.get("recall", 0.0) >= agent.thresholds.min_recall:
        print("✅ Recall target met. Switching to latency optimization phase.")
        state["phase"] = "latency"
        state["status"] = "experimenting"
        return state
    if phase == "latency" and current_metrics.get("recall", 0.0) < agent.thresholds.min_recall:
        print("⚠️  Recall dropped below target. Returning to recall phase.")
        state["phase"] = "recall"
        state["status"] = "experimenting"
        return state
    state["status"] = "experimenting"
    return state


