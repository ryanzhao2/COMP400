"""
LangGraph-based Agentic AI for Vector Database Optimization
A basic template for tuning FAISS HNSW parameters using intelligent agents.
"""

import os
import time
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, TypedDict, Optional
from dataclasses import dataclass
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# FAISS imports
try:
    import faiss
except ImportError:
    raise ImportError("FAISS not installed. Run: pip install faiss-cpu")

# LangGraph imports
from langgraph.graph import StateGraph, END
try:
    import google.genai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:
    ChatGoogleGenerativeAI = None  # type: ignore
    SystemMessage = None  # type: ignore
    HumanMessage = None  # type: ignore
from langchain_core.messages import SystemMessage, HumanMessage

# Configuration
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

# -------- Experiment logging (Archivist) --------
class ArchivistAgent:
    """Append experiment records to a JSONL file with a run_id."""
    def __init__(self, log_path: Optional[str], run_id: str):
        self.log_path = log_path or os.getenv("EXPERIMENT_LOG", "experiments.jsonl")
        self.run_id = run_id

    def log(self, record: Dict[str, Any]) -> None:
        try:
            enriched = {"run_id": self.run_id, **record}
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(enriched) + "\n")
        except Exception as e:
            print(f"⚠️  Failed to write experiment record: {e}")

# -------- Sub-agents --------
class ParamProposerAgent:
    """Delegates to LLM or heuristic to propose parameters."""
    def __init__(self, parent: "VectorDatabaseAgent"):
        self.parent = parent

    def propose(self, state: "OptimizationState") -> Dict[str, int]:
        if self.parent.lc_llm or self.parent.genai_client:
            return self.parent._generate_llm_parameters(state)
        return self.parent._generate_fallback_parameters(state)


# (Note) We compute exact recall inline in the evaluation node below.
# A prior helper class for exact recall was removed to reduce duplication.



# -------- JSON I/O helpers for GraphData --------
def graph_data_to_dict(graph: "GraphData") -> Dict[str, Any]:
    return {
        "dataset_size": graph.dataset_size,
        "dimension": graph.dimension,
        "global_distance_stats": {
            "mean": graph.global_distance_stats.mean,
            "std": graph.global_distance_stats.std,
            "minimum": graph.global_distance_stats.minimum,
            "maximum": graph.global_distance_stats.maximum,
            "p25": graph.global_distance_stats.p25,
            "p50": graph.global_distance_stats.p50,
            "p75": graph.global_distance_stats.p75,
        },
        "trials": [
            {
                "params": t.params,
                "metrics": t.metrics,
                "distance_stats": {
                    "mean": t.distance_stats.mean,
                    "std": t.distance_stats.std,
                    "minimum": t.distance_stats.minimum,
                    "maximum": t.distance_stats.maximum,
                    "p25": t.distance_stats.p25,
                    "p50": t.distance_stats.p50,
                    "p75": t.distance_stats.p75,
                },
            }
            for t in graph.trials
        ],
    }


def graph_data_from_dict(d: Dict[str, Any]) -> "GraphData":
    gstats = d.get("global_distance_stats", {})
    global_stats = DistanceStats(
        mean=float(gstats.get("mean", 0.0)),
        std=float(gstats.get("std", 0.0)),
        minimum=float(gstats.get("minimum", 0.0)),
        maximum=float(gstats.get("maximum", 0.0)),
        p25=float(gstats.get("p25", 0.0)),
        p50=float(gstats.get("p50", 0.0)),
        p75=float(gstats.get("p75", 0.0)),
    )

    trials: List[TrialRecord] = []
    for td in d.get("trials", []):
        dstats_raw = td.get("distance_stats", {})
        dstats = DistanceStats(
            mean=float(dstats_raw.get("mean", 0.0)),
            std=float(dstats_raw.get("std", 0.0)),
            minimum=float(dstats_raw.get("minimum", 0.0)),
            maximum=float(dstats_raw.get("maximum", 0.0)),
            p25=float(dstats_raw.get("p25", 0.0)),
            p50=float(dstats_raw.get("p50", 0.0)),
            p75=float(dstats_raw.get("p75", 0.0)),
        )
        trials.append(
            TrialRecord(
                params={k: int(v) for k, v in td.get("params", {}).items()},
                metrics={k: float(v) for k, v in td.get("metrics", {}).items()},
                distance_stats=dstats,
            )
        )

    return GraphData(
        dataset_size=int(d.get("dataset_size", 0)),
        dimension=int(d.get("dimension", 0)),
        global_distance_stats=global_stats,
        trials=trials,
    )


def save_graph_data_json(graph: "GraphData", path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph_data_to_dict(graph), f, indent=2)


def load_graph_data_json(path: str) -> "GraphData":
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return graph_data_from_dict(data)

# State definition
class OptimizationState(TypedDict):
    """State for the optimization workflow"""
    dataset_size: int
    dimension: int
    current_params: Dict[str, int]
    current_metrics: Dict[str, float]
    experiment_history: List[Dict[str, Any]]
    best_config: Dict[str, Any]
    iteration_count: int
    status: str
    error_message: Optional[str]
    # Internal cached data (not persisted):
    _vectors: Optional[Any]
    _queries: Optional[Any]
    _last_build_ms: Optional[float]
    phase: str  # "recall" or "latency"
    tried_params: List[Dict[str, int]]
    exploration_count: int

class VectorDatabaseAgent:
    """Basic agent for vector database optimization"""
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 constraints: Optional[ParameterConstraints] = None,
                 thresholds: Optional[PerformanceThresholds] = None,
                 log_path: Optional[str] = None,
                 dataset_vectors_path: Optional[str] = None,
                 dataset_queries_path: Optional[str] = None,
                 num_threads: Optional[int] = None,
                 use_gpu_exact: Optional[bool] = None,
                 initial_exploration_trials: int = 10):
        
        # Set up LLMs (prefer LangChain wrapper, fallback to google-genai client)
        self.genai_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        gemini_key = os.getenv("GEMINI_API_KEY")
        self.lc_llm = None
        self.genai_client = None
        if gemini_key and ChatGoogleGenerativeAI is not None:
            try:
                # Ensure LangChain sees the key
                if not os.getenv("GOOGLE_API_KEY"):
                    os.environ["GOOGLE_API_KEY"] = gemini_key
                try:
                    self.lc_llm = ChatGoogleGenerativeAI(model=self.genai_model, temperature=0.1, google_api_key=gemini_key)
                except Exception:
                    # Fallback to a widely available model name
                    self.lc_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1, google_api_key=gemini_key)
            except Exception as e:
                print(f"⚠️  Failed to initialize LangChain Gemini: {e}. Will try direct client.")
        if gemini_key and self.lc_llm is None and genai is not None:
            try:
                self.genai_client = genai.Client(api_key=gemini_key)
            except Exception as e:
                print(f"⚠️  Failed to initialize Google GenAI client: {e}. Using heuristic fallback.")
        
        # Set up constraints and thresholds
        self.constraints = constraints or ParameterConstraints()
        self.thresholds = thresholds or PerformanceThresholds()
        self.run_id: Optional[str] = None
        self.archivist: Optional[ArchivistAgent] = ArchivistAgent(log_path or os.getenv("EXPERIMENT_LOG", "experiments.jsonl"), run_id=str(uuid.uuid4()))
        self.param_proposer = ParamProposerAgent(self)
        # Optional dataset paths
        self.dataset_vectors_path = dataset_vectors_path
        self.dataset_queries_path = dataset_queries_path
        # GPU for exact baseline (optional)
        env_use_gpu = os.getenv("FAISS_USE_GPU_EXACT")
        self.use_gpu_exact = (use_gpu_exact if use_gpu_exact is not None else (env_use_gpu == "1" or env_use_gpu == "true"))
        # Initial exploration trials before convergence
        self.initial_exploration_trials = max(0, int(os.getenv("INITIAL_EXPLORATION_TRIALS", initial_exploration_trials)))
        # Threads (FAISS OpenMP)
        try:
            env_threads = os.getenv("FAISS_NUM_THREADS")
            threads = num_threads or (int(env_threads) if env_threads else None) or (os.cpu_count() or 1)
            faiss.omp_set_num_threads(int(threads))  # type: ignore
        except Exception:
            pass
        # LLM history configuration
        try:
            self.llm_history_trials = int(os.getenv("LLM_HISTORY_TRIALS", "5"))
        except Exception:
            self.llm_history_trials = 5
        self.include_past_log_history = os.getenv("INCLUDE_PAST_LOG_HISTORY", "0").lower() in ("1", "true", "yes")
        
        # Build the workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> Any:
        """Build the LangGraph workflow"""
        workflow = StateGraph(OptimizationState)
        
        # Add nodes
        workflow.add_node("analyze_dataset", self._analyze_dataset_node)
        workflow.add_node("generate_parameters", self._generate_parameters_node)
        workflow.add_node("build_index", self._build_index_node)
        workflow.add_node("evaluate_performance", self._evaluate_performance_node)
        workflow.add_node("evaluate_exact_recall", self._evaluate_exact_recall_node)
        workflow.add_node("update_best_config", self._update_best_config_node)
        workflow.add_node("decide_next_action", self._decide_next_action_node)
        
        # Set entry point
        workflow.set_entry_point("analyze_dataset")
        
        # Add edges
        workflow.add_edge("analyze_dataset", "generate_parameters")
        workflow.add_edge("generate_parameters", "build_index")
        workflow.add_edge("build_index", "evaluate_performance")
        workflow.add_edge("evaluate_performance", "evaluate_exact_recall")
        workflow.add_edge("evaluate_exact_recall", "update_best_config")
        workflow.add_edge("update_best_config", "decide_next_action")
        
        # Conditional edges
        workflow.add_conditional_edges(
            "decide_next_action",
            self._should_continue,
            {
                "continue": "generate_parameters",
                "done": END
            }
        )
        
        return workflow.compile()
    
    # Node implementations
    def _analyze_dataset_node(self, state: OptimizationState) -> OptimizationState:
        """Analyze or prepare dataset and queries used across nodes in this run.

        - Tries to load vectors/queries from disk if paths were provided.
        - Otherwise generates a one-off synthetic dataset and query set.
        - Stores both in state so all subsequent nodes reuse the same data.
        """
        print("🔍 Analyzing dataset...")
        
        # Load provided dataset if available; otherwise generate once and reuse
        vectors = None
        queries = None
        try:
            if self.dataset_vectors_path and os.path.isfile(self.dataset_vectors_path):
                vectors = np.load(self.dataset_vectors_path).astype(np.float32)
                if self.dataset_queries_path and os.path.isfile(self.dataset_queries_path):
                    queries = np.load(self.dataset_queries_path).astype(np.float32)
        except Exception as e:
            print(f"⚠️  Failed to load dataset from disk: {e}. Falling back to synthetic.")
        
        if vectors is None:
            # Generate synthetic dataset once and store in state for reuse
            dataset_size = 10000
            dimension = 128
            vectors = np.random.random((dataset_size, dimension)).astype(np.float32)
            queries = np.random.random((min(100, dataset_size), dimension)).astype(np.float32)
        else:
            dataset_size, dimension = int(vectors.shape[0]), int(vectors.shape[1])
            if queries is None:
                # Create queries near the data manifold
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
    
    def _generate_parameters_node(self, state: OptimizationState) -> OptimizationState:
        """Generate HNSW parameters for the next trial.

        Behavior:
        - First N trials (exploration): sample diverse params across ranges.
        - Afterwards: phase-aware proposals; reduce ef_search in 'latency' phase.
        - Avoids duplicate parameter sets tried earlier in this run.
        """
        print("🧠 Generating parameters...")
        
        # Exploration first: diversify parameters for the first N trials
        exploring = state.get("exploration_count", 0) < getattr(self, "initial_exploration_trials", 10)
        proposed = self.param_proposer.propose(state) if not exploring else {}
        # Phase-aware adjustment and diversification
        phase = state.get("phase", "recall")
        constraints = self.constraints
        tried = state.get("tried_params", [])
        tried_set = { (p.get("hnsw_m"), p.get("ef_construction"), p.get("ef_search")) for p in tried }
        if exploring:
            # Deterministic coverage over a small grid to ensure diversity
            grid = self._get_exploration_grid()
            idx = min(len(tried), len(grid) - 1)
            hnsw_m = grid[idx]["hnsw_m"]
            ef_c = grid[idx]["ef_construction"]
            ef_s = grid[idx]["ef_search"]
        else:
            hnsw_m = int(proposed.get("hnsw_m", 16))
            ef_c = int(proposed.get("ef_construction", 200))
            ef_s = int(proposed.get("ef_search", 50))
        # In latency phase, bias toward reducing ef_search to lower latency while maintaining recall
        if phase == "latency":
            ef_s = max(constraints.ef_search_min, int(max(ef_s * 0.8, ef_s - 50)))
        # Clamp within constraints
        hnsw_m = max(constraints.hnsw_m_min, min(constraints.hnsw_m_max, hnsw_m))
        ef_c = max(constraints.ef_construction_min, min(constraints.ef_construction_max, ef_c))
        ef_s = max(constraints.ef_search_min, min(constraints.ef_search_max, ef_s))
        candidate = {"hnsw_m": hnsw_m, "ef_construction": ef_c, "ef_search": ef_s}
        # Avoid duplicates by random perturbation if already tried
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
        # Track tried params
        tried.append(candidate.copy())
        state["tried_params"] = tried
        
        print(f"🎯 Generated params ({'explore' if exploring else phase} phase): {state['current_params']}")
        return state

    def _get_exploration_grid(self) -> List[Dict[str, int]]:
        """Return a small, diverse grid of parameter combinations for exploration."""
        c = self.constraints
        # Candidates across range; dedup by set conversion later
        m_candidates = [c.hnsw_m_min, 16, 32, 48, 64, c.hnsw_m_max]
        m_candidates = sorted({max(c.hnsw_m_min, min(c.hnsw_m_max, v)) for v in m_candidates})
        c_candidates = [c.ef_construction_min, 200, 400, 800, 1200, c.ef_construction_max]
        c_candidates = sorted({max(c.ef_construction_min, min(c.ef_construction_max, v)) for v in c_candidates})
        s_candidates = [c.ef_search_min, 20, 30, 40, 50, 75, 100, 200, 400, 600, 800, c.ef_search_max]
        s_candidates = sorted({max(c.ef_search_min, min(c.ef_search_max, v)) for v in s_candidates})
        # Build a curated set: combine positions across the lists to avoid full Cartesian blow-up
        combos: List[Dict[str, int]] = []
        for i, s in enumerate(s_candidates):
            m = m_candidates[min(i % len(m_candidates), len(m_candidates)-1)]
            ec = c_candidates[min((i // 2) % len(c_candidates), len(c_candidates)-1)]
            combos.append({"hnsw_m": m, "ef_construction": ec, "ef_search": s})
        # Ensure we have at least initial_exploration_trials combos
        if len(combos) < self.initial_exploration_trials:
            # Add more by rotating pairs
            for m in m_candidates:
                for ec in c_candidates:
                    for s in s_candidates:
                        combos.append({"hnsw_m": m, "ef_construction": ec, "ef_search": s})
                        if len(combos) >= self.initial_exploration_trials * 2:
                            break
                    if len(combos) >= self.initial_exploration_trials * 2:
                        break
                if len(combos) >= self.initial_exploration_trials * 2:
                    break
        # Deduplicate preserving order
        seen = set()
        unique: List[Dict[str, int]] = []
        for d in combos:
            key = (d["hnsw_m"], d["ef_construction"], d["ef_search"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(d)
        return unique
    
    def _build_index_node(self, state: OptimizationState) -> OptimizationState:
        """Build a fresh FAISS HNSW index and add all vectors.

        Note: For clarity we build per node; caching the index can reduce latency,
        but is omitted here to keep node boundaries explicit and state minimal.
        """
        print("🔨 Building index...")
        
        try:
            params = state["current_params"]
            dimension = state["dimension"]
            dataset_size = state["dataset_size"]
            
            # Create HNSW index
            build_start = time.time()
            index = faiss.IndexHNSWFlat(dimension, params["hnsw_m"])
            index.hnsw.efConstruction = params["ef_construction"]
            index.hnsw.efSearch = params["ef_search"]
            
            # Use provided/generated dataset (reused across nodes)
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
    
    def _evaluate_performance_node(self, state: OptimizationState) -> OptimizationState:
        """Evaluate index performance (ANN search).

        Measures:
        - Average query latency (ms/query) over a batch of queries.
        - Estimated recall via a bounded sigmoid growth model of ef_search.
        - Rough memory footprint based on vector storage (does not include graph overhead).

        Note: This node rebuilds the index for isolation. It could reuse the index
        built earlier in the iteration to save time; we leave it explicit for readability.
        """
        print("📊 Evaluating performance...")
        
        try:
            params = state["current_params"]
            dimension = state["dimension"]
            dataset_size = state["dataset_size"]
            
            # Rebuild index for evaluation
            build_start = time.time()
            index = faiss.IndexHNSWFlat(dimension, params["hnsw_m"])
            index.hnsw.efConstruction = params["ef_construction"]
            index.hnsw.efSearch = params["ef_search"]
            
            vectors = state.get("_vectors")
            if vectors is None or int(vectors.shape[0]) != dataset_size:
                vectors = np.random.random((dataset_size, dimension)).astype(np.float32)
                state["_vectors"] = vectors
            index.add(vectors)  # type: ignore
            build_ms = (time.time() - build_start) * 1000.0
            
            # Generate test queries
            queries = state.get("_queries")
            if queries is None or int(queries.shape[1]) != dimension:
                num_queries = min(200, dataset_size)
                queries = np.random.random((num_queries, dimension)).astype(np.float32)
                state["_queries"] = queries
            num_queries = int(queries.shape[0])
            
            # Measure performance
            start_time = time.time()
            distances, indices = index.search(queries, k=10)  # type: ignore
            search_time = (time.time() - start_time) * 1000
            
            # Calculate metrics
            avg_latency = search_time / num_queries
            # Sigmoid growth model for estimated recall (proxy): saturates between r_min and r_max
            ef_max = max(1, int(self.constraints.ef_search_max))
            x = max(0.0, min(1.0, params["ef_search"] / ef_max))
            slope = 10.0  # larger -> sharper transition around mid-range
            sig = 1.0 / (1.0 + np.exp(-slope * (x - 0.5)))
            r_min, r_max = 0.6, 0.98
            estimated_recall = float(r_min + (r_max - r_min) * sig)
            memory_usage_gb = (index.ntotal * dimension * 4) / (1024**3)
            
            metrics = {
                "recall": estimated_recall,
                "latency_ms": avg_latency,
                "memory_gb": memory_usage_gb,
                "index_size": index.ntotal,
                "build_ms": float(build_ms)
            }
            
            state["current_metrics"] = metrics
            
            print(f"📈 Metrics: Recall={metrics['recall']:.3f}, "
                  f"Latency={metrics['latency_ms']:.1f}ms, "
                  f"Memory={metrics['memory_gb']:.2f}GB")
            
        except Exception as e:
            print(f"❌ Error evaluating performance: {e}")
            state["error_message"] = f"Performance evaluation failed: {str(e)}"
            state["status"] = "error"
        
        return state
    
    def _evaluate_exact_recall_node(self, state: OptimizationState) -> OptimizationState:
        """Compute true recall@k using an exact FAISS baseline (IndexFlatL2).

        - Runs ANN (HNSW) search and exact (FlatL2) search on the same queries.
        - Computes average overlap fraction over queries.
        - If FAISS GPU is available and enabled, the FlatL2 baseline runs on GPU.
        """
        print("🎯 Computing exact recall@k...")
        try:
            params = state["current_params"]
            dimension = state["dimension"]
            dataset_size = state["dataset_size"]
            k = 10

            # Build HNSW index and run ANN search
            ann_index = faiss.IndexHNSWFlat(dimension, params["hnsw_m"])
            ann_index.hnsw.efConstruction = params["ef_construction"]
            ann_index.hnsw.efSearch = params["ef_search"]

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
            ann_dist, ann_idx = ann_index.search(queries, k)  # type: ignore

            # Build exact baseline (L2) and run exact search (GPU if available/enabled)
            gt_dist = None
            gt_idx = None
            use_gpu = False
            try:
                use_gpu = bool(self.use_gpu_exact) and hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0  # type: ignore
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
                    # Fallback to CPU exact if GPU path fails
                    exact_index = faiss.IndexFlatL2(dimension)
                    exact_index.add(vectors)  # type: ignore
                    gt_dist, gt_idx = exact_index.search(queries, k)  # type: ignore
            else:
                exact_index = faiss.IndexFlatL2(dimension)
                exact_index.add(vectors)  # type: ignore
                gt_dist, gt_idx = exact_index.search(queries, k)  # type: ignore

            # Compute recall@k
            recalls = []
            for i in range(num_queries):
                ann_set = set(ann_idx[i].tolist())
                gt_set = set(gt_idx[i].tolist())
                inter = len(ann_set & gt_set)
                recalls.append(inter / k)
            true_recall = float(np.mean(recalls))

            # Attach to metrics
            metrics = state.get("current_metrics", {})
            metrics["true_recall"] = true_recall
            metrics["k"] = k
            state["current_metrics"] = metrics

            print(f"✅ True Recall@{k}: {true_recall:.3f}")
        except Exception as e:
            print(f"❌ Error computing exact recall: {e}")
            # Non-fatal: continue without true_recall
        return state

    def _update_best_config_node(self, state: OptimizationState) -> OptimizationState:
        """Update best configuration"""
        print("🏆 Updating best configuration...")
        
        current_metrics = state["current_metrics"]
        current_params = state["current_params"]
        
        # Check if current config is better
        if not state["best_config"] or self._is_better_config(current_metrics, state["best_config"].get("metrics", {})):
            state["best_config"] = {
                "params": current_params.copy(),
                "metrics": current_metrics.copy(),
                "iteration": state["iteration_count"]
            }
            print("🎉 New best configuration found!")
        
        # Add to experiment history
        experiment = {
            "iteration": state["iteration_count"],
            "params": current_params.copy(),
            "metrics": current_metrics.copy(),
            "timestamp": time.time(),
            "datetime": datetime.utcnow().isoformat() + "Z"
        }
        state["experiment_history"].append(experiment)
        # Count exploration trials
        state["exploration_count"] = int(state.get("exploration_count", 0)) + 1
        # Persist to JSONL via Archivist
        if self.archivist is not None:
            try:
                ds = {
                    "dataset_size": state.get("dataset_size"),
                    "dimension": state.get("dimension"),
                }
                self.archivist.log({"dataset": ds, **experiment})
            except Exception as e:
                print(f"⚠️  Logging failed: {e}")
        state["iteration_count"] += 1
        
        return state
    
    def _decide_next_action_node(self, state: OptimizationState) -> OptimizationState:
        """Decide whether to continue optimization"""
        print("🤔 Deciding next action...")
        
        # Check stopping conditions
        if state["iteration_count"] >= self.thresholds.max_experiments:
            print("🛑 Maximum experiments reached")
            state["status"] = "done"
            return state
        
        # Force initial exploration for N trials regardless of thresholds
        exploration_count = int(state.get("exploration_count", 0))
        if exploration_count < getattr(self, "initial_exploration_trials", 5):
            state["status"] = "experimenting"
            return state
        
        # Phase-based progression
        current_metrics = state["current_metrics"]
        phase = state.get("phase", "recall")
        if phase == "recall" and current_metrics.get("recall", 0.0) >= self.thresholds.min_recall:
            # Switch to latency optimization while maintaining recall target
            print("✅ Recall target met. Switching to latency optimization phase.")
            state["phase"] = "latency"
            state["status"] = "experimenting"
            return state
        if phase == "latency" and current_metrics.get("recall", 0.0) < self.thresholds.min_recall:
            # If we dipped below recall, go back to recall phase
            print("⚠️  Recall dropped below target. Returning to recall phase.")
            state["phase"] = "recall"
            state["status"] = "experimenting"
            return state
        
        # Continue optimization
        state["status"] = "experimenting"
        return state
    
    # Helper methods
    def _generate_llm_parameters(self, state: OptimizationState) -> Dict[str, int]:
        """Generate parameters using LLM"""
        # If no LLM available, fallback
        if not (self.lc_llm or self.genai_client):
            return self._generate_fallback_parameters(state)
        # Build recent history/past log payloads
        recent_trials = self._get_recent_trials(state, max(0, int(getattr(self, "llm_history_trials", 5))))
        past_log_trials: List[Dict[str, Any]] = []
        if self.include_past_log_history:
            try:
                past_log_trials = self._load_past_log_trials(state, limit=5)
            except Exception:
                past_log_trials = []
        # Build a single plain-text prompt
        prompt_text = (
            "You are an expert on FAISS HNSW parameter tuning. Propose parameters to maximize recall while respecting latency and memory targets.\n\n"
            "Constraints:\n"
            f"- hnsw_m: {self.constraints.hnsw_m_min}-{self.constraints.hnsw_m_max}\n"
            f"- ef_construction: {self.constraints.ef_construction_min}-{self.constraints.ef_construction_max}\n"
            f"- ef_search: {self.constraints.ef_search_min}-{self.constraints.ef_search_max}\n\n"
            "Targets:\n"
            f"- min_recall: {self.thresholds.min_recall}\n"
            f"- max_latency_ms: {self.thresholds.max_latency_ms}\n"
            f"- max_memory_gb: {self.thresholds.max_memory_gb}\n\n"
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

        try:
            if self.lc_llm is not None and SystemMessage is not None and HumanMessage is not None:
                system = SystemMessage(content="You are an expert on FAISS HNSW parameter tuning. Respond in strict JSON.")
                user = HumanMessage(content=prompt_text)
                lc_resp = self.lc_llm.invoke([system, user])
                content = getattr(lc_resp, "content", "")
            else:
                assert self.genai_client is not None
                response = self.genai_client.models.generate_content(model=self.genai_model, contents=prompt_text)
                content = getattr(response, "text", "")
            # Try strict JSON parse, otherwise extract braces
            try:
                parsed = json.loads(content)
            except Exception:
                start = content.find("{")
                end = content.rfind("}")
                parsed = json.loads(content[start:end+1]) if start != -1 and end != -1 else {}

            params = {
                "hnsw_m": int(parsed.get("hnsw_m", 16)),
                "ef_construction": int(parsed.get("ef_construction", 200)),
                "ef_search": int(parsed.get("ef_search", 50)),
            }
            return self._validate_parameters(params)
        except Exception:
            return self._generate_fallback_parameters(state)

    def _get_recent_trials(self, state: OptimizationState, n: int) -> List[Dict[str, Any]]:
        """Return last n trials from current run in compact form for LLM context (most recent first)."""
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

    def _load_past_log_trials(self, state: OptimizationState, limit: int = 5) -> List[Dict[str, Any]]:
        """Load up to 'limit' recent trials from JSONL log matching current dimension and dataset_size."""
        log_path = getattr(self.archivist, "log_path", os.getenv("EXPERIMENT_LOG", "experiments.jsonl"))
        if not log_path or not os.path.isfile(log_path):
            return []
        dim = state.get("dimension")
        size = state.get("dataset_size")
        rows: List[Dict[str, Any]] = []
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except Exception:
                        continue
                    ds = obj.get("dataset", {})
                    if (dim and ds.get("dimension") != dim) or (size and ds.get("dataset_size") != size):
                        continue
                    params = obj.get("params", {})
                    metrics = obj.get("metrics", {})
                    rows.append({
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
            return []
        return rows[-limit:][::-1]

    def recommend_parameters_from_graph(self, graph_data: "GraphData") -> Dict[str, Any]:
        """Use LLM (or heuristic) to recommend params from external graph data.

        Returns a dict with: params, rationale, considered_trials
        """
        # If no LLM, use a simple heuristic: increase ef_search when distances are tight
        if not (self.lc_llm or self.genai_client):
            heuristic_m = min(max(16, int(graph_data.dimension // 8)), self.constraints.hnsw_m_max)
            # If nearest distances are small (dense space), favor higher ef_search
            tight_space = graph_data.global_distance_stats.p50 < 0.5
            ef_search = 200 if tight_space else 100
            ef_construction = 400 if tight_space else 200
            params = self._validate_parameters({
                "hnsw_m": heuristic_m,
                "ef_construction": ef_construction,
                "ef_search": ef_search,
            })
            return {"params": params, "rationale": "Heuristic recommendation without LLM.", "considered_trials": len(graph_data.trials)}

        # Build a compact JSON payload for the LLM
        def serialize_distance(d: DistanceStats) -> Dict[str, float]:
            return {
                "mean": d.mean,
                "std": d.std,
                "min": d.minimum,
                "max": d.maximum,
                "p25": d.p25,
                "p50": d.p50,
                "p75": d.p75,
            }

        trials_payload = []
        for t in graph_data.trials[:20]:  # limit to first 20 for brevity
            trials_payload.append({
                "params": t.params,
                "metrics": t.metrics,
                "distance_stats": serialize_distance(t.distance_stats),
            })

        payload = {
            "dataset_size": graph_data.dataset_size,
            "dimension": graph_data.dimension,
            "global_distance_stats": serialize_distance(graph_data.global_distance_stats),
            "trials": trials_payload,
            "constraints": {
                "hnsw_m": [self.constraints.hnsw_m_min, self.constraints.hnsw_m_max],
                "ef_construction": [self.constraints.ef_construction_min, self.constraints.ef_construction_max],
                "ef_search": [self.constraints.ef_search_min, self.constraints.ef_search_max],
            },
            "targets": {
                "min_recall": self.thresholds.min_recall,
                "max_latency_ms": self.thresholds.max_latency_ms,
                "max_memory_gb": self.thresholds.max_memory_gb,
            },
        }

        prompt_text = (
            "You are an expert on FAISS HNSW for vector search. Recommend parameters that maximize recall while respecting latency and memory targets. Always respond with strict JSON.\n\n"
            "Here is graph data and constraints as JSON. Return JSON with keys: hnsw_m, ef_construction, ef_search, rationale.\n"
            + json.dumps(payload)
        )

        try:
            if self.lc_llm is not None and SystemMessage is not None and HumanMessage is not None:
                system = SystemMessage(content="You are an expert on FAISS HNSW. Respond in strict JSON.")
                user = HumanMessage(content=prompt_text)
                lc_resp = self.lc_llm.invoke([system, user])
                content = getattr(lc_resp, "content", "")
            else:
                assert self.genai_client is not None
                response = self.genai_client.models.generate_content(model=self.genai_model, contents=prompt_text)
                content = getattr(response, "text", "")
            try:
                parsed = json.loads(content)
            except Exception:
                start = content.find("{")
                end = content.rfind("}")
                parsed = json.loads(content[start:end+1]) if start != -1 and end != -1 else {}

            params = self._validate_parameters({
                "hnsw_m": int(parsed.get("hnsw_m", 16)),
                "ef_construction": int(parsed.get("ef_construction", 200)),
                "ef_search": int(parsed.get("ef_search", 50)),
            })
            rationale = parsed.get("rationale", "")
            return {"params": params, "rationale": rationale, "considered_trials": len(trials_payload)}
        except Exception as e:
            # Fallback to heuristic if parsing/LLM fails
            heuristic_m = min(max(16, int(graph_data.dimension // 8)), self.constraints.hnsw_m_max)
            tight_space = graph_data.global_distance_stats.p50 < 0.5
            ef_search = 200 if tight_space else 100
            ef_construction = 400 if tight_space else 200
            params = self._validate_parameters({
                "hnsw_m": heuristic_m,
                "ef_construction": ef_construction,
                "ef_search": ef_search,
            })
            return {"params": params, "rationale": "Heuristic fallback due to LLM error.", "considered_trials": len(graph_data.trials)}

    def recommend_parameters_from_json_file(self, json_path: str) -> Dict[str, Any]:
        """Load graph data from JSON and return a recommendation."""
        graph = load_graph_data_json(json_path)
        return self.recommend_parameters_from_graph(graph)

    def _generate_fallback_parameters(self, state: OptimizationState) -> Dict[str, int]:
        """Generate fallback parameters"""
        import random
        iteration = state["iteration_count"]
        if iteration == 0:
            hnsw_m, ef_construction, ef_search = 16, 200, 50
        elif iteration == 1:
            hnsw_m, ef_construction, ef_search = 32, 400, 100
        elif iteration == 2:
            hnsw_m, ef_construction, ef_search = 8, 100, 20
        else:
            hnsw_m = random.randint(self.constraints.hnsw_m_min, self.constraints.hnsw_m_max)
            ef_construction = random.randint(self.constraints.ef_construction_min, self.constraints.ef_construction_max)
            ef_search = random.randint(self.constraints.ef_search_min, self.constraints.ef_search_max)
        params = {
            "hnsw_m": hnsw_m,
            "ef_construction": ef_construction,
            "ef_search": ef_search
        }
        return self._validate_parameters(params)

    def _validate_parameters(self, params: Dict[str, int]) -> Dict[str, int]:
        """Validate and clamp parameters"""
        validated = {}
        validated["hnsw_m"] = max(self.constraints.hnsw_m_min,
                                   min(self.constraints.hnsw_m_max, params.get("hnsw_m", 16)))
        validated["ef_construction"] = max(self.constraints.ef_construction_min,
                                            min(self.constraints.ef_construction_max,
                                                params.get("ef_construction", 200)))
        validated["ef_search"] = max(self.constraints.ef_search_min,
                                      min(self.constraints.ef_search_max,
                                          params.get("ef_search", 50)))
        return validated

    def _is_better_config(self, current: Dict[str, float], best: Dict[str, float]) -> bool:
        """Determine if current configuration is better"""
        if not best:
            return True
        if current["recall"] > best.get("recall", -1.0):
            return True
        elif current["recall"] == best.get("recall", -1.0):
            if current["latency_ms"] < best.get("latency_ms", float("inf")):
                return True
            elif current["latency_ms"] == best.get("latency_ms", float("inf")):
                return current["memory_gb"] < best.get("memory_gb", float("inf"))
        return False

    def _should_continue(self, state: OptimizationState) -> str:
        """Determine the next step"""
        if state["status"] == "done":
            return "done"
        else:
            return "continue"

    def optimize(self, max_iterations: int = 10) -> Dict[str, Any]:
        """Run the optimization workflow"""
        print("🚀 Starting vector database optimization...")
        self.thresholds.max_experiments = max_iterations
        initial_state = OptimizationState(
            dataset_size=0,
            dimension=0,
            current_params={},
            current_metrics={},
            experiment_history=[],
            best_config={},
            iteration_count=0,
            status="analyzing",
            error_message=None,
            _vectors=None,
            _queries=None,
            _last_build_ms=None,
            phase="recall",
            tried_params=[],
            exploration_count=0
        )
        # Increase recursion limit to accommodate multi-iteration loops (each iteration traverses multiple nodes)
        recursion_limit = max(100, int(self.thresholds.max_experiments) * 10)
        final_state = self.workflow.invoke(initial_state, config={"recursion_limit": recursion_limit})
        print(f"🏁 Optimization completed")
        print(f"📊 Total experiments: {final_state['iteration_count']}")
        if final_state["best_config"]:
            print(f"🎯 Best configuration:")
            print(f"   Parameters: {final_state['best_config']['params']}")
            print(f"   Metrics: {final_state['best_config']['metrics']}")
        return final_state


def generate_dummy_graph_data(dataset_size: int = 5000, dimension: int = 128, num_trials: int = 6) -> GraphData:
    """Create dummy graph data with plausible distance stats and trial outcomes."""
    # Simulate a random dataset and compute simple distance summaries on a small sample
    sample_vectors = np.random.random((min(dataset_size, 2000), dimension)).astype(np.float32)

    # Approximate pairwise distances for a subset of samples
    sample = sample_vectors[:256]
    # Compute dot products and derive cosine-like distances for simplicity
    norms = np.linalg.norm(sample, axis=1, keepdims=True) + 1e-9
    normalized = sample / norms
    sims = normalized @ normalized.T
    dists = 1.0 - sims  # pseudo cosine distance
    # Use upper triangle values excluding diagonal
    iu = np.triu_indices_from(dists, k=1)
    values = dists[iu]

    def summarize(arr: np.ndarray) -> DistanceStats:
        qs = np.quantile(arr, [0.25, 0.5, 0.75])
        return DistanceStats(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            minimum=float(np.min(arr)),
            maximum=float(np.max(arr)),
            p25=float(qs[0]),
            p50=float(qs[1]),
            p75=float(qs[2]),
        )

    global_stats = summarize(values)

    # Create a few synthetic trials with metrics correlated to ef_search
    trials: List[TrialRecord] = []
    preset_params = [
        {"hnsw_m": 16, "ef_construction": 200, "ef_search": 50},
        {"hnsw_m": 32, "ef_construction": 400, "ef_search": 100},
        {"hnsw_m": 8,  "ef_construction": 100, "ef_search": 20},
    ]
    while len(preset_params) < num_trials:
        preset_params.append({
            "hnsw_m": int(np.random.randint(8, 48)),
            "ef_construction": int(np.random.randint(100, 800)),
            "ef_search": int(np.random.randint(20, 400)),
        })

    for p in preset_params[:num_trials]:
        # Synthetic metrics
        recall = float(min(0.98, 0.6 + (p["ef_search"] / 1000.0) * 0.35))
        latency_ms = float(max(2.0, 1.0 + p["ef_search"] * 0.15))
        memory_gb = float((dataset_size * dimension * 4) / (1024**3))
        metrics = {"recall": recall, "latency_ms": latency_ms, "memory_gb": memory_gb}
        trials.append(TrialRecord(params=p, metrics=metrics, distance_stats=global_stats))

    return GraphData(
        dataset_size=dataset_size,
        dimension=dimension,
        global_distance_stats=global_stats,
        trials=trials,
    )

 