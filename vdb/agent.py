"""
Agent orchestration for FAISS HNSW optimization.
"""

import os
import time
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
from dotenv import load_dotenv
from vdb.config import ParameterConstraints, PerformanceThresholds, get_database_config, DATABASE_TYPES
from vdb.models import DistanceStats, TrialRecord, GraphData
from vdb.io import load_graph_data_json
from vdb.state import OptimizationState
from langgraph.graph import StateGraph, END

# Optional LLM clients
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
from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore

from vdb.logging import ArchivistAgent
from vdb.prompt import (
    get_recent_trials as _prompt_get_recent_trials,
    load_past_log_trials as _prompt_load_past,
    build_tuning_prompt as _build_prompt,
)
from vdb import nodes


load_dotenv()


class ParamProposerAgent:
    def __init__(self, parent: "VectorDatabaseAgent"):
        self.parent = parent

    def propose(self, state: "OptimizationState") -> Dict[str, int]:
        if self.parent.lc_llm or self.parent.genai_client:
            return self.parent._generate_llm_parameters(state)
        return self.parent._generate_fallback_parameters(state)


class VectorDatabaseAgent:
    """Basic agent for vector database optimization"""
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        constraints: Optional[ParameterConstraints] = None,
        thresholds: Optional[PerformanceThresholds] = None,
        log_path: Optional[str] = None,
        dataset_vectors_path: Optional[str] = None,
        dataset_queries_path: Optional[str] = None,
        num_threads: Optional[int] = None,
        use_gpu_exact: Optional[bool] = None,
        initial_exploration_trials: int = 10,
        database_type: str = "knowledge_reasoning",
    ):
        # LLM init
        self.genai_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        gemini_key = os.getenv("GEMINI_API_KEY")
        self.lc_llm = None
        self.genai_client = None
        if gemini_key and ChatGoogleGenerativeAI is not None:
            try:
                if not os.getenv("GOOGLE_API_KEY"):
                    os.environ["GOOGLE_API_KEY"] = gemini_key
                try:
                    self.lc_llm = ChatGoogleGenerativeAI(model=self.genai_model, temperature=0.1, google_api_key=gemini_key)
                except Exception:
                    self.lc_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1, google_api_key=gemini_key)
            except Exception as e:
                print(f"⚠️  Failed to initialize LangChain Gemini: {e}. Will try direct client.")
        if gemini_key and self.lc_llm is None and genai is not None:
            try:
                self.genai_client = genai.Client(api_key=gemini_key)
            except Exception as e:
                print(f"⚠️  Failed to initialize Google GenAI client: {e}. Using heuristic fallback.")

        # Database type-specific configuration
        self.database_type = database_type
        db_config = get_database_config(database_type)
        self.constraints = constraints or db_config.get("constraints", ParameterConstraints())
        self.thresholds = thresholds or db_config.get("thresholds", PerformanceThresholds())
        
        # Print database type info
        db_info = db_config.get("name", database_type)
        db_desc = db_config.get("description", "")
        print(f"📊 Database Type: {db_info}")
        if db_desc:
            print(f"   {db_desc}")
        self.run_id: Optional[str] = None
        self.archivist: Optional[ArchivistAgent] = ArchivistAgent(log_path or os.getenv("EXPERIMENT_LOG", "experiments.jsonl"), run_id=str(uuid.uuid4()))
        self.param_proposer = ParamProposerAgent(self)
        self.dataset_vectors_path = dataset_vectors_path
        self.dataset_queries_path = dataset_queries_path
        env_use_gpu = os.getenv("FAISS_USE_GPU_EXACT")
        self.use_gpu_exact = (use_gpu_exact if use_gpu_exact is not None else (env_use_gpu == "1" or env_use_gpu == "true"))
        self.initial_exploration_trials = max(0, int(os.getenv("INITIAL_EXPLORATION_TRIALS", initial_exploration_trials)))
        # Threads are set inside FAISS usage within nodes; keep env-only here
        try:
            self.llm_history_trials = int(os.getenv("LLM_HISTORY_TRIALS", "5"))
        except Exception:
            self.llm_history_trials = 5
        self.include_past_log_history = os.getenv("INCLUDE_PAST_LOG_HISTORY", "0").lower() in ("1", "true", "yes")

        self.workflow = self._build_workflow()

    def _build_workflow(self) -> Any:
        workflow = StateGraph(OptimizationState)
        workflow.add_node("analyze_dataset", lambda state: nodes.analyze_dataset_node(self, state))
        workflow.add_node("generate_parameters", lambda state: nodes.generate_parameters_node(self, state))
        workflow.add_node("build_index", lambda state: nodes.build_index_node(self, state))
        workflow.add_node("evaluate_performance", lambda state: nodes.evaluate_performance_node(self, state))
        workflow.add_node("evaluate_exact_recall", lambda state: nodes.evaluate_exact_recall_node(self, state))
        workflow.add_node("update_best_config", lambda state: nodes.update_best_config_node(self, state))
        workflow.add_node("decide_next_action", lambda state: nodes.decide_next_action_node(self, state))
        workflow.set_entry_point("analyze_dataset")
        workflow.add_edge("analyze_dataset", "generate_parameters")
        workflow.add_edge("generate_parameters", "build_index")
        workflow.add_edge("build_index", "evaluate_performance")
        workflow.add_edge("evaluate_performance", "evaluate_exact_recall")
        workflow.add_edge("evaluate_exact_recall", "update_best_config")
        workflow.add_edge("update_best_config", "decide_next_action")
        workflow.add_conditional_edges(
            "decide_next_action",
            self._should_continue,
            {"continue": "generate_parameters", "done": END},
        )
        return workflow.compile()

    # Helper methods
    def _generate_llm_parameters(self, state: OptimizationState) -> Dict[str, int]:
        if not (self.lc_llm or self.genai_client):
            return self._generate_fallback_parameters(state)
        recent_trials = _prompt_get_recent_trials(dict(state), max(0, int(getattr(self, "llm_history_trials", 5))))
        past_log_trials: List[Dict[str, Any]] = []
        if self.include_past_log_history:
            try:
                log_path = getattr(self.archivist, "log_path", os.getenv("EXPERIMENT_LOG", "experiments.jsonl")) if self.archivist is not None else os.getenv("EXPERIMENT_LOG", "experiments.jsonl")
                database_type = state.get("database_type") or self.database_type
                past_log_trials = _prompt_load_past(log_path, state.get("dimension"), state.get("dataset_size"), database_type=database_type, limit=5)
            except Exception:
                past_log_trials = []
        prompt_text = _build_prompt(self, dict(state), recent_trials, past_log_trials)
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

    def recommend_parameters_from_graph(self, graph_data: "GraphData") -> Dict[str, Any]:
        if not (self.lc_llm or self.genai_client):
            heuristic_m = min(max(16, int(graph_data.dimension // 8)), self.constraints.hnsw_m_max)
            tight_space = graph_data.global_distance_stats.p50 < 0.5
            ef_search = 200 if tight_space else 100
            ef_construction = 400 if tight_space else 200
            params = self._validate_parameters({
                "hnsw_m": heuristic_m,
                "ef_construction": ef_construction,
                "ef_search": ef_search,
            })
            return {"params": params, "rationale": "Heuristic recommendation without LLM.", "considered_trials": len(graph_data.trials)}

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
        for t in graph_data.trials[:20]:
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
        except Exception:
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
        graph = load_graph_data_json(json_path)
        return self.recommend_parameters_from_graph(graph)

    def _generate_fallback_parameters(self, state: OptimizationState) -> Dict[str, int]:
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
        validated = {}
        validated["hnsw_m"] = max(self.constraints.hnsw_m_min, min(self.constraints.hnsw_m_max, params.get("hnsw_m", 16)))
        validated["ef_construction"] = max(self.constraints.ef_construction_min, min(self.constraints.ef_construction_max, params.get("ef_construction", 200)))
        validated["ef_search"] = max(self.constraints.ef_search_min, min(self.constraints.ef_search_max, params.get("ef_search", 50)))
        return validated

    def _is_better_config(self, current: Dict[str, float], best: Dict[str, float], phase: str = "recall") -> bool:
        """
        Determine if current config is better than best config.
        In recall phase: prioritize recall, then latency.
        In latency phase: prioritize latency (if recall >= min), then recall.
        """
        if not best:
            return True
        
        current_recall = current.get("recall", 0.0)
        best_recall = best.get("recall", 0.0)
        current_latency = current.get("latency_ms", float("inf"))
        best_latency = best.get("latency_ms", float("inf"))
        min_recall = self.thresholds.min_recall
        
        if phase == "latency":
            # In latency phase: prioritize lower latency if recall is acceptable
            # Accept if: (recall >= min AND latency is lower) OR (same latency but higher recall >= min)
            if current_recall >= min_recall and best_recall >= min_recall:
                # Both meet recall target - prioritize latency
                if current_latency < best_latency:
                    return True
                elif current_latency == best_latency:
                    # Same latency - prefer higher recall or lower memory
                    if current_recall > best_recall:
                        return True
                    elif current_recall == best_recall:
                        return current.get("memory_gb", float("inf")) < best.get("memory_gb", float("inf"))
            elif current_recall >= min_recall:
                # Current meets target, best doesn't
                return True
            elif best_recall < min_recall:
                # Neither meets target - prefer higher recall
                return current_recall > best_recall
            # Best meets target, current doesn't
            return False
        else:
            # In recall phase: prioritize recall first
            if current_recall > best_recall:
                return True
            elif current_recall == best_recall:
                # Same recall - prefer lower latency
                if current_latency < best_latency:
                    return True
                elif current_latency == best_latency:
                    return current.get("memory_gb", float("inf")) < best.get("memory_gb", float("inf"))
            return False

    def _should_continue(self, state: OptimizationState) -> str:
        return "done" if state["status"] == "done" else "continue"

    def optimize(self, max_iterations: int = 10) -> Dict[str, Any]:
        print("🚀 Starting vector database optimization...")
        self.thresholds.max_experiments = max_iterations
        initial_state = OptimizationState(
            dataset_size=0,
            dimension=0,
            database_type=self.database_type,
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
    sample_vectors = np.random.random((min(dataset_size, 2000), dimension)).astype(np.float32)
    sample = sample_vectors[:256]
    norms = np.linalg.norm(sample, axis=1, keepdims=True) + 1e-9
    normalized = sample / norms
    sims = normalized @ normalized.T
    dists = 1.0 - sims
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


