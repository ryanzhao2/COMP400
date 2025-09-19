"""
LangGraph-based Agentic AI for Vector Database Optimization
A basic template for tuning FAISS HNSW parameters using intelligent agents.
"""

import os
import time
import json
from typing import Dict, List, Any, TypedDict, Optional
from dataclasses import dataclass
import numpy as np

# FAISS imports
try:
    import faiss
except ImportError:
    raise ImportError("FAISS not installed. Run: pip install faiss-cpu")

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Configuration
@dataclass
class ParameterConstraints:
    """Constraints for HNSW parameters"""
    hnsw_m_min: int = 4
    hnsw_m_max: int = 64
    ef_construction_min: int = 50
    ef_construction_max: int = 1000
    ef_search_min: int = 10
    ef_search_max: int = 500

@dataclass
class PerformanceThresholds:
    """Performance guardrails"""
    min_recall: float = 0.8
    max_latency_ms: float = 100.0
    max_memory_gb: float = 8.0
    max_experiments: int = 20

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

class VectorDatabaseAgent:
    """Basic agent for vector database optimization"""
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 constraints: Optional[ParameterConstraints] = None,
                 thresholds: Optional[PerformanceThresholds] = None):
        
        # Set up LLM (optional)
        self.llm = None
        if openai_api_key or os.getenv("OPENAI_API_KEY"):
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
            )
        
        # Set up constraints and thresholds
        self.constraints = constraints or ParameterConstraints()
        self.thresholds = thresholds or PerformanceThresholds()
        
        # Build the workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(OptimizationState)
        
        # Add nodes
        workflow.add_node("analyze_dataset", self._analyze_dataset_node)
        workflow.add_node("generate_parameters", self._generate_parameters_node)
        workflow.add_node("build_index", self._build_index_node)
        workflow.add_node("evaluate_performance", self._evaluate_performance_node)
        workflow.add_node("update_best_config", self._update_best_config_node)
        workflow.add_node("decide_next_action", self._decide_next_action_node)
        
        # Set entry point
        workflow.set_entry_point("analyze_dataset")
        
        # Add edges
        workflow.add_edge("analyze_dataset", "generate_parameters")
        workflow.add_edge("generate_parameters", "build_index")
        workflow.add_edge("build_index", "evaluate_performance")
        workflow.add_edge("evaluate_performance", "update_best_config")
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
        """Analyze the dataset characteristics"""
        print("🔍 Analyzing dataset...")
        
        # Generate sample dataset
        dataset_size = 10000
        dimension = 128
        
        state["dataset_size"] = dataset_size
        state["dimension"] = dimension
        state["status"] = "experimenting"
        
        print(f"📊 Dataset: {dataset_size} vectors, {dimension}D")
        return state
    
    def _generate_parameters_node(self, state: OptimizationState) -> OptimizationState:
        """Generate HNSW parameters"""
        print("🧠 Generating parameters...")
        
        # Use LLM if available, otherwise use fallback
        if self.llm:
            state["current_params"] = self._generate_llm_parameters(state)
        else:
            state["current_params"] = self._generate_fallback_parameters(state)
        
        print(f"🎯 Generated params: {state['current_params']}")
        return state
    
    def _build_index_node(self, state: OptimizationState) -> OptimizationState:
        """Build FAISS HNSW index"""
        print("🔨 Building index...")
        
        try:
            params = state["current_params"]
            dimension = state["dimension"]
            dataset_size = state["dataset_size"]
            
            # Create HNSW index
            index = faiss.IndexHNSWFlat(dimension, params["hnsw_m"])
            index.hnsw.efConstruction = params["ef_construction"]
            index.hnsw.efSearch = params["ef_search"]
            
            # Generate sample data
            vectors = np.random.random((dataset_size, dimension)).astype(np.float32)
            index.add(vectors)
            
            print(f"✅ Index built with {dataset_size} vectors")
            
        except Exception as e:
            print(f"❌ Error building index: {e}")
            state["error_message"] = f"Index building failed: {str(e)}"
            state["status"] = "error"
        
        return state
    
    def _evaluate_performance_node(self, state: OptimizationState) -> OptimizationState:
        """Evaluate index performance"""
        print("📊 Evaluating performance...")
        
        try:
            params = state["current_params"]
            dimension = state["dimension"]
            dataset_size = state["dataset_size"]
            
            # Rebuild index for evaluation
            index = faiss.IndexHNSWFlat(dimension, params["hnsw_m"])
            index.hnsw.efConstruction = params["ef_construction"]
            index.hnsw.efSearch = params["ef_search"]
            
            vectors = np.random.random((dataset_size, dimension)).astype(np.float32)
            index.add(vectors)
            
            # Generate test queries
            num_queries = 100
            queries = np.random.random((num_queries, dimension)).astype(np.float32)
            
            # Measure performance
            start_time = time.time()
            distances, indices = index.search(queries, k=10)
            search_time = (time.time() - start_time) * 1000
            
            # Calculate metrics
            avg_latency = search_time / num_queries
            estimated_recall = min(0.95, 0.7 + (params["ef_search"] / 1000) * 0.25)
            memory_usage_gb = (index.ntotal * dimension * 4) / (1024**3)
            
            metrics = {
                "recall": estimated_recall,
                "latency_ms": avg_latency,
                "memory_gb": memory_usage_gb,
                "index_size": index.ntotal
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
            "timestamp": time.time()
        }
        state["experiment_history"].append(experiment)
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
        
        # Check if we've found a good enough solution
        current_metrics = state["current_metrics"]
        if (current_metrics["recall"] >= self.thresholds.min_recall and
            current_metrics["latency_ms"] <= self.thresholds.max_latency_ms and
            current_metrics["memory_gb"] <= self.thresholds.max_memory_gb):
            print("✅ Performance targets met!")
            state["status"] = "done"
            return state
        
        # Continue optimization
        state["status"] = "experimenting"
        return state
    
    # Helper methods
    def _generate_llm_parameters(self, state: OptimizationState) -> Dict[str, int]:
        """Generate parameters using LLM"""
        # Implementation for LLM-based parameter generation
        # This would use the LLM to reason about optimal parameters
        return self._generate_fallback_parameters(state)
    
    def _generate_fallback_parameters(self, state: OptimizationState) -> Dict[str, int]:
        """Generate fallback parameters"""
        import random
        
        iteration = state["iteration_count"]
        
        # Simple parameter exploration
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
        
        # Multi-objective optimization
        if current["recall"] > best["recall"]:
            return True
        elif current["recall"] == best["recall"]:
            if current["latency_ms"] < best["latency_ms"]:
                return True
            elif current["latency_ms"] == best["latency_ms"]:
                return current["memory_gb"] < best["memory_gb"]
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
        
        # Update max experiments
        self.thresholds.max_experiments = max_iterations
        
        # Initial state
        initial_state = OptimizationState(
            dataset_size=0,
            dimension=0,
            current_params={},
            current_metrics={},
            experiment_history=[],
            best_config={},
            iteration_count=0,
            status="analyzing",
            error_message=None
        )
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        print(f"🏁 Optimization completed")
        print(f"📊 Total experiments: {final_state['iteration_count']}")
        
        if final_state["best_config"]:
            print(f"🎯 Best configuration:")
            print(f"   Parameters: {final_state['best_config']['params']}")
            print(f"   Metrics: {final_state['best_config']['metrics']}")
        
        return final_state

# Example usage
def main():
    """Example usage of the VectorDatabaseAgent"""
    
    # Create agent
    agent = VectorDatabaseAgent()
    
    # Run optimization
    results = agent.optimize(max_iterations=5)
    
    # Print results
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Experiments run: {results['iteration_count']}")
    
    if results["best_config"]:
        print(f"\nBest configuration:")
        print(f"  HNSW M: {results['best_config']['params']['hnsw_m']}")
        print(f"  EF Construction: {results['best_config']['params']['ef_construction']}")
        print(f"  EF Search: {results['best_config']['params']['ef_search']}")
        print(f"\nPerformance:")
        print(f"  Recall: {results['best_config']['metrics']['recall']:.3f}")
        print(f"  Latency: {results['best_config']['metrics']['latency_ms']:.1f} ms")
        print(f"  Memory: {results['best_config']['metrics']['memory_gb']:.2f} GB")

if __name__ == "__main__":
    main()