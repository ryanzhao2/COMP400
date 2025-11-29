"""
Experiment logging and archival.

The ArchivistAgent records all experiments to organized JSONL files,
with automatic folder structure: experiments/{database_type}/experiments_n{size}.jsonl
"""

from typing import Optional, Dict, Any
import os
import json


class ArchivistAgent:
    """
    Records experiment results to JSONL log files.
    
    Automatically organizes logs by database type and dataset size
    for easy analysis and visualization.
    """
    def __init__(self, log_path: Optional[str], run_id: str):
        self.log_path = log_path or os.getenv("EXPERIMENT_LOG", "experiments.jsonl")
        self.run_id = run_id

    def _resolve_log_path(self, record: Dict[str, Any]) -> str:
        """
        Determine target log file path based on experiment metadata.
        
        Organizes logs into: experiments/{database_type}/experiments_n{size}.jsonl
        This structure enables efficient filtering and comparison across runs.
        """
        try:
            base = self.log_path
            base_dir = os.path.dirname(base) or "."
            dataset = record.get("dataset", {}) or {}
            database_type = record.get("database_type") or dataset.get("database_type") or "knowledge_reasoning"
            size = dataset.get("dataset_size")
            
            # Ensure database_type subfolders are under "experiments/" folder
            # If base_dir is "." or doesn't contain "experiments", use "experiments" as base
            if base_dir == "." or "experiments" not in base_dir:
                base_dir = "experiments"
            
            # Create subfolder for database type under experiments/
            db_subfolder = os.path.join(base_dir, database_type)
            
            # Build filename: experiments_n{size}.jsonl (without database_type prefix since it's in subfolder)
            if isinstance(size, int):
                filename = f"experiments_n{size}.jsonl"
            else:
                filename = "experiments.jsonl"
            return os.path.join(db_subfolder, filename)
        except Exception:
            return self.log_path

    def log(self, record: Dict[str, Any]) -> None:
        try:
            enriched = {"run_id": self.run_id, **record}
            target_path = self._resolve_log_path(enriched)
            os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
            with open(target_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(enriched) + "\n")
        except Exception as e:
            print(f"Warning: Failed to write experiment record: {e}")


