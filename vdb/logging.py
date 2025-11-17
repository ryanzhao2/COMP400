from typing import Optional, Dict, Any
import os
import json


class ArchivistAgent:
    """Append experiment records to a JSONL file with a run_id."""
    def __init__(self, log_path: Optional[str], run_id: str):
        self.log_path = log_path or os.getenv("EXPERIMENT_LOG", "experiments.jsonl")
        self.run_id = run_id

    def _resolve_log_path(self, record: Dict[str, Any]) -> str:
        """
        Resolve a database type and dataset-specific JSONL path in subfolders.
        Example:
          base: experiments/experiments.jsonl
          database_type: "knowledge_reasoning", dataset_size: 100000 
          -> experiments/knowledge_reasoning/experiments_n100000.jsonl
        """
        try:
            base = self.log_path
            base_dir = os.path.dirname(base) or "."
            dataset = record.get("dataset", {}) or {}
            database_type = record.get("database_type") or dataset.get("database_type") or "knowledge_reasoning"
            size = dataset.get("dataset_size")
            
            # Create subfolder for database type
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
            print(f"⚠️  Failed to write experiment record: {e}")


