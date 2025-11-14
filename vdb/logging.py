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
        Resolve a dataset-specific JSONL path if dataset_size is present.
        Example:
          base: experiments/experiments.jsonl
          dataset_size: 100000 -> experiments/experiments_n100000.jsonl
        """
        try:
            base = self.log_path
            base_dir = os.path.dirname(base) or "."
            dataset = record.get("dataset", {}) or {}
            size = dataset.get("dataset_size")
            if isinstance(size, int):
                filename = f"experiments_n{size}.jsonl"
                return os.path.join(base_dir, filename)
            return base
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


