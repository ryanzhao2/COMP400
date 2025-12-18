"""
Experiment logging and archival.

The ArchivistAgent records all experiments to organized JSONL files,
with automatic folder structure: experiments/{database_type}/experiments_n{size}.jsonl
"""

from typing import Optional, Dict, Any, List
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


def load_past_log_trials(log_path: Optional[str], dim: Optional[int], size: Optional[int], database_type: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Load past trials from log files for LLM context.
    
    Intelligently searches for relevant past experiments by:
    1. Exact database_type + size match (highest priority)
    2. Similar dataset sizes from same database_type
    3. Similar sizes from other database types (fallback)
    
    Filters require exact dimension match and prefer matching database_type.
    """
    if not log_path:
        return []
    
    base_dir = os.path.dirname(log_path) or "."
    # Ensure database_type subfolders are under "experiments/" folder
    if base_dir == "." or "experiments" not in base_dir:
        base_dir = "experiments"
    rows: List[Dict[str, Any]] = []
    
    database_type = database_type or "knowledge_reasoning"
    
    # Collect candidate files: exact match first, then closest sizes
    candidate_files: List[tuple[str, int, str]] = []  # (path, dataset_size, database_type)
    
    if size is not None:
        # Try exact database_type subfolder + size-specific file first
        db_subfolder = os.path.join(base_dir, database_type)
        exact_path = os.path.join(db_subfolder, f"experiments_n{size}.jsonl")
        if os.path.isfile(exact_path):
            candidate_files.append((exact_path, size, database_type))
        
        # Find other database type-specific files in subfolders and sort by distance from target size
        try:
            # First, check the target database_type subfolder
            if os.path.isdir(db_subfolder):
                for filename in os.listdir(db_subfolder):
                    if filename.startswith("experiments_n") and filename.endswith(".jsonl"):
                        try:
                            size_str = filename.replace("experiments_n", "").replace(".jsonl", "")
                            file_size = int(size_str)
                            file_path = os.path.join(db_subfolder, filename)
                            if file_path != exact_path and os.path.isfile(file_path):
                                candidate_files.append((file_path, file_size, database_type))
                        except (ValueError, AttributeError):
                            continue
            
            # Also check other database type subfolders
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path) and item != database_type:
                    # Check if this looks like a database_type folder (contains experiments files)
                    for filename in os.listdir(item_path):
                        if filename.startswith("experiments") and filename.endswith(".jsonl"):
                            try:
                                # Extract size from filename: experiments_n100000.jsonl
                                if filename.startswith("experiments_n"):
                                    size_str = filename.replace("experiments_n", "").replace(".jsonl", "")
                                    file_size = int(size_str)
                                    file_path = os.path.join(item_path, filename)
                                    candidate_files.append((file_path, file_size, item))  # item is the database_type
                            except (ValueError, AttributeError):
                                continue
            
            # Also handle old format (flat structure) for backward compatibility
            for filename in os.listdir(base_dir):
                if os.path.isfile(os.path.join(base_dir, filename)):
                    # Match pattern: experiments_{database_type}_n{size}.jsonl (old format)
                    if filename.startswith("experiments_") and filename.endswith(".jsonl"):
                        try:
                            parts = filename.replace("experiments_", "").replace(".jsonl", "").split("_n")
                            if len(parts) == 2:
                                file_db_type = parts[0]
                                file_size = int(parts[1])
                                file_path = os.path.join(base_dir, filename)
                                if file_path != exact_path and os.path.isfile(file_path):
                                    candidate_files.append((file_path, file_size, file_db_type))
                        except (ValueError, AttributeError):
                            continue
            
            # Sort by: 1) database_type match, 2) size distance
            candidate_files.sort(key=lambda x: (0 if x[2] == database_type else 1, abs(x[1] - size)))
        except Exception:
            pass
    
    # Fallback to original log_path if no size-specific files found
    if not candidate_files and os.path.isfile(log_path):
        candidate_files.append((log_path, size or 0, database_type))
    
    # Load from candidate files until we have enough results
    for file_path, _, _ in candidate_files:
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
                    # Filter by database_type (prefer exact match, but accept if not specified in record)
                    record_db_type = obj.get("database_type") or ds.get("database_type")
                    if record_db_type and record_db_type != database_type:
                        # Skip if database type doesn't match (unless record has no database type, then accept)
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
                        "database_type": record_db_type or database_type,  # Keep for sorting
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
    
    # Sort by: 1) database_type match, 2) dataset_size proximity (exact matches first, then closest)
    if size is not None:
        rows.sort(key=lambda x: (0 if x.get("database_type") == database_type else 1, abs(x.get("dataset_size", float("inf")) - size)))
    
    # Remove dataset_size and database_type from final output and return most recent first
    result = []
    for row in rows[:limit]:
        row_copy = {k: v for k, v in row.items() if k not in ("dataset_size", "database_type")}
        result.append(row_copy)
    
    return result[::-1]  # Most recent first
