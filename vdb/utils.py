import math


def format_bytes(num: int) -> str:
    """Format byte count as human-readable string with appropriate units."""
    try:
        n = float(num)
    except Exception:
        return str(num)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def calculate_estimated_recall(ef_search: int, ef_search_max: int, r_min: float = 0.6, r_max: float = 0.98) -> float:
    """
    Estimate recall based on ef_search parameter using a sigmoid function.
    
    Used as a fallback when exact recall computation is too expensive (large datasets).
    """
    ef_max = max(1, int(ef_search_max))
    x = max(0.0, min(1.0, ef_search / ef_max))
    slope = 10.0
    sig = 1.0 / (1.0 + math.exp(-slope * (x - 0.5)))
    return float(r_min + (r_max - r_min) * sig)
