def format_bytes(num: int) -> str:
    """Format a byte count as a human-readable string."""
    try:
        n = float(num)
    except Exception:
        return str(num)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


