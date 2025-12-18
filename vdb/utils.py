#Utility function for formatting and display.
def format_bytes(num: int) -> str:
    # Format byte count as human-readable string with appropriate units.
    try:
        n = float(num)
    except Exception:
        return str(num)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


