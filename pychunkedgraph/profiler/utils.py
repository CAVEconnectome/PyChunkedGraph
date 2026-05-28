def _fmt_time(s: float) -> str:
    """Auto-scale seconds → ``12.3 ms`` / ``1.23 s``."""
    if s < 1.0:
        return f"{s * 1000:.1f} ms"
    return f"{s:.2f} s"


def _fmt_bytes(n: int, *, signed: bool = False) -> str:
    """Auto-scale bytes (binary) → ``512 B`` / ``1.5 KB`` / ``45.6 MB`` / ``1.23 GB``.

    With ``signed=True``, positive values get a ``+`` prefix (for delta columns).
    """
    if signed:
        sign = "+" if n > 0 else "-" if n < 0 else ""
    else:
        sign = "-" if n < 0 else ""
    n = abs(int(n))
    if n < 1024:
        return f"{sign}{n} B"
    if n < 1024**2:
        return f"{sign}{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{sign}{n / 1024**2:.1f} MB"
    return f"{sign}{n / 1024**3:.2f} GB"


def _fmt_count(n: int) -> str:
    """Thousands-separator integer: ``1234567`` → ``1,234,567``."""
    return f"{int(n):,}"
