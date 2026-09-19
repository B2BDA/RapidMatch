"""Tiny SQL helpers shared by DuckDB-facing modules."""


def quote_ident(name: str) -> str:
    """Quote a column/relation name for DuckDB (handles spaces and quotes)."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"invalid identifier: {name!r}")
    return '"' + name.replace('"', '""') + '"'
