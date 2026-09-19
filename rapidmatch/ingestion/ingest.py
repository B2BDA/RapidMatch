"""Module 1 — Ingestion.

RapidSegment's `UniversalDataLoader` writes a disk-backed DuckDB file and
closes its own connection. From that moment RapidMatch owns the connection
and the lazy pipeline begins.

Always read from the `udl_data` view — the loader always creates that alias.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from types import TracebackType
from typing import Any, Optional

import duckdb
import pyarrow as pa

from rapidmatch.ingestion.data_loader import UniversalDataLoader

EXCEL_EXTS = {".xlsx", ".xls"}
STREAMABLE_EXTS = {".csv", ".tsv", ".parquet", ".pq", ".arrow", ".feather"}


class MatchSession:
    """Owns the DuckDB file + connection for one run.

    Closing the session always closes the connection. Unless `keep_db=True`
    the `.duckdb` file is removed so a crash cannot leak temp files.
    """

    def __init__(self, db_path: str, con: duckdb.DuckDBPyConnection, keep_db: bool = False) -> None:
        self.db_path = db_path
        self.con = con
        self.keep_db = keep_db
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self.con.close()
        if not self.keep_db:
            for path in (self.db_path, self.db_path + ".wal"):
                if os.path.exists(path):
                    os.remove(path)
        self._closed = True

    def __enter__(self) -> "MatchSession":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()


def ingest(
    data: Any,
    work_dir: Optional[str] = None,
    keep_db: bool = False,
) -> MatchSession:
    """Route `data` through UniversalDataLoader and reopen our own connection.

    CSV/TSV/Parquet/Feather go through DuckDB native readers (never a Python
    object). Excel is the exception: `.load()` into Arrow, stream, then `del`.
    """
    db_path = _resolve_db_path(work_dir)
    _stream(data, db_path)
    con = duckdb.connect(db_path)
    # Stable integer id used by scoring/matching. ROW_NUMBER is 1-based.
    con.execute(
        """
        CREATE OR REPLACE VIEW v_source AS
        SELECT ROW_NUMBER() OVER () AS _rm_id, * FROM udl_data
        """
    )
    return MatchSession(db_path=db_path, con=con, keep_db=keep_db)


def _resolve_db_path(work_dir: Optional[str]) -> str:
    name = f"rapidmatch_{uuid.uuid4().hex[:8]}.duckdb"
    if work_dir:
        os.makedirs(work_dir, exist_ok=True)
        return os.path.abspath(os.path.join(work_dir, name))
    return os.path.abspath(os.path.join(tempfile.gettempdir(), name))


def _stream(data: Any, db_path: str) -> str:
    if _is_in_memory(data):
        return UniversalDataLoader().stream_to_duckdb(
            db_path=db_path, data=data, scorer_view_name=None
        )
    if not isinstance(data, str):
        raise TypeError(
            "data must be a file path, pandas DataFrame, or PyArrow Table"
        )
    ext = os.path.splitext(data)[1].lower()
    if ext in EXCEL_EXTS:
        # Loader cannot stream Excel. Parse once, persist, drop the RAM copy.
        table = UniversalDataLoader(file_path=data).load()
        try:
            return UniversalDataLoader().stream_to_duckdb(
                db_path=db_path, data=table, scorer_view_name=None
            )
        finally:
            del table
    if ext not in STREAMABLE_EXTS:
        raise ValueError(f"unsupported input format: {ext!r}")
    return UniversalDataLoader(file_path=data).stream_to_duckdb(
        db_path=db_path, scorer_view_name=None
    )


def _is_in_memory(data: Any) -> bool:
    if isinstance(data, pa.Table):
        return True
    return type(data).__module__.startswith("pandas") and type(data).__name__ == "DataFrame"
