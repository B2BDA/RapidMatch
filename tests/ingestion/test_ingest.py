from pathlib import Path

import pandas as pd

from rapidmatch.ingestion.ingest import ingest
from rapidmatch.ingestion.validate import column_types


def test_csv_stream_creates_udl_data(tmp_path: Path) -> None:
    path = tmp_path / "tiny.csv"
    pd.DataFrame(
        {"income": [1.0, 2.0], "region": ["A", "B"], "is_target": [1, 0]}
    ).to_csv(path, index=False)
    with ingest(str(path), work_dir=str(tmp_path), keep_db=True) as session:
        types = column_types(session.con, "udl_data")
        n = session.con.execute("SELECT COUNT(*) FROM udl_data").fetchone()[0]
        assert n == 2
        assert "income" in types
        assert "DOUBLE" in types["income"]
