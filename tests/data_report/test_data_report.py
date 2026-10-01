import pandas as pd
import pytest

from rapidmatch.data_report.report import DataReport, InsufficientControlPoolWarning
from rapidmatch.ingestion.ingest import ingest


def test_data_report_is_lazy_then_batched() -> None:
    df = pd.DataFrame(
        {
            "income": [10.0, None, 30.0, 40.0],
            "region": ["N", "S", None, "N"],
            "is_target": [1, 1, 0, 0],
        }
    )
    with ingest(df, keep_db=True) as session:
        report = DataReport(session.con, "is_target")
        assert report._cache is None
        summary = report.summary()
        assert report.summary() is summary
        assert summary["n_rows"] == 4
        assert summary["n_target"] == 2
        assert summary["n_control"] == 2
        assert summary["null_counts"]["income"] == 1
        assert summary["null_counts"]["region"] == 1
        assert summary["n_numeric"] >= 1
        assert summary["n_categorical"] >= 1


def test_data_report_warns_when_controls_are_fewer_than_targets() -> None:
    df = pd.DataFrame(
        {
            "income": [10.0, 20.0, 30.0],
            "is_target": [1, 1, 0],
        }
    )
    with ingest(df, keep_db=True) as session:
        report = DataReport(session.con, "is_target")
        with pytest.warns(InsufficientControlPoolWarning, match="Control pool has 1"):
            summary = report.summary()
        assert summary["n_target"] == 2
        assert summary["n_control"] == 1
