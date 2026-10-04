"""Preparation contracts shared by the two real-data notebook examples."""

import gzip
import importlib.util
from pathlib import Path

import duckdb
import pytest

pytest.importorskip("matplotlib")  # Optional notebook plotting extra.

# The example helper is intentionally outside the installed rapidmatch package.
spec = importlib.util.spec_from_file_location(
    "notebook_downsample", Path(__file__).resolve().parents[1] / "notebook_downsample.py",
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
download_dataset = helpers.download_dataset
prepare_dataset = helpers.prepare_dataset


def source_fixture(tmp_path, text="0,30\n1,10\n0,40\n1,20\n0,999\n1,998\n"):
    source = tmp_path / "source.csv.gz"
    with gzip.open(source, "wt") as stream:
        stream.write(text)
    downloaded = tmp_path / "cache" / "download.csv.gz"
    info = download_dataset(source.as_uri(), downloaded)
    return source, downloaded, info


def test_source_order_holdout_and_offline_cache(tmp_path, monkeypatch):
    source, downloaded, info = source_fixture(tmp_path)
    cache = tmp_path / "prepared"
    train, test, metadata = prepare_dataset(
        downloaded, cache, ["x"], total_rows=6, test_rows=2, source=info,
    )
    with duckdb.connect() as con:
        assert con.execute("SELECT source_row_id, x FROM read_parquet(?)", [str(train)]).fetchall() == [
            (1, 30.), (2, 10.), (3, 40.), (4, 20.),
        ]
        assert con.execute("SELECT source_row_id, x FROM read_parquet(?)", [str(test)]).fetchall() == [
            (5, 999.), (6, 998.),
        ]
    source.unlink()  # Cache must work without the download endpoint.
    assert download_dataset(source.as_uri(), downloaded, allow_download=False) == info

    def unexpected_connection(*args, **kwargs):
        raise AssertionError("Cache reuse must not reparse the CSV")

    monkeypatch.setattr(duckdb, "connect", unexpected_connection)
    assert prepare_dataset(downloaded, cache, ["x"], total_rows=6, test_rows=2, source=info) == (train, test, metadata)
    with train.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        prepare_dataset(downloaded, cache, ["x"], total_rows=6, test_rows=2, source=info)


@pytest.mark.parametrize("text,total", [
    ("0,1\n1,2\n", 3),  # Truncated dataset.
    ("0,1\n0.5,2\n", 2),  # Invalid binary label.
    ("0,1\n1,nan\n", 2),  # Nonfinite predictor.
])
def test_invalid_source_is_not_promoted_to_prepared_cache(tmp_path, text, total):
    _, downloaded, info = source_fixture(tmp_path, text)
    cache = tmp_path / "prepared"
    with pytest.raises(ValueError, match="Expected"):
        prepare_dataset(downloaded, cache, ["x"], total_rows=total, test_rows=1, source=info)
    assert not (cache / "prepared.json").exists()
    assert not (cache / "train.parquet").exists()
    assert not list(cache.glob("*.partial"))
    assert not list(cache.glob("prepare_*"))


def test_missing_cache_and_non_gzip_download(tmp_path):
    source = tmp_path / "error.html"
    source.write_text("<html>Download unavailable</html>")
    target = tmp_path / "cached.csv.gz"
    with pytest.raises(FileNotFoundError, match="ALLOW_DOWNLOAD"):
        download_dataset(source.as_uri(), target, allow_download=False)
    with pytest.raises(ValueError, match="Expected gzip"):
        download_dataset(source.as_uri(), target)
    assert not target.exists()
    assert not target.with_suffix(".gz.partial").exists()
