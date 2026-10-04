"""File preparation and plotting helpers for the HIGGS/SUSY example notebooks.

Sampling configuration and interpretation live in the notebooks. Population
data stays in DuckDB/Parquet; only bounded plotting data enters pandas.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import time
from urllib.request import urlopen

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_dataset(url, path, *, allow_download=True):
    """Cache a gzip download atomically; fingerprints are local, not publisher hashes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = path.with_suffix(path.suffix + ".json")
    if path.exists():
        digest = sha256_file(path)
        if metadata_path.exists():
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            if previous["url"] != url or previous["sha256"] != digest:
                raise ValueError(f"Cached download provenance mismatch: {path}")
        print(f"Using cached source: {path}", flush=True)
    else:
        if not allow_download:
            raise FileNotFoundError(f"Place {url} at {path}, or set ALLOW_DOWNLOAD = True.")
        partial = path.with_suffix(path.suffix + ".partial")
        try:
            print(f"Downloading {url}", flush=True)
            digest_state = hashlib.sha256()
            downloaded, last_message = 0, 0
            with urlopen(url, timeout=120) as response, partial.open("wb") as stream:
                expected = response.headers.get("Content-Length")
                for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                    stream.write(block)
                    digest_state.update(block)
                    downloaded += len(block)
                    if downloaded - last_message >= 128 * 1024 * 1024:
                        print(f"  {downloaded / 1024**2:,.0f} MiB downloaded", flush=True)
                        last_message = downloaded
            if expected and downloaded != int(expected):
                raise IOError(f"Incomplete download: {downloaded} of {expected} bytes")
            with partial.open("rb") as stream:
                if stream.read(2) != b"\x1f\x8b":
                    raise ValueError("Expected gzip data; the server returned a different payload")
            digest = digest_state.hexdigest()
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    metadata = {"url": url, "bytes": path.stat().st_size, "sha256": digest}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def prepare_dataset(raw_path, cache_dir, features, *, total_rows, test_rows, source):
    """Persist source-order IDs and the published tail holdout; validate cached files."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_path, test_path = cache_dir / "train.parquet", cache_dir / "test.parquet"
    metadata_path = cache_dir / "prepared.json"
    train_rows = total_rows - test_rows
    if train_rows <= 0 or test_rows <= 0:
        raise ValueError("Both training and test partitions must be nonempty")
    signature = {
        "format_version": 1, "source_sha256": source["sha256"],
        "features": list(features), "total_rows": total_rows,
        "train_rows": train_rows, "test_rows": test_rows,
        "split": "1-based source row ID; final test_rows records reserved",
    }
    if metadata_path.exists() and train_path.exists() and test_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous["signature"] != signature:
            raise ValueError("Prepared cache configuration changed; choose a new CACHE_DIR")
        for name, path in (("train", train_path), ("test", test_path)):
            if sha256_file(path) != previous[f"{name}_sha256"]:
                raise ValueError(f"Prepared cache fingerprint mismatch: {path}")
        print("Using validated training/test Parquet cache", flush=True)
        return train_path, test_path, previous

    started = time.perf_counter()
    # All column names are notebook-defined; quote even these trusted identifiers.
    columns = ["label", *features]
    schema = "{" + ", ".join("'" + c.replace("'", "''") + "': 'DOUBLE'" for c in columns) + "}"
    invalid = " OR ".join(
        'NOT COALESCE(isfinite("' + c.replace('"', '""') + '"), FALSE)' for c in columns
    )
    partial_train = train_path.with_suffix(".parquet.partial")
    partial_test = test_path.with_suffix(".parquet.partial")
    try:
        with tempfile.TemporaryDirectory(prefix="prepare_", dir=cache_dir) as work:
            # Serial scan + ROW_NUMBER preserves the published file-order split.
            with duckdb.connect(str(Path(work) / "prepare.duckdb")) as con:
                con.execute("SET threads = 1")
                con.execute("SET memory_limit = '2GB'")
                con.execute("SET preserve_insertion_order = true")
                print("Parsing the complete gzip source into file-backed DuckDB...", flush=True)
                con.execute(f"""
                    CREATE TABLE source AS
                    SELECT ROW_NUMBER() OVER ()::BIGINT AS source_row_id, *
                    FROM read_csv(?, header=false, delim=',', columns={schema},
                                  auto_detect=false, compression='gzip', parallel=false)
                """, [str(raw_path)])
                n, bad = con.execute(f"""
                    SELECT COUNT(*), COUNT(*) FILTER (
                        WHERE {invalid} OR label NOT IN (0, 1)) FROM source
                """).fetchone()
                if n != total_rows or bad:
                    raise ValueError(f"Expected {total_rows:,} valid binary rows; got {n:,}, invalid={bad}")
                for path, condition in (
                    (partial_train, f"source_row_id <= {train_rows}"),
                    (partial_test, f"source_row_id > {train_rows}"),
                ):
                    con.execute(f"""
                        COPY (SELECT * FROM source WHERE {condition} ORDER BY source_row_id)
                        TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
                    """, [str(path)])
        partial_train.replace(train_path)
        partial_test.replace(test_path)
    finally:
        partial_train.unlink(missing_ok=True)
        partial_test.unlink(missing_ok=True)
    metadata = {
        "signature": signature, "preparation_seconds": time.perf_counter() - started,
        "train_sha256": sha256_file(train_path), "test_sha256": sha256_file(test_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared {train_rows:,} training and {test_rows:,} reserved test records", flush=True)
    return train_path, test_path, metadata


def plot_distributions(train_path, sample, features, balance, *, seed=42, plot_rows=50_000):
    """Approximate visualizations, labeled separately from the exact report KS."""
    cols = ", ".join('"' + c.replace('"', '""') + '"' for c in features)
    with duckdb.connect() as con:
        con.execute("SET threads = 1")
        con.execute("SET memory_limit = '1GB'")
        population = con.execute(f"""
            SELECT {cols} FROM read_parquet(?)
            USING SAMPLE reservoir({int(plot_rows)} ROWS) REPEATABLE({int(seed)})
        """, [str(train_path)]).df()
    selected = sample.select(features).to_pandas().sample(
        n=min(plot_rows, len(sample)), random_state=seed,
    )
    fig, axes = plt.subplots(len(features), 2, figsize=(13, 3.4 * len(features)), squeeze=False)
    for row, feature in enumerate(features):
        p = np.sort(population[feature].dropna().to_numpy())
        s = np.sort(selected[feature].dropna().to_numpy())
        low, high = np.quantile(np.concatenate([p, s]), [0.005, 0.995])
        if low == high:
            low, high = low - 0.5, high + 0.5
        bins = np.linspace(low, high, 51)
        for values, label, color in ((p, "Training population (plot subset)", "C0"),
                                     (s, "100k sample (plot subset)", "C1")):
            axes[row, 0].hist(values, bins=bins, density=True, histtype="step", label=label, color=color)
            # Bound rendered points as well as the population data pulled to Python.
            indices = np.unique(np.linspace(0, len(values) - 1, min(2000, len(values))).astype(int))
            axes[row, 1].step(values[indices], (indices + 1) / len(values), where="post", label=label, color=color)
        check = balance[(balance.variable == feature) & (balance.scope == "overall")].iloc[0]
        axes[row, 0].set_title(f"{feature}: central 99% display range")
        axes[row, 1].set_title(
            f"Approximate ECDF | exact report KS={check.statistic:.4f} "
            f"/ {check.threshold:.2f}: {check.status.upper()}"
        )
        for ax in axes[row]:
            ax.set_xlabel(feature)
            ax.legend(fontsize=8)
        axes[row, 0].set_ylabel("Density within displayed range")
        axes[row, 1].set_ylabel("Cumulative share")
    fig.tight_layout()
    return fig


def export_sample(result, output_dir, metadata):
    """Keep each run's sample/report together, including explicitly unaccepted candidates."""
    sample = result.sample
    for column, dtype in (("source_row_id", pa.int64()), ("label", pa.int8())):
        index = sample.schema.get_field_index(column)
        sample = sample.set_column(index, column, sample[column].cast(dtype, safe=True))
    accepted = result.report.summary["balance_status"] == "pass"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(output_dir) / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    sample_path = run_dir / ("training_sample_100k.parquet" if accepted else "candidate_100k.parquet")
    pq.write_table(sample, sample_path, compression="zstd")
    restored = pq.read_table(sample_path)
    if not restored.equals(sample):
        raise AssertionError("Exported sample did not round-trip exactly")
    (run_dir / "report.json").write_text(json.dumps(result.report.to_dict(), indent=2), encoding="utf-8")
    ids = sample["source_row_id"].to_numpy().astype("<i8", copy=False)
    manifest = {
        **metadata, "accepted": accepted, "sample_rows": len(sample),
        "sample_file": sample_path.name, "sample_sha256": sha256_file(sample_path),
        "sample_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "created_utc": stamp,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{'Accepted sample' if accepted else 'Unaccepted candidate'}: {sample_path}", flush=True)
    return sample, run_dir
