import json

import duckdb
import numpy as np
import pyarrow as pa
import pytest

from rapidmatch import random_downsample


def class_counts(result):
    return {json.loads(r["label"])[0]: r["sampled_rows"] for r in result.report.classes.to_pylist()}


def test_exact_size_label_proportions_and_feature_representation():
    data = pa.table({
        "id": np.arange(10000), "Y": [0]*9000 + [1]*1000,
        "region": ["north", "south"]*5000, "x": np.tile(np.arange(10), 1000),
    })
    r = random_downsample(data, sample_size=1000, label_col="Y",
                          stratify_vars=["region", "x"], n_bins=10, random_state=5,
                          check_by_label=True)
    assert len(r.sample) == 1000
    assert class_counts(r) == {0: 900, 1: 100}
    assert len(set(r.row_ids.to_pylist())) == 1000
    assert r.sample.column_names == data.column_names
    assert r.report.summary["balance_status"] == "pass"
    assert r.report.summary["class_budgets_passed"]
    assert all(row["allocated_rows"] == row["sampled_rows"] for row in r.report.strata.to_pylist())
    json.dumps(r.report.to_dict())


def test_label_first_not_flat_joint_rounding_and_bins_do_not_change_classes():
    # Flat joint rounding would choose four B groups and only one A group.
    data = pa.table({"Y": ["A"]*36 + ["B"]*64,
                     "g": sum(([str(i)]*12 for i in range(3)), [])
                          + sum(([str(i)]*16 for i in range(4)), []),
                     "x": np.arange(100)})
    for grouping, bins in ((["g"], 4), (["x"], 2), (["g", "x"], 8), ([], 4)):
        r = random_downsample(data, sample_size=5, label_col="Y",
                              stratify_vars=grouping, n_bins=bins, random_state=5)
        assert class_counts(r) == {"A": 2, "B": 3}
        assert len(r.sample) == 5


@pytest.mark.parametrize("size", [1, 17, 100])
def test_no_label_no_strata_and_full_population(size):
    data = pa.table({"x": np.arange(100), "g": [None, "a"]*50})
    r = random_downsample(data, sample_size=size, random_state=7)
    assert len(r.sample) == size
    assert r.report.summary["balance_status"] == "not_evaluated"
    assert r.report.summary["n_classes"] == 0
    if size == 100:
        assert r.sample["g"].equals(data["g"])
        assert r.sample["x"].to_pylist() == data["x"].to_pylist()


def test_seed_reproducible_across_threads_and_changes_selection():
    data = pa.table({"id": np.arange(1000), "Y": np.arange(1000) % 3,
                     "x": np.arange(1000) % 17})
    kwargs = dict(sample_size=117, label_col="Y", stratify_vars=["x"], random_state=81)
    a = random_downsample(data, **kwargs, duckdb_threads=1)
    b = random_downsample(data, **kwargs, duckdb_threads=4)
    assert a.row_ids.equals(b.row_ids) and a.sample.equals(b.sample)
    c = random_downsample(data, **{**kwargs, "random_state": 82})
    assert not a.row_ids.equals(c.row_ids)


def test_label_and_scope_columns_do_not_collide_with_balance_aliases():
    data = pa.table({"label": [0, 0, 1, 1] * 20, "scope": [1., 2., 1., 2.] * 20})
    result = random_downsample(
        data, sample_size=40, label_col="label", stratify_vars=["scope"],
        check_by_label=True, random_state=42,
    )
    assert class_counts(result) == {0: 20, 1: 20}
    assert result.report.summary["balance_status"] == "pass"
    rows = result.report.balance.to_pylist()
    assert len(rows) == 4  # Three feature scopes plus the descriptive label check.
    assert all(row["statistic"] == 0 for row in rows)


def test_duplicates_missing_categories_and_label_normalization():
    data = pa.table({"Y": [None, None, 0, 0, 1, 1]*10,
                     "g": [None, "__MISSING__", "a|b", "a", "[null]", ""]*10,
                     "h": ["x", "x", "c", "b|c", "x", "x"]*10,
                     "x": pa.array([None]*60, type=pa.float64())})
    r = random_downsample(data, sample_size=60, label_col="Y", stratify_vars=["Y", "g", "h", "x"],
                          check_vars=["Y", "g"], random_state=10)
    assert class_counts(r) == {None: 20, 0: 20, 1: 20}
    assert r.report.summary["n_strata"] == 6
    assert r.report.summary["balance_status"] == "not_evaluated"
    assert r.sample["g"].equals(data["g"])
    assert r.sample["x"].null_count == 60
    assert "Y" not in r.report.summary["checked_features"]
    assert len(set(r.row_ids.to_pylist())) == 60


def test_rare_class_unavailable_and_bounded_reports_keep_complete_summary():
    data = pa.table({"Y": [0]*99 + [1], "x": [1.]*100, "g": [str(i) for i in range(100)]})
    r = random_downsample(data, sample_size=2, label_col="Y", stratify_vars=["g"],
                          check_vars=["x"], check_by_label=True, max_report_rows=1, random_state=0)
    assert len(r.sample) == 2
    assert r.report.summary["omitted_classes"] == 1
    assert r.report.summary["omitted_strata"] == 98
    assert r.report.summary["strata_truncated"] and r.report.summary["balance_truncated"]
    assert r.report.summary["classes_truncated"]
    assert r.report.summary["checks_not_evaluated"] == 2
    assert len(r.report.strata) == len(r.report.classes) == len(r.report.balance) == 1
    assert any("outer budgets" in rec["reason"] for rec in r.report.recommendations)


def test_threshold_boundary_failure_keeps_rows_and_budgets():
    data = pa.table({"Y": [0, 0, 1, 1], "x": [0., 1., 0., 1.]})
    passed = random_downsample(data, sample_size=2, label_col="Y", check_vars=["x"],
                               check_by_label=True, ks_threshold=.5, random_state=5)
    failed = random_downsample(data, sample_size=2, label_col="Y", check_vars=["x"],
                               check_by_label=True, ks_threshold=.49, random_state=5)
    assert passed.report.summary["balance_status"] == "pass"
    assert failed.report.summary["balance_status"] == "fail"
    assert passed.sample.equals(failed.sample)
    assert class_counts(failed) == {0: 1, 1: 1}
    assert failed.report.summary["ks_threshold"] == .49


@pytest.mark.parametrize("kwargs", [
    {"sample_size": 0}, {"sample_size": -1}, {"sample_size": 1.0}, {"sample_size": True},
    {"sample_size": 11}, {"n_bins": 1}, {"n_bins": 2.5}, {"random_state": -1},
    {"random_state": True}, {"random_state": 2**63}, {"ks_threshold": float("nan")},
    {"js_threshold": 1.1}, {"duckdb_threads": 0}, {"max_report_rows": 0},
    {"stratify_vars": "x"}, {"stratify_vars": ["x", "x"]}, {"label_col": "absent"},
    {"check_vars": ["absent"]}, {"check_by_label": True},
])
def test_invalid_requests(kwargs, tmp_path):
    with pytest.raises(ValueError):
        random_downsample(pa.table({"x": np.arange(10)}), **{"sample_size": 2, **kwargs},
                          work_dir=str(tmp_path))
    assert not list(tmp_path.glob("*.duckdb*"))


def test_empty_input_and_failure_cleanup(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="population size"):
        random_downsample(pa.table({"x": pa.array([], type=pa.float64())}), sample_size=1,
                          work_dir=str(tmp_path))
    import rapidmatch.sampling.sampler as sampler
    def fail(*args):
        raise RuntimeError("balance failed")
    monkeypatch.setattr(sampler, "check_population_balance", fail)
    with pytest.raises(RuntimeError, match="balance failed"):
        sampler.random_downsample(pa.table({"x": [1., 2., 3.]}), sample_size=2,
                                   work_dir=str(tmp_path))
    assert not list(tmp_path.glob("*.duckdb*"))


def test_ingestion_failure_removes_partial_database(tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module("rapidmatch.ingestion.ingest")
    def partial_failure(data, path):
        with duckdb.connect(path) as con:
            con.execute("CREATE TABLE partial AS SELECT 1 AS x")
        raise RuntimeError("partial ingestion")
    monkeypatch.setattr(module, "_stream", partial_failure)
    with pytest.raises(RuntimeError, match="partial ingestion"):
        random_downsample(pa.table({"x": [1., 2.]}), sample_size=1, work_dir=str(tmp_path))
    assert not list(tmp_path.glob("*.duckdb*"))


def test_generated_seed_can_replay_and_nonfinite_numeric_values_are_explicit():
    data = pa.table({"x": [1., 2., 3., 4., None, float("nan"), float("inf"), -float("inf")]})
    a = random_downsample(data, sample_size=4, stratify_vars=["x"])
    b = random_downsample(data, sample_size=4, stratify_vars=["x"],
                          random_state=a.report.summary["random_state"])
    assert a.row_ids.equals(b.row_ids)
    assert all(np.isfinite(e) for e in a.report.bin_edges["x"])
    row = a.report.balance.to_pylist()[0]
    assert row["population_missing_rate"] == .5


def test_large_file_backed_sampling_has_no_population_arrow_pull(tmp_path, monkeypatch):
    import rapidmatch.sampling.sampler as sampler
    original = sampler.ingest

    class RelationGuard:
        def __init__(self, relation, sql):
            self.relation, self.sql = relation, sql
        def to_arrow_table(self):
            assert "LIMIT" in self.sql or "FROM _rs_sample ORDER BY _rs_id" in self.sql
            return self.relation.to_arrow_table()

    class ConnectionGuard:
        def __init__(self, con):
            self.con = con
        def sql(self, sql):
            return RelationGuard(self.con.sql(sql), sql)
        def __getattr__(self, name):
            return getattr(self.con, name)

    def guarded_ingest(*args, **kwargs):
        session = original(*args, **kwargs)
        session.con.execute("SET memory_limit = '128MB'")
        session.con = ConnectionGuard(session.con)
        return session

    path = tmp_path / "population.parquet"
    with duckdb.connect() as con:
        con.execute("COPY (SELECT i AS id, i % 10 = 0 AS Y, i % 20 AS x, "
                    "CAST(i % 3 AS VARCHAR) AS g FROM range(200000) t(i)) TO ? (FORMAT PARQUET)", [str(path)])
    monkeypatch.setattr(sampler, "ingest", guarded_ingest)
    r = sampler.random_downsample(str(path), sample_size=2000, label_col="Y",
                                  stratify_vars=["g", "x"], n_bins=20, duckdb_threads=1,
                                  max_report_rows=10, random_state=42, work_dir=str(tmp_path))
    assert len(r.sample) == 2000
    assert class_counts(r) == {False: 1800, True: 200}
    assert not list(tmp_path.glob("*.duckdb*"))
