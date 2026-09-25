# RapidMatch (RiMatch) — working memory

Repo: https://github.com/B2BDA/RapidMatch.git
Package import: `rapidmatch`. Pronunciation: "rematch".
Python >= 3.11. Compute: DuckDB (prep) then NumPy on a PyArrow pull (score/match). No ML, no PySpark, no sklearn.

Maps to read first: `codegraph.md` (running code), `plan.md` (locked design). Modules 1-13 are built. Module 14 (FastAPI) is not.

## What it does

Given a table with a binary treatment flag (`1` = campaign target, `0` = untreated) and pre-campaign features, build a control group from the `0` rows whose distribution mirrors the `1` group. Rule-based: target-only quantile bins, composite strata, global z-score, weighted Euclidean distance, greedy assignment without replacement, global strength cutoff.

## Public API

```python
from rapidmatch import ControlMatcher, MatchConfig, MatchResult

result = ControlMatcher(MatchConfig(
    match_vars=["income", "age", "region"],
    treatment_col="is_target",
    id_col="id",
    weights={"income": 1.5},
    n=1,
    tolerance=0.8,           # default: keep strongest 20%
    min_control_pool_size=5,
    n_bins=4,
    monitor_vars=["tenure"],
    js_threshold=0.10,
    ks_threshold=0.05,
)).fit_match(df)  # path, pandas, or Arrow
```

Exports (`rapidmatch/__init__.py`): `ControlMatcher`, `MatchConfig`, `MatchResult`, `Report`.

Entry: `ControlMatcher.fit_match` in `rapidmatch/pipeline.py`.

## Pipeline (`ControlMatcher._run`)

1. `ingest` -> DuckDB file, `udl_data` then `v_source` (`_rm_id` = ROW_NUMBER).
2. `DataReport.summary()` — batched nulls/types/target counts (lazy until `.summary()`).
3. `validate_schema` + `classify_match_vars` (numeric vs categorical).
4. `handle_missing` -> `v_prepared`. Numeric null -> 0 + `is_missing_<var>`. Cat null -> `"__MISSING__"`. Flags stratify only, never enter distance.
5. `Stratifier.run` -> `v_stratified`. Quantile edges from `_treatment=1` only; same edges on control. Stratum = `_bin_<num>` + cats + missing flags.
6. `flag_coverage`: `n_control==0` -> `no_control` (not scored). Thin (`< min_control_pool_size` / ratio) still eligible and still matched; `thin_stratum` is a boolean, not a status.
7. `_pull(v_stratified)` to Arrow. `target_moments` once on whole target group. `score_all_strata` on eligible keys only.
8. `greedy_match` global sort by strength desc (stable), no control reuse, `n` slots per target. Occupancy is a dense mask on `_rm_id`, not a Python set.
9. `apply_tolerance`: cutoff = quantile(accepted strengths, tolerance). `0` keeps all; `0.8` keeps top 20%.
10. `check_balance` (JS cats, KS nums) on match_vars + monitor_vars. Flagged *monitor* groups: `correct_drift` trims weakest matched controls only (no swap-in). Re-check after.
11. `assemble` then `build_report`.

DuckDB views in order: `udl_data` -> `v_source` -> `v_prepared` -> `v_stratified`.

## Scoring (Module 7)

- Global target mean/std per numeric match_var. Constant std treated as 1.
- `z = (x - mean) / std`, same moments on control.
- Distance = weighted Euclidean on z: `d = sqrt(sum_k (a_k - b_k)^2)`,
  computed as `||a||^2 + ||b||^2 - 2 a·b`. `match_strength = exp(-d)` in `(0, 1]`.
- Categorical-only stratum: distance 0, strength 1 for every pair.
- Pair ids: `repeat`/`tile` target-major. Chunk target rows if the 2D
  distance grid > 16e6 cells; never chunk controls.
- `max_candidates_per_target`: keep K strongest controls per target after scoring (`np.argpartition`). Untouched if `n_control <= K`.
- `n_workers > 1`: ThreadPoolExecutor, order-preserving, bit-identical to serial.

## Match status (`output/assemble.py`)

| status | meaning |
|---|---|
| `matched` | at least one pair survived tolerance |
| `no_control_available` | stratum had zero control rows |
| `below_tolerance` | had assignments, all below cutoff |
| `unmatched` | eligible but lost every control slot to stronger pairs |

Every target appears in `result.targets`. Never silently dropped.

`MatchResult`: `pairs`, `targets`, `cutoff`, `coverage_summary`, `report`.

## Config defaults (`config.py`, frozen)

- `n=1`, `tolerance=0.8`, `min_control_pool_size=5`, `n_bins=4`
- `js_threshold=0.10`, `ks_threshold=0.05`
- `match_vars` and `monitor_vars` must not overlap; weight keys subset of `match_vars`
- `n_workers` / `duckdb_threads` / `progress` opt-in; unset path is the serial identity

## Layout

```
rapidmatch/
  pipeline.py           ControlMatcher
  config.py             MatchConfig
  ingestion/            ingest, validate, vendored UniversalDataLoader
  data_report/          DataReport
  missingness/          handle_missing
  binning/              Stratifier + edges/bucket/stratum/counts
  coverage/             flag_coverage
  scoring/              scorer, score_strata
  matching/             greedy_match
  tolerance/            apply_tolerance
  output/               assemble, MatchResult
  balance/              JS/KS checker
  drift/                diagnose + trim-only correct
  reporting/            coverage / balance / drift_log / profile
tests/                  1:1 with step files
```

`ingestion/data_loader.py` is a vendored RapidSegment copy. Do not rewrite.

## Invariants

1. No control row reused.
2. Strength in `(0, 1]`.
3. Bin edges and z-moments from target group only, never per stratum.
4. Every target in `result.targets`.
5. Missing flags stratify only, not distance.
6. pandas is not used for computation.
7. We open DuckDB after `stream_to_duckdb` returns a path; temp `.duckdb` deleted unless `keep_db`.
8. Thin strata still match.
9. Drift only removes excess control rows; weakest first.

## Tests / demo

```
python3 -m pytest tests -q
python3 demo.py
```

Do not add FastAPI, PySpark, or sklearn.
