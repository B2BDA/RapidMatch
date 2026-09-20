# Identity-preserving speed pack

**Status:** Implemented on branch `260920-feat-identity-preserving-speed-pack`. Implementation plan: `docs/superpowers/plans/2026-09-20-identity-preserving-speed-pack.md`.

**Scope:** Rewrite three hot paths so the same input and `MatchConfig` produce bit-identical `pairs`, `targets`, cutoff, ranks, statuses, thin flags, and `match_strength` values. No new dependencies. Serial default stays the default.

**Out of scope:** Gram/2D distance, float32, approx quantiles, nearest-K / KD-tree, parallel greedy, Numba, per-stratum sort + k-way merge, parallel JS/KS, streaming merge into greedy, changing `_pull`, Parquet export, `memory_limit`, CLI, viz, `save_run()`, cardinality guardrails, FastAPI.

---

## 1. Goal

Make scoring and greedy matching cheaper in CPU and peak RAM **without changing any matching outcome**.

Identity bar:

- Same input + same `MatchConfig` → identical `result.pairs` and `result.targets` (ids, ranks, statuses, thin flags).
- Identical `match_strength` bits (`np.array_equal`, not `allclose`).
- Identical `result.cutoff`.
- Default `n_workers=None` remains serial and bit-identical to today's serial path.
- Existing parallel-vs-serial scoring tests keep passing.

If a change cannot meet that bar, it does not belong in this pack.

---

## 2. Current bottlenecks (code as of `origin/main`)

### 2.1 `score_all_strata` (`rapidmatch/scoring/score_strata.py`)

For each eligible stratum key, the worker does `strata_labels == stratum` over the **full** `N`-length array. Cost is `O(N × n_strata)` before any distance math. Results are concatenated in `eligible` order (`pool.map` already preserves that order).

### 2.2 `score_pairs` (`rapidmatch/scoring/scorer.py`)

Builds a full `(n_target, n_control, n_dim)` float64 tensor:

```text
delta = zt[:, None, :] - zc[None, :, :]
dist  = sqrt(sum(delta * delta, axis=2))
```

Then `meshgrid(..., indexing="ij")` ravel order: target-major, control inner. A 5k×5k×4 stratum is ~800MB for `delta` alone.

### 2.3 `greedy_match` (`rapidmatch/matching/greedy_match.py`)

`np.argsort(-strengths, kind="mergesort")`, then a Python loop that:

- boxes every id with `int(...)`
- membership-tests `used_control: set[int]`
- updates `target_slots: defaultdict(int)`

Walk is batched in 4096-row chunks only so the progress bar can tick. Algorithm is already global greedy without replacement.

### 2.4 Why array indexing is valid

`_rm_id` is assigned as `ROW_NUMBER() OVER ()` in `v_source` (`rapidmatch/ingestion/ingest.py`). It is 1-based, dense, unique for the run. A boolean / int mask of length `max(_rm_id)+1` is therefore a correct replacement for a Python set / defaultdict.

---

## 3. Changes

Three files. No public API, config, or pipeline orchestration changes.

### 3.1 Split-once by stratum — `score_all_strata`

**Keep:**

- Score only keys in `strata` (the eligible list), in that list's order.
- Parallel path: `ThreadPoolExecutor.map` with the same worker count rule (`n_workers is not None and n_workers > 1`).
- Concatenate non-empty `(t_ids, c_ids, strengths)` in that same order.
- Empty eligible list still returns three empty arrays (`int64`, `int64`, `float64`).
- Progress bar still updated once with `len(results)` after scoring.

**Change:**

Index rows by stratum **once** before scoring:

1. One forward scan of `strata_labels` / `treatment` that appends row indices into per-key target and control lists (appearance order). Drop keys that are not in the `strata` argument.
2. `_score(stratum)` slices `ids` / `x` with those precomputed index arrays and calls `score_pairs` unchanged.

Do **not** rescan `strata_labels == stratum` inside `_score`. Do not sort ids inside a stratum. Row order must match today's `in_stratum & (treatment == 1)` mask order (first appearance in the pulled table first).

### 3.2 Chunked 3D scoring — `score_pairs`

**Keep:**

- Global z-score with `std = where(target_std == 0, 1.0, target_std)`.
- Weights as `float64` via `config.weight_for`.
- Distance `sqrt(sum(delta * delta, axis=2))` on a `(chunk_t, n_control, n_dim)` tensor — **same formula, not the Gram rewrite**. Never chunk on the control axis.
- Categorical-only (`weights.size == 0`): zeros of shape `(n_target, n_control)`, then `exp`.
- Output ravel order: target-major, control inner (today's `meshgrid(..., indexing="ij").ravel()`).
- Empty target or control: three empty arrays as today.

**Change:**

1. Replace `meshgrid` index arrays with equivalent construction:
   - `target_ids_out = np.repeat(target_ids, n_control)`
   - `control_ids_out = np.tile(control_ids, n_target)`
   - `strengths_out` raveled the same way.
2. If the 3D tensor would exceed **16,000,000 cells** (`n_target * n_control * n_dim`), score successive target-row blocks against **all** controls and concatenate in block order. `block_t = max(1, 16_000_000 // (n_control * max(n_dim, 1)))`. Categorical-only (no numeric dim) does not allocate a 3D tensor and is not chunked.

Chunking is an internal implementation detail, not a `MatchConfig` knob. The constant lives as a module-private name in `scorer.py` (e.g. `_MAX_DISTANCE_CELLS = 16_000_000`) so tests can monkeypatch it to force more than one block on a tiny fixture.

Block order: targets `[0:b)`, `[b:2b)`, … in original `target_ids` order. Within a block, control order is the original `control_ids` order. Concatenating those blocks is identical to scoring the full matrix at once.

`exp(-dist)` is applied per block (or once on the concatenated `dist`); both are the same IEEE op on the same `dist` values.

### 3.3 Array greedy — `greedy_match`

**Keep:**

- `np.argsort(-strengths, kind="mergesort")` as the only ordering.
- Skip if control already used; skip if target already has `n` matches; else assign with `rank = slots[t] + 1`.
- Return `list[tuple[int, int, float, int]]` in walk order (same as today).
- Empty `target_ids` → `[]`.
- Progress bar: `total = len(order)`; update every 4096 steps with `min(4096, remaining)`.

**Change:**

```text
max_id = int(max(target_ids.max(), control_ids.max()))
used_control = np.zeros(max_id + 1, dtype=bool)
target_slots = np.zeros(max_id + 1, dtype=np.int32)
```

Index with the raw `_rm_id` values. Do not use a Python `set` or `defaultdict`.

`n` stays a Python int compared against `target_slots[t]`. Assignment still appends `(int(t), int(c), float(strengths[i]), rank)` so the public tuple types do not change.

If `target_ids` is non-empty, ids are positive (`ROW_NUMBER` is 1-based). No extra empty-id branch beyond today's `len == 0` guard.

---

## 4. Explicitly not in this pack

| Idea | Reason |
|---|---|
| Gram form `zt² + zc² − 2 zt·zc` | Different float association; strengths / cutoff / assignments can move |
| `float32` scoring | Same |
| `approx_quantile` | Different bin edges |
| Nearest-K / KD-tree | Drops pairs; changes Module 7 contract |
| Parallel greedy | Breaks strongest-first / no-reuse |
| Numba | New extra; not required for the identity bar |
| Per-stratum sort + k-way merge | Same algorithm only with an extra tie-key; no RAM win if pairs are still concatenated; wait for a benchmark |
| Parallel JS/KS | Identity-safe but not these three files |
| Never-materialize-all-pairs / streaming merge | Real RAM win, larger proof; next pack |
| Split `_pull` by stratum | Pipeline change; balance/drift still need the full table |
| Config knobs for chunk size, memory_limit, etc. | API change |

---

## 5. Files

| File | Role |
|---|---|
| `rapidmatch/scoring/score_strata.py` | Split-once indices; `_score` uses slices |
| `rapidmatch/scoring/scorer.py` | Chunked 3D distance; `repeat`/`tile` instead of `meshgrid` |
| `rapidmatch/matching/greedy_match.py` | Boolean / int masks instead of `set` / `defaultdict` |
| `tests/scoring/test_score_strata.py` | Keep existing parallel identity tests |
| `tests/scoring/test_scorer.py` | Keep bounded/identical-row tests; add chunk-vs-full identity (monkeypatch `_MAX_DISTANCE_CELLS`) |
| `tests/matching/test_greedy_match.py` | Keep reuse / n-slot tests; add 1-based / gapped ids so the mask is `max_id+1` |

Do not keep a permanent dual implementation in production modules. Tests may inline a tiny reference (copy of today's formula) to lock bits; production has one path.

`pipeline.py`, `config.py`, public `__init__`, and reporting are untouched.

---

## 6. Testing

1. **Existing tests** must stay green without assertion changes that weaken identity (`array_equal` stays `array_equal`).
2. **`score_pairs` chunk identity:** construct `n_target`, `n_control` large enough to force more than one block (monkeypatch the private chunk constant in the test if needed). Compare concatenated output to an unchunked reference that uses today's `delta` formula. `array_equal` on all three arrays.
3. **`score_all_strata` order:** several strata, mixed empty/non-empty, `n_workers` in `{None, 1, 2}`. Outputs `array_equal` across worker counts; pair order matches `eligible` order then in-stratum appearance order.
4. **`greedy_match`:** existing n=1 / n=2 fixtures; plus ids that are not `0..n-1` (1-based, gaps unused) to prove mask length is `max_id+1`, not `len(ids)`.
5. **No pipeline golden.** `_rm_id` is `ROW_NUMBER() OVER ()` without `ORDER BY`, so a full `fit_match` golden is not a reliable identity lock. Unit tests on arrays are the source of truth for this pack. Existing `tests/test_pipeline.py` still must pass unchanged.

No performance assertion in CI (no "must be faster"). Speed is not the merge gate; identity is.

---

## 7. Error handling and progress

- No new exceptions.
- Empty inputs keep today's empty returns.
- `pbar` in scoring still updated after all strata finish (`pbar.update(len(results))`), not per stratum — current behavior.
- `pbar` in greedy still 4096-batched.

---

## 8. Rollout

Single PR. No feature flag. Behavior is the new (identical) path for everyone, including `n_workers=None`.

After merge, `plan.md` changelog notes: identity-preserving rewrite of scoring index, pair materialization, and greedy occupancy; no algorithm change.

---

## 9. Success criteria

- `python3 -m pytest tests -q` passes.
- New identity tests prove chunked `score_pairs` equals the full 3D formula, and greedy assignments equal the current walk on the same pair list.
- No new package extra or dependency.
- `MatchConfig` field list unchanged.
- Default serial path remains the documented default.
