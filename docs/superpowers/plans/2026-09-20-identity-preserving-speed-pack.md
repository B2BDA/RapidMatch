# Identity-preserving speed pack — implementation plan

**Status:** Implemented (Tasks 1–4). Branch `260920-feat-identity-preserving-speed-pack`. Not committed.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite three hot paths so scoring and greedy matching use less CPU and peak RAM without changing any matching outcome.

**Architecture:** Production still has one path (no dual implementation, no feature flag). Tests may inline a tiny copy of today's `delta` formula to lock bits. Chunk size is a module-private constant, not a `MatchConfig` knob.

**Tech stack:** NumPy, existing `ThreadPoolExecutor` scoring, no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-20-identity-preserving-speed-pack-design.md`

---

## File map

| File | Change |
|---|---|
| `rapidmatch/scoring/scorer.py` | `_MAX_DISTANCE_CELLS`, chunked 3D distance, `repeat`/`tile` |
| `rapidmatch/scoring/score_strata.py` | One forward scan into per-stratum index lists |
| `rapidmatch/matching/greedy_match.py` | Boolean / int32 masks instead of `set` / `defaultdict` |
| `tests/scoring/test_scorer.py` | Chunk-vs-full identity (monkeypatch `_MAX_DISTANCE_CELLS`) |
| `tests/scoring/test_score_strata.py` | Keep existing tests; they already lock worker-count identity |
| `tests/matching/test_greedy_match.py` | 1-based / gapped ids |
| `plan.md` | Changelog note after tests pass |

Do not edit `pipeline.py`, `config.py`, public `__init__`, or reporting.

---

### Task 1: Chunked `score_pairs`

**Files:**

- Modify: `rapidmatch/scoring/scorer.py`
- Modify: `tests/scoring/test_scorer.py`

**Steps:**

1. In `scorer.py`, add a module-private constant:

```python
_MAX_DISTANCE_CELLS = 16_000_000
```

2. Keep `target_moments` unchanged.

3. Rewrite `score_pairs` so that:

   - Empty `target_ids` or `control_ids` still returns three empty arrays (`int64`, `int64`, `float64`).
   - `weights` is still `float64` via `config.weight_for`.
   - `std = np.where(target_std == 0, 1.0, target_std)`.
   - If `weights.size == 0` (categorical-only): `dist = np.zeros((n_t, n_c), dtype=np.float64)` — **not chunked**. Then `strength = np.exp(-dist)`.
   - Else z-score and weight `zt` / `zc` as today, then:
     - `n_dim = zt.shape[1]`
     - If `n_t * n_c * n_dim <= _MAX_DISTANCE_CELLS`, score the full tensor once.
     - Else `block_t = max(1, _MAX_DISTANCE_CELLS // (n_c * max(n_dim, 1)))` and score target slices `[0:b)`, `[b:2b)`, … against **all** controls. Never chunk on the control axis.
   - Distance formula on each block (must be this, not Gram):

```python
delta = zt_block[:, None, :] - zc[None, :, :]
dist = np.sqrt(np.sum(delta * delta, axis=2))
```

   - Apply `np.exp(-dist)` per block (same IEEE op as once on the full `dist`).
   - Replace `meshgrid` with:

```python
t_out = np.repeat(target_ids, n_c)
c_out = np.tile(control_ids, n_t)
```

     For the chunked path, build `t_out` / `c_out` / `s_out` per block with `repeat(target_ids[start:end], n_c)` and `tile(control_ids, block_len)`, then `np.concatenate`. Concatenation order must equal full-matrix ravel order (target-major, control inner).

4. Do not add comments unless needed to match file style (this file already has short comments; keep the existing ones about constant columns and categorical-only strata).

5. In `tests/scoring/test_scorer.py`, keep the two existing tests. Add:

   - `test_repeat_tile_order_matches_meshgrid` on a small 2×3 numeric fixture: compare `score_pairs` outputs to an inline reference that uses today's `meshgrid` + full `delta`. `np.testing.assert_array_equal` on all three arrays.
   - `test_chunked_score_equals_full_tensor` that monkeypatches `rapidmatch.scoring.scorer._MAX_DISTANCE_CELLS` to a tiny value (e.g. `4`) so a 3-target × 3-control × 2-dim fixture takes more than one block. Compare to the same inline full-tensor reference. `array_equal`.
   - `test_categorical_only_not_chunked` (`numeric_vars=[]`, empty `target_x`/`control_x` with shape `(n, 0)`): strengths all `1.0`, ids in target-major order.

6. Run:

```bash
python3 -m pytest tests/scoring/test_scorer.py -q
```

Expected: all pass.

---

### Task 2: Split-once `score_all_strata`

**Files:**

- Modify: `rapidmatch/scoring/score_strata.py`
- Modify: `tests/scoring/test_score_strata.py` (only if a new case is needed; existing parallel tests must stay)

**Steps:**

1. After the empty-`strata` early return, build index lists with **one** forward scan:

```python
wanted = set(strata)
target_idx: dict[str, list[int]] = {s: [] for s in wanted}
control_idx: dict[str, list[int]] = {s: [] for s in wanted}
for i, key in enumerate(strata_labels):
    if key not in wanted:
        continue
    if treatment[i] == 1:
        target_idx[key].append(i)
    elif treatment[i] == 0:
        control_idx[key].append(i)
```

   Appearance order only. Do not sort. Drop keys not in `strata`.

2. `_score(stratum)` must **not** do `strata_labels == stratum`. Instead:

```python
ti = np.asarray(target_idx[stratum], dtype=np.int64)
ci = np.asarray(control_idx[stratum], dtype=np.int64)
return score_pairs(ids[ti], ids[ci], x[ti], x[ci], ...)
```

   Empty lists → empty int64 arrays so `score_pairs` hits its existing empty branch.

3. Keep: `ThreadPoolExecutor` only when `n_workers is not None and n_workers > 1`; `pool.map` over `strata` in given order; concatenate non-empty results in that order; `pbar.update(len(results))` once after scoring.

4. Existing tests in `tests/scoring/test_score_strata.py` already lock serial vs `n_workers=2` and order across worker counts. Leave them. Optionally add `test_ineligible_stratum_is_ignored` (a label present in `strata_labels` but absent from `eligible` must not appear in the concatenated ids). Only add it if cheap.

5. Run:

```bash
python3 -m pytest tests/scoring -q
```

Expected: all pass, including `assert_array_equal` worker-count tests.

---

### Task 3: Array-mask `greedy_match`

**Files:**

- Modify: `rapidmatch/matching/greedy_match.py`
- Modify: `tests/matching/test_greedy_match.py`

**Steps:**

1. Remove `from collections import defaultdict`. Drop `used_control: set` and `target_slots: defaultdict`.

2. After the `len(target_ids) == 0` guard:

```python
max_id = int(max(int(target_ids.max()), int(control_ids.max())))
used_control = np.zeros(max_id + 1, dtype=bool)
target_slots = np.zeros(max_id + 1, dtype=np.int32)
```

3. Keep `order = np.argsort(-strengths, kind="mergesort")` and the 4096 progress batching. Inside the walk:

```python
t = int(target_ids[i])
c = int(control_ids[i])
if used_control[c]:
    continue
if target_slots[t] >= n:
    continue
rank = int(target_slots[t]) + 1
assigned.append((t, c, float(strengths[i]), rank))
target_slots[t] = rank
used_control[c] = True
```

4. Return type stays `list[tuple[int, int, float, int]]`. Empty input still `[]`.

5. Keep the two existing tests. Add `test_gapped_one_based_ids`:

   - `target_ids = [10, 20, 10]`, `control_ids = [100, 100, 30]`, strengths `[0.9, 0.8, 0.7]`, `n=2`.
   - Expect assignments `(10, 100, 0.9, 1)` and `(10, 30, 0.7, 2)`; skip the second use of 100.
   - This fails if someone sizes the mask as `len(ids)` instead of `max_id+1`.

6. Run:

```bash
python3 -m pytest tests/matching/test_greedy_match.py -q
```

Expected: all pass.

---

### Task 4: Full regression + changelog

**Files:**

- Modify: `plan.md` (changelog only)
- Verify: entire `tests/` tree

**Steps:**

1. Run:

```bash
python3 -m pytest tests -q
```

Expected: same pass count as before plus the new tests. Do not weaken any `array_equal` to `allclose`.

2. Append a changelog bullet to `plan.md` §8, matching the existing style, stating: identity-preserving rewrite of scoring index, pair materialization, and greedy occupancy; no algorithm change; no public API change.

3. Do not touch `codegraph.md` unless a reader of that file would be misled (it currently describes all-pairs scoring and global greedy — still true). Skip it unless a sentence claims `set()` occupancy or per-stratum `==` scans.

---

## Verification (done when all of these hold)

- `python3 -m pytest tests -q` passes.
- Chunked `score_pairs` equals the inline full 3D formula (`array_equal`).
- `score_all_strata` still identical across `n_workers` in `{None, 1, 2}`.
- Greedy still no-reuse / n-slots, including gapped 1-based ids.
- No new dependency in `pyproject.toml`.
- `MatchConfig` field list unchanged.

## Out of scope (do not implement)

Gram distance, float32, approx quantiles, nearest-K, parallel greedy, Numba, k-way merge, parallel JS/KS, streaming pairs, `_pull` changes, `memory_limit`, CLI, viz, `save_run()`, FastAPI, pipeline goldens.
