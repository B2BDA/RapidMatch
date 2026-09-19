import pyarrow as pa

from rapidmatch.balance.checker import BalanceRow
from rapidmatch.drift.correct import correct_drift


def test_trims_overrepresented_category() -> None:
    target = pa.table(
        {
            "_rm_id": [1, 2],
            "seg": ["A", "B"],
        }
    )
    control = pa.table(
        {
            "_rm_id": [10, 11, 12, 13],
            "seg": ["A", "A", "A", "B"],
        }
    )
    flagged = [
        BalanceRow(
            variable="seg",
            role="monitor",
            kind="js",
            statistic=0.2,
            threshold=0.1,
            flagged=True,
        )
    ]
    strength = {10: 0.9, 11: 0.2, 12: 0.1, 13: 0.8}
    result = correct_drift(target, control, strength, flagged)
    assert 12 not in result.kept_control_ids or 11 not in result.kept_control_ids
    assert result.events
    assert result.events[0].variable == "seg"
    assert result.events[0].n_removed >= 1


def test_trims_numeric_bin() -> None:
    target = pa.table(
        {
            "_rm_id": [1, 2, 3],
            "income": [70.0, 71.0, 72.0],
        }
    )
    control = pa.table(
        {
            "_rm_id": [10, 11, 12, 13, 14],
            "income": [70.5, 71.5, 72.5, 90.0, 90.5],
        }
    )
    flagged = [
        BalanceRow(
            variable="income",
            role="monitor",
            kind="ks",
            statistic=0.4,
            threshold=0.1,
            flagged=True,
        )
    ]
    strength = {i: 0.5 for i in (10, 11, 12, 13, 14)}
    result = correct_drift(target, control, strength, flagged, n_bins=2)
    assert result.events
    assert result.events[0].kind == "ks"