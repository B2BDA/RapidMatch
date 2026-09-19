import pandas as pd

from rapidmatch.balance.checker import BalanceRow
from rapidmatch.drift.correct import correct_drift


def test_trims_overrepresented_category() -> None:
    target = pd.DataFrame(
        {
            "_rm_id": [1, 2],
            "seg": ["A", "B"],
        }
    )
    control = pd.DataFrame(
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
