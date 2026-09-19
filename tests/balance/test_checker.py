import pandas as pd

from rapidmatch.balance.checker import check_balance


def test_flags_monitor_var_over_threshold() -> None:
    target = pd.DataFrame({"income": [1.0, 2.0, 3.0], "seg": ["A", "A", "B"]})
    control = pd.DataFrame({"income": [1.1, 2.1, 3.1], "seg": ["A", "A", "A"]})
    rows = check_balance(
        target,
        control,
        match_vars=["income"],
        monitor_vars=["seg"],
        dtypes={"income": "DOUBLE", "seg": "VARCHAR"},
        js_threshold=0.05,
        ks_threshold=0.5,
    )
    by_var = {r.variable: r for r in rows}
    assert by_var["income"].role == "match"
    assert by_var["income"].kind == "ks"
    assert by_var["seg"].role == "monitor"
    assert by_var["seg"].kind == "js"
    assert by_var["seg"].flagged
