from rapidmatch.tolerance.apply_tolerance import apply_tolerance


def test_below_tolerance_when_uncovered() -> None:
    assignments = [
        (1, 10, 0.95, 1),
        (2, 11, 0.10, 1),
        (1, 12, 0.05, 2),
    ]
    kept, below, cutoff = apply_tolerance(assignments, tolerance=0.5)
    assert cutoff >= 0.10
    assert any(a[2] >= cutoff for a in kept)
    assert 2 in {a[0] for a in kept}
    assert 2 not in below
    assert {a[0] for a in kept} == {1, 2}


def test_tolerance_can_drop_primary_assignment_when_requested() -> None:
    assignments = [
        (1, 10, 0.95, 1),
        (2, 11, 0.10, 1),
    ]
    kept, below, _ = apply_tolerance(
        assignments,
        tolerance=0.5,
        preserve_primary=False,
    )
    assert {a[0] for a in kept} == {1}
    assert below == {2}
