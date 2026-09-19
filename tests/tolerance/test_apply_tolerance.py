from rapidmatch.tolerance.apply_tolerance import apply_tolerance


def test_below_tolerance_when_uncovered() -> None:
    assignments = [
        (1, 10, 0.95, 1),
        (2, 11, 0.10, 1),
        (1, 12, 0.05, 2),
    ]
    kept, below, cutoff = apply_tolerance(assignments, tolerance=0.5)
    assert cutoff >= 0.10
    assert all(a[2] >= cutoff for a in kept)
    if 2 not in {a[0] for a in kept}:
        assert 2 in below
    assert 1 not in below or 1 in {a[0] for a in kept}
