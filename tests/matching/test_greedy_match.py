import numpy as np

from rapidmatch.matching.greedy_match import greedy_match


def test_control_never_reused() -> None:
    target_ids = np.array([1, 2, 1, 3])
    control_ids = np.array([7, 7, 3, 9])
    strengths = np.array([0.91, 0.88, 0.85, 0.80])
    assigned = greedy_match(target_ids, control_ids, strengths, n=2)
    controls = [a[1] for a in assigned]
    assert len(controls) == len(set(controls))
    assert (1, 7, 0.91, 1) in assigned
    assert (1, 3, 0.85, 2) in assigned
    assert (2, 7, 0.88, 1) not in assigned
    assert (3, 9, 0.80, 1) in assigned


def test_n_slots() -> None:
    target_ids = np.array([1, 1, 1])
    control_ids = np.array([8, 9, 10])
    strengths = np.array([0.9, 0.8, 0.7])
    assigned = greedy_match(target_ids, control_ids, strengths, n=1)
    assert len(assigned) == 1
    assert assigned[0][1] == 8
