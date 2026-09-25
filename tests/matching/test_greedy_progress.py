import numpy as np

from rapidmatch.matching.greedy_match import greedy_match


def test_on_progress_reports_assigned_count() -> None:
    target_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    control_ids = np.array([10, 11, 12, 13], dtype=np.int64)
    strengths = np.array([0.9, 0.8, 0.7, 0.6])
    seen = []

    def _note(n_assigned, n_seen):
        seen.append((n_assigned, n_seen))

    assigned = greedy_match(
        target_ids,
        control_ids,
        strengths,
        n=1,
        on_progress=_note,
    )

    assert len(assigned) == 4
    assert seen
    assert seen[-1] == (4, 4)
