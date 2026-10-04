"""Small runnable example: exact size, class proportions, and feature acceptance."""

import numpy as np
import pyarrow as pa

from rapidmatch import random_downsample


def main():
    data = pa.table({
        "id": np.arange(10_000),
        "Y": [0] * 9_000 + [1] * 1_000,
        "region": ["north", "south"] * 5_000,
        "activity": np.tile(np.arange(10), 1_000),
    })
    result = random_downsample(
        data, sample_size=1_000, label_col="Y",
        stratify_vars=["region", "activity"], n_bins=10,
        random_state=42, check_by_label=True,
    )
    print("Requested / returned:", 1_000, len(result.sample))
    print("Feature acceptance:", result.report.summary["balance_status"])
    print("Class allocation:", result.report.classes.to_pylist())
    print("Sample preview:", result.sample.slice(0, 5).to_pylist())


if __name__ == "__main__":
    main()
