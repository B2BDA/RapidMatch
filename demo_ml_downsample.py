"""End-to-end ML walkthrough from README. Writes artifacts under the given folder.

Usage: python demo_ml_downsample.py [output_directory]
"""

import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from rapidmatch import random_downsample


def make_training_population():
    rng = np.random.default_rng(42)
    n = 100_000
    y = np.array([0] * 90_000 + [1] * 10_000)
    rng.shuffle(y)
    return pa.table({
        "customer_id": np.arange(n),
        "Y": y,
        "region": rng.choice(["north", "south", "west"], n, p=[0.5, 0.3, 0.2]),
        "age": np.clip(rng.normal(40 + 5 * y, 12, n), 18, 80),
        "income": rng.lognormal(10.5 + 0.2 * y, 0.45, n),
        "tenure": rng.integers(0, 15, n),
        "spend": rng.gamma(2.0 + y, 300, n),
    })


def describe(result, title):
    print(f"\n{title}")
    summary = result.report.summary
    print({k: summary[k] for k in (
        "n_population", "requested_rows", "sampled_rows", "balance_status",
        "checks_failed", "checks_not_evaluated", "omitted_strata",
    )})
    print("Class budgets:", result.report.classes.to_pylist())
    balance = result.report.balance
    issues = balance.filter(pc.and_(balance["required"], pc.not_equal(balance["status"], "pass")))
    print("Failed/unavailable feature checks:", issues.to_pylist())
    for rec in result.report.recommendations:
        print(
            f"Recommendation [{rec['evidence']}]: {rec['setting']} -> {rec['proposed_value']}\n"
            f"  Reason: {rec['reason']}\n  Trade-off: {rec['tradeoff']}"
        )
    if not result.report.recommendations:
        print("No recommendations for this run.")


def main(output_directory="ml_downsampling_demo"):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    input_path = output / "training_population.parquet"
    population = make_training_population()
    pq.write_table(population, input_path)
    del population

    settings = dict(
        sample_size=1_000, label_col="Y", stratify_vars=["region", "age", "income"],
        check_vars=["tenure", "spend"], n_bins=4, random_state=42,
        ks_threshold=0.05, js_threshold=0.10, check_by_label=True,
    )
    result = random_downsample(str(input_path), **settings)
    describe(result, "Initial 1,000-row candidate")
    # An explicit caller decision for this example, not an automatic library retry.
    # Assume the training memory budget can accommodate 10,000 rows.
    if result.report.summary["balance_status"] != "pass":
        settings = {**settings, "sample_size": 10_000}
        result = random_downsample(str(input_path), **settings)
        describe(result, "Explicitly requested 10,000-row candidate")

    report_path = output / "training_sample.report.json"
    report_path.write_text(json.dumps(result.report.to_dict(), indent=2), encoding="utf-8")
    if result.report.summary["balance_status"] == "pass":
        pq.write_table(result.sample, output / "training_sample.parquet")
        x_train = result.sample.drop(["customer_id", "Y"])
        y_train = result.sample["Y"]
        print(f"Training-ready features: {x_train.num_rows} rows x {x_train.num_columns} columns")
        print(f"Training labels: {len(y_train)}")
        print("Saved accepted sample and report to:", output.resolve())
    else:
        print("Feature acceptance did not pass; inspect the saved report before choosing another configuration.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ml_downsampling_demo")
