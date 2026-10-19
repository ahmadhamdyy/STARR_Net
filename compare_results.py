"""Collect Table 1 metrics from completed variant runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

VARIANTS = ("mscm_ms", "betarisk", "starr_loc", "starr_time", "starr_full")
METRICS = ("brier", "auc", "ece", "selective_ausc", "f1", "recall")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="runs/comparison.csv")
    args = parser.parse_args()

    rows = []
    for variant in VARIANTS:
        path = Path(args.runs_root) / f"{variant}_seed{args.seed}" / "test_metrics.json"
        if not path.is_file():
            print(f"Skipping {variant}: {path} not found")
            continue
        with open(path, encoding="utf-8") as file:
            metrics = json.load(file)
        rows.append(
            {
                "variant": variant,
                "checkpoint_epoch": metrics.get("checkpoint_epoch", ""),
                **{name: metrics[name] for name in METRICS},
            }
        )

    if not rows:
        raise SystemExit("No completed test_metrics.json files were found.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("variant", "checkpoint_epoch", *METRICS))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
