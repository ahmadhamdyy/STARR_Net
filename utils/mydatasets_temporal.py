"""
Temporal extension of `mydatasets.MultiScaleProbabilistic`. Positive samples use
their recorded crash time; negatives receive reproducible uniform 7 x 24 queries.
"""

from __future__ import annotations

import csv
import os
import random
import warnings
from typing import Dict, Tuple

import torch

from mydatasets import MultiScaleProbabilistic


CoordinateKey = Tuple[float, float]


def _load_time_lookup_from_csv(path: str) -> Dict[CoordinateKey, list[float]]:
    lut: Dict[CoordinateKey, list[float]] = {}
    if not os.path.isfile(path):
        return lut
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return lut
        need = {"latitude", "longitude", "DAY_WEEK", "HOUR"}
        if not need.issubset(set(reader.fieldnames)):
            warnings.warn(f"{path}: missing columns {need - set(reader.fieldnames)}; skipping")
            return lut
        for row in reader:
            try:
                key = (float(row["latitude"]), float(row["longitude"]))
            except (TypeError, ValueError):
                continue
            vec = [float(row["DAY_WEEK"] or 0), float(row["HOUR"] or 0)]
            lut[key] = vec
    return lut


class MultiScaleProbabilisticTemporal(MultiScaleProbabilistic):
    """
    Like `MultiScaleProbabilistic`, plus `temporal` in __getitem__:
      tensor shape [2] = [DAY_WEEK, HOUR]

    Time rows are matched using unrounded float latitude/longitude values.
    """

    _DEFAULT_TIME = [4.0, 12.0]

    def __init__(
        self,
        split: str = "train",
        time_csv_root: str = ".",
        temporal_seed: int = 0,
        **kwargs,
    ):
        super().__init__(split=split, **kwargs)
        self.split = split
        root = os.path.abspath(time_csv_root)
        pos_path = os.path.join(root, f"{split}_pos_time_data.csv")
        pos_lut = _load_time_lookup_from_csv(pos_path)
        self._time_lut = pos_lut
        split_offset = {"train": 0, "val": 1, "test": 2}.get(split, 3)
        rng = random.Random(temporal_seed + split_offset)
        self._negative_times = [
            [float(rng.randint(1, 7)), float(rng.randrange(24))]
            for _ in range(len(self.metadata))
        ]
        self._warned_missing = False

    def __getitem__(self, index):
        filename = str(self.metadata[index][0]).replace("\ufeff", "")
        filename_parts = filename.split("_")
        key = (float(filename_parts[0]), float(filename_parts[1]))
        label = min(int(filename_parts[-1][0]), 1)

        out = super().__getitem__(index)
        # The paper uses recorded crash time for positives and an independent
        # uniform draw over the 7 x 24 grid for each negative.
        vec = self._time_lut.get(key) if label == 1 else self._negative_times[index]
        if vec is None:
            if not self._warned_missing:
                warnings.warn(
                    f"No time row for lat/lon key {key}; "
                    f"using default {self._DEFAULT_TIME}. Suppressing further warnings."
                )
                self._warned_missing = True
            vec = list(self._DEFAULT_TIME)
        out["temporal"] = torch.tensor(vec, dtype=torch.float32)
        return out
