from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd


class GateCalibrator:
    def __init__(self, path: str, 
                 cutoffs: list[float]) -> None:
        self.df = pd.read_csv(path)
        self.cutoffs = cutoffs
        self.label_col = "expected_service"
        self.general = "general"
        self.score_cols = [
            c for c in self.df.columns
            if c.endswith("_score") and c not in ("max_score", "mean_score", "std_score")
        ]
        if not self.score_cols:
            raise ValueError("no '<name>_score' columns found")
        self.names = [c[: -len("_score")] for c in self.score_cols]
        self.k = len(self.names)


    def wobble(self) -> float:
        scores = self.df[self.score_cols].to_numpy(float)
        mad = np.median(np.abs(scores - np.median(scores, axis=0)), axis=0)
        return float(mad.mean())


    def route(self, df: pd.DataFrame, tau: float, cutoff: float) -> np.ndarray:
        scores = df[self.score_cols].to_numpy(float)
        ordered = np.sort(scores, axis=1)[:, ::-1]
        trust = (ordered[:, 0] - ordered[:, 1:].mean(axis=1)) / tau
        winners = np.array(self.names, dtype=object)[scores.argmax(axis=1)]
        return np.where(trust >= cutoff, winners, self.general)


    def calibrate(self) -> Tuple[float, float]:
        tau = self.wobble()
        y = self.df[self.label_col].to_numpy(object)
        accuracies = [float(np.mean(self.route(self.df, tau, c) == y)) for c in self.cutoffs]
        tied = [c for c, a in zip(self.cutoffs, accuracies) if a == max(accuracies)]
        return tau, tied[len(tied) // 2]