from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from GateCalibrator import GateCalibrator


class GatePredictor:
    def __init__(self, 
                 val_path: str, 
                 data_path: str, 
                 cutoffs: Sequence[float]) -> None:
        self.calibrator = GateCalibrator(val_path, cutoffs)
        self.tau, self.cutoff = self.calibrator.calibrate()
        self.data = pd.read_csv(data_path)


    def predict(self) -> np.ndarray:
        return self.calibrator.route(self.data, self.tau, self.cutoff)


    def accuracy(self) -> Tuple[float, float]:
        """Returns (general vs specialist accuracy, full routing accuracy)."""
        general = self.calibrator.general
        y = self.data[self.calibrator.label_col].to_numpy(object)
        pred = self.predict()
        gate_accuracy = float(np.mean((pred == general) == (y == general)))
        real_accuracy = float(np.mean(pred == y))
        return gate_accuracy, real_accuracy