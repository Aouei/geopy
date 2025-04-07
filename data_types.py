from __future__ import annotations
import numpy as np
import pandas as pd

from typing import Tuple
from dataclasses import dataclass

def do_linear_regression(x : np.ndarray, y : np.ndarray) -> Tuple[float, float, float]:
    m, n = np.polyfit(x, y, 1)
    r_square = np.corrcoef(x, y)[0, 1] ** 2

    return r_square, m, n

@dataclass
class CalibrationSummary():
    pseudomodel : np.ndarray
    in_situ : np.ndarray
    lon : np.ndarray
    lat : np.ndarray
    r_square : float
    m : float
    n : float
    name : str = ''

    def predict(self, x : np.ndarray) -> np.ndarray:
        return self.m * x + self.n
    
    def __str__(self) -> str:
        return f'N: {len(self.in_situ)} | R: {self.r_square:.4f} | y = {self.m:.3f}x{self.n:+.3f} | Name {self.name}'
    
    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def load(filename : str) -> CalibrationSummary:
        df = pd.read_csv(filename)
        r_square, m, n = do_linear_regression(df.pseudomodel, df.in_situ)

        return CalibrationSummary(df.pseudomodel.values, df.in_situ.values, df.lon.values, 
                                  df.lat.values, r_square, m, n)

@dataclass
class ValidationSummary():
    model : np.ndarray
    in_situ : np.ndarray


    @property
    def error(self) -> np.ndarray:
        return self.in_situ - self.model

    @property
    def MSD(self) -> float:
        return np.nanmean(self.model - self.in_situ)

    @property
    def MAE(self) -> float:
        return round(np.nanmean(np.abs(self.error)), 5)

    @property
    def MedAE(self) -> float:
        return round(np.nanmedian(np.abs(self.error)), 5)

    @property
    def RMSE(self) -> float:
        return round(np.sqrt(np.nanmean(self.error ** 2)), 5)

    @property
    def RMedSE(self) -> float:
        return round(np.sqrt(np.nanmedian(self.error ** 2)), 5)

    @property
    def Abs_std(self) -> float:
        return round(np.nanstd(np.abs(self.error)), 5)
    
    @property
    def N(self) -> float:
        return len(self.error)
    
    def __str__(self) -> str:
        return f"N: {self.N} | MSD: {self.MSD:.4f} | MedAE: {self.MedAE:.4f} | Abs_std: {self.Abs_std}"
    
    def __repr__(self) -> str:
        return str(self)