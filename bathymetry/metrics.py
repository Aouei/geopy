import numpy as np

from dataclasses import dataclass


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

    def __getitem__(self, key):
        return getattr(self, key)

    def __str__(self) -> str:
        return f"N: {self.N} | MSD: {self.MSD:.4f} | MedAE: {self.MedAE:.4f} | Abs_std: {self.Abs_std}"
    
    def __repr__(self) -> str:
        return str(self)