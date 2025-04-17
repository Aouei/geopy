import numpy as np
import scipy

from typing import Self
from bathymetry.metrics import ValidationSummary


def stumpf_pseudomodel(blue : np.ndarray, other : np.ndarray, n : float = np.pi * 1_000):
    return np.log(blue * n) / np.log(other * n)

def multi_image_pseudomodel(p_greens : np.ndarray, p_reds : np.ndarray) -> np.ndarray:
    return np.nanmax(p_greens, axis = 0), np.nanmax(p_reds, axis = 0), np.argmax(p_greens, axis = 0)


def switching_model(green_model : np.ndarray, red_model : np.ndarray, green_coef : float = 3.5, red_coef : float = 2) -> np.ndarray:
    a = (green_coef - red_model) / (green_coef - red_coef)
    b = 1 - a
    switching_model = a * red_model + b * green_model

    model = np.zeros(red_model.shape)
    model[:] = np.nan

    model = np.where(red_model < red_coef, red_model, model)
    model = np.where((red_model > red_coef) & (green_model > green_coef), green_model, model)
    model = np.where((red_model >= red_coef) & (green_model <= green_coef), switching_model, model)
    model[model < 0] = np.nan
    
    return model


class LinearModel(object):
    def fit(self, pseudomodel : np.ndarray, in_situ : np.ndarray) -> Self:
        self._set_linear_regression(pseudomodel, in_situ)
        return self

    def _set_linear_regression(self, X : np.ndarray, y : np.ndarray) -> np.ndarray:
        slope, intercept, r_value, *_ = scipy.stats.linregress(X, y)
        self.slope = slope
        self.intercept = intercept
        self.r_square = r_value ** 2

    def predict(self, pseudomodel : np.ndarray) -> np.ndarray:
        return self.slope * pseudomodel + self.intercept
    
    def predict_and_evaluate(self, pseudomodel : np.ndarray, in_situ : np.ndarray) -> ValidationSummary:
        return ValidationSummary(self.predict(pseudomodel), in_situ)
    def __str__(self) -> str:
        return f'R: {self.r_square:.4f} | y = {self.slope:.3f}x{self.intercept:+.3f}'
    
    def __repr__(self) -> str:
        return str(self)