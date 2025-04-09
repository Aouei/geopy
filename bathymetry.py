import numpy as np
import scipy

from data_types import CalibrationSummary, ValidationSummary


def stumpf_pseudomodel(blue : np.ndarray, other : np.ndarray, n : float = np.pi * 1_000):
    return np.log(blue * n) / np.log(other * n)

def multi_image(p_Greens : np.ndarray, p_Reds : np.ndarray) -> np.ndarray:
    return np.nanmax(p_Greens, axis = 0), np.nanmax(p_Reds, axis = 0), \
            np.argmax(p_Greens, axis = 0)


def calibrate(p_model : np.ndarray, in_situ : np.ndarray, lon : np.ndarray, lat : np.ndarray) -> CalibrationSummary:
    slope, intercept, r_value, *_ = scipy.stats.linregress(p_model, in_situ)
    return CalibrationSummary(p_model, in_situ, lon, lat, r_value ** 2, slope, intercept)

def validate(model : np.ndarray, in_situ : np.ndarray) -> ValidationSummary:
    return ValidationSummary(model, in_situ)


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