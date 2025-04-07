from typing import Tuple
import numpy as np
import scipy.linalg
import scipy.stats
from data_types import CalibrationSummary, ValidationSummary
import scipy

def stumpf_pseudomodel(blue : np.ndarray, other : np.ndarray, n : float = np.pi * 1_000):
    return np.log(blue * n) / np.log(other * n)

def get_stratified_depths(depths : np.ndarray, n_segments : int = 10, sample_size_per_depth : int = 5, min_depth = None, max_depth = None, random = None) -> np.ndarray:
    min_depth = np.nanmin(depths) if min_depth is None else min_depth
    max_depth = np.nanmax(depths) if max_depth is None else max_depth
    
    segments = np.linspace(min_depth, max_depth, n_segments)
    indexes = np.arange(depths.size)

    selected_indexes = []
    for idx, depth in enumerate(segments[:-1]):
        min_depth = depth
        max_depth = segments[idx + 1]

        to_select = (min_depth <= depths) & (depths <= max_depth)

        if random is None:
            selected_indexes.extend(np.random.choice(indexes[to_select], sample_size_per_depth, 
                                                 replace = False))
        else:
            selected_indexes.extend(random.choice(indexes[to_select], sample_size_per_depth, 
                                                 replace = False))

    return np.array(selected_indexes)

def calibrate(pseudomodel : np.ndarray, in_situ : np.ndarray, lon : np.ndarray, lat : np.ndarray) -> CalibrationSummary:
    slope, intercept, r_value, *_ = scipy.stats.linregress(pseudomodel, in_situ)
    return CalibrationSummary(pseudomodel, in_situ, lon, lat, r_value ** 2, slope, intercept)


def validate(model : np.ndarray, in_situ : np.ndarray) -> ValidationSummary:
    return ValidationSummary(model, in_situ)
    

def composite(green_pseudomodels : np.ndarray, red_pseudomodels : np.ndarray) -> np.ndarray:
    return np.nanmax(green_pseudomodels, axis = 0), np.nanmax(red_pseudomodels, axis = 0), \
            np.argmax(green_pseudomodels, axis = 0)


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