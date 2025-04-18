import numpy as np

from scipy.stats import mode


def hedley(deep_area : np.ndarray, to_correct : np.ndarray, nir : np.ndarray, nir_band : int) -> np.ndarray:
    no_nans = ~np.isnan(deep_area[nir_band])

    for idx in range(len(to_correct)):
        m = np.polyfit(deep_area[nir_band][no_nans].ravel(), deep_area[idx][no_nans].ravel(), 1)[0]
        to_correct[idx] = to_correct[idx] - m * (nir - np.nanmin(deep_area[nir_band]))
        to_correct[idx][to_correct[idx] < 0] = np.nan
    
    return to_correct


def lyzenga(deep_area : np.ndarray, to_correct : np.ndarray, nir : np.ndarray, nir_band : int) -> np.ndarray:
    no_nans = ~np.isnan(deep_area[nir_band])
    
    for idx in range(len(to_correct)):
        m = np.cov(deep_area[nir_band][no_nans].ravel(), deep_area[idx][no_nans].ravel())[0, 1]
        to_correct[idx] = to_correct[idx] - m * (nir - np.nanmean(deep_area[nir_band]))
        to_correct[idx][to_correct[idx] < 0] = np.nan
    
    return to_correct


def joyce(deep_area : np.ndarray, to_correct : np.ndarray, nir : np.ndarray, nir_band : int) -> np.ndarray:
    no_nans = ~np.isnan(deep_area[nir_band])
    
    for idx in range(len(to_correct)):
        m = np.polyfit(deep_area[nir_band][no_nans].ravel(), deep_area[idx][no_nans].ravel(), 1)[0]
        to_correct[idx] = to_correct[idx] - m * (nir - mode(deep_area[nir_band][no_nans].ravel()).mode)
        to_correct[idx][to_correct[idx] < 0] = np.nan
    
    return to_correct