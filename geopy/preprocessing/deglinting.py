import numpy as np

from scipy.stats import mode


def hedley(deep_area_mask : np.ndarray, to_correct : np.ndarray, nir : np.ndarray) -> np.ndarray:
    """Hedley method for deglinting.

    Args:
        deep_area (np.ndarray): Area to use for correction
        to_correct (np.ndarray): bands to correct
        nir (np.ndarray): nir band values

    Returns:
        np.ndarray: to_correct with deglinting applied
    """

    for idx in range(len(to_correct)):
        deep_value = to_correct[idx][deep_area_mask]
        deep_nir = nir[deep_area_mask]
        is_valid = ~np.isnan(deep_value) & ~np.isnan(deep_nir)

        m = np.polyfit(deep_nir[is_valid].ravel(), deep_value[is_valid].ravel(), 1)[0]
        to_correct[idx] = to_correct[idx] - m * (nir - np.nanmin(deep_nir[is_valid]))
        to_correct[idx][to_correct[idx] < 0] = np.nan
    
    return to_correct


def lyzenga(deep_area_mask : np.ndarray, to_correct : np.ndarray, nir : np.ndarray) -> np.ndarray:
    """Lyzenga method for deglinting.

    Args:
        deep_area_mask (np.ndarray): Area to use for correction
        to_correct (np.ndarray): bands to correct
        nir (np.ndarray): nir band values

    Returns:
        np.ndarray: to_correct with deglinting applied
    """

    for idx in range(len(to_correct)):
        deep_value = to_correct[idx][deep_area_mask]
        deep_nir = nir[deep_area_mask]
        is_valid = ~np.isnan(deep_value) & ~np.isnan(deep_nir)

        m = np.cov(deep_nir[is_valid].ravel(), deep_value[is_valid].ravel())[0, 1]
        to_correct[idx] = to_correct[idx] - m * (nir - np.nanmean(deep_nir[is_valid]))
        to_correct[idx][to_correct[idx] < 0] = np.nan
    
    return to_correct


def joyce(deep_area_mask : np.ndarray, to_correct : np.ndarray, nir : np.ndarray) -> np.ndarray:
    """Joyce method for deglinting.

    Args:
        deep_area_mask (np.ndarray): Area to use for correction
        to_correct (np.ndarray): bands to correct
        nir (np.ndarray): nir band values

    Returns:
        np.ndarray: to_correct with deglinting applied
    """

    for idx in range(len(to_correct)):
        deep_value = to_correct[idx][deep_area_mask]
        deep_nir = nir[deep_area_mask]
        is_valid = ~np.isnan(deep_value) & ~np.isnan(deep_nir)

        m = np.polyfit(deep_nir[is_valid].ravel(), deep_value[is_valid].ravel(), 1)[0]
        to_correct[idx] = to_correct[idx] - m * (nir - mode(deep_nir[is_valid].ravel()).mode)
        to_correct[idx][to_correct[idx] < 0] = np.nan
    
    return to_correct