import numpy as np

from scipy.stats import mode


def hedley(deep_area_mask : np.ndarray, to_correct : np.ndarray, nir : np.ndarray) -> np.ndarray:
    """
    Hedley method for deglinting.
    
    This method corrects sun glint in visible bands using a linear regression 
    between NIR and visible bands over optically deep water areas. It assumes a 
    linear relationship between sun glint in NIR and visible bands.
    
    Parameters
    ----------
    deep_area_mask : np.ndarray
        Boolean mask identifying optically deep water areas to use for correction
    to_correct : np.ndarray
        Array of bands to correct for sun glint
    nir : np.ndarray
        Near-infrared band values used as the glint predictor
        
    Returns
    -------
    np.ndarray
        Array of bands with sun glint correction applied
    
    Notes
    -----
    The algorithm uses the slope of the linear regression between each visible 
    band and the NIR band over deep water to determine the correction factor.
    Values < 0 after correction are set to NaN.
    
    References
    ----------
    Hedley, J. D., Harborne, A. R., & Mumby, P. J. (2005). Simple and robust 
    removal of sun glint for mapping shallow-water benthos. International Journal 
    of Remote Sensing, 26(10), 2107-2112.
    https://doi.org/10.1080/01431160500034086
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
    """
    Lyzenga method for deglinting.
    
    This method corrects sun glint in visible bands using covariance between 
    NIR and visible bands over optically deep water areas. It is based on the 
    statistical relationship between NIR brightness and glint intensity.
    
    Parameters
    ----------
    deep_area_mask : np.ndarray
        Boolean mask identifying optically deep water areas to use for correction
    to_correct : np.ndarray
        Array of bands to correct for sun glint
    nir : np.ndarray
        Near-infrared band values used as the glint predictor
        
    Returns
    -------
    np.ndarray
        Array of bands with sun glint correction applied
    
    Notes
    -----
    This implementation uses the covariance between each visible band and 
    the NIR band over deep water to determine the correction factor. The method 
    uses mean NIR rather than minimum NIR as the reference point. Values < 0
    after correction are set to NaN.
    
    References
    ----------
    Lyzenga, D. R., Malinas, N. P., & Tanis, F. J. (2006). Multispectral 
    bathymetry using a simple physically based algorithm. IEEE Transactions on 
    Geoscience and Remote Sensing, 44(8), 2251-2259.
    https://doi.org/10.1109/TGRS.2006.872909
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
    """
    Joyce method for deglinting.
    
    This method corrects sun glint by using the statistical mode of NIR values 
    in deep water as the reference point. It combines aspects of both Hedley and
    Hochberg approaches to glint removal.
    
    Parameters
    ----------
    deep_area_mask : np.ndarray
        Boolean mask identifying optically deep water areas to use for correction
    to_correct : np.ndarray
        Array of bands to correct for sun glint
    nir : np.ndarray
        Near-infrared band values used as the glint predictor
        
    Returns
    -------
    np.ndarray
        Array of bands with sun glint correction applied
    
    Notes
    -----
    The algorithm uses the slope of the linear regression between each visible 
    band and the NIR band over deep water, similar to Hedley's method. However,
    it uses the mode of NIR values as the reference point rather than the minimum.
    Values < 0 after correction are set to NaN.
    
    References
    ----------
    Joyce, K. E. (2004). A method for mapping live coral cover using remote sensing.
    PhD Thesis, University of Queensland, Brisbane, Australia.
    
    Kay, S., Hedley, J. D., & Lavender, S. (2009). Sun glint correction of high 
    and low spatial resolution images of aquatic scenes: a review of methods for 
    visible and near-infrared wavelengths. Remote Sensing, 1(4), 697-730.
    https://doi.org/10.3390/rs1040697
    """

    for idx in range(len(to_correct)):
        deep_value = to_correct[idx][deep_area_mask]
        deep_nir = nir[deep_area_mask]
        is_valid = ~np.isnan(deep_value) & ~np.isnan(deep_nir)

        m = np.polyfit(deep_nir[is_valid].ravel(), deep_value[is_valid].ravel(), 1)[0]
        to_correct[idx] = to_correct[idx] - m * (nir - mode(deep_nir[is_valid].ravel()).mode)
        to_correct[idx][to_correct[idx] < 0] = np.nan
    
    return to_correct