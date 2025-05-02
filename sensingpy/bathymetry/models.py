import numpy as np
import scipy

from typing import Self
from sensingpy.bathymetry.metrics import ValidationSummary
from sensingpy.selector import argcomposite



def stumpf_pseudomodel(blue : np.ndarray, other : np.ndarray, n : float = np.pi * 1_000) -> np.ndarray:
    """
    Calculate Stumpf pseudomodel for satellite-derived bathymetry.
    
    This function implements the ratio transform algorithm developed by Stumpf et al. (2003)
    for estimating water depth from multispectral imagery. The algorithm uses the ratio
    of logarithms of reflectance in blue and another band.
    
    Parameters
    ----------
    blue : np.ndarray
        Blue band data to use as base reflectance
    other : np.ndarray
        Band such as green or red to compare with blue
    n : float, optional
        Constant to prevent negative values in logarithm calculation,
        by default np.pi*1_000
        
    Returns
    -------
    np.ndarray
        Pseudomodel values representing linearized depth estimates
    
    Notes
    -----
    The algorithm leverages the differential attenuation of light with depth
    between blue and other wavelengths.
    
    References
    ----------
    Stumpf, R.P., Holderied, K., Sinclair, M. (2003). Determination of water depth
    with high-resolution satellite imagery over variable bottom types.
    Limnology and Oceanography, 48(1), 547-556.
    https://doi.org/10.4319/lo.2003.48.1_part_2.0547
    """

    return np.log(blue * n) / np.log(other * n)

def multi_image_pseudomodel(p_greens : np.ndarray, p_reds : np.ndarray) -> np.ndarray:
    """
    Apply multi-image composition method for improved bathymetry estimates.
    
    This function implements the multi-image compositing method developed by
    Caballero & Stumpf (2020) for optimizing satellite-derived bathymetry from
    multiple scenes. It creates composite bathymetry models by selecting the
    maximum values from multiple pseudomodels.
    
    Parameters
    ----------
    p_greens : np.ndarray
        List of green-band pseudomodels to compose
    p_reds : np.ndarray
        List of red-band pseudomodels to compose
        
    Returns
    -------
    tuple
        Tuple containing:
        - np.ndarray: Maximum green pseudomodel values
        - np.ndarray: Maximum red pseudomodel values
        - np.ndarray: Index array identifying which image was selected at each pixel
    
    Notes
    -----
    The multi-image approach helps overcome limitations from individual images
    such as sun glint, clouds, or varying water quality conditions.
    
    References
    ----------
    Caballero, I., & Stumpf, R. P. (2020). Towards routine mapping of shallow
    bathymetry in environments with variable turbidity: Contribution of
    Sentinel-2A/B satellites mission. Remote Sensing, 12(3), 451.
    https://doi.org/10.3390/rs12030451
    """

    return np.nanmax(p_greens, axis = 0), np.nanmax(p_reds, axis = 0), argcomposite(p_greens, np.nanargmax)


def switching_model(green_model : np.ndarray, red_model : np.ndarray, green_coef : float = 3.5, red_coef : float = 2) -> np.ndarray:
    """
    Create a depth model by combining green and red models using a weighted approach.
    
    This function implements the linear weighted model presented by Caballero & Stumpf (2020)
    that combines green and red models based on depth thresholds. It uses only the red model
    for shallow areas, only the green model for deeper areas, and a weighted combination
    for intermediate depths.
    
    Parameters
    ----------
    green_model : np.ndarray
        Green band-based bathymetry model, generally better for deeper waters
    red_model : np.ndarray
        Red band-based bathymetry model, generally better for shallow waters
    green_coef : float, optional
        Minimum threshold value where the green model starts to be used exclusively,
        by default 3.5
    red_coef : float, optional
        Maximum threshold value where the red model is used exclusively,
        by default 2
        
    Returns
    -------
    np.ndarray
        Combined bathymetry model with optimized depth estimates
    
    Notes
    -----
    The switching approach leverages the strengths of different spectral bands:
    red bands perform better in shallow waters while green bands perform better
    in deeper waters. This method provides a smooth transition between the models.
    
    References
    ----------
    Caballero, I., & Stumpf, R. P. (2020). Towards routine mapping of shallow
    bathymetry in environments with variable turbidity: Contribution of
    Sentinel-2A/B satellites mission. Remote Sensing, 12(3), 451.
    https://doi.org/10.3390/rs12030451
    """

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


def optical_deep_water_model(model: np.ndarray, blue: np.ndarray, green: np.ndarray, vnir: np.ndarray) -> np.ndarray:
    """
    Filter depth estimations based on optical properties of water.
    
    This function applies optical water property-based filters to depth estimation
    models to improve accuracy. It filters out pixels based on reflectance thresholds
    for clear waters and an upper depth limit equation for turbid waters as described
    in peer-reviewed literature.
    
    Parameters
    ----------
    model : np.ndarray
        The initial depth estimation model output to be filtered
    blue : np.ndarray
        Blue band reflectance values, used for clear water filtering
    green : np.ndarray
        Green band reflectance values, used for clear water filtering
    vnir : np.ndarray
        Near-infrared band reflectance values, used for turbid water depth limit calculation
        
    Returns
    -------
    np.ndarray
        Filtered depth model with invalid estimations set to NaN
    
    Notes
    -----
    The function applies two filtering steps:
    1. Clear water filtering: removes pixels with reflectance <= 0.003 in blue or green bands
    2. Turbid water filtering: applies depth limitation based on NIR reflectance using the equation:
       Ymax = -0.251 * ln(NIR) + 0.8
    
    References
    ----------
    Caballero, I., & Stumpf, R. P. (2019). Retrieval of nearshore bathymetry from Sentinel-2A and 2B 
    satellites in South Florida coastal waters. Estuarine, Coastal and Shelf Science, 226, 106277.
    https://doi.org/10.1016/j.ecss.2019.106277
    """
    
    ## Clear waters
    model[blue <= 0.003] = np.nan
    model[green <= 0.003] = np.nan
    
    ## Turbid waters
    Ymax = (-0.251 * np.log(vnir)) + 0.8
    Ymax[Ymax < 0] = np.nan

    y = np.log(model)
    y[y < 0] = np.nan

    # Remove values exceeding depth limit from 2019 paper
    model[y > Ymax] = np.nan
    
    return model


class LinearModel(object):
    """
    Linear regression model for satellite-derived bathymetry.
    
    This class implements a simple linear regression approach for converting
    bathymetric pseudomodels to actual depth values. It provides methods for
    fitting the model to known depth measurements and predicting depths from
    new pseudomodel values.
    
    Parameters
    ----------
    None
        
    Attributes
    ----------
    slope : float
        Slope coefficient of the linear model
    intercept : float
        Y-intercept of the linear model
    r_square : float
        Coefficient of determination (R²) indicating goodness of fit
    
    Notes
    -----
    Linear models are commonly used in satellite-derived bathymetry to establish
    the relationship between optical properties and actual water depths. This
    implementation uses scipy's linregress for the underlying calculations.
    """

    def fit(self, pseudomodel : np.ndarray, in_situ : np.ndarray) -> Self:
        """
        Fit linear regression model using pseudomodel and in-situ depth data.
        
        Parameters
        ----------
        pseudomodel : np.ndarray
            Predictor values (typically from ratio transform algorithms)
        in_situ : np.ndarray
            Target values (measured water depths)
            
        Returns
        -------
        Self
            Returns the instance for method chaining
            
        Notes
        -----
        This method calculates the linear relationship between pseudomodel values
        and actual water depths using ordinary least squares regression.
        """

        self._set_linear_regression(pseudomodel, in_situ)
        return self

    def _set_linear_regression(self, X : np.ndarray, y : np.ndarray) -> None:
        """
        Calculate and store linear regression parameters.
        
        Parameters
        ----------
        X : np.ndarray
            Predictor data (pseudomodel values)
        y : np.ndarray
            Target data (measured water depths)
            
        Notes
        -----
        This internal method sets the slope, intercept, and R² attributes
        based on the input data.
        """

        slope, intercept, r_value, *_ = scipy.stats.linregress(X, y)
        self.slope = slope
        self.intercept = intercept
        self.r_square = r_value ** 2

    def predict(self, pseudomodel : np.ndarray) -> np.ndarray:
        """
        Predict depths using the fitted linear model.
        
        Parameters
        ----------
        pseudomodel : np.ndarray
            Predictor values to convert to depth estimates
            
        Returns
        -------
        np.ndarray
            Predicted depth values
            
        Notes
        -----
        Applies the linear transformation using stored slope and intercept values.
        """

        return self.slope * pseudomodel + self.intercept
    
    def predict_and_evaluate(self, pseudomodel : np.ndarray, in_situ : np.ndarray) -> ValidationSummary:
        """
        Predict depths and evaluate model performance against in-situ measurements.
        
        Parameters
        ----------
        pseudomodel : np.ndarray
            Predictor values to convert to depth estimates
        in_situ : np.ndarray
            Reference depth values for validation
            
        Returns
        -------
        ValidationSummary
            Object containing various error metrics and validation statistics
            
        Notes
        -----
        This convenience method combines prediction and validation in a single step.
        """

        return ValidationSummary(self.predict(pseudomodel), in_situ)
    
    def __str__(self) -> str:
        """
        Return string representation of the model.
        
        Returns
        -------
        str
            Formatted string with R² value and linear equation
        """
        return f'R: {self.r_square:.4f} | y = {self.slope:.3f}x{self.intercept:+.3f}'
    
    def __repr__(self) -> str:
        """
        Return official string representation of the model.
        
        Returns
        -------
        str
            Same as __str__ output
        """
        return str(self)