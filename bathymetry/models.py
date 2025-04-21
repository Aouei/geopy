import numpy as np
import scipy

from typing import Self
from bathymetry.metrics import ValidationSummary
from selector import argcomposite



def stumpf_pseudomodel(blue : np.ndarray, other : np.ndarray, n : float = np.pi * 1_000) -> np.ndarray:
    """Stumpf pseudomodel based on https://doi.org/10.4319/lo.2003.48.1_part_2.0547

    Args:
        blue (np.ndarray): blue band data to use as base reflectance.
        other (np.ndarray): band such as green or red to compare with blue.
        n (float, optional): constant to prevent negative values. Defaults to np.pi*1_000.

    Returns:
        np.ndarray: pseudomodel.
    """

    return np.log(blue * n) / np.log(other * n)

def multi_image_pseudomodel(p_greens : np.ndarray, p_reds : np.ndarray) -> np.ndarray:
    """Multi-image method done by Isabel Caballero & Richard Stumpf on https://doi.org/10.3390/rs12030451

    Args:
        p_greens (np.ndarray): list of green pseudomodels to compose.
        p_reds (np.ndarray): list of red pseudomodels to compose.

    Returns:
        np.ndarray: pseudomodel.
    """

    return np.nanmax(p_greens, axis = 0), np.nanmax(p_reds, axis = 0), argcomposite(p_greens, np.nanargmax)


def switching_model(green_model : np.ndarray, red_model : np.ndarray, green_coef : float = 3.5, red_coef : float = 2) -> np.ndarray:
    """Linear weighted model presented on https://doi.org/10.3390/rs12030451 that combines green and red models.

    Args:
        green_model (np.ndarray): green model.
        red_model (np.ndarray): red model.
        green_coef (float, optional): min limit where start using green model. Defaults to 3.5.
        red_coef (float, optional): max limite where use red model. Defaults to 2.

    Returns:
        np.ndarray: combined model.
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


class LinearModel(object):
    """Class for apply a simple Linear regression"""

    def fit(self, pseudomodel : np.ndarray, in_situ : np.ndarray) -> Self:
        """linear regression creation.

        Args:
            pseudomodel (np.ndarray): data to use as predictor.
            in_situ (np.ndarray): data to predict.

        Returns:
            Self: fitted object.
        """

        self._set_linear_regression(pseudomodel, in_situ)
        return self

    def _set_linear_regression(self, X : np.ndarray, y : np.ndarray) -> None:
        """Method that creates the regression object.

        Args:
            X (np.ndarray): predictor data.
            y (np.ndarray): objetive data.
        """

        slope, intercept, r_value, *_ = scipy.stats.linregress(X, y)
        self.slope = slope
        self.intercept = intercept
        self.r_square = r_value ** 2

    def predict(self, pseudomodel : np.ndarray) -> np.ndarray:
        """Uses the regression created to predict new depths.

        Args:
            pseudomodel (np.ndarray): data to use for prediction.

        Returns:
            np.ndarray: predicted data.
        """

        return self.slope * pseudomodel + self.intercept
    
    def predict_and_evaluate(self, pseudomodel : np.ndarray, in_situ : np.ndarray) -> ValidationSummary:
        """Predicts new depths and creates a ValidationSummary object.

        Args:
            pseudomodel (np.ndarray): data to use as predictor.
            in_situ (np.ndarray): depths to predict.

        Returns:
            ValidationSummary: ValidationSummary
        """

        return ValidationSummary(self.predict(pseudomodel), in_situ)
    
    def __str__(self) -> str:
        return f'R: {self.r_square:.4f} | y = {self.slope:.3f}x{self.intercept:+.3f}'
    
    def __repr__(self) -> str:
        return str(self)