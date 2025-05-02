import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Any, Tuple, Mapping
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
from scipy.stats import gaussian_kde
from matplotlib.axes import Axes


from sensingpy.bathymetry.metrics import ValidationSummary
from sensingpy.bathymetry.models import LinearModel


class CalibrationPlot(object):
    """
    Class for creating standardized bathymetry calibration plots.
    
    This class provides methods for plotting calibration data for satellite-derived
    bathymetry models, including regression lines, scatterplots, and statistics.
    It handles consistent styling and formatting across plots.
    
    Parameters
    ----------
    title_font_size : int, optional
        Font size for plot titles, by default 30
    label_font_size : int, optional
        Font size for axis labels, by default 20
    tick_font_size : int, optional
        Font size for axis ticks, by default 15
    legend_font_size : int, optional
        Font size for legend text, by default 20
    font_family : str, optional
        Font family for all text elements, by default 'Times New Roman'
    """

    def __init__(self, title_font_size : int = 30, label_font_size : int = 20, tick_font_size : int = 15, 
                 legend_font_size : int = 20, font_family : str = 'Times New Roman'):
        """Constructor with the size of the texts"""

        self.legend_font_size = legend_font_size
        self.label_font_size = label_font_size
        self.tick_font_size = tick_font_size
        self.title_font_size = title_font_size
        self.font_family = font_family
        
    def add_calibration_scatter(self, model: LinearModel, x: np.ndarray, y: np.ndarray, ax: Axes, c: str = 'g', **kwargs) -> Axes:
        """
        Plot regression data with model fit line and statistics.
        
        Parameters
        ----------
        model : LinearModel
            Calibrated linear model with slope, intercept and R² values
        x : np.ndarray
            x-axis values (predictor variables)
        y : np.ndarray
            y-axis values (observed depths)
        ax : Axes
            Matplotlib axes on which to plot
        c : str, optional
            Color for scatter points, by default 'g'
        **kwargs
            Additional keyword arguments passed to scatter()
            
        Returns
        -------
        Axes
            Matplotlib axes with plotted data
            
        Notes
        -----
        The plot includes the best-fit line, R² value, linear equation,
        and sample size in the legend.
        """

        ax.tick_params(axis='both', which='major', labelsize = self.tick_font_size)
        ax.tick_params(axis='both', which='minor', labelsize = self.tick_font_size)

        x_range = np.linspace(np.nanmin(x), np.nanmax(x), 100).reshape(-1, 1)
        y_pred = model.predict(x_range)

        ax.plot(x_range, y_pred, '--k', label = f'$R² = {model.r_square:.4f}$')
        ax.scatter(x, y, label = f'$y = {model.slope:.4f}x {model.intercept:+.4f}$', c = c, **kwargs)
        ax.scatter(x[0], y[0], label = f'$n = {x.size}$', alpha = 0)
        ax.grid()

        return ax

    def add_legend(self, ax: Axes) -> Axes:
        """
        Add formatted legend to axes.
        
        Parameters
        ----------
        ax : Axes
            Matplotlib axes to which legend will be added
            
        Returns
        -------
        Axes
            Matplotlib axes with legend
            
        Notes
        -----
        The legend is formatted according to the font settings specified
        during class initialization, with handles hidden.
        """

        legend = ax.legend(loc = 'upper left', handlelength = 0, handletextpad = 0, 
                           prop={'size' : self.legend_font_size, 'family' : self.font_family})
        for item in legend.legend_handles:
            item.set_visible(False)

        return ax
    
    def add_labels(self, ax: Axes, title: str = None, xlabel: str = None, ylabel: str = None) -> Axes:
        """
        Add formatted title and axis labels to plot.
        
        Parameters
        ----------
        ax : Axes
            Matplotlib axes to which labels will be added
        title : str, optional
            Plot title, by default None
        xlabel : str, optional
            x-axis label, by default None
        ylabel : str, optional
            y-axis label, by default None
            
        Returns
        -------
        Axes
            Matplotlib axes with labels
            
        Notes
        -----
        Labels are formatted according to the font settings specified
        during class initialization.
        """
        
        if title is not None:
            ax.set_title(title, fontdict={'size' : self.title_font_size, 'family' : self.font_family})
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontdict={'size' : self.label_font_size, 'family' : self.font_family})
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontdict={'size' : self.label_font_size, 'family' : self.font_family})

        return ax


class ValidationPlot(object):
    """
    Class for creating standardized bathymetry validation plots.
    
    This class provides methods for visualizing and evaluating the performance
    of satellite-derived bathymetry models, including density scatter plots
    and error histograms with statistical metrics.
    
    Parameters
    ----------
    title_font_size : int, optional
        Font size for plot titles, by default 30
    label_font_size : int, optional
        Font size for axis labels, by default 20
    tick_font_size : int, optional
        Font size for axis ticks, by default 15
    legend_font_size : int, optional
        Font size for legend text, by default 20
    font_family : str, optional
        Font family for all text elements, by default 'Times New Roman'
    """

    def __init__(self, title_font_size : int = 30, label_font_size : int = 20, tick_font_size : int = 15, 
                 legend_font_size : int = 20, font_family : str = 'Times New Roman'):
        """Constructor with the size of the texts"""

        self.legend_font_size = legend_font_size
        self.label_font_size = label_font_size
        self.tick_font_size = tick_font_size
        self.title_font_size = title_font_size
        self.font_family = font_family

    def add_densed_scatter(self, summary: ValidationSummary, ax: Axes, s: float = 5, 
                          cmap: str = 'viridis_r', vmin: float = None, vmax: float = None, 
                          density: Mapping[str, Any] = None) -> Tuple[Axes, Any]:
        """
        Create a density-colored scatter plot comparing modeled vs. in-situ depths.
        
        Parameters
        ----------
        summary : ValidationSummary
            Validation summary object containing model predictions and ground truth
        ax : Axes
            Matplotlib axes on which to plot
        s : float, optional
            Point size for scatter plot, by default 5
        cmap : str, optional
            Matplotlib colormap for density representation, by default 'viridis_r'
        vmin : float, optional
            Minimum value for colormap normalization, by default None
        vmax : float, optional
            Maximum value for colormap normalization, by default None
        density : Mapping[str, Any], optional
            Settings for density calculation, by default None.
            Format: {'method': 'precise'|'approximate', 'bins': int}
            
        Returns
        -------
        Tuple[Axes, Any]
            Matplotlib axes with plot and the colorbar mappable object
            
        Notes
        -----
        Point color represents data density, helping visualize the distribution
        of points in crowded scatter plots. A 1:1 line is added to show perfect
        agreement between model and observations.
        """

        x, y, z, norm = self.__select_density_method(summary, density)
        
        mappable = ax.scatter(x, y, c = z, s = s, cmap = cmap, vmin = vmin, vmax = vmax, norm = norm)
        ax.plot([x.min(), x.max()], [x.min(), x.max()], '--k', alpha = 0.75, zorder = 9)
        self.add_labels(ax, title = 'SDB vs In Situ', xlabel = 'In Situ (m)', ylabel = 'SDB (m)')

        colorbar = plt.colorbar(mappable, ax = ax)
        colorbar.ax.tick_params(axis = 'both', which = 'major', labelsize = self.tick_font_size)
        colorbar.ax.tick_params(axis = 'both', which = 'minor', labelsize = self.tick_font_size)

        ax.tick_params(axis='both', which='major', labelsize = self.tick_font_size)
        ax.tick_params(axis='both', which='minor', labelsize = self.tick_font_size)

        return ax, colorbar

    def __select_density_method(self, summary: ValidationSummary, density: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        """
        Select and apply appropriate density calculation method.
        
        Parameters
        ----------
        summary : ValidationSummary
            Validation summary object containing model predictions and ground truth
        density : Mapping[str, Any]
            Settings for density calculation
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, Any]
            x-values, y-values, density values, and colormap normalization
            
        Raises
        ------
        ValueError
            If an unknown density method is specified
            
        Notes
        -----
        Two density calculation methods are supported:
        - 'precise': Uses Gaussian KDE for accurate but slower density estimation
        - 'approximate': Uses 2D histogram and interpolation for faster but less accurate results
        """

        if density is None:
            x, y, z, norm = self.__get_precise_density(summary.in_situ, summary.model)
        else:
            if density['method'] == 'precise':
                x, y, z, norm = self.__get_precise_density(summary.in_situ, summary.model)
            elif density['method'] == 'approximate':
                x, y, z, norm = self.__get_approximate_density(summary.in_situ, summary.model, bins = density.get('bins', 10))
            else:
                raise ValueError(f"Unknown density method: {density['method']}")
            
        return x,y,z,norm
    
    def add_residuals(self, summary: ValidationSummary, ax: Axes, x_lim: int = 5, 
                     metrics: List[str] = None, **hist_kwargs) -> Axes:
        """
        Create a histogram of model residuals with statistical metrics.
        
        Parameters
        ----------
        summary : ValidationSummary
            Validation summary object containing error statistics
        ax : Axes
            Matplotlib axes on which to plot
        x_lim : int, optional
            Symmetric x-axis limit for histogram, by default 5
        metrics : List[str], optional
            List of metric names to display in legend, by default None
        **hist_kwargs
            Additional keyword arguments passed to seaborn's histplot()
            
        Returns
        -------
        Axes
            Matplotlib axes with plotted histogram
            
        Notes
        -----
        The histogram shows the distribution of residual errors (in-situ minus 
        model predictions), with a kernel density estimate overlay. Statistical
        metrics are shown in a text box on the plot.
        """

        if metrics is None:
            metrics = []

        ax = sns.histplot(summary.error, ax = ax, kde = True, color = 'skyblue', edgecolor = 'black', **hist_kwargs)
        ax.set_xlim(-x_lim, x_lim)

        ax.tick_params(axis='both', which='major', labelsize = self.tick_font_size)
        ax.tick_params(axis='both', which='minor', labelsize = self.tick_font_size)

        self.add_labels(ax, title = 'In Situ - SDB',  xlabel = 'Residual error (m)')

        legend = '\n'.join( [f'N = {summary.N}'] + [ f'{metric} = {summary[metric]:.3f} (m)' for metric in metrics ] )
        background = { 'facecolor':'white', 'alpha':0.3, 'boxstyle':'round,pad=0.3' }
        ax.text(0.1, 0.95, legend, fontsize = self.legend_font_size, transform = ax.transAxes, 
                verticalalignment = 'top', bbox = background)

        return ax
    
    def add_labels(self, ax: Axes, title: str = None, xlabel: str = None, ylabel: str = None) -> Axes:
        """
        Add formatted title and axis labels to plot.
        
        Parameters
        ----------
        ax : Axes
            Matplotlib axes to which labels will be added
        title : str, optional
            Plot title, by default None
        xlabel : str, optional
            x-axis label, by default None
        ylabel : str, optional
            y-axis label, by default None
            
        Returns
        -------
        Axes
            Matplotlib axes with labels
            
        Notes
        -----
        Labels are formatted according to the font settings specified
        during class initialization.
        """

        if title is not None:
            ax.set_title(title, fontdict={'size' : self.title_font_size, 'family' : self.font_family})
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontdict={'size' : self.label_font_size, 'family' : self.font_family})
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontdict={'size' : self.label_font_size, 'family' : self.font_family})

        return ax

    def __get_precise_density(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        """
        Calculate accurate point density using Gaussian kernel density estimation.
        
        Parameters
        ----------
        X : np.ndarray
            x-coordinates (in-situ depth values)
        y : np.ndarray
            y-coordinates (modeled depth values)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, Any]
            x-values, y-values, density values, and colormap normalization
            
        Notes
        -----
        This method uses scipy's gaussian_kde for accurate kernel density
        estimation. It provides smooth density estimates but can be computationally
        intensive for large datasets.
        """

        xy = np.vstack([X, y])
        density = gaussian_kde(xy)(xy)

        idx = density.argsort()
        X, y, density = X[idx], y[idx], density[idx]
        norm = Normalize(vmin = np.min(density), vmax = np.max(density))

        return X, y, density, norm


    def __get_approximate_density(self, X: np.ndarray, y: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        """
        Calculate approximate point density using 2D histogram and interpolation.
        
        Parameters
        ----------
        X : np.ndarray
            x-coordinates (in-situ depth values)
        y : np.ndarray
            y-coordinates (modeled depth values)
        bins : int
            Number of bins for the 2D histogram
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, Any]
            x-values, y-values, density values, and colormap normalization
            
        Notes
        -----
        This method uses numpy's histogram2d and scipy's interpn for faster but
        less accurate density estimation. It's more suitable for large datasets
        where performance is a concern.
        """

        data , x_e, y_e = np.histogram2d(X, y, bins = bins, density = True )
        density = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([X, y]).T , method = "splinef2d", bounds_error = False)

        density[np.where(np.isnan(density))] = 0.0
        density[density < 0] = 0.0
        
        idx = density.argsort()
        X, y, density = X[idx], y[idx], density[idx]

        norm = Normalize(vmin = np.min(density), vmax = np.max(density))

        return X, y, density, norm