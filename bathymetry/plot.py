import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Self, Tuple
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn


from .metrics import ValidationSummary
from .models import LinearModel


class CalibrationPlot:

    def __init__(self, nrows : int, ncols : int, figsize : Tuple[int, int], **kwargs):
        self.fig, self.axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
        self.legend_font_size = 20
        self.label_font_size = 20
        self.tick_font_size = 15
        self.title_font_size = 30
        self.font_family = 'Times New Roman'
        
    def add_calibration_scatter(self, model : LinearModel, x : np.ndarray, y : np.ndarray, ax, c : str = 'g', **kwargs) -> Self:
        ax.tick_params(axis='both', which='major', labelsize = self.tick_font_size)
        ax.tick_params(axis='both', which='minor', labelsize = self.tick_font_size)

        x_range = np.linspace(np.nanmin(x), np.nanmax(x), 100).reshape(-1, 1)
        y_pred = model.predict(x_range)

        ax.plot(x_range, y_pred, '--k', label = f'$R² = {model.r_square:.4f}$')
        ax.scatter(x, y, label = f'$y = {model.slope:.4f}x {model.intercept:+.4f}$', c = c, **kwargs)
        ax.scatter(x[0], y[0], label = f'$n = {x.size}$', alpha = 0)
        ax.grid()
        return self

    def add_legend(self, ax) -> Self:
        legend = ax.legend(loc = 'upper left', handlelength = 0, handletextpad = 0, 
                           prop={'size' : self.legend_font_size, 'family' : self.font_family})
        for item in legend.legend_handles:
            item.set_visible(False)
        return self
    
    def add_labels(self, ax, title : str = '', xlabel : str = '', ylabel : str = '') -> Self:
        ax.set_title(title, fontdict={'size' : self.title_font_size, 'family' : self.font_family})
        ax.set_xlabel(xlabel, fontdict={'size' : self.label_font_size, 'family' : self.font_family})
        ax.set_ylabel(ylabel, fontdict={'size' : self.label_font_size, 'family' : self.font_family})

        return self


def plot_validation(summary : ValidationSummary, scatter_ax, residuals_ax, density_method : None):
    if density_method == DENSITY_METHODS.AUTO:
        __density_scatter(scatter_ax, summary)
    elif density_method == DENSITY_METHODS.PRECISE:
        pass
    elif density_method == DENSITY_METHODS.APPROXIMATE:
        pass

    sns.histplot(x = summary.error.flatten(), kde = True, ax = residuals_ax)
    max_w = 5
    residuals_ax.set_xlim(-max_w, max_w)
    max_h = 100
    residuals_ax.set_ylim(0, max_h)

    residuals_ax.tick_params(axis='both', which='major')
    residuals_ax.tick_params(axis='both', which='minor')

    h_offset = 10
    residuals_ax.set_xlabel("Residual error (m)")
    residuals_ax.set_ylabel("")
    residuals_ax.set_title("Error")
    residuals_ax.text(-max_w + 2, max_h - h_offset * 1, f'MedAE = {summary.MedAE:.3f} (m)')
    residuals_ax.text(-max_w + 2, max_h - h_offset * 2, f'Abs_std = {summary.Abs_std:.3f} (m)')
    residuals_ax.text(-max_w + 2, max_h - h_offset * 3, f'MSD = {summary.MSD:.3f} (m)')
    residuals_ax.text(-max_w + 2, max_h - h_offset * 4, f'N = {summary.N}')


def __density_scatter(ax, summary : ValidationSummary , sort : bool = True, bins : int = 20, cmap = 'viridis_r', **kwargs):
    label_font_size = 20
    tick_font_size = 15
    title_font_size = 30
    font_family = 'Times New Roman'

    
    x = summary.in_situ
    y = summary.model

    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0
    z[z < 0] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    mappable = ax.scatter(x, y, c = z, s = 5, cmap = cmap, norm = norm)

    min_x_lim, max_x_lim = kwargs.get('min_x', 0), kwargs.get('max_x', 7)
    min_y_lim, max_y_lim = kwargs.get('min_y', 0), kwargs.get('max_y', 7)
    ax.set_xlim(min_x_lim, max_x_lim)
    ax.set_ylim(min_y_lim, max_y_lim)

    ax.plot([min_x_lim, max_x_lim], [min_y_lim, max_y_lim], 'r-', alpha = 0.75, zorder = 0)
    ax.set_xlabel(kwargs.get('xlabel', 'Depth (m)'), fontdict={'size' : label_font_size, 'family' : font_family})
    ax.set_ylabel(kwargs.get('ylabel', 'pSDB Green'), fontdict={'size' : label_font_size, 'family' : font_family})
    ax.set_title(kwargs.get('title', 'pSDB Green vs Prof'), fontdict={'size' : title_font_size, 'family' : font_family})

    return ax, mappable