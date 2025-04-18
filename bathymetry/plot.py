import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Self, Tuple, List
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn


from .metrics import ValidationSummary
from .models import LinearModel


class CalibrationPlot(object):

    def __init__(self, title_font_size : int = 30, label_font_size : int = 20, tick_font_size : int = 15, 
                 legend_font_size : int = 20, font_family : str = 'Times New Roman'):
        self.legend_font_size = legend_font_size
        self.label_font_size = label_font_size
        self.tick_font_size = tick_font_size
        self.title_font_size = title_font_size
        self.font_family = font_family
        
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
    
    def add_labels(self, ax, title : str = None, xlabel : str = None, ylabel : str = None) -> Self:
        if title is not None:
            ax.set_title(title, fontdict={'size' : self.title_font_size, 'family' : self.font_family})
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontdict={'size' : self.label_font_size, 'family' : self.font_family})
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontdict={'size' : self.label_font_size, 'family' : self.font_family})

        return self


class ValidationPlot(object):
    def __init__(self, title_font_size : int = 30, label_font_size : int = 20, tick_font_size : int = 15, 
                 legend_font_size : int = 20, font_family : str = 'Times New Roman'):
        self.legend_font_size = legend_font_size
        self.label_font_size = label_font_size
        self.tick_font_size = tick_font_size
        self.title_font_size = title_font_size
        self.font_family = font_family

    def add_densed_scatter(self, summary : ValidationSummary, ax, s = 5, cmap = 'viridis_r', vmin = None, vmax = None, density = None):
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

    def __select_density_method(self, summary, density):
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
    
    def add_residuals(self, summary : ValidationSummary, ax, x_lim : int = 5, metrics : List[str] = None, **hist_kwargs):
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
    
    def add_labels(self, ax, title : str = None, xlabel : str = None, ylabel : str = None) -> Self:
        if title is not None:
            ax.set_title(title, fontdict={'size' : self.title_font_size, 'family' : self.font_family})
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontdict={'size' : self.label_font_size, 'family' : self.font_family})
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontdict={'size' : self.label_font_size, 'family' : self.font_family})

        return self

    def __get_precise_density(self, X, y):
        xy = np.vstack([X, y])
        density = gaussian_kde(xy)(xy)

        idx = density.argsort()
        X, y, density = X[idx], y[idx], density[idx]
        norm = Normalize(vmin = np.min(density), vmax = np.max(density))

        return X, y, density, norm


    def __get_approximate_density(self, X, y, bins):
        data , x_e, y_e = np.histogram2d(X, y, bins = bins, density = True )
        density = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([X, y]).T , method = "splinef2d", bounds_error = False)

        density[np.where(np.isnan(density))] = 0.0
        density[density < 0] = 0.0
        
        idx = density.argsort()
        X, y, density = X[idx], y[idx], density[idx]

        norm = Normalize(vmin = np.min(density), vmax = np.max(density))
        return X, y, density, norm