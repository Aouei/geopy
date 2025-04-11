from __future__ import annotations

import geopandas as gpd
import rasterio.features
import xarray as xr
import numpy as np
import rasterio
import selector
import pyproj
import enums


from rasterio.warp import reproject, Resampling, calculate_default_transform
from shapely.geometry.base import BaseGeometry
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box
from geopandas import GeoSeries
from itertools import pairwise
from typing import Tuple, List
from affine import Affine
from copy import deepcopy



class Image(object):
    grid_mapping : str = 'projection'

    def __init__(self, data: xr.Dataset, crs: pyproj.CRS) -> None:
        self.crs: pyproj.CRS = crs
        self.data: xr.Dataset = data
        self.name: str = ''

    def replace(self, old : str, new : str) -> None:
        new_names = {
            var: var.replace(old, new) for var in self.data.data_vars if old in var
        }

        self.data = self.data.rename(new_names)

    def rename(self, new_names):
        self.data = self.data.rename(new_names)

    def rename_by_enum(self, enum : enums.Enum) -> None:
        for band in enum:
            for wavelenght in band.value:
                self.replace(wavelenght, band.name)

    @property
    def band_names(self) -> List[str]:
        return list(self.data.data_vars.keys())
    
    @property
    def width(self) -> int:
        return len(self.data.x)
    
    @property
    def height(self) -> int:
        return len(self.data.y)
    
    @property
    def count(self) -> int:
        return len(self.data.data_vars)

    @property
    def x_res(self) -> float | int:
        return float(abs(self.data.x[0] - self.data.x[1]))

    @property
    def y_res(self) -> float | int:
        return float(abs(self.data.y[0] - self.data.y[1]))
    
    @property
    def transform(self) -> Affine:
        return from_origin(self.left, self.top, self.x_res, self.y_res)

    @property
    def xs_ys(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.data.x, self.data.y)

    @property
    def left(self) -> float:
        return float(self.data.x.min()) - abs(self.x_res / 2
)
    @property
    def right(self) -> float:
        return float(self.data.x.max()) + abs(self.x_res / 2
)
    @property
    def top(self) -> float:
        return float(self.data.y.max()) + abs(self.y_res / 2)

    @property
    def bottom(self) -> float:
        return float(self.data.y.min()) - abs(self.y_res / 2)

    @property
    def bbox(self) -> Polygon:
        return box(self.left, self.bottom, self.right, self.top)


    @property
    def values(self) -> np.ndarray:
        return np.array( [self.data[band].values.copy() for band in self.band_names] )


    def reproject(self, new_crs: pyproj.CRS, interpolation : Resampling = Resampling.nearest) -> None:        
        # Obtener la información del CRS actual y el nuevo
        src_crs = self.crs
        dst_crs = new_crs
        
        # Obtener las dimensiones actuales
        src_height, src_width = self.height, self.width
        
        # Calcular la transformación para la reproyección
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, src_width, src_height,
            left=float(self.data.x.min()), bottom=float(self.data.y.min()),
            right=float(self.data.x.max()), top=float(self.data.y.max())
        )
        
        dst_x, _ = rasterio.transform.xy(dst_transform, np.zeros(dst_width), np.arange(dst_width))
        _, dst_y = rasterio.transform.xy(dst_transform, np.arange(dst_height), np.zeros(dst_height))

        # Preparar los metadatos para las nuevas coordenadas
        self.crs = dst_crs
        x_meta, y_meta = self.crs.cs_to_cf()
        
        if x_meta['standard_name'] == 'latitude':
            x_meta, y_meta = y_meta, x_meta
        
        wkt_meta = self.crs.to_cf()
        
        # Crear las nuevas coordenadas para el dataset
        coords = {
            'x': xr.DataArray(
                data=dst_x,
                coords={'x': dst_x},
                attrs=x_meta
            ),
            'y': xr.DataArray(
                data=dst_y,
                coords={'y': dst_y},
                attrs=y_meta
            ),
            self.grid_mapping: xr.DataArray(
                data=0,
                attrs=wkt_meta
            )
        }
        
        # Crear un nuevo dataset con las variables reproyectadas
        new_data_vars = {}
        
        for var_name, var_data in self.data.data_vars.items():
            # Preparar el array de destino
            dst_array = np.zeros((dst_height, dst_width), dtype=np.float32)
            dst_array[:] = np.nan

            # Realizar la reproyección
            dst_array, _ = reproject(
                source=var_data.values,
                destination=dst_array,
                src_transform=self.transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata = np.nan,
                resampling=interpolation
            )
            
            # Agregar la variable reproyectada al nuevo dataset
            new_data_vars[var_name] = xr.DataArray(
                data=dst_array,
                dims=('y', 'x'),
                coords={'y': coords['y'], 'x': coords['x']},
                attrs={'grid_mapping': self.grid_mapping}
            )
        
        # Actualizar el dataset con las nuevas coordenadas y datos reproyectados
        self.data = xr.Dataset(
            data_vars=new_data_vars,
            coords=coords,
            attrs = self.data.attrs
        )

    def align(self, reference: Image, interpolation: Resampling = Resampling.nearest) -> None:
        """
        Alinea esta imagen con respecto a una imagen de referencia.
        Adopta el CRS, resolución y extensión de la imagen de referencia.
        
        Args:
            reference: Imagen que se usará como referencia para la alineación.
            interpolation: Método de remuestreo a utilizar durante la alineación.
        """
        # Verificar si los CRS son diferentes y reproyectar si es necesario
        if self.crs != reference.crs:
            self.reproject(reference.crs, interpolation)
        
        # Obtener la transformación y dimensiones de la referencia
        dst_transform = reference.transform
        dst_width, dst_height = reference.width, reference.height
        
        # Crear un nuevo dataset con las variables alineadas
        new_data_vars = {}
        
        for var_name, var_data in self.data.data_vars.items():
            # Preparar el array de destino
            dst_array = np.zeros((dst_height, dst_width), dtype=np.float32)
            dst_array[:] = np.nan
            
            # Realizar la alineación (reproyección con el mismo CRS)
            dst_array, _ = reproject(
                source=var_data.values,
                destination=dst_array,
                src_transform=self.transform,
                src_crs=self.crs,
                dst_transform=dst_transform,
                dst_crs=reference.crs,
                dst_nodata=np.nan,
                resampling=interpolation
            )
            
            # Agregar la variable alineada al nuevo dataset
            new_data_vars[var_name] = xr.DataArray(
                data=dst_array,
                dims=('y', 'x'),
                coords={'y': reference.data.y, 'x': reference.data.x},
                attrs={'grid_mapping': self.grid_mapping}
            )
        
        # Actualizar el dataset con las nuevas coordenadas y datos alineados
        self.data = xr.Dataset(
            data_vars=new_data_vars,
            coords={
                'x': reference.data.x.copy(),
                'y': reference.data.y.copy(),
                self.grid_mapping: xr.DataArray(
                    data=0,
                    attrs=reference.crs.to_cf()
                )
            },
            attrs=self.data.attrs
        )
        
        # Actualizar el CRS y la transformación para que coincidan con la referencia
        self.crs = reference.crs

    def clip(self, geometries : List[BaseGeometry]):
        inshape = rasterio.features.geometry_mask(geometries = geometries, out_shape = (self.height, self.width), 
                                                  transform = self.transform, invert = True)
            
        rows, cols = self.__find_empty_borders(inshape)
        self.data = self.data.isel({'y' : rows, 'x' : cols})

    def mask(self, condition : np.ndarray, bands : str | List[str] = None):     
        if bands is not None:
            self.data[bands] = self.data[bands].where( xr.DataArray(data = condition, dims = ('y', 'x')) )
        else:
            self.data = self.data.where( xr.DataArray(data = condition, dims = ('y', 'x')) )

    def geometry_mask(self, geometries : List[BaseGeometry], mask_out : bool = True,  bands : str | List[str] = None):
        condition = rasterio.features.geometry_mask(geometries = geometries, out_shape = (self.height, self.width), 
                                                    transform = self.transform, invert = mask_out)
            
        self.mask(condition, bands)


    def select(self, bands : str | List[str], only_values : bool = True) -> np.ndarray | xr.DataArray:
        result = None

        if only_values:
            if isinstance(bands, list):
                result = np.array([self.data[band].values.copy() for band in bands])
            else:
                result = self.data[bands].values.copy()
        else:
            result = deepcopy(self.data[bands])
    
        return result
    
    def add_band(self, band_name : str, data : np.ndarray | xr.DataArray):
        if isinstance(data, np.ndarray):
            self.data[band_name] = (('y', 'x'), data)
        else:
            self.data[band_name] = data

    def drop_bands(self, bands):
        self.data = self.data.drop_vars(bands)


    def dropna(self):
        mask = np.zeros((self.height, self.width))
        for data in self.data.data_vars.values():
            mask = np.logical_or(mask, ~np.isnan(data.values))
            
        rows, cols = self.__find_empty_borders(mask)
        self.data = self.data.isel({'y' : rows, 'x' : cols})

    def __find_empty_borders(self, array : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rows = np.where(array.any(axis=1))[0]
        rows = np.arange(rows.min(), rows.max() + 1)

        cols = np.where(array.any(axis=0))[0]
        cols = np.arange(cols.min(), cols.max() + 1)

        return rows, cols

    def extract_values(self, xs : np.ndarray, ys : np.ndarray, bands : List[str] = None, is_1D : bool = False) -> np.ndarray:
        bands = self.band_names if bands is None else bands
        filtered = self.data.sel({'x' : xs, 'y' : ys}, method = 'nearest')
        values = np.array( [ filtered[band].values.copy() for band in bands ] )

        if xs.ndim == ys.ndim and is_1D:
            values = values[:, np.arange(len(xs)), np.arange(len(xs))]
            
        return values

    def interval_choice(self, band, size, intervals, replace : bool = True, add_indexes = False):
        if not isinstance(band, str):
            raise ValueError('band argument must a string')


        array = self.select(band).ravel()        
        selected_indexes = selector.arginterval_choice(array, size, intervals, replace)
        selected_values = array[selected_indexes]

        if add_indexes:
            return selected_values, selected_indexes
        else:
            return selected_values


    def empty_like(self) -> Image:
        result = Image(deepcopy(self.data), deepcopy(self.crs))
        result.drop_bands(result.band_names)
        return result
    
    def _repr_html_(self) -> str:
        return self.data._repr_html_()

    def to_netcdf(self, filename):
        self.data.attrs['proj4_string'] = self.crs.to_proj4()
        self.data.attrs['crs_wkt'] = self.crs.to_wkt()
        
        return self.data.to_netcdf(filename)
    
    def to_tif(self, filename):
        height, width = self.height, self.width
        count = self.count
        
        # Prepare the metadata for rasterio
        meta = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': count,
            'dtype': next(iter(self.data.data_vars.values())).dtype,
            'crs': self.crs,
            'transform': self.transform
        }
        
        with rasterio.open(filename, 'w', **meta) as dst:
            # Write each band
            for idx, (band_name, band_data) in enumerate(self.data.data_vars.items(), start=1):
                dst.write(band_data.values, idx)
                dst.set_band_description(idx, band_name)