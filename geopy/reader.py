# from __future__ import annotations

import xarray as xr
import numpy as np
import rasterio
import pyproj
import enums
from typing import Dict

from image import Image


class ImageReader:
    """Base class for reading images from different formats.
    """

    def read(self, filename: str) -> Image:
        raise NotImplementedError("This method must be implemented by subclasses")


class NetCDFReader(ImageReader):
    """Reader for NetCDF files.
    This class reads a NetCDF file and extracts the image data, coordinates, and CRS information.
    """

    def read(self, filename: str) -> Image:
        """This method reads a NetCDF file and extracts the image data, coordinates, and CRS information.

        Args:
            filename (str): the path to the NetCDF file.

        Returns:
            Image: a custom Image object containing the data, coordinates, and CRS information.
        """

        grid_mapping = 'projection'
        
        with xr.open_dataset(filename) as src:
            crs = None
            crs_var_name = None
            
            # Buscar un DataArray con atributo crs_wkt en las coordenadas
            for coord_name, coord in src.coords.items():
                if 'crs_wkt' in coord.attrs:
                    crs = pyproj.CRS.from_wkt(coord.attrs['crs_wkt'])
                    crs_var_name = coord_name
                    break
            
            # Si no se encontró en las coordenadas, buscar en las variables
            if crs is None:
                for var_name, var in src.data_vars.items():
                    if 'crs_wkt' in var.attrs:
                        crs = pyproj.CRS.from_wkt(var.attrs['crs_wkt'])
                        crs_var_name = var_name
                        src.coords[var_name] = var
                        break
            
            # Si aún no se ha encontrado, buscar en los atributos globales
            if crs is None:
                if 'crs_wkt' in src.attrs:
                    crs = pyproj.CRS.from_wkt(src.attrs['crs_wkt'])
                elif 'proj4_string' in src.attrs:
                    crs = pyproj.CRS.from_proj4(src.attrs['proj4_string'])
            
            # Si no se encontró un nombre de variable para la proyección, usar el predeterminado
            if crs_var_name is None:
                crs_var_name = grid_mapping
                src.coords[grid_mapping] = xr.DataArray(0, attrs=crs.to_cf())
            
            # Asegurarse de que todas las variables tengan el atributo grid_mapping
            for var in src.data_vars:
                src[var].attrs['grid_mapping'] = crs_var_name

            src.attrs['grid_mapping'] = crs_var_name
            
            return Image(data=src, crs=crs)


class GeoTIFFReader(ImageReader):
    """Reader for GeoTIFF files.
    This class reads a GeoTIFF file and extracts the image data, coordinates, and CRS information.
    """

    def read(self, filename: str) -> Image:
        """This method reads a GeoTIFF file and extracts the image data, coordinates, and CRS information.

        Args:
            filename (str): The path to the GeoTIFF file.

        Returns:
            Image: a custom Image object containing the data, coordinates, and CRS information.
        """

        grid_mapping = 'projection'
        
        with rasterio.open(filename) as src:
            crs = pyproj.CRS.from_proj4(src.crs.to_proj4())
            coords = self._prepare_coords(src, crs, grid_mapping)
            variables = self._prepare_vars(src, coords, grid_mapping)
            
            # Crear atributos globales del dataset
            attrs = {}
            
            # Añadir nodata
            for i in range(1, src.count + 1):
                nodata = src.nodatavals[i-1]
                if nodata is not None:
                    attrs[f'_FillValue_band_{i}'] = nodata
            
            # Añadir otros metadatos relevantes del archivo GeoTIFF
            for key, value in src.tags().items():
                # Filtrar algunos tags que podrían causar problemas o no ser relevantes
                if key not in ['TIFFTAG_DATETIME', 'TIFFTAG_SOFTWARE']:
                    attrs[f'tiff_{key}'] = value
            
            # Añadir resumen de metadatos de bandas
            for i in range(1, src.count + 1):
                band_tags = src.tags(i)
                for tag_key, tag_value in band_tags.items():
                    attrs[f'band_{i}_{tag_key}'] = tag_value
            
            attrs['grid_mapping'] = grid_mapping
            
            # Crear el dataset con todos los atributos
            dataset = xr.Dataset(data_vars=variables, coords=coords, attrs=attrs)
            
            return Image(data=dataset, crs=crs)
    
    def _prepare_coords(self, src : rasterio.DatasetReader, crs: pyproj.CRS, grid_mapping: str) -> Dict[str, xr.DataArray]:
        """Generates the coordinates for the dataset based on the CRS and the source data.

        Args:
            src (rasterio.DatasetReader): rasterio object representing the source data.
            crs (pyproj.CRS): CRS object representing the coordinate reference system.
            grid_mapping (str): name of the projection variable.

        Returns:
            Dict[str, xr.DataArray]: Coordinates.
        """

        x_meta, y_meta = crs.cs_to_cf()
        wkt_meta = crs.to_cf()
            
        x = np.array([src.xy(row, col)[0] for row, col in zip(np.zeros(src.width), np.arange(src.width))])
        y = np.array([src.xy(row, col)[-1] for row, col in zip(np.arange(src.height), np.zeros(src.height))])
        
        coords = {
            'x': xr.DataArray(
                data=x,
                coords={'x': x},
                attrs=x_meta
            ),
            'y': xr.DataArray(
                data=y,
                coords={'y': y},
                attrs=y_meta
            ),
            grid_mapping: xr.DataArray(
                data=0,
                attrs=wkt_meta
            )
        }
        
        return coords
    
    def _prepare_vars(self, src : rasterio.DatasetReader, coords: Dict[str, xr.DataArray], grid_mapping: str) -> Dict[str, xr.DataArray]:
        """Generates the variables for the dataset based on the source data.

        Args:
            src (rasterio.DatasetReader): rasterio object representing the source data.
            crs (pyproj.CRS): CRS object representing the coordinate reference system.
            grid_mapping (str): name of the projection variable.

        Returns:
            Dict[str, xr.DataArray]: The variables or bands.
        """

        band_names = src.descriptions if not None in src.descriptions else [f'Band {i}' for i in range(1, src.count + 1)]
        
        variables = {}
        
        for idx, band_name in enumerate(band_names, start=1):
            band_data = src.read(idx)
            nodata = src.nodatavals[idx-1]
            
            # Crear atributos específicos para esta banda
            attrs = {
                'grid_mapping': grid_mapping,
                'long_name': band_name
            }
            
            # Añadir valor nodata si existe
            if nodata is not None:
                attrs['_FillValue'] = nodata
            
            # Añadir metadatos específicos de la banda
            for key, value in src.tags(idx).items():
                attrs[f'tiff_{key}'] = value
            
            variables[band_name] = xr.DataArray(
                data=band_data,
                dims=('y', 'x'),
                coords={'y': coords['y'], 'x': coords['x']},
                attrs=attrs
            )
        
        return variables


def open(filename: str) -> Image:
    """Factory function to open an image file.

    Args:
        filename (str): the path to the image file.

    Raises:
        ValueError: Error if the file format is not supported.

    Returns:
        Image: custom Image object containing the data, coordinates, and CRS information.
    """

    extension = filename.split('.')[-1].lower()
    
    if extension in enums.FILE_EXTENTIONS.TIF.value:
        return GeoTIFFReader().read(filename)
    elif extension in enums.FILE_EXTENTIONS.NETCDF.value:
        return NetCDFReader().read(filename)
    else:
        raise ValueError(f"Formato de archivo no soportado: {extension}")