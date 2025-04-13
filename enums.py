from enum import Enum
from typing import List


class SENTINEL2_BANDS(Enum):
    B1 : List[str] =  ['443', '442', '444']
    B2 : List[str] =  ['493', '492', '489']
    B3 : List[str] =  ['560', '559', '561']
    B4 : List[str] =  ['665', '665', '667']
    B5 : List[str] =  ['704', '704', '707']
    B6 : List[str] =  ['740', '739', '741']
    B7 : List[str] =  ['783', '780', '785']
    B8 : List[str] =  ['833', '833', '835']
    B8A : List[str] = ['865', '864', '866']
    B9 : List[str] =  ['945', '943', '947']
    B10 : List[str] = ['1373', '1377', '1372']
    B11 : List[str] = ['1614', '1610', '1612']
    B12 : List[str] = ['2202', '2186', '2191']

class MICASENSE_BANDS(Enum):
    BLUE : List[str] = ['1']
    GREEN : List[str] = ['2']
    RED : List[str] = ['3']
    NIR : List[str] = ['4']
    RED_EDGE : List[str] = ['5']


class FILE_EXTENTIONS(Enum):
    TIF = ['tiff', 'tif']
    NETCDF = ['nc']