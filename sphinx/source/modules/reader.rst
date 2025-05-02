Reader Module
===============

The Reader module provides functionality for reading geospatial data from different file formats.

.. currentmodule:: sensingpy.reader

Classes and Functions
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   
   ImageReader
   GeoTIFFReader
   NetCDFReader
   open

ImageReader Class
----------

.. autoclass:: ImageReader
   :members:
   :undoc-members:
   :show-inheritance:
   
   .. rubric:: Methods
   
   .. autosummary::
      :nosignatures:
      
      ~ImageReader.read

GeoTIFFReader Class
----------

.. autoclass:: GeoTIFFReader
   :members:
   :undoc-members:
   :show-inheritance:
   
   .. rubric:: Methods
   
   .. autosummary::
      :nosignatures:
      
      ~GeoTIFFReader.read
      ~GeoTIFFReader._prepare_coords
      ~GeoTIFFReader._prepare_vars

NetCDFReader Class
----------

.. autoclass:: NetCDFReader
   :members:
   :undoc-members:
   :show-inheritance:
   
   .. rubric:: Methods
   
   .. autosummary::
      :nosignatures:
      
      ~NetCDFReader.read

Module Functions
--------------

.. autofunction:: open