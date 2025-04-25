from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

requirements = [
    'accessible-pygments==0.0.5','affine==2.4.0','alabaster==1.0.0','asttokens==3.0.0','attrs==25.3.0','babel==2.17.0',
    'beautifulsoup4==4.13.4','bleach==6.2.0','Cartopy==0.24.1','certifi==2025.1.31','cftime==1.6.4.post1','charset-normalizer==3.4.1',
    'click==8.1.8','click-plugins==1.1.1','cligj==0.7.2','cmocean==4.0.3','colorama==0.4.6','comm==0.2.2','contextily==1.6.2',
    'contourpy==1.3.1','coverage==7.8.0','cycler==0.12.1','debugpy==1.8.13','decorator==5.2.1','defusedxml==0.7.1','docutils==0.21.2',
    'executing==2.2.0','fastjsonschema==2.21.1','fonttools==4.57.0','geographiclib==2.0','geopandas==1.0.1','geopy==2.4.1','h5netcdf==1.6.1',
    'h5py==3.13.0','idna==3.10','imagesize==1.4.1','iniconfig==2.1.0','ipykernel==6.29.5','ipython==9.0.2','ipython_pygments_lexers==1.1.1',
    'jedi==0.19.2','Jinja2==3.1.6','joblib==1.4.2','jsonschema==4.23.0','jsonschema-specifications==2025.4.1','jupyter_client==8.6.3',
    'jupyter_core==5.7.2','jupyterlab_pygments==0.3.0','kiwisolver==1.4.8','MarkupSafe==3.0.2','matplotlib==3.10.1','matplotlib-inline==0.1.7',
    'mercantile==1.2.1','mistune==3.1.3','nbclient==0.10.2','nbconvert==7.16.6','nbformat==5.10.4','nbsphinx==0.9.7','nest-asyncio==1.6.0',
    'netCDF4==1.7.2','numpy==2.2.4','packaging==24.2','pandas==2.2.3','pandocfilters==1.5.1','parso==0.8.4','pillow==11.1.0','platformdirs==4.3.7',
    'pluggy==1.5.0','prompt_toolkit==3.0.50','psutil==7.0.0','pure_eval==0.2.3','pydata-sphinx-theme==0.15.4','Pygments==2.19.1','pyogrio==0.10.0',
    'pyparsing==3.2.3','pyproj==3.7.1','pyshp==2.3.1','pytest==8.3.5','pytest-cov==6.1.1','python-dateutil==2.9.0.post0','pytz==2025.2',
    'pywin32==310','pyzmq==26.4.0','rasterio==1.4.3','referencing==0.36.2','requests==2.32.3','roman-numerals-py==3.1.0','rpds-py==0.24.0',
    'scikit-learn==1.6.1','scipy==1.15.2','seaborn==0.13.2','shapely==2.1.0','six==1.17.0','snowballstemmer==2.2.0','soupsieve==2.7','Sphinx',
    'sphinx-book-theme==1.1.4','sphinx-rtd-theme==3.0.2','sphinxcontrib-applehelp==2.0.0','sphinxcontrib-devhelp==2.0.0','sphinxcontrib-htmlhelp==2.1.0',
    'sphinxcontrib-jquery==4.1','sphinxcontrib-jsmath==1.0.1','sphinxcontrib-qthelp==2.0.0','sphinxcontrib-serializinghtml==2.0.0','stack-data==0.6.3',
    'threadpoolctl==3.6.0','tinycss2==1.4.0','tornado==6.4.2','traitlets==5.14.3','typing_extensions==4.13.1','tzdata==2025.2','urllib3==2.3.0',
    'wcwidth==0.2.13','webencodings==0.5.1','xarray==2025.3.1','xyzservices==2025.1.0'
]

setup(
    name="sensingpy",
    version="1.0.0",
    author="Sergio Heredia",
    author_email="sergiohercar1@gmail.com",
    description="A package for geospatial image processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aouei/geopy",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
)