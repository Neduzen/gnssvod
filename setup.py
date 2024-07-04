"""
GNSSVOD setup file based on

gnssvod setup file by
Mustafa Serkan ISIK and Volkan Ozbey
"""
from setuptools import setup
import re

def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), open(project + '/__init__.py').read())
    return result.group(1)

setup(
  name = 'gnssvod',
  install_requires=[
    "pandas",
    "numpy",
    "matplotlib",
    "plotly"
    "pyunpack",
    "hatanaka",
    "tqdm",
    "xarray",
    "netcdf4",
    "dask",
    "scipy"
  ],
  include_package_data = True,
  package_data = {"gnssvod.doc": ["IGSList.txt"]},
  data_files = [("", ["LICENSE"])],
  version = get_property('__version__', 'gnssvod'),
  description = 'Python Toolkit for GNSS Data',
  author = get_property('__author__', 'gnssvod'),
  author_email = 'vincent.humphrey@geo.uzh.ch',
  license = '',
  url = 'https://github.com/gnssvod-Project/gnssvod',
  download_url = 'https://github.com/gnssvod-Project/gnssvod/archive/0.1.tar.gz',
  classifiers = [],
  zip_safe=False
)   

