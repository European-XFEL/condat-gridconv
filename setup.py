from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name="condat_gridconv",
    version="0.1",
    author="European XFEL GmbH",
    author_email="da-support@xfel.eu",
    description="Convert data from hexagonal pixels to cartesian grid",
    packages=find_packages(),
    ext_modules = cythonize("condat_gridconv/shift.pyx"),
    install_requires=[
        'numpy',
    ],
)
