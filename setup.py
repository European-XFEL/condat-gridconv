from setuptools import setup
from Cython.Build import cythonize

with open("README.md") as f:
    readme = f.read()

setup(
    name="condat_gridconv",
    version="0.1",
    author="European XFEL GmbH",
    author_email="da-support@xfel.eu",
    description="Convert data from hexagonal pixels to cartesian grid",
    long_description=readme,
    long_description_content_type='text/markdown',
    url="https://github.com/European-XFEL/condat-gridconv",
    package_dir={"": "src"},
    packages=["condat_gridconv"],
    ext_modules = cythonize("src/condat_gridconv/shift.pyx"),
    install_requires=[
        'numpy',
    ],
    python_requires='>=3.7',
)
