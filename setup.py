from setuptools import setup, find_packages

setup(
    name='portiloop',
    version='0.0.1',
    packages=[package for package in find_packages()],
    description='Library for portiloop',
    install_requires=['wheel',
                      'EDFlib-Python',
                      'numpy',
                      'matplotlib',
                      'portilooplot',
                      'ipywidgets',
                      'python-periphery',
                      'spidev'
                     ]
)
