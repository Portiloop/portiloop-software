from setuptools import setup, find_packages

setup(
    name='portiloop',
    version='0.0.1',
    packages=[package for package in find_packages(where='src')],
    description='Library for portiloop',
    install_requires=['numpy',
                     'matplotlib',
                     'portilooplot']
)
