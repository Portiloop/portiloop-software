from setuptools import setup, find_packages

setup(
    name='portiloop',
    version='0.0.1',
    packages=[package for package in find_packages()],
    description='Portiloop software library',
    install_requires=['wheel',
                      'EDFlib-Python',
                      'numpy',
                      'portilooplot',
                      'ipywidgets',
                      'python-periphery',
                      'scipy',
                      'matplotlib',
                     ],
    extras_require={
        'Portiloop': ['pycoral', 
                'spidev', 
                'pylsl-coral', 
                'pyalsaaudio'],
        'PC': ['gradio',
               'tensorflow',
               'pyxdf',
               'wonambi']
    },
)
