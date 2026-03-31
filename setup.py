from setuptools import setup, find_packages
import io


VERSION = '0.1.3'  # Note: don't forget to update this in the top __init__.py


def is_coral():
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
            line = m.read().lower()
            if 'phanbell' in line or "coral" in line: return True
    except Exception: pass
    return False

requirements_list = ['wheel',
                     'numpy',
                     'portilooplot',
                     'ipywidgets',
                     'python-periphery',
                     'scipy',
                     'matplotlib']

if is_coral():
    requirements_list += ['spidev',
                          'pylsl-coral',
                          'pyalsaaudio==0.9.2',
                          'python-socketio==5.9',
                          'nicegui']
else:
    requirements_list += ['gradio',
                          'tensorflow',
                          'pyxdf',
                          'wonambi']


setup(
    name='portiloop',
    version=VERSION,
    packages=[package for package in find_packages()],
    description='Portiloop software library',
    install_requires=requirements_list,
)
