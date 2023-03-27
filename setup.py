from setuptools import setup, find_packages
import io


def is_coral():
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
            string = m.read().lower()
            return any([x in string for x in ["phanbell", "coral"]])
    except Exception: pass
    return False

requirements_list = ['wheel',
                     'pyEDFLib',
                     'numpy',
                     'portilooplot',
                     'ipywidgets',
                     'python-periphery',
                     'scipy',
                     'matplotlib']

if is_coral():
    requirements_list += ['pycoral',
                          'spidev',
                          'pylsl-coral',
                          'pyalsaaudio']
else:
    requirements_list += ['gradio',
                          'tensorflow',
                          'pyxdf',
                          'wonambi']


setup(
    name='portiloop',
    version='0.0.1',
    packages=[package for package in find_packages()],
    description='Portiloop software library',
    install_requires=requirements_list,
)
