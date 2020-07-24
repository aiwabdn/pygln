import os
from setuptools import find_packages, setup

# Extract install_requires from requirements.txt
install_requires = list()
directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'requirements.txt'), 'r') as filehandle:
    for line in filehandle:
        line = line.strip()
        if line:
            install_requires.append(line)

setup(
    name='PyGLN',
    version='0.1.0',
    description=
    'PyGLN: Gated Linear Networks implementations for NumPy, PyTorch, TensorFlow and JAX',
    long_description=
    'PyGLN: Gated Linear Networks implementations for NumPy, PyTorch, TensorFlow and JAX',
    long_description_content_type='text/markdown',
    author='Anindya Basu, Alexander Kuhnle',
    author_email='aiwabdn@gmail.com, alexkuhnle@t-online.de',
    url='https://github.com/aiwabdn/pygln',
    packages=find_packages(exclude=('test', )),
    license='GNU GPLv3',
    python_requires='>=3.5',
    install_requires=install_requires)
