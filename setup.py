#!/usr/bin/python3

from distutils.core import setup

setup(name='tf_utils',
      version='1.1',
      description='Tensorflow layers and utilities for working in Python.',
      author='Jason St George',
      author_email='stgeorgejas@gmail.com',
      packages=['tf_utils', '_utils', 'losses', 'layers', 'activations', 'callbacks'],
      install_requires=['numpy',
                        'scipy',
                        'tensorflow',
                        'tensorflow_addons',
                        'tensorflow_probability']
     )