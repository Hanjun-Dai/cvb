from setuptools import setup

import os
BASEPATH = os.path.dirname(os.path.abspath(__file__))

setup(name='cvb',
      py_modules=['cvb'],
      install_requires=[
          'torch'
      ],
)
