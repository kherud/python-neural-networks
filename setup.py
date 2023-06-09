#!/usr/bin/env python3

from setuptools import setup
import unittest


setup(name='GigaNN',
      version='1.0',
      description='GIGA Neural Network library',
      author='Konstantin Herud, Filipp Roos, Julian Zarges',
      url='https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/lectures/machine-learning/student-material/ws19/team12/code',
      packages=['gigann', 'gigann.layer'],
      test_suite='test'
     )
