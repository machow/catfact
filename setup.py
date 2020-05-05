#!/usr/bin/env python

import re
import ast
from setuptools import setup, find_namespace_packages

_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('forcats.py', 'rb') as f:
    version = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))

with open('README.md') as f:
    README = f.read()

setup(
    name='forcats',
    version=version,
    py_modules=["forcats"],
    install_requires=['siuba', 'pandas'],
    description="Tools for working with pandas categoricals",
    author='Michael Chow',
    author_email='mc_al_github@fastmail.com',
    long_description=README,
    long_description_content_type="text/markdown",
    url='https://github.com/machow/forcats-py'
    )
