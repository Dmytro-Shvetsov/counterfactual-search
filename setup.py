import io
import os
from setuptools import find_packages, setup

NAME = 'src'
REQUIRES_PYTHON = '>=3.0.0'
VERSION = '0.0.0'

here = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as fid:
        REQUIRED = [r for r in fid if not r.startswith('git')]
except:
    REQUIRED = []

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as fid:
        long_description = '\n' + fid.read()
except FileNotFoundError:
    long_description = ''


setup(
    name=NAME,
    version=VERSION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude=('data', 'configs', 'lightning_logs')),
    install_requires=REQUIRED,
    include_package_data=True,
)