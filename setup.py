# -*- coding: utf-8 -*-
"""
setup.py

Package install script for fm2p.


DMM, 2024-2026
"""


import setuptools


setuptools.setup(
    name = 'fm2p',
    packages = setuptools.find_packages(),
    description = 'Preprocessing and analysis for freely moving two-photon data.',
    author = 'Dylan Martins',
    version = 0.1,
)
