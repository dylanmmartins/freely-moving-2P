# -*- coding: utf-8 -*-
"""
fm2p/utils/linalg.py

Linear algebra utilities for triangular matrix construction.

Functions
---------
make_U_triangular
    Upper-triangular binary matrix of given size.
make_L_triangular
    Lower-triangular binary matrix of given size.


DMM, May 2025
"""


import numpy as np


def make_U_triangular(size):
    """ Create an upper-triangular matrix.

    Parameters
    ----------
    size : int
        The size of the matrix to create.
    
    Returns
    -------
    np.ndarray
        An upper-triangular matrix of the given size.
    """

    matrix = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(i, size):
            matrix[i, j] = 1
    return matrix


def make_L_triangular(size):
    """ Create a lower-triangular matrix.

    Parameters
    ----------
    size : int
        The size of the matrix to create.

    Returns
    -------
    np.ndarray
        A lower-triangular matrix of the given size.
    """

    matrix = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(i, size):
            matrix[j, i] = 1
    return matrix

