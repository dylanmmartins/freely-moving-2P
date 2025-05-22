# -*- coding: utf-8 -*-
"""
Linear algebra utilities for matrix operations.

Functions
---------
make_U_triangular(size)
    Create an upper-triangular matrix of the given size.
make_L_triangular(size)
    Create a lower-triangular matrix of the given size.

Author: DMM, 2025
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

