# -*- coding: utf-8 -*-
"""
fm2p/utils/LNP_helpers.py

Helper functions for the linear-nonlinear-Poisson (LNP) model.

Functions
---------
rough_penalty
    Roughness penalty (value, gradient, Hessian) for a parameter vector.
find_param
    Slice a concatenated parameter vector into per-model-component arrays.
make_varmap
    One-hot encoding of a variable trace relative to a set of bin centres.


DMM, March 2025
"""


import numpy as np
from scipy import sparse


def rough_penalty(param, beta, circ=False):
    """ Compute roughness penalty for a parameter.

    Parameters
    ----------
    param : array_like
        Parameter values to compute the roughness penalty for.
    beta : float
        Weight of the roughness penalty.
    circ : bool, optional
        Whether the parameter is circular. Default is False.

    Returns
    -------
    J : float
        Roughness penalty value.
    J_g : array_like
        Gradient of the roughness penalty.
    J_h : array_like
        Hessian of the roughness penalty.
    """

    n = np.size(param)

    D = sparse.spdiags(
        (np.ones([n,1]) @ np.array([-1,1])[np.newaxis,:]).T,
        (0,1),
        (n-1,n)
    ).toarray()

    DD = D.T @ D

    if circ is True:

        DD[0, :] = np.roll(DD[1, :], -1)
        DD[-1, :] = np.roll(DD[-2, :], 1)

    param1 = param[np.newaxis,:]

    J = 0.5 * beta * param1 @ DD @ param1.T
    J_g = beta * DD @ param1.T
    J_h = beta * DD

    return float(J), J_g, J_h


def find_param(param, modelType, numA, numB, numC, numD):
    # more flexible approach

    # Map model names to their parameter counts
    counts = {'A': numA, 'B': numB, 'C': numC, 'D': numD}
    
    # Initialize all results to empty arrays
    results = {k: np.array([]) for k in ['A', 'B', 'C', 'D']}
    
    current_idx = 0
    
    # Iterate through the canonical order A -> B -> C -> D
    for model in ['A', 'B', 'C', 'D']:
        if model in modelType:
            # How many params does this specific model need?
            count = counts[model]
            
            # Slice the param array
            # If it's the last one, take everything remaining to be safe
            end_idx = current_idx + count
            results[model] = param[current_idx : end_idx]
            
            # Move the index pointer
            current_idx += count

    # If the last slice didn't capture the very end (due to rounding or logic), 
    # the last active model in the loop typically grabs `param[current_idx:]` 
    # in the hardcoded version. The slicing above handles explicit counts.
            
    return results['A'], results['B'], results['C'], results['D']


def find_param(param, modelType, numA, numB, numC, numD):
    """ Find the parameters for the model type.

    Given a model type (e.g., 'AB'), find the parameters for
    that model by indexing over the empty parameter values in
    the parameter array.

    Parameters
    ----------
    param : array_like
        Array of parameters.
    modelType : str
        Model type string. Must contain only the character A, B, C, and/or D.
    numA : int
        Number of parameters for model A.
    numB : int
        Number of parameters for model B.
    numC : int
        Number of parameters for model C.
    numD : int
        Number of parameters for model D.
    """

    # Initialize empty arrays
    pA = np.array([])
    pB = np.array([])
    pC = np.array([])
    pD = np.array([])

    # --- Single Models ---
    if modelType == 'A':      # 1
        pA = param
    elif modelType == 'B':    # 2
        pB = param
    elif modelType == 'C':    # 3
        pC = param
    elif modelType == 'D':    # 4
        pD = param

    # --- Double Models ---
    elif modelType == 'AB':   # 5
        pA = param[:numA]
        pB = param[numA:]
    elif modelType == 'AC':   # 6
        pA = param[:numA]
        pC = param[numA:]
    elif modelType == 'AD':   # 7
        pA = param[:numA]
        pD = param[numA:]
    elif modelType == 'BC':   # 8
        pB = param[:numB]
        pC = param[numB:]
    elif modelType == 'BD':   # 9
        pB = param[:numB]
        pD = param[numB:]
    elif modelType == 'CD':   # 10
        pC = param[:numC]
        pD = param[numC:]
    
    # --- Triple Models ---
    elif modelType == 'ABC':  # 11
        pA = param[:numA]
        pB = param[numA : numA+numB]
        pC = param[numA+numB :]
    elif modelType == 'ABD':  # 12
        pA = param[:numA]
        pB = param[numA : numA+numB]
        pD = param[numA+numB :]
    elif modelType == 'ACD':  # 13
        pA = param[:numA]
        pC = param[numA : numA+numC]
        pD = param[numA+numC :]
    elif modelType == 'BCD':  # 14
        pB = param[:numB]
        pC = param[numB : numB+numC]
        pD = param[numB+numC :]
    
    # --- Quad Model ---
    elif modelType == 'ABCD': # 15
        pA = param[:numA]
        pB = param[numA : numA+numB]
        pC = param[numA+numB : numA+numB+numC]
        pD = param[numA+numB+numC :]

    return pA, pB, pC, pD


def make_varmap(var, bin_cents, circ=False):
    """ Make a one-hot encoding of variable values relative to bins.

    Parameters
    ----------
    var : array_like
        Array of variable values to encode.
    bin_cents : array_like
        Array of bin centers for the encoding.
    circ : bool, optional
        Whether the variable is circular. Default is False.
    
    Returns
    -------
    varmap : array_like
        One-hot encoding of the variable values with two dimensions:
        the first is the timepoint in var (matches the length of var),
        and the second is the bin (matches the length of bin_cents).

    Example output
    --------------
    var = [1, 1.5, 3, 3.5, 5]
    bin_cents = [1, 3, 5]

    varmap = [[1, 0, 0],
              [1, 0, 0],
              [0, 1, 0],
              [0, 1, 0],
              [0, 0, 1]]
    
    """

    varmap = np.zeros([len(var), len(bin_cents)])

    for i in range(len(var)):

        # Index of the bin that is closest to the variable value
        b_ind = np.argmin(np.abs(var[i] - bin_cents))

        # If this is a circular variable and the value is at the edge of the bins
        if (circ is True) and ((b_ind==0) or (b_ind==bin_cents[-1])):

            # if at an edge bin, make sure it's not closer to the other edge
            # for circular variables
            if np.abs(var[i] - bin_cents[0]) < np.abs(var[i] - bin_cents[-1]):
                varmap[i, 0] = 1
            else:
                varmap[i, -1] = 1

        # Set the one-hot encoding   
        else:
            varmap[i, b_ind] = 1

    return varmap


