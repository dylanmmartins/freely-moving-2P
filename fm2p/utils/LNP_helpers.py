# -*- coding: utf-8 -*-
"""
Helper functions for the linear-nonlinear-Poisson (LNP) model.

Functions
---------
rough_penalty(param, beta, circ=False)
    Compute roughness penalty for a parameter.
find_param(param, modelType, numP, numR, numE)
    Find the parameters for the model type.
make_varmap(var, bin_cents, circ=False)
    Make a one-hot encoding of variable values relative to bins.

Author: DMM, 2024
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



def find_param(param, modelType, numP, numR, numE):
    """ Find the parameters for the model type.

    Given a model type (e.g., 'PR'), find the parameters for
    that model by indexing over the empty parameter values in
    the parameter array.

    Parameters
    ----------
    param : array_like
        Array of parameters.
    modelType : str
        Model type string. Must contain only the character P, R, E, and/or D.
    numP : int
        Number of parameters for the pupil position model.
    numR : int
        Number of parameters for the retinotopic model.
    numE : int
        Number of parameters for the egocentric model.
    numD : int
        Number of parameters for the distance model.
    """

    # Number of parameters
    pP = np.array([])
    pR = np.array([])
    pE = np.array([])
    pD = np.array([])

    if modelType == 'P':      # 1
        pP = param
    elif modelType == 'R':    # 2
        pR = param
    elif modelType == 'E':    # 3
        pE = param
    elif modelType == 'D':    # 4
        pD = param

    elif modelType == 'PR':   # 5
        pP = param[:numP]
        pR = param[numP:]
    elif modelType == 'PE':   # 6
        pP = param[:numP]
        pE = param[numP:]
    elif modelType == 'PD':   # 7
        pP = param[:numP]
        pD = param[numP:]
    elif modelType == 'RE':   # 8
        pR = param[:numR]
        pE = param[numR:]
    elif modelType == 'RD':   # 9
        pR = param[:numR]
        pD = param[numR:]
    elif modelType == 'ED':   # 10
        pE = param[:numE]
        pD = param[numE:]
    
    elif modelType == 'PRE':  # 11
        pP = param[:numP]
        pR = param[numP : numP+numR]
        pE = param[numP+numR :]
    elif modelType == 'PRD':  # 12
        pP = param[:numP]
        pR = param[numP : numP+numR]
        pD = param[numP+numR :]
    elif modelType == 'PED':  # 13
        pP = param[:numP]
        pE = param[numP : numP+numE]
        pD = param[numP+numE :]
    elif modelType == 'RED':  # 14
        pR = param[:numR]
        pE = param[numR : numR+numE]
        pD = param[numR+numE :]
    
    elif modelType == 'PRED': # 15
        pP = param[:numP]
        pR = param[numP : numP+numR]
        pE = param[numP+numR : numP+numR+numE]
        pD = param[numP+numR+numE :]

    return pP, pR, pE



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


