"""
File: basis.py

Description: Contains methods of each basis as shape functions in parametric space
"""
import numpy as np


def Monomial1d(xCoords, p):
    """
    @:brief Get modal monomials in one-dimension
    :param xCoords: Coordinates in the x-direction [-1, 1]
    :param p:       Polynomial order in the x-direction
    :return: Monomial values given points
    """
    # Use numpy power series polynomials
    monomials = np.polynomial.polynomial.Polynomial
    # Initialise end result: basis values
    values = np.zeros((xCoords.shape[0], p + 1))
    # values[:, :] = 0.0
    xCoords.shape = -1
    # Polynomial orders start at the zeroth order
    for i in range(p + 1):
        values[:, i] = monomials.basis(i)(xCoords)

    # Reshape x coordinates back to normal
    xCoords.shape = -1, 1

    return values


def Legendre1d(xCoords, p):
    """
    @:brief Get modal Legendre polynomials in one-dimension
    :param xCoords: Coordinates in the x-direction [-1, 1]
    :param p:       Polynomial order in the x-direction
    :return: Legendre polynomial values given points
    """
    # Use numpy legendre polynomials
    legendre = np.polynomial.legendre.Legendre
    # Initialise end result: basis values
    values = np.zeros((xCoords.shape[0], p + 1))
    xCoords.shape = -1

    for i in range(p + 1):
        values[:, i] = legendre.basis(i)(xCoords)
        # values[:, i] *= np.sqrt(i + 0.5)

    xCoords.shape = -1, 1

    return values


def Legendre1dGrad(xCoords, p):
    """
    @:brief Get modal Legendre polynomial derivatives in one-dimension
    :param xCoords: Coordinates in the x-direction [-1, 1]
    :param p:       Polynomial order in the x-direction
    :return: Legendre polynomial derivatives given points
    """
    # Use numpy legendre polynomials
    legendre = np.polynomial.legendre.Legendre
    # Initialise end result: basis gradients
    gradients = np.zeros((xCoords.shape[0], p + 1, 1))
    xCoords.shape = -1

    for i in range(p + 1):
        derivative = legendre.basis(i).deriv(1)
        gradients[:, i, 0] = derivative(xCoords)
        # gradients[:, i, 0] *= np.sqrt(i + 0.5)

    xCoords.shape = -1, 1

    return gradients


def Lagrange1d(xCoords, nodeCoords):
    """
    @:brief Get nodal Lagrange polynomials in one-dimension
    :param xCoords:     Coordinates in the x-direction [-1, 1]
    :param nodeCoords:  Node coordinates as solution nodes where all else but one polynomial has a non-zero value
    :return: Lagrange polynomial values given points and nodes
    """
    nNodes = nodeCoords.shape[0]
    temp = np.ones(nNodes, bool)
    values = np.ones((xCoords.shape[0], nNodes))
    xCoords.shape = -1, 1

    # Loop through number of nodes and evaluate basis values for each polynomial
    for j in range(nNodes):
        temp[j] = False
        qCoordsProd = np.repeat(xCoords.reshape(-1, 1), nNodes - 1, axis=1)
        nodeCoordsProd = np.tile(nodeCoords[temp].reshape(-1), (xCoords.shape[0], 1))

        # np.prod should be evaluated for nNodes - 1 no. of terms, nx no. of times,
        # if 5 nodes and 6 points, something like
        # ([[u u u u], [u u u u], [u u u u], [u u u u], [u u u u], [u u u u]])
        # each array is (xCoords - nodeCoords[temp]) / (nodeCoords[j] - nodeCoords[temp])
        # https://numpy.org/doc/stable/reference/generated/numpy.prod.html

        values[:, j] = np.prod((qCoordsProd - nodeCoordsProd) / (nodeCoords[j] - nodeCoordsProd), axis=1)
        temp[j] = True

    return values


def Legendre2d(coords, p, q):
    """
    @:brief Get modal Legendre polynomials in two-dimensions using tensor products
    :param coords: x and y-coordinates in the [-1, 1]x[-1, 1] domain
    :param p:      Polynomial order in the x-direction
    :param q:      Polynomial order in the y-direction
    :return: Legendre tensorial bases given coordinates
    """
    nPoints = coords.shape[0]
    # Initialise end result: tensor base values
    values = np.zeros((nPoints, (p + 1) * (q + 1)))
    # Get one-dimensional polynomial values first in both x and y-directions
    valuex = Legendre1d(coords[:, 0], p)
    valuey = Legendre1d(coords[:, 1], q)

    for i in range(nPoints):
        # Tensor product
        # https://numpy.org/doc/stable/reference/generated/numpy.outer.html
        values[i, :] = np.reshape(np.outer(valuex[i, :], valuey[i, :]), (-1,), 'F')

    return values


def Legendre2dGrad(coords, p, q):
    """
    @:brief Get modal Legendre polynomial derivatives in two-dimensions using tensor products
    :param coords: x and y-coordinates in the [-1, 1]x[-1, 1] domain
    :param p:      Polynomial order in the x-direction
    :param q:      Polynomial order in the y-direction
    :return: Legendre tensorial base derivatives given coordinates
    """
    nPoints = coords.shape[0]
    # Initialise end result: tensor base gradients
    gradients = np.zeros((nPoints, (p + 1) * (q + 1), 2))
    # Get one-dimensional polynomial derivatives first in both x and y-directions
    valuex = Legendre1d(coords[:, 0], p)
    valuey = Legendre1d(coords[:, 1], q)
    gradientx = Legendre1dGrad(coords[:, 0], p)
    gradienty = Legendre1dGrad(coords[:, 1], q)

    for i in range(nPoints):
        gradients[i, :, 0] = np.reshape(np.outer(gradientx[i, :, 0], valuey[i, :]), (-1,), 'F')
        gradients[i, :, 1] = np.reshape(np.outer(valuex[i, :], gradienty[i, :, 0]), (-1,), 'F')

    return gradients


def Lagrange2d(coords, xnodes, ynodes):
    """
    @:brief Get nodal Lagrange polynomials in two-dimensions using tensor products
            (vertex-based shape not supported yet)
    :param coords: Coordinates in the x and y-direction within [-1, 1]x[-1, 1] space
    :param xnodes: Node coordinates in the x-direction
    :param ynodes: Node coordinates in the y-direction
    :return: Lagrange tensorial base values given points and nodes
    """
    nPoints = coords.shape[0]
    NxNodes = xnodes.shape[0]
    NyNodes = ynodes.shape[0]
    values = np.zeros((nPoints, NxNodes * NyNodes))

    # Get one-dimensional basis values first
    valuex = Lagrange1d(coords[:, 0].reshape(-1, 1), xnodes)
    valuey = Lagrange1d(coords[:, 1].reshape(-1, 1), ynodes)

    # Tensor products to get two-dimensional bases values
    for i in range(nPoints):
        values[i, :] = np.reshape(np.outer(valuex[i, :], valuey[i, :]), (-1,), 'F')

    return values
