"""
File: paramSeg.py

Description: Subclass to evaluate quadrature of a segment/line.

Add: Gauss Radau, Vertex-based shape where quadrature points are treated as solution nodes.
     Collocation scheme for nodal DG.
"""
import numpy as np
from src.library.paramCells.paramCell import ParamCell as Cell


class ParamSeg(Cell):

    def __init__(self, quadrature, p):
        """
        @:brief Main constructor
        :param p:          Polynomial order in the xi_1 direction
        :param quadrature: Type of quadrature selected
        """
        # if nNodes is None:
        #     self.nNodes = 0
        # else:
        #     self.nNodes = nNodes
        self.P = p
        self.Quadrature = quadrature
        self.Zeros, self.Weights = self.GetQuadratureZerosWeights()

    @property
    def p(self):
        return self.P

    @p.setter
    def p(self, value):
        self.P = value

    @property
    def quadrature(self):
        return self.Quadrature

    @property
    def zeros(self):
        return self.Zeros

    @property
    def weights(self):
        return self.Weights

    def GetQuadratureZerosWeights(self):

        if self.quadrature == "GL" or self.p == 1:  # Gauss-Legendre
            qZeros, qWeights = self.GetQuadratureGL()
        elif self.quadrature == "GLL":  # Gauss-Lobatto-Legendre
            qZeros, qWeights = self.GetQuadratureGLL()
        else:
            raise NotImplementedError

        return qZeros, qWeights

    def GetQuadratureGL(self):
        """
        @:brief Get quadrature zeros and weights using numpy Gauss-Legendre library
        :return: Quadrature point coordinates and weights
        """
        # If p is even, add 1
        # if self.p % 2 == 0:
        #     self.p += 1
        # Number of integration points to obtain exact integration given polynomial order
        nZeros = (self.p + 1)  # // 2

        qZeros, qWeights = np.polynomial.legendre.leggauss(nZeros)

        qZeros.shape = -1, 1
        qWeights.shape = -1, 1

        return qZeros, qWeights

    def GetQuadratureGLL(self):
        """
        @:brief Get Gauss-Lobatto-Legendre quadrature zeros and weights
        :return: Quadrature point coordinates and weights
        """
        # If p is even, add 1
        if self.p % 2 == 0:
            self.p += 1
        # Number of integration points to obtain exact integration given polynomial order
        nZeros = (self.p + 3) // 2

        qZeros, qWeights = self.GetGLLZerosWeights(nZeros, np.finfo(float).eps)
        qZeros = qZeros.reshape(qZeros.shape[0], 1)
        qWeights = qWeights.reshape(qWeights.shape[0], 1)

        return qZeros, qWeights

    @staticmethod
    def GetGLLZerosWeights(m, tol):
        """
        @:brief Get Gauss-Lobatto-Legendre zeros and weights using Newton-Raphson iteration
        :param m:   nZeros - 1
        :param tol: Tolerance set as machine precision/epsilon
        :return: Quadrature zero coordinates and weights
        """
        legendre = np.polynomial.legendre.Legendre

        # i = 0, ..., m-1
        i = np.arange(m)
        # Initialise Legendre polynomials
        temp = np.zeros([m, m])
        # Chebyshev polynomial as initial guess
        x_im = -np.cos(((2 * i + 1) / (2 * (m - 1))) * np.pi)
        # Max no. of iterations
        niter = 1000
        # Iterative evaluation to get zeros of Legendre polynomial derivatives
        for k in range(m):

            r = x_im[k]

            if k > 0:
                r = (r + x_im[k - 1]) * 0.5

            for j in range(niter):
                s = np.sum((1 / (r - x_im))[0:k])
                # Initialise 1D Vandermonde matrix
                V = np.polynomial.legendre.legvander(r, m - 1)
                delta = - 1 / (((m * V[0][m - 1]) / (r * V[0][m - 1] - V[0][m - 2])) - s)
                r += delta

                # Check if tolerance is met
                if np.amax(np.abs(delta)) < tol:
                    break
                # Convergence error
                if j == niter - 1:
                    raise ValueError

            x_im[k] = r

        # Evaluate Legendre polynomials
        for i in range(m):
            temp[:, i] = legendre.basis(i)(x_im)

        # Quadrature weights
        weights = 2.0 / (m * (m - 1) * temp[:, m - 1] ** 2)

        return x_im, weights
