"""
File: paramQuad.py

Description: Subclass to evaluate quadrature of a quadrilateral
"""
import numpy as np
from src.library.paramCells.paramCell2d import ParamCell2d as Cell
from src.library.paramCells.paramSeg import ParamSeg as Segment


class ParamQuad(Cell):
    """
    @:brief Subclass to represent a 2d quad cell in parametric space
    """
    def __init__(self, quadrature, p, q):
        """
        @:brief Main constructor
        :param p:          Polynomial order in the xi_1 direction
        :param q:          Polynomial order in the xi_2 direction
        :param quadrature: Type of quadrature selected
        """
        # super().__init__(quadrature, p, q, nNodesP, nNodesQ)
        self.P = p
        self.Q = q
        self.Quadrature = quadrature
        self.Zeros, self.Weights = self.GetQuadratureZerosWeights()

    @property
    def p(self):
        return self.P

    @property
    def q(self):
        return self.Q

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
        """
        @:brief Calls the segment quadrature function to get quadrature zeros and weights.
                Construct quadrilateral shapes using tensor products
        :return: Quadrature point coordinates and weights
        """
        # Get quadrature zeros and weights from 1D
        Segment1 = Segment(self.quadrature, self.p)
        Segment2 = Segment(self.quadrature, self.q)
        qZeros1, qWeights1 = Segment1.GetQuadratureZerosWeights()
        qZeros2, qWeights2 = Segment2.GetQuadratureZerosWeights()

        # Weights
        qWeights = np.reshape(np.outer(qWeights1, qWeights2), (-1,), 'F').reshape(-1, 1)

        # Zeros
        qZeros = np.zeros([qWeights.shape[0], 2])
        # https://numpy.org/doc/stable/reference/generated/numpy.tile.html
        qZeros[:, 0] = np.tile(qZeros1, (qZeros2.shape[0], 1)).reshape(-1)
        qZeros[:, 1] = np.repeat(qZeros2, qZeros1.shape[0], axis=0).reshape(-1)

        return qZeros, qWeights
