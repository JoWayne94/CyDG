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
    def __init__(self, quadrature, p1, p2):
        """
        @:brief Main constructor
        :param p1:         Polynomial order in the xi_1 direction
        :param p2:         Polynomial order in the xi_2 direction
        :param quadrature: Type of quadrature selected
        """
        self.P1 = p1
        self.P2 = p2
        self.Quadrature = quadrature
        self.Zeros, self.Weights = self.GetQuadratureZerosWeights()

    @property
    def p1(self):
        return self.P1

    @property
    def p2(self):
        return self.P2

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
        Segment1 = Segment(self.quadrature, self.p1)
        Segment2 = Segment(self.quadrature, self.p2)
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
