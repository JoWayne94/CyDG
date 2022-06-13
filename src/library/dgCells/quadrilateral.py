"""
File: quadrilateral.py

Description: Quadrilateral cell subclass.
"""
import numpy as np

from src.library.dgCells.dgCell import DgCell as Cell
from src.library.paramCells.basis import *
from src.library.paramCells.paramQuad import ParamQuad
from src.library.geometries.quadrilateral import Quadrilateral as QuadGeom


class Quadrilateral(Cell):

    def __init__(self, shape, pointlabels, points, neighbourlabels, p1, p2):
        super().__init__(shape, pointlabels, points, neighbourlabels)
        self.P1 = p1
        self.P2 = p2
        self.paramQuad = ParamQuad("GL", p1, p2)
        self.basisMatrix = GetLegendre2d(self.paramQuad.zeros, p1, p2)
        # np.full(len(self.weights), 2).reshape(-1, 1)
        self.quad = QuadGeom(self.geomData.pointLabels, self.geomData.points,
                             np.array([i[0] for i in self.paramQuad.zeros]),
                             np.array([i[1] for i in self.paramQuad.zeros]))
        self.jacobian = (self.quad.detJacobian()).reshape(-1, 1)
        self.invJacobian = self.quad.invJacobianMatrix()

    def calculateCellVolume(self):
        # https://en.wikipedia.org/wiki/Shoelace_formula
        # https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
        x = np.array([self.geomData.points[point][0] for point in self.geomData.pointLabels])
        y = np.array([self.geomData.points[point][1] for point in self.geomData.pointLabels])
        volume = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        return volume

    def GetMassMatrix(self):
        """
        @:brief Construct Basis transposed, Weights, and Basis matrices to assemble the mass matrix
                B^T W B = M
        :return: Cellular mass matrix ([p+1, p+1]) per cell entries
        """
        # https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
        # https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        massMatrix = np.matmul(self.basisMatrix.transpose(), self.basisMatrix * self.paramQuad.weights *
                               abs(self.jacobian))

        return massMatrix

    def GetStiffnessMatrix(self):
        derivMatrix = GetLegendre2dGrad(self.paramQuad.zeros, self.P1, self.P2)

        # print("np.einsum")
        # print(np.einsum('ij', self.invJacobian))
        # print(np.einsum('ijk', derivMatrix[:, :]))
        # print(self.invJacobian[0][0] * derivMatrix[:, :, 0] + self.invJacobian[1][0] * derivMatrix[:, :, 1] +
        #       self.invJacobian[0][1] * derivMatrix[:, :, 0] + self.invJacobian[1][1] * derivMatrix[:, :, 1])

        # stiffnessMatrix = np.matmul((self.invJacobian[0][0] * derivMatrix[:, :, 0] +
        #                              self.invJacobian[1][0] * derivMatrix[:, :, 1]).transpose(),
        #                             self.basisMatrix * self.paramQuad.weights * abs(self.jacobian)) + \
        #     np.matmul((self.invJacobian[0][1] * derivMatrix[:, :, 0] +
        #                self.invJacobian[1][1] * derivMatrix[:, :, 1]).transpose(),
        #               self.basisMatrix * self.paramQuad.weights * abs(self.jacobian))

        stiffnessMatrix = np.matmul(np.einsum('kij, kli -> kl', self.invJacobian, derivMatrix[:, :]).transpose(),
                                    self.basisMatrix * self.paramQuad.weights * abs(self.jacobian))

        return stiffnessMatrix

    # def GetLaplacianMatrix(self):
    #     """
    #     @:brief Construct Derivative transposed, Weights, and Derivative matrices to assemble the Laplacian matrix
    #             D_xi1^T W D_xi1 + D_xi2^T W D_xi2 = L
    #     :return: Cellular Laplacian matrix
    #     """
    #     derivMatrix = GetLegendre2dGrad(self.zeros, 1, 1)
    #
    #     laplacian = np.matmul(derivMatrix[:, :, 0].transpose(), derivMatrix[:, :, 0] * self.weights) + \
    #                 np.matmul(derivMatrix[:, :, 1].transpose(), derivMatrix[:, :, 1] * self.weights)
    #
    #     return laplacian
