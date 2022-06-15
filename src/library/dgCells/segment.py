"""
File: segment.py

Description: Segment cell subclass. 1D implementation of a line
"""
import numpy as np

from src.library.dgCells.dgCell import DgCell as Cell
from src.library.paramCells.basis import *
from src.library.paramCells.paramSeg import ParamSeg
from src.library.geometries.segment import Segment as SegGeom


class Segment(Cell):

    def __init__(self, shape, pointlabels, points, neighbourlabels, p1):
        super().__init__(shape, pointlabels, points, neighbourlabels)
        self.P1 = p1
        self.uCoeffs = np.empty(p1 + 1)
        self.paramSeg = ParamSeg("GL", p1)
        self.uSoln = np.empty(len(self.paramSeg.zeros))
        self.basisMatrix = GetLegendre1d(self.paramSeg.zeros, p1)
        self.seg = SegGeom(self.geomData.pointLabels, self.geomData.points,
                           np.array([i[0] for i in self.paramSeg.zeros]))
        self.jacobian = (self.seg.detJacobian()).reshape(-1, 1)

    def calculateCellVolume(self):
        return abs(self.geomData.points[self.geomData.pointLabels[0]][0] -
                   self.geomData.points[self.geomData.pointLabels[1]][0])

    @property
    def GetQuadratureCoords(self):
        return self.seg.parametricMapping()

    def GetMassMatrix(self):
        """
        @:brief Construct Basis transposed, Weights, and Basis matrices to assemble the mass matrix
                B^T W B = M
        :return: Cellular mass matrix ([p+1, p+1]) per cell entries
        """
        massMatrix = np.matmul(self.basisMatrix.transpose(), self.basisMatrix * self.paramSeg.weights *
                               abs(self.jacobian))

        return massMatrix

    def GetStiffnessMatrix(self):
        """
        @:brief Construct Basis transposed, Weights, and Derivative matrices to assemble the stiffness matrix
                (dxi1dx1 D B)^T W B = S
        :return: Cellular stiffness matrix (singular) per cell entries
        """
        derivMatrix = GetLegendre1dGrad(self.paramSeg.zeros, self.P1)

        stiffnessMatrix = np.matmul(self.seg.dxi1dx1 * derivMatrix[:, :, 0].transpose(),
                                    self.basisMatrix * self.paramSeg.weights * abs(self.jacobian))

        return stiffnessMatrix

    def GetLaplacianMatrix(self):

        derivMatrix = GetLegendre1dGrad(self.paramSeg.zeros, self.P1)

        laplacianMatrix = np.matmul(self.seg.dxi1dx1 * derivMatrix[:, :, 0].transpose(), self.paramSeg.weights *
                                    self.seg.dxi1dx1 * derivMatrix[:, :, 0] * abs(self.jacobian))

        return laplacianMatrix
