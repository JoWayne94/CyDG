"""
File: segment.py

Description: Segment cell subclass. 1D implementation of a line
"""
from src.library.dgCells.dgCell import DgCell as Cell
from src.library.paramCells.basis import *
from src.library.paramCells.paramSeg import ParamSeg
from src.library.geometries.segment import Segment as SegGeom


class Segment(Cell):
    class CellMatricesData:
        """
        @:brief Cellular matrices data subclass
        """

        def __init__(self):
            self.basisMatrix = None
            self.derivMatrix = None

    class CellGeometricData:
        """
        @:brief Cellular geometric data subclass
        """

        def __init__(self):
            self.segment = None
            self.detJacobian = None

    class CellSolutionData:
        """
        @:brief Cellular solution data subclass
        """

        def __init__(self):
            self.uCoeffs = None
            self.uPhysical = None

    def __init__(self, shape, pointlabels, points, neighbourlabels, nvars, p1):
        super().__init__(shape, pointlabels, points, neighbourlabels)
        self.paramSeg = ParamSeg("GL", p1)
        self.matCell = self.CellMatricesData()
        self.geomCell = self.CellGeometricData()
        self.solnCell = self.CellSolutionData()

        # Matrices data
        self.matCell.basisMatrix = GetLegendre1d(self.paramSeg.zeros, p1)
        self.matCell.derivMatrix = GetLegendre1dGrad(self.paramSeg.zeros, p1)

        # Geometric data
        self.geomCell.segment = SegGeom(self.geomData.pointLabels, self.geomData.points,
                                        np.array([i[0] for i in self.paramSeg.zeros]))
        self.geomCell.detJacobian = (self.geomCell.segment.detJacobian()).reshape(-1, 1)

        # Solution data
        self.solnCell.uCoeffs = np.zeros((p1 + 1, nvars))
        self.solnCell.uPhysical = np.zeros((len(self.paramSeg.zeros), nvars))

    def calculateCellVolume(self):
        return abs(self.geomData.points[self.geomData.pointLabels[0]][0] -
                   self.geomData.points[self.geomData.pointLabels[1]][0])

    def calculateFaceNormals(self):
        return np.array([-1, 1])

    @property
    def GetQuadratureCoords(self):
        return self.geomCell.segment.parametricMapping()

    def GetMassMatrix(self):
        """
        @:brief Construct Basis transposed, Weights, and Basis matrices to assemble the mass matrix
                B^T W B = M
        :return: Cellular mass matrix ([p+1, p+1]) per cell entries
        """
        massMatrix = np.matmul(self.matCell.basisMatrix.transpose(), self.matCell.basisMatrix * self.paramSeg.weights *
                               abs(self.geomCell.detJacobian))
        # massMatrix = np.matmul(self.matCell.basisMatrix.transpose(), self.matCell.basisMatrix)

        return massMatrix

    def GetStiffnessMatrix(self):
        """
        @:brief Construct Basis transposed, Weights, and Derivative matrices to assemble the stiffness matrix
                (dxi1dx1 D B)^T W B = S
        :return: Cellular stiffness matrix (singular) per cell entries
        """
        stiffnessMatrix = np.matmul(self.geomCell.segment.dxi1dx1 * self.matCell.derivMatrix[:, :, 0].transpose(),
                                    self.matCell.basisMatrix * self.paramSeg.weights
                                    * abs(self.geomCell.detJacobian))
        # stiffnessMatrix = np.matmul(self.matCell.basisMatrix.transpose() * self.paramSeg.weights
        #                             * abs(self.geomCell.detJacobian), self.geomCell.segment.dxi1dx1 *
        #                             self.matCell.derivMatrix[:, :, 0])
        # stiffnessMatrix = self.geomCell.segment.dxi1dx1 * self.matCell.derivMatrix[:, :, 0].transpose() \
        #                   * self.paramSeg.weights * abs(self.geomCell.detJacobian)

        return stiffnessMatrix

    def GetLaplacianMatrix(self):
        laplacianMatrix = np.matmul(self.geomCell.segment.dxi1dx1 * self.matCell.derivMatrix[:, :, 0].transpose(),
                                    self.paramSeg.weights * self.geomCell.segment.dxi1dx1 *
                                    self.matCell.derivMatrix[:, :, 0] * abs(self.geomCell.detJacobian))

        return laplacianMatrix
