"""
File: segment.py

Description: Segment cell child class.
"""
from src.library.dgCells.dgCell import DgCell as Cell
from src.library.paramCells.basis import *
from src.library.paramCells.paramSeg import ParamSeg
from src.library.geometries.segment import Segment as SegGeom


class Segment(Cell):
    """
    @:brief 1D implementation of a line
    """

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
            self.geometry = None
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

        # Geometric data
        self.geomCell.geometry = SegGeom(self.geomData.pointLabels, self.geomData.points,
                                         np.array([i[0] for i in self.paramSeg.zeros]))
        self.geomCell.detJacobian = (self.geomCell.geometry.detJacobian()).reshape(-1, 1)

        # Matrices data
        self.matCell.basisMatrix = Legendre1d(self.paramSeg.zeros, p1)
        self.matCell.derivMatrix = Legendre1dGrad(self.paramSeg.zeros, p1)

        # Solution data
        self.solnCell.uCoeffs = np.zeros((p1 + 1, nvars))
        self.solnCell.uPhysical = np.zeros((len(self.paramSeg.zeros), nvars))

        # Matrix operators
        self.massMatrix = self.GetMassMatrix()
        self.invMassMatrix = np.linalg.inv(self.massMatrix)
        self.stiffnessMatrix = self.GetStiffnessMatrix()
        self.laplacianMatrix = self.GetLaplacianMatrix()

    def calculateCellVolume(self):
        return abs(self.geomData.points[self.geomData.pointLabels[0]][0] -
                   self.geomData.points[self.geomData.pointLabels[1]][0])

    def calculateFaceNormals(self):
        return np.array([-1, 1])

    @property
    def GetQuadratureCoords(self):
        return self.geomCell.geometry.parametricMapping()

    def GetMassMatrix(self):
        """
        @:brief Construct Basis transposed, Weights, and Basis matrices to assemble the mass matrix
                B^T W B = M
        :return: Cellular mass matrix ([p+1, no. of gauss points]) per cell entries
        """
        massMatrix = np.matmul(self.matCell.basisMatrix.transpose(), self.matCell.basisMatrix * self.paramSeg.weights *
                               abs(self.geomCell.detJacobian))

        return massMatrix

    def GetStiffnessMatrix(self):
        """
        @:brief Construct Derivative transposed, Weights, and Basis matrices to assemble the stiffness matrix
                dxi/dx (D B)^T W B = S
        :return: Cellular stiffness matrix (singular) per cell entries
        """
        stiffnessMatrix = np.matmul(self.geomCell.geometry.dxi1dx1 * self.matCell.derivMatrix[:, :, 0].transpose(),
                                    self.matCell.basisMatrix * self.paramSeg.weights
                                    * abs(self.geomCell.detJacobian))

        return stiffnessMatrix

    def GetLaplacianMatrix(self):
        """
        @:brief Construct Derivative transposed, Weights, and Derivative matrices to assemble the laplacian matrix
                dxi/dx (D B)^T W dxi/dx (D B) = L
        :return: Cellular stiffness matrix (singular) per cell entries
        """
        laplacianMatrix = np.matmul(self.geomCell.geometry.dxi1dx1 * self.matCell.derivMatrix[:, :, 0].transpose(),
                                    self.geomCell.geometry.dxi1dx1 * self.matCell.derivMatrix[:, :, 0] *
                                    self.paramSeg.weights * abs(self.geomCell.detJacobian))

        return laplacianMatrix
