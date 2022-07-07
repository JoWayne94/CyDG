"""
File: quadrilateral.py

Description: Quadrilateral cell child class.
"""
from src.library.dgCells.dgCell import DgCell as Cell
from src.library.dgCells.segment import Segment as Seg
from src.library.paramCells.basis import *
from src.library.paramCells.paramQuad import ParamQuad
from src.library.geometries.quadrilateral import Quadrilateral as QuadGeom


class Quadrilateral(Cell):
    """
    @:brief 2D implementation of a quadrilateral
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
            self.invJacobian = None

    class CellSolutionData:
        """
        @:brief Cellular solution data subclass
        """

        def __init__(self):
            self.uCoeffs = None
            self.uPhysical = None

    class CellFacesObjects:
        """
        @:brief Cell faces class instances
        """
        def __init__(self):
            self.F0 = None
            self.F1 = None
            self.F2 = None
            self.F3 = None

    def __init__(self, shape, pointlabels, points, neighbourlabels, nvars, p1, p2):
        super().__init__(shape, pointlabels, points, neighbourlabels)
        self.paramQuad = ParamQuad("GL", p1, p2)
        self.matCell = self.CellMatricesData()
        self.geomCell = self.CellGeometricData()
        self.solnCell = self.CellSolutionData()
        self.facesCell = self.CellFacesObjects()

        # Geometric data
        # np.full(len(self.weights), 2).reshape(-1, 1)
        self.geomCell.geometry = QuadGeom(self.geomData.pointLabels, self.geomData.points,
                                          np.array([i[0] for i in self.paramQuad.zeros]),
                                          np.array([i[1] for i in self.paramQuad.zeros]))
        self.geomCell.detJacobian = (self.geomCell.geometry.detJacobian()).reshape(-1, 1)
        self.geomCell.invJacobian = self.geomCell.geometry.invJacobianMatrix()

        # Matrices data
        self.matCell.basisMatrix = Legendre2d(self.paramQuad.zeros, p1, p2)
        self.matCell.derivMatrix = Legendre2dGrad(self.paramQuad.zeros, p1, p2)

        # Solution data
        self.solnCell.uCoeffs = np.zeros(((p1 + 1) * (p2 + 1), nvars))
        self.solnCell.uPhysical = np.zeros((len(self.paramQuad.zeros), nvars))  # revisit

        # Matrix operators
        self.massMatrix = self.GetMassMatrix()
        self.invMassMatrix = np.linalg.inv(self.massMatrix)
        self.stiffnessMatrix = self.GetStiffnessMatrix()
        self.laplacianMatrix = self.GetLaplacianMatrix()

        # Cell faces
        self.facesCell.F0 = Seg('S', self.geomData.pointLabels[0:2], self.geomData.points[:, 0].reshape(-1, 1),
                                [None, None], nvars, p1)
        self.facesCell.F1 = Seg('S', self.geomData.pointLabels[1:3], self.geomData.points[:, 1].reshape(-1, 1),
                                [None, None], nvars, p2)
        self.facesCell.F2 = Seg('S', [self.geomData.pointLabels[2], self.geomData.pointLabels[3]],
                                self.geomData.points[:, 0].reshape(-1, 1), [None, None], nvars, p1)
        self.facesCell.F3 = Seg('S', [self.geomData.pointLabels[3], self.geomData.pointLabels[0]],
                                self.geomData.points[:, 1].reshape(-1, 1), [None, None], nvars, p2)

    def calculateCellVolume(self):
        # https://en.wikipedia.org/wiki/Shoelace_formula
        # https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
        x = np.array([self.geomData.points[point][0] for point in self.geomData.pointLabels])
        y = np.array([self.geomData.points[point][1] for point in self.geomData.pointLabels])
        volume = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        return volume

    def calculateFaceNormals(self):
        """
        @:brief Unit face normals of all faces [F0, F1, F2, F3]. Is outward-facing (un-normalized) normal vector needed?
        :return: Unit normal vector
        """
        n = np.empty((4, 2))
        n[0] = np.array([0, -1])  # F0
        n[1] = np.array([1, 0])  # F1
        n[2] = np.array([0, 1])  # F2
        n[3] = np.array([-1, 0])  # F3

        return n

    @property
    def GetQuadratureCoords(self):
        return self.geomCell.geometry.parametricMapping()

    def GetMassMatrix(self):
        """
        @:brief Construct Basis transposed, Weights, and Basis matrices to assemble the mass matrix
                B^T W B = M
        :return: Cellular mass matrix ([(p1+1) * (p2+1), no. of gauss points]) per cell entry
        """
        # https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
        # https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        massMatrix = np.matmul(self.matCell.basisMatrix.transpose(), self.matCell.basisMatrix * self.paramQuad.weights *
                               abs(self.geomCell.detJacobian))

        return massMatrix

    def GetStiffnessMatrix(self):
        """
        @:brief Construct Derivative transposed, Weights, and Basis matrices to assemble the stiffness matrix
                ((dxi1/dx1 + dxi1/dx2) (D_xi1 B)^T + (dxi2/dx1 + dxi2dx2) (D_xi2 B)^T) W B = S
        :return: Cellular stiffness matrix per cell entry
        """
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

        stiffnessMatrix = np.matmul(np.einsum('kij, kli -> kl', self.geomCell.invJacobian,
                                              self.matCell.derivMatrix[:, :]).transpose(),
                                    self.matCell.basisMatrix * self.paramQuad.weights * abs(self.geomCell.detJacobian))

        return stiffnessMatrix

    def GetLaplacianMatrix(self):
        """
        @:brief Construct Derivative transposed, Weights, and Derivative matrices to assemble the Laplacian matrix
        :return: Cellular weak Laplacian matrix
        """
        # laplacian = np.matmul(np.einsum('kij, kli -> kl', self.geomCell.geometry.invJacobian,
        #                                 self.matCell.derivMatrix[:, :]).transpose(),
        #                       self.paramQuad.weights * abs(self.geomCell.detJacobian) *
        #                       np.einsum('kij, kli -> kl', self.geomCell.geometry.invJacobian,
        #                                 self.matCell.derivMatrix[:, :]))
        laplacian = np.matmul((self.geomCell.invJacobian[:, :, 0][:, 0] * self.matCell.derivMatrix[:, :, 0] +
                               self.geomCell.invJacobian[:, :, 1][:, 0] * self.matCell.derivMatrix[:, :, 1]).transpose(),
                              self.paramQuad.weights * abs(self.geomCell.detJacobian) *
                              (self.geomCell.invJacobian[:, :, 0][:, 0] * self.matCell.derivMatrix[:, :, 0] +
                               self.geomCell.invJacobian[:, :, 1][:, 0] * self.matCell.derivMatrix[:, :, 1])) + \
                    np.matmul((self.geomCell.invJacobian[:, :, 0][:, 1] * self.matCell.derivMatrix[:, :, 0] +
                               self.geomCell.invJacobian[:, :, 1][:, 1] * self.matCell.derivMatrix[:, :, 1]).transpose(),
                              self.paramQuad.weights * abs(self.geomCell.detJacobian) *
                              (self.geomCell.invJacobian[:, :, 0][:, 1] * self.matCell.derivMatrix[:, :, 0] +
                               self.geomCell.invJacobian[:, :, 1][:, 1] * self.matCell.derivMatrix[:, :, 1]))

        return laplacian
