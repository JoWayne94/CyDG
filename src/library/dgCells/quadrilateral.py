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
    @:brief 2D implementation of a straight-sided quadrilateral
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
            self.F = np.empty(4, dtype=Seg)

    def __init__(self, shape, pointlabels, points, neighbourlabels, nvars, p1, p2):
        """
        @:brief Main constructor
        :param shape:           Shape of the cell
        :param pointlabels:     Point IDs that make up the cell
        :param points:          Point coordinates list parsed in by reference
        :param neighbourlabels: Neighbour cell IDs
        :param nvars:           Number of state variables
        :param p1:              Polynomial order in the x-direction
        :param p2:              Polynomial order in the y-direction
        """
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
        """
        :param basisMatrix: Cellular basis matrix -> B ([Number of Gauss points, Number of polynomials])
        :param derivMatrix: Cellular derivative matrix, d Phi/d xi -> (D_xi1 B) & (D_xi2 B) 
        """
        self.matCell.basisMatrix = Legendre2d(self.paramQuad.zeros, p1, p2)
        self.matCell.derivMatrix = Legendre2dGrad(self.paramQuad.zeros, p1, p2)

        # Solution data
        self.solnCell.uCoeffs = np.zeros(((p1 + 1) * (p2 + 1), nvars))
        self.solnCell.uPhysical = np.zeros((len(self.paramQuad.zeros), nvars))  # revisit

        # Matrix operators
        """
        :param massMatrix:      Cellular mass matrix -> B^T W B
        :param stiffnessMatrix: Cellular stiffness matrix -> d xi/d x (D B)^T W B
        :param laplacianMatrix: Cellular weak Laplacian matrix -> d xi/d x (D B)^T W (D B) d xi/d x
        """
        self.massMatrix = self.GetMassMatrix()
        self.invMassMatrix = np.linalg.inv(self.massMatrix)
        self.stiffnessMatrix = self.GetStiffnessMatrix()
        self.laplacianMatrix = self.GetLaplacianMatrix()

        # Cell faces
        self.facesCell.F[0] = Seg('S', self.geomData.pointLabels[0:2], self.geomData.points[:, 0].reshape(-1, 1),
                                  [None, None], nvars, p1)
        self.facesCell.F[0].faceZeros = np.array([[zero[0], -1] for zero in self.facesCell.F[0].paramSeg.zeros])
        self.facesCell.F[0].faceBasis = Legendre2d(self.facesCell.F[0].faceZeros, p1, p2)
        self.facesCell.F[0].faceDeriv = Legendre2dGrad(self.facesCell.F[0].faceZeros, p1, p2)
        self.facesCell.F[0].unitNormal = np.full((len(self.facesCell.F[0].GetQuadratureCoords), 2), [0, -1], dtype=list)

        self.facesCell.F[1] = Seg('S', self.geomData.pointLabels[1:3], self.geomData.points[:, 1].reshape(-1, 1),
                                  [None, None], nvars, p2)
        self.facesCell.F[1].faceZeros = np.array([[1, zero[0]] for zero in self.facesCell.F[1].paramSeg.zeros])
        self.facesCell.F[1].faceBasis = Legendre2d(self.facesCell.F[1].faceZeros, p1, p2)
        self.facesCell.F[1].faceDeriv = Legendre2dGrad(self.facesCell.F[1].faceZeros, p1, p2)
        self.facesCell.F[1].unitNormal = np.full((len(self.facesCell.F[1].GetQuadratureCoords), 2), [1, 0], dtype=list)

        self.facesCell.F[2] = Seg('S', [self.geomData.pointLabels[3], self.geomData.pointLabels[2]],
                                  self.geomData.points[:, 0].reshape(-1, 1), [None, None], nvars, p1)
        self.facesCell.F[2].faceZeros = np.array([[zero[0], 1] for zero in self.facesCell.F[2].paramSeg.zeros])
        self.facesCell.F[2].faceBasis = Legendre2d(self.facesCell.F[2].faceZeros, p1, p2)
        self.facesCell.F[2].faceDeriv = Legendre2dGrad(self.facesCell.F[2].faceZeros, p1, p2)
        self.facesCell.F[2].unitNormal = np.full((len(self.facesCell.F[2].GetQuadratureCoords), 2), [0, 1], dtype=list)

        self.facesCell.F[3] = Seg('S', [self.geomData.pointLabels[0], self.geomData.pointLabels[3]],
                                  self.geomData.points[:, 1].reshape(-1, 1), [None, None], nvars, p2)
        self.facesCell.F[3].faceZeros = np.array([[-1, zero[0]] for zero in self.facesCell.F[3].paramSeg.zeros])
        self.facesCell.F[3].faceBasis = Legendre2d(self.facesCell.F[3].faceZeros, p1, p2)
        self.facesCell.F[3].faceDeriv = Legendre2dGrad(self.facesCell.F[3].faceZeros, p1, p2)
        self.facesCell.F[3].unitNormal = np.full((len(self.facesCell.F[3].GetQuadratureCoords), 2), [-1, 0], dtype=list)

    def calculateCellVolume(self):
        """
        @:brief Calculates area of a polygon using the Shoelace formula
        :return: Area of the cell
        """
        # https://en.wikipedia.org/wiki/Shoelace_formula
        # https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
        x = np.array([self.geomData.points[point][0] for point in self.geomData.pointLabels])
        y = np.array([self.geomData.points[point][1] for point in self.geomData.pointLabels])
        volume = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        return volume

    def calculateFaceNormals(self):
        """
        @:brief Unit face normals of all faces [F0, F1, F2, F3]. Is outward-facing (un-normalized) normal vector needed?
        :return: Unit normal vectors
        """
        n = np.empty((4, 2))  # uniform P1 = P2 for now
        # n[0] = np.full((len(self.facesCell.F[0].GetQuadratureCoords), 2), [0, -1], dtype=list)  # F0
        # n[1] = np.full((len(self.facesCell.F[1].GetQuadratureCoords), 2), [1, 0], dtype=list)  # F1
        # n[2] = np.full((len(self.facesCell.F[2].GetQuadratureCoords), 2), [0, 1], dtype=list)  # F2
        # n[3] = np.full((len(self.facesCell.F[3].GetQuadratureCoords), 2), [-1, 0], dtype=list)  # F3

        return n

    @property
    def GetQuadratureCoords(self):
        """
        @:brief Quadrature coordinates getter
        :return: Quadrature coordinates in real space
        """
        return self.geomCell.geometry.parametricMapping()

    def GetMassMatrix(self):
        """
        @:brief Construct Basis transposed, Weights, and Basis matrices to assemble the mass matrix
                B^T W B = M
        :return: Cellular mass matrix ([no. of gauss points, (p1+1) * (p2+1)]) per cell entry
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
        """ Stiffness matrix in total """
        # stiffnessMatrix = np.matmul(np.einsum('kij, kli -> kl', self.geomCell.invJacobian,
        #                                       self.matCell.derivMatrix[:, :]).transpose(),
        #                            self.matCell.basisMatrix * self.paramQuad.weights * abs(self.geomCell.detJacobian))

        """ Stiffness matrix seperated in x and y-direction """
        stiffness_x = np.matmul((self.geomCell.invJacobian[:, :, 0][:, 0] * self.matCell.derivMatrix[:, :, 0] +
                                 self.geomCell.invJacobian[:, :, 1][:, 0] * self.matCell.derivMatrix[:, :, 1])
                                .transpose(),
                                self.matCell.basisMatrix * self.paramQuad.weights * abs(self.geomCell.detJacobian))

        stiffness_y = np.matmul((self.geomCell.invJacobian[:, :, 0][:, 1] * self.matCell.derivMatrix[:, :, 0] +
                                 self.geomCell.invJacobian[:, :, 1][:, 1] * self.matCell.derivMatrix[:, :, 1])
                                .transpose(),
                                self.matCell.basisMatrix * self.paramQuad.weights * abs(self.geomCell.detJacobian))

        return [stiffness_x, stiffness_y]

    def GetLaplacianMatrix(self):
        """
        @:brief Construct Derivative transposed, Weights, and Derivative matrices to assemble the Laplacian matrix
        :return: Cellular weak Laplacian matrix per cell entry
        """
        # laplacian = np.matmul(np.einsum('kij, kli -> kl', self.geomCell.geometry.invJacobian,
        #                                 self.matCell.derivMatrix[:, :]).transpose(),
        #                       self.paramQuad.weights * abs(self.geomCell.detJacobian) *
        #                       np.einsum('kij, kli -> kl', self.geomCell.geometry.invJacobian,
        #                                 self.matCell.derivMatrix[:, :]))
        laplacian = np.matmul((self.geomCell.invJacobian[:, :, 0][:, 0] * self.matCell.derivMatrix[:, :, 0] +
                               self.geomCell.invJacobian[:, :, 1][:, 0] * self.matCell.derivMatrix[:, :,
                                                                          1]).transpose(),
                              self.paramQuad.weights * abs(self.geomCell.detJacobian) *
                              (self.geomCell.invJacobian[:, :, 0][:, 0] * self.matCell.derivMatrix[:, :, 0] +
                               self.geomCell.invJacobian[:, :, 1][:, 0] * self.matCell.derivMatrix[:, :, 1])) + \
                    np.matmul((self.geomCell.invJacobian[:, :, 0][:, 1] * self.matCell.derivMatrix[:, :, 0] +
                               self.geomCell.invJacobian[:, :, 1][:, 1] * self.matCell.derivMatrix[:, :,
                                                                          1]).transpose(),
                              self.paramQuad.weights * abs(self.geomCell.detJacobian) *
                              (self.geomCell.invJacobian[:, :, 0][:, 1] * self.matCell.derivMatrix[:, :, 0] +
                               self.geomCell.invJacobian[:, :, 1][:, 1] * self.matCell.derivMatrix[:, :, 1]))

        return laplacian
