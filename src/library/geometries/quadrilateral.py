"""
File: quadrilateral.py

Description: Quadrilateral shape derived class containing constant data
"""
import numpy as np

from src.library.geometries.geometry import Geometry


class Quadrilateral(Geometry):
    """
    @:brief Quadrilateral geometry class to evaluate transformation and Jacobian with methods defined
    """
    xmin = [0, 1]
    xmax = [2, 3]
    ymin = [3, 0]
    ymax = [1, 2]

    def __init__(self, pointlabels, points, x, y):
        """
        @:brief Main constructor for quad geometry
        :param pointlabels: Point IDs that make up the cell
        :param points:      Array of all point coordinates parsed in by reference
        :param x:           Array of quadrature zeros x-coordinates
        :param y:           Array of quadrature zeros y-coordinates
        """
        super().__init__()
        # self.x1A = points[pointlabels[0]][0]
        # self.x2A = points[pointlabels[0]][1]
        # self.x1B = points[pointlabels[1]][0]
        # self.x2B = points[pointlabels[1]][1]
        # self.x1C = points[pointlabels[2]][0]
        # self.x2C = points[pointlabels[2]][1]
        # self.x1D = points[pointlabels[3]][0]
        # self.x2D = points[pointlabels[3]][1]
        self.A = points[pointlabels[0]]
        self.B = points[pointlabels[1]]
        self.C = points[pointlabels[2]]
        self.D = points[pointlabels[3]]
        self.x = x
        self.y = y

    def parametricMapping(self):
        """
        @:brief Linear mapping between a parametric quad cell and a straight sided quad cell in real space
        :return: Arrays of x and y-coordinates in physical space given arrays in parametric space
        """
        coords = np.empty((2, self.x.shape[0]), dtype=float)
        for i in range(2):
            coords[i] = self.A[i] * ((1 - self.x) / 2) * ((1 - self.y) / 2) +\
                        self.B[i] * ((1 + self.x) / 2) * ((1 - self.y) / 2) +\
                        self.D[i] * ((1 - self.x) / 2) * ((1 + self.y) / 2) +\
                        self.C[i] * ((1 + self.x) / 2) * ((1 + self.y) / 2)

        return coords

    # def dx1dxi1(self):
    #     A = self.x1A
    #     B = self.x1B
    #     C = self.x1C
    #     D = self.x1D
    #
    #     return - ((D - C + B - A) * self.y + D - C - B + A) / 4

    # def dx2dxi2(self):
    #     A = self.x2A
    #     B = self.x2B
    #     C = self.x2C
    #     D = self.x2D
    #
    #     return - ((D - C + B - A) * self.x - D - C + B + A) / 4
    #
    # def dx1dxi2(self):
    #     A = self.x1A
    #     B = self.x1B
    #     C = self.x1C
    #     D = self.x1D
    #
    #     return - ((D - C + B - A) * self.x - D - C + B + A) / 4

    @property
    def dxdxi2(self):

        dxdxi2 = np.empty((2, self.x.shape[0]), dtype=float)
        for i in range(2):
            dxdxi2[i] = -((self.D[i] - self.C[i] + self.B[i] - self.A[i]) * self.x -
                          self.D[i] - self.C[i] + self.B[i] + self.A[i]) / 4

        return dxdxi2

    # def dx2dxi1(self):
    #     A = self.x2A
    #     B = self.x2B
    #     C = self.x2C
    #     D = self.x2D
    #
    #     return - ((D - C + B - A) * self.y + D - C - B + A) / 4

    @property
    def dxdxi1(self):

        dxdxi1 = np.empty((2, self.x.shape[0]), dtype=float)
        for i in range(2):
            dxdxi1[i] = -((self.D[i] - self.C[i] + self.B[i] - self.A[i]) * self.y +
                          self.D[i] - self.C[i] - self.B[i] + self.A[i]) / 4

        return dxdxi1

    @property
    def jacobianMatrix(self):
        """
        @:brief Jacobian calculation routines
        :return: 2D Jacobian based on linear mapping above
        """
        return np.array([[self.dxdxi1.transpose()[i], self.dxdxi2.transpose()[i]] for i in range(len(self.dxdxi1[0]))])

    def invJacobianMatrix(self):
        # j = np.array([self.dxdxi1.reshape(-1), self.dxdxi2.reshape(-1)]).transpose()

        return np.linalg.inv(self.jacobianMatrix)

    def detJacobian(self):
        # print(self.dxdxi1[0] * self.dxdxi2[1] - self.dxdxi2[0] * self.dxdxi1[1])

        return np.linalg.det(self.jacobianMatrix)

    # def GetGradients(self, qPoints):
    #
    #     p = self.p
    #     nBasis = (p + 1)**2
    #     nPoints = qPoints.shape[0]
    #
    #     basisGrad = np.zeros([nPoints, nBasis, 2])
    #
    #     if p > 0:
    #         nodeCoords = equidistant_nodes_1d(-1.0, 1.0, p + 1)
    #         GetLagrange2d(qPoints, nodeCoords, basisGrad=basisGrad)
    #
    #     return basisGrad  # [nq, nb, ndims]
    #
    # def CalculateJacobian(self, qPoints, points):
    #     basisGrad = self.GetGradients(qPoints)
    #
    #     print(points.transpose())
    #
    #     # for i in range(len(points)):
    #     #     points[i] = equidistant_nodes_1d(points[i][0], points[i][1], self.p + 1)
    #
    #     jac = np.tensordot(basisGrad, points.transpose(), axes=[[1], [1]]).transpose((0, 2, 1))
    #
    #     return jac
