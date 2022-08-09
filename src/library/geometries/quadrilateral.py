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
        @:brief Main constructor for linear quad geometry
        :param pointlabels: Point IDs that make up the cell
        :param points:      Array of all point coordinates parsed in by reference
        :param x:           Array of quadrature zeros xi1-coordinates
        :param y:           Array of quadrature zeros xi2-coordinates
        """
        super().__init__()
        self.A = points[pointlabels[0]]
        self.B = points[pointlabels[1]]
        self.C = points[pointlabels[2]]
        self.D = points[pointlabels[3]]
        self.x = x
        self.y = y

    def parametricMapping(self):
        """
        @:brief Linear mapping between a parametric quad cell and a straight-sided quad cell in real space
        :return: Arrays of x and y-coordinates in physical space given arrays in parametric space
        """
        coords = np.empty((2, self.x.shape[0]), dtype=float)
        for i in range(2):
            coords[i] = self.A[i] * ((1 - self.x) / 2) * ((1 - self.y) / 2) +\
                        self.B[i] * ((1 + self.x) / 2) * ((1 - self.y) / 2) +\
                        self.D[i] * ((1 - self.x) / 2) * ((1 + self.y) / 2) +\
                        self.C[i] * ((1 + self.x) / 2) * ((1 + self.y) / 2)

        return coords

    @property
    def dxdxi1(self):

        dxdxi1 = np.empty((2, self.x.shape[0]), dtype=float)
        for i in range(2):
            dxdxi1[i] = -((self.D[i] - self.C[i] + self.B[i] - self.A[i]) * self.y +
                          self.D[i] - self.C[i] - self.B[i] + self.A[i]) / 4

        return dxdxi1

    @property
    def dxdxi2(self):

        dxdxi2 = np.empty((2, self.x.shape[0]), dtype=float)
        for i in range(2):
            dxdxi2[i] = -((self.D[i] - self.C[i] + self.B[i] - self.A[i]) * self.x -
                          self.D[i] - self.C[i] + self.B[i] + self.A[i]) / 4

        return dxdxi2

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
