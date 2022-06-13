"""
File: segment.py

Description: Segment shape subclass. All data held constant
"""
from src.library.geometries.geometry import Geometry


class Segment(Geometry):
    """
    @:brief Segment geometry class to evaluate transformation and Jacobian with methods defined
    """
    def __init__(self, pointlabels, points, x):
        """
        @:brief Main constructor. A and B are the first and second points of the geometry
        :param pointlabels: Point IDs that make up the cell
        :param points:      Point coordinates from dgMesh class parsed in by reference
        :param x:           Arbitrary zeros coordinate in the x-direction
        """
        super().__init__()
        self.x1A = points[pointlabels[0]][0]
        self.x1B = points[pointlabels[1]][0]
        self.x = x

    def parametricMapping(self):
        """
        @:brief Linear mapping for a line
        :return: x-coordinate in physical space corresponding to xi_1 inputs
        """
        x1 = self.x1A * ((1 - self.x) / 2) + self.x1B * ((1 + self.x) / 2)

        return x1

    @property
    def dx1dxi1(self):
        """
        @:brief dx/dxi derivative based on linear map above
        :return: dx/dxi
        """
        return (self.x1B - self.x1A) / 2

    @property
    def dxi1dx1(self):

        return 1 / self.dx1dxi1

    def detJacobian(self):
        """
        @:brief Jacobian getter routines
        :return: 1D Jacobian based on linear mapping
        """
        return self.dx1dxi1
