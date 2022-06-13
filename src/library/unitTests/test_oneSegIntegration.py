"""
File: test_oneSegIntegration.py

Description: Test if numerical integration routines on a parametric segment is correct
"""
import unittest
import numpy as np
from src.library.geometries.segment import Segment
from src.library.paramCells.paramSeg import ParamSeg


def functionInCell(x1):
    """
    @brief: Parse in function to be evaluated in the domain
    :param x1: Coordinates in the x-direction
    :return: Values at specified coordinates given a function
    """
    return x1**12  # np.sin(x1)


def calculateLineIntegration(func, quadrature, p):
    """
    @brief: One-dimensional integration in a parametric line
    :param func:       Function to be evaluated
    :param quadrature: Type of Gaussian quadrature
    :param p:          Polynomial order
    :return: Integrand
    """
    paramSeg = ParamSeg(quadrature, p)
    seg = Segment(np.array([0, 1]), np.array([[-1], [1]]), np.array([i[0] for i in paramSeg.zeros]))
    x1 = seg.parametricMapping()  # (points[i][0])
    jacobian = seg.calculateJacobian()
    integrand = 0

    for i in range(len(paramSeg.weights)):
        integrand += paramSeg.weights[i][0] * func(x1[i]) * abs(jacobian)

    print("Number of quadrature points = ", len(paramSeg.weights), "\n")
    print("Error = ", (2.0/13.0) - integrand, "\n")

    return integrand


class TestCase(unittest.TestCase):

    def test_GL_integration(self):
        # self.assertEqual(True, False)  # add assertion here
        self.assertTrue(-np.finfo(np.float32).eps <= (2.0/13.0) - calculateLineIntegration(functionInCell, "GL", 12)
                        <= np.finfo(np.float32).eps)

    def test_GLL_integration(self):
        self.assertTrue(-np.finfo(np.float64).eps <= (2.0/13.0) - calculateLineIntegration(functionInCell, "GLL", 12)
                        <= np.finfo(np.float64).eps)


if __name__ == '__main__':
    unittest.main()


# https://stackoverflow.com/questions/455612/limiting-floats-to-two-decimal-points
# print("%.6f" % answer)
# print("{:.6f}".format(abs((2 / 13) - answer)))
