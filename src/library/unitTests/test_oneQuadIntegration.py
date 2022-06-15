"""
File: test_oneQuadIntegration.py

Unit test - make sure refactoring and updates don't have any unintended consequence.
        - shows exactly what is broken.

Description: Test if numerical integration routines over a parametric quad cell is correct
"""
import unittest
import numpy as np
from src.library.geometries.quadrilateral import Quadrilateral as Quad
from src.library.paramCells.paramQuad import ParamQuad


def functionInCell(x1, x2):
    """
    @brief: Parse in function to be evaluated in the domain
    :param x1: Coordinates in the x-direction
    :param x2: Coordinates in the y-direction
    :return: Values at specified coordinates given a function
    """
    return x1**12 * x2**14  # np.sin(x1) * np.cos(x2)


def calculateStdIntegration(func, quadrature, p, q):
    """
    @brief: Two-dimensional integration in a standard and local quad region
    :param p:          Polynomial order in the xi_1 direction
    :param q:          Polynomial order in the xi_2 direction
    :param func:       Function to be evaluated
    :param quadrature: Type of Gaussian quadrature selected
    :return: Integrand in parametric space
    """
    paramQuad = ParamQuad(quadrature, p, q)
    quad = Quad(np.array([0, 1, 2, 3]), np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]),
                np.array([i[0] for i in paramQuad.zeros]), np.array([i[1] for i in paramQuad.zeros]))
    coords = quad.parametricMapping()  # points[i][0], points[i][1]
    jacobian = quad.detJacobian()
    integrand = 0

    for i in range(len(paramQuad.weights)):
        integrand += paramQuad.weights[i][0] * func(coords[0][i], coords[1][i]) * abs(jacobian[i])

    print("Number of quadrature points in total = ", len(paramQuad.weights), "\n")
    print("Error = ", abs((4.0 / 195.0) - integrand), "\n")

    return integrand


def calculateLocIntegration(func, quadrature, p, q):
    """
    @brief: Two-dimensional integration in a standard and local quad region
    :param p:          Polynomial order in the xi_1 direction
    :param q:          Polynomial order in the xi_2 direction
    :param func:       Function to be evaluated
    :param quadrature: Type of Gaussian quadrature selected
    :return: Integrand in real space
    """
    paramQuad = ParamQuad(quadrature, p, q)
    quad = Quad(np.array([0, 1, 2, 3]), np.array([[0, -1], [1, -1], [1, 1], [0, 0]]),
                np.array([i[0] for i in paramQuad.zeros]), np.array([i[1] for i in paramQuad.zeros]))
    coords = quad.parametricMapping()
    jacobian = quad.detJacobian()
    integrand = 0

    for i in range(len(paramQuad.weights)):
        integrand += paramQuad.weights[i][0] * func(coords[0][i], coords[1][i]) * abs(jacobian[i])

    print("Number of quadrature points in total = ", len(paramQuad.weights), "\n")
    print("Error = ", abs(1.0/195.0 + 1.0/420.0 - integrand), "\n")

    return integrand


class TestCase(unittest.TestCase):

    def test_GLL_integration(self):
        # self.assertEqual(True, False)  # add assertion here
        self.assertEqual("{:.8f}".format(abs((4.0/195.0) - calculateStdIntegration(functionInCell, "GLL", 9, 11))),
                         "0.00178972")

    def test_GLL_exact(self):
        self.assertTrue(abs((4.0/195.0) - calculateStdIntegration(functionInCell, "GLL", 13, 15))
                        <= np.finfo(np.float64).eps)

    def test_GLL_local(self):
        self.assertEqual("{:.9f}".format(abs(1.0/195.0 + 1.0/420.0 -
                                             calculateLocIntegration(functionInCell, "GLL", 13, 15))), "0.000424657")


if __name__ == '__main__':
    unittest.main()

# Naive implementation:
# for i in range(len(weightsSeg1)):
#
#     temp = 0
#
#     for j in range(len(weightsSeg2)):
#         temp += weightsSeg2[j][0] * (pointsSeg1[i][0]**12) * (pointsSeg2[j][0]**14)
#
#     answer += weightsSeg1[i][0] * temp
