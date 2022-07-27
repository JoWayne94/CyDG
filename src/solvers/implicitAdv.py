"""
File: implicitAdv.py

Description: Solve linear advection implicitly in 1D
"""
import numpy as np
import math
from src.library.dgMesh.dgMesh import *
from src.library.paramCells.basis import *
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator


def upwind_flux(qleft, qright, v, n):
    """
    @:brief Upwind flux of a scalar advection
    :param qleft:  Left state value
    :param qright: Right state value
    :param v:      Constant advection velocity
    :param n:      Normal vector
    :return: Upwind flux value
    """

    if v * n >= 0:
        return v * qleft * n
    elif v * n < 0:
        return v * qright * n
    else:
        raise NotImplementedError


def forwardTransform(meshObj, physicalValues, cell):
    return np.matmul(meshObj.connectivityData.cells[cell].invMassMatrix,
                     np.matmul(meshObj.connectivityData.cells[cell].matCell.basisMatrix.transpose(),
                               meshObj.connectivityData.cells[cell].paramSeg.weights *
                               abs(meshObj.connectivityData.cells[cell].geomCell.detJacobian) *
                               physicalValues))


def linearAdvFlux(speed, q):
    return speed * q


def minmod(prev, current, next, cell):

    p_counter = P1
    for var in range(nVars):
        for p in range(P1, 0, -1):
            coeff_tilde = 0
            temp_a = current[p][var]
            temp_b = next[p - 1][var] - current[p - 1][var]
            temp_c = current[p - 1][var] - prev[p - 1][var]
            if np.sign(temp_a) == np.sign(temp_b) and np.sign(temp_b) == np.sign(temp_c):
                coeff_tilde = np.sign(temp_a) * min(abs(temp_a), min(abs(temp_b), abs(temp_c)))

            if abs(temp_a - coeff_tilde) < 1.0e-6:
                break
            else:
                cell[p][var] = coeff_tilde
                p_counter -= 1

        for remaining in range(p_counter, -1, -1):
            cell[remaining][var] = current[remaining][var]


if __name__ == '__main__':
    """
    main()
    """

    name = "/Users/jwtan/PycharmProjects/PyDG/polyMesh/100x0"
    nDims = 1
    nVars = 1
    # Uniform polynomial orders in the x and y-directions for now
    P1 = 3
    P2 = 0
    # Read in the mesh
    mesh = DgMesh.constructFromPolyMeshFolder(name, nDims)
    mesh.constructShapeBasedCells(name, nVars, P1, P2)

    """
    Prototype time-loop with forward Euler time-stepping
    """

    """ Set test case """
    time = 0.0
    nCells = len(mesh.connectivityData.cells)
    quadCoords = mesh.connectivityData.cells[0].GetQuadratureCoords

    """ Number of quadrature points within a single cell """
    nCoords = len(quadCoords)

    """ Numerical flux array [Number of cells in mesh, left and right numerical fluxes, number of variables]"""
    numericalFluxArray = np.empty((nCells, 2, nVars))

    """ Left and right face variable values extrapolated using basis matrices at -1 and 1 in parametric space """
    basisMatrixforF0 = Legendre1d(np.array([[-1]]), P1)
    basisMatrixforF1 = Legendre1d(np.array([[1]]), P1)
    leftFaceValueArray = np.empty((nCells, nVars))
    rightFaceValueArray = np.empty((nCells, nVars))

    """
    Set initial values using the first coefficients for constant state, or using physical values then 
    transform back to coefficient space
    """
    test_case = "sine"
    a = 1.0  # constant velocity
    numerical_flux = upwind_flux

    if test_case == "sine":
        endTime = 1.0
        flux = linearAdvFlux
        boundaryConditions = "Periodic"

        for i in range(nCells):
            # Initial condition for sine wave
            for coords in range(nCoords):
                mesh.connectivityData.cells[i].solnCell.uPhysical[coords] = \
                    np.sin(2 * np.pi * mesh.connectivityData.cells[i].GetQuadratureCoords[coords])

            mesh.connectivityData.cells[i].solnCell.uCoeffs = \
                forwardTransform(mesh, mesh.connectivityData.cells[i].solnCell.uPhysical, i)

    elif test_case == "step":
        uL = 0.0
        uR = 1.0
        xL = 0.25
        xR = 0.75
        endTime = 1.0
        flux = linearAdvFlux
        boundaryConditions = "Periodic"

        for i in range(nCells):
            # Initial condition for step function
            mesh.connectivityData.cells[i].solnCell.uCoeffs[0] = np.array(
                [uR if xL <= coords <= xR else uL for coords in mesh.connectivityData.cells[i].calculations.cellCentre])
            # Set physical values, no need * abs(mesh.connectivityData.cells[i].geomCell.detJacobian)
            mesh.connectivityData.cells[i].solnCell.uPhysical = np.matmul(
                mesh.connectivityData.cells[i].matCell.basisMatrix, mesh.connectivityData.cells[i].solnCell.uCoeffs)

    else:
        raise NotImplementedError

    pltPhysical = np.zeros((nCells, nCoords, nVars))
    for i in range(nCells):
        pltPhysical[i] = mesh.connectivityData.cells[i].solnCell.uPhysical

    pltCoords = np.array([mesh.connectivityData.cells[i].GetQuadratureCoords for i in range(nCells)]).reshape(-1)
    plt.plot(pltCoords, pltPhysical[:, :, 0].reshape(-1))

    CFL = 0.01
    # Constant mesh size for now
    deltax = mesh.connectivityData.cells[0].calculations.V

    """ Global diagonal upwind matrix """
    globalDiag = np.zeros((nCells, nCells, P1 + 1, P1 + 1))

    """ Global off-diagonal matrix """
    globalOffDiag = np.zeros_like(globalDiag)

    """ Global mass matrix """
    massMatrices = np.zeros_like(globalDiag)

    """ Global stiffness matrix """
    stiffnessMatrices = np.zeros_like(globalDiag)

    u = np.block([
        [mesh.connectivityData.cells[i].solnCell.uCoeffs] for i in range(nCells)
    ])

    uCoeffsGlobal = np.empty_like(u)

    """ Go through all cells """
    for i in range(nCells):

        massMatrices[i][i] = mesh.connectivityData.cells[i].massMatrix
        stiffnessMatrices[i][i] = -a * mesh.connectivityData.cells[i].stiffnessMatrix

        """ Upwind """
        # globalDiag[i][i] = a * np.matmul(basisMatrixforF1.transpose(), basisMatrixforF1)

        """ Central flux """
        globalDiag[i][i] = (a / 2) * (np.matmul(basisMatrixforF1.transpose(), basisMatrixforF1) -
                                      np.matmul(basisMatrixforF0.transpose(), basisMatrixforF0))

    """ Periodic BC contribution """
    """ Upwind """
    # globalOffDiag[-1][0] = -a * np.matmul(basisMatrixforF0.transpose(), basisMatrixforF1)

    """ Central flux """
    globalOffDiag[-1][0] = -(a / 2) * np.matmul(basisMatrixforF0.transpose(), basisMatrixforF1)
    globalOffDiag[0][-1] = (a / 2) * np.matmul(basisMatrixforF1.transpose(), basisMatrixforF0)

    """ Off-diagonal block coefficients """
    for i in range(nCells - 1):
        """ Upwind """
        # globalOffDiag[i][i + 1] = -a * np.matmul(basisMatrixforF0.transpose(), basisMatrixforF1)

        """ Central flux """
        globalOffDiag[i][i + 1] = -(a / 2) * np.matmul(basisMatrixforF0.transpose(), basisMatrixforF1)

    for i in range(nCells - 1):
        globalOffDiag[i + 1][i] = (a / 2) * np.matmul(basisMatrixforF1.transpose(), basisMatrixforF0)

    massMatrices = np.block([
        [massMatrices[j][i] for j in range(nCells)] for i in range(nCells)
    ])
    globalDiag = np.block([
        [globalDiag[j][i] for j in range(nCells)] for i in range(nCells)
    ])
    stiffnessMatrices = np.block([
        [stiffnessMatrices[j][i] for j in range(nCells)] for i in range(nCells)
    ])
    globalOffDiag = np.block([
        [globalOffDiag[j][i] for j in range(nCells)] for i in range(nCells)
    ])

    """ Calculate time-step size """
    deltaT = CFL * deltax / a
    globalMatrix = (massMatrices / deltaT) + globalDiag + stiffnessMatrices + globalOffDiag
    invGlobalMatrix = np.linalg.inv(globalMatrix)

    """ Start time-loop """
    while endTime - time > 1e-10:

        # Increment time
        if time + deltaT > endTime:
            deltaT = endTime - time
        time += deltaT
        print("Current time: " + str(time))

        """ Initialise new solution coefficients """
        # uCoeffsNew = np.zeros((nCells, P1 + 1, nVars))

        RHS = np.matmul((massMatrices / deltaT), u)

        uCoeffsGlobal = np.matmul(invGlobalMatrix, RHS)

        u = uCoeffsGlobal

        """ No limiter solution """
        # for i in range(nCells):
        #     mesh.connectivityData.cells[i].solnCell.uCoeffs = uCoeffsNew[i]

    """ Plot """
    for i in range(nCells):
        pltPhysical[i] = np.matmul(mesh.connectivityData.cells[i].matCell.basisMatrix,
                                   uCoeffsGlobal.reshape(nCells, P1 + 1)[i].reshape(-1, 1))

    plt.plot(pltCoords, pltPhysical[:, :, 0].reshape(-1), linestyle="", marker="o", label=str(time), markersize=2)
    plt.grid()
    plt_name = test_case + "_" + "P" + str(P1) + "_" + str(endTime)
    # plt.savefig(plt_name + '_limited.png', dpi=100)
    # np.savetxt(plt_name + "_coords.dat", pltCoords, delimiter=',')
    # np.savetxt(plt_name + "_values.dat", pltPhysical[:, :, 0].reshape(-1), delimiter=',')
    plt.show()
