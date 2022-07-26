"""
File: ADR.py

Description: Solve ADR equations in 1D
"""
import numpy as np
import math
from src.library.dgMesh.dgMesh import *
from src.library.paramCells.basis import *
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator


def computeDtForBurgers(meshObj, dx, ncells, ncoords, cfl):
    """
    @:brief Calculate time-step size for Burgers' equation with CFL constraints
    :param ncoords:   Number of quadrature points in a cell
    :param meshObj:   Mesh object to obtain physical scalar values in each cell
    :param dx:        Mesh size
    :param ncells:    Number of cells
    :param cfl:       Courant number
    :return: Time-step size
    """

    smax = -1.0e6  # a big negative number
    for cell in range(ncells):
        for quadrature in range(ncoords):
            value = abs(meshObj.connectivityData.cells[cell].solnCell.uPhysical[quadrature][0])
            if value > smax:
                smax = value

    return cfl * dx / smax  # make sure dt addition does not exceed final time output


def upwind_flux(qleft, qright, v, n):
    """
    @:brief Upwind flux of a scalar advection
    :param qleft:  Left state value
    :param qright: Right state value
    :param v:      Constant advection velocity
    :param n:      Normal vector
    :return: Upwind flux value
    """
    if qright * n >= 0:
        return 0.5 * qleft * qleft * n
    elif qright * n < 0:
        return 0.5 * qright * qright * n
    else:
        raise NotImplementedError

    # if v * n >= 0:
    #     return v * qleft * n
    # elif v * n < 0:
    #     return v * qright * n
    # else:
    #     raise NotImplementedError


def lax_friedrichs_flux(qleft, qright, dx, dt, left_flux, right_flux, n):
    return 0.5 * (dx / dt) * (qleft - qright) + 0.5 * (left_flux + right_flux) * n


def forwardTransform(meshObj, physicalValues, cell):
    return np.matmul(meshObj.connectivityData.cells[cell].invMassMatrix,
                     np.matmul(meshObj.connectivityData.cells[cell].matCell.basisMatrix.transpose(),
                               meshObj.connectivityData.cells[cell].paramSeg.weights *
                               abs(meshObj.connectivityData.cells[cell].geomCell.detJacobian) *
                               physicalValues))


def linearAdvFlux(speed, q):
    return speed * q


def burgersFlux(speed, q):
    return 0.5 * q * q


def minmod(prev, current, next, cell):

    p_counter = P1
    for var in range(nVars):
        for p in range(P1, 0, -1):
            coeff_tilde = 0
            temp_a = current[p][var]
            temp_b = next[p - 1][var] - current[p - 1][var]
            temp_c = current[p - 1][var] - prev[p - 1][var]
            # sign = np.where(np.logical_and(np.sign(temp_a) == np.sign(temp_b), np.sign(temp_b) == np.sign(temp_c)))[0]
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

    name = "/Users/jwtan/PycharmProjects/PyDG/polyMesh/64x0"
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
    test_case = "burgers_sine"
    a = 1.0  # constant velocity
    numerical_flux = upwind_flux

    if test_case == "sine":
        endTime = 0.0
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
        endTime = 0.5
        flux = linearAdvFlux
        boundaryConditions = "Periodic"

        for i in range(nCells):
            # Initial condition for step function
            mesh.connectivityData.cells[i].solnCell.uCoeffs[0] = np.array(
                [uR if xL <= coords <= xR else uL for coords in mesh.connectivityData.cells[i].calculations.cellCentre])
            # Set physical values, no need * abs(mesh.connectivityData.cells[i].geomCell.detJacobian)
            mesh.connectivityData.cells[i].solnCell.uPhysical = np.matmul(
                mesh.connectivityData.cells[i].matCell.basisMatrix, mesh.connectivityData.cells[i].solnCell.uCoeffs)

    elif test_case == "burgers_step":
        endTime = 0.1
        flux = burgersFlux
        boundaryConditions = "Neumann"

        for i in range(nCells):
            # Initial condition for Burgers' equation
            mesh.connectivityData.cells[i].solnCell.uCoeffs[0] = np.array(
                [2 if coords < 0.5 else 1 for coords in mesh.connectivityData.cells[i].calculations.cellCentre])
            mesh.connectivityData.cells[i].solnCell.uPhysical = np.matmul(
                mesh.connectivityData.cells[i].matCell.basisMatrix, mesh.connectivityData.cells[i].solnCell.uCoeffs)

    elif test_case == "burgers_sine":
        endTime = 1.0
        flux = burgersFlux
        boundaryConditions = "Periodic"

        for i in range(nCells):
            for coords in range(nCoords):
                mesh.connectivityData.cells[i].solnCell.uPhysical[coords] = \
                    0.5 + 0.5 * np.sin(np.pi * mesh.connectivityData.cells[i].GetQuadratureCoords[coords])

            mesh.connectivityData.cells[i].solnCell.uCoeffs = \
                forwardTransform(mesh, mesh.connectivityData.cells[i].solnCell.uPhysical, i)

    else:
        raise NotImplementedError

    # print("Initial coefficients: ")
    # for i in range(nCells):
    #     print(mesh.connectivityData.cells[i].solnCell.uCoeffs)
    #
    # print("Initial physical values: ")
    # for i in range(nCells):
    #     print(mesh.connectivityData.cells[i].solnCell.uPhysical)

    CFL = 0.1
    # CFL = 1 / (2 * P1 + 1)
    deltaT = 0.0
    # Constant mesh size for now
    deltax = mesh.connectivityData.cells[0].calculations.V

    """ Start time-loop """
    while endTime - time > 1e-10:
        """ Populate the faces values """
        for i in range(nCells):
            leftFaceValueArray[i] = np.matmul(basisMatrixforF0, mesh.connectivityData.cells[i].solnCell.uCoeffs)[0]
            rightFaceValueArray[i] = np.matmul(basisMatrixforF1, mesh.connectivityData.cells[i].solnCell.uCoeffs)[0]

        """ Calculate time-step size """
        deltaT = computeDtForBurgers(mesh, deltax, nCells, nCoords, CFL)
        # if flux == burgersFlux:
        #     deltaT = computeDtForBurgers(mesh, deltax, nCells, nCoords, CFL)
        # elif flux == linearAdvFlux:
        #     deltaT = CFL * deltax / a
        # else:
        #     raise NotImplementedError

        # Increment time
        if time + deltaT > endTime:
            deltaT = endTime - time
        time += deltaT
        print("Current time: " + str(time))

        if boundaryConditions == "Periodic":
            if numerical_flux == upwind_flux:
                """ Set periodic boundary conditions, i.e., copy end cell values to the other end """
                numericalFluxArray[0][0] = upwind_flux(leftFaceValueArray[0], rightFaceValueArray[-1], a,
                                                       mesh.connectivityData.cells[0].calculations.faceNormals[0])
                numericalFluxArray[0][1] = upwind_flux(rightFaceValueArray[0], leftFaceValueArray[1], a,
                                                       mesh.connectivityData.cells[0].calculations.faceNormals[1])
                numericalFluxArray[-1][0] = upwind_flux(leftFaceValueArray[-1], rightFaceValueArray[-2], a,
                                                        mesh.connectivityData.cells[0].calculations.faceNormals[0])
                numericalFluxArray[-1][1] = upwind_flux(rightFaceValueArray[-1], leftFaceValueArray[0], a,
                                                        mesh.connectivityData.cells[0].calculations.faceNormals[1])
            elif numerical_flux == lax_friedrichs_flux:
                """ Set Periodic boundary conditions, Lax Friedrichs flux """
                numericalFluxArray[0][0] = lax_friedrichs_flux(leftFaceValueArray[0], rightFaceValueArray[-1], deltax,
                                                               deltaT,
                                                               flux(a, leftFaceValueArray[0]),
                                                               flux(a, rightFaceValueArray[-1]),
                                                               mesh.connectivityData.cells[0].calculations.faceNormals[
                                                                   0])
                numericalFluxArray[0][1] = lax_friedrichs_flux(rightFaceValueArray[0], leftFaceValueArray[1], deltax,
                                                               deltaT,
                                                               flux(a, rightFaceValueArray[0]),
                                                               flux(a, leftFaceValueArray[1]),
                                                               mesh.connectivityData.cells[0].calculations.faceNormals[
                                                                   1])
                numericalFluxArray[-1][0] = lax_friedrichs_flux(leftFaceValueArray[-1], rightFaceValueArray[-2], deltax,
                                                                deltaT,
                                                                flux(a, leftFaceValueArray[-1]),
                                                                flux(a, rightFaceValueArray[-2]),
                                                                mesh.connectivityData.cells[-1].calculations.faceNormals[
                                                                    0])
                numericalFluxArray[-1][1] = lax_friedrichs_flux(rightFaceValueArray[-1], leftFaceValueArray[0], deltax,
                                                                deltaT,
                                                                flux(a, rightFaceValueArray[-1]),
                                                                flux(a, leftFaceValueArray[0]),
                                                                mesh.connectivityData.cells[-1].calculations.faceNormals[
                                                                    1])
            else:
                raise NotImplementedError

        elif boundaryConditions == "Neumann":
            """ Set Neumann boundary conditions, Lax Friedrichs flux """
            numericalFluxArray[0][0] = lax_friedrichs_flux(leftFaceValueArray[0], leftFaceValueArray[0], deltax,
                                                           deltaT,
                                                           flux(a, leftFaceValueArray[0]),
                                                           flux(a, leftFaceValueArray[0]),
                                                           mesh.connectivityData.cells[0].calculations.faceNormals[0]
                                                           )
            numericalFluxArray[0][1] = lax_friedrichs_flux(rightFaceValueArray[0], leftFaceValueArray[1], deltax,
                                                           deltaT,
                                                           flux(a, rightFaceValueArray[0]),
                                                           flux(a, leftFaceValueArray[1]),
                                                           mesh.connectivityData.cells[0].calculations.faceNormals[1]
                                                           )
            numericalFluxArray[-1][0] = lax_friedrichs_flux(leftFaceValueArray[-1], rightFaceValueArray[-2], deltax,
                                                            deltaT,
                                                            flux(a, leftFaceValueArray[-1]),
                                                            flux(a, rightFaceValueArray[-2]),
                                                            mesh.connectivityData.cells[-1].calculations.faceNormals[0]
                                                            )
            numericalFluxArray[-1][1] = lax_friedrichs_flux(rightFaceValueArray[-1], rightFaceValueArray[-1], deltax,
                                                            deltaT,
                                                            flux(a, rightFaceValueArray[-1]),
                                                            flux(a, rightFaceValueArray[-1]),
                                                            mesh.connectivityData.cells[-1].calculations.faceNormals[1]
                                                            )
        else:
            raise NotImplementedError

        """ Calculate numerical fluxes for internal faces, qleft is always on the left side of a face instead """
        if numerical_flux == upwind_flux:
            for i in range(nCells - 2):
                numericalFluxArray[i + 1][0] = upwind_flux(leftFaceValueArray[i + 1], rightFaceValueArray[i], a,
                                                           mesh.connectivityData.cells[0].calculations.faceNormals[0])
                numericalFluxArray[i + 1][1] = upwind_flux(rightFaceValueArray[i + 1], leftFaceValueArray[i + 2], a,
                                                           mesh.connectivityData.cells[0].calculations.faceNormals[1])
        elif numerical_flux == lax_friedrichs_flux:
            for i in range(nCells - 2):
                numericalFluxArray[i + 1][0] = lax_friedrichs_flux(leftFaceValueArray[i + 1], rightFaceValueArray[i],
                                                                   deltax, deltaT, flux(a, leftFaceValueArray[i + 1]),
                                                                   flux(a, rightFaceValueArray[i]),
                                                                   mesh.connectivityData.cells[
                                                                       i + 1].calculations.faceNormals[0])
                numericalFluxArray[i + 1][1] = lax_friedrichs_flux(rightFaceValueArray[i + 1],
                                                                   leftFaceValueArray[i + 2],
                                                                   deltax, deltaT, flux(a, rightFaceValueArray[i + 1]),
                                                                   flux(a, leftFaceValueArray[i + 2]),
                                                                   mesh.connectivityData.cells[
                                                                        i + 1].calculations.faceNormals[1])
        else:
            raise NotImplementedError

        """ Initialise new solution coefficients """
        uCoeffsNew = np.zeros((nCells, P1 + 1, nVars))

        """ Go through all cells """
        for i in range(nCells):
            """ Backward transformation from coefficient space to physical space """
            mesh.connectivityData.cells[i].solnCell.uPhysical = np.matmul(
                mesh.connectivityData.cells[i].matCell.basisMatrix, mesh.connectivityData.cells[i].solnCell.uCoeffs)
            """ Flux values at quadrature points in the cell """
            fluxValues = np.array([flux(a, q) for q in mesh.connectivityData.cells[i].solnCell.uPhysical])
            """ Flux coefficients vector in coefficient space, f hat = (M)^-1 B^T W f """
            fluxCoeffs = forwardTransform(mesh, fluxValues, i)
            # f0Coeffs = np.matmul(np.linalg.inv(mesh.connectivityData.cells[i].GetMassMatrix()),
            #                      np.matmul(mesh.connectivityData.cells[i].matCell.basisMatrix.transpose(),
            #                                mesh.connectivityData.cells[i].paramSeg.weights *
            #                                abs(mesh.connectivityData.cells[i].geomCell.detJacobian) *
            #                                numericalFluxArray[i][0].reshape(-1, 1).transpose()))
            # f1Coeffs = np.matmul(np.linalg.inv(mesh.connectivityData.cells[i].GetMassMatrix()),
            #                      np.matmul(mesh.connectivityData.cells[i].matCell.basisMatrix.transpose(),
            #                                mesh.connectivityData.cells[i].paramSeg.weights *
            #                                abs(mesh.connectivityData.cells[i].geomCell.detJacobian) *
            #                                numericalFluxArray[i][1].reshape(-1, 1).transpose()))
            # divFlux = np.matmul(np.linalg.inv(mesh.connectivityData.cells[i].GetMassMatrix()),
            #                     (np.matmul(massMatrixforF0, f0Coeffs) +
            #                      np.matmul(massMatrixforF1, f1Coeffs) -
            #                      np.matmul(mesh.connectivityData.cells[i].GetStiffnessMatrix(), fluxCoeffs)))
            divFlux = np.matmul(mesh.connectivityData.cells[i].invMassMatrix,
                                (np.matmul(basisMatrixforF0.transpose(),
                                           numericalFluxArray[i][0].reshape(-1, 1).transpose()) +
                                 np.matmul(basisMatrixforF1.transpose(),
                                           numericalFluxArray[i][1].reshape(-1, 1).transpose()) -
                                 np.matmul(mesh.connectivityData.cells[i].stiffnessMatrix, fluxCoeffs)))
            uCoeffsNew[i] = mesh.connectivityData.cells[i].solnCell.uCoeffs - deltaT * divFlux

        """ c-1, c, c+1, periodic """
        minmod(uCoeffsNew[-1], uCoeffsNew[0], uCoeffsNew[1], mesh.connectivityData.cells[0].solnCell.uCoeffs)
        minmod(uCoeffsNew[-2], uCoeffsNew[-1], uCoeffsNew[0], mesh.connectivityData.cells[-1].solnCell.uCoeffs)
        for i in range(nCells - 2):
            minmod(uCoeffsNew[i], uCoeffsNew[i + 1], uCoeffsNew[i + 2],
                   mesh.connectivityData.cells[i + 1].solnCell.uCoeffs)

        """ No limiter solution """
        # for i in range(nCells):
        #     mesh.connectivityData.cells[i].solnCell.uCoeffs = uCoeffsNew[i]

    """ Plot """
    pltPhysical = np.zeros((nCells, nCoords, nVars))
    for i in range(nCells):
        pltPhysical[i] = np.matmul(mesh.connectivityData.cells[i].matCell.basisMatrix,
                                   mesh.connectivityData.cells[i].solnCell.uCoeffs)

    pltCoords = np.array([mesh.connectivityData.cells[i].GetQuadratureCoords for i in range(nCells)]).reshape(-1)
    plt.plot(pltCoords, pltPhysical[:, :, 0].reshape(-1))
    # plt.rc('axes', linewidth=1.25)
    # plt.xticks([-1, 0, 1])
    # plt.tick_params(axis='x', direction='in', pad=5)
    # plt.yticks([-1, 0, 1])
    # plt.tick_params(axis='y', direction='in', pad=5)
    plt.grid()
    plt_name = test_case + "_" + "P" + str(P1) + "_" + str(endTime)
    plt.savefig(plt_name + '_limited.png', dpi=100)
    # np.savetxt(plt_name + "_coords.dat", pltCoords, delimiter=',')
    # np.savetxt(plt_name + "_values.dat", pltPhysical[:, :, 0].reshape(-1), delimiter=',')
    plt.show()
