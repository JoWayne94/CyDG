"""
File: explicitAdv.py

Description: Solve the scalar advection equation in 1D with forward Euler time-stepping
"""
import numpy as np
import math
from src.library.dgMesh.dgMesh import *
from src.library.paramCells.basis import *
from src.utilities.functions import *


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


def upwind_flux(qleft, qright, v, n, left_flux, right_flux):
    """
    @:brief Upwind flux of a scalar advection
    :param qleft:      Left state value
    :param qright:     Right state value
    :param v:          Constant advection velocity
    :param n:          Normal vector
    :param right_flux: Right state flux value
    :param left_flux:  Left state flux value
    :return: Upwind flux value
    """
    # if qright * n >= 0:
    #     return left_flux * n
    # elif qright * n < 0:
    #     return right_flux * n
    # else:
    #     raise NotImplementedError

    if v * n >= 0:
        return left_flux * n
    elif v * n < 0:
        return right_flux * n
    else:
        raise NotImplementedError


def central_flux(qleft, qright, v, n, left_flux, right_flux):
    return v * 0.5 * (qleft + qright) * n


def local_lax_friedrichs_flux(qleft, qright, v, n, left_flux, right_flux):
    """
    @:brief Local Lax-Friedrichs flux, only for linear advection
    :param qleft:       Left state value
    :param qright:      Right state value
    :param v:           Constant advection speed, or dx / dt
    :param left_flux:   Left state flux value
    :param right_flux:  Right state flux value
    :param n:           Unit normal vector
    :return: Numerical flux value
    """
    return n * (0.5 * v * (qleft - qright) + 0.5 * (left_flux + right_flux))


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


def minmod(prev, current, subs, cell):
    """
    @:brief Moment limiter using the minmod function
    :param prev:    Previous cell coefficients
    :param current: Current cell coefficients
    :param subs:    Subsequent cell coefficients
    :param cell:    Current cell solution coefficients to be replaced
    :return: None
    """

    for var in range(nVars):
        p_counter = P1
        for p in range(P1, 0, -1):
            coeff_tilde = 0.
            temp_a = current[p][var]
            temp_b = subs[p - 1][var] - current[p - 1][var]
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

    name = "/Users/jwtan/PycharmProjects/PyDG/polyMesh/50x0"
    nDims = 1
    nVars = 1
    # Uniform polynomial orders in the x and y-directions for now
    P1 = 1
    P2 = 0
    # Read in the mesh
    mesh = DgMesh.constructFromPolyMeshFolder(name, nDims)
    mesh.constructShapeBasedCells(name, nVars, P1, P2)

    """ Set test case """
    time = 0.
    nCells = len(mesh.connectivityData.cells)
    quadCoords = mesh.connectivityData.cells[0].GetQuadratureCoords

    """ Number of quadrature points within a single cell """
    nCoords = len(quadCoords)

    """ Numerical flux array [Number of cells in mesh, left and right numerical fluxes, number of variables]"""
    numericalFluxArray = np.empty((nCells, 2, nVars))

    """ Left and right face state values extrapolated using basis matrices at -1 and 1 in parametric space """
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
    exact_soln = np.empty((nCells, nCoords, nVars))

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
        xD = 0.5
        endTime = 1.0
        flux = linearAdvFlux
        boundaryConditions = "Periodic"

        for i in range(nCells):
            # Initial condition for step function
            mesh.connectivityData.cells[i].solnCell.uCoeffs[0] = np.array(
                [uR if coords <= xD else uL for coords in mesh.connectivityData.cells[i].calculations.cellCentre])
            # Set physical values
            mesh.connectivityData.cells[i].solnCell.uPhysical = np.matmul(
                mesh.connectivityData.cells[i].matCell.basisMatrix, mesh.connectivityData.cells[i].solnCell.uCoeffs)

    elif test_case == "top_hat":
        uL = 0.0
        uR = 1.0
        xL = 0.25
        xR = 0.75
        endTime = 1.0
        flux = linearAdvFlux
        boundaryConditions = "Periodic"

        for i in range(nCells):
            # Initial condition for top hat function
            mesh.connectivityData.cells[i].solnCell.uCoeffs[0] = np.array(
                [uR if xL <= coords <= xR else uL for coords in mesh.connectivityData.cells[i].calculations.cellCentre])
            # Set physical values
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

    for i in range(nCells):
        exact_soln[i] = mesh.connectivityData.cells[i].solnCell.uPhysical

    CFL = 0.01  # 1. / (2. * P1 + 1.)
    deltaT = 0.0
    # Constant mesh size for now
    deltax = mesh.connectivityData.cells[0].calculations.V

    """ Initialise new solution coefficients """
    uCoeffsNew = np.zeros((nCells, P1 + 1, nVars))

    """ Start time-loop """
    while endTime - time > 1e-10:
        """ Populate the face values """
        for i in range(nCells):
            leftFaceValueArray[i] = np.matmul(basisMatrixforF0, mesh.connectivityData.cells[i].solnCell.uCoeffs)[0]
            rightFaceValueArray[i] = np.matmul(basisMatrixforF1, mesh.connectivityData.cells[i].solnCell.uCoeffs)[0]

        """ Calculate time-step size """
        if flux == burgersFlux:
            deltaT = computeDtForBurgers(mesh, deltax, nCells, nCoords, CFL)
        elif flux == linearAdvFlux:
            deltaT = CFL * deltax / a
        else:
            raise NotImplementedError

        if boundaryConditions == "Periodic":
            """ Set Periodic boundary conditions, i.e., copy end cell values to the other end """
            numericalFluxArray[0][0] = numerical_flux(leftFaceValueArray[0], rightFaceValueArray[-1], a,
                                                      mesh.connectivityData.cells[0].calculations.faceNormals[0],
                                                      flux(a, leftFaceValueArray[0]),
                                                      flux(a, rightFaceValueArray[-1]))
            numericalFluxArray[-1][1] = numerical_flux(rightFaceValueArray[-1], leftFaceValueArray[0], a,
                                                       mesh.connectivityData.cells[-1].calculations.faceNormals[1],
                                                       flux(a, rightFaceValueArray[-1]),
                                                       flux(a, leftFaceValueArray[0]))

        elif boundaryConditions == "Neumann":
            """ Set Neumann boundary conditions """
            numericalFluxArray[0][0] = numerical_flux(leftFaceValueArray[0], leftFaceValueArray[0], a,
                                                      mesh.connectivityData.cells[0].calculations.faceNormals[0],
                                                      flux(a, leftFaceValueArray[0]),
                                                      flux(a, leftFaceValueArray[0]))
            numericalFluxArray[-1][1] = numerical_flux(rightFaceValueArray[-1], rightFaceValueArray[-1], a,
                                                       mesh.connectivityData.cells[-1].calculations.faceNormals[1],
                                                       flux(a, rightFaceValueArray[-1]),
                                                       flux(a, rightFaceValueArray[-1]))
        else:
            raise NotImplementedError

        numericalFluxArray[0][1] = numerical_flux(rightFaceValueArray[0], leftFaceValueArray[1], a,
                                                  mesh.connectivityData.cells[0].calculations.faceNormals[1],
                                                  flux(a, rightFaceValueArray[0]),
                                                  flux(a, leftFaceValueArray[1]))
        numericalFluxArray[-1][0] = numerical_flux(leftFaceValueArray[-1], rightFaceValueArray[-2], a,
                                                   mesh.connectivityData.cells[-1].calculations.faceNormals[0],
                                                   flux(a, leftFaceValueArray[-1]),
                                                   flux(a, rightFaceValueArray[-2]))

        """ Calculate numerical fluxes for internal faces, q left is internal value """
        for i in range(nCells - 2):
            numericalFluxArray[i + 1][0] = numerical_flux(leftFaceValueArray[i + 1], rightFaceValueArray[i], a,
                                                          mesh.connectivityData.cells[i + 1].calculations.faceNormals[0]
                                                          , flux(a, leftFaceValueArray[i + 1]),
                                                          flux(a, rightFaceValueArray[i]))
            numericalFluxArray[i + 1][1] = numerical_flux(rightFaceValueArray[i + 1], leftFaceValueArray[i + 2], a,
                                                          mesh.connectivityData.cells[i + 1].calculations.faceNormals[1]
                                                          , flux(a, rightFaceValueArray[i + 1]),
                                                          flux(a, leftFaceValueArray[i + 2]))

        """ Go through all cells """
        for i in range(nCells):
            """ Backward transformation from coefficient space to physical space """
            mesh.connectivityData.cells[i].solnCell.uPhysical = np.matmul(
                mesh.connectivityData.cells[i].matCell.basisMatrix, mesh.connectivityData.cells[i].solnCell.uCoeffs)
            """ Flux values at quadrature points in the cell """
            fluxValues = np.array([flux(a, q) for q in mesh.connectivityData.cells[i].solnCell.uPhysical])
            """ Flux coefficients vector in coefficient space, f hat = (M)^-1 B^T W f """
            fluxCoeffs = forwardTransform(mesh, fluxValues, i)

            divFlux = np.matmul(mesh.connectivityData.cells[i].invMassMatrix,
                                (np.matmul(basisMatrixforF0.transpose(),
                                           numericalFluxArray[i][0].reshape(-1, 1).transpose()) +
                                 np.matmul(basisMatrixforF1.transpose(),
                                           numericalFluxArray[i][1].reshape(-1, 1).transpose()) -
                                 np.matmul(mesh.connectivityData.cells[i].stiffnessMatrix, fluxCoeffs)))

            uCoeffsNew[i] = mesh.connectivityData.cells[i].solnCell.uCoeffs - deltaT * divFlux

        """ Increment time """
        if time + deltaT > endTime:
            deltaT = endTime - time
        time += deltaT
        print("Current time: " + str(time))

        """ c-1, c, c+1, Periodic """
        # minmod(uCoeffsNew[-1], uCoeffsNew[0], uCoeffsNew[1], mesh.connectivityData.cells[0].solnCell.uCoeffs)
        # minmod(uCoeffsNew[-2], uCoeffsNew[-1], uCoeffsNew[0], mesh.connectivityData.cells[-1].solnCell.uCoeffs)
        # for i in range(nCells - 2):
        #     minmod(uCoeffsNew[i], uCoeffsNew[i + 1], uCoeffsNew[i + 2],
        #            mesh.connectivityData.cells[i + 1].solnCell.uCoeffs)

        """ No limiter solution """
        for i in range(nCells):
            mesh.connectivityData.cells[i].solnCell.uCoeffs = uCoeffsNew[i]

    """ Plot solution """
    directory = "/Users/jwtan/PycharmProjects/PyDG/data/explicitAdv/"
    plt_name = test_case + "_P" + str(P1) + "_T" + str(endTime) + "_" + str(numerical_flux.__name__)
    title = "Step profile, final time = {0} s".format(str(endTime))
    plotSolution(mesh, nCells, nCoords, nVars, P1, 0., 1., directory, exact_soln, "Central flux", title, plt_name,
                 save=False)

    """ Calculate L2 error """
    calculateL2err(mesh, exact_soln, nCells)
