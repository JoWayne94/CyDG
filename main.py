import numpy as np
import pandas as pd
import math
from src.library.dgMesh.dgMesh import *
from src.library.paramCells.basis import *


def conservativeToPrimitive(u):
    w = np.empty(3)
    rho = u[0]
    vx = u[1] / u[0]
    E = u[2]
    kin = 0.5 * rho * vx * vx
    e = (E - kin) / rho
    gamma = 1.4
    pressure = rho * e * (gamma - 1.0)

    w[0] = rho
    w[1] = vx
    w[2] = pressure

    return w


def primitiveToConservative(w):
    u = np.empty(3)
    rho = w[0]
    vx = w[1]
    pressure = w[2]
    kin = 0.5 * rho * vx * vx
    gamma = 1.4
    e = pressure / (rho * (gamma - 1.0))
    E = rho * e + kin

    u[0] = rho
    u[1] = rho * vx
    u[2] = E

    return u


def eulerFlux(u):
    F = np.empty(3)
    w = conservativeToPrimitive(u)
    rho = w[0]
    vx = w[1]
    pressure = w[2]
    E = u[2]

    F[0] = rho * vx
    F[1] = rho * vx * vx + pressure
    F[2] = vx * (E + pressure)

    return F


def computeSoundSpeed(rho, pressure):
    # Ideal Gas EoS
    a = math.sqrt(1.4 * pressure / rho)

    return a


def waveEstimates(ul, ur):
    # wl = np.empty(3)
    # wr = np.empty(3)
    wl = conservativeToPrimitive(ul)
    wr = conservativeToPrimitive(ur)

    rhol = wl[0]
    rhor = wr[0]
    vxl = wl[1]
    vxr = wr[1]
    pressurel = wl[2]
    pressurer = wr[2]

    # Ideal Gas EoS
    al = computeSoundSpeed(rhol, pressurel)
    ar = computeSoundSpeed(rhor, pressurer)
    gamma = 1.4
    # /** Pressure–Based Wave Speed Estimates (ideal gases) **/
    """ Two-rarefaction Riemann solver TRRS for computing Pstar
        Toro's book page 301 & 330, section 9.4.1 """
    z = (gamma - 1) / (2.0 * gamma)
    pLR = (pressurel / pressurer) ** z
    vstar = (pLR * vxl / al + vxr / ar + 2.0 * (pLR - 1.0) / (gamma - 1.0)) / (pLR / al + 1.0 / ar)
    pstar = 0.5 * (pressurel * (1.0 + (gamma - 1.0) / (2.0 * al) * (vxl - vstar)) ** (1.0 / z) + pressurer
                   * (1.0 + (gamma - 1.0) / (2.0 * ar) * (vstar - vxr)) ** (1.0 / z))

    # /** Toro's book, page 330, eq (10.59 & 10.60) **/
    if pstar <= pressurel:
        ql = 1.0
    else:
        ql = math.sqrt(1.0 + (gamma + 1.0) / (2.0 * gamma) * (pstar / pressurel - 1.0))

    if pstar <= pressurer:
        qr = 1.0
    else:
        qr = math.sqrt(1.0 + (gamma + 1.0) / (2.0 * gamma) * (pstar / pressurer - 1.0))

    sleft = vxl - al * ql
    sright = vxr + ar * qr

    return sleft, sright


def computeDt(leftfacearray, rightfacearray, deltax, ncells, cfl):
    """
    @:brief Calculate time-step size with CFL constraints
    :param leftfacearray: Conservative values on the left face of the cell
    :param rightfacearray: Conservative values on the right face of the cell
    :param deltax: Mesh size
    :param ncells: Number of cells
    :param cfl: Courant number
    :return: Time-step size
    """
    sleft, sright, smax = 0.0, 0.0, 0.0
    # Left and right conservative states of the RP
    # ql = np.empty(3)
    # qr = np.empty(3)

    for cell in range(ncells - 1):  # n cells + 1, meshObj.connectivityData.cells[i].solnCell.uPhysical
        ql = rightfacearray[cell]
        qr = leftfacearray[cell + 1]

        sleft, sright = waveEstimates(ql, qr)  # calculate left and right waves

        smax = max(smax, max(abs(sleft), abs(sright)))  # eq (6.19)

    return cfl * deltax / smax  # // eq (6.17), make sure dt addition does not exceed final time output


def HLLFlux(qleft, qright):
    sleft, sright = waveEstimates(qleft, qright)  # calculate Sl and Sr

    # fl = np.empty(3)
    # fr = np.empty(3)
    fhll = np.empty(3)
    fl = eulerFlux(qleft)
    fr = eulerFlux(qright)

    # /** Eq (10.21) **/
    if sleft >= 0:
        return fl
    elif sright <= 0:
        return fr
    else:
        for n in range(3):
            fhll[n] = (sright * fl[n] - sleft * fr[n] + sleft * sright * (qright[n] - qleft[n])) / (sright - sleft)
        return fhll


if __name__ == '__main__':
    """
    main()
    """

    name = "polyMesh/10x0"
    nDims = 1
    # Uniform polynomial orders in the x and y-directions for now
    P1 = 1
    P2 = 0
    # Read in the mesh
    mesh = DgMesh.constructFromPolyMeshFolder(name, nDims)
    mesh.constructShapeBasedCells(name, P1, P2)

    # Print data to check code correctness
    # print(mesh.derivedData.faces)
    # print(mesh.connectivityData.cells[0].geomData.neighbourLabels)
    # print(mesh.connectivityData.cells[0].calculations.V)
    # print(mesh.connectivityData.cells[0].matCell.basisMatrix)
    print(mesh.connectivityData.cells[0].GetMassMatrix())
    print(mesh.connectivityData.cells[0].GetStiffnessMatrix())

    """
    Prototype time loop with forward Euler time-stepping
    """

    """ Set test case """
    test_case = "Toro1a"

    if test_case == "Toro1a":
        rhoL = 1.0
        uL = 0.0
        pL = 1.0
        rhoR = 0.125
        uR = 0.0
        pR = 0.1
        xd = 0.5
        endTime = 0.25
    elif test_case == "Toro1b":
        rhoL = 1.0
        uL = 0.75
        pL = 1.0
        rhoR = 0.125
        uR = 0.0
        pR = 0.1
        xd = 0.3
        endTime = 0.2
    elif test_case == "linear_advection":
        rhoL = 1.0
        uL = 0.0
        pL = 1.0
        rhoR = 0.125
        uR = 0.0
        pR = 1.0
        xd = 0.5
        endTime = 0.25
    else:
        raise NotImplementedError

    """ Initialise primitive and conservative initial conditions """
    primL = np.array([rhoL, uL, pL])
    primR = np.array([rhoR, uR, pR])
    consL = primitiveToConservative(primL)
    consR = primitiveToConservative(primR)

    time = 0.0
    endTime = 0.0001
    nCells = len(mesh.connectivityData.cells)
    # quadCoords = np.array([mesh.connectivityData.cells[i].GetQuadratureCoords for i in range(nCells)]).reshape(-1)
    quadCoords = mesh.connectivityData.cells[0].GetQuadratureCoords
    """ Number of quadrature points within a single cell """
    nCoords = len(quadCoords)

    # Conservative variables
    # uArray = np.empty((nCells, nCoords, 3))
    # wArray = np.empty((nCells, nCoords, 3))

    """ Numerical flux array [Number of cells in mesh, left and right numerical fluxes, number of variables]"""
    numericalFluxArray = np.empty((nCells, 2, 3))
    """ Left and right face variable values extrapolated using basis matrices at -1 and 1 in parametric space """
    basisMatrixforF0 = GetLegendre1d(np.array([[-1]]), P1)
    basisMatrixforF1 = GetLegendre1d(np.array([[1]]), P1)
    massMatrixforF0 = np.matmul(basisMatrixforF0.transpose(), basisMatrixforF0)
    massMatrixforF1 = np.matmul(basisMatrixforF1.transpose(), basisMatrixforF1)
    # print(massMatrixforF0)
    # print(massMatrixforF1)
    basisMatrixforFace = GetLegendre1d(np.array([[0]]), P1)
    leftFaceValueArray = np.empty((nCells, 3))
    rightFaceValueArray = np.empty((nCells, 3))

    """ Set initial values using the first coefficients for constant state """
    for i in range(nCells):
        mesh.connectivityData.cells[i].solnCell.uCoeffs[0] = np.array(
            [consL if coords <= xd else consR for coords in mesh.connectivityData.cells[i].calculations.cellCentre])
        # Set physical values, no need * abs(mesh.connectivityData.cells[i].geomCell.detJacobian)
        mesh.connectivityData.cells[i].solnCell.uPhysical = np.matmul(mesh.connectivityData.cells[i].matCell.basisMatrix
                                                                      , mesh.connectivityData.cells[i].solnCell.uCoeffs)

    print("Initial coefficients: ")
    print(mesh.connectivityData.cells[0].solnCell.uCoeffs)
    print("Initial physical values: ")
    print(mesh.connectivityData.cells[0].solnCell.uPhysical)

    CFL = 0.9
    deltaT = 0.0
    # Constant mesh size for now
    deltax = mesh.connectivityData.cells[0].calculations.V

    """ Start time-loop """
    while endTime - time > 1e-10:

        """ Populate the faces values """
        for i in range(nCells):
            leftFaceValueArray[i] = np.matmul(basisMatrixforF0, mesh.connectivityData.cells[i].solnCell.uCoeffs)[0]
            # print(leftFaceValueArray[i])
            rightFaceValueArray[i] = np.matmul(basisMatrixforF1, mesh.connectivityData.cells[i].solnCell.uCoeffs)[0]
            # print(rightFaceValueArray[i])

        """ Set Neumann boundary conditions, i.e., copy end cell values """
        numericalFluxArray[0][0] = HLLFlux(leftFaceValueArray[0], leftFaceValueArray[0]) * \
                                   mesh.connectivityData.cells[0].calculations.faceNormals[0]
        numericalFluxArray[0][1] = HLLFlux(rightFaceValueArray[0], leftFaceValueArray[1]) * \
                                   mesh.connectivityData.cells[0].calculations.faceNormals[1]
        numericalFluxArray[-1][0] = HLLFlux(rightFaceValueArray[-2], leftFaceValueArray[-1]) * \
                                    mesh.connectivityData.cells[-1].calculations.faceNormals[0]
        numericalFluxArray[-1][1] = HLLFlux(rightFaceValueArray[-1], rightFaceValueArray[-1]) * \
                                    mesh.connectivityData.cells[-1].calculations.faceNormals[1]

        """ Calculate numerical fluxes for internal faces """
        for i in range(nCells - 2):
            numericalFluxArray[i + 1][0] = HLLFlux(rightFaceValueArray[i], leftFaceValueArray[i + 1]) \
                                           * mesh.connectivityData.cells[i + 1].calculations.faceNormals[0]
            # numericalFluxArray[i + 1][0] = HLLFlux(leftFaceValueArray[i + 1], rightFaceValueArray[i]) \
            #                                * mesh.connectivityData.cells[i + 1].calculations.faceNormals[0]
            numericalFluxArray[i + 1][1] = HLLFlux(rightFaceValueArray[i + 1], leftFaceValueArray[i + 2]) \
                                           * mesh.connectivityData.cells[i + 1].calculations.faceNormals[1]

        # print(numericalFluxArray)

        """ Calculate time-step size """
        deltaT = computeDt(leftFaceValueArray, rightFaceValueArray, deltax, nCells, CFL)
        print(deltaT)

        # Increment time
        # if time + deltaT > endTime:
        #     deltaT = endTime - time
        time += deltaT

        """ Initialise divergence of flux and new solution coefficients """
        # divFlux = np.zeros((P1 + 1, 3))
        uCoeffsNew = np.zeros((nCells, P1 + 1, 3))

        """ Go through all cells """
        for i in range(nCells):
            """ Backward transformation from coefficient space to physical space """
            mesh.connectivityData.cells[i].solnCell.uPhysical = np.matmul(
                mesh.connectivityData.cells[i].matCell.basisMatrix, mesh.connectivityData.cells[i].solnCell.uCoeffs)
            """ Euler flux values at quadrature points in the cell """
            eulerFluxValues = np.array([eulerFlux(q) for q in mesh.connectivityData.cells[i].solnCell.uPhysical])
            """ Flux coefficients vector in coefficient space, f hat = (M)^-1 B^T W f """
            eulerFluxCoeffs = np.matmul(np.linalg.inv(mesh.connectivityData.cells[i].GetMassMatrix()),
                                        np.matmul(mesh.connectivityData.cells[i].matCell.basisMatrix.transpose(),
                                                  mesh.connectivityData.cells[i].paramSeg.weights *
                                                  abs(mesh.connectivityData.cells[i].geomCell.detJacobian) *
                                                  eulerFluxValues))
            # numericalFluxLeftCoeffs = np.matmul(np.linalg.inv(massMatrixforF0), np.matmul(basisMatrixforF0.transpose(),
            #                                                                               numericalFluxArray[i][0]))
            # numericalFluxRightCoeffs = np.matmul(np.linalg.inv(massMatrixforF1), np.matmul(basisMatrixforF1.transpose(),
            #                                                                                numericalFluxArray[i][1]))
            """ Test discrete Galerkin projection operation """
            # print(np.matmul(mesh.connectivityData.cells[i].matCell.basisMatrix, eulerFluxMatrix2))
            # print(np.matmul(mesh.connectivityData.cells[i].GetStiffnessMatrix(), eulerFluxMatrix2))
            print("Checkpoint... ")
            # print(numericalFluxLeftCoeffs)
            # print(np.matmul(basisMatrixforF0.transpose(), numericalFluxArray[i][0].reshape(-1, 1).transpose()))
            # print(numericalFluxArray[i][1])
            # print(np.matmul(basisMatrixforFace.transpose(), numericalFluxArray[i][1].reshape(-1, 1).transpose()))
            # + np.matmul(basisMatrixforF1.transpose(), numericalFluxArray[i][1].reshape(-1, 1).transpose()))
            # divFlux = np.matmul(np.linalg.inv(mesh.connectivityData.cells[i].GetMassMatrix()),
            #                     (numericalFluxLeftCoeffs + numericalFluxRightCoeffs -
            #                      np.matmul(mesh.connectivityData.cells[i].GetStiffnessMatrix(), eulerFluxCoeffs)))
            divFlux = np.matmul(np.linalg.inv(mesh.connectivityData.cells[i].GetMassMatrix()),
                                (np.matmul(basisMatrixforF0.transpose(),
                                           numericalFluxArray[i][0].reshape(-1, 1).transpose()) +
                                 np.matmul(basisMatrixforF1.transpose(),
                                           numericalFluxArray[i][1].reshape(-1, 1).transpose()) -
                                 np.matmul(mesh.connectivityData.cells[i].GetStiffnessMatrix(), eulerFluxCoeffs)))
            uCoeffsNew[i] = mesh.connectivityData.cells[i].solnCell.uCoeffs - deltaT * divFlux

        print("Results: ")
        for i in range(nCells):
            mesh.connectivityData.cells[i].solnCell.uCoeffs = uCoeffsNew[i]
            print(np.matmul(
                mesh.connectivityData.cells[i].matCell.basisMatrix, mesh.connectivityData.cells[i].solnCell.uCoeffs))

    # for face in mesh.connectivityData.cells[i].geomData.neighbourLabels:
    #
    #     if face is not None:
