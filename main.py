import numpy as np
import pandas as pd
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
    a = np.sqrt(1.4 * pressure/rho)

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
    z = (gamma - 1)/(2.0 * gamma)
    pLR = pressurel/pressurer ** z
    vstar = (pLR * vxl / al + vxr / ar + 2.0 * (pLR - 1.0) / (gamma - 1.0)) / (pLR / al + 1.0 / ar)
    pstar = 0.5 * (pressurel * (1.0 + (gamma - 1.0) / (2.0 * al) * (vxl - vstar)) ** (1.0/z) + pressurer
                   * (1.0 + (gamma - 1.0) / (2.0 * ar) * (vstar - vxr)) ** (1.0/z))

    # /** Toro's book, page 330, eq (10.59 & 10.60) **/
    if pstar <= pressurel:
        ql = 1.0
    else:
        ql = np.sqrt(1.0 + (gamma + 1.0) / (2.0 * gamma) * (pstar / pressurel - 1.0))

    if pstar <= pressurer:
        qr = 1.0
    else:
        qr = np.sqrt(1.0 + (gamma + 1.0) / (2.0 * gamma) * (pstar / pressurer - 1.0))

    sleft = vxl - al * ql
    sright = vxr + ar * qr

    return sleft, sright


def computeDt(leftfacearray, rightfacearray, deltax, ncells, cfl):

    sleft, sright, smax = 0.0, 0.0, 0.0
    # Left and right conservative states of the RP
    # ql = np.empty(3)
    # qr = np.empty(3)

    for i in range(ncells - 1):  # n cells + 1, meshObj.connectivityData.cells[i].solnCell.uPhysical
        ql = primitiveToConservative(rightfacearray[i])
        qr = primitiveToConservative(leftfacearray[i + 1])

        sleft, sright = waveEstimates(ql, qr)  # calculate left and right waves

        smax = max(smax, max(abs(sleft), abs(sright)))  # eq (6.19)

    return cfl * deltax / smax  # // eq (6.17), make sure dt addition does not exceed final time output


def HLLFlux(qleft, qright):

    sleft, sright = waveEstimates(qleft, qright)  # calculate Sl and Sr

    fl = np.empty(3)
    fr = np.empty(3)
    fhll = np.empty(3)
    fl = eulerFlux(qleft)
    fr = eulerFlux(qright)

    # /** Eq (10.21) **/
    if sleft >= 0:
        fhll = fl

    if sright <= 0:
        fhll = fr

    for n in range(3):
        fhll[n] = (sright * fl[n] - sleft * fr[n] + sleft * sright * (qright[n] - qleft[n]))/(sright - sleft)

    return fhll


if __name__ == '__main__':
    """
    main()
    """

    name = "polyMesh/100x0"
    nDims = 1
    # Uniform polynomial orders in the x and y-directions for now
    P1 = 2
    P2 = 0
    # Read in the mesh
    mesh = DgMesh.constructFromPolyMeshFolder(name, nDims)
    mesh.constructShapeBasedCells(name, P1, P2)

    # Print data to check code correctness
    # print(mesh.derivedData.faces)
    # print(mesh.connectivityData.cells[0].geomData.neighbourLabels)
    # print(mesh.connectivityData.cells[0].calculations.V)
    # print(mesh.connectivityData.cells[0].GetMassMatrix())
    # print(mesh.connectivityData.cells[0].GetStiffnessMatrix())

    """
    Prototype time loop with forward Euler time-stepping
    """
    rhoL = 1.0
    uL = 0.0
    pL = 1.0
    rhoR = 0.125
    uR = 0.0
    pR = 0.1
    xd = 0.5
    primL = [rhoL, uL, pL]
    primR = [rhoR, uR, pR]

    time = 0.0
    endTime = 0.0001
    nCells = len(mesh.connectivityData.cells)
    quadCoords = np.array([mesh.connectivityData.cells[i].GetQuadratureCoords for i in range(nCells)]).reshape(-1)
    # print(quadCoords)

    # Initial conditions
    nCoords = len(quadCoords)
    # Conservative variables
    uArray = np.empty((nCells, 3))
    wArray = np.empty((nCells, 3))
    numericalFluxArray = np.empty((nCells + 1, 3))
    leftFaceValueArray = np.empty((nCells, 3))
    rightFaceValueArray = np.empty((nCells, 3))

    # for i in range(nCoords):
    #     if quadCoords[i] < xd:
    #         u[i] = primL
    #     else:
    #         u[i] = primR

    for i in range(nCells):
        # mesh.connectivityData.cells[i].solnCell.uPhysical = np.array(
        #     [primL if coords < xd else primR for coords in mesh.connectivityData.cells[i].GetQuadratureCoords])
        mesh.connectivityData.cells[i].solnCell.uCoeffs[0] = np.array(
            [primL if coords < xd else primR for coords in mesh.connectivityData.cells[i].calculations.cellCentre])
        # Set physical values, no need * abs(mesh.connectivityData.cells[i].geomCell.detJacobian)
        mesh.connectivityData.cells[i].solnCell.uPhysical = np.matmul(mesh.connectivityData.cells[i].matCell.basisMatrix
                                                                      , mesh.connectivityData.cells[i].solnCell.uCoeffs)

    print(mesh.connectivityData.cells[50].GetQuadratureCoords)
    print("Initial coefficients: ")
    print(mesh.connectivityData.cells[50].solnCell.uCoeffs)
    print("Initial physical values: ")
    print(mesh.connectivityData.cells[50].solnCell.uPhysical)

    basisMatrixforF0 = GetLegendre1d(np.array([[-1]]), P1)
    basisMatrixforF1 = GetLegendre1d(np.array([[1]]), P1)

    for i in range(nCells):
        leftFaceValueArray[i] = np.matmul(basisMatrixforF0, mesh.connectivityData.cells[i].solnCell.uCoeffs)
        rightFaceValueArray[i] = np.matmul(basisMatrixforF1, mesh.connectivityData.cells[i].solnCell.uCoeffs)

    # print(leftFaceValueArray)
    CFL = 0.9
    deltaT = 0.0
    deltax = mesh.connectivityData.cells[0].calculations.V

    while endTime - time > 1e-10:

        """ Set Neumann boundary conditions """
        leftBoundaryPrim = np.matmul(basisMatrixforF0, mesh.connectivityData.cells[0].solnCell.uCoeffs)
        rightBoundaryPrim = np.matmul(basisMatrixforF1, mesh.connectivityData.cells[-1].solnCell.uCoeffs)
        leftBoundaryCons = primitiveToConservative(leftBoundaryPrim[0])
        rightBoundaryCons = primitiveToConservative(rightBoundaryPrim[0])
        numericalFluxArray[0] = HLLFlux(leftBoundaryCons, leftBoundaryCons)
        numericalFluxArray[-1] = HLLFlux(rightBoundaryCons, rightBoundaryCons)

        """ Calculate time step size """
        deltaT = computeDt(leftFaceValueArray, rightFaceValueArray, deltax, nCells, CFL)

        # Increment time
        if time + deltaT > endTime:
            deltaT = endTime - time
        time += deltaT

        divFlux = np.zeros((nCells, P1 + 1))

        # for i in range(nCells):
        #
        #     for face in mesh.connectivityData.cells[i].geomData.neighbourLabels:
        #
        #         if face is not None:
