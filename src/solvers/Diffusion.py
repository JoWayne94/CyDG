import numpy as np
import math
from scipy.sparse.linalg import spsolve
from src.library.dgMesh.dgMesh import *
from src.library.paramCells.basis import *
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator


def sipFlux(cell_left, cell_right, bleft, bright, dleft, dright, n):
    dxi1dx1_left = cell_left.geomCell.geometry.dxi1dx1
    dxi1dx1_right = cell_right.geomCell.geometry.dxi1dx1

    owner = n * (- 0.5 * np.matmul(bleft.transpose(), (dxi1dx1_left * dleft)[:, :, 0])
                 - 0.5 * np.matmul((dxi1dx1_left * dleft)[:, :, 0].transpose(), bleft)) \
            + eta * np.matmul(bleft.transpose(), bleft)

    own_neigh = n * (0.5 * np.matmul(bright.transpose(), (dxi1dx1_left * dleft)[:, :, 0])
                     - 0.5 * np.matmul((dxi1dx1_right * dright)[:, :, 0].transpose(), bleft)) \
                - eta * np.matmul(bright.transpose(), bleft)

    neigh_own = n * (- 0.5 * np.matmul(bleft.transpose(), (dxi1dx1_right * dright)[:, :, 0])
                     + 0.5 * np.matmul((dxi1dx1_left * dleft)[:, :, 0].transpose(), bright)) \
                - eta * np.matmul(bleft.transpose(), bright)

    neighbour = n * (0.5 * np.matmul(bright.transpose(), (dxi1dx1_right * dright)[:, :, 0])
                     + 0.5 * np.matmul((dxi1dx1_right * dright)[:, :, 0].transpose(), bright)) \
                + eta * np.matmul(bright.transpose(), bright)

    return owner, own_neigh, neigh_own, neighbour


def sipFluxBoundary(cell_left, bleft, dleft, n):
    dxi1dx1_left = cell_left.geomCell.geometry.dxi1dx1

    sip_left = n * (- np.matmul(bleft.transpose(), (dxi1dx1_left * dleft)[:, :, 0])
                    - np.matmul((dxi1dx1_left * dleft)[:, :, 0].transpose(), bleft)) + \
               eta * np.matmul(bleft.transpose(), bleft)

    return sip_left


def forwardTransform(meshObj, physicalValues, cell):
    return np.matmul(meshObj.connectivityData.cells[cell].invMassMatrix,
                     np.matmul(meshObj.connectivityData.cells[cell].matCell.basisMatrix.transpose(),
                               meshObj.connectivityData.cells[cell].paramSeg.weights *
                               abs(meshObj.connectivityData.cells[cell].geomCell.detJacobian) *
                               physicalValues))


if __name__ == '__main__':
    """
    main()
    """

    name = "/Users/jwtan/PycharmProjects/PyDG/polyMesh/5x0"
    nDims = 1
    nVars = 1
    # Uniform polynomial orders in the x and y-directions for now
    P1 = 3
    P2 = 0
    # Read in the mesh
    mesh = DgMesh.constructFromPolyMeshFolder(name, nDims)
    mesh.constructShapeBasedCells(name, nVars, P1, P2)

    """
    Prototype backward Euler implicit time-stepping
    """

    """ Set test case """
    time = 0.0
    endTime = 1.0
    nCells = len(mesh.connectivityData.cells)
    quadCoords = mesh.connectivityData.cells[0].GetQuadratureCoords

    """ Number of quadrature points within a single cell """
    nCoords = len(quadCoords)

    """ Global Laplacian matrix """
    laplacian = np.zeros((nCells, nCells, P1 + 1, P1 + 1))

    """ Global SIP matrix [P1 + 1, P1 + 1] """
    sipMatrices = np.zeros_like(laplacian)

    """ Global mass matrix """
    massMatrices = np.zeros_like(laplacian)

    """ Left and right face variable values extrapolated using basis matrices at -1 and 1 in parametric space """
    basisMatrixforF0 = Legendre1d(np.array([[-1]]), P1)
    basisMatrixforF1 = Legendre1d(np.array([[1]]), P1)
    derivMatrixforF0 = Legendre1dGrad(np.array([[-1]]), P1)
    derivMatrixforF1 = Legendre1dGrad(np.array([[1]]), P1)

    """
    Set initial values using the first coefficients for constant state, or using physical values then 
    transform back to coefficient space
    """
    test_case = "unsteady_diffusion"
    numerical_flux = sipFlux

    if test_case == "unsteady_diffusion":
        boundaryConditions = "Dirichlet"
        u_l = 5
        u_r = 2
        forcing = 0  # -Laplace(u) = forcing

    else:
        raise NotImplementedError

    for i in range(nCells):
        # Initial condition for sine wave
        for coords in range(nCoords):
            # mesh.connectivityData.cells[i].solnCell.uPhysical[coords] = \
            #     np.sin(np.pi * mesh.connectivityData.cells[i].GetQuadratureCoords[coords])
            mesh.connectivityData.cells[i].solnCell.uPhysical[coords] = \
                0 * mesh.connectivityData.cells[i].GetQuadratureCoords[coords]

        mesh.connectivityData.cells[i].solnCell.uCoeffs = \
            forwardTransform(mesh, mesh.connectivityData.cells[i].solnCell.uPhysical, i)

    pltPhysical = np.zeros((nCells, nCoords, nVars))
    for i in range(nCells):
        pltPhysical[i] = mesh.connectivityData.cells[i].solnCell.uPhysical

    pltCoords = np.array([mesh.connectivityData.cells[i].GetQuadratureCoords for i in range(nCells)]).reshape(-1)
    plt.plot(pltCoords, pltPhysical[:, :, 0].reshape(-1), linestyle="", marker="o", label=str(time))

    deltaT = 0.1
    # Constant mesh size for now
    deltax = mesh.connectivityData.cells[0].calculations.V

    fCoeffsGlobal = np.zeros((nCells, P1 + 1))
    eta = (P1 ** 2) * (1.0 / deltax)

    """ Domain boundaries """
    laplacian[0][0] = mesh.connectivityData.cells[0].laplacianMatrix

    sipMatrices[0][0] = \
        sipFluxBoundary(mesh.connectivityData.cells[0], basisMatrixforF0, derivMatrixforF0,
                        mesh.connectivityData.cells[0].calculations.faceNormals[0])

    temp1, temp2, temp3, temp4 = \
        sipFlux(mesh.connectivityData.cells[0], mesh.connectivityData.cells[1],
                basisMatrixforF1, basisMatrixforF0, derivMatrixforF1, derivMatrixforF0,
                mesh.connectivityData.cells[0].calculations.faceNormals[1])

    sipMatrices[0][0] += temp1
    sipMatrices[0][1] += temp2
    sipMatrices[1][0] += temp3
    sipMatrices[1][1] += temp4
    fCoeffsGlobal[0] = - u_l * mesh.connectivityData.cells[0].calculations.faceNormals[0] * \
                       mesh.connectivityData.cells[0].geomCell.geometry.dxi1dx1 * derivMatrixforF0[:, :, 0] + \
                       eta * u_l * basisMatrixforF0

    laplacian[-1][-1] = mesh.connectivityData.cells[-1].laplacianMatrix

    sipMatrices[-1][-1] = \
        sipFluxBoundary(mesh.connectivityData.cells[-1], basisMatrixforF1, derivMatrixforF1,
                        mesh.connectivityData.cells[-1].calculations.faceNormals[1])

    fCoeffsGlobal[-1] = - u_r * mesh.connectivityData.cells[-1].calculations.faceNormals[1] * \
                        mesh.connectivityData.cells[-1].geomCell.geometry.dxi1dx1 * derivMatrixforF1[:, :, 0] + \
                        eta * u_r * basisMatrixforF1

    """ Populate internal values """
    for i in range(nCells - 2):
        laplacian[i + 1][i + 1] = mesh.connectivityData.cells[i + 1].laplacianMatrix

        temp1, temp2, temp3, temp4 = \
            sipFlux(mesh.connectivityData.cells[i + 1], mesh.connectivityData.cells[i + 2],
                    basisMatrixforF1, basisMatrixforF0, derivMatrixforF1, derivMatrixforF0,
                    mesh.connectivityData.cells[i + 1].calculations.faceNormals[1])
        sipMatrices[i + 1][i + 1] += temp1
        sipMatrices[i + 1][i + 2] += temp2
        sipMatrices[i + 2][i + 1] += temp3
        sipMatrices[i + 2][i + 2] += temp4

    for i in range(nCells):
        massMatrices[i][i] = mesh.connectivityData.cells[i].massMatrix

        tempf = forcing * np.ones((nCoords, nVars))

        fCoeffsGlobal[i] += np.matmul(mesh.connectivityData.cells[i].matCell.basisMatrix.transpose(),
                                      tempf * \
                                      mesh.connectivityData.cells[i].paramSeg.weights * \
                                      abs(mesh.connectivityData.cells[i].geomCell.detJacobian)).reshape(-1)

    """ Global block-matrix """
    A = laplacian + sipMatrices

    """ Row runs first then column !! """
    A = np.block([
        [A[j][i] for j in range(nCells)] for i in range(nCells)
    ])
    f = np.block([
        [fCoeffsGlobal[i]] for i in range(nCells)
    ])
    M = np.block([
        [massMatrices[j][i] for j in range(nCells)] for i in range(nCells)
    ])
    u = np.block([
        [mesh.connectivityData.cells[i].solnCell.uCoeffs] for i in range(nCells)
    ])
    uCoeffsGlobal = np.empty_like(u)

    """ Start time-loop """
    while endTime - time > 1e-10:

        time += deltaT
        LHS = f.reshape(-1, 1) + np.matmul((1 / deltaT) * M, u.reshape(-1, 1))

        uCoeffsGlobal = np.matmul(np.linalg.inv((1 / deltaT) * M + A), LHS)

        if abs(time - 0.1) < 1e-6 or abs(time - 0.2) < 1e-6:
            for i in range(nCells):
                pltPhysical[i] = np.matmul(mesh.connectivityData.cells[i].matCell.basisMatrix,
                                           uCoeffsGlobal.reshape(nCells, P1 + 1)[i].reshape(-1, 1))
            label = "{time:.1f}".format(time=time)
            plt.plot(pltCoords, pltPhysical[:, :, 0].reshape(-1), linestyle="", marker="o", label=label)

        u = uCoeffsGlobal

    np.set_printoptions(precision=2, suppress=True)
    # print(A)
    # print(f.reshape(-1, 1))
    print(uCoeffsGlobal)

    for i in range(nCells):
        pltPhysical[i] = np.matmul(mesh.connectivityData.cells[i].matCell.basisMatrix,
                                   uCoeffsGlobal.reshape(nCells, P1 + 1)[i].reshape(-1, 1))

    plt.plot(pltCoords, pltPhysical[:, :, 0].reshape(-1), linestyle="", marker="o", color="blue", label=str(endTime))
    # plt.plot([0.0, 2.0],
    #          np.array([np.matmul(basisMatrixforF0, uCoeffsGlobal.reshape(2, 2)[0].reshape(-1, 1)),
    #                    np.matmul(basisMatrixforF1, uCoeffsGlobal.reshape(2, 2)[1].reshape(-1, 1))])[:, :,
    #          0].reshape(-1),
    #          linestyle="", marker="o", color="blue")
    plt.grid()
    # plt_name = test_case + "_" + "P" + str(P1) + "_" + str(endTime) + '.png'
    # plt.savefig(plt_name, dpi=100)
    # np.savetxt(plt_name + "_coords.dat", pltCoords, delimiter=',')
    # np.savetxt(plt_name + "_values.dat", pltPhysical[:, :, 0].reshape(-1), delimiter=',')
    plt.legend()
    plt.show()
