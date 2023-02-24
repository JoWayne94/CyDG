"""
File: ADR2d.py

Description: Solve linear advection diffusion equation implicitly in 2D with backward Euler time-stepping
"""
import matplotlib.pyplot as plt
import os
import imageio
import numpy as np
import math
from scipy.special import kn, kv, kvp
from src.solvers.ADR import *
from src.library.dgMesh.dgMesh import *
from src.library.paramCells.basis import *


def sipFlux(cell_left, cell_right, face, bleft, bright, dleft, dright, n, eta):
    """
    @:brief Symmetric Interior Penalty method for interior faces
    :param cell_left:  Owner cell
    :param cell_right: Neighbour cell
    :param face:       Individual interior face, face-based addressing might suit better
    :param bleft:      Owner face basis matrix
    :param bright:     Neighbour face basis matrix
    :param dleft:      Owner face derivative matrix
    :param dright:     Neighbour face derivative matrix
    :param n:          Owner face unit normal vector
    :return: x4 Block-matrix contributions
    """
    dxi1dx1_left = cell_left.geomCell.invJacobian[:, :, 0][:, 0]
    dxi2dx1_left = cell_left.geomCell.invJacobian[:, :, 1][:, 0]
    dxi1dx2_left = cell_left.geomCell.invJacobian[:, :, 0][:, 1]
    dxi2dx2_left = cell_left.geomCell.invJacobian[:, :, 1][:, 1]

    dxi1dx1_right = cell_right.geomCell.invJacobian[:, :, 0][:, 0]
    dxi2dx1_right = cell_right.geomCell.invJacobian[:, :, 1][:, 0]
    dxi1dx2_right = cell_right.geomCell.invJacobian[:, :, 0][:, 1]
    dxi2dx2_right = cell_right.geomCell.invJacobian[:, :, 1][:, 1]

    weight_mat = face.paramSeg.weights * abs(face.geomCell.detJacobian)

    """ Consistency, symmetry, and penalty terms """
    owner = - 0.5 * np.matmul(bleft.transpose(),
                              (n[0] * (dxi1dx1_left * dleft[:, :, 0] + dxi2dx1_left * dleft[:, :, 1]) +
                               n[1] * (dxi1dx2_left * dleft[:, :, 0] + dxi2dx2_left * dleft[:, :, 1])) * weight_mat) \
            - 0.5 * np.matmul(((n[0] * (dxi1dx1_left * dleft[:, :, 0] + dxi2dx1_left * dleft[:, :, 1]) +
                                n[1] * (dxi1dx2_left * dleft[:, :, 0] + dxi2dx2_left * dleft[:, :, 1]))).transpose(),
                              bleft * weight_mat) \
            + eta * np.matmul(bleft.transpose(), bleft * weight_mat)

    own_neigh = 0.5 * np.matmul(bright.transpose(),
                                (n[0] * (dxi1dx1_left * dleft[:, :, 0] + dxi2dx1_left * dleft[:, :, 1]) +
                                 n[1] * (dxi1dx2_left * dleft[:, :, 0] + dxi2dx2_left * dleft[:, :, 1])) * weight_mat) \
                - 0.5 * np.matmul(((n[0] * (dxi1dx1_right * dright[:, :, 0] + dxi2dx1_right * dright[:, :, 1]) +
                                    n[1] * (dxi1dx2_right * dright[:, :, 0] + dxi2dx2_left * dright[:, :,
                                                                                             1]))).transpose(),
                                  bleft * weight_mat) \
                - eta * np.matmul(bright.transpose(), bleft * weight_mat)

    neigh_own = - 0.5 * np.matmul(bleft.transpose(),
                                  (n[0] * (dxi1dx1_right * dright[:, :, 0] + dxi2dx1_right * dright[:, :, 1]) +
                                   n[1] * (dxi1dx2_right * dright[:, :, 0] + dxi2dx2_right * dright[:, :, 1])) *
                                  weight_mat) \
                + 0.5 * np.matmul(((n[0] * (dxi1dx1_left * dleft[:, :, 0] + dxi2dx1_left * dleft[:, :, 1]) +
                                    n[1] * (dxi1dx2_left * dleft[:, :, 0] + dxi2dx2_left * dleft[:, :,
                                                                                           1]))).transpose(),
                                  bright * weight_mat) \
                - eta * np.matmul(bleft.transpose(), bright * weight_mat)

    neighbour = 0.5 * np.matmul(bright.transpose(),
                                (n[0] * (dxi1dx1_right * dright[:, :, 0] + dxi2dx1_right * dright[:, :, 1]) +
                                 n[1] * (dxi1dx2_right * dright[:, :, 0] + dxi2dx2_right * dright[:, :, 1])) *
                                weight_mat) \
                + 0.5 * np.matmul(((n[0] * (dxi1dx1_right * dright[:, :, 0] + dxi2dx1_right * dright[:, :, 1]) +
                                    n[1] * (dxi1dx2_right * dright[:, :, 0] + dxi2dx2_right * dright[:, :,
                                                                                              1]))).transpose(),
                                  bright * weight_mat) \
                + eta * np.matmul(bright.transpose(), bright * weight_mat)

    return owner, own_neigh, neigh_own, neighbour


def sipFluxBoundary(cell_left, face_left, eta):
    """
    @:brief Symmetric Interior Penalty method on Dirichlet boundaries
    :param cell_left: Boundary cell
    :param face_left: Dirichlet boundary face
    :return: Diagonal block coefficient contribution
    """
    dxi1dx1_left = cell_left.geomCell.invJacobian[:, :, 0][:, 0]
    dxi2dx1_left = cell_left.geomCell.invJacobian[:, :, 1][:, 0]
    dxi1dx2_left = cell_left.geomCell.invJacobian[:, :, 0][:, 1]
    dxi2dx2_left = cell_left.geomCell.invJacobian[:, :, 1][:, 1]

    weight_mat = face_left.paramSeg.weights * abs(face_left.geomCell.detJacobian)

    bleft = face_left.faceBasis
    dleft = face_left.faceDeriv
    n = face_left.unitNormal[0]

    sip_left = - np.matmul(bleft.transpose(),
                           (n[0] * (dxi1dx1_left * dleft[:, :, 0] + dxi2dx1_left * dleft[:, :, 1]) +
                            n[1] * (dxi1dx2_left * dleft[:, :, 0] + dxi2dx2_left * dleft[:, :, 1])) * weight_mat) \
               - np.matmul(((n[0] * (dxi1dx1_left * dleft[:, :, 0] + dxi2dx1_left * dleft[:, :, 1]) +
                             n[1] * (dxi1dx2_left * dleft[:, :, 0] + dxi2dx2_left * dleft[:, :, 1]))).transpose(),
                           bleft * weight_mat) \
               + eta * np.matmul(bleft.transpose(), bleft * weight_mat)

    return sip_left


def forwardTransform(meshObj, physicalValues, cell):
    return np.matmul(meshObj.connectivityData.cells[cell].invMassMatrix,
                     np.matmul(meshObj.connectivityData.cells[cell].matCell.basisMatrix.transpose(),
                               meshObj.connectivityData.cells[cell].paramQuad.weights *
                               abs(meshObj.connectivityData.cells[cell].geomCell.detJacobian) *
                               physicalValues))


def zeroDirichletBCs(x, y):
    return 0.


def analyticDirichletBCs(x, y):
    bc = (point_source / (2 * np.pi * kappa)) * kn(0, (a[0] * np.sqrt(x ** 2 + y ** 2)) / (2 * kappa)) * \
         np.exp((a[0] * x) / (2 * kappa))

    return bc.reshape(-1, 1)


def analyticDirichletBCsDeriv_x(x, y):
    bc = (point_source / (2 * np.pi * kappa)) * (kvp(0, (a[0] * np.sqrt(x ** 2 + y ** 2)) / (2 * kappa), 1) *
                                                 np.exp((a[0] * x) / (2 * kappa)) +
                                                 kn(0, (a[0] * np.sqrt(x ** 2 + y ** 2)) / (2 * kappa)) *
                                                 (a[0] / (2 * kappa)) * np.exp((a[0] * x) / (2 * kappa)))
    return bc.reshape(-1, 1)


def analyticDirichletBCsDeriv_y(x, y):
    bc = (point_source / (2 * np.pi * kappa)) * kvp(0, (a[0] * np.sqrt(x ** 2 + y ** 2)) / (2 * kappa), 1) * \
         np.exp((a[0] * x) / (2 * kappa))

    return bc.reshape(-1, 1)


def analyticDirichletBCsDeriv_xy(x, y):
    bc = (point_source / (2 * np.pi * kappa)) * (kvp(0, (a[0] * np.sqrt(x ** 2 + y ** 2)) / (2 * kappa), 2) *
                                                 np.exp((a[0] * x) / (2 * kappa)) +
                                                 kvp(0, (a[0] * np.sqrt(x ** 2 + y ** 2)) / (2 * kappa), 1) *
                                                 (a[0] / (2 * kappa)) * np.exp((a[0] * x) / (2 * kappa)))
    return bc.reshape(-1, 1)


if __name__ == '__main__':
    """
    main()
    """

    name = "/Users/jwtan/PycharmProjects/CyDG/polyMesh/3x3"
    nDims = 2
    nVars = 1
    # Uniform polynomial orders in the x and y-directions for now
    P1 = 1
    P2 = 1
    dims = (P1 + 1) * (P2 + 1)
    # Read in the mesh
    mesh = DgMesh.constructFromPolyMeshFolder(name, nDims)
    mesh.constructShapeBasedCells(name, nVars, P1, P2)

    """ Set test case """
    time = 0.
    deltaT = 0.
    endTime = 0.
    nCells = len(mesh.connectivityData.cells)
    quadCoords = mesh.connectivityData.cells[0].GetQuadratureCoords

    """ Number of quadrature points within a single cell """
    nCoords = len(quadCoords[0])

    """
    Set initial values using the first coefficients for constant state, or using physical values then 
    transform back to coefficient space
    """
    test_case = "Poisson"
    a = np.array([0., 0.])  # constant velocity
    kappa = 0.
    forcing = 0.
    point_source = None
    left_boundary = None
    bottom_boundary = None
    right_boundary = None
    top_boundary = None
    g_d = None
    numerical_flux = upwind_flux
    exact_soln = np.empty((nCells, nCoords, nVars))

    if test_case == "pure_advection_gaussian_wave":
        a = np.array([1., 1.])
        endTime = 2.
        left_boundary = -1.
        bottom_boundary = -1.
        right_boundary = 1.
        top_boundary = 1.
        flux = linearAdvFlux
        left_BCs = "Periodic"
        bottom_BCs = "Periodic"
        right_BCs = "Periodic"
        top_BCs = "Periodic"

        for i in range(nCells):
            # Initial condition for gaussian wave
            for coords in range(nCoords):
                mesh.connectivityData.cells[i].solnCell.uPhysical[coords] = \
                    np.exp(-4 * (mesh.connectivityData.cells[i].GetQuadratureCoords[:, coords][0] ** 2 +
                                 mesh.connectivityData.cells[i].GetQuadratureCoords[:, coords][1] ** 2))

            mesh.connectivityData.cells[i].solnCell.uCoeffs = \
                forwardTransform(mesh, mesh.connectivityData.cells[i].solnCell.uPhysical, i)

            exact_soln[i] = mesh.connectivityData.cells[i].solnCell.uPhysical

    elif test_case == "pure_advection_circular_func":
        u_l = 0.
        u_r = 1.
        x_centre = 0.0
        y_centre = 0.0
        radius = 0.25
        a = np.array([1., 1.])
        endTime = 2.
        left_boundary = -1.
        bottom_boundary = -1.
        right_boundary = 1.
        top_boundary = 1.
        flux = linearAdvFlux
        left_BCs = "Periodic"
        bottom_BCs = "Periodic"
        right_BCs = "Periodic"
        top_BCs = "Periodic"

        for i in range(nCells):
            # Initial condition for a circular step profile
            for coords in range(nCoords):
                if (mesh.connectivityData.cells[i].GetQuadratureCoords[:, coords][0] - x_centre) ** 2 + \
                        (mesh.connectivityData.cells[i].GetQuadratureCoords[:, coords][1] - y_centre) ** 2 <= radius:
                    mesh.connectivityData.cells[i].solnCell.uPhysical[coords] = u_r
                else:
                    mesh.connectivityData.cells[i].solnCell.uPhysical[coords] = u_l

            mesh.connectivityData.cells[i].solnCell.uCoeffs = \
                forwardTransform(mesh, mesh.connectivityData.cells[i].solnCell.uPhysical, i)

            exact_soln[i] = mesh.connectivityData.cells[i].solnCell.uPhysical

    elif test_case == "Poisson":
        g_d = zeroDirichletBCs
        left_boundary = -1.
        bottom_boundary = -1.
        right_boundary = 1.
        top_boundary = 1.
        kappa = 1.
        forcing = 1.  # -Laplace(u) = forcing
        left_BCs = "Dirichlet"
        bottom_BCs = "Dirichlet"
        right_BCs = "Dirichlet"
        top_BCs = "Dirichlet"

        """ No initial conditions """

    elif test_case == "Hinze":
        a = np.array([1., 0.])
        kappa = 0.05
        forcing = 0.
        point_source = 16.67
        g_d = analyticDirichletBCs

        left_boundary = 0.05
        bottom_boundary = -0.5
        right_boundary = 4.05
        top_boundary = 0.5
        endTime = 0.1
        left_BCs = "Dirichlet"
        bottom_BCs = "Dirichlet"
        right_BCs = "Neumann"
        top_BCs = "Dirichlet"

        """ Zeros initial condition not needed """
        for i in range(nCells):
            for coord in range(nCoords):
                x_coord = mesh.connectivityData.cells[i].GetQuadratureCoords[:, coord][0]
                y_coord = mesh.connectivityData.cells[i].GetQuadratureCoords[:, coord][1]
                exact_soln[i][coord] = (point_source / (2 * np.pi * kappa)) * \
                                       kn(0, (np.sqrt(x_coord ** 2 + y_coord ** 2)) / (2 * kappa)) * \
                                       np.exp(x_coord / (2 * kappa))

    else:
        raise NotImplementedError

    """ Plot exact solution """
    # xCoords = np.array([[mesh.connectivityData.cells[i].GetQuadratureCoords[:, j][0] for j in range(nCoords)]
    #                     for i in range(nCells)])
    # yCoords = np.array([[mesh.connectivityData.cells[i].GetQuadratureCoords[:, j][1] for j in range(nCoords)]
    #                     for i in range(nCells)])
    # X, Y = np.meshgrid(xCoords, yCoords)
    # Z = np.empty((X.shape[0], Y.shape[0]))
    #
    # for i in range(X.shape[0]):
    #     for j in range(Y.shape[0]):
    #         Z[i][j] = (point_source / (2 * np.pi * kappa)) * \
    #                   kn(0, (np.sqrt(X[i][j] ** 2 + Y[i][j] ** 2)) / (2 * kappa)) * \
    #                   np.exp(X[i][j] / (2 * kappa))
    #
    # fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    #
    # # Plot the surface
    # surf = ax.contourf(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=False)
    # ax.title.set_text("Exact solution")
    # ax.set_xlim(left_boundary, right_boundary)
    # ax.set_ylim(bottom_boundary, top_boundary)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    #
    # plt.show()

    if a[0] < 0 or a[1] < 0:
        print("Advection velocity has to be positive for now.")
        raise ValueError
    if kappa < 0:
        print("Negative diffusivity will blow up the solution.")
        raise ValueError

    """ Identify boundary cells """
    boundaries = [[bottom_boundary, right_boundary], [top_boundary, left_boundary]]
    boundaryIDs = []
    for boundary in boundaries:  # 4 domain boundaries
        for axis in range(1, -1, -1):  # y then x axes
            tmp = np.array([i for i, p in enumerate(mesh.connectivityData.points[:, axis], 0)
                            if abs(p - boundary[int(abs(1 - axis))]) < 1e-6])

            IDs = [[i for cell in tmp if mesh.connectivityData.cells[i].geomData.pointLabels.count(cell) > 0]
                   for i in range(nCells)]
            boundaryIDs.append(np.unique(np.array([item for elem in IDs for item in elem])))

    directory = "/Users/jwtan/PycharmProjects/PyDG/data/2D_adr/"
    plt_name = test_case + "_P" + str(P1) + "_N" + str(nCells) + "_IC"
    # title = "Initial condition"
    # plotSolution2d(mesh, nCells, nCoords, nVars, left_boundary, right_boundary, bottom_boundary, top_boundary,
    #                directory, exact_soln, " ", title, plt_name, save=True)

    CFL = 0.01
    # Constant mesh size for now
    deltax = mesh.connectivityData.cells[0].facesCell.F[0].calculations.V
    deltay = mesh.connectivityData.cells[0].facesCell.F[1].calculations.V

    """ Initialise global block-matrix coefficients """
    """ Global diagonal upwind matrix """
    globalDiag = np.zeros((nCells, nCells, dims, dims))

    """ Global off-diagonal matrix """
    globalOffDiag = np.zeros_like(globalDiag)

    """ Global mass matrix """
    massMatrices = np.zeros_like(globalDiag)

    """ Global stiffness matrix """
    stiffnessMatrices = np.zeros_like(globalDiag)

    """ Global Laplacian matrix """
    laplacian = np.zeros_like(globalDiag)

    """ Global SIP matrix, x4 [dims, dims] blocks from each face """
    sipMatrices = np.zeros_like(globalDiag)

    """ Global RHS coefficients """
    fCoeffsGlobal = np.zeros((nCells, dims))

    """ Penalty factor, 10 to 20 in paper """
    # mesh.connectivityData.cells[0].calculations.V
    eta_x = (P1 ** 2) * (10.0 / deltax)
    eta_y = (P2 ** 2) * (10.0 / deltay)
    eta_ = [eta_x, eta_y]

    """ Global solution coefficients """
    u = np.block([
        [mesh.connectivityData.cells[i].solnCell.uCoeffs] for i in range(nCells)
    ])
    uCoeffsGlobal = np.empty_like(u)

    """ Domain boundary condition contributions, Dirichlet or Periodic or zero Neumann """
    """ SIP shared faces indices convention """
    sipNeighbourFaces = [3, 0]
    sipOwnerFaces = [1, 2]  # all internal faces

    """ Upwind shared faces indices convention """
    upwindNeighbourFaces = [2, 1]
    upwindOwnerFaces = [0, 3]  # receiving flow from bottom and left

    BCs = [bottom_BCs, right_BCs, top_BCs, left_BCs]
    BC_coord = [bottom_boundary, right_boundary, top_boundary, left_boundary]
    n_list = [1, -1, -1, 1]
    x = None
    y = None
    dxi1dx = None
    dxi2dx = None

    """ Iterate through domain boundaries """
    for j in range(4):
        if BCs[j] == "Dirichlet":
            for i in range(len(boundaryIDs[j])):
                """ SIP """
                sipMatrices[boundaryIDs[j][i]][boundaryIDs[j][i]] = \
                    sipFluxBoundary(mesh.connectivityData.cells[boundaryIDs[j][i]],
                                    mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j], eta_[j % 2])
                """ Varying x and y-coordinates of face Gauss points """
                if j == 0 or j == 2:
                    x = mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].geomCell.geometry. \
                        parametricMapping()
                    y = BC_coord[j]

                    dxi1dx = mesh.connectivityData.cells[boundaryIDs[j][i]].geomCell.invJacobian[:, :, 0][:, 1]
                    dxi2dx = mesh.connectivityData.cells[boundaryIDs[j][i]].geomCell.invJacobian[:, :, 1][:, 1]
                elif j == 1 or j == 3:
                    y = mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].geomCell.geometry. \
                        parametricMapping()
                    x = BC_coord[j]

                    dxi1dx = mesh.connectivityData.cells[boundaryIDs[j][i]].geomCell.invJacobian[:, :, 0][:, 0]
                    dxi2dx = mesh.connectivityData.cells[boundaryIDs[j][i]].geomCell.invJacobian[:, :, 1][:, 0]
                else:
                    raise NotImplementedError

                weight_matrix = mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].paramSeg.weights * \
                                abs(mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].geomCell.detJacobian)
                tmpOwnerBasis = mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].faceBasis
                tmpOwnerDeriv = mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].faceDeriv
                term1 = n_list[j] * np.matmul(
                    (dxi1dx * tmpOwnerDeriv[:, :, 0] + dxi2dx * tmpOwnerDeriv[:, :, 1]).transpose(),
                    weight_matrix * g_d(x, y))
                term2 = eta_[j % 2] * np.matmul(tmpOwnerBasis.transpose(), weight_matrix * g_d(x, y))

                fCoeffsGlobal[boundaryIDs[j][i]] += kappa * (term1 + term2).reshape(-1)

                """ Upwind """
                vel = np.array(
                    [np.matmul(a, mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].
                               unitNormal[k].transpose())
                     for k in range(len(mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].
                                        paramSeg.zeros))])

                faceMatrix = np.matmul(tmpOwnerBasis.transpose(), g_d(x, y) * vel.reshape(-1, 1) *
                                       mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].
                                       paramSeg.weights *
                                       abs(mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].
                                           geomCell.detJacobian))
                fCoeffsGlobal[boundaryIDs[j][i]] -= faceMatrix.reshape(-1)

        elif BCs[j] == "Periodic":
            for i in range(len(boundaryIDs[j])):
                """ SIP """
                sipMatrices[boundaryIDs[j][i]][boundaryIDs[j][i]] = \
                    sipFluxBoundary(mesh.connectivityData.cells[boundaryIDs[j][i]],
                                    mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j], eta_[j % 2])
                if j != 1 and j != 2:
                    """ Upwind """
                    tmpOwnerBasis = mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].faceBasis
                    vel = np.array(
                        [np.matmul(a, mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].
                                   unitNormal[k].transpose())
                         for k in range(len(mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].
                                            paramSeg.zeros))])
                    tmpNeighbourBasis = mesh.connectivityData.cells[boundaryIDs[j - 2][i]].facesCell.F[j - 2].faceBasis

                    faceMatrix = np.matmul(tmpOwnerBasis.transpose(), tmpNeighbourBasis * vel.reshape(-1, 1) *
                                           mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].
                                           paramSeg.weights *
                                           abs(mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].
                                               geomCell.detJacobian))
                    globalOffDiag[boundaryIDs[j - 2][i]][boundaryIDs[j][i]] += faceMatrix

    """ Populate internal values """
    for i in range(nCells):
        laplacian[i][i] = mesh.connectivityData.cells[i].laplacianMatrix
        massMatrices[i][i] = mesh.connectivityData.cells[i].massMatrix
        # different velocity requires mod
        stiffnessMatrices[i][i] = - a[0] * mesh.connectivityData.cells[i].stiffnessMatrix[0] - \
                                  a[1] * mesh.connectivityData.cells[i].stiffnessMatrix[1]

        sipNeighbourIDs = [mesh.connectivityData.cells[i].geomData.neighbourLabels[1],
                           mesh.connectivityData.cells[i].geomData.neighbourLabels[2]]

        upwindNeighbourIDs = [mesh.connectivityData.cells[i].geomData.neighbourLabels[upwindOwnerFaces[0]],
                              mesh.connectivityData.cells[i].geomData.neighbourLabels[upwindOwnerFaces[1]]]

        for j in range(len(sipNeighbourIDs)):  # faces F1 and F2
            if sipNeighbourIDs[j] is not None:
                tmpOwnerBasis = mesh.connectivityData.cells[i].facesCell.F[sipOwnerFaces[j]].faceBasis
                tmpNeighbourBasis = mesh.connectivityData.cells[sipNeighbourIDs[j]].facesCell.F[sipNeighbourFaces[j]]. \
                    faceBasis
                tmpOwnerDeriv = mesh.connectivityData.cells[i].facesCell.F[sipOwnerFaces[j]].faceDeriv
                tmpNeighbourDeriv = mesh.connectivityData.cells[sipNeighbourIDs[j]].facesCell.F[sipNeighbourFaces[j]]. \
                    faceDeriv

                temp1, temp2, temp3, temp4 = \
                    sipFlux(mesh.connectivityData.cells[i], mesh.connectivityData.cells[sipNeighbourIDs[j]],
                            mesh.connectivityData.cells[i].facesCell.F[sipOwnerFaces[j]],
                            tmpOwnerBasis, tmpNeighbourBasis, tmpOwnerDeriv, tmpNeighbourDeriv,
                            mesh.connectivityData.cells[i].facesCell.F[sipOwnerFaces[j]].unitNormal[0], eta_[1 - j])

                sipMatrices[i][i] += temp1
                sipMatrices[i][sipNeighbourIDs[j]] += temp2
                sipMatrices[sipNeighbourIDs[j]][i] += temp3
                sipMatrices[sipNeighbourIDs[j]][sipNeighbourIDs[j]] += temp4

            """ Upwind, local contributions """
            tmpBasis = mesh.connectivityData.cells[i].facesCell.F[j + 1].faceBasis
            vel = np.array([np.matmul(a, mesh.connectivityData.cells[i].facesCell.F[j + 1].unitNormal[k].transpose())
                            for k in range(len(mesh.connectivityData.cells[i].facesCell.F[j + 1].paramSeg.zeros))])

            faceMatrix = np.matmul(tmpBasis.transpose(), tmpBasis * vel.reshape(-1, 1) *
                                   mesh.connectivityData.cells[i].facesCell.F[j + 1].paramSeg.weights *
                                   abs(mesh.connectivityData.cells[i].facesCell.F[j + 1].geomCell.detJacobian))
            globalDiag[i][i] += faceMatrix

            """ Upwind, find internal cells and calculate neighbour contributions from faces F0 and F3 """
            if upwindNeighbourIDs[j] is not None:
                tmpOwnerBasis = mesh.connectivityData.cells[i].facesCell.F[upwindOwnerFaces[j]].faceBasis
                vel = np.array(
                    [np.matmul(a, mesh.connectivityData.cells[i].facesCell.F[upwindOwnerFaces[j]].unitNormal[
                        k].transpose())
                     for k in range(
                        len(mesh.connectivityData.cells[i].facesCell.F[upwindOwnerFaces[j]].paramSeg.zeros))])
                tmpNeighbourBasis = mesh.connectivityData.cells[upwindNeighbourIDs[j]].facesCell. \
                    F[upwindNeighbourFaces[j]].faceBasis

                faceMatrix = np.matmul(tmpOwnerBasis.transpose(), tmpNeighbourBasis * vel.reshape(-1, 1) *
                                       mesh.connectivityData.cells[i].facesCell.F[
                                           upwindOwnerFaces[j]].paramSeg.weights *
                                       abs(mesh.connectivityData.cells[i].facesCell.F[upwindOwnerFaces[j]].
                                           geomCell.detJacobian))
                globalOffDiag[upwindNeighbourIDs[j]][i] += faceMatrix

        tempf = forcing * np.ones((nCoords, nVars))

        fCoeffsGlobal[i] += np.matmul(mesh.connectivityData.cells[i].matCell.basisMatrix.transpose(),
                                      tempf * mesh.connectivityData.cells[i].paramQuad.weights * \
                                      abs(mesh.connectivityData.cells[i].geomCell.detJacobian)).reshape(-1)

    """ Row runs first then column !! """
    A = laplacian + sipMatrices
    A = np.block([
        [A[j][i] for j in range(nCells)] for i in range(nCells)
    ])
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
    f = np.block([
        [fCoeffsGlobal[i]] for i in range(nCells)
    ]).reshape(-1, 1)

    time_loop = True
    """ Calculate time-step size """
    if abs(endTime - 0.) < 1e-6:  # no time derivative, pure second-order solver
        M = np.zeros_like(massMatrices)
        time_loop = False
    else:
        deltaT = CFL / ((a[0] / deltax) + (a[1] / deltay))
        if abs(a[0] - 0.) < 1e-6 and abs(a[1] - 0.) < 1e-6:  # no advection, pure diffusion
            deltaT = 0.1
        M = massMatrices / deltaT

    globalMatrix = M + globalDiag + stiffnessMatrices + globalOffDiag + kappa * A
    print("Inverting global matrix.")
    invGlobalMatrix = np.linalg.inv(globalMatrix)
    RHS = f + np.matmul(M, u.reshape(-1, 1))

    """ Create a gif """
    xCoords = np.array([[mesh.connectivityData.cells[i].GetQuadratureCoords[:, j][0] for j in range(nCoords)]
                        for i in range(nCells)])
    yCoords = np.array([[mesh.connectivityData.cells[i].GetQuadratureCoords[:, j][1] for j in range(nCoords)]
                        for i in range(nCells)])
    X, Y = np.meshgrid(xCoords, yCoords)
    frame_rate = 0.002
    no_of_frames = int(endTime / frame_rate) + 1
    Z = np.zeros((no_of_frames, X.shape[0], Y.shape[0]))
    frame_counter = 0
    record_time = 0.0

    gif_data(mesh, u, Z, X, Y, xCoords, yCoords, nCells, nCoords, nVars, dims, frame_counter)
    frame_counter += 1
    record_time += frame_rate

    if time_loop:
        """ Start time-loop """
        while endTime - time > 1e-10:

            """ Iterate through domain boundaries """
            for j in range(4):
                if BCs[j] == "Periodic":
                    for i in range(len(boundaryIDs[j])):
                        tmpNeighbourBasis = mesh.connectivityData.cells[boundaryIDs[j - 2][i]].facesCell.F[j - 2]. \
                            faceBasis
                        g_d = np.matmul(tmpNeighbourBasis, u.reshape(nCells, dims)[boundaryIDs[j - 2][i]].
                                        reshape(-1, 1))
                        if j == 0 or j == 2:
                            dxi1dx = mesh.connectivityData.cells[boundaryIDs[j][i]].geomCell.invJacobian[:, :, 0][:, 1]
                            dxi2dx = mesh.connectivityData.cells[boundaryIDs[j][i]].geomCell.invJacobian[:, :, 1][:, 1]
                        elif j == 1 or j == 3:
                            dxi1dx = mesh.connectivityData.cells[boundaryIDs[j][i]].geomCell.invJacobian[:, :, 0][:, 0]
                            dxi2dx = mesh.connectivityData.cells[boundaryIDs[j][i]].geomCell.invJacobian[:, :, 1][:, 0]
                        else:
                            raise NotImplementedError

                        weight_matrix = mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].paramSeg.weights \
                                        * abs(mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].geomCell.
                                              detJacobian)
                        tmpOwnerBasis = mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].faceBasis
                        tmpOwnerDeriv = mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].faceDeriv
                        term1 = n_list[j] * np.matmul(
                            (dxi1dx * tmpOwnerDeriv[:, :, 0] + dxi2dx * tmpOwnerDeriv[:, :, 1]).transpose(),
                            weight_matrix * g_d)
                        term2 = eta_[j % 2] * np.matmul(tmpOwnerBasis.transpose(), weight_matrix * g_d)

                        fCoeffsGlobal[boundaryIDs[j][i]] = kappa * (term1 + term2).reshape(-1)

            if bottom_BCs or right_BCs or top_BCs or left_BCs == "Periodic":
                f = np.block([
                    [fCoeffsGlobal[i]] for i in range(nCells)
                ])

            RHS = f + np.matmul(M, u.reshape(-1, 1))

            uCoeffsGlobal = np.matmul(invGlobalMatrix, RHS)

            u = uCoeffsGlobal

            # Increment time
            if time + deltaT > endTime:
                deltaT = endTime - time
            time += deltaT
            print("Current time: " + str(time))

            # if abs(record_time - time) < 1e-8:
            #     gif_data(mesh, u, Z, X, Y, xCoords, yCoords, nCells, nCoords, nVars, dims, frame_counter)
            #     frame_counter += 1
            #     record_time += frame_rate

    else:
        uCoeffsGlobal = np.matmul(invGlobalMatrix, RHS)  # spsolve(A, RHS) for iterative methods

    """ No limiter solution """
    for i in range(nCells):
        mesh.connectivityData.cells[i].solnCell.uCoeffs = uCoeffsGlobal.reshape(nCells, dims)[i].reshape(-1, 1)
        mesh.connectivityData.cells[i].solnCell.uPhysical = np.matmul(
            mesh.connectivityData.cells[i].matCell.basisMatrix, mesh.connectivityData.cells[i].solnCell.uCoeffs)

    """ Plot solution """
    plt_name = test_case + "_P" + str(P1) + "_T" + str(endTime) + "_N" + str(nCells)
    title = "Numerical solution, final time = {0} s".format(str(endTime))
    dat = np.array([xCoords, yCoords])
    a = np.column_stack(dat)
    hdrtxt = '# rc in AU, #vc in km/s'
    np.savetxt('test.dat', a, delimiter=' ', header=title)

    # plotSolution2d(mesh, nCells, nCoords, nVars, left_boundary, right_boundary,
    #                bottom_boundary, top_boundary, directory, exact_soln, " ", title,
    #                plt_name, save=True)

    """ Calculate L2 error """
    # calculateL2err2d(mesh, exact_soln, nCells)

    # gif_name = plt_name + "_movie"
    # filenames = []
    # for i in range(no_of_frames):
    #     levels = MaxNLocator(nbins=20).tick_values(Z[i].min(), Z[i].max())
    #     cmap = plt.get_cmap('coolwarm')
    #     norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    #
    #     fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    #
    #     cf = ax.contourf(X, Y, Z[i], levels=levels, cmap=cmap)
    #     fig.colorbar(cf, ax=ax)
    #
    #     # Plot the surface
    #     ax.title.set_text("Numerical solution, time = {0}".format(str(i * frame_rate)))
    #     ax.set_xlim(left_boundary, right_boundary)
    #     ax.set_ylim(bottom_boundary, top_boundary)
    #     color_tuple = (1.0, 1.0, 1.0, 0.0)
    #
    #     # Customize the axes
    #     ax.xaxis.set_major_formatter('{x:.01f}')
    #     ax.yaxis.set_major_formatter('{x:.01f}')
    #
    #     # create file name and append it to a list
    #     filename = f'{i}.png'
    #     filenames.append(filename)
    #
    #     # save frame
    #     plt.savefig(filename)
    #     plt.close()
    #
    # # build gif
    # with imageio.get_writer(f'/Users/jwtan/PycharmProjects/PyDG/src/solvers/gifs/{gif_name}.gif', mode='I') as writer:
    #     for filename in filenames:
    #         image = imageio.v2.imread(filename)
    #         writer.append_data(image)
    #
    # # Remove files
    # for filename in set(filenames):
    #     os.remove(filename)
