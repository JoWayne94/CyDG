"""
File: ADR.py

Description: Solve linear advection diffusion equation implicitly in 1D with backward Euler time-stepping
"""
import numpy as np
import math
from src.solvers.explicitAdv import *
from scipy.sparse.linalg import spsolve
from src.library.dgMesh.dgMesh import *
from src.library.paramCells.basis import *


def sipFlux(cell_left, cell_right, bleft, bright, dleft, dright, n):
    """
    @:brief Symmetric Interior Penalty method
    :param cell_left:   Owner cell
    :param cell_right:  Neighbour cell
    :param bleft:       Owner face basis matrix
    :param bright:      Neighbour face basis matrix
    :param dleft:       Owner face derivative matrix
    :param dright:      Neighbour face derivative matrix
    :param n:           Owner face unit normal vector
    :return: Block-matrix contributions
    """
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
    """
    @:brief Symmetric Interior Penalty method on Dirichlet boundaries
    :param cell_left: Boundary cell
    :param bleft:     Boundary face basis matrix
    :param dleft:     Boundary face derivative matrix
    :param n:         Boundary face unit normal vector
    :return: Diagonal block coefficient contribution
    """
    dxi1dx1_left = cell_left.geomCell.geometry.dxi1dx1

    sip_left = n * (- np.matmul(bleft.transpose(), (dxi1dx1_left * dleft)[:, :, 0])
                    - np.matmul((dxi1dx1_left * dleft)[:, :, 0].transpose(), bleft)) \
               + eta * np.matmul(bleft.transpose(), bleft)

    return sip_left


def G(x, gamma_, z_):
    return np.exp(- gamma_ * (x - z_) ** 2)


def F(x, alpha_, b_):
    return np.sqrt(max(1 - alpha_ ** 2 * (x - b_) ** 2, 0.))


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

    """ Set test case """
    time = 0.
    deltaT = 0.
    endTime = 0.
    nCells = len(mesh.connectivityData.cells)
    quadCoords = mesh.connectivityData.cells[0].GetQuadratureCoords

    """ Number of quadrature points within a single cell """
    nCoords = len(quadCoords)

    """ Left and right face state values extrapolated using basis matrices at -1 and 1 in parametric space """
    basisMatrixforF0 = Legendre1d(np.array([[-1]]), P1)
    basisMatrixforF1 = Legendre1d(np.array([[1]]), P1)
    derivMatrixforF0 = Legendre1dGrad(np.array([[-1]]), P1)
    derivMatrixforF1 = Legendre1dGrad(np.array([[1]]), P1)

    """
    Set initial values using the first coefficients for constant state, or using physical values then 
    transform back to coefficient space
    """
    test_case = "pure_advection_sine_wave"
    a = 0.  # constant velocity
    kappa = 0.
    forcing = 0.
    g_d_left = None
    g_d_right = None
    numerical_flux = upwind_flux
    exact_soln = np.empty((nCells, nCoords, nVars))

    if test_case == "pure_advection_sine_wave":
        a = 1.
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

            exact_soln[i] = mesh.connectivityData.cells[i].solnCell.uPhysical

    elif test_case == "pure_advection_step_profile":
        u_l = 0.
        u_r = 1.
        x_l = 0.25
        x_r = 0.75
        a = 1.
        endTime = 1.0
        flux = linearAdvFlux
        boundaryConditions = "Periodic"

        for i in range(nCells):
            # Initial condition for step profile
            mesh.connectivityData.cells[i].solnCell.uCoeffs[0] = np.array(
                [u_r if x_l <= coords <= x_r else u_l for coords in
                 mesh.connectivityData.cells[i].calculations.cellCentre])
            # Set physical values, no need * abs(mesh.connectivityData.cells[i].geomCell.detJacobian)
            mesh.connectivityData.cells[i].solnCell.uPhysical = np.matmul(
                mesh.connectivityData.cells[i].matCell.basisMatrix, mesh.connectivityData.cells[i].solnCell.uCoeffs)

            exact_soln[i] = mesh.connectivityData.cells[i].solnCell.uPhysical

    elif test_case == "pure_advection_shu_test":
        a = 1.
        endTime = 2.0
        flux = linearAdvFlux
        boundaryConditions = "Periodic"

        delta = 0.005
        z = -0.7
        gamma = np.log(2) / (36 * delta ** 2)
        b = 0.5
        alpha = 10

        for i in range(nCells):
            # Initial condition for shu test
            for coords in range(nCoords):
                x_coord = mesh.connectivityData.cells[i].GetQuadratureCoords[coords]
                if -0.8 <= x_coord <= -0.6:
                    mesh.connectivityData.cells[i].solnCell.uPhysical[coords] = \
                        (1 / 6) * (G(x_coord, gamma, z - delta) + G(x_coord, gamma, z + delta) + 4 * G(x_coord, gamma,
                                                                                                       z))
                elif -0.4 <= x_coord <= -0.2:
                    mesh.connectivityData.cells[i].solnCell.uPhysical[coords] = 1.
                elif 0. <= x_coord <= 0.2:
                    mesh.connectivityData.cells[i].solnCell.uPhysical[coords] = 1. - abs(10 * (x_coord - 0.1))
                elif 0.4 <= x_coord <= 0.6:
                    mesh.connectivityData.cells[i].solnCell.uPhysical[coords] = \
                        (1 / 6) * (F(x_coord, alpha, b - delta) + F(x_coord, alpha, b + delta) + 4 * F(x_coord, alpha,
                                                                                                       b))
                else:
                    mesh.connectivityData.cells[i].solnCell.uPhysical[coords] = 0.

            mesh.connectivityData.cells[i].solnCell.uCoeffs = \
                forwardTransform(mesh, mesh.connectivityData.cells[i].solnCell.uPhysical, i)

            exact_soln[i] = mesh.connectivityData.cells[i].solnCell.uPhysical

    elif test_case == "Laplace":
        g_d_left = 5.
        g_d_right = 2.
        kappa = 1.
        forcing = 0.  # -Laplace(u) = forcing
        boundaryConditions = "Dirichlet"

        """ No initial conditions """
        m = ((g_d_right - g_d_left) / (1. + 1.))
        c = 2. - m * 1.
        for i in range(nCells):
            for coords in range(nCoords):
                exact_soln[i][coords] = m * mesh.connectivityData.cells[i].GetQuadratureCoords[coords] + c

    elif test_case == "Poisson":
        g_d_left = 0.
        g_d_right = 0.
        kappa = 1.
        forcing = 2.  # -Laplace(u) = forcing
        boundaryConditions = "Dirichlet"

        """ No initial conditions """

        for i in range(nCells):
            for coords in range(nCoords):
                exact_soln[i][coords] = 1. - mesh.connectivityData.cells[i].GetQuadratureCoords[coords] ** 2

    elif test_case == "Poisson2":
        g_d_left = 0.
        g_d_right = 0.
        kappa = 1.
        boundaryConditions = "Dirichlet"

        """ No initial conditions """

        for i in range(nCells):
            for coords in range(nCoords):
                exact_soln[i][coords] = np.sin(np.pi * mesh.connectivityData.cells[i].GetQuadratureCoords[coords])

    elif test_case == "pure_diffusion_sine_wave":
        g_d_left = 0.
        g_d_right = 0.
        kappa = 1.
        forcing = 0.  # du/dt - Laplace(u) = forcing
        endTime = 1.6
        boundaryConditions = "Dirichlet"

        for i in range(nCells):
            # Initial condition for sine wave
            for coords in range(nCoords):
                mesh.connectivityData.cells[i].solnCell.uPhysical[coords] = \
                    np.sin(np.pi / 2 * (mesh.connectivityData.cells[i].GetQuadratureCoords[coords] + 1))

            mesh.connectivityData.cells[i].solnCell.uCoeffs = \
                forwardTransform(mesh, mesh.connectivityData.cells[i].solnCell.uPhysical, i)

    elif test_case == "pure_diffusion_segment":
        g_d_left = 5.
        g_d_right = 17.
        kappa = 1.
        forcing = 0.  # du/dt - Laplace(u) = forcing
        endTime = 2.2
        boundaryConditions = "Dirichlet"

        for i in range(nCells):
            # Zeros initial condition, not necessary
            mesh.connectivityData.cells[i].solnCell.uCoeffs[0] = 0.
            # Set physical values
            mesh.connectivityData.cells[i].solnCell.uPhysical = np.matmul(
                mesh.connectivityData.cells[i].matCell.basisMatrix, mesh.connectivityData.cells[i].solnCell.uCoeffs)

    elif test_case == "advection_diffusion_sine_wave":
        a = 1.
        kappa = 0.01
        forcing = 0.  # du/dt + a du/dx - Laplace(u) = forcing
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

    else:
        raise NotImplementedError

    if a < 0:
        print("Advection velocity has to be positive for now.")
        raise ValueError
    if kappa < 0:
        print("Negative diffusivity will blow up the solution.")
        raise ValueError

    CFL = 0.01
    # Constant mesh size for now
    deltax = mesh.connectivityData.cells[0].calculations.V

    """ Initialise global block-matrix coefficients """
    """ Global diagonal upwind matrix """
    globalDiag = np.zeros((nCells, nCells, P1 + 1, P1 + 1))

    """ Global off-diagonal matrix """
    globalOffDiag = np.zeros_like(globalDiag)

    """ Global mass matrix """
    massMatrices = np.zeros_like(globalDiag)

    """ Global stiffness matrix """
    stiffnessMatrices = np.zeros_like(globalDiag)

    """ Global Laplacian matrix """
    laplacian = np.zeros_like(globalDiag)

    """ Global SIP matrix, x4 [P1 + 1, P1 + 1] blocks from each surface integral """
    sipMatrices = np.zeros_like(globalDiag)

    """ Global RHS coefficients """
    fCoeffsGlobal = np.zeros((nCells, P1 + 1))

    """ Penalty factor, 10 to 20 in paper """
    eta = (P1 ** 2) * (10.0 / deltax)

    """ Global solution coefficients """
    u = np.block([
        [mesh.connectivityData.cells[i].solnCell.uCoeffs] for i in range(nCells)
    ])
    uCoeffsGlobal = np.empty_like(u)

    """ Domain boundary condition contributions """
    # Dirichlet or Periodic contributions at both ends
    sipMatrices[0][0] = \
        sipFluxBoundary(mesh.connectivityData.cells[0], basisMatrixforF0, derivMatrixforF0,
                        mesh.connectivityData.cells[0].calculations.faceNormals[0])

    sipMatrices[-1][-1] = \
        sipFluxBoundary(mesh.connectivityData.cells[-1], basisMatrixforF1, derivMatrixforF1,
                        mesh.connectivityData.cells[-1].calculations.faceNormals[1])

    """ Dirichlet contributions from SIP method, came from LHS hence kappa needed """
    if boundaryConditions == "Dirichlet":
        fCoeffsGlobal[0] += kappa * (- g_d_left * mesh.connectivityData.cells[0].calculations.faceNormals[0] *
                                     mesh.connectivityData.cells[0].geomCell.geometry.dxi1dx1 *
                                     derivMatrixforF0[:, :, 0] + eta * g_d_left * basisMatrixforF0).reshape(-1)

        fCoeffsGlobal[-1] += kappa * (- g_d_right * mesh.connectivityData.cells[-1].calculations.faceNormals[1] *
                                      mesh.connectivityData.cells[-1].geomCell.geometry.dxi1dx1 *
                                      derivMatrixforF1[:, :, 0] + eta * g_d_right * basisMatrixforF1).reshape(-1)

    if boundaryConditions == "Periodic":
        """ Upwind """
        globalOffDiag[-1][0] -= a * np.matmul(basisMatrixforF0.transpose(), basisMatrixforF1)

        """ Central flux """
        # globalOffDiag[-1][0] -= (a / 2) * np.matmul(basisMatrixforF0.transpose(), basisMatrixforF1)
        # globalOffDiag[0][-1] += (a / 2) * np.matmul(basisMatrixforF1.transpose(), basisMatrixforF0)

    """ Populate internal faces -> surface integral values for off-diagonal block coefficients """
    for i in range(nCells - 1):
        """ Upwind """
        globalOffDiag[i][i + 1] -= a * np.matmul(basisMatrixforF0.transpose(), basisMatrixforF1)

        """ Central flux """
        # globalOffDiag[i][i + 1] -= (a / 2) * np.matmul(basisMatrixforF0.transpose(), basisMatrixforF1)
        # globalOffDiag[i + 1][i] += (a / 2) * np.matmul(basisMatrixforF1.transpose(), basisMatrixforF0)

        temp1, temp2, temp3, temp4 = \
            sipFlux(mesh.connectivityData.cells[i], mesh.connectivityData.cells[i + 1],
                    basisMatrixforF1, basisMatrixforF0, derivMatrixforF1, derivMatrixforF0,
                    mesh.connectivityData.cells[i].calculations.faceNormals[1])
        sipMatrices[i][i] += temp1
        sipMatrices[i][i + 1] += temp2
        sipMatrices[i + 1][i] += temp3
        sipMatrices[i + 1][i + 1] += temp4

    """ Go through all cells """
    for i in range(nCells):
        laplacian[i][i] = mesh.connectivityData.cells[i].laplacianMatrix
        massMatrices[i][i] = mesh.connectivityData.cells[i].massMatrix
        stiffnessMatrices[i][i] = -a * mesh.connectivityData.cells[i].stiffnessMatrix

        tempf = forcing * np.ones((nCoords, nVars))
        # tempf = np.empty((nCoords, nVars))
        # for coord in range(nCoords):
        #     tempf[coord][0] = np.pi ** 2 * \
        #             np.sin(np.pi * mesh.connectivityData.cells[i].GetQuadratureCoords[coord])

        fCoeffsGlobal[i] += np.matmul(mesh.connectivityData.cells[i].matCell.basisMatrix.transpose(),
                                      tempf * mesh.connectivityData.cells[i].paramSeg.weights * \
                                      abs(mesh.connectivityData.cells[i].geomCell.detJacobian)).reshape(-1)

        """ Upwind """
        globalDiag[i][i] += a * np.matmul(basisMatrixforF1.transpose(), basisMatrixforF1)

        """ Central flux """
        # globalDiag[i][i] += (a / 2) * (np.matmul(basisMatrixforF1.transpose(), basisMatrixforF1) -
        #                                np.matmul(basisMatrixforF0.transpose(), basisMatrixforF0))

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
    ])

    time_loop = True
    """ Calculate time-step size """
    if abs(endTime - 0.) < 1e-6:  # no time derivative, pure second-order solver
        M = np.zeros_like(massMatrices)
        time_loop = False
    else:
        deltaT = CFL * deltax / a
        if abs(a - 0.) < 1e-6:  # no advection, pure diffusion
            deltaT = 0.1
        M = massMatrices / deltaT

    globalMatrix = M + globalDiag + stiffnessMatrices + globalOffDiag + kappa * A
    print("Inverting global matrix.")
    invGlobalMatrix = np.linalg.inv(globalMatrix)
    RHS = f.reshape(-1, 1) + np.matmul(M, u.reshape(-1, 1))

    """ Create a gif """
    # xCoords = np.array([[mesh.connectivityData.cells[i].GetQuadratureCoords[j] for j in range(nCoords)]
    #                     for i in range(nCells)])
    # frame_rate = 0.1
    # no_of_frames = int(endTime / frame_rate) + 1
    # Z = np.zeros((no_of_frames, xCoords.shape[0]))
    # frame_counter = 0
    # record_time = 0.0
    #
    # gif_data(mesh, u, Z, xCoords, nCells, nCoords, nVars, dims, frame_counter)
    # frame_counter += 1
    # record_time += frame_rate

    if time_loop:
        """ Start time-loop """
        while endTime - time > 1e-10:

            if boundaryConditions == "Periodic":
                """ Periodic SIP is just a dynamic Dirichlet BC """
                u_l = np.matmul(basisMatrixforF1, u.reshape(nCells, P1 + 1)[-1].reshape(-1, 1))
                u_r = np.matmul(basisMatrixforF0, u.reshape(nCells, P1 + 1)[0].reshape(-1, 1))
                # Revisit if Advection Diffusion equation has an inhomogeneous forcing function, has to +=
                fCoeffsGlobal[0] = kappa * (- u_l * mesh.connectivityData.cells[0].calculations.faceNormals[0] *
                                            mesh.connectivityData.cells[0].geomCell.geometry.dxi1dx1 *
                                            derivMatrixforF0[:, :, 0] + eta * u_l * basisMatrixforF0).reshape(-1)

                fCoeffsGlobal[-1] = kappa * (- u_r * mesh.connectivityData.cells[-1].calculations.faceNormals[1] *
                                             mesh.connectivityData.cells[-1].geomCell.geometry.dxi1dx1 *
                                             derivMatrixforF1[:, :, 0] + eta * u_r * basisMatrixforF1).reshape(-1)
                f = np.block([
                    [fCoeffsGlobal[i]] for i in range(nCells)
                ])
            """ Initialise new RHS coefficients """
            RHS = f.reshape(-1, 1) + np.matmul(M, u.reshape(-1, 1))

            uCoeffsGlobal = np.matmul(invGlobalMatrix, RHS)  # spsolve(globalMatrix, RHS)

            # if abs(time - 0.25) < 1e-6 or abs(time - 0.75) < 1e-6:
            #     for i in range(nCells):
            #         mesh.connectivityData.cells[i].solnCell.uCoeffs = uCoeffsGlobal.reshape(nCells, P1 + 1)[i].reshape(
            #             -1, 1)
            #     label = "{time:.1f}".format(time=time)
            #     plotSolution(mesh, nCells, nCoords, nVars, test_case, P1, label)

            u = uCoeffsGlobal

            # Increment time
            if time + deltaT > endTime:
                deltaT = endTime - time
            time += deltaT
            print("Current time: " + str(time))

    else:
        uCoeffsGlobal = np.matmul(invGlobalMatrix, RHS)  # spsolve(A, RHS) for iterative methods

    # np.set_printoptions(precision=2, suppress=True)
    """ No limiter solution, unwrap global coefficients """
    for i in range(nCells):
        mesh.connectivityData.cells[i].solnCell.uCoeffs = uCoeffsGlobal.reshape(nCells, P1 + 1)[i].reshape(-1, 1)
        mesh.connectivityData.cells[i].solnCell.uPhysical = np.matmul(
            mesh.connectivityData.cells[i].matCell.basisMatrix, mesh.connectivityData.cells[i].solnCell.uCoeffs)

    """ Plot solution """
    directory = "/Users/jwtan/PycharmProjects/PyDG/data/ADR/"
    plt_name = test_case + "_P" + str(P1) + "_T" + str(endTime) + "_N" + str(nCells) + "_CFL" + str(CFL)
    title = "AD equation, final time = {0} s".format(str(endTime))
    # title = "Laplace's equation"
    plotSolution(mesh, nCells, nCoords, nVars, P1, 0., 1., directory, exact_soln, "Numerical solution", title,
                 plt_name,
                 save=True)

    """ Calculate L2 error """
    # calculateL2err(mesh, exact_soln, nCells)
