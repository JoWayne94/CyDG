"""
File: non_const_Adv.py

Description: Solve non-constant linear advection equation implicitly in 2D with backward Euler time-stepping
"""
import numpy as np
import math
from src.solvers.ADR2d import *
from src.library.dgMesh.dgMesh import *
from src.library.paramCells.basis import *

if __name__ == '__main__':
    """
    main()
    """

    name = "/Users/jwtan/PycharmProjects/PyDG/polyMesh/20x20"
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
    test_case = "pure_advection_non_constant_vel"
    a = np.array([0., 0.])  # constant velocity
    kappa = 0.
    forcing = 0.
    point_source = None
    left_boundary = None
    bottom_boundary = None
    right_boundary = None
    top_boundary = None
    g_d_left = None
    g_d_right = None
    g_d_top = None
    g_d_bottom = None
    numerical_flux = upwind_flux

    if test_case == "pure_advection_non_constant_vel":
        u_l = 0.
        u_r = 1.
        x_centre = 0.
        y_centre = -1.25
        a = np.empty((nCells, nCoords, 2))
        endTime = 8.
        left_boundary = -2.
        bottom_boundary = -2.
        right_boundary = 2.
        top_boundary = 2.
        flux = linearAdvFlux
        left_BCs = "Dirichlet"
        bottom_BCs = "Dirichlet"
        right_BCs = "Dirichlet"
        top_BCs = "Dirichlet"
        g_d = np.array([0., 0., 0., 0.])

        for i in range(nCells):
            # Initial condition for gaussian pulse
            for coords in range(nCoords):
                mesh.connectivityData.cells[i].solnCell.uPhysical[coords] = \
                    np.exp(-32. * ((mesh.connectivityData.cells[i].GetQuadratureCoords[:, coords][0] - x_centre) ** 2 +
                                   (mesh.connectivityData.cells[i].GetQuadratureCoords[:, coords][1] - y_centre) ** 2))

                a[i][coords] = np.array([-mesh.connectivityData.cells[i].GetQuadratureCoords[:, coords][1],
                                         mesh.connectivityData.cells[i].GetQuadratureCoords[:, coords][0]])

            mesh.connectivityData.cells[i].solnCell.uCoeffs = \
                forwardTransform(mesh, mesh.connectivityData.cells[i].solnCell.uPhysical, i)

    else:
        raise NotImplementedError

    plotSolution2d(mesh, nCells, nCoords, nVars, test_case, P1, P2, 0.0, left_boundary, right_boundary,
                   bottom_boundary, top_boundary)

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

    CFL = 0.01
    # Constant mesh size for now
    deltax = mesh.connectivityData.cells[0].facesCell.F[0].calculations.V
    deltay = mesh.connectivityData.cells[0].facesCell.F[1].calculations.V

    """ Global diagonal upwind matrix """
    globalDiag = np.zeros((nCells, nCells, dims, dims))

    """ Global off-diagonal matrix """
    globalOffDiag = np.zeros_like(globalDiag)

    """ Global mass matrix """
    massMatrices = np.zeros_like(globalDiag)

    """ Global stiffness matrix """
    stiffnessMatrices = np.zeros_like(globalDiag)

    """ Global RHS coefficients """
    fCoeffsGlobal = np.zeros((nCells, dims))

    """ Global solution coefficients """
    u = np.block([
        [mesh.connectivityData.cells[i].solnCell.uCoeffs] for i in range(nCells)
    ])
    uCoeffsGlobal = np.empty_like(u)

    """ Domain boundary condition contributions, zero Neumann """
    BCs = [bottom_BCs, right_BCs, top_BCs, left_BCs]
    BC_coord = [bottom_boundary, right_boundary, top_boundary, left_boundary]
    for j in range(4):  # 4 faces
        if BCs[j] == "Dirichlet":
            """ Iterate domain boundaries """
            for i in range(len(boundaryIDs[j])):
                """ Upwind """
                if j == 0 or j == 2:
                    x = mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].geomCell.geometry. \
                        parametricMapping()
                    y = BC_coord[j]
                    vel = np.array(
                        [np.matmul(np.array([-y, x[k]]), mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].
                                   unitNormal[k].transpose())
                         for k in range(len(mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].
                                            paramSeg.zeros))])
                elif j == 1 or j == 3:
                    y = mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].geomCell.geometry. \
                        parametricMapping()
                    x = BC_coord[j]
                    vel = np.array(
                        [np.matmul(np.array([-y[k], x]), mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].
                                   unitNormal[k].transpose())
                         for k in range(len(mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].
                                            paramSeg.zeros))])
                else:
                    raise NotImplementedError

                tmpOwnerBasis = mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].faceBasis
                """ Let's say all face Gauss points same v cdot n for now """
                if vel[0] >= 0:
                    """ Local contribution """
                    faceMatrix = np.matmul(tmpOwnerBasis.transpose(), tmpOwnerBasis * vel.reshape(-1, 1) *
                                           mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].
                                           paramSeg.weights *
                                           abs(mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].
                                               geomCell.detJacobian))
                    globalDiag[boundaryIDs[j][i]][boundaryIDs[j][i]] += faceMatrix

                elif vel[0] < 0:
                    # print(boundaryIDs[j][i])
                    """ RHS contribution """
                    faceMatrix = np.matmul(tmpOwnerBasis.transpose(), g_d[j].reshape(-1, 1) * vel.reshape(-1, 1) *
                                           mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].
                                           paramSeg.weights *
                                           abs(mesh.connectivityData.cells[boundaryIDs[j][i]].facesCell.F[j].
                                               geomCell.detJacobian))
                    fCoeffsGlobal[boundaryIDs[j][i]] -= faceMatrix.reshape(-1)

    """ Populate internal values """
    for i in range(nCells):
        massMatrices[i][i] = mesh.connectivityData.cells[i].massMatrix
        # different velocity requires mod
        stiffnessMatrices[i][i] = - a[i, :, 0] * mesh.connectivityData.cells[i].stiffnessMatrix[0] - \
                                  a[i, :, 1] * mesh.connectivityData.cells[i].stiffnessMatrix[1]
        neighbourLabels = mesh.connectivityData.cells[i].geomData.neighbourLabels
        vertex_coords = [mesh.connectivityData.cells[i].geomCell.geometry.A,
                         mesh.connectivityData.cells[i].geomCell.geometry.B,
                         mesh.connectivityData.cells[i].geomCell.geometry.C,
                         mesh.connectivityData.cells[i].geomCell.geometry.D]

        for j in range(4):  # 4 faces
            """ Internal faces """
            if neighbourLabels[j] is not None:
                if j == 0 or j == 2:
                    x = mesh.connectivityData.cells[i].facesCell.F[j].geomCell.geometry.parametricMapping()
                    y = vertex_coords[j][1]
                    vel = np.array(
                        [np.matmul(np.array([-y, x[k]]), mesh.connectivityData.cells[i].facesCell.F[j].
                                   unitNormal[k].transpose())
                         for k in range(len(mesh.connectivityData.cells[i].facesCell.F[j].paramSeg.zeros))])
                elif j == 1 or j == 3:
                    y = mesh.connectivityData.cells[i].facesCell.F[j].geomCell.geometry.parametricMapping()
                    x = vertex_coords[j][0]
                    vel = np.array(
                        [np.matmul(np.array([-y[k], x]), mesh.connectivityData.cells[i].facesCell.F[j].
                                   unitNormal[k].transpose())
                         for k in range(len(mesh.connectivityData.cells[i].facesCell.F[j].paramSeg.zeros))])
                else:
                    raise NotImplementedError

                tmpOwnerBasis = mesh.connectivityData.cells[i].facesCell.F[j].faceBasis
                """ Let's say all face Gauss points same v cdot n for now """
                if vel[0] >= 0:
                    """ Local contribution """
                    faceMatrix = np.matmul(tmpOwnerBasis.transpose(), tmpOwnerBasis * vel.reshape(-1, 1) *
                                           mesh.connectivityData.cells[i].facesCell.F[j].
                                           paramSeg.weights *
                                           abs(mesh.connectivityData.cells[i].facesCell.F[j].geomCell.detJacobian))
                    globalDiag[i][i] += faceMatrix

                elif vel[0] < 0:
                    """ Off-diagonal contribution """
                    tmpNeighbourBasis = mesh.connectivityData.cells[neighbourLabels[j]].facesCell.F[j - 2].faceBasis
                    faceMatrix = np.matmul(tmpOwnerBasis.transpose(), tmpNeighbourBasis * vel.reshape(-1, 1) *
                                           mesh.connectivityData.cells[i].facesCell.F[j].
                                           paramSeg.weights *
                                           abs(mesh.connectivityData.cells[i].facesCell.F[j].geomCell.detJacobian))
                    globalOffDiag[neighbourLabels[j]][i] += faceMatrix

    """ Global block-matrix """
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

    """ Calculate time-step size """
    deltaT = CFL / ((2. / deltax) + (2. / deltay))
    M = massMatrices / deltaT

    globalMatrix = M + globalDiag + stiffnessMatrices + globalOffDiag
    invGlobalMatrix = np.linalg.inv(globalMatrix)

    """ Create a gif """
    xCoords = np.array([[mesh.connectivityData.cells[i].GetQuadratureCoords[:, j][0] for j in range(nCoords)]
                        for i in range(nCells)])
    yCoords = np.array([[mesh.connectivityData.cells[i].GetQuadratureCoords[:, j][1] for j in range(nCoords)]
                        for i in range(nCells)])
    X, Y = np.meshgrid(xCoords, yCoords)
    no_of_frames = int(endTime / 0.1) + 1
    Z = np.zeros((no_of_frames, X.shape[0], Y.shape[0]))
    frame_counter = 0
    record_time = 0.0

    """ Start time-loop """
    while endTime - time > 1e-10:

        if abs(record_time - time) < 1e-6:
            save_data(mesh, Z, X, Y, xCoords, yCoords, nCells, nCoords, nVars, frame_counter)
            frame_counter += 1
            record_time += 0.1

        RHS = f.reshape(-1, 1) + np.matmul(M, u.reshape(-1, 1))

        """ Initialise new solution coefficients """
        uCoeffsGlobal = np.matmul(invGlobalMatrix, RHS)

        u = uCoeffsGlobal

        """ No limiter solution """
        for i in range(nCells):
            mesh.connectivityData.cells[i].solnCell.uCoeffs = u.reshape(nCells, dims)[i].reshape(-1, 1)

        # Increment time
        if time + deltaT > endTime:
            deltaT = endTime - time
        time += deltaT
        print("Current time: " + str(time))

    gif_name = 'non_constant_vel'
    filenames = []
    for i in range(no_of_frames):
        levels = MaxNLocator(nbins=15).tick_values(Z[i].min(), Z[i].max())
        cmap = plt.get_cmap('coolwarm')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)

        cf = ax.contourf(X, Y, Z[i], levels=levels, cmap=cmap)
        fig.colorbar(cf, ax=ax)

        # Plot the surface
        ax.title.set_text("Numerical solution, time = {0}".format(str(i * 0.1)))
        ax.set_xlim(left_boundary, right_boundary)
        ax.set_ylim(bottom_boundary, top_boundary)
        color_tuple = (1.0, 1.0, 1.0, 0.0)

        # Customize the axes
        ax.xaxis.set_major_formatter('{x:.01f}')
        ax.yaxis.set_major_formatter('{x:.01f}')

        # create file name and append it to a list
        filename = f'{i}.png'
        filenames.append(filename)

        # save frame
        plt.savefig(filename)
        plt.close()

    # build gif
    with imageio.get_writer(f'{gif_name}.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.v2.imread(filename)
            writer.append_data(image)

    # Remove files
    for filename in set(filenames):
        os.remove(filename)
