"""
File: functions.py

Description: Memory and start statement print functions, plot functions
"""
import os
import psutil
import numpy as np
import matplotlib.pyplot as plt
from src.library.paramCells.basis import *
from matplotlib.ticker import LinearLocator
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


def printMemoryUsageInMB():
    print("\n Memory usage is: " + str(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2) + ' MB \n')


def startStatement(solverName, caseDir, date):
    print("""--------------- PyDG v1.0 ---------------""")
    print("Solver           : " + solverName)
    print("Case             : " + caseDir)
    print("Date and time    : " + date)
    print("\n")


def save_solution(coords, solution, path, name):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)

    np.savetxt(path + name + "_coords.dat", coords, delimiter=',')
    np.savetxt(path + name + "_values.dat", solution, delimiter=',')


def plotSolution(mesh, ncells, ncoords, nvars, p1, left, right, directory, exact, label, title, plt_name, save=False):
    """
    @:brief Plot physical solutions
    :return: Plot images
    """

    pltPhysical = np.zeros((ncells, ncoords, nvars))

    for i in range(ncells):
        pltPhysical[i] = np.matmul(mesh.connectivityData.cells[i].matCell.basisMatrix,
                                   mesh.connectivityData.cells[i].solnCell.uCoeffs)

    pltPhysical = pltPhysical[:, :, 0].reshape(-1)
    pltCoords = np.array([mesh.connectivityData.cells[i].GetQuadratureCoords for i in range(ncells)]).reshape(-1)

    # tmp = int((ncells * ncoords) / 4)

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)

    # ax.plot(pltCoords, exact[:, :, 0].reshape(-1), linestyle="-", linewidth=1, label="Exact solution", color="blue")
    # ax.plot(pltCoords[tmp:tmp * 3 + 1], (exact[:, :, 0].reshape(-1))[tmp:tmp * 3 + 1], linestyle="-", linewidth=1,
    #         color="blue")
    # ax.plot(pltCoords[tmp * 3 + 1:], (exact[:, :, 0].reshape(-1))[tmp * 3 + 1:], linestyle="-", linewidth=1,
    #         color="blue")

    # ax.plot([left, pltCoords[0]], np.array([5., exact[:, :, 0].reshape(-1)[0]]), linestyle="-", linewidth=1,
    #         color="blue")
    # ax.plot([pltCoords[-1], right], np.array([exact[:, :, 0].reshape(-1)[-1], 2.]), linestyle="-", linewidth=1,
    #         color="blue")

    ax.plot(pltCoords, pltPhysical, linestyle="", marker="o", label=label, markersize=5, color="black")
    ax.plot([left, right],
            np.array([np.matmul(Legendre1d(np.array([[-1]]), p1), mesh.connectivityData.cells[0].solnCell.uCoeffs),
                      np.matmul(Legendre1d(np.array([[1]]), p1), mesh.connectivityData.cells[-1].solnCell.uCoeffs)])[:,
            :, 0].reshape(-1), linestyle="", marker="o", markersize=5, color="black")

    # plt.rc('axes', linewidth=1.25)
    # plt.xticks([-1, 0, 1])
    # plt.yticks([-1, 0, 1])
    # plt.tick_params(axis='x', direction='in', pad=5)
    # plt.tick_params(axis='y', direction='in', pad=5)
    ax.title.set_text(title)
    plt.grid()
    plt.legend()
    if save:
        save_solution(pltCoords, pltPhysical, directory, plt_name)
        plt.savefig(directory + plt_name + '_plot.eps', dpi=1000)

    plt.show()


def calculateL2err(mesh, exact, ncells):
    err = 0.

    for i in range(ncells):
        err += np.matmul((mesh.connectivityData.cells[i].paramSeg.weights *
                          abs(mesh.connectivityData.cells[i].geomCell.detJacobian)).transpose(),
                         (exact[i] - mesh.connectivityData.cells[i].solnCell.uPhysical) ** 2)[0][0]

    print("L2 error is: ")
    print(np.sqrt(err))


def plotSolution2d(mesh, ncells, ncoords, nvars, left, right, bottom, top, directory, exact, label, title,
                   plt_name, save=False):
    """
    @:brief Plot physical solutions in 2D, set figure parameters
    :return: Plot images
    """
    # xyCoords = np.array([[mesh.connectivityData.cells[i].GetQuadratureCoords[:, j] for j in range(nCoords)]
    #                     for i in range(nCells)]).reshape(nCells * nCoords, 2)
    xCoords = np.array([[mesh.connectivityData.cells[i].GetQuadratureCoords[:, j][0] for j in range(ncoords)]
                        for i in range(ncells)])
    yCoords = np.array([[mesh.connectivityData.cells[i].GetQuadratureCoords[:, j][1] for j in range(ncoords)]
                        for i in range(ncells)])
    X, Y = np.meshgrid(xCoords, yCoords)
    pltPhysical = np.array([mesh.connectivityData.cells[i].solnCell.uPhysical for i in range(ncells)])

    Z = np.zeros((X.shape[0], Y.shape[0]))

    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            loc = np.where((abs(xCoords - X[i][j]) < 1e-7) & (abs(yCoords - Y[i][j]) < 1e-7))
            # if len(loc[0]) > 1 or len(loc[1]) > 1:
            #     print(loc[0])
            #     print(loc[1])
            #     raise ValueError
            Z[i][j] = pltPhysical[loc[0][0]][loc[1][0]]

    levels = MaxNLocator(nbins=20).tick_values(Z.min(), Z.max())
    cmap = plt.get_cmap('coolwarm')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)

    # pcmesh = ax.pcolormesh(X, Y, Z, linewidth=0, antialiased=False)
    # contour = ax.contour(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=False)
    # scatter = ax.scatter(xCoords.reshape(-1), yCoords.reshape(-1), pltPhysical.reshape(-1),
    #                      linewidth=0, antialiased=False)
    # surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=False)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    fig.colorbar(cf, ax=ax)

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6.4, 4.8), dpi=100)

    # Plot the surface
    ax.title.set_text(title)
    color_tuple = (1.0, 1.0, 1.0, 0.0)

    # Customize the axes
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    # ax.xaxis.set_major_locator(LinearLocator(5))
    # ax.yaxis.set_major_locator(LinearLocator(5))
    # ax.zaxis.set_major_locator(LinearLocator(10))
    ax.xaxis.set_major_formatter('{x:.01f}')
    ax.yaxis.set_major_formatter('{x:.01f}')
    # ax.zaxis.set_major_formatter('{x:.02f}')

    # ax.w_xaxis.set_pane_color(color_tuple)
    # ax.w_xaxis.line.set_color(color_tuple)
    # ax.w_yaxis.set_pane_color(color_tuple)
    # ax.w_yaxis.line.set_color(color_tuple)

    # Add a color bar which maps values to colour
    # fig.colorbar(cf, shrink=0.5, aspect=5)
    if save:
        isExist = os.path.exists(directory)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(directory)
        # save_solution(pltCoords, pltPhysical, directory, plt_name)
        plt.savefig(directory + plt_name + '_plot.eps', dpi=1000)

    plt.show()


def calculateL2err2d(mesh, exact, ncells):
    err = 0.

    for i in range(ncells):
        err += np.matmul((mesh.connectivityData.cells[i].paramQuad.weights *
                          abs(mesh.connectivityData.cells[i].geomCell.detJacobian)).transpose(),
                         (exact[i] - mesh.connectivityData.cells[i].solnCell.uPhysical) ** 2)[0][0]

    print("L2 error is: ")
    print(np.sqrt(err))


def gif_data(mesh, u, values, x, y, xcoords, ycoords, ncells, ncoords, nvars, dims, counter):
    pltPhysical = np.empty((ncells, ncoords, nvars))

    for i in range(ncells):
        pltPhysical[i] = np.matmul(mesh.connectivityData.cells[i].matCell.basisMatrix,
                                   u.reshape(ncells, dims)[i].reshape(-1, 1))

    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            loc = np.where((abs(xcoords - x[i][j]) < 1e-6) & (abs(ycoords - y[i][j]) < 1e-6))
            values[counter][i][j] = pltPhysical[loc[0][0]][loc[1][0]]


def gif_data1d(mesh, u, values, ncells, ncoords, nvars, dims, counter):
    pltPhysical = np.empty((ncells, ncoords, nvars))

    for i in range(ncells):
        pltPhysical[i] = np.matmul(mesh.connectivityData.cells[i].matCell.basisMatrix,
                                   u.reshape(ncells, dims)[i].reshape(-1, 1))
        for j in range(ncoords):
            values[counter][i][j] = pltPhysical[i][j]
