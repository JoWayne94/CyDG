"""
File: functions.py

Description: Memory and start statement print functions, plot functions
"""
import os
import psutil
import numpy as np
import matplotlib.pyplot as plt
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


def plotSolution(mesh, ncells, ncoords, nvars, test_case, p1, end_time):
    """
    @:brief Plot physical solutions
    :return: Plot images
    """

    pltPhysical = np.zeros((ncells, ncoords, nvars))

    for i in range(ncells):
        pltPhysical[i] = np.matmul(mesh.connectivityData.cells[i].matCell.basisMatrix,
                                   mesh.connectivityData.cells[i].solnCell.uCoeffs)

    pltCoords = np.array([mesh.connectivityData.cells[i].GetQuadratureCoords for i in range(ncells)]).reshape(-1)
    plt.plot(pltCoords, pltPhysical[:, :, 0].reshape(-1), linestyle="", marker="o", label=str(end_time), markersize=1)
    # color="blue"
    # plt.rc('axes', linewidth=1.25)
    # plt.xticks([-1, 0, 1])
    # plt.tick_params(axis='x', direction='in', pad=5)
    # plt.yticks([-1, 0, 1])
    # plt.tick_params(axis='y', direction='in', pad=5)

    plt.grid()
    plt_name = test_case + "_" + "P" + str(p1) + "_" + str(end_time)
    # plt.plot([0.0, 2.0],
    #          np.array([np.matmul(basisMatrixforF0, uCoeffsGlobal.reshape(2, 2)[0].reshape(-1, 1)),
    #                    np.matmul(basisMatrixforF1, uCoeffsGlobal.reshape(2, 2)[1].reshape(-1, 1))])[:, :,
    #          0].reshape(-1),
    #          linestyle="", marker="o", color="blue")
    # plt.savefig(plt_name + '_limited.png', dpi=100)
    # np.savetxt(plt_name + "_coords.dat", pltCoords, delimiter=',')
    # np.savetxt(plt_name + "_values.dat", pltPhysical[:, :, 0].reshape(-1), delimiter=',')
    plt.legend()
    plt.show()


def plotSolution2d(mesh, ncells, ncoords, nvars, test_case, p1, p2, end_time, left, right, bottom, top):
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
    pltPhysical = np.empty((ncells, ncoords, nvars))

    for i in range(ncells):
        pltPhysical[i] = np.matmul(mesh.connectivityData.cells[i].matCell.basisMatrix,
                                   mesh.connectivityData.cells[i].solnCell.uCoeffs)

    Z = np.zeros((X.shape[0], Y.shape[0]))

    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            loc = np.where((abs(xCoords - X[i][j]) < 1e-6) & (abs(yCoords - Y[i][j]) < 1e-6))
            # if len(loc[0]) > 1 or len(loc[1]) > 1:
            #     print(loc[0])
            #     print(loc[1])
            #     raise ValueError
            Z[i][j] = pltPhysical[loc[0][0]][loc[1][0]]

    levels = MaxNLocator(nbins=15).tick_values(Z.min(), Z.max())
    cmap = plt.get_cmap('coolwarm')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)

    # surf = ax.pcolormesh(X, Y, Z, cmap='', linewidth=0, antialiased=False)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    fig.colorbar(cf, ax=ax)
    # surf = ax.contour(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=False)

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6.4, 4.8), dpi=100)

    # Plot the surface
    # surf = ax.scatter(xCoords.reshape(-1), yCoords.reshape(-1), pltPhysical.reshape(-1),
    #                   cmap='coolwarm', linewidth=0, antialiased=False)
    # surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=False)
    ax.title.set_text("Numerical solution, time = {0}".format(str(end_time)))
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

    # Add a color bar which maps values to colour
    # fig.colorbar(cf, shrink=0.5, aspect=5)
    plt.show()


def save_data(mesh, values, x, y, xcoords, ycoords, ncells, ncoords, nvars, counter):

    pltPhysical = np.empty((ncells, ncoords, nvars))

    for i in range(ncells):
        pltPhysical[i] = np.matmul(mesh.connectivityData.cells[i].matCell.basisMatrix,
                                   mesh.connectivityData.cells[i].solnCell.uCoeffs)

    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            loc = np.where((abs(xcoords - x[i][j]) < 1e-6) & (abs(ycoords - y[i][j]) < 1e-6))
            values[counter][i][j] = pltPhysical[loc[0][0]][loc[1][0]]
