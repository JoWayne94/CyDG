"""
File: meshGen.py

Description: Generate structured triangular and quadrilateral mesh in 2D
"""
# typedef std::vector<int> IntVec;
# typedef std::vector<double> DoubleVec;
# typedef std::vector<std::array<double, 2> > ArrayVec;
import numpy as np
import math
import fileinput
import sys
import matplotlib.pyplot as plt


def populatePointCoords(r, coord, n, delta, max_, min_, gp):
    """
    @:brief Fill in numpy array with x, y point coordinates
    :param r:       Geometric ratio
    :param coord:   Coordinates container
    :param n:       Number of cells in one-direction
    :param delta:   Mesh size
    :param max_:    End of physical domain coordinate
    :param min_:    Start of physical domain coordinate
    :param gp:      Geometric progression
    :return:        void
    """

    if r == 1.0:
        for i in range(n):
            delta[i] = (max_ - min_) / n
            coord[i + 1] = coord[i] + delta[i]

    else:

        if gp == "single_gp":
            """ One-way geometric progression """
            delta[0] = (max_ - min_) * (1 - r) / (1 - r ** n)
            coord[1] = coord[0] + delta[0]
            for i in range(1, n):
                delta[i] = delta[i - 1] * r
                coord[i + 1] = coord[i] + delta[i]

        elif gp == "double_gp":
            """ Two-way geometric progression """
            coord[-1] = max_
            number1 = (-1 * (-n // 2))
            number2 = n - number1
            delta[0] = ((max_ - min_) / 2) * (1 - r) / (1 - r ** number1)
            coord[1] = coord[0] + delta[0]
            delta[-1] = ((max_ - min_) / 2) * (1 - r) / (1 - r ** number2)
            coord[-2] = coord[-1] - delta[-1]

            for i in range(1, number1):
                delta[i] = delta[i - 1] * r
                coord[i + 1] = coord[i] + delta[i]

            for i in range(1, number2):
                delta[-1 - i] = delta[-i] * r
                coord[-2 - i] = coord[-1 - i] - delta[-1 - i]

        else:
            raise NotImplementedError


if __name__ == '__main__':

    """ x min """
    x_min = -1.

    """ y min """
    y_min = -1.

    """ x max """
    x_max = 1.

    """ y max """
    y_max = 1.

    """ No. of cells in x-direction """
    Nx = 3

    """ No. of cells in y-direction """
    Ny = 3

    """ Geometric progression ratio in x-direction """
    rx = 1.0

    """ Geometric progression ratio in y-direction """
    ry = 1.0

    """ Cell shape """
    shape = "T"
    tri_direction = "diagonal"

    save_dat = True  # save points and cells data
    gpx = "single_gp"  # single or double geometric progression
    gpy = "single_gp"

    """ Total no. of points """
    N = (Nx + 1) * (Ny + 1)

    """ Point IDs list """
    # listP = np.empty(N, dtype=int)
    listP = np.array([i for i in range(N)])

    """ Point coordinates lists """
    listPx = np.empty(Nx + 1, dtype=np.double)
    listPy = np.empty(Ny + 1, dtype=np.double)

    """ Mesh sizes """
    dx = np.empty(Nx, dtype=np.double)
    dy = np.empty(Ny, dtype=np.double)
    listPx[0] = x_min
    listPy[0] = y_min

    populatePointCoords(rx, listPx, Nx, dx, x_max, x_min, gpx)
    populatePointCoords(ry, listPy, Ny, dy, y_max, y_min, gpy)

    """
    Gmsh bump algo
    https://gitlab.onelab.info/gmsh/gmsh/-/blob/gmsh_4_8_4/Mesh/meshGEdge.cpp
    """
    # if rx > 1.0:
    #     dx[0] = -4.0 * math.sqrt(rx - 1.0) * math.atan2(1.0, math.sqrt(rx - 1.0)) / ((Nx + 1) * (x_max - x_min))
    # else:
    #     dx[0] = 2.0 * math.sqrt(1.0 - rx) *\
    #         math.log(abs((1.0 + 1.0 / math.sqrt(1.0 - rx)) / (1.0 - 1.0 / math.sqrt(1.0 - rx)))) /\
    #         ((Nx + 1) * (x_max - x_min))
    #
    # b = -dx[0] * (x_max - x_min) * (x_max - x_min) / (4.0 * (rx - 1.0))
    # for i in range(1, Nx + 1):
    #     listPx[i] = listPx[i - 1] + dx[i - 1]
    #     dx[i] = (-dx[0] * (listPx[i] - (x_max - x_min) * 0.5) ** 2) + b

    """ Output the data """
    # outputList = np.zeros((N, 2))
    #     outputList[i][0] = listPx[i % (Nx + 1)]
    #     outputList[i][1] = listPy[i // (Nx + 1)]
    # np.savetxt(name, outputList, delimiter=' ')
    # for line in fileinput.input(['./' + name], inplace=True):
    #     sys.stdout.write('({l}'.format(l=line))

    if save_dat:
        name = 'rawOutput/' + str(Nx) + '_' + str(Ny) + "_points_list.dat"
        with open(name, 'w+') as f:
            f.write(str(N) + '\n')
            f.write('( \n')
            for i in range(N):
                f.write('(' + "{:.5f}".format(listPx[i % (Nx + 1)]) + ' ' + "{:.5f}".format(listPy[i // (Nx + 1)]) +
                        ') \n')
            f.write(') \n')

    # totalEdges = Nx * (Ny + 1) + Ny * (Nx + 1)
    # internalEdges = totalEdges - 2 * (Nx + Ny)

    if shape == "Q":
        """ Total number of cells """
        nCells = Nx * Ny

        cells = np.empty((nCells, 4), dtype=int)

        temp, count = 1, 1
        for i in range(0, nCells, 1):
            temp = i // Nx
            cells[i][0] = listP[count + temp - 1]
            cells[i][1] = listP[count + temp]
            cells[i][2] = listP[count + temp + (Nx + 1)]
            cells[i][3] = listP[count + temp + (Nx + 1) - 1]
            count += 1

    elif shape == "T":
        """ Total number of cells """
        nCells = Nx * Ny * 2

        cells = np.empty((nCells, 3), dtype=int)

        temp, count = 1, 1
        if tri_direction == "diagonal":
            for i in range(0, nCells, 2):
                temp = i // (Nx * 2)
                cells[i][0] = listP[count + temp - 1]
                cells[i][1] = listP[count + temp]
                cells[i][2] = listP[count + temp + (Nx + 1) - 1]

                cells[i + 1][0] = listP[count + temp]
                cells[i + 1][1] = listP[count + temp + (Nx + 1)]
                cells[i + 1][2] = listP[count + temp + (Nx + 1) - 1]
                count += 1

        elif tri_direction == "anti-diagonal":
            for i in range(0, nCells, 2):
                temp = i // (Nx * 2)
                cells[i][0] = listP[count + temp - 1]
                cells[i][1] = listP[count + temp + (Nx + 1)]
                cells[i][2] = listP[count + temp + (Nx + 1) - 1]

                cells[i + 1][0] = listP[count + temp - 1]
                cells[i + 1][1] = listP[count + temp]
                cells[i + 1][2] = listP[count + temp + (Nx + 1)]
                count += 1

        else:
            print("Diagonalisation direction not specified. Please try again.")
            raise NotImplementedError

    else:
        print("Shape input not defined. Please pick Q or T.")
        raise NotImplementedError

    if save_dat:
        name = 'rawOutput/' + str(Nx) + '_' + str(Ny) + "_cells_list.dat"
        with open(name, 'w+') as f:
            f.write(str(nCells) + '\n')
            f.write('( \n')
            for i in range(nCells):
                f.write('(' + shape + ' ' + ' '.join([str(cells[i][j]) for j in range(len(cells[i]))]) + ') \n')
            f.write(') \n')

    plt.plot(np.full(Ny + 1, listPx[0]), listPy, 'k')
    for i in range(nCells):
        plt.plot([listPx[j % (Nx + 1)] for j in cells[i]], [listPy[j // (Nx + 1)] for j in cells[i]], 'k')

    # for i in range(Ny + 1):
    #     plt.plot(listPx, np.full(Nx + 1, listPy[i]), 'k')
    # for i in range(Nx + 1):
    #     plt.plot(np.full(Ny + 1, listPx[i]), listPy, 'k')

    plt.axis('square')
    plt.show()
