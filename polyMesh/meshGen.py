"""
File: meshGen.py

Description: Generate structured quadrilateral mesh in 2D
"""
# typedef std::vector<int> IntVec;
# typedef std::vector<double> DoubleVec;
# typedef std::vector<std::array<double, 2> > ArrayVec;
import numpy as np
import math
import fileinput
import sys
import matplotlib.pyplot as plt


def populatePointCoords(r, coord, N, delta, max_, min_, gp):

    if r == 1.0:
        for n in range(N):
            delta[n] = (max_ - min_) / N
            coord[n + 1] = coord[n] + delta[n]

    else:
        """ Geometric progression """
        if gp == "single_gp":
            delta[0] = (max_ - min_) * (1 - r) / (1 - r ** N)
            coord[1] = coord[0] + delta[0]
            for n in range(1, N):
                delta[n] = delta[n - 1] * r
                coord[n + 1] = coord[n] + delta[n]

        elif gp == "double_gp":
            coord[-1] = max_
            number1 = (-1 * (-N // 2))
            number2 = N - number1
            delta[0] = ((max_ - min_) / 2) * (1 - r) / (1 - r ** number1)
            coord[1] = coord[0] + delta[0]
            delta[-1] = ((max_ - min_) / 2) * (1 - r) / (1 - r ** number2)
            coord[-2] = coord[-1] - delta[-1]
            for n in range(1, number1):
                delta[n] = delta[n - 1] * r
                coord[n + 1] = coord[n] + delta[n]

            for n in range(1, number2):
                delta[-1 - n] = delta[-n] * r
                coord[-2 - n] = coord[-1 - n] - delta[-1 - n]

        else:
            raise NotImplementedError


if __name__ == '__main__':

    """ x min """
    x_min = 0.

    """ y min """
    y_min = 0.

    """ x max """
    x_max = 1.

    """ y max """
    y_max = 0.

    """ No. of cells in x-direction """
    Nx = 50

    """ No. of cells in y-direction """
    Ny = 1

    """ Geometric progression ratio in x-direction """
    rx = 1.0

    """ Geometric progression ratio in y-direction """
    ry = 1.0

    save_dat = True  # save points and cells data
    gpx = "single_gp"  # single or double geometric progression
    gpy = "single_gp"

    """ Total no. of points """
    N = (Nx + 1) * (Ny + 1)

    """ Total number of cells """
    nCells = Nx * Ny

    """ Point IDs list """
    listP = np.empty(N, dtype=int)

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
    for i in range(N):
        listP[i] = i
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

    cells = np.empty((nCells, 4), dtype=int)
    # totalEdges = Nx * (Ny + 1) + Ny * (Nx + 1)
    # internalEdges = totalEdges - 2 * (Nx + Ny)

    temp, count = 1, 1
    for i in range(0, nCells, 1):
        temp = i // Nx
        cells[i][0] = listP[count + temp - 1]
        cells[i][1] = listP[count + temp]
        cells[i][2] = listP[count + temp + (Nx + 1)]
        cells[i][3] = listP[count + temp + (Nx + 1) - 1]
        count += 1

    if save_dat:
        name = 'rawOutput/' + str(Nx) + '_' + str(Ny) + "_cells_list.dat"
        with open(name, 'w+') as f:
            f.write(str(nCells) + '\n')
            f.write('( \n')
            for i in range(nCells):
                f.write('(Q ' + str(cells[i][0]) + ' ' + str(cells[i][1]) + ' ' + str(cells[i][2]) + ' ' +
                        str(cells[i][3]) + ') \n')
            f.write(') \n')

    for i in range(Ny + 1):
        plt.plot(listPx, np.full(Nx + 1, listPy[i]), 'k')

    for i in range(Nx + 1):
        plt.plot(np.full(Ny + 1, listPx[i]), listPy, 'k')

    plt.axis('square')
    plt.show()

    # edges = np.empty(internalEdges, dtype=int)
    # edgesP1 = np.empty_like(edges)
    # edgesP2 = np.empty_like(edges)
    # ownerArray = np.empty_like(edges)
    # neighbourArray = np.empty_like(edges)
    #
    # edgesP1x = np.empty(internalEdges, dtype=np.double)
    # edgesP2x = np.empty_like(edgesP1x)
    # edgesP1y = np.empty_like(edgesP1x)
    # edgesP2y = np.empty_like(edgesP1x)
    #
    # cellVol = np.empty_like(edgesP1x)
    # fxGeom = np.empty_like(edgesP1x)
    # edgesLength = np.empty_like(edgesP1x)
    #
    # faceNormal = np.empty((internalEdges, 2), dtype=np.double)
    # cellCentre = np.empty_like(faceNormal)
    # distance = np.empty_like(faceNormal)
    #
    # for i in range(0, internalEdges - ((Ny - 1) + (Nx - 1)), 2):
    #     temp = i // (2 * (Nx - 1))
    #     edgesP1[i + temp] = listP[count + 2 * temp]  # point i+1
    #     edgesP2[i + temp] = listP[count + 2 * temp + (Nx + 1)]
    #     edgesP1[i + temp + 1] = edgesP2[i + temp]
    #     edgesP2[i + temp + 1] = listP[count + 2 * temp + (Nx + 1) - 1]
    #
    #     edgesP1x[i + temp] = listPx[(count + 2 * temp) % (Nx + 1)]
    #     edgesP1y[i + temp] = listPy[(count + 2 * temp) // (Nx + 1)]
    #     edgesP2x[i + temp] = listPx[(count + 2 * temp + (Nx + 1)) % (Nx + 1)]
    #     edgesP2y[i + temp] = listPy[(count + 2 * temp + (Nx + 1)) // (Nx + 1)]
    #
    #     edgesP1x[i + temp + 1] = edgesP2x[i + temp]
    #     edgesP1y[i + temp + 1] = edgesP2y[i + temp]
    #     edgesP2x[i + temp + 1] = listPx[(count + 2 * temp + (Nx + 1) - 1) % (Nx + 1)]
    #     edgesP2y[i + temp + 1] = listPy[(count + 2 * temp + (Nx + 1) - 1) // (Nx + 1)]
    #
    #     ownerArray[i + temp] = count + temp
    #     ownerArray[i + temp + 1] = count + temp
    #     neighbourArray[i + temp] = ownerArray[i + temp] + 1
    #     neighbourArray[i + temp + 1] = neighbourArray[i + temp] + (Nx - 1)
    #     count += 1
    #
    #     distance[i + temp][0] = edgesP2x[i + temp] - edgesP1x[i + temp]
    #     distance[i + temp][1] = edgesP2y[i + temp] - edgesP1y[i + temp]
    #     distance[i + temp + 1][0] = edgesP2x[i + temp + 1] - edgesP1x[i + temp + 1]
    #     distance[i + temp + 1][1] = edgesP2y[i + temp + 1] - edgesP1y[i + temp + 1]
    #
    #     edgesLength[i + temp] = math.sqrt(distance[i + temp][0] * distance[i + temp][0] + distance[i + temp][1] *
    #                                       distance[i + temp][1])
    #     edgesLength[i + temp + 1] = math.sqrt(distance[i + temp + 1][0] * distance[i + temp + 1][0] +
    #                                           distance[i + temp + 1][1] * distance[i + temp + 1][1])
    #
    #     cellVol[i + temp] = edgesLength[i + temp] * edgesLength[i + temp + 1]
    #     cellVol[i + temp + 1] = edgesLength[i + temp] * edgesLength[i + temp + 1]
    #
    #     faceNormal[i + temp][0] = distance[i + temp][1]
    #     faceNormal[i + temp][1] = -distance[i + temp][0]
    #     faceNormal[i + temp + 1][0] = distance[i + temp + 1][1]
    #     faceNormal[i + temp + 1][1] = -distance[i + temp + 1][0]
    #
    #     cellCentre[i + temp][0] = edgesP2x[i + temp] - abs(distance[i + temp + 1][0] / 2)
    #     cellCentre[i + temp][1] = edgesP2y[i + temp] - abs(distance[i + temp][1] / 2)
    #     cellCentre[i + temp + 1][0] = edgesP1x[i + temp + 1] - abs(distance[i + temp + 1][0] / 2)
    #     cellCentre[i + temp + 1][1] = edgesP1y[i + temp + 1] - abs(distance[i + temp][1] / 2)
    #
    # for i in range(Ny - 1):
    #
    #     index = (i + 1) * (2 * (Nx - 1) + 1) - 1
    #
    #     edgesP1[index] = edgesP1[index - 1] + 1
    #     edgesP2[index] = edgesP2[index - 1] + 1
    #
    #     edgesP1x[index] = edgesP1x[index - 1] + dx[Nx - 1]
    #     edgesP1y[index] = edgesP1y[index - 1]
    #     edgesP2x[index] = edgesP2x[index - 1] + dx[Nx - 2]
    #     edgesP2y[index] = edgesP2y[index - 1]
    #
    #     ownerArray[index] = (i+1) * Nx
    #     neighbourArray[index] = neighbourArray[index - 1] + 1
    #
    #     distance[index][0] = edgesP2x[index] - edgesP1x[index]
    #     distance[index][1] = edgesP2y[index] - edgesP1y[index]
    #
    #     edgesLength[index] = math.sqrt(distance[index][0] * distance[index][0] + distance[index][1] *
    #                                    distance[index][1])
    #
    #     cellVol[index] = edgesLength[index] * edgesLength[index - 2]
    #
    #     faceNormal[index][0] = distance[index][1]
    #     faceNormal[index][1] = -distance[index][0]
    #
    #     cellCentre[index][0] = edgesP1x[index] - abs(distance[index][0] / 2)
    #     cellCentre[index][1] = edgesP1y[index] - abs(distance[index - 2][1] / 2)
    #
    # for i in range(Nx - 1):
    #
    #     index = internalEdges - (Nx - 1) + i
    #
    #     edgesP1[index] = listP[(N-1) - (Nx-1) - (Nx+1) + i]
    #     edgesP2[index] = listP[(N-1) - (Nx-1) + i]
    #
    #     edgesP1x[index] = listPx[((N-1) - (Nx-1) - (Nx+1) + i) % (Nx+1)]
    #     edgesP1y[index] = listPy[((N-1) - (Nx-1) - (Nx+1) + i) // (Nx+1)]
    #     edgesP2x[index] = listPx[((N-1) - (Nx-1) + i) % (Nx+1)]
    #     edgesP2y[index] = listPy[((N-1) - (Nx-1) + i) // (Nx+1)]
    #
    #     ownerArray[index] = (Nx * Ny) - Nx + i+1
    #     neighbourArray[index] = ownerArray[index] + 1
    #
    #     distance[index][0] = edgesP2x[index] - edgesP1x[index]
    #     distance[index][1] = edgesP2y[index] - edgesP1y[index]
    #
    #     edgesLength[index] = math.sqrt(distance[index][0] * distance[index][0] + distance[index][1] *
    #                                    distance[index][1])
    #
    #     cellVol[index] = edgesLength[index] * edgesLength[index - (2 * Nx - 2 - i)]
    #
    #     faceNormal[index][0] = distance[index][1]
    #     faceNormal[index][1] = -distance[index][0]
    #
    #     cellCentre[index][0] = edgesP2x[index] - abs(distance[index - (2 * Nx - 2 - i)][0] / 2)
    #     cellCentre[index][1] = edgesP2y[index] - abs(distance[index][1] / 2)
