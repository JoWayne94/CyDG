import numpy as np
import pandas as pd
import math
from src.library.dgMesh.dgMesh import *
from src.utilities.factory import *

if __name__ == '__main__':
    """
    main()
    """

    name = "polyMesh/2x2"
    nDims = 2
    nVars = 1
    # Uniform polynomial orders in the x and y-directions for now
    P1 = 1
    P2 = 1
    # Read in the mesh
    mesh = DgMesh.constructFromPolyMeshFolder(name, nDims)
    mesh.constructShapeBasedCells(name, nVars, P1, P2)

    """ Print data to check code correctness """
    tmp = np.array([i for i, p in enumerate(mesh.connectivityData.points[:, 0], 0) if abs(p - 2.) < 1e-6])
    nCells = len(mesh.connectivityData.cells)

    test = [[i for cell in tmp if mesh.connectivityData.cells[i].geomData.pointLabels.count(cell) > 0]
            for i in range(nCells)]
    flatList = np.unique(np.array([item for elem in test for item in elem]))
    print(flatList)
    # print(mesh.connectivityData.cells[0].geomData.neighbourLabels)
    # print(mesh.connectivityData.cells[1].calculations.V)
    # print(len(mesh.connectivityData.cells[0].GetQuadratureCoords[0]))
    # print(mesh.connectivityData.cells[0].solnCell.uPhysical)
    # print(mesh.connectivityData.cells[0].facesCell.F3.unitNormals)

    # def shapeFunctionCalc(x1, x2):
    #     a0 = 0.0
    #     a1 = 0.0
    #     a2 = 2.7
    #
    #     basis = GetLegendre2D(np.array([x1, x2]), 3, 3)
    #
    #     return a0 * basis[0] + a1 * basis[1] + a2 * basis[2]
