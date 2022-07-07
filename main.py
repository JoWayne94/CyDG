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
    # print(mesh.derivedData.faces)
    # print(mesh.connectivityData.cells[0].geomData.neighbourLabels)
    # print(mesh.connectivityData.cells[0].calculations.V)
    print(mesh.connectivityData.cells[0].GetQuadratureCoords)
    print(mesh.connectivityData.cells[0].massMatrix)
    print(mesh.connectivityData.cells[0].facesCell.F0.massMatrix)
