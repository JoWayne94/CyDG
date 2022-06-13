import numpy as np
import pandas as pd
from src.library.dgMesh.dgMesh import *


if __name__ == '__main__':
    """
    main()
    """

    name = "polyMesh/2x0"
    nDims = 1
    # Uniform polynomial orders in the x and y-directions for now
    P1 = 2
    P2 = 0
    # Read in the mesh
    mesh = DgMesh.constructFromPolyMeshFolder(name, nDims)
    mesh.constructShapeBasedCells(name, P1, P2)

    # Print data to check code correctness
    # print(mesh.derivedData.faces)
    # print(mesh.connectivityData.cells[0].geomData.neighbourLabels)
    # print(mesh.connectivityData.cells[0].calculations.V)
    # print(mesh.connectivityData.cells[0].GetMassMatrix())
    # print(mesh.connectivityData.cells[0].GetStiffnessMatrix())
    # print(mesh.connectivityData.cells[1].GetLaplacianMatrix())

    """
    Prototype time loop with forward Euler time-stepping
    """
    rhoL = 1.0
    uL = 0.0
    pL = 1.0
    rhoR = 0.125
    uR = 0.0
    pR = 0.1
    xd = 0.5

    time = 0.0
    endTime = 1.0
    nCells = len(mesh.connectivityData.cells)
    quadCoords = np.array([mesh.connectivityData.cells[i].GetQuadratureCoords for i in range(nCells)]).reshape(-1)
    # while endTime - time > 1e-10:
    #
    #     deltaT = 0.5
    #
    #     # Increment time
    #     if time + deltaT > endTime:
    #         deltaT = endTime - time
    #     time += deltaT
    #
    #     divFlux = np.zeros(nCells)
    #
    #     for i in range(nCells):
    #
    #         for face in mesh.connectivityData.cells[i].geomData.neighbourLabels:
    #
    #             if face is not None:

