"""
File: dgMesh.py

Description: Mesh data needed to do the discontinuous Galerkin discretisation
"""
import numpy as np
from src.library.dgCells.segment import Segment as Seg
from src.library.dgCells.quadrilateral import Quadrilateral as Quad  # Mesh not mixed for now


class DgMesh:
    """
    @:brief Contains information of the mesh
    """
    class DgMeshConnectivityData:
        """
        @:brief Mesh connectivity data subclass
        """
        def __init__(self):
            self.points = None
            self.pointLabels = None
            self.cells = None  # DgCell objects stored in a list
            self.cellLabels = None  # DgCell IDs stored in a numpy array
            self.boundary = None
            self.nDims = None

    class DgMeshDerivedData:
        """
        @:brief Mesh derived data subclass
        """
        def __init__(self):
            self.pointNeighbours = None
            self.faces = None

    def __init__(self, points, pointlabels, boundary, ndims):
        """
        @:brief Main constructor
        :param points:          Coordinates of all points stored in a numpy array
        :param pointlabels:     Point IDs stored in a numpy array
        :param boundary:        Dictionary of boundary patches for the simulation
        """
        self.connectivityData = self.DgMeshConnectivityData()
        self.derivedData = self.DgMeshDerivedData()

        # Connectivity data
        self.connectivityData.points = points
        self.connectivityData.pointLabels = pointlabels
        self.connectivityData.boundary = boundary
        self.connectivityData.nDims = ndims

    @classmethod
    def constructFromPolyMeshFolder(cls, polyMeshLocation, ndims):
        """
        @:brief Read data from polyMesh directory and create mesh for simulation
        :param ndims:            Number of spatial dimensions
        :param polyMeshLocation: Location of polyMesh directory in machine
        :return: Returns main constructor with data from methods defined
        """
        print("Creating mesh. \n")

        pointlabels, points = cls.readPointsFromPolyMesh(polyMeshLocation, ndims)
        boundary = cls.readBoundaryFromPolyMesh(polyMeshLocation)

        return cls(points, pointlabels, boundary, ndims)

    def constructShapeBasedCells(self, polyMeshLocation, nvars, p1, p2=0):
        """
        @:brief Create shape-based cells only after points and point IDs arrays are created
        :param nvars:            Number of state variables
        :param p1:               Polynomial order in the x-direction
        :param p2:               Polynomial order in the y-direction
        :param polyMeshLocation: Location of polyMesh directory in machine
        :return: Return cell IDs and cell objects list in mesh object
        """
        self.connectivityData.cellLabels, self.connectivityData.cells, self.derivedData.pointNeighbours, \
            self.derivedData.faces = self.readCellsFromPolyMesh(polyMeshLocation, nvars, p1, p2)

        print("Mesh created. \n")

    @staticmethod
    def readPointsFromPolyMesh(polyMeshLocation, ndims):
        """
        @:brief Read points data from points file and create corresponding arrays
        :param ndims:            Number of spatial dimensions
        :param polyMeshLocation: Location of polyMesh directory in machine
        :return: Return arrays of point IDs and coordinates
        """
        # Read in file
        file = open(polyMeshLocation + "/points").read()

        # Split the file into tokens
        file = file.split()

        # Number of points in mesh is in the first line of points file
        nPoints = int(file[0])

        # Initialise the end result: array of points and point IDs
        pointsArray = np.empty((nPoints, ndims), dtype=float)  # 1 for 1D, 2 for 2D, 3 for 3D
        pointLabelsArray = np.empty(nPoints, dtype=int)

        # Variable used to know in which point the file is
        pointCounter = 0

        # Variable used to determine point ID and the x, y, z coordinates
        componentCounter = 0

        for i in file[2:-1]:
            # Remove brackets in file
            component = (i.replace('(', '')).replace(')', '')
            # Assign point ID to array with increment of 1
            pointLabelsArray[pointCounter] = pointCounter
            # Assign coordinates to array
            pointsArray[pointCounter][componentCounter] = component

            if componentCounter == ndims - 1:  # 0 for 1D, 1 for 2D, 2 for 3D
                componentCounter = 0
                pointCounter += 1
            else:
                componentCounter += 1

        return pointLabelsArray, pointsArray

    def readCellsFromPolyMesh(self, polyMeshLocation, nvars, p1, p2):
        """
        @:brief Read cells data from cells file and create corresponding containers
        :param nvars:            Number of state variables
        :param p1:               Polynomial order in the x-direction
        :param p2:               Polynomial order in the y-direction
        :param polyMeshLocation: Location of polyMesh directory in machine
        :return: Return arrays of cell IDs, cell objects, point neighbours, and faces
        """
        # Read in file
        file = open(polyMeshLocation + "/cells").read().splitlines()

        # Read number of cells in mesh as the first line of file
        nCells = int(file[0])

        # Initialise end results
        """ Cell IDs array """
        ownersArray = np.empty(nCells, dtype=int)

        """ Cell shapes array """
        shapesArray = np.empty(nCells, dtype=str)

        """ Vertex IDs list """
        # Vertices list not array as no. of vertices that make up a cell depends on cell shape hence size not const
        verticesList = []

        # Variable used to know which cell it is in the file
        cellCounter = 0

        for i in file[2:-1]:
            # Remove brackets in line
            i = (i.replace('(', '')).replace(')', '')
            # Shape char read as the first element in line
            shape = i[0]
            # Remove shape char from line so that vertex IDs remain
            vertices = ((i.replace(shape + ' ', '')).replace(')', '')).split()
            # List comprehension to convert vertex IDs to integer and store in a list
            vertices = [int(v) for v in vertices]
            # Append vertex IDs to list of all vertex IDs in the mesh
            verticesList.append(vertices)
            # Assign cell ID to array with increment of 1
            ownersArray[cellCounter] = cellCounter
            # Assign cell shape char to array
            shapesArray[cellCounter] = shape
            # Next cell in line
            cellCounter += 1

        # Initialise end result: List of cell IDs corresponding to the neighbours of each point
        pointNeighboursList = []
        # Iterate through all points
        for i in range(len(self.connectivityData.points)):
            # Go through all elements in the vertices list. If list contains current point ID, append cell ID to temp
            temp = [x for x, v in enumerate(verticesList, 0) if v.count(i) > 0]
            pointNeighboursList.append(temp)

        # Initialise end result: faces list
        facesList = []
        if self.connectivityData.nDims == 1:
            # Iterate through all points with neighbouring cell IDs
            for index in range(len(pointNeighboursList)):
                # Ignore point if point only has one neighbour; they don't contribute to internal faces
                if len(pointNeighboursList[index]) == 1:
                    continue
                else:
                    facesList.append([index, set(pointNeighboursList[index])])
        else:
            # Iterate through all points with neighbouring cell IDs
            for index in range(len(pointNeighboursList)):
                # Initialise temporary faces list
                tempFaceList = []
                # Ignore point if point only has one neighbour; they don't contribute to internal faces
                if len(pointNeighboursList[index]) == 1:
                    continue
                else:
                    # List comprehension to go through all remaining points below current point. If contains common cell
                    # IDs (> 1), add common cell IDs to list
                    commonNeighbour = [self.common_member(pointNeighboursList[index], point) for point in
                                       pointNeighboursList[index + 1:]]
                    # Iterate through common cell IDs list
                    # https://www.geeksforgeeks.org/python-non-none-elements-indices/
                    for enum, val in enumerate(commonNeighbour):
                        # If contains common neighbouring cell IDs, append [point ID who shares common neighbouring IDs
                        # with current point, common neighbouring cell IDs] to temporary list
                        if val is not None:
                            tempFaceList.append([enum + index + 1, val])
                    # If temporary list contains something, iterate through the list
                    if len(tempFaceList) != 0:
                        for k in range(len(tempFaceList)):
                            # Append [[current point ID, point ID who shares common neighbouring cell IDs], common
                            # cell IDs in a set] to final faces list
                            facesList.append([[index, tempFaceList[k][0]], tempFaceList[k][1]])

        if self.connectivityData.nDims == 1:
            # Initialise all cell objects array
            cellsArray = np.empty(nCells, dtype=Seg)
            # Iterate over all cells
            for i in range(nCells):
                # Iterate through all faces, if current cell ID corresponds to an internal face, add face ID to list
                cellInFaceList = [x for x, j in enumerate(facesList, 0) if len({i}.intersection(j[1])) > 0]
                # Cell neighbours IDs list
                neighboursList = []
                # Iterate over all vertices of a shape
                for vertex in range(2):  # 2 if 1D, 3 if 2D tri, 4 if 2D quad
                    # Iterate over all internal faces corresponding to current cell, if vertex IDs correspond to
                    # point IDs that make up said internal face, store neighbouring cell ID to list
                    neighboursList.append(next(iter([list(facesList[face][1] ^ {i})[0] for face in cellInFaceList if
                                                     len({verticesList[i][vertex]}.intersection(
                                                         {facesList[face][0]})) > 0]), None))

                cellsArray[i] = Seg(shapesArray[i], verticesList[i], self.connectivityData.points, neighboursList, nvars
                                    , p1)
        else:
            # Initialise all cell objects array
            cellsArray = np.empty(nCells, dtype=Quad)
            # Iterate over all cells
            for i in range(nCells):
                # Iterate through all faces, if current cell ID corresponds to an internal face, add face ID to list
                cellInFaceList = [x for x, j in enumerate(facesList, 0) if len({i}.intersection(j[1])) > 0]
                # Cell neighbours IDs list
                neighboursList = []
                # Iterate over all vertices of a shape
                for vertex in range(4):  # 2 if 1D, 3 if 2D tri, 4 if 2D quad
                    nextVertex = vertex + 1
                    if nextVertex == 4:  # 2 if 1D, 3 if 2D tri, 4 if 2D quad
                        nextVertex = 0  # go back to first vertex

                    # Iterate over all internal faces corresponding to current cell, if vertex IDs correspond to
                    # point IDs that make up said internal face, store neighbouring cell ID to list
                    neighboursList.append(next(iter([list(facesList[face][1] ^ {i})[0] for face in cellInFaceList if
                                                     len({verticesList[i][vertex],
                                                          verticesList[i][nextVertex]}.intersection(
                                                         set(facesList[face][0]))) > 1]), None))

                cellsArray[i] = Quad(shapesArray[i], verticesList[i], self.connectivityData.points, neighboursList, p1,
                                     p2)

        return ownersArray, cellsArray, pointNeighboursList, facesList

    def common_member(self, a, b):
        """
        https://www.geeksforgeeks.org/python-check-two-lists-least-one-element-common/
        :param a: First list input
        :param b: Second list input
        :return: Common elements if number of common elements > 1, else None
        """
        a_set = set(a)
        b_set = set(b)
        intersect = a_set.intersection(b_set)
        if len(intersect) > self.connectivityData.nDims - 1:  # 0 for 1D, 1 for 2D
            return intersect
        pass

    @staticmethod
    def readBoundaryFromPolyMesh(polyMeshLocation):
        """
        @:brief Read boundary patches data from boundary file and create dictionaries
        :param polyMeshLocation: Location of polyMesh directory in machine
        :return: Returns a dictionary of boundary patches
        """
        # Read in file
        file = open(polyMeshLocation + "/boundary").read()

        # Split the file into tokens
        file = file.split()

        boundaryDict = {}

        # Start with name first
        lookingForName = True
        lookingForType = False
        lookingFornFaces = False
        lookingForboundaryList = False

        foundnFaces = False
        foundboundaryList = False
        assembledBoundary = False

        # Initialise end results
        boundaryArray = None
        nFaces = None
        boundaryName = None
        boundaryType = None

        # Variable used to count different boundary pairs
        pairCounter = 0

        # Variable used to determine cell and face IDs
        componentCounter = 0

        for i in file[2:-1]:
            if (i == "}") or (i == "{"):
                continue

            elif lookingForName:
                boundaryName = i
                lookingForName = False
                lookingForType = True

            elif lookingForType:
                if i == 'type':
                    continue
                else:
                    boundaryType = i.replace(";", "")
                    lookingForType = False
                    lookingFornFaces = True

            elif lookingFornFaces:
                if i == 'nFaces':
                    foundnFaces = True
                elif not foundnFaces:
                    continue
                else:
                    nFaces = i.replace(";", "")
                    foundnFaces = False
                    lookingFornFaces = False
                    lookingForboundaryList = True

                    # Array of (Cell, Face) IDs
                    boundaryArray = np.empty((int(nFaces), 2), dtype=int)

            elif lookingForboundaryList:
                if i == 'boundaryList':
                    foundboundaryList = True
                elif not foundboundaryList:
                    continue
                else:
                    # Convert boundaryList to an array
                    component = ((i.replace('(', '')).replace(')', '')).replace(";", "")

                    boundaryArray[pairCounter][componentCounter] = component

                    if componentCounter == 1:
                        componentCounter = 0
                        pairCounter += 1
                    else:
                        componentCounter += 1

                    if pairCounter == int(nFaces):
                        pairCounter = 0
                        componentCounter = 0
                        foundboundaryList = False
                        lookingForboundaryList = False
                        lookingForName = True

                        assembledBoundary = True

            if assembledBoundary:
                boundaryDict[boundaryName] = \
                    {"type": boundaryType,
                     "nFaces": int(nFaces),
                     "boundaryList": boundaryArray
                     }

                assembledBoundary = False

        return boundaryDict
