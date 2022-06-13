"""
File: dgCell.py

Description: Individual cell object class/struct containing cell and geometry data.
            Abstract base class for extensions of different calculation routines in child classes.
"""
import numpy as np
from abc import ABC, abstractmethod


class DgCell(ABC):
    """
    @:brief Contains information of a single cell
    """
    class DgCellGeometryData:
        """
        @:brief Cell-based shape data subclass
        """
        def __init__(self):
            self.shape = None
            self.pointLabels = None
            self.points = None
            self.neighbourLabels = None

    class DgCellCalc:
        """
        @:brief Calculation routines subclass
        """
        def __init__(self):
            self.V = None
            self.cellCentre = None

    def __init__(self, shape, pointlabels, points, neighbourlabels):
        """
        @:brief Main constructor
        :param shape:           Cell-based shape currently stored as a char
        :param pointlabels:     Point IDs making up this cell stored in a list
        :param points:          Point coordinates from dgMesh class supposedly parsed in here by reference/assignment
        :param neighbourlabels: Neighbouring cell IDs, starting from first face of cell. If no neighbour corresponding
                                to that face, a None value is stored
        """
        self.geomData = self.DgCellGeometryData()
        self.calculations = self.DgCellCalc()

        # Geometry data
        self.geomData.shape = shape
        self.geomData.pointLabels = pointlabels
        self.geomData.points = points
        self.geomData.neighbourLabels = neighbourlabels

        # Calculation routines
        """
        :param V:           Cell volume
        :param cellCentre:  Cell centre
        :param massMatrix:  Cellular mass matrix -> B^T W B
        :param stiffnessMatrix: Cellular stiffness matrix -> (D B)^T W B
        """
        self.calculations.V = self.calculateCellVolume()
        self.calculations.cellCentre = self.calculateCellCentre()

        print("Cell created.\n")

    def calculateCellCentre(self):
        """
        @:brief Average of coordinates of vertices
        :return: Cell centre coordinates of this cell
        """
        # Initialise number of points that make up the cell
        nPointsInCell = 0
        # Initialise end result: (x, y) coordinates
        result = np.zeros(len(self.geomData.points[0]))  # 1 for 1D, 2 for 2D, 3 for 3D

        # For every point in cell
        for i in range(len(self.geomData.pointLabels)):
            # Add up the coordinates of points
            result += self.geomData.points[self.geomData.pointLabels[i]]

            # Counter for numbers of points in cell
            nPointsInCell += 1

        # Divide sums of coordinates by number of points in cell
        result /= nPointsInCell

        return result

    @abstractmethod
    def calculateCellVolume(self):
        """
        @:brief Calculates area of a polygon using the Shoelace formula
        :return: Volume of the cell
        """
        pass
