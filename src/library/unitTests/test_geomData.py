import unittest
from src.library.dgMesh.dgMesh import *


class TestCase(unittest.TestCase):

    mesh1d = None
    mesh2d = None

    @classmethod
    def setUpClass(cls) -> None:
        # Read in the mesh
        cls.mesh1d = DgMesh.constructFromPolyMeshFolder("/Users/jwtan/PycharmProjects/PyDG/polyMesh/2x0", 1)
        cls.mesh1d.constructShapeBasedCells("/Users/jwtan/PycharmProjects/PyDG/polyMesh/2x0", 1)
        cls.mesh2d = DgMesh.constructFromPolyMeshFolder("/Users/jwtan/PycharmProjects/PyDG/polyMesh/2x2", 2)
        cls.mesh2d.constructShapeBasedCells("/Users/jwtan/PycharmProjects/PyDG/polyMesh/2x2", 1, 1)

    def tearDown(self) -> None:

        pass

    def test_volume(self):
        self.assertEqual(TestCase.mesh1d.connectivityData.cells[0].calculations.V, 2)  # add assertion here
        self.assertEqual(TestCase.mesh1d.connectivityData.cells[1].calculations.V, 3)
        self.assertEqual(TestCase.mesh2d.connectivityData.cells[0].calculations.V, 1)
        self.assertEqual(TestCase.mesh2d.connectivityData.cells[1].calculations.V, 1)
        self.assertEqual(TestCase.mesh2d.connectivityData.cells[2].calculations.V, 1)
        self.assertEqual(TestCase.mesh2d.connectivityData.cells[3].calculations.V, 1)

    def test_cellCentre(self):
        self.assertEqual(TestCase.mesh1d.connectivityData.cells[0].calculations.cellCentre, 1)
        self.assertEqual(TestCase.mesh1d.connectivityData.cells[1].calculations.cellCentre, 3.5)
        self.assertEqual(TestCase.mesh2d.connectivityData.cells[0].calculations.cellCentre[0], 0.5)
        self.assertEqual(TestCase.mesh2d.connectivityData.cells[0].calculations.cellCentre[1], 0.5)
        self.assertEqual(TestCase.mesh2d.connectivityData.cells[1].calculations.cellCentre[0], 1.5)
        self.assertEqual(TestCase.mesh2d.connectivityData.cells[1].calculations.cellCentre[1], 0.5)
        self.assertEqual(TestCase.mesh2d.connectivityData.cells[2].calculations.cellCentre[0], 0.5)
        self.assertEqual(TestCase.mesh2d.connectivityData.cells[2].calculations.cellCentre[1], 1.5)
        self.assertEqual(TestCase.mesh2d.connectivityData.cells[3].calculations.cellCentre[0], 1.5)
        self.assertEqual(TestCase.mesh2d.connectivityData.cells[3].calculations.cellCentre[1], 1.5)


if __name__ == '__main__':
    unittest.main()
