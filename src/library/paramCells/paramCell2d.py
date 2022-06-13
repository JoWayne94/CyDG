"""
File: paramCell2d.py

Description: Abstract base class for a two-dimensional parametric cell

Add: Triangle subclass
"""
from abc import ABC, abstractmethod
from src.library.paramCells.paramCell import ParamCell as Cell


class ParamCell2d(Cell, ABC):
    """
    @:brief Abstract subclass to represent a 2d cell in parametric space
    """
    # def __init__(self, quadrature, p, q, nNodesP=None, nNodesQ=None):
    #     """
    #     @:brief Main constructor
    #     """
    #     super().__init__(quadrature, p)
    #     if nNodesP is None:
    #         self.nNodesP = 0
    #     else:
    #         self.nNodesP = nNodesP
    #
    #     if nNodesQ is None:
    #         self.nNodesQ = 0
    #     else:
    #         self.nNodesQ = nNodesQ

    @property
    @abstractmethod
    def q(self):
        """
        @:brief q: Polynomial order in the xi_2 direction
        """
        pass

    @abstractmethod
    def GetQuadratureZerosWeights(self):
        """
        @:brief Quadrature zeros and weights to be stored as const data, overloaded depending on shapes
        :return: Quadrature point coordinates in [-1, 1] and weights
        """
        return None, None
