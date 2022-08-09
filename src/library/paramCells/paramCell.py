"""
File: paramCell.py

Description: Abstract base class for a parametric cell. Contains constant data
"""
from abc import ABC, abstractmethod


class ParamCell(ABC):
    """
    @:brief Base class to represent a cell in parametric space
    """

    @property
    @abstractmethod
    def p1(self):
        """
        @:brief p1: Polynomial order in the xi_1 direction
        """
        pass

    @p1.setter
    @abstractmethod
    def p1(self, value):
        return None

    @property
    @abstractmethod
    def quadrature(self):
        """
        @:brief quadrature: Type of quadrature selected
        """
        pass

    @property
    @abstractmethod
    def zeros(self):
        """
        @:brief zeros: Zeros of polynomials
        """
        pass

    @property
    @abstractmethod
    def weights(self):
        """
        @:brief weights: Quadrature weights
        """
        pass
