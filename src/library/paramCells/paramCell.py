"""
File: paramCell.py

Description: Base class for a parametric cell. Contains constant data
"""
from abc import ABC, abstractmethod


class ParamCell(ABC):
    """
    @:brief Base class to represent a cell in parametric space
    """

    @property
    @abstractmethod
    def p(self):
        """
        @:brief p: Polynomial order in the xi_1 direction
        """
        pass

    @p.setter
    @abstractmethod
    def p(self, value):
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
