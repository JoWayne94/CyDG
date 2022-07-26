"""
File: geometry.py

Description: Abstract base class for various shapes. Contains constant data

Add: Triangle subclass
"""
from abc import ABC, abstractmethod


class Geometry(ABC):

    def __init__(self):
        """
        @:brief Main constructor
        """
        pass

    @abstractmethod
    def parametricMapping(self):
        pass

    @abstractmethod
    def detJacobian(self):
        pass
