"""
File: factory.py

Description: Prototype factory classes
"""
from enum import Enum, auto


class ShapeEnum(Enum):
    """
    @:brief Cell shape enumeration
    """
    Segment = auto()
    Quad = auto()
