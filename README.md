# CyDG 
(v1.0) is a cythonizable, open-source, modal discontinuous Galerkin code written in Python for prototyping purposes.
Currently, CyDG solves first- and second-order linear systems of Partial Differential Equations (PDEs).


**Background**

Author: Jo Wayne Tan, 2022.

The source code is in the src/ folder.

Implemented ADR class equation sets.

scriptSetCython does a Cython compilation of the most computational intensive functions.

scriptUnsetCython reverts the code back to a standard Python implementation.


**Setup**

Library requirements to be installed are listed in requirements.txt (tested version numbers provided).


**Using CyDG**

Examples of Allrun and Allclean scripts are in the tutorials/ directory, with examples for different solvers.

All parameters required for running the case are defined in caseSetup.py.

Meshes are in a cell-based format and in the constant/polyMesh location inside the case directory, generated using meshGen.py.


**Dependencies**

The code is tested using:

python 3.9  
numpy 1.16.6  
scipy 1.7.3  
cython 0.29.30  
psutil 5.8.0  
