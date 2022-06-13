"""
File: printFuncs.py

Description: Memory and start statement print functions
"""
import os
import psutil


# from src.foam.fields.include import *

def printMemoryUsageInMB():

    print("\n Memory usage is: " + str(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2) + ' MB \n')


def startStatement(solverName, caseDir, date):

    print("""--------------- PyDG v1.0 ---------------""")
    print("Solver           : " + solverName)
    print("Case             : " + caseDir)
    print("Date and time    : " + date)
    print("\n")
