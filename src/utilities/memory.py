"""
File: memory.py

Description: Test memory usage if Python is parsing inputs by assignment for objects
"""
import numpy as np
from functions import printMemoryUsageInMB as PrintMem


class NumpyData:

    def __init__(self, array):
        self.array = array


if __name__ == '__main__':

    # data = np.repeat(np.array([[1, 2, 3]]), 10, axis=0)
    #
    # Object1 = NumpyData(data)
    #
    # d = {}
    # for i in range(2, 10000):
    #     d["Object{0}".format(i)] = NumpyData(data)
    #
    # PrintMem()
    #
    # data[3][1] = 7
    # print(Object1.array[3])

    """ Test if using same variable name in for loop to initialise class uses extra memory """
    # https://stackoverflow.com/questions/6181935/how-do-you-create-different-variable-names-while-in-a-loop
    d = {}
    for i in range(1, 10):
        d["data{0}".format(i)] = np.repeat(np.array([[1, 2, 3]]), 10, axis=0)

    # print(d["data6"])

    for i in range(1, 10):
        Object = NumpyData(d["data{0}".format(i)])

    print(i)
    PrintMem()
