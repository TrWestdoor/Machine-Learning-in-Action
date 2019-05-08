import numpy as np


def loadSimpData():
    datMat = np.matrix([[1. , 2.1],
                     [2. , 1.1],
                     [1.3, 1. ],
                     [1. , 1. ],
                     [2. , 1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def main():
    datMat, classLabels = loadSimpData()


if __name__ == '__main__':
    main()
