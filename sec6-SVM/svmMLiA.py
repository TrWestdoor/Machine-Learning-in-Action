import numpy as np


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    fr.close()
    return dataMat, labelMat


def selectJrand(i, m):
    j = i
    while(j == i):
        j = int(np.random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while (iter <  maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T)) + b
            


def main():
    dataArr, labelArr = loadDataSet('testSet.txt')
    print(labelArr)


if __name__ == '__main__':
    main()
