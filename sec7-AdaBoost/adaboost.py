import numpy as np
import matplotlib.pyplot as plt


def loadSimpData():
    datMat = np.matrix([[1. , 2.1],
                     [2. , 1.1],
                     [1.3, 1. ],
                     [1. , 1. ],
                     [2. , 1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def stumpClassify(dataMatrix, dimen, threashVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threashVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threashVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    # Recurrent each dimension.
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps

        # Recurrent all value according stepSize in this dimension.
        for j in range(-1, int(numSteps)+1):
            # Two judge way.
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                # print("split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f" % \
                #       (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLables, numIt=40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)
    # Record class aggregate value in each data point.
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLables, D)
        print("D: ", D.T)
        alpha = float(0.5*np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alhpa'] = alpha
        weakClassArr.append(bestStump)
        print("classEst: ", classEst.T)
        # which has some question need to study!!!!!!!!!!!!!! as end to end.
        expon = np.multiply(-1*alpha*np.mat(classLables).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        print("aggClassEst: ", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLables).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate, "\n")
        if errorRate == 0.0:
            break
    return weakClassArr


def main():
    datMat, classLabels = loadSimpData()
    # In p121
    # D = np.mat(np.ones((5, 1)) / 5)
    # print(buildStump(datMat, classLabels, D))
    ###################

    # programming 7-2
    classifierArray = adaBoostTrainDS(datMat, classLabels, 9)


if __name__ == '__main__':
    main()
