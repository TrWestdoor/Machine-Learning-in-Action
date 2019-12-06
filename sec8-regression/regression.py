import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    fr.close()
    return dataMat, labelMat


def standRegres(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def resError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i-10))
        wMat[i, :] = ws.T
    return wMat


def regularize(x_mat):
    xMeans = np.mean(x_mat, 0)
    xVar = np.var(x_mat, 0)
    return (x_mat - xMeans) / xVar


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    """
    This part code is different with book, cause primary code can't directly run.
    So i fix some error content.
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean

    # xMat = np.regularize(xMat)
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar

    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowerError = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = resError(yMat.A, yTest.A)
                if rssE < lowerError:
                    lowerError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


from time import sleep
import json
from urllib.request import urlopen


def search_for_set(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'get from code.google.com'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d' \
                '&alt=json' % (myAPIstr, setNum)
    pg = urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, sellingPrice])
                    retY.append(sellingPrice)
        except:
            print("problem with item %d" % i)


def setDataCollect(retX, retY):
    search_for_set(retX, retY, 8288, 2006, 800, 49.99)
    # search_for_set(retX, retY, 10030, 2002, 3096, 269.99)
    # search_for_set(retX, retY, 10179, 2007, 5195, 499.99)
    # search_for_set(retX, retY, 10181, 2007, 3428, 199.99)
    # search_for_set(retX, retY, 10189, 2008, 5922, 299.99)
    # search_for_set(retX, retY, 10196, 2009, 3263, 249.99)


def main():
    # xArr, yArr = loadDataSet('ex0.txt')
    # print(xArr[0:2])

    # ws = standRegres(xArr, yArr)
    # # print(ws)
    # xMat = np.mat(xArr)
    # yMat = np.mat(yArr)
    # yHat = xMat * ws

    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()
    '''
    # print(np.corrcoef(yHat.T, yMat))
    '''
    # 8-2
    yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    xMat = np.mat(xArr)
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s=2, c='red')
    plt.show()
    '''

    '''
    # 8-3, abalone age predict
    abX, abY = loadDataSet('abalone.txt')
    yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    print('++++++++++++predict train set++++++++++++++')
    print(resError(abY[0:99], yHat01.T))
    print(resError(abY[0:99], yHat1.T))
    print(resError(abY[0:99], yHat10.T))
    print('++++++++++++predict test set+++++++++++++++')
    yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    print(resError(abY[100:199], yHat01.T))
    yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    print(resError(abY[100:199], yHat1.T))
    yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    print(resError(abY[100:199], yHat10.T))
    print('++++++++++compare simple regression++++++++')
    ws = standRegres(abX[0:99], abY[0:99])
    yHat = np.mat(abX[100:199]) * ws
    print(resError(abY[100:199], yHat.T.A))
    '''

    # abX, abY = loadDataSet('abalone.txt')
    # ridgeWeights = ridgeTest(abX, abY)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(ridgeWeights)
    # plt.show()

    # 8.4.3
    # xArr, yArr = loadDataSet('abalone.txt')
    # # print(stageWise(xArr, yArr, 0.001, 5000))
    # xMat = np.mat(xArr)
    # yMat = np.mat(yArr).T
    # xMat = regularize(xMat)
    # yM = np.mean(yMat, 0)
    # yMat = yMat - yM
    # weights = standRegres(xMat, yMat.T)
    # print(weights)

    # 8.6
    # Because given website address has been down, so these code cannot run successfully.
    # We will try found new address and fix these code.
    # 原网址已经关闭，我会尝试找到原来的数据修改代码，或者寻找谷歌放置此API的新网址并修正代码。
    # lgX = []
    # lgY = []
    # setDataCollect(lgX, lgY)
    # print(lgX)


if __name__ == '__main__':
    main()
