from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


def file2matrix(filename):
    fr = open(filename)
    # 此处书中为 fr.readline(),根据上下文代码判断，应当为 fr.readlines()函数才正确。
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0 * array(datingLabels))
plt.show()
