from numpy import *


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open(
        'C:/JKerving/Programming Documents/Machine Learning Action/MLiA_SourceCode/machinelearninginaction/Ch05/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


dataMat, labelMat = loadDataSet()
print(dataMat)
print(labelMat)
