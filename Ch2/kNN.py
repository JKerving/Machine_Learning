from numpy import *
import operator
from os import listdir


def create_data_set():
    group = array(
        [
            [1.0, 1.1],
            [1.0, 1.0],
            [0, 0],
            [0, 0.1]
        ]
    )
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    print(sortedClassCount)
    return sortedClassCount[0][0]


group, labels = create_data_set()
label = classify0([1, 1], group, labels, 3)
print(label)


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.50
    datingDataMat, datingLabels = file2matrix(
        'C:/JKerving/Programming Documents/Machine Learning Action/MLiA_SourceCode/machinelearninginaction/Ch02/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print('the classifier came back with: %d,the real answer is : %d' % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print('the total error rate is :%f', errorCount / float(numTestVecs))
    print(errorCount)


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir(
        'C:/JKerving/Programming Documents/Machine Learning Action/MLiA_SourceCode/machinelearninginaction/Ch02/digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('C:/JKerving/Programming Documents/Machine Learning Action/MLiA_SourceCode/machinelearninginaction/Ch02/digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('C:/JKerving/Programming Documents/Machine Learning Action/MLiA_SourceCode/machinelearninginaction/Ch02/digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('C:/JKerving/Programming Documents/Machine Learning Action/MLiA_SourceCode/machinelearninginaction/Ch02/digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print('the classifier came back with: %d,the real answer is: %d' % (classifierResult,classNumStr))
        if(classifierResult != classNumStr):
            errorCount += 1.0
    print('the total number of errors is: %d' % errorCount)
    print('the total error rate is: %f' % (errorCount/float(mTest)))

handwritingClassTest()
# datingDataMat, datingLabels = file2matrix(
#     'C:/JKerving/Programming Documents/Machine Learning Action/MLiA_SourceCode/machinelearninginaction/Ch02/datingTestSet2.txt')
# print(datingDataMat)
# print(datingLabels)
#
# import matplotlib
# import matplotlib.pyplot as plt
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
# plt.show()
