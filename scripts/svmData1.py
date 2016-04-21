import numpy as np
import csv
import random
import math
import sys
from sklearn import svm
from sklearn import decomposition


def loadCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [x for x in dataset[i]]
    return dataset


def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


def createXandy(data):
    X = []
    y = []
    for sample in data:
        new_sample = sample[:len(sample)-1]
        X.append(new_sample)
        y.append(sample[-1])
    return [np.array(X), y]


def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


def main(filename):
    filename = filename[0]
    splitRatio = 0.5
    dataset = loadCsv(filename)
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
    X = createXandy(trainingSet)[0]
    y = createXandy(trainingSet)[1]

    pca = decomposition.PCA()
    pca.n_components = 2
    X_reduced = pca.fit_transform(X)

    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(X, y)
    clf2 = svm.SVC(kernel='rbf', C=1.0)
    clf2.fit(X, y)
    clf3 = svm.SVC(kernel='linear', C=1.0)
    clf3.fit(X_reduced, y)
    clf4 = svm.SVC(kernel='rbf', C=1.0)
    clf4.fit(X_reduced, y)

    testX = createXandy(testSet)[0]

    testX_reduced = pca.fit_transform(testX)

    predictions = (clf.predict(testX)).tolist()
    accuracy = round(getAccuracy(testSet, predictions), 2)
    print('Accuracy linear (all components): {0}%').format(accuracy)
    predictions2 = (clf2.predict(testX)).tolist()
    accuracy2 = round(getAccuracy(testSet, predictions2), 2)
    print('Accuracy RBF (all components): {0}%').format(accuracy2)

    predictions3 = (clf3.predict(testX_reduced)).tolist()
    accuracy3 = round(getAccuracy(testSet, predictions3), 2)
    print('Accuracy linear (reduced components): {0}%').format(accuracy3)
    predictions4 = (clf4.predict(testX_reduced)).tolist()
    accuracy4 = round(getAccuracy(testSet, predictions4), 2)
    print('Accuracy RBF (reduced components): {0}%').format(accuracy4)


if __name__ == '__main__':
    main(sys.argv[1:])
