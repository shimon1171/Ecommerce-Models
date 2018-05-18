# Example of kNN implemented from Scratch in Python

import csv
import random
import math
import operator


def l2_norm_similarty(z1, z2):
    return pow((z1 - z2), 2)

def getNeighbors(tx_id, md, k):
    distances = []
    l2_norm_colunm_names = 'l2_norm'
    md[l2_norm_colunm_names] = (md[fee_colunm_names] - z) ** 2
    md_z = md.sort_values(fee_colunm_names, ascending=[True])
    md_z = md_z.head(k)

    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def find_k_similar_to_z(z , k , md):
    l2_norm_colunm_names = 'l2_norm'
    md[l2_norm_colunm_names] = (md[fee_colunm_names] - z) ** 2
    md_z = md.sort_values(fee_colunm_names, ascending=[True])
    md_z = md_z.head(k)
    return list(md_z[TXID_colunm_names])


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    # prepare data
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset('iris.data', split, trainingSet, testSet)
    print 'Train set: ' + repr(len(trainingSet))
    print 'Test set: ' + repr(len(testSet))
    # generate predictions
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


main()