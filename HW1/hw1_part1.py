"""
HW1 part1
"""

import random

import numpy as np
# Necessary to run on the server, without display
from matplotlib import use

use('Agg')
from matplotlib import pyplot as plt

t = 1082791719

'''
Question 2 parts 2.a-2.c
'''


def H_features(time=t):
    """
    TODO:
    this is an example code of 3 arbitrary nodes with arbitrary features for each node,
    you should delete this code and write your own, according to the instructions in HW1 document
    """
    nodes_feat_dict = {}
    nodes = range(0, 6, 2)
    print(len(nodes))
    num_features = 4
    demovalues = [np.random.randn() for _ in nodes]
    demivalues = [random.uniform(0, 1) for _ in nodes]
    degree = [random.randint(0, 100) for _ in nodes]
    clustering_coeff = [random.uniform(0, 1) for _ in nodes]
    closeness_centrality = [random.choice(demovalues) for _ in nodes]
    betweenness_centrality = [random.sample(demivalues, 1) for _ in nodes]
    for i, node in enumerate(nodes):
        nodes_feat_dict[node] = [degree[i], clustering_coeff[i], closeness_centrality[i], betweenness_centrality[i]]
    """
    example return format:
    {0:[1,0.2,0.3,0.4], 2:[5,0.1,0.2,0.3], 3:[3,0.01,0.7,0.9]}
    """
    print('nodes_feat_dict:', nodes_feat_dict)
    return nodes_feat_dict


"""
Question 3
"""


def calc_no(time=t):
    """
    TODO:
    this is an example code of 3 arbitrary nodes with arbitrary features for each node!!!
    you should delete this code and write your own as in the HW1 document instructions
    """
    cont = [np.random.randn() for _ in range(500)]
    plt.hist(cont)
    plt.xlabel('Neighborhood Overlap for Strong Edges')
    plt.ylabel('Frequencies')
    plt.title('Histogram of Neighborhood Overlap for Strong Edges')
    plt.savefig('HistogramExample.png')
    plt.close()
    # In order to show, comment save and close functions and uncomment show function
    # plt.show()

    edges_no_overlap = {(n, 2 * n): n / 100 for n in range(1, 100, 13)}
    """
    example return format:
    {(1, 2):0.7, (1, 5):0.3, (1,3): 0.001}
    """
    return edges_no_overlap


'''
Question 4
'''


def stc_index(time=t):
    """
    TODO:
    this is an example code of 50 arbitrary nodes when some are collected randomly as vilates stc
    you should delete this code and write your own as in the HW1 document
    """
    indices = set(np.arange(0, 3, 0.3))
    nodes = np.arange(1, 10)
    n = random.randint(4, 10)
    random_stc_index = {}
    for node in nodes:  # Pick a random number between 1 and 100.
        index = random.sample(indices, 1)[0]
        random_stc_index[node] = index
    """
    example return format:
    {1:0.7, 3:0.3, 2: 0.001}
    """
    return random_stc_index


######################################################

def main():
    print(H_features())
    # print(calc_no())
    print(stc_index())


if __name__ == '__main__':
    main()
