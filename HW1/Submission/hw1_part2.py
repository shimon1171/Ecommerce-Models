"""
HW1 part2
"""

import numpy as np
import random
import pandas as pd
from datetime import datetime as dt
import collections
from matplotlib import pyplot as plt
from collections import Counter
import math
import operator
import random
import networkx as nx
import numpy as np
from sklearn.model_selection import KFold
import csv

time = 1098770179

"""
competitive part
"""


def predict_future_links(k):
    df = read_graph_by_time_file()
    number_of_nodes_to_predict = 10000
    node_probability = get_probability_for_each_node(df)
    i = 0.90
    neighbors_node_probability = get_probability_for_each_neighbors_node(df, node_probability, i)
    future_links = []
    for l in range(0, number_of_nodes_to_predict):
        node1 = random_distr(node_probability)
        neighbors = neighbors_node_probability[node1]
        node2 = random_distr(neighbors)
        future_links.append((node1, node2))

    write_file(future_links,"037036209","00000")

def write_file(edges, id1, id2=None):
    """writing dictionary to a csv files"""
    if id2 is None:
        filename = 'hw1_part2_{}_.csv'.format(id1)
    else:
        filename = 'hw1_part2_{}_{}.csv'.format(id1, id2)
    with open(filename, 'w') as csv_file:
        writer = csv.writer(csv_file, lineterminator='\n')
        fieldnames2 = ["user1", "user2"]
        writer.writerow(fieldnames2)
        for edge in edges:
            writer.writerow([edge[0], edge[1]])

# <editor-fold desc="Competitive part">
graph_by_time_file_name = '..\graph_by_time.txt'
user1_column_name = 'user1'
user2_column_name = 'user2'
ts_column_name = 'ts'

h_graph_file_name = 'h_graph.csv'
h_graph_user1_column_name = 'user1'
h_graph_user2_column_name = 'user2'
h_graph_weight_column_name = 'weight'

strong_edge = "s"
weak_edge = "w"

def read_graph_by_time_file():
    with open(graph_by_time_file_name) as f:
        df = pd.read_table(f, sep=' ', index_col=False, header=0, lineterminator='\n')
    df = df.sort_values(by=ts_column_name)
    return df

def get_probability_for_each_node(df):
    degrees = get_out_degree_for_all_nods(df)
    probabilities = normalizing_dictionary_values(degrees)
    return probabilities

def get_probability_for_each_neighbors_node(df , node_probability , factor_neighbor_probability):
    probabilities = {}
    graph = get_for_each_node_is_neighbors(df);
    for node in graph.keys():
        frequencies = Counter(graph[node])
        frequencies = normalizing_dictionary_values(frequencies)
        for neighbor in frequencies.keys():
            neighbor_general_weight = node_probability[neighbor]
            neighbor_probability = frequencies[neighbor]
            frequencies[neighbor] = (1-factor_neighbor_probability) * neighbor_general_weight + factor_neighbor_probability * neighbor_probability
        probabilities[node] = normalizing_dictionary_values(frequencies)
    return probabilities

def get_out_degree_for_all_nods(df):
    degrees = {}
    all_nodes_set = set(df[user1_column_name]) | set(df[user2_column_name])
    for node in all_nodes_set:
        degrees[node] = 0
    df_group_by_user1 = df.groupby(by=user1_column_name)
    for node, user1_group in df_group_by_user1:
        degrees[node] = len(user1_group)
    return degrees

def random_distr(l):
    r = random.uniform(0, 1)
    s = 0
    for item in l.keys():
        prob = l[item]
        s += prob
        if s >= r:
            return item
    return item  # Might occur because of floating point inaccuracies

def normalizing_dictionary_values(d):
    if(len(d) == 0 or sum(d.itervalues()) == 0):
        return d
    factor = 1.0 / sum(d.itervalues())
    for k in d:
        d[k] = d[k] * factor
    key_for_max = max(d.iteritems(), key=operator.itemgetter(1))[0]
    diff = 1.0 - math.fsum(d.itervalues())
    d[key_for_max] += diff
    return d

def get_for_each_node_is_neighbors(df):
    graph = {}
    df_group_by_user1 = df.groupby(by=user1_column_name)
    for user1, user1_group in df_group_by_user1:
        graph[user1] = []
        for index, row in user1_group.iterrows():
            user2 = row[user2_column_name]
            graph[user1].append(user2)
    return graph

# </editor-fold>


if __name__ == '__main__':
    predict_future_links(10000)


###########################################
###########################################
###########################################
###########################################
###########################################
###########################################
###########################################
###########################################
###########################################
###########################################
###########################################





# <editor-fold desc="Competitive part">













def predict_future_links(k):
    df = read_graph_by_time_file()
    node_probability = get_probability_for_each_node(df)
    neighbors_node_probability = get_probability_for_each_neighbors_node(df , node_probability,  0.8)

    future_links = []
    for i in range(0,k):
        node1 = random_distr(node_probability)
        neighbors = neighbors_node_probability[node1]
        node2 = random_distr(neighbors)
        future_links.append( (node1,node2))

    return future_links

def save_predict_future_links_in_file(future_links):
    df = pd.DataFrame(columns=[user1_column_name,user2_column_name])
    df_index = 0
    for link in future_links:
        user1 = link[0]
        user2 = link[1]
        data = [user1, user2]
        df.loc[df_index] = data
        df_index = df_index + 1
    df.to_csv('predict_future_link.csv',index=False)


#cross validation
def cross_validation(df):

    n = len(df)
    k = int(n / (n * 0.25))

    vlaues = range(10, 21, 1) # [0 ,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    vlaues = [x / 2.0 for x in vlaues]
    vlaues = [x / 10.0 for x in vlaues]

    #vlaues = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    vlaues_dic = {}
    for i in vlaues:
        vlaues_dic[i] = 0

    it = 1
    for yy in range(0,5):
        kf = KFold(n_splits=k, shuffle=True, random_state=2)
        for train_index, test_index in kf.split(df):
            print("Iteratio {}".format(it))
            it = it + 1
            train = df.iloc[train_index]
            test = df.iloc[test_index]

            test_node = get_nodes(test)
            test_node_size = len(test_node)

            node_probability = get_probability_for_each_node(train)
            for i in vlaues:
                neighbors_node_probability = get_probability_for_each_neighbors_node(train, node_probability, i)
                future_links = []
                for l in range(0, test_node_size):
                    node1 = random_distr(node_probability)
                    neighbors = neighbors_node_probability[node1]
                    node2 = random_distr(neighbors)
                    future_links.append((node1, node2))

                c = 0
                test_node_temp = list(test_node)
                for future_link in future_links:
                    if future_link in test_node_temp:
                        test_node_temp.remove(future_link)
                        c = c + 1

                p = 1.0*c / test_node_size
                print("For lamda {} recive {}".format(i,p))
                if vlaues_dic[i] == 0:
                    vlaues_dic[i] = p
                else:
                    vlaues_dic[i] = (vlaues_dic[i] + p) / 2

    print("final result : ")
    max = -1

    for key in vlaues_dic.keys():
        print("For lamda {} recive {}".format(key, vlaues_dic[key]))
        if max == -1:
            max = vlaues_dic[key]
            maxkey = key
        elif max < vlaues_dic[key]:
            max = vlaues_dic[key]
            maxkey = key

    print("Max lamda {} recive {}".format(maxkey, max))

def get_nodes(df):
    tupels = []
    for index, row in df.iterrows():
        user1 = row[user1_column_name]
        user2 = row[user2_column_name]
        tupels.append((user1, user2))
    return tupels



# </editor-fold>


