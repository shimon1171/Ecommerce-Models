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


# def predict_future_links(k):
#     df = read_graph_by_time_file()
#     number_of_nodes_to_predict = 10000
#     node_probability = get_probability_for_each_node(df)
#     i = 0.90
#     neighbors_node_probability = get_probability_for_each_neighbors_node(df, node_probability, i)
#     future_links = []
#     for l in range(0, number_of_nodes_to_predict):
#         node1 = random_distr(node_probability)
#         neighbors = neighbors_node_probability[node1]
#         node2 = random_distr(neighbors)
#         future_links.append((node1, node2))
#
#     write_file(future_links,"037036209","00000")

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
graph_by_time_file_name = 'graph_by_time.txt'
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


def get_degree_for_all_nods(df , nodes):
    degrees = {}
    all_nodes_set = nodes
    for node in nodes:
        degrees[node] = 0
    df_group_by_user1 = df.groupby(by=user1_column_name)
    for node, user1_group in df_group_by_user1:
        degrees[node] = len(user1_group)

    df_group_by_user2 = df.groupby(by=user2_column_name)
    for node, user2_group in df_group_by_user2:
        degrees[node] = degrees[node] + len(user2_group)

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



def get_all_graph_nodes(df):
    all_nodes_set = set(df[user1_column_name]) | set(df[user2_column_name])
    return all_nodes_set

def predict_future_links(k):
    df = read_graph_by_time_file()
    nodes = list(get_all_graph_nodes(df))

    list_tupels_of_edge = []
    for i in range(0,len(nodes)-1):
        for j in range(i+1, len(nodes) - 1):
            list_tupels_of_edge.append( (nodes[i], nodes[j]) )


    H = build_h_graph(df,nodes )

    edges_popularity = get_edges_popularity(list_tupels_of_edge, df)
    neighborhood_overlap = get_neighborhood_overlap(list_tupels_of_edge, H)
    probability = get_probability_for_each_edges(list_tupels_of_edge ,edges_popularity , neighborhood_overlap , 0.5)


    # number_of_nodes_to_predict = 10000
    # node_probability = get_probability_for_each_node(df)
    # i = 0.90
    # neighbors_node_probability = get_probability_for_each_neighbors_node(df, node_probability, i)
    # future_links = []
    # for l in range(0, number_of_nodes_to_predict):
    #     node1 = random_distr(node_probability)
    #     neighbors = neighbors_node_probability[node1]
    #     node2 = random_distr(neighbors)
    #     future_links.append((node1, node2))
    #
    # write_file(future_links,"037036209","00000")



def build_h_graph(df , nods):
    G = nx.Graph()
    G.add_nodes_from(list(nods))
    graph = {}
    for index, row in df.iterrows():
        user1 = row[user1_column_name]
        user2 = row[user2_column_name]
        user1_to_user2 = (user1, user2)
        user2_to_user1 = (user2, user1)
        if user1_to_user2 not in graph and user2_to_user1 not in graph:
            graph[user1_to_user2] = weak_edge
            G.add_edge(user1, user2 , weight=weak_edge)
        elif user2_to_user1 in graph:
            graph[user2_to_user1] = strong_edge
            G.add_edge(user1, user2, weight=strong_edge)

    return G

def get_probability_for_each_edges(list_tupels_of_edge ,dges_popularity , neighborhood_overlap , l ):
    probabilities = {}

    for edge_data in list_tupels_of_edge:
        probabilities[edge_data] = l * dges_popularity[edge_data] + (1-l)*neighborhood_overlap[edge_data]

    return probabilities



def get_probability_for_each_node(df , nodes):
    degrees = get_degree_for_all_nods(df , nodes)
    probabilities = normalizing_dictionary_values(degrees)
    return probabilities

def get_edges_popularity(list_tupels_of_edge , df , nodes):
    degrees = get_degree_for_all_nods(df , nodes)
    probabilities = normalizing_dictionary_values(degrees)
    edges_popularity = {}
    for edge_data in list_tupels_of_edge:
        x = edge_data[0]
        y = edge_data[1]
        px = probabilities[x]
        py = probabilities[y]
        edges_popularity[(x, y)] = 0.5 * px + 0.5 * py
    return edges_popularity


def get_neighborhood_overlap(list_tupels_of_edge,G , nodes):
    edges_no_overlap = {}
    for edge_data in list_tupels_of_edge:
        x = edge_data[0]
        y = edge_data[1]
        Ex = set(G.neighbors(x))
        Ey = set(G.neighbors(y))
        if len(Ex) == 0 or len(Ey) == 0:
            edges_no_overlap[(x, y)] = 0
            continue
        union = Ex | Ey
        intersection = Ex & Ey
        no_xy = float(len(intersection)) / float(len(union))
        edges_no_overlap[(x,y)] = no_xy
    return edges_no_overlap

def get_train_df(df,train):
    isFirst = True
    for train_row in train:
        x = train_row[0]
        y = train_row[1]
        df_temp = df[ (df[user1_column_name].isin([x,y]) ) ]
        df_temp = df_temp[ (df_temp[user2_column_name].isin([x,y]) ) ]
        #df_temp = df[(df[user1_column_name] == x & df[user1_column_name] == y) | (df[user1_column_name] == y & df[user1_column_name] == x)]
        if isFirst :
            train_df = df_temp
            isFirst = False
        else:
            frames = [train_df, df_temp]
            train_df = pd.concat(frames)
    return train_df

# def remove_train_edges(list_tupels_of_edge ,train ):
#     list_tupels_of_edge_to_check = []
#     for i in range(0, len(list_tupels_of_edge)):
#         edge = list_tupels_of_edge[i]
#         x = edge[0]
#         y = edge[1]
#         if (x,y) not in train and (y,x) not in train:
#             list_tupels_of_edge_to_check.append(edge)
#     return list_tupels_of_edge_to_check

def remove_train_edges(list_tupels_of_edge ,train, train2 ):
    s = set(list_tupels_of_edge)
    t = set(train)
    t2 = set(train2)
    s = s - t
    s = s - t2
    list_tupels_of_edge_to_check = list(s)

    return list_tupels_of_edge_to_check


def get_edge_driection(list):
    list2 = []
    for r in range(0, len(list)):
        edge = list[r]
        x = edge[0]
        y = edge[1]
        list2.append((y, x))
    return list2

def cross_validation():

    df = read_graph_by_time_file()
    nodes = list(get_all_graph_nodes(df))
    H = build_h_graph(df, nodes)
    edges = H.edges()
    edges_list = list(edges)

    n = len(edges_list)
    k = int(n / (n * 0.20))

    list_tupels_of_edge = []
    for i in range(0, len(nodes) - 1):
        for j in range(i + 1, len(nodes) - 1):
            list_tupels_of_edge.append((nodes[i], nodes[j]))

    # vlaues = range(10, 21, 1) # [0 ,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # vlaues = [x / 2.0 for x in vlaues]
    # vlaues = [x / 10.0 for x in vlaues]

    vlaues = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    vlaues_dic = {}
    for i in vlaues:
        vlaues_dic[i] = 0

    it = 1
    for yy in range(0, 5):
        kf = KFold(n_splits=k, shuffle=True, random_state=2)
        for train_index, test_index in kf.split(edges_list):
            print("Iteratio {}".format(it))
            it = it + 1
            train = [edges_list[index] for index in train_index]
            train2 = get_edge_driection(train)

            test = [edges_list[index] for index in test_index]
            test2 = get_edge_driection(test)

            test_set = set(test) | set(test2)

            train_df = get_train_df(df,train)
            list_tupels_of_edge_to_check = remove_train_edges(list_tupels_of_edge ,train ,train2)

            H = build_h_graph(train_df, nodes)
            edges_popularity = get_edges_popularity(list_tupels_of_edge_to_check, train_df , nodes)
            neighborhood_overlap = get_neighborhood_overlap(list_tupels_of_edge_to_check, H , nodes)

            for l in vlaues:
                probability = get_probability_for_each_edges(list_tupels_of_edge_to_check, edges_popularity, neighborhood_overlap, l)
                future_links = dict(sorted(probability.iteritems(), key=operator.itemgetter(1), reverse=True)[:10000]).keys()

                future_links2 = get_edge_driection(future_links)
                future_links_set = set(future_links)

                c = len(test_set & future_links_set)
                print("test_set len {0} , future_links_set len {1} , interction {2} ".format(len(test_set) , len(future_links_set) , c))
                p = 1.0 * c / len(test_set)
                print("For lamda {} recive {}".format(l, p))
                if vlaues_dic[l] == 0:
                    vlaues_dic[l] = p
                else:
                    vlaues_dic[l] = (vlaues_dic[l] + p) / 2

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



# </editor-fold>


if __name__ == '__main__':
    cross_validation()
    predict_future_links(10000)