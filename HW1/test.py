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

def nCk(n,k):
    f = math.factorial
    return (f(n) / (f(k) * f(n-k)))



def normalizing_dictionary_values(d):
    factor = 1.0 / sum(d.itervalues())
    for k in d:
        d[k] = d[k] * factor
    key_for_max = max(d.iteritems(), key=operator.itemgetter(1))[0]
    diff = 1.0 - math.fsum(d.itervalues())
    d[key_for_max] += diff
    return d

def read_graph_by_time_file():
    with open(graph_by_time_file_name) as f:
        df = pd.read_table(f, sep=' ', index_col=False, header=0, lineterminator='\n')
    #df[ts_column_name] = df[ts_column_name].apply(lambda x: dt.fromtimestamp(x))
    df = df.sort_values(by=ts_column_name)
    return df

def get_graph_by_time_as_multiGraph(df):
    all_nodes_set = set(df[user1_column_name]) | set(df[user2_column_name])
    G = nx.MultiGraph()
    #for node in all_nodes_set:
    G.add_nodes_from(list(all_nodes_set))
    for index, row in df.iterrows():
        user1 = row[user1_column_name]
        user2 = row[user2_column_name]
        G.add_edge(user1, user2)
    return G

def get_all_graph_nodes(df):
    all_nodes_set = set(df[user1_column_name]) | set(df[user2_column_name])
    return all_nodes_set

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

def build_h_graph_with_time(df , time):
    nodes = get_all_graph_nodes(df)
    filtered_df = df[ (df[ts_column_name] < time)] # get all samples that are lower that the given time
    return build_h_graph(filtered_df, nodes)




def compute_distance_between_every_pair_of_nodes(G):
    spl = nx.all_pairs_shortest_path_length(G)
    return spl

def create_histogram_as_fig_2_10(frequencies , file_name="Empyt"):
    plt.xlabel('Number of intermediaries')
    plt.ylabel("Number of paris", rotation=90)
    plt.plot(frequencies.keys(), frequencies.values(), '-bo')
    if(file_name != "Empyt"):
        plt.savefig(file_name, format='png', dpi=600)
    plt.close()

def create_histogram_as_fig_2_11(frequencies , file_name="Empyt"):
    normalizing_frequencies = normalizing_dictionary_values(frequencies)
    plt.xlabel('i, Path length in graph')
    plt.ylabel("P(i), Probability", rotation=90)
    plt.plot(normalizing_frequencies.keys(), normalizing_frequencies.values(), '-bo')
    if(file_name != "Empyt"):
        plt.savefig(file_name, format='png', dpi=600)
    plt.close()

def Q1():
    df = read_graph_by_time_file()
    nodes = get_all_graph_nodes(df)
    H = build_h_graph(df, nodes)
    spl = compute_distance_between_every_pair_of_nodes(H)

    nodes = H.nodes()
    length_list = []
    for node1_data in spl:
        node1 = node1_data[0]
        dic_of__length = node1_data[1]
        for node2 in nodes:
            if dic_of__length.has_key(node2):
                length_list.append(dic_of__length[node2])

    frequencies = Counter(length_list)
    create_histogram_as_fig_2_10(frequencies , 'q1a_2_10_fig.png')
    create_histogram_as_fig_2_11(frequencies , 'q1a_2_11_fig.png')










def compute_degree(G):
    degrees = {}
    for (node, val) in G.degree():
        degrees[node] = val
    return degrees

# def compute_clustering_coefficiente(G):
# def compute_closeness_centrality(G):
# def compute_betweenness_centrality(G):

def create_histogram_for_degree_frequencies(frequencies , file_name="Empyt"):
    plt.xlabel('Degree')
    plt.ylabel("Frequencies", rotation=90)
    plt.plot(frequencies.keys(), frequencies.values(), '-bo')
    if(file_name != "Empyt"):
        plt.savefig(file_name, format='png', dpi=600)
    plt.close()

def H_features(time):
    nodes_feat_dict = {}
    df = read_graph_by_time_file()
    nodes = get_all_graph_nodes(df)
    H = build_h_graph_with_time(df, time)
    degree = compute_degree(H)

    demovalues = [np.random.randn() for _ in nodes]
    demivalues = [random.uniform(0, 1) for _ in nodes]
    clustering_coeff = [random.uniform(0, 1) for _ in nodes]
    closeness_centrality = [random.choice(demovalues) for _ in nodes]
    betweenness_centrality = [random.sample(demivalues, 1) for _ in nodes]

    for i, node in enumerate(nodes):
        nodes_feat_dict[node] = [degree[i], clustering_coeff[i], closeness_centrality[i], betweenness_centrality[i]]

    return nodes_feat_dict

def Q2():
    df = read_graph_by_time_file()
    time = df[ts_column_name].max;
    nodes_feat_dict = H_features(time)







###################################################
###################################################
################## Q4 #############################

def stc_index(time):
    df = read_graph_by_time_file()
    nodes = get_all_graph_nodes(df)
    H = build_h_graph_with_time(df, time)

    edgeDataView = H.edges(data=True)


    list_of_strong_edges = []
    for edge_data in edgeDataView:
        node1 = edge_data[0]
        node2 = edge_data[1]
        weight = edge_data[2]['weight']
        if weight == strong_edge:
            list_of_strong_edges.append( (node1,node2) )

    G = nx.Graph()
    G.add_nodes_from(list(nodes))
    G.add_edges_from(list_of_strong_edges)

    nodes_stc_dict = {}
    for node in nodes:
        neighbors = list(G.neighbors(node))
        if( len(neighbors) == 0):
            nodes_stc_dict[node] = 0
        else:
            number_edges_between_strong_neighbors_of_node = 0
            for i in range(0,len(neighbors)-1):
                for j in range(i+1,len(neighbors)):
                    n1 = neighbors[i]
                    n2 = neighbors[j]
                    if n1 != n2 and H.has_edge(n1,n2) :
                        number_edges_between_strong_neighbors_of_node = number_edges_between_strong_neighbors_of_node + 1
            if number_edges_between_strong_neighbors_of_node == 0 :
                nodes_stc_dict[node] = 0
            else:
                nodes_stc_dict[node] = float(number_edges_between_strong_neighbors_of_node) / float(nCk( len(neighbors) , 2))
    return nodes_stc_dict
def Q4():
    df = read_graph_by_time_file()
    time = df[ts_column_name].max;
    nodes_feat_dict = stc_index(time)



if __name__ == '__main__':
    #Q1()
    #Q2()
    Q4()
    # df = read_graph_by_time_file()
    # #g = get_graph_by_time_as_multiGraph(df)
    # g = build_h_graph(df,get_all_graph_nodes(df))
    # compute_distance_between_every_pair_of_nodes(g)