"""
HW1 part1
"""
import numpy as np
import pandas as pd
from datetime import datetime as dt
import collections
from collections import Counter
import math
import operator
import random
import networkx as nx
from collections import deque


# Necessary to run on the server, without display


t = 1082791719

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
    if 0 < k and k <= n :
        f = math.factorial
        return ( f(n) / (f(k) * f(n-k)))
    return 0

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

def get_h_sub_graph(H ,nodes , edge_weight ):
    edgeDataView = H.edges(data=True)
    list_of_strong_edges = []
    for edge_data in edgeDataView:
        node1 = edge_data[0]
        node2 = edge_data[1]
        weight = edge_data[2]['weight']
        if weight == edge_weight:
            list_of_strong_edges.append((node1, node2))
    G = nx.Graph()
    G.add_nodes_from(list(nodes))
    G.add_edges_from(list_of_strong_edges)
    return G

def compute_degree(G):
    degrees = {}
    for (node, val) in G.degree():
        degrees[node] = val
    return degrees

def compute_clustering_coefficiente(G):
    clustering_coefficiente = {}
    nodes = list(G.nodes())
    for x in nodes:
        clustering_coefficiente[x] = cc(x,G)
    return clustering_coefficiente

def cc(x, G):
    Ex = list(G.neighbors(x))
    num_of_neighbors = len(Ex)
    if num_of_neighbors < 2 :
        return 0
    up = 0
    for i in range(0,num_of_neighbors-1):
        for j in range(i+1, num_of_neighbors ):
            z = Ex[i]
            y = Ex[j]
            if G.has_edge(y,z):
                up = up + 1
    return float(up) / float(nCk(num_of_neighbors, 2))

def compute_closeness_centrality(G):
    closeness_centrality = {}
    nodes = nx.nodes(G)
    number_of_all_nodes = len(nodes)
    for node in nodes:
        sum = 0
        neighbors = nx.node_connected_component(G, node)
        for neighbor in neighbors:
            if neighbor != node:
                sum += nx.shortest_path_length(G, neighbor, node)
        n = len(neighbors)-1
        if n == 0:
            closeness_centrality[node] = 0
            continue
        closeness_centrality[node] = (n / (number_of_all_nodes-1)) * (n / sum)
    return closeness_centrality

def compute_betweenness_centrality(G):
    betweenness = {}
    nodes = list(G.nodes())
    num_of_nodes = len(nodes)

    for x in nodes:
        betweenness[x] = 0

    for i in range(0,num_of_nodes-1):
        for j in range(i+1, num_of_nodes ):
            s = nodes[i]
            t = nodes[j]
            try:
                shortest_paths = list(nx.all_shortest_paths(G, source=s, target=t))
            except nx.exception.NetworkXNoPath:
                continue
            number_of_shortest_paths = len(shortest_paths)
            if (number_of_shortest_paths == 0):
                continue
            for x in nodes:
                if ((s != x) and (t != x)):
                    number_of_shortest_paths_include_x = 0
                    for path in shortest_paths:
                        if len(path) < 3 :
                            continue
                        if x in path:
                            number_of_shortest_paths_include_x = number_of_shortest_paths_include_x + 1
                    betweenness[x] = betweenness[x] + number_of_shortest_paths_include_x / number_of_shortest_paths
        factor = 1.0 / float(nCk(num_of_nodes-1,2))
        for node in nodes:
            betweenness[node] = factor * betweenness[node]

    return betweenness

def compute_betweenness_centrality2(G):
    betweenness = {}
    nodes = list(G.nodes())
    for i in nodes:
        betweenness[i] = 0
    for node in nodes:
        s = []
        p = {}
        for i in nodes:
            p[i] = []
        shortest = {}
        for i in nodes:
            shortest[i] = 0
        shortest[node] = 1
        distance = {}
        for x in nodes:
            distance[x] = -1
        distance[node] = 0
        q = deque()
        q.append(node)
        while q:
            v = q.popleft()
            s.append(v)
            for neighbor in nx.all_neighbors(G, v):
                if distance[neighbor] < 0:
                    q.append(neighbor)
                    distance[neighbor] = distance[v] + 1

                if distance[neighbor] == (distance[v] + 1):
                    shortest[neighbor] = shortest[neighbor] + shortest[v]
                    p[neighbor].append(v)
        e = {}
        for i in nodes:
            e[i] = 0
        while s:
            w = s.pop()
            for y in p[w]:
                e[y] = e[y] + float(float(shortest[y] * (1 + e[w])) / shortest[w])
                if w != node:
                    betweenness[w] = (betweenness[w] + e[w])

    return betweenness


def get_neighborhood_overlap(G):
    edges_no_overlap = {}
    edgeDataView = G.edges(data=True)
    for edge_data in edgeDataView:
        x = edge_data[0]
        y = edge_data[1]
        Ex = set(G.neighbors(x))
        Ey = set(G.neighbors(y))
        if len(Ex) == 0 or len(Ey) == 0:
            continue
        union = Ex | Ey
        intersection = Ex & Ey
        no_xy = float(len(intersection)) / float(len(union))
        edges_no_overlap[(x,y)] = no_xy
    return edges_no_overlap

'''
Question 2 parts 2.a-2.c
'''

def H_features(time):
    nodes_feat_dict = {}
    df = read_graph_by_time_file()
    nodes = get_all_graph_nodes(df)
    H = build_h_graph_with_time(df, time)
    betweenness_centrality = compute_betweenness_centrality2(H)
    degree = compute_degree(H)
    clustering_coeff = compute_clustering_coefficiente(H)
    closeness_centrality = compute_closeness_centrality(H)

    for node in nodes:
        nodes_feat_dict[node] = [degree[node], clustering_coeff[node], closeness_centrality[node], betweenness_centrality[node]]
    return nodes_feat_dict

"""
Question 3
"""


def calc_no(time=t):
        df = read_graph_by_time_file()
        H = build_h_graph_with_time(df, time)
        edges_no_overlap = get_neighborhood_overlap(H)
        return edges_no_overlap

'''
Question 4
'''

def stc_index(time):
    df = read_graph_by_time_file()
    nodes = get_all_graph_nodes(df)
    H = build_h_graph_with_time(df, time)
    G = get_h_sub_graph(H, nodes, strong_edge)
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
                    if n1 != n2 and n1 != node and n2 != node and H.has_edge(n1,n2) :
                        number_edges_between_strong_neighbors_of_node = number_edges_between_strong_neighbors_of_node + 1
            if number_edges_between_strong_neighbors_of_node == 0 :
                nodes_stc_dict[node] = 0
            else:
                nodes_stc_dict[node] = float(number_edges_between_strong_neighbors_of_node) / float(nCk( len(neighbors) , 2))
    return nodes_stc_dict

######################################################
