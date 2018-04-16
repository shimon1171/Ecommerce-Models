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



###################################################
###################################################
################## Q1 #############################
###################################################
###################################################

def compute_distance_between_every_pair_of_nodes(G):
    spl = nx.all_pairs_shortest_path_length(G)
    return spl

def create_histogram_as_fig_2_10(frequencies , file_name="Empyt"):
    plt.xlabel('Number of intermediaries')
    plt.ylabel("Number of paris", rotation=90)
    plt.plot(frequencies.keys(), frequencies.values(), '-bo')
    plt.subplots_adjust(top=0.8)
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


###################################################
###################################################
################## Q2 #############################
###################################################
###################################################

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
    nodes = G.nodes()
    number_of_nodes = len(nodes)
    for node in nodes:
        sum = 0
        for node2 in nodes:
            if node2 != node:
                try:
                    sum+=nx.shortest_path_length(G,node2,node)
                except nx.exception.NetworkXNoPath:
                    continue
        if sum != 0:
            closeness_centrality[node] = (number_of_nodes-1) / float(sum)
        else:
            closeness_centrality[node] = 0
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
    betweenness_centrality = compute_betweenness_centrality(H)
    degree = compute_degree(H)
    clustering_coeff = compute_clustering_coefficiente(H)
    closeness_centrality = compute_closeness_centrality(H)


    for i, node in enumerate(nodes):
        nodes_feat_dict[node] = [degree[i], clustering_coeff[i], closeness_centrality[i], betweenness_centrality[i]]

    return nodes_feat_dict

def Q2():
    df = read_graph_by_time_file()
    time = df[ts_column_name].max;
    nodes_feat_dict = H_features(time)


###################################################
###################################################
################## Q3 #############################
###################################################
###################################################

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

def get_neighborhood_overlap_list_with_weight(G, edge_weight):
    list_of_no = []
    edgeDataView = G.edges(data=True)
    for edge_data in edgeDataView:
        x = edge_data[0]
        y = edge_data[1]
        weight = edge_data[2]['weight']
        if weight == edge_weight:
            Ex = set(G.neighbors(x))
            Ey = set(G.neighbors(y))
            if len(Ex) == 0 or len(Ey) == 0:
                continue
            union = Ex | Ey
            intersection = Ex & Ey
            no_xy = float(len(intersection)) / float(len(union))
            list_of_no.append(no_xy)
    return Counter(list_of_no)

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

def create_histogram_for_neighborhood_overlap_frequencies(frequencies , file_name="Empyt"):
    plt.xlabel('neighborhood_overlap')
    plt.ylabel("Frequencies", rotation=90)
    plt.plot(frequencies.keys(), frequencies.values(), '-bo')
    if(file_name != "Empyt"):
        plt.savefig(file_name, format='png', dpi=600)
    plt.close()

def calc_no(time):
    df = read_graph_by_time_file()
    H = build_h_graph_with_time(df, time)
    edges_no_overlap = get_neighborhood_overlap(H)
    return edges_no_overlap

def Q3():
    df = read_graph_by_time_file()
    nodes = get_all_graph_nodes(df)
    time = df[ts_column_name].max
    H = build_h_graph_with_time(df, time)
    frequencies = get_neighborhood_overlap_list_with_weight(H,strong_edge)
    create_histogram_for_neighborhood_overlap_frequencies(frequencies , 'q3a_strong.png')
    frequencies = get_neighborhood_overlap_list_with_weight(H,weak_edge)
    create_histogram_for_neighborhood_overlap_frequencies(frequencies, 'q3a_weak.png')

###################################################
###################################################
################## Q4 #############################
###################################################
###################################################


def stc_index(time):
    df = read_graph_by_time_file()
    nodes = get_all_graph_nodes(df)
    H = build_h_graph_with_time(df, time)
    G = get_h_sub_graph(H, nodes ,strong_edge)
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

def Q4():
    df = read_graph_by_time_file()
    time = df[ts_column_name].max;
    nodes_feat_dict = stc_index(time)

###################################################
###################################################
################## Competitive part ##############
###################################################
###################################################

class ShortestPathsHelper:
    def __init__(self, G , nodes):
        num_of_nodes = len(nodes)
        self.tuple_index = {}
        index = 0
        for i in range(0,num_of_nodes):
            for j in range(0, num_of_nodes ):
                s = nodes[i]
                t = nodes[j]
                if s != t:
                    if (t,s) in self.tuple_index:
                        self.tuple_index[(s,t)] =  self.tuple_index[(t,s)]
                    else:
                        self.tuple_index[(s, t)] = index
                        index = index + 1
        self.shortest_paths_dic = {}
        self.number_of_shortest_paths_dic = {}
        for i in range(0,num_of_nodes-1):
            for j in range(i+1, num_of_nodes ):
                s = nodes[i]
                t = nodes[j]
                try:
                    shortest_paths = list(nx.all_shortest_paths(G, source=s, target=t))
                except nx.exception.NetworkXNoPath:
                    continue
                number_of_shortest_paths = len(shortest_paths)
                ind1 = self.tuple_index[(s,t)]
                ind2 = self.tuple_index[(t,s)]
                self.number_of_shortest_paths_dic[ind1] = number_of_shortest_paths
                self.number_of_shortest_paths_dic[ind2] = number_of_shortest_paths
                if (number_of_shortest_paths == 0):
                    continue
                self.shortest_paths_dic[ ind1 ] = shortest_paths
                self.shortest_paths_dic[ ind2 ] = shortest_paths

    def get_number_of_shortest_paths_include_x(self,s,t,x):
        if (s, t) not in self.tuple_index:
            return 0
        if ((s == x) or (t == x)):
            return 0
        ind = self.tuple_index[(s, t)]
        if ind not in self.shortest_paths_dic:
            return 0
        shortest_paths = self.shortest_paths_dic[ind]
        number_of_shortest_paths_include_x = 0
        for path in shortest_paths:
            if len(path) < 3:
                continue
            if x in path:
                number_of_shortest_paths_include_x = number_of_shortest_paths_include_x + 1
        return number_of_shortest_paths_include_x

    def get_number_of_shortest_paths(self, s,t):
        if (s,t) not in self.tuple_index:
            return 0
        ind = self.tuple_index[(s, t)]
        return self.number_of_shortest_paths_dic[ind]

    def get_shortest_path_length(self,s,t):
        if (s,t) not in self.tuple_index:
            return 0
        ind = self.tuple_index[(s, t)]
        if ind not in self.shortest_paths_dic:
            return 0
        shortest_paths = self.shortest_paths_dic[ind]
        return len(shortest_paths[0])

class NodeProbabilityClass:
    def __init__(self, G ,H):
        self.nodes = list(G.nodes())

        #sph = ShortestPathsHelper(H,self.nodes)

        degrees_for_orig = compute_degree(G)
        degrees_for_h = compute_degree(H)
        clustering = compute_clustering_coefficiente(H)
        closeness = compute_closeness_centrality(H)
        #betweenness = compute_betweenness_centrality_(H,sph)

        self.degrees_for_orig = normalizing_dictionary_values(degrees_for_orig)
        self.degrees_for_h = normalizing_dictionary_values(degrees_for_h)
        self.clustering = normalizing_dictionary_values(clustering)
        self.closeness = normalizing_dictionary_values(closeness)
       # self.betweenness = normalizing_dictionary_values(betweenness)


    def get_probability_for_each_node(self,degrees_for_orig_cof,degrees_for_h_cof,clustering_cof,closeness_cof,betweenness_cof):
        probability = {}
        for x in self.nodes:
            probability[x] = degrees_for_orig_cof * self.degrees_for_orig[x] + \
                             degrees_for_h_cof * self.degrees_for_orig[x] +\
                             clustering_cof * self.clustering[x] + \
                              closeness_cof * self.closeness[x]
                             # betweenness_cof * self.betweenness[x]

        return normalizing_dictionary_values(probability)

def get_probability_for_each_neighbors_node(G , node_probability , factor_neighbor_probability):
    probabilities = {}
    nodes = list(G.nodes())
    for x in nodes:
        n = list([c[1] for c in list(G.edges(x))])
        frequencies = Counter(n)
        frequencies = normalizing_dictionary_values(frequencies)
        for neighbor in frequencies.keys():
            neighbor_general_weight = node_probability[neighbor]
            neighbor_probability = frequencies[neighbor]
            frequencies[neighbor] = (1-factor_neighbor_probability) * neighbor_general_weight + factor_neighbor_probability * neighbor_probability
        probabilities[x] = normalizing_dictionary_values(frequencies)
    return probabilities

def compute_closeness_centrality_(G , SPH):
    closeness_centrality = {}
    nodes = G.nodes()
    number_of_nodes = len(nodes)
    for node in nodes:
        sum = 0
        for node2 in nodes:
            if node2 != node:
                sum += SPH.get_shortest_path_length(node2, node)
        if sum != 0:
            closeness_centrality[node] = (number_of_nodes-1) / float(sum)
        else:
            closeness_centrality[node] = 0
    return closeness_centrality

def compute_betweenness_centrality_(G , SPH):
    betweenness = {}
    nodes = list(G.nodes())
    num_of_nodes = len(nodes)

    for x in nodes:
        betweenness[x] = 0

    for i in range(0,num_of_nodes-1):
        for j in range(i+1, num_of_nodes ):
            s = nodes[i]
            t = nodes[j]
            number_of_shortest_paths = SPH.get_number_of_shortest_paths(s,t)
            if (number_of_shortest_paths == 0):
                continue
            for x in nodes:
                number_of_shortest_paths_include_x = SPH.get_number_of_shortest_paths_include_x(s,t,x)
                betweenness[x] = betweenness[x] + number_of_shortest_paths_include_x / number_of_shortest_paths

        factor = 1.0 / float(nCk(num_of_nodes-1,2))
        for node in nodes:
            betweenness[node] = factor * betweenness[node]

    return betweenness


def CompetitivePart():
    df = read_graph_by_time_file()
    cross_validation(df)
    # nodes = get_all_graph_nodes(df)
    # G = get_graph_by_time_as_multiGraph(df)
    # H = build_h_graph(df,nodes)


def predict_future_links(k):
    df = read_graph_by_time_file()
    nodes = get_all_graph_nodes(df)
    G = get_graph_by_time_as_multiGraph(df)
    H = build_h_graph(df, nodes)
    nodeProbabilityClass = NodeProbabilityClass(G, H)

    factor_neighbor_probability = 0.9
    degrees_for_orig_cof = 0.6
    degrees_for_h_cof = 0.1
    clustering_cof = 0.1
    closeness_cof = 0.1
    betweenness_cof = 0.1

    node_probability = nodeProbabilityClass.get_probability_for_each_node(degrees_for_orig_cof, degrees_for_h_cof,
                                                                      clustering_cof, closeness_cof, betweenness_cof)
    neighbors_node_probability = get_probability_for_each_neighbors_node(G, node_probability,
                                                                         factor_neighbor_probability)

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


from sklearn.model_selection import KFold
def cross_validation(df):
    n = len(df)
    k = int(n / (n * 0.1))

    vlaues = [0 , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    NodeProbability = {}
    for factor_neighbor_probability in [0.9,0.8]:
        for degrees_for_orig_cof in vlaues:
            for degrees_for_h_cof in vlaues:
                for clustering_cof in vlaues:
                    for closeness_cof in vlaues:
                        for betweenness_cof in [0.0]:
                            if ((degrees_for_orig_cof + degrees_for_h_cof + clustering_cof + closeness_cof + betweenness_cof) == 1 ):
                                NodeProbability[(factor_neighbor_probability ,degrees_for_orig_cof , degrees_for_h_cof , clustering_cof , closeness_cof , betweenness_cof)] = 0

    kf = KFold(n_splits=k, shuffle=True, random_state=2)
    for train_index, test_index in kf.split(df):
        train = df.iloc[train_index]
        test = df.iloc[test_index]

        nodes = get_all_graph_nodes(df)
        G = get_graph_by_time_as_multiGraph(train)
        H = build_h_graph(train, nodes)
        nodeProbabilityClass = NodeProbabilityClass(G,H)

        test_node = get_nodes(test)
        test_node_size = len(test_node)

        prop = NodeProbability.keys()
        for coefs in prop:
            factor_neighbor_probability = coefs[0]
            degrees_for_orig_cof = coefs[1]
            degrees_for_h_cof = coefs[2]
            clustering_cof = coefs[3]
            closeness_cof = coefs[4]
            betweenness_cof = coefs[5]

            node_probability = nodeProbabilityClass.get_probability_for_each_node(degrees_for_orig_cof,degrees_for_h_cof,clustering_cof,closeness_cof,betweenness_cof)
            neighbors_node_probability = get_probability_for_each_neighbors_node(G, node_probability, factor_neighbor_probability)

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

            p = 1.0 * c / test_node_size
            print("For lamda {} recive {}".format(coefs, p))
            if NodeProbability[coefs] == 0:
                NodeProbability[coefs] = p
            else:
                NodeProbability[coefs] = (NodeProbability[coefs] + p) / 2

    print("final result : ")
    max = -1

    for key in NodeProbability.keys():
        print("For lamda {} recive {}".format(key, NodeProbability[key]))
        if max == -1 :
            max = NodeProbability[key]
            maxkey = key
        elif max < NodeProbability[key]:
            max = NodeProbability[key]
            maxkey = key


    print("Max lamda {} recive {}".format(maxkey, max))



def get_nodes(df):
    tupels = []
    for index, row in df.iterrows():
        user1 = row[user1_column_name]
        user2 = row[user2_column_name]
        tupels.append((user1, user2))
    return tupels

def random_distr(l):
    r = random.uniform(0, 1)
    s = 0
    for item in l.keys():
        prob = l[item]
        s += prob
        if s >= r:
            return item
    return item  # Might occur because of floating point inaccuracies


if __name__ == '__main__':
    #Q1()
    #Q2()
    #Q3()
    #Q4()
    CompetitivePart()
    # df = read_graph_by_time_file()
    # #g = get_graph_by_time_as_multiGraph(df)
    # g = build_h_graph(df,get_all_graph_nodes(df))
    # compute_distance_between_every_pair_of_nodes(g)