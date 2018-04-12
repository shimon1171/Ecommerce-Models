import pandas as pd
from datetime import datetime as dt
import collections
from matplotlib import pyplot as plt
from collections import Counter
import math
import operator
import random
from sklearn.model_selection import KFold
# <editor-fold desc="Common">

graph_by_time_file_name = 'graph_by_time.txt'
user1_column_name = 'user1'
user2_column_name = 'user2'
ts_column_name = 'ts'

def read_graph_by_time_file():
    with open(graph_by_time_file_name) as f:
        df = pd.read_table(f, sep=' ', index_col=False, header=0, lineterminator='\n')
    df[ts_column_name] = df[ts_column_name].apply(lambda x: dt.fromtimestamp(x))
    df = df.sort_values(by=ts_column_name)
    return df


def get_for_each_node_is_neighbors(df):
    graph = {}
    df_group_by_user1 = df.groupby(by=user1_column_name)
    for user1, user1_group in df_group_by_user1:
        graph[user1] = []
        for index, row in user1_group.iterrows():
            user2 = row[user2_column_name]
            graph[user1].append(user2)
    return graph

def get_all_graph_nodes(df):
    all_nodes_set = set(df[user1_column_name]) | set(df[user2_column_name])
    return all_nodes_set

def get_graph_as_matrix(df):
    graph = get_for_each_node_is_neighbors(df)
    all_nodes_set = get_all_graph_nodes(df)

    n = len(all_nodes_set)
    matrix = [[0 for x in range(n)] for y in range(n)]

    node_to_index_mapping = {}
    index = 0
    for node in all_nodes_set:
        node_to_index_mapping[node] = index
        index = index + 1

    for node in graph.keys():
        node_index = node_to_index_mapping[node]
        neighbors = graph[node]
        for neighbor in neighbors:
            neighbor_index = node_to_index_mapping[neighbor]
            matrix[node_index][neighbor_index] = 1

    return matrix,node_to_index_mapping





def normalizing_dictionary_values(d):
    factor = 1.0 / sum(d.itervalues())
    for k in d:
        d[k] = d[k] * factor
    key_for_max = max(d.iteritems(), key=operator.itemgetter(1))[0]
    diff = 1.0 - math.fsum(d.itervalues())
    d[key_for_max] += diff
    return d

def random_distr(l):
    r = random.uniform(0, 1)
    s = 0
    for item in l.keys():
        prob = l[item]
        s += prob
        if s >= r:
            return item
    return item  # Might occur because of floating point inaccuracies
# </editor-fold>


# <editor-fold desc="Question 1">


def bfs(graph, start):
    visited = set()
    next = set()
    open =  set([start])
    distance = 0
    nods_with_distance = {}
    nods_with_distance[start] = 0
    visited.add(start)
    while len(open) != 0:
        distance = distance + 1
        for vertex in open:
            neighbors = graph[vertex]
            for neighbor in neighbors:
                if neighbor not in visited | open | next :
                    next = next | set([neighbor])
                    nods_with_distance[neighbor] = distance
        visited = visited | open
        open = next
        next = set()
    return nods_with_distance



def compute_distance_between_every_pair_of_nodes2(graph):

    distance_histogram = {} # key is distance , value is pairs of nods

    df = pd.DataFrame(columns=[h_graph_user1_column_name, h_graph_user2_column_name, h_graph_weight_column_name])
    df_index = 0

    for vertex in graph.keys():
        nods_with_distance_dic = bfs(graph, vertex)
        for node in nods_with_distance_dic.keys():
            distance = nods_with_distance_dic[node]
            if distance != 0:
                if distance not in distance_histogram:
                    distance_histogram[distance] = 1
                else:
                    distance_histogram[distance] = distance_histogram[distance] + 1

                data = [vertex, node, distance]
                df.loc[df_index] = data
                df_index = df_index + 1

    df.to_csv('distance_between_every_pair_of_nodes.csv')

    return distance_histogram

def compute_distance_between_every_pair_of_nodes(graph):

    distance_histogram = {} # key is distance , value is pairs of nods

    df = pd.DataFrame(columns=['distance','count'])
    df_index = 0

    for vertex in graph.keys():
        nods_with_distance_dic = bfs(graph, vertex)
        for node in nods_with_distance_dic.keys():
            distance = nods_with_distance_dic[node]
            if distance != 0:
                if distance not in distance_histogram:
                    distance_histogram[distance] = 1
                else:
                    distance_histogram[distance] = distance_histogram[distance] + 1


    for distance in distance_histogram.keys():
        data = [distance, distance_histogram[distance]]
        df.loc[df_index] = data
        df_index = df_index + 1

    df.to_csv('distance_between_every_pair_of_nodes.csv')
    return distance_histogram



def read_distance_between_every_pair_of_nodes():
    df = pd.read_csv('distance_between_every_pair_of_nodes.csv')
    bins_names =[]
    bins_values=[]
    for index, row in df.iterrows():
        bins_names.append(row['distance'])
        bins_values.append(row['count'])

    _sum = sum(bins_values)
    norm = [float(i) / _sum for i in bins_values]

    PlotSingleColumnHist(bins_names, norm,'distance_between_every_pair_of_nodes.png',y_axis_name='Number Of Paris', x_axis_name='Number Of Intermediaries')



def PlotSingleColumnHist(bins_names, bins_values, file_name, y_axis_name=None, x_axis_name=None, title_name=None):
    histogram = plt.figure()

    if (title_name != None):
        plt.title(title_name)
    if (x_axis_name != None):
        plt.xlabel(x_axis_name)
    if (y_axis_name != None):
        plt.ylabel(y_axis_name, rotation=90)

    bar_width = 0.4
    bins = range(0, len(bins_names))
    bar1 = plt.bar(bins, bins_values,
                   width=bar_width,
                   color='r',
                   align='center',
                   label="label",
                   tick_label=bins_names,
                   edgecolor='black',
                   linewidth=1.2)
    # plt.xticks(np.arange(len(bins)), (bins_names), rotation=0)

    plt.hlines(0, 0, len(bins_names))
    plt.savefig(file_name, format='eps', dpi=600)
    plt.close()



# </editor-fold>

# <editor-fold desc="Question 2">

h_graph_file_name = 'h_graph.csv'
h_graph_user1_column_name = 'user1'
h_graph_user2_column_name = 'user2'
h_graph_weight_column_name = 'weight'

strong_edge = "s"
weak_edge = "w"

def build_h_graph(df):
    graph = {}
    for index, row in df.iterrows():
        user1 = row[user1_column_name]
        user2 = row[user2_column_name]
        user1_to_user2 = (user1, user2)
        user2_to_user1 = (user2, user1)
        if user1_to_user2 not in graph and user2_to_user1 not in graph:
            graph[user1_to_user2] = weak_edge
        elif user1_to_user2 in graph or user2_to_user1 in graph:
            if user1_to_user2 in graph:
                graph[user1_to_user2] = strong_edge
            else:
                graph[user2_to_user1] = strong_edge

    h_df = pd.DataFrame(columns=[h_graph_user1_column_name, h_graph_user2_column_name, h_graph_weight_column_name])
    h_df_index = 0
    for key in graph.keys():
        user1 = key[0]
        user2 = key[1]
        weight = graph[key]
        data = [user1, user2, weight]
        h_df.loc[h_df_index] = data
        h_df_index = h_df_index + 1

    return h_df, graph

def build_h_graph_with_time(df , time, time_format_is_unit = True ):
    if time_format_is_unit:
        time = dt.fromtimestamp(time)
    filtered_df = df[ (df[ts_column_name] < time)] # get all samples that are lower that the given time
    return build_h_graph(filtered_df)

def add_node_to_dic(dic, key_node , value_node):
    if key_node not in dic:
        dic[key_node] = [value_node]
    else:
        dic[key_node].append(value_node)

def read_h_graph_from_file(file_name):
    df = pd.read_csv(file_name)
    graph = {}
    for index, row in df.iterrows():
        user1 = row[user1_column_name]
        user2 = row[user2_column_name]
        add_node_to_dic(graph,user1,user2)
        add_node_to_dic(graph,user2,user1)
    return df, graph


#  Compute : Degree, Clustering coefficient, Closeness centrality, Betweenness centrality

def get_degree_for_all_nods(df):
    degrees = {}

    all_nodes_set = set(df[user1_column_name]) | set(df[user2_column_name])
    for node in all_nodes_set:
        degrees[node] = 0

    df_group_by_user1 = df.groupby(by=user1_column_name)
    for node, user1_group in df_group_by_user1:
        degrees[node] = len(user1_group)

    return degrees

 # a = Counter(degrees.values())
 #    PlotSingleColumnHist(a.keys(), a.values(), 'degree_for_all_nods.png', y_axis_name='frequencie',
 #                         x_axis_name='degree of nods')

# betweness
def get_betweenness_for_all_nods(df):
    betweenness = {}
    all_nodes_set = set(df[user1_column_name]) | set(df[user2_column_name])
    for node in all_nodes_set:
        betweenness[node] = 0

    graph = get_for_each_node_is_neighbors(df)
    for s in all_nodes_set:
        for t in all_nodes_set:
            if s == t:
                continue
            shortest_paths = find_all_shortest_paths(graph,s,t)
            number_of_shortest_paths = len(shortest_paths)
            if(number_of_shortest_paths == 0):
                continue
            for x in all_nodes_set:
                if((s != x) and (t != x)):
                    number_of_shortest_paths_include_x = 0
                    for path in shortest_paths:
                        if x in path:
                            number_of_shortest_paths_include_x = number_of_shortest_paths_include_x + 1
                    betweenness[x] = betweenness[x] + number_of_shortest_paths_include_x / number_of_shortest_paths
    n = len(all_nodes_set)
    factor = 1 / ((n-1)*(n-2) / 2)
    for node in all_nodes_set:
        betweenness[node] = factor *  betweenness[node]







def find_all_shortest_paths(graph, start, end, path=[]):
    paths = find_all_paths(graph, start, end)
    number_of_paths = len(paths)
    if(number_of_paths == 0):
        return []
    min = len(paths[0])
    for path in paths:
        temp_len = len(paths)
        if temp_len < min:
            min = temp_len

    shortest_paths = []
    for path in paths:
        temp_len = len(paths)
        if temp_len == min :
            shortest_paths.append(path)

    return shortest_paths



def find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if not graph.has_key(start):
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths

# </editor-fold>


# <editor-fold desc="Question 3">

# </editor-fold>

# <editor-fold desc="Question 4">

# </editor-fold>

# <editor-fold desc="Competitive part">

def get_probability_for_each_node(df):
    probabilities = {}

    degrees = get_degree_for_all_nods(df)
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
    k = int(n / (n * 0.1))

    vlaues = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    vlaues_dic = {}
    for i in vlaues:
        vlaues_dic[i] = 0

    kf = KFold(n_splits=k, shuffle=True, random_state=2)
    for train_index, test_index in kf.split(df):
        train = df.iloc[train_index]
        test = df.iloc[test_index]

        test_node = get_nodes(test)
        test_node_size = len(test_node)

        for i in vlaues:
            node_probability = get_probability_for_each_node(train)
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

    for key in vlaues_dic.keys():
        print("For lamda {} recive {}".format(key, vlaues_dic[key]))






def get_nodes(df):
    tupels = []
    for index, row in df.iterrows():
        user1 = row[user1_column_name]
        user2 = row[user2_column_name]
        tupels.append((user1, user2))
    return tupels


# </editor-fold>










if __name__ == '__main__':
    #future_links = predict_future_links(100)
    #save_predict_future_links_in_file(future_links)

    df = read_graph_by_time_file()
    #h_df, graph = build_h_graph(df)
    #h_df, graph = read_h_graph_from_file(h_graph_file_name)
    #nods_with_distance_dic = compute_distance_between_every_pair_of_nodes(graph)
    #read_distance_between_every_pair_of_nodes()
    #get_degree_for_all_nods(df)
    #get_graph_as_matrix(df)
    #cross_validation(df)
    get_betweenness_for_all_nods(df)