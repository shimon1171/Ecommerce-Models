import pandas as pd
from datetime import datetime as dt
import collections
from matplotlib import pyplot as plt

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
    df[ts_column_name] = df[ts_column_name].apply(lambda x: dt.fromtimestamp(x))
    df = df.sort_values(by=ts_column_name)
    return df


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


if __name__ == '__main__':
    #df = read_graph_by_time_file()
    #h_df, graph = build_h_graph(df)
    #h_df, graph = read_h_graph_from_file(h_graph_file_name)
    #nods_with_distance_dic = compute_distance_between_every_pair_of_nodes(graph)
    read_distance_between_every_pair_of_nodes()

