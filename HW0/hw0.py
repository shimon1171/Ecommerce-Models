import csv

Graph = {}






########################################################################################
###################  csv files to dictionaries #########################################
with open("graph.csv","r") as csvfile:
    reader = csv.DictReader(csvfile)
    # note : field_names = ['Node 1', 'Node 2','W or S' ]
    for row in reader:
        Graph[(row['Node 1'],row['Node 2'])] = row['W or S']
csvfile.close()


#######################################################################################
############## Insert Your Answers ####################################################

strong_edge = "s"
weak_edge = "w"
empyt_edge = ""
not_exsist_edge = "null"

def add_node_to_dic(dic, key_node , value_node):
    if key_node not in nodes_dic:
        dic[key_node] = [value_node]
    else:
        dic[key_node].append(value_node)

def get_edge_weight(dic, a, b):
    edge_weight = not_exsist_edge
    edge_a_b = (a,b)
    edge_b_a = (b,a)
    edge = ()
    if edge_a_b in dic:
        edge_weight = dic[edge_a_b]
        edge = edge_a_b
    elif edge_b_a in dic:
        edge_weight = dic[edge_b_a]
        edge = edge_b_a
    return edge_weight , edge

def build_empyt_edges_weight_lists(empyt_edges_weight, nodes_dic,Graph):
    for x in nodes_dic:
        list_of_connected_nodes = nodes_dic[x]
        number_of_connected_nodes = len(list_of_connected_nodes)
        if (number_of_connected_nodes == 1):
            continue
        for i in range(0, number_of_connected_nodes - 2):
            a = list_of_connected_nodes[i]
            for j in range(1, number_of_connected_nodes - 1):
                b = list_of_connected_nodes[j]
                a_b_edge_weight, a_b_edge = get_edge_weight(Graph, a, b)

                if a_b_edge_weight == not_exsist_edge:
                    continue
                x_a_edge_weight, x_a_edge = get_edge_weight(Graph, x, a)
                x_b_edge_weight, x_b_edge = get_edge_weight(Graph, x, b)

                if a_b_edge_weight == empyt_edge:
                    if x_a_edge_weight == strong_edge and x_b_edge_weight == strong_edge:
                        empyt_edges_weight[a_b_edge].append(weak_edge)

                if a_b_edge_weight == strong_edge:
                    if x_a_edge_weight == strong_edge and x_b_edge_weight == empyt_edge:
                        empyt_edges_weight[x_b_edge].append(strong_edge)
                    elif x_a_edge_weight == empyt_edge and x_b_edge_weight == strong_edge:
                        empyt_edges_weight[x_a_edge].append(strong_edge)


# for ech node we bulid list of all connected nodes
nodes_dic = {}
for key in Graph:
    a = key[0]
    b = key[1]
    add_node_to_dic(nodes_dic,a,b)
    add_node_to_dic(nodes_dic,b,a)

# for eche empty edge we bulid list of all optinal weight option
empyt_edges_weight = {}
for nodes_tuple in Graph:
    edge_weight = Graph[nodes_tuple]
    if edge_weight == empyt_edge:
        empyt_edges_weight[nodes_tuple] = []

build_empyt_edges_weight_lists(empyt_edges_weight, nodes_dic,Graph)

# we choose for eche empty edge tha maximum value of weight that it can get : max is strong.
for nodes_tuple in empyt_edges_weight:
    weights = empyt_edges_weight[nodes_tuple]
    if strong_edge in weights:
        Graph[nodes_tuple] = strong_edge
    else:
        Graph[nodes_tuple] = weak_edge



#######################################################################################
###################### End Answers ####################################################




########################################################################################
#######  output file ###################################################################        
with open('HW0_037036209_XXXXXXXXX.csv', 'w' ) as write_file:
    writer = csv.writer(write_file, lineterminator='\n')
    fieldnames2 = ["Node 1" , "Node 2" ,"W or S"]
    writer.writerow(fieldnames2)
    for arc in Graph:
        writer.writerow([  arc[0] , arc[1] , Graph[arc]  ])
write_file.close
#######################################################################################
#######################################################################################





