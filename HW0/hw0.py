import csv

Graph = {}

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

def change_weak_edge_to_strong(nodes_dic,Graph):
    is_change = False
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

                if x_a_edge_weight == strong_edge and x_b_edge_weight == strong_edge:
                    if Graph[a_b_edge] != strong_edge:
                        Graph[a_b_edge] = strong_edge
                        is_change = True
    return is_change


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

# get all nodes
nodes_dic = {}
for key in Graph:
    a = key[0]
    b = key[1]
    add_node_to_dic(nodes_dic,a,b)
    add_node_to_dic(nodes_dic,b,a)


# first : we put in each empty edge weak symbol
for key in Graph:
    edge_weight = Graph[key]
    if edge_weight == empyt_edge:
        Graph[key] = weak_edge

# second : change weak edge to strong until all Graph is OK
# our Graph goal definition : if there is a strong edge between A-B and A-C , we expect to be a strong edge between B-C
while( change_weak_edge_to_strong(nodes_dic,Graph) == True):
    print("change_weak_edge_to_strong")



########################################################################################
#######  output file ###################################################################        
with open('HW0_037036209_123456789.csv', 'w' ) as write_file:
    writer = csv.writer(write_file, lineterminator='\n')
    fieldnames2 = ["Node 1" , "Node 2" ,"W or S"]
    writer.writerow(fieldnames2)
    for arc in Graph:
        writer.writerow([  arc[0] , arc[1] , Graph[arc]  ])
write_file.close
#######################################################################################
#######################################################################################





