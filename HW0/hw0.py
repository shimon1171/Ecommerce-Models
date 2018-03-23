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

# for example:
Graph[('A','B')] = 's'





########################################################################################
#######  output file ###################################################################        
with open('HW0_200202141_123456789.csv', 'w' ) as write_file:
    writer = csv.writer(write_file, lineterminator='\n')
    fieldnames2 = ["Node 1" , "Node 2" ,"W or S"]
    writer.writerow(fieldnames2)
    for arc in Graph:
        writer.writerow([  arc[0] , arc[1] , Graph[arc]  ])
write_file.close
#######################################################################################
#######################################################################################

