#!/usr/local/bin/python
# -*- coding: utf-8 -*-

'''
we will be working with similar main, you should not change this file!
all your imports/constants/classes/func should be held in hw2_students.py
'''

from hw2_part1 import *



##### knn ########

def get_list_of_posible_z(md , k):
    z_list = []
    for index,row in md.iterrows():
        z = row[satoshi_per_byte_colunm_names]
        neighbors = getNeighbors(z, md, k)
        md_z = md[(md[TXID_colunm_names].isin(neighbors))]
        z_list.append(md_z[satoshi_per_byte_colunm_names].mean())

    z_list.extend(range(0,5000,10))

    return list(set(z_list))

def l2_norm_similarty(z1, z2):
    return pow((z1 - z2), 2)

def getNeighbors(z,md, k):
    l2_norm_colunm_names = 'l2_norm'
    md[l2_norm_colunm_names] = l2_norm_similarty(md[satoshi_per_byte_colunm_names],z) # (md[fee_colunm_names] - z) ** 2
    md_z = md.sort_values(l2_norm_colunm_names, ascending=[True])
    md_z = md_z.head(k)
    neighbors = list(md_z[TXID_colunm_names])
    return neighbors

##########################################################################




def find_k_similar_to_z(z , k , md):
    neighbors = getNeighbors(z,md,k)
    return list(neighbors)


def agent_gains(z , k , mempool_data , v , r , size):
    md = mempool_data.copy()
    tx_list = find_k_similar_to_z(z, k, md)
    md = md[(md[TXID_colunm_names].isin(tx_list))]
    w_colunm_names = 'w'
    md[w_colunm_names] = v * pow(2 , -1*md[diff_time_colunm_names] * (r / 1000.0) )
    diff_w_colunm_names = 'diff w'
    md[diff_w_colunm_names] = md[w_colunm_names] - z * size
    sum = md[diff_w_colunm_names].sum()
    gains = (1.0 / k) * sum
    return gains


def get_z_to_tx(v,r,size , md , z_list):

    maxZ = 0
    min_gains_value = float("inf")
    for z in z_list:
        gains = agent_gains(z, k, md, v, r, size)
        if gains < min_gains_value:
            min_gains_value = gains
            maxZ = z
    return maxZ


def set_z_to_file():
    block_size = 100000
    mempool_data_name = 'bitcoin_mempool_data.json'
    # load the data once for all func of the mandatory part , at your choosing (class, dict, pandas, etc..)
    mempool_data_full_path = os.path.abspath(mempool_data_name)
    mempool_data = load_mempool_data(mempool_data_full_path)

    hw2_part2_file_name = 'hw2_part2.csv'
    hw2_part2_full_path = os.path.abspath(hw2_part2_file_name)
    df = pd.read_csv(hw2_part2_full_path)

    k = 10
    z_list = get_list_of_posible_z(mempool_data , k)

    for index, row in df.iterrows():
        v = row['v']
        r = row['r']
        size = row['size']
        z = get_z_to_tx(v, r, size, mempool_data,z_list)
        df.loc[index, 'z'] = z



if __name__ == "__main__":
    set_z_to_file()


