

import json
import pandas as pd
import os

# const
TXID_colunm_names = 'TXID'
fee_colunm_names = 'fee'
output_colunm_names = 'output'
removed_colunm_names = 'removed'
size_colunm_names = 'size'
time_colunm_names = 'time'
diff_time_colunm_names = 'diff_time'
satoshi_per_byte_colunm_names = 'satoshi_per_byte'
NEVER = -1

def convert_bitcoin_mempool_data_json_to_csv(file_name):
    #file_name = 'bitcoin_mempool_data.json'
    listOfDic = []
    with open(file_name, "r") as file:
        for cnt, line in enumerate(file):
            data = json.loads(line)
            for txid in data.keys():
                dic = data[txid]
                dic[TXID_colunm_names] = txid
                listOfDic.append(dic)
    df = pd.DataFrame(listOfDic)
    df = df[ (df[output_colunm_names] != -1) &  (df[removed_colunm_names] != -1) ]

    df[diff_time_colunm_names] = df[removed_colunm_names] - df[time_colunm_names]
    df[satoshi_per_byte_colunm_names] = df[fee_colunm_names] / df[size_colunm_names]

    return df
    #df.to_csv('bitcoin_mempool_data.csv', index=False)




############################
# Part 1
############################

# return a list of the tx id's to insert into a block
def load_mempool_data(mempool_data_full_path, current_time=1510264253.0):
    df = convert_bitcoin_mempool_data_json_to_csv(mempool_data_full_path)
    #mempool_data = df[ (df[output_colunm_names] == -1) & (df[time_colunm_names] < current_time) & (current_time < df[removed_colunm_names]) ]
    mempool_data = df[(df[time_colunm_names] < current_time) & (current_time < df[removed_colunm_names])]
    return mempool_data

# return a list of the tx id's to insert into a block
def greedy_knapsack(block_size, mempool_data):

    md = mempool_data.sort_values([satoshi_per_byte_colunm_names, TXID_colunm_names], ascending=[False,  True])
    tx_ids = []
    for index,row in md.iterrows():
        size = row[size_colunm_names]
        if size < block_size:
            tx_ids.append(row[TXID_colunm_names])
            block_size = block_size - size
    return tx_ids

 # sum_colunm_names = 'sum'
 #    md[sum_colunm_names] = md[size_colunm_names].cumsum()
 #    md = md[ (md[sum_colunm_names] < block_size)]
 #    tx_ids = list(md[TXID_colunm_names])

def evaluate_block(tx_list, mempool_data):
    md = mempool_data[(mempool_data[TXID_colunm_names].isin(tx_list))]
    revenue = md[fee_colunm_names].sum()
    return revenue


def Vs(block_size , mempool_data,txi_id):
    md = mempool_data[mempool_data[TXID_colunm_names] != txi_id]
    tx_list = greedy_knapsack(block_size, md)
    revenue = evaluate_block(tx_list, md)
    return revenue

def Vs_j(block_size , mempool_data, txi_id):
    md = mempool_data[mempool_data[TXID_colunm_names] == txi_id]
    size = md[size_colunm_names].iloc[0]
    block_size = block_size - size
    return Vs(block_size , mempool_data, txi_id)


# return a dict of tx_id as keys, for each tx_id it VCG price [satoshi)]
def VCG(block_size, tx_list, mempool_data):
    Pi = {}
    for txi_id in tx_list:
        vs = Vs(block_size, mempool_data, txi_id)
        vs_j = Vs_j(block_size , mempool_data, txi_id)
        res = vs - vs_j
        if res < 0 :
            res = 0
        Pi[txi_id] = res
    return Pi


############################
# Part 2
############################

def truthful_bidding_agent(tx_size, value, urgency, mempool_data, block_size):

	z = value*(2**(-3.6*urgency))
	return z

def forward_bidding_agent(tx_size, value, urgency, mempool_data, block_size):

    md = mempool_data.copy()
    tx_list = greedy_knapsack(block_size, md)
    md = md[(md[TXID_colunm_names].isin(tx_list))]

    z_dic = {}
    for z in range(0,5000,10):
        md_z = md[ (md[satoshi_per_byte_colunm_names] <= z) ]
        if len(md_z) ==0:
            z_dic[z] = NEVER
        else:
            md_z = md_z.sort_values(satoshi_per_byte_colunm_names, ascending=[False])
            md_z = md_z.head(1)
            diff_time = md_z[diff_time_colunm_names].iloc[0]
            z_dic[z] = diff_time

    maxZ = 0
    max_utility_value = float("-inf")
    for z in z_dic.keys():
        GTz = z_dic[z]
        if GTz != NEVER:
            utility_value = utility(value,urgency,tx_size,z,GTz)
            if utility_value > max_utility_value:
                max_utility_value = utility_value
                maxZ = z

    return maxZ




def utility(v,r,size,z,t):
    if t == NEVER : # t==never
        return 0
    return v * 2 ** ( (-1*t)* (r/1000.0) ) - z * size



if __name__ == '__main__':
    mempool_data = load_mempool_data('bitcoin_mempool_data.json')
    block_size = 5600
    tx_list = greedy_knapsack(block_size,mempool_data)
    revenue = evaluate_block(tx_list, mempool_data)
    revenue_prices = VCG(block_size,tx_list,mempool_data)
    #convert_bitcoin_mempool_data_json_to_csv()







    # from scipy import stats
    # import matplotlib.pyplot as plt
    # x = md[fee_colunm_names]
    # y = md[diff_time_colunm_names]
    #
    # slope, intercept, r_value, p_value, std_err = stats.linregress(x ,y)
    # line1 = intercept + slope * x
    #
    # plt.plot(line1, 'r-')
    # plt.plot(x, y, 'ro')