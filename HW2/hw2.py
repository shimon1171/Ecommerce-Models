

import json
import pandas as pd


# const
TXID_colunm_names = 'TXID'
fee_colunm_names = 'fee'
output_colunm_names = 'output'
removed_colunm_names = 'removed'
size_colunm_names = 'size'
time_colunm_names = 'time'


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
    sum_colunm_names = 'sum'
    md = mempool_data.sort_values([fee_colunm_names, TXID_colunm_names], ascending=[True, False])
    md[sum_colunm_names] = md[size_colunm_names].cumsum()
    md = md[ (md[sum_colunm_names] < block_size)]
    tx_ids = list(md[TXID_colunm_names])
    return tx_ids


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
        Pi[txi_id] = vs - vs_j
    return Pi











if __name__ == '__main__':
    mempool_data = load_mempool_data('bitcoin_mempool_data.json')
    block_size = 5600
    tx_list = greedy_knapsack(block_size,mempool_data)
    revenue = evaluate_block(tx_list, mempool_data)
    revenue_prices = VCG(block_size,tx_list,mempool_data)
    #convert_bitcoin_mempool_data_json_to_csv()

