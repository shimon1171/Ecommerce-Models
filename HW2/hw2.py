

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
    mempool_data = df[ (df[output_colunm_names] == -1) & (df[time_colunm_names] < current_time) & (current_time < df[removed_colunm_names]) ]
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



def VCG(block_size, tx_list, mempool_data):
    keys = ("513b062d7bce674e90f089e19028da4a4c7e6347711fd08a714d69e9ed60180f",
            "d42d24c1605a920866c209b8bdd11da01623d698e5a30319440e272332d660c7",
            "35882476b1074182635d0093617e7a477821830bd9d5785077148a62005b023a",
            "03ee8ba1bfd5882c4d1c51e6ed70933d4dced572187fc2dfe45e7a1849f0ef02",
            "5799ad0f83095a7942f9d925a70d9c4137f73bce54b93e95455a6f4e97877cfd",
            "63f1e5eef1f1eadf1ed89423b6ccb164344e81fa0bc8234ddbed065d20014988")
    values = (264927.0, 85939.0, 85939.0, 85939.0, 171878.0, 85939.0)

    # return a dict of tx_id as keys, for each tx_id it VCG price [satoshi)]
    return dict(zip(keys, values))



if __name__ == '__main__':
    mempool_data = load_mempool_data('bitcoin_mempool_data.json')
    block_size = 20000
    tx_list = greedy_knapsack(block_size,mempool_data)
    revenue = evaluate_block(tx_list, mempool_data)
    #convert_bitcoin_mempool_data_json_to_csv()

