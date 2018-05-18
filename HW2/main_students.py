#!/usr/local/bin/python
# -*- coding: utf-8 -*-

'''
we will be working with similar main, you should not change this file!
all your imports/constants/classes/func should be held in hw2_students.py
'''

from hw2_part1 import * 


block_size=5000
mempool_data_name='bitcoin_mempool_data.json'

# load the data once for all func of the mandatory part , at your choosing (class, dict, pandas, etc..)
mempool_data_full_path=os.path.abspath(mempool_data_name)
mempool_data=load_mempool_data(mempool_data_full_path)


def main():

	############################
	# Question 1
	############################

	# tx_to_insert_list=greedy_knapsack(block_size, mempool_data)
	# print ('pay-your-bid revenue: ', evaluate_block(tx_to_insert_list, mempool_data))
    #
	#
	# vcg=VCG(block_size, tx_to_insert_list, mempool_data)
	# print ('vcg revenue: ',sum(vcg.values()))
	#
	
	############################
	# Question 2
	############################

	bid_true=truthful_bidding_agent(225, 4000, 0.4, mempool_data, block_size)
	bid_students=forward_bidding_agent(225, 4000, 0.4, mempool_data, block_size)
	print ('bid of truthful agent: %s, bid of student agent: %s' %(bid_true, bid_students))
	

	
if __name__ == "__main__":
	main()

	
