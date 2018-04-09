"""
HW1 part2
"""

import numpy as np
import random

time = 1098770179

"""
competitive part
"""


def predict_future_links(k: int):
    """
    this is an example code of 3 arbitrary nodes with arbitrary features for each node!!!
    you should delete this code and write your own as in the HW1 document
    """
    nodes = range(100)
    return tuple(random.shuffle(nodes)[:k])
