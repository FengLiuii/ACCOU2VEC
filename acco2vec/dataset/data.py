# -*- codeing = utf-8 -*-
# @Time : 2021/11/4 11:09
# @ Author : LF
# @ File : data.py
# @ Software : PyCharm
import pandas as pd
import numpy as np
user_id = []
user_labler = {}
with open('./91263_sybil.txt', 'r', encoding='utf-8') as f:
# with open(twitter_userdata_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        # splits = line.split(' ')
        splits = line.split('/n')
        user = splits[0]
        user_id.append(user)
    for i in range(len(user_id)):
        user_labler = {user_id[i]:1}




