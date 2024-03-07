# -*- codeing = utf-8 -*-
# @Time : 2022/4/21 14:21
# @ Author : LF
# @ File : pca_T20.py
# @ Software : PyCharm
import numpy as np
nodes = []
emb = []
arr = []
#with open('dataset/twitter_lables.txt','r') as f2:
with open('id_label1.txt','r') as f2:
    for line in f2:
        splits = line.split(' ')
        user_id1 = splits[0]
        arr.append([splits[0], splits[1][0]])
        '''id_label=dict(zip(user_id1,label))
        id_label.update(id_label)'''
print(user_id1)
arr2=[]
with open('GAT.emb','r') as f1:
    for line in f1:
        splits = line.split(' ')
        user_id = splits[0]
        print(user_id)
        emberdding = splits[1:-1]
        arr2.append(arr[int(user_id)])
        '''nodes.append(user_id)
        emb.append(emberdding)'''
np.savetxt('id_label_t_emb4.txt',arr2,delimiter=' ',fmt='%s')
# print(len(nodes))
# id_label = {}
#
#
#
# print(id_label)
