# -*- codeing = utf-8 -*-
# @Time : 2021/11/2 15:06
# @ Author : LF
# @ File : test.py
# @ Software : PyCharm
import json

import collections
with open('./cresci_membership280.json', 'r', encoding='utf8')as fp:
    json_data = json.load(fp)
    #print(json_data)
    key_value = list(json_data.keys())
    value_list = list(json_data.values())
    #print(value_list)
    value_list2 = list(set(value_list))
    #print(value_list2)
    print(len(value_list2))
    list3 = []
    for i in value_list2:

        list3.append([[value_list2[i]]])


    #print(value_list)
    for i in range(0, len(value_list)):

        list3[value_list[i]].append(i)



    dict = {}
    for i in range(0, len(list3)):
        dict[list3[i][0][0]] = list3[i][1:]

    dic = collections.Counter(value_list)
    for key in dic:
        print(key, dic[key])


