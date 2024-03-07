# -*- codeing = utf-8 -*-
# @Time : 2021/10/14 9:24
# @ Author : LF
# @ File : txt_to_csv.py
# @ Software : PyCharm
import pandas as pd
import numpy as np
# txt = np.loadtxt('C:/Users/LF/PycharmProjects/papercode/dataset/twiter_edges.txt')
txt = np.loadtxt('C:/Users/LF/PycharmProjects/papercode/dataset/222.txt')
txtDF = pd.DataFrame(txt)
txtDF = txtDF.astype(np.int64)
print(txtDF.dtypes)
txtDF.to_csv('C:/Users/LF/PycharmProjects/papercode/dataset/test/222.csv',index=False)