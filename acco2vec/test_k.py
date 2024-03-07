# -*- codeing = utf-8 -*-
# @Time : 2021/11/3 11:04
# @ Author : LF
# @ File : test_k.py
# @ Software : PyCharm
import numpy as np
import matplotlib.pyplot as plt
from surprise.model_selection import cross_validate
from surprise import Dataset,Reader
import os

import os
from surprise import Reader, Dataset
from surprise.prediction_algorithms.matrix_factorization import NMF




data = Dataset.load_builtin('ml-100k')


RMSE=[]
MAE=[]



for k in range(2,20,2):
    nmf = NMF(n_factors=k)
    nmf_result =cross_validate(nmf,data,measures=['rmse','mae'],cv=3)
    RMSE.append(np.mean(nmf_result['test_rmse']))
    MAE.append(np.mean(nmf_result['test_mae']))
    print("k:{};RMSE:{};MAE:{}".format(k,nmf_result['test_rmse'],nmf_result['test_mae']))

print("min k:{} for RMSE:{},MAE:{}".format(k,np.argmin(RMSE),np.argmin(MAE)))
print("min RMSE:{},MAE:{}".format(min(RMSE),min(MAE)))
x=[2,4,6,8,10,12,14,16,18]
plt.plot(x,RMSE)
plt.plot(x,MAE)
plt.show()

# import numpy as np
# from sklearn.manifold import TSNE
# from sklearn import datasets
# import time
# import numpy as np
# import matplotlib.pyplot as plt
# n_components = 2
#
#
# # %%
# digits = datasets.load_digits(n_class=10)
# data = digits['data']
# label = digits['target']
# n_samples, n_features = data.shape
# n_features
#
#
# # %%
# tsne = TSNE(n_components=n_components, init='random', random_state=0, perplexity=30)
# start = time.time()
# result = tsne.fit_transform(data)
# end = time.time()
# print('t-SNE time: {}'.format(end-start))
#
#
# # %%
# # result
#
#
# # %%
# x_min, x_max = np.min(result, 0), np.max(result, 0)
# result = (result-x_min)/(x_max-x_min)
# plt.figure(figsize=(8, 8))
# for i in range(n_samples):
#
#     plt.text(result[i, 0], result[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10.), fontdict={'weight': 'bold', 'size': 9})
#
# # plt.xticks([])
# # plt.yticks([])
# # plt.title('t-SNE-digits')
# plt.show()

