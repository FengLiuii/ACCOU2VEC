# -*- codeing = utf-8 -*-
# @Time : 2021/11/12 21:47
# @ Author : LF
# @ File : open.py
# @ Software : PyCharm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
import pandas as pd
from sklearn.manifold import TSNE
from  sklearn.metrics import homogeneity_score

nodes = []
e = []
# d = {}
#with open('cresci-2015.emb','r',) as f:
#with open('cresci_2015_dbot2vec3.emb','r',) as f:
#with open('embeddings.emb','r',) as f:
#with open('deepwalk.emb','r',) as f:

with open('cresci_2015_dbot2vec3.emb','r',) as f:
    for line in f:
        splits = line.split(' ')
        user_id = splits[0]
        emberdding = splits[1:-1]


        nodes.append(user_id)
        e.append(emberdding)
        # d.update({user_id:emberdding})
# data = nodes
# df = pd.DataFrame(data)
# df.to_csv('cresce_nodes.csv',index=False,header=False)
for i in range(len(nodes)):
    if(int(nodes[i])<1950):
        nodes[i] = 0
    else:
        nodes[i] = 1
# data = nodes
# df = pd.DataFrame(data)
# df.to_csv('cresce_labels.csv',index=False,header=False)

b = []
for i in range(len(e)):
    a = []
    for j in range(len(e[i])):
        a.append(float(e[i][j]))

    b.append(a)

c = np.array(b)

# print(y)

x = b
# print(x)

y =np.array(nodes)



#pca = decomposition.PCA(n_components=3)

#pca.fit(x)

#X = pca.transform(x)
#X = TSNE(n_components=3,random_state=3).fit_transform(x)
X = TSNE(n_components=2,  perplexity=100.0, early_exaggeration=12.0,
         learning_rate=200.0, n_iter=1000, n_iter_without_progress=300,
         min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0,
         random_state=None, method='barnes_hut', angle=0.5, n_jobs=None,
         square_distances='legacy').fit_transform(x)
# print(X)



plt.figure(figsize=(4, 3),dpi=300)
'''for label in [0,1]:

    plt.scatter(X[y==label, 0],
                X[y==label, 1]
                ,label =label
                 )'''
plt.scatter(X[y==0, 0],
                X[y==0, 1]
                ,label ='Bot',c='r',s=5,alpha = 1,edgecolors = 'r'
                 )
plt.scatter(X[y==1, 0],
                X[y==1, 1]
                ,label ='Human'
            ,c='lime',
            s=5,alpha = 1,edgecolors = 'lime'
                 )

plt.xlabel(' ')
plt.ylabel(' ')
# X = [-40,-20,0,20,40]
# Y = [-40,-20,0,20,40]
# X = [-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5]
# Y = [-1.5,-1,0,1,2]
# plt.xticks(X,['-2','-1','0','1','2'],fontproperties = 'Times New Roman', fontsize=12)
# plt.yticks(Y,['-2','-1','0','1','2'],fontproperties = 'Times New Roman', fontsize=12)
plt.legend(loc='best',prop={'family':'Times New Roman', 'size':12},frameon=True)
# plt.title('(a) PCA 2D projections of Ours model.')
# plt.title('T-sne 2D projections of Dbot2vec.')

plt.tight_layout()
# plt.savefig('../Figure/PCA-struc.png',dpi=100, bbox_inches="tight")
plt.savefig('C:/Users/LF/PycharmProjects/papercode/Figure/Accou2vec.png',dpi=300, bbox_inches="tight")
plt.show()


# fig = plt.figure(1, figsize=(4, 3))
# plt.clf()  #只会清除数字 仍然可以在其上绘制另一个绘图
# ax = Axes3D(fig)
# plt.cla()
#
# pca = decomposition.PCA(n_components=3)   #这里为维数
# pca.fit(x)
# X = pca.transform(x)
# ####显示方差
# # print(pca.explained_variance_ratio_)  #投影后的三个维度的方差分布 [0.92461872 0.05306648 0.01710261]
# # print(pca.explained_variance_) #方差 [4.22824171 0.24267075 0.0782095 ]
#
# for name, label in [('human', 0), ('bot', 1)]:
#     ax.text3D(X[y == label, 0].mean(),
#               X[y == label, 1].mean() + 3,
#               X[y == label, 2].mean(), name,
#               # horizontalalignment='center',
#               bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# # Reorder the labels to have colors matching the cluster results
# y = np.choose(y, [1, 2, 0]).astype(np.float)
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,edgecolor='k')
# #ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.nipy_spectral,edgecolor='k')
# for i in range(len(X[:,0])):
#     print(X[i])
# ax.w_xaxis.set_ticklabels([])
# ax.w_yaxis.set_ticklabels([])
# ax.w_zaxis.set_ticklabels([])
# plt.show()


















