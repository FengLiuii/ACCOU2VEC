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

nodes = []
e = []

with open('./twibot20_1.emb','r',) as f:

    for line in f:
        splits = line.split(' ')
        user_id = splits[0]
        emberdding = splits[1:-1]


        nodes.append(user_id)
        e.append(emberdding)
        
l = []
with open('./tree_t/id_label_t_emb2.txt','r') as f1:
    for line in f1:
        splits = line.split(' ')
        label = splits[1].replace('\n','')
        label = int(label)
        l.append(label)

print(len(l),len(e))


b = []
for i in range(len(e)):
    a = []
    for j in range(len(e[i])):
        a.append(float(e[i][j]))

    b.append(a)

c = np.array(b)



x = b

y =np.array(l)





X = TSNE(n_components=2,  perplexity=10.0, early_exaggeration=12.0,
         learning_rate=20.0, n_iter=300, n_iter_without_progress=300,
         min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0,
         random_state=8, method='barnes_hut', angle=0.5, n_jobs=None,
         square_distances='legacy').fit_transform(x)


plt.scatter(X[y==0, 0],
                X[y==0, 1]
                ,label ='Bot',c='lime',s=5,alpha = 1,edgecolors = 'lime'
                 )
plt.scatter(X[y==1, 0],
                X[y==1, 1]
                ,label ='Human'
            ,c='r',
            s=5,alpha = 1,edgecolors = 'r'
                 )

plt.xlabel(' ')
plt.ylabel(' ')

plt.legend(loc='best',prop={'family':'Times New Roman', 'size':18},frameon=True)


plt.tight_layout()
plt.savefig('C:/Users/LF/PycharmProjects/papercode/Twi20_figure/t20_accou2vec.png',dpi=300, bbox_inches="tight")
plt.show()




















