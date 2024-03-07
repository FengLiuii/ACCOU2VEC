# -*- codeing = utf-8 -*-
# @Time : 2021/11/21 19:39
# @ Author : LF
# @ File : community.py
# @ Software : PyCharm
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

path = './222.txt'
source = []
target = []
edges = []
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        splits = line.split(' ')
        edges.append((splits[0],splits[1]))
        source.append(splits[0])
        target.append(splits[1])
nodes = source
edges = edges

G2 = nx.DiGraph()
G2.add_nodes_from(nodes)
G2.add_edges_from(edges)
pos2 = nx.spring_layout(G2)
nx.draw(G2, pos2, with_labels=True, font_weight='bold',alpha = 0.1)
# plt.axis('on')
plt.xticks([])
plt.yticks([])

plt.show()



# edges = pd.DataFrame()
# edges["source"] = source
# edges["target"] = target

# G = nx.from_pandas_edgelist(edges,source='source',target='target')
# nx.draw_networkx(G)
#
# plt.show()

