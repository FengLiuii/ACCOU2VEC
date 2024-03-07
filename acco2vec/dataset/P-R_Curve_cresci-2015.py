# -*- codeing = utf-8 -*-
# @Time : 2021/11/17 15:11
# @ Author : LF
# @ File : P-R_Curve.py
# @ Software : PyCharm
import numpy as np
from gensim.models import KeyedVectors
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
import warnings
from sklearn.metrics import precision_recall_curve
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
warnings.filterwarnings("ignore", category=FutureWarning)

cresci2015_userdata_file_path = 'C:/Users/LF/PycharmProjects/papercode/userdata.txt'
cresci2015_output_emb_file_path_dbot = 'cresci_model_emb/cresci_2015_dbot2vec3.emb'
cresci2015_output_emb_file_path_node2vec = 'cresci_model_emb/embeddings.emb'
cresci2015_output_emb_file_path_struc2vec = 'cresci_model_emb/struc2vec-cresci2015.emb'
cresci2015_output_emb_file_path_bot2vec = 'cresci_model_emb/cresci-2015.emb'
cresci2015_output_emb_file_path_deepwalk = 'cresci_model_emb/deepwalk2.emb'
cresci2015_emb_model_dbot2vec = KeyedVectors.load_word2vec_format(cresci2015_output_emb_file_path_dbot)
cresci2015_emb_model_node2vec = KeyedVectors.load_word2vec_format(cresci2015_output_emb_file_path_node2vec)
cresci2015_emb_model_struc2vec = KeyedVectors.load_word2vec_format(cresci2015_output_emb_file_path_struc2vec)
cresci2015_emb_model_bot2vec = KeyedVectors.load_word2vec_format(cresci2015_output_emb_file_path_bot2vec)
cresci2015_emb_model_deepwalk = KeyedVectors.load_word2vec_format(cresci2015_output_emb_file_path_deepwalk)


# twitter1_userdata_file_path = './twitter_model_emb/twitter_edges.txt'
# twitter2_userdata_file_path ='C:/Users/LF/PycharmProjects/papercode/twitter_dataset/50k_node_label.txt'
# twitter_output_emb_file_path_dbot = 'C:/Users/LF/PycharmProjects/papercode/twitter_dataset/50k_graph.emb'
# twitter_output_emb_file_path_node2vec = 'twitter_model_emb/node2vec_12k_graph.emb'
# twitter_output_emb_file_path_bot2vec = 'twitter_model_emb/bot2vec_12k_graph.emb'
# twitter_output_emb_file_path_deepwalk = 'twitter_model_emb/deepwakl_12k_graph.emb'
# twitter_emb_model_dbot2vec = KeyedVectors.load_word2vec_format(twitter_output_emb_file_path_dbot)
# twitter_emb_model_node2vec = KeyedVectors.load_word2vec_format(twitter_output_emb_file_path_node2vec)
# twitter_emb_model_bot2vec = KeyedVectors.load_word2vec_format(twitter_output_emb_file_path_bot2vec)
# twitter_emb_model_deepwalk = KeyedVectors.load_word2vec_format(twitter_output_emb_file_path_deepwalk)
id1_label1_dict = {}

#cresci_2015标签和节点
# with open(cresci2015_userdata_file_path, 'r', encoding='utf-8') as f:
# # with open(twitter_userdata_file_path, 'r', encoding='utf-8') as f:
#     for line in f:
#         # splits = line.split(' ')
#         splits = line.split('\t')
#         user_id = splits[0]
#         label = splits[4].replace('\n', '')
#         id_label_dict.update({user_id: int(label)})


with open(cresci2015_userdata_file_path, 'r', encoding='utf-8') as f:
# with open(twitter_userdata_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        # splits = line.split(' ')
        splits = line.split('\t')
        user_id = splits[0]
        label = splits[4].replace('\n', '')
        id1_label1_dict.update({user_id: int(label)})
# id2_label2_dict = {}
# with open(twitter1_userdata_file_path, 'r', encoding='utf-8') as f:
# # with open(twitter_userdata_file_path, 'r', encoding='utf-8') as f:
#     for line in f:
#         # splits = line.split(' ')
#         splits = line.split(' ')
#         user_id = splits[0]
#         label = splits[1].replace('\n', '')
#         id2_label2_dict.update({user_id: int(label)})

X1 = []
y1 = []
X2 = []
y2 = []
X3 = []
y3 = []
X4 = []
y4 = []
X5 = []
y5 = []
# a

for idx, key in enumerate(cresci2015_emb_model_dbot2vec.wv.vocab):
    emb_vector = cresci2015_emb_model_dbot2vec.wv[key]
    X1.append(emb_vector)
    y1.append(id1_label1_dict[key])
for idx, key in enumerate(cresci2015_emb_model_bot2vec.wv.vocab):
    emb_vector = cresci2015_emb_model_bot2vec.wv[key]
    X2.append(emb_vector)
    y2.append(id1_label1_dict[key])
for idx, key in enumerate(cresci2015_emb_model_node2vec.wv.vocab):
    emb_vector = cresci2015_emb_model_node2vec.wv[key]
    X3.append(emb_vector)
    y3.append(id1_label1_dict[key])
for idx, key in enumerate(cresci2015_emb_model_deepwalk.wv.vocab):
    emb_vector = cresci2015_emb_model_deepwalk.wv[key]
    X4.append(emb_vector)
    y4.append(id1_label1_dict[key])
for idx, key in enumerate(cresci2015_emb_model_struc2vec.wv.vocab):
    emb_vector = cresci2015_emb_model_struc2vec.wv[key]
    X5.append(emb_vector)
    y5.append(id1_label1_dict[key])
# for idx, key in enumerate(cresci2015_emb_model_deepwalk.wv.vocab):
#     emb_vector = cresci2015_emb_model_deepwalk.wv[key]
#     X5.append(emb_vector)
#     y5.append(id_label_dict[key])

train_x1, test_x1, train_y1, test_y1 = train_test_split(X1, y1, test_size=5/ 10, random_state=2)
train_x2, test_x2, train_y2, test_y2 = train_test_split(X2, y2, test_size=8/ 10, random_state=2)
train_x3, test_x3, train_y3, test_y3 = train_test_split(X3, y3, test_size=9/ 10, random_state=2)
train_x4, test_x4, train_y4, test_y4 = train_test_split(X4, y4, test_size=9/ 10, random_state=2)
train_x5, test_x5, train_y5, test_y5 = train_test_split(X5, y5, test_size=9/ 10, random_state=2)


# classify
svm_classifier1 = svm.SVC(kernel='rbf', C=1)
svm_classifier1.fit(train_x1, train_y1)
svm_classifier2 = svm.SVC(kernel='rbf', C=10)
svm_classifier2.fit(train_x2, train_y2)
svm_classifier3 = svm.SVC(kernel='rbf', C=1)
svm_classifier3.fit(train_x3, train_y3)
svm_classifier4 = svm.SVC(kernel='rbf', C=1)
svm_classifier4.fit(train_x4, train_y4)
svm_classifier5 = svm.SVC(kernel='rbf', C=1)
svm_classifier5.fit(train_x5, train_y5)

y1_score = svm_classifier1.decision_function(test_x1)
y2_score = svm_classifier2.decision_function(test_x2)
y3_score = svm_classifier3.decision_function(test_x3)
y4_score = svm_classifier4.decision_function(test_x4)
y5_score = svm_classifier5.decision_function(test_x5)

# roc_auc
precision1, recall1, _ = precision_recall_curve(test_y1, y1_score)
precision2, recall2, _ = precision_recall_curve(test_y2, y2_score)
precision3, recall3, _ = precision_recall_curve(test_y3, y3_score)
precision4, recall4, _ = precision_recall_curve(test_y4, y4_score)
precision5, recall5, _ = precision_recall_curve(test_y5, y5_score)


# metrics.plot_confusion_matrix(svm_classifier, test_x, test_y)

#roc_auc curve
# fpr1,tpr1,threshholds = roc_curve(test_y1,svm_classifier.fit(train_x1, train_y1).decision_function(test_x1))
# roc_auc1 = auc(fpr1,tpr1)
# fpr2,tpr2,threshholds = roc_curve(test_y2,svm_classifier.fit(train_x2, train_y2).decision_function(test_x2))
# roc_auc2 = auc(fpr2,tpr2)
# fpr3,tpr3,threshholds = roc_curve(test_y3,svm_classifier.fit(train_x3, train_y3).decision_function(test_x3))
# roc_auc3 = auc(fpr3,tpr3)
# fpr4,tpr4,threshholds = roc_curve(test_y4,svm_classifier.fit(train_x4, train_y4).decision_function(test_x4))
# roc_auc4 = auc(fpr4,tpr4)
# fpr5,tpr5,threshholds = roc_curve(test_y5,svm_classifier.fit(train_x5, train_y5).decision_function(test_x5))
# roc_auc5 = auc(fpr5,tpr5)
recall3_new = []
precision3_new = []
recall4_new = []
precision4_new = []
recall5_new = []
precision5_new = []
for i in range(0, len(recall3)):
    if  i<10 or i%10 == 0 :
        recall3_new.append(recall3[i])
        precision3_new.append(precision3[i])
for i in range(0, len(recall4)):
    if i < 10 or i % 10 == 0:
        recall4_new.append(recall4[i])
        precision4_new.append(precision4[i])
for i in range(0, len(recall5)):
    if i < 10 or i % 10 == 0:
        recall5_new.append(recall5[i])
        precision5_new.append(precision5[i])
    # else:
    #     recall3_new.append(recall3[i])
    #     precision3_new.append(precision3[i])
print(precision3_new,len(precision3_new))

fig = plt.figure(figsize=(6,4),dpi=300)

ax1 = fig.add_subplot(111)
# draw roc_auc curve
# ax1.plot(fpr1,tpr1,label='dbot2vec (area = %0.4f)'% roc_auc1)
# ax1.plot(fpr2,tpr2,label='node2vec (area = %0.4f)'% roc_auc2)
# ax1.plot(fpr3,tpr3,label='struc2vec (area = %0.4f)'% roc_auc3)
# ax1.plot(fpr4,tpr4,label='bot2vec (area = %0.4f)'% roc_auc4)
# ax1.plot(fpr5,tpr5,label='deepwalk (area = %0.4f)'% roc_auc5)
# draw p_r curve

ax1.plot(recall2,precision2,'#00008B',label='Bot2vec',linewidth=1.5,linestyle='-' )
ax1.plot( recall3_new,precision3_new,'#FF8C00',label='Node2vec',linewidth=1.5,linestyle='-' )
ax1.plot(recall4_new,precision4_new,'#808080',label='Deepwalk',linewidth=1.5,linestyle='-' )
ax1.plot(recall5_new,precision5_new,'#008000',label='Struc2vec',linewidth=1.5,linestyle='-' )
ax1.plot(recall1,precision1,'#FF0000',label='Accou2vec',linewidth=1.5,linestyle='-' )

ax1.legend(loc='best', prop={'family':'Times New Roman', 'size':8}, frameon=True)
plt.xlabel('Recall',fontdict={'family' : 'Times New Roman', 'size':12})
plt.ylabel('Precision',fontdict={'family' : 'Times New Roman', 'size':12})
# plt.title('The Precision-Recall Curve of Diffent Model in Cresci-2015 Data Set')

# 嵌入放大子图
axins = inset_axes(ax1, width="50%", height="50%", loc='lower left',
                   bbox_to_anchor=(0.3, 0.1, 1, 1),
                   bbox_transform=ax1.transAxes)

axins.plot(recall2,precision2,'#00008B',label='Bot2vec',linewidth=1.5,linestyle='-' )
axins.plot(recall3,precision3,'#FF8C00',label='Node2vec',linewidth=1.5,linestyle='-' )
axins.plot(recall4,precision4,'#808080',label='Deepwalk',linewidth=1.5,linestyle='-' )
axins.plot(recall5,precision5,'#008000',label='Struc2vec',linewidth=1.5,linestyle='-' )
axins.plot(recall1,precision1,'#FF0000',label='Accou2vec',linewidth=1.5,linestyle='-' )
# 设置放大区间
zone_left = 1000
zone_right = 1000

# 坐标轴的扩展比例（根据实际数据调整）
# x_ratio = 0.9 # x轴显示范围的扩展比例
# y_ratio = 0.9  # y轴显示范围的扩展比例

# X轴的显示范围
xlim0 = 0.9
xlim1 = 1.01

# Y轴的显示范围
y = np.hstack((precision1[zone_left:zone_right], precision2[zone_left:zone_right],
               precision3[zone_left:zone_right],precision4[zone_left:zone_right],
               precision5[zone_left:zone_right]))
ylim0 = 0.85
ylim1 = 1.01

tx0 = xlim0
tx1 = xlim1
ty0 = ylim0
ty1 = ylim1
sx = [tx0,tx1,tx1,tx0,tx0]
sy = [ty0,ty0,ty1,ty1,ty0]
ax1.plot(sx,sy,"#BC8F8F",linewidth=1.5,linestyle='--',alpha=1.0)

# 画两条线
xy = (xlim0,ylim0)
xy2 = (xlim0,ylim1)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax1,alpha=0.2,linewidth=0.5,linestyle='--')
axins.add_artist(con)

xy = (xlim1,ylim0)
xy2 = (xlim1,ylim1)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax1,alpha=0.2,linewidth=0.5,linestyle='--')
axins.add_artist(con)


# 调整子坐标系的显示范围
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)
plt.savefig('C:/Users/LF/PycharmProjects/papercode/Figure/p-r-cresci.png',dpi=300, bbox_inches="tight")

plt.show()