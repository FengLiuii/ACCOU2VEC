# -*- codeing = utf-8 -*-
# @Time : 2021/11/17 10:39
# @ Author : LF
# @ File : roc-auc.py
# @ Software : PyCharm
import numpy as np
from gensim.models import KeyedVectors
from sklearn import svm
from sklearn.model_selection import cross_val_score,train_test_split,cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pylab as plt
import warnings
from sklearn import tree,metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score,auc
from sklearn.metrics import roc_curve,SCORERS
import pandas as pd
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


id_label_dict = {}
with open(cresci2015_userdata_file_path, 'r', encoding='utf-8') as f:
# with open(twitter_userdata_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        # splits = line.split(' ')
        splits = line.split('\t')
        user_id = splits[0]
        label = splits[4].replace('\n', '')
        id_label_dict.update({user_id: int(label)})

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

# for idx, key in enumerate(cresci2015_emb_model.wv.vocab):
#     emb_vector = cresci2015_emb_model.wv[key]
#     X.append(emb_vector)
#     y.append(id_label_dict[key])
for idx, key in enumerate(cresci2015_emb_model_dbot2vec.wv.vocab):
    emb_vector = cresci2015_emb_model_dbot2vec.wv[key]
    X1.append(emb_vector)
    y1.append(id_label_dict[key])
for idx, key in enumerate(cresci2015_emb_model_node2vec.wv.vocab):
    emb_vector = cresci2015_emb_model_node2vec.wv[key]
    X2.append(emb_vector)
    y2.append(id_label_dict[key])
for idx, key in enumerate(cresci2015_emb_model_struc2vec.wv.vocab):
    emb_vector = cresci2015_emb_model_struc2vec.wv[key]
    X3.append(emb_vector)
    y3.append(id_label_dict[key])
for idx, key in enumerate(cresci2015_emb_model_bot2vec.wv.vocab):
    emb_vector = cresci2015_emb_model_bot2vec.wv[key]
    X4.append(emb_vector)
    y4.append(id_label_dict[key])
for idx, key in enumerate(cresci2015_emb_model_deepwalk.wv.vocab):
    emb_vector = cresci2015_emb_model_deepwalk.wv[key]
    X5.append(emb_vector)
    y5.append(id_label_dict[key])
# print(y)
# y1 == 1
# t = pd.Series(y1).value_counts()
# print(t)
train_x1, test_x1, train_y1, test_y1 = train_test_split(X1, y1, test_size=9/ 10, random_state=2)
train_x2, test_x2, train_y2, test_y2 = train_test_split(X2, y2, test_size=9/ 10, random_state=2)
train_x3, test_x3, train_y3, test_y3 = train_test_split(X3, y3, test_size=9/ 10, random_state=2)
train_x4, test_x4, train_y4, test_y4 = train_test_split(X4, y4, test_size=5/ 10, random_state=2)
train_x5, test_x5, train_y5, test_y5 = train_test_split(X5, y5, test_size=9/ 10, random_state=2)


svm_classifier = svm.SVC(kernel='sigmoid', C=1)
svm_classifier.fit(train_x1, train_y1)
svm_classifier.fit(train_x2, train_y2)
svm_classifier.fit(train_x3, train_y3)
svm_classifier.fit(train_x4, train_y4)
svm_classifier.fit(train_x5, train_y5)
# metrics.plot_confusion_matrix(svm_classifier, test_x, test_y)


fpr1,tpr1,threshholds = roc_curve(test_y1,svm_classifier.fit(train_x1, train_y1).decision_function(test_x1))
roc_auc1 = auc(fpr1,tpr1)
fpr2,tpr2,threshholds = roc_curve(test_y2,svm_classifier.fit(train_x2, train_y2).decision_function(test_x2))
roc_auc2 = auc(fpr2,tpr2)
fpr3,tpr3,threshholds = roc_curve(test_y3,svm_classifier.fit(train_x3, train_y3).decision_function(test_x3))
roc_auc3 = auc(fpr3,tpr3)
fpr4,tpr4,threshholds = roc_curve(test_y4,svm_classifier.fit(train_x4, train_y4).decision_function(test_x4))
roc_auc4 = auc(fpr4,tpr4)
fpr5,tpr5,threshholds = roc_curve(test_y5,svm_classifier.fit(train_x5, train_y5).decision_function(test_x5))
roc_auc5 = auc(fpr5,tpr5)

fig = plt.figure(figsize=(6,4),dpi=300)

ax1 = fig.add_subplot(111)


ax1.plot(fpr2,tpr2,color='#FF8C00',label='Node2vec (AUC = %0.4f)'% roc_auc2)
ax1.plot(fpr3,tpr3,'#008000',label='Struc2vec (AUC = %0.4f)'% roc_auc3)
ax1.plot(fpr4,tpr4,'#00008B',label='Bot2vec (AUC = %0.4f)'% roc_auc4)
ax1.plot(fpr5,tpr5,'#808080',label='Deepwalk (AUC = %0.4f)'% roc_auc5)
ax1.plot(fpr1,tpr1,'#FF0000',label='Accou2vec (AUC = %0.4f)'% roc_auc1)
ax1.legend(loc='best', prop={'family':'Times New Roman', 'size':8},frameon=True)
plt.xlabel('False Positive Rate',fontdict={'family' : 'Times New Roman', 'size':12})
plt.ylabel('True Positive Rate',fontdict={'family' : 'Times New Roman', 'size':12})
plt.savefig('C:/Users/LF/PycharmProjects/papercode/Figure/roc-auc-cresci.png',dpi=300, bbox_inches="tight")
# plt.title('The ROC-AUC Curve of Diffent Model in Cresci-2015 Data Set')
plt.show()




