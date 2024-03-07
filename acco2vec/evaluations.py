import numpy as np
from gensim.models import KeyedVectors
from sklearn import svm
from sklearn.model_selection import cross_val_score,train_test_split,cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pylab as plt
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import accuracy_score
import warnings
from sklearn import tree,metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score,roc_auc_score,homogeneity_score

from sklearn.metrics import roc_curve,SCORERS
import pandas as pd



warnings.filterwarnings("ignore", category=FutureWarning)

twitter_accou2vec_path = './tree_t/Our_e.emb'

twitter_accou2vec_label = './tree_t/id_label_paper3.txt'

cresci2015_emb_model = KeyedVectors.load_word2vec_format(twitter_accou2vec_path)


id_label_dict = {}

with open(twitter_accou2vec_label, 'r', encoding='utf-8') as f:
    for line in f:
        # splits = line.split(' ')
        splits = line.split(' ')
        user_id = splits[0]
        label = splits[1].replace('\n', '')
        id_label_dict.update({user_id: int(label)})

X = []
y = []



for idx, key in enumerate(cresci2015_emb_model.wv.vocab):
    emb_vector = cresci2015_emb_model.wv[key]
    X.append(emb_vector)
    y.append(id_label_dict[key])
print(X,len(X))
print(y,len(y))



train_x, test_x, train_y, test_y = train_test_split(X, y,test_size=0.95, random_state=2)



svm_classifier = svm.SVC(kernel='rbf', C=1)
svm_classifier.fit(train_x, train_y)
metrics.plot_confusion_matrix(svm_classifier, test_x, test_y)


_10_folds_cross_val_scores_svm = cross_val_score(svm_classifier,test_x, test_y, cv=10,scoring='accuracy')
_10_folds_cross_val_scores_svm1 = cross_val_score(svm_classifier,test_x, test_y, cv=10,scoring='f1')
_10_folds_cross_val_scores_svm2 = cross_val_score(svm_classifier,test_x, test_y, cv=10,scoring='precision')
_10_folds_cross_val_scores_svm3 = cross_val_score(svm_classifier,test_x, test_y, cv=10,scoring='roc_auc')
_10_folds_cross_val_scores_svm5 = cross_val_score(svm_classifier, test_x,test_y, cv=10,scoring='homogeneity_score')

print('Evaluating accuracy performance of Daccou2vec model for bot classification task...')


print('Average 10-folds cross validation accuracy (SVM): {}'.format(np.mean(_10_folds_cross_val_scores_svm)))
print('Average 10-folds cross validation f1-score (SVM): {}'.format(np.mean(_10_folds_cross_val_scores_svm1)))
print('Average 10-folds cross validation precision (SVM): {}'.format(np.mean(_10_folds_cross_val_scores_svm2)))
print('Average 10-folds cross validation roc_auc (SVM): {}'.format(np.mean(_10_folds_cross_val_scores_svm3)))
print('Average 10-folds cross validation H-score (svm): {}'.format(np.mean(_10_folds_cross_val_scores_svm5)))

